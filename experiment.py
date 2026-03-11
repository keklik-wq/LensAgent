from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from src.agent_shell.llm_router import LlmRouterClient
from src.agent_shell.config import ShellConfig


OUTPUT_DIR = Path("output")
RUNS_DIR = OUTPUT_DIR / "runs"
DEFAULT_SCORING = {
    "spill_gb_penalty": 0.0,
    "small_file_penalty": 0.0,
}


@dataclass(frozen=True)
class Variant:
    shuffle_partitions: int
    executor_cores: int
    executor_instances: int
    executor_memory_gb: int


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _parse_memory_gb(value: str) -> int:
    raw = value.strip().lower()
    if raw.endswith("g"):
        return int(raw[:-1])
    if raw.endswith("m"):
        return max(1, int(int(raw[:-1]) / 1024))
    if raw.endswith("t"):
        return int(raw[:-1]) * 1024
    raise ValueError(f"Unsupported memory format: {value}")


def _format_memory_gb(gb: int) -> str:
    return f"{gb}g"


def _get_spark_conf(manifest: dict[str, Any]) -> dict[str, str]:
    return manifest.get("spec", {}).get("sparkConf", {}) or {}


def _get_executor_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("spec", {}).get("executor", {}) or {}


def _get_driver_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("spec", {}).get("driver", {}) or {}


def _get_arguments(manifest: dict[str, Any]) -> list[str]:
    return list(manifest.get("spec", {}).get("arguments", []) or [])


def _replace_argument(args: list[str], key: str, value: str) -> list[str]:
    replaced = []
    for item in args:
        raw = item
        if raw.startswith("--"):
            raw = raw[2:]
            prefix = "--"
        else:
            prefix = ""
        if raw.startswith(f"{key}="):
            replaced.append(f"{prefix}{key}={value}")
        else:
            replaced.append(item)
    return replaced


def _update_manifest(
    manifest: dict[str, Any],
    variant: Variant,
    run_id: str,
    dataversionid: str,
) -> dict[str, Any]:
    data = _deep_copy(manifest)
    data.setdefault("metadata", {})
    base_name = data["metadata"].get("name", "spark-app")
    data["metadata"]["name"] = f"{base_name}-r{run_id}"

    spark_conf = data.setdefault("spec", {}).setdefault("sparkConf", {})
    spark_conf["spark.sql.shuffle.partitions"] = str(variant.shuffle_partitions)
    spark_conf["spark.kubernetes.executor.request.cores"] = str(variant.executor_cores)

    executor = data["spec"].setdefault("executor", {})
    executor["cores"] = int(variant.executor_cores)
    executor["instances"] = int(variant.executor_instances)
    executor["memory"] = _format_memory_gb(variant.executor_memory_gb)

    args = _get_arguments(data)
    args = _replace_argument(
        args,
        "Output.RAW.ViewingEvent.dataversionid",
        dataversionid,
    )
    data["spec"]["arguments"] = args
    return data


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_variants(
    base: Variant,
    max_total_memory_gb: int,
    driver_memory_gb: int,
    count: int,
) -> list[Variant]:
    cores_candidates = sorted(
        {max(1, base.executor_cores - 1), base.executor_cores, base.executor_cores + 1, base.executor_cores + 2}
    )
    instances_candidates = sorted(
        {max(1, base.executor_instances - 2), base.executor_instances, base.executor_instances + 2, base.executor_instances + 4}
    )
    memory_candidates = sorted(
        {max(1, base.executor_memory_gb - 4), base.executor_memory_gb, base.executor_memory_gb + 4, base.executor_memory_gb + 8}
    )
    shuffle_candidates = sorted(
        {max(200, int(base.shuffle_partitions / 2)), base.shuffle_partitions, base.shuffle_partitions * 2}
    )

    all_variants: list[Variant] = []
    for shuffle in shuffle_candidates:
        for cores in cores_candidates:
            for instances in instances_candidates:
                for mem_gb in memory_candidates:
                    total_gb = driver_memory_gb + instances * mem_gb
                    if total_gb > max_total_memory_gb:
                        continue
                    all_variants.append(
                        Variant(
                            shuffle_partitions=shuffle,
                            executor_cores=cores,
                            executor_instances=instances,
                            executor_memory_gb=mem_gb,
                        )
                    )

    if not all_variants:
        raise SystemExit("No valid variants generated under memory constraint.")

    random.seed(42)
    if count >= len(all_variants):
        return all_variants
    return random.sample(all_variants, count)


def _build_tuning_prompt(
    history: list[dict[str, Any]],
    base_params: dict[str, Any],
    constraints: dict[str, Any],
    history_url: str,
) -> tuple[str, str]:
    system = (
        "You are a Spark tuning assistant. "
        "You must propose the next run parameters to minimize requested_gb_seconds, "
        "reduce spill_gb, and avoid too many small files. "
        "Return ONLY valid JSON that matches the schema exactly."
    )
    user = {
        "task": "Propose next Spark config parameters for the next run.",
        "base_params": base_params,
        "history": history,
        "history_url": history_url,
        "constraints": constraints,
        "response_schema": {
            "params": {
                "spark.sql.shuffle.partitions": "int",
                "executor.cores": "int",
                "executor.instances": "int",
                "executor.memory_gb": "int",
            },
            "rationale": "string",
        },
    }
    return system, json.dumps(user, ensure_ascii=True)


def _load_llm_client(config_path: Path) -> LlmRouterClient:
    cfg = ShellConfig.load(config_path)
    return LlmRouterClient(
        base_url=cfg.llm_router.base_url,
        api_key_env=cfg.llm_router.api_key_env,
        model=cfg.llm_router.model,
        timeout_seconds=cfg.llm_router.timeout_seconds,
        allow_models=cfg.llm_router.allow_models,
    )


def _apply_constraints(
    params: dict[str, Any],
    constraints: dict[str, Any],
    driver_memory_gb: int,
) -> Variant:
    def _clamp(name: str, value: int) -> int:
        lo = int(constraints[name]["min"])
        hi = int(constraints[name]["max"])
        return max(lo, min(hi, value))

    shuffle = _clamp("spark.sql.shuffle.partitions", int(params.get("spark.sql.shuffle.partitions", 200)))
    cores = _clamp("executor.cores", int(params.get("executor.cores", 1)))
    instances = _clamp("executor.instances", int(params.get("executor.instances", 1)))
    mem_gb = _clamp("executor.memory_gb", int(params.get("executor.memory_gb", 4)))

    total_gb = driver_memory_gb + instances * mem_gb
    if total_gb > int(constraints["total_memory_gb"]["max"]):
        max_total = int(constraints["total_memory_gb"]["max"]) - driver_memory_gb
        if max_total < mem_gb:
            mem_gb = max(1, max_total)
            instances = max(1, int(max_total / max(mem_gb, 1)))
        else:
            instances = max(1, int(max_total / mem_gb))

    return Variant(
        shuffle_partitions=shuffle,
        executor_cores=cores,
        executor_instances=instances,
        executor_memory_gb=mem_gb,
    )


def _sanitize_value(value: str) -> str:
    if not value:
        return value
    return "****"


SENSITIVE_KEY_RE = re.compile(
    r"(secret|password|access\.key|secret\.key|apikey|api_key|token|keytab)",
    re.IGNORECASE,
)


def _mask_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    data = _deep_copy(manifest)

    def mask_node(node: Any) -> Any:
        if isinstance(node, dict):
            masked = {}
            for key, val in node.items():
                if SENSITIVE_KEY_RE.search(str(key)):
                    masked[key] = _sanitize_value(str(val))
                else:
                    masked[key] = mask_node(val)
            return masked
        if isinstance(node, list):
            return [mask_node(item) for item in node]
        return node

    data = mask_node(data)

    args = _get_arguments(data)
    masked_args = []
    for item in args:
        prefix = "--" if item.startswith("--") else ""
        raw = item[2:] if item.startswith("--") else item
        if "=" in raw:
            key, value = raw.split("=", 1)
            if SENSITIVE_KEY_RE.search(key):
                masked_args.append(f"{prefix}{key}={_sanitize_value(value)}")
            else:
                masked_args.append(item)
        else:
            masked_args.append(item)
    data.setdefault("spec", {})["arguments"] = masked_args
    return data


def _build_run_id(index: int) -> str:
    return f"{index:03d}"


def _derive_dataversionid(base_value: str, run_index: int) -> str:
    if base_value.isdigit():
        return str(int(base_value) + run_index)
    return f"{base_value}-r{run_index:03d}"


def _ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _load_scoring(path: Path | None) -> dict[str, float]:
    scoring_path = path or (OUTPUT_DIR / "scoring.json")
    if not scoring_path.exists():
        scoring_path.parent.mkdir(parents=True, exist_ok=True)
        scoring_path.write_text(json.dumps(DEFAULT_SCORING, indent=2), encoding="utf-8")
        return dict(DEFAULT_SCORING)
    return json.loads(scoring_path.read_text(encoding="utf-8"))


def _score_run(run: dict[str, Any], scoring: dict[str, float]) -> float:
    base = float(run.get("requested_gb_seconds", 0.0))
    spill_gb = float(run.get("spill_gb", 0.0))
    small_files = float(run.get("small_files", 0.0))
    return (
        base
        + spill_gb * float(scoring.get("spill_gb_penalty", 0.0))
        + small_files * float(scoring.get("small_file_penalty", 0.0))
    )


def cmd_plan(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    transform_path = Path(args.transform)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not transform_path.exists():
        raise SystemExit(f"Transform not found: {transform_path}")

    manifest = _read_yaml(manifest_path)
    spark_conf = _get_spark_conf(manifest)
    executor = _get_executor_spec(manifest)
    driver = _get_driver_spec(manifest)

    base_variant = Variant(
        shuffle_partitions=int(spark_conf.get("spark.sql.shuffle.partitions", 200)),
        executor_cores=int(executor.get("cores", 1)),
        executor_instances=int(executor.get("instances", 1)),
        executor_memory_gb=_parse_memory_gb(str(executor.get("memory", "4g"))),
    )
    driver_memory_gb = _parse_memory_gb(str(driver.get("memory", "4g")))

    arguments = _get_arguments(manifest)
    dataversionid = ""
    for item in arguments:
        raw = item[2:] if item.startswith("--") else item
        if raw.startswith("Output.RAW.ViewingEvent.dataversionid="):
            dataversionid = raw.split("=", 1)[1]
            break
    if not dataversionid:
        raise SystemExit("Output.RAW.ViewingEvent.dataversionid not found in arguments.")

    variants = _generate_variants(
        base=base_variant,
        max_total_memory_gb=args.max_total_memory_gb,
        driver_memory_gb=driver_memory_gb,
        count=args.runs,
    )

    _ensure_dirs()
    transform_hash = _hash_file(transform_path)
    for idx, variant in enumerate(variants, start=1):
        run_id = _build_run_id(idx)
        run_dir = RUNS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        derived_dataversionid = _derive_dataversionid(dataversionid, idx)
        updated_manifest = _update_manifest(
            manifest=manifest,
            variant=variant,
            run_id=run_id,
            dataversionid=derived_dataversionid,
        )

        manifest_out = run_dir / f"manifest_{run_id}.yaml"
        manifest_masked_out = run_dir / f"manifest_{run_id}.masked.yaml"
        _write_yaml(manifest_out, updated_manifest)
        _write_yaml(manifest_masked_out, _mask_manifest(updated_manifest))

        transform_out = run_dir / transform_path.name
        shutil.copy2(transform_path, transform_out)

        requested_gb = driver_memory_gb + variant.executor_instances * variant.executor_memory_gb
        run_meta = {
            "run_id": run_id,
            "manifest_path": str(manifest_out),
            "manifest_masked_path": str(manifest_masked_out),
            "transform_path": str(transform_out),
            "transform_sha256": transform_hash,
            "dataversionid": derived_dataversionid,
            "params": {
                "spark.sql.shuffle.partitions": variant.shuffle_partitions,
                "executor.cores": variant.executor_cores,
                "executor.instances": variant.executor_instances,
                "executor.memory_gb": variant.executor_memory_gb,
                "driver.memory_gb": driver_memory_gb,
            },
            "spark_ui": "",
            "runtime_seconds": None,
            "requested_gb": requested_gb,
            "requested_gb_seconds": None,
            "spill_gb": None,
            "output_files": None,
            "small_files": None,
            "conclusion": {
                "good": "",
                "bad": "",
                "next_params": "",
            },
        }
        (run_dir / f"run_{run_id}.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )

    print(f"Generated {len(variants)} runs in {RUNS_DIR}")


def cmd_record(args: argparse.Namespace) -> None:
    run_id = _build_run_id(int(args.run_id))
    run_dir = RUNS_DIR / f"run_{run_id}"
    run_meta_path = run_dir / f"run_{run_id}.json"
    if not run_meta_path.exists():
        raise SystemExit(f"Run metadata not found: {run_meta_path}")

    run = json.loads(run_meta_path.read_text(encoding="utf-8"))
    if args.spark_ui:
        run["spark_ui"] = args.spark_ui
    if args.runtime_seconds is not None:
        run["runtime_seconds"] = float(args.runtime_seconds)
    if args.requested_gb is not None:
        run["requested_gb"] = float(args.requested_gb)
    if run.get("runtime_seconds") is not None and run.get("requested_gb") is not None:
        run["requested_gb_seconds"] = float(run["requested_gb"]) * float(run["runtime_seconds"])
    if args.spill_gb is not None:
        run["spill_gb"] = float(args.spill_gb)
    if args.output_files is not None:
        run["output_files"] = int(args.output_files)
    if args.small_files is not None:
        run["small_files"] = int(args.small_files)
    if args.good is not None:
        run["conclusion"]["good"] = args.good
    if args.bad is not None:
        run["conclusion"]["bad"] = args.bad
    if args.next_params is not None:
        run["conclusion"]["next_params"] = args.next_params

    run_meta_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"Recorded results for run {run_id}")


def cmd_propose(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    config_path = Path(args.config)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    manifest = _read_yaml(manifest_path)
    spark_conf = _get_spark_conf(manifest)
    executor = _get_executor_spec(manifest)
    driver = _get_driver_spec(manifest)

    base_params = {
        "spark.sql.shuffle.partitions": int(spark_conf.get("spark.sql.shuffle.partitions", 200)),
        "executor.cores": int(executor.get("cores", 1)),
        "executor.instances": int(executor.get("instances", 1)),
        "executor.memory_gb": _parse_memory_gb(str(executor.get("memory", "4g"))),
    }
    driver_memory_gb = _parse_memory_gb(str(driver.get("memory", "4g")))

    constraints = {
        "spark.sql.shuffle.partitions": {"min": 200, "max": 10000},
        "executor.cores": {"min": 1, "max": 16},
        "executor.instances": {"min": 1, "max": 500},
        "executor.memory_gb": {"min": 1, "max": 256},
        "total_memory_gb": {"max": args.max_total_memory_gb},
    }

    history = []
    for run_path in sorted(RUNS_DIR.glob("run_*/run_*.json")):
        history.append(json.loads(run_path.read_text(encoding="utf-8")))
    if not history:
        raise SystemExit("No runs found in output/runs.")

    system, user = _build_tuning_prompt(history, base_params, constraints, args.history_url)
    llm = _load_llm_client(config_path)
    response = llm.chat(system, user)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"LLM response was not valid JSON: {exc}") from exc

    raw_params = parsed.get("params", {})
    variant = _apply_constraints(raw_params, constraints, driver_memory_gb)

    arguments = _get_arguments(manifest)
    dataversionid = ""
    for item in arguments:
        raw = item[2:] if item.startswith("--") else item
        if raw.startswith("Output.RAW.ViewingEvent.dataversionid="):
            dataversionid = raw.split("=", 1)[1]
            break
    if not dataversionid:
        raise SystemExit("Output.RAW.ViewingEvent.dataversionid not found in arguments.")

    _ensure_dirs()
    next_index = len(history) + 1
    run_id = _build_run_id(next_index)
    derived_dataversionid = _derive_dataversionid(dataversionid, next_index)
    updated_manifest = _update_manifest(
        manifest=manifest,
        variant=variant,
        run_id=run_id,
        dataversionid=derived_dataversionid,
    )

    run_dir = RUNS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = run_dir / f"manifest_{run_id}.yaml"
    manifest_masked_out = run_dir / f"manifest_{run_id}.masked.yaml"
    _write_yaml(manifest_out, updated_manifest)
    _write_yaml(manifest_masked_out, _mask_manifest(updated_manifest))

    decision = {
        "run_id": run_id,
        "manifest_path": str(manifest_out),
        "manifest_masked_path": str(manifest_masked_out),
        "params": {
            "spark.sql.shuffle.partitions": variant.shuffle_partitions,
            "executor.cores": variant.executor_cores,
            "executor.instances": variant.executor_instances,
            "executor.memory_gb": variant.executor_memory_gb,
            "driver.memory_gb": driver_memory_gb,
        },
        "rationale": parsed.get("rationale", ""),
        "constraints": constraints,
    }
    (run_dir / f"proposal_{run_id}.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print(f"Wrote next proposal to {manifest_out}")


def _run_kubectl(args: list[str]) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _wait_for_application(
    namespace: str,
    name: str,
    poll_seconds: int = 20,
    timeout_seconds: int = 6 * 60 * 60,
) -> dict[str, Any]:
    start = time.time()
    while True:
        raw = _run_kubectl(["get", "sparkapplication", name, "-n", namespace, "-o", "json"])
        obj = json.loads(raw)
        state = (
            obj.get("status", {})
            .get("applicationState", {})
            .get("state", "")
        )
        if state in {"COMPLETED", "FAILED"}:
            return obj
        if time.time() - start > timeout_seconds:
            raise SystemExit(f"Timed out waiting for {name} to finish.")
        time.sleep(poll_seconds)


def _find_driver_pod(namespace: str, app_name: str) -> str:
    labels = f"sparkoperator.k8s.io/app-name={app_name},spark-role=driver"
    raw = _run_kubectl(["get", "pods", "-n", namespace, "-l", labels, "-o", "json"])
    obj = json.loads(raw)
    items = obj.get("items", [])
    if items:
        return items[0]["metadata"]["name"]
    raw = _run_kubectl(["get", "pods", "-n", namespace, "-o", "json"])
    obj = json.loads(raw)
    for item in obj.get("items", []):
        name = item.get("metadata", {}).get("name", "")
        if name.startswith(f"{app_name}-driver"):
            return name
    raise SystemExit(f"Driver pod not found for app {app_name}")


def _app_id_from_status(obj: dict[str, Any]) -> str:
    status = obj.get("status", {})
    for key in ("sparkApplicationId", "appId", "applicationId"):
        value = status.get(key)
        if value:
            return str(value)
    return ""


def _extract_app_id(driver_logs: str) -> str:
    patterns = [
        r"app[-_]?id\s*[:=]\s*(app-\d{14}-\d{4})",
        r"(app-\d{14}-\d{4})",
        r"(spark-[a-f0-9]{32})",
    ]
    for pattern in patterns:
        match = re.search(pattern, driver_logs, re.IGNORECASE)
        if match:
            return match.group(1)
    raise SystemExit("Could not find appId in driver logs.")


def _parse_stage_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fGMT", "%Y-%m-%dT%H:%M:%SGMT"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _collect_metrics_from_stages(stages: list[dict[str, Any]]) -> dict[str, Any]:
    memory_spill = 0
    disk_spill = 0
    start_times: list[datetime] = []
    end_times: list[datetime] = []
    status_set = set()
    for stage in stages:
        memory_spill += int(stage.get("memoryBytesSpilled", 0) or 0)
        disk_spill += int(stage.get("diskBytesSpilled", 0) or 0)
        status_set.add(stage.get("status", ""))
        start = _parse_stage_time(stage.get("submissionTime") or stage.get("submissionTimest"))
        end = _parse_stage_time(stage.get("completionTime"))
        if start:
            start_times.append(start)
        if end:
            end_times.append(end)
    runtime_seconds = None
    if start_times and end_times:
        runtime_seconds = (max(end_times) - min(start_times)).total_seconds()
    return {
        "status": "COMPLETE" if "COMPLETE" in status_set else "UNKNOWN",
        "runtime_seconds": runtime_seconds,
        "spill_gb": (memory_spill + disk_spill) / (1024 ** 3),
    }


def _fetch_json(url: str) -> Any:
    from urllib.request import urlopen
    with urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _history_stages_url(base: str, app_id: str) -> str:
    return f"{base.rstrip('/')}/api/v1/applications/{app_id}/stages"


def _history_ui_url(base: str, app_id: str) -> str:
    return f"{base.rstrip('/')}/history/{app_id}/stages/"


def _load_iterations_from_env() -> Optional[int]:
    load_dotenv()
    raw = os.getenv("ITERATIONS")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit("ITERATIONS in .env must be an integer.")


def cmd_iterate(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    transform_path = Path(args.transform)
    config_path = Path(args.config)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not transform_path.exists():
        raise SystemExit(f"Transform not found: {transform_path}")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    iterations = args.iterations or _load_iterations_from_env()
    if not iterations:
        raise SystemExit("Iterations not provided. Use --iterations or set ITERATIONS in .env")

    base_manifest = _read_yaml(manifest_path)
    driver = _get_driver_spec(base_manifest)
    driver_memory_gb = _parse_memory_gb(str(driver.get("memory", "4g")))

    arguments = _get_arguments(base_manifest)
    dataversionid = ""
    for item in arguments:
        raw = item[2:] if item.startswith("--") else item
        if raw.startswith("Output.RAW.ViewingEvent.dataversionid="):
            dataversionid = raw.split("=", 1)[1]
            break
    if not dataversionid:
        raise SystemExit("Output.RAW.ViewingEvent.dataversionid not found in arguments.")

    namespace = base_manifest.get("metadata", {}).get("namespace", "default")

    constraints = {
        "spark.sql.shuffle.partitions": {"min": 200, "max": 10000},
        "executor.cores": {"min": 1, "max": 16},
        "executor.instances": {"min": 1, "max": 500},
        "executor.memory_gb": {"min": 1, "max": 256},
        "total_memory_gb": {"max": args.max_total_memory_gb},
    }

    llm = _load_llm_client(config_path)
    transform_hash = _hash_file(transform_path)
    history: list[dict[str, Any]] = []

    for iteration in range(1, iterations + 1):
        run_id = _build_run_id(iteration)
        run_dir = RUNS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if iteration == 1 and args.use_base_for_first:
            spark_conf = _get_spark_conf(base_manifest)
            executor = _get_executor_spec(base_manifest)
            variant = Variant(
                shuffle_partitions=int(spark_conf.get("spark.sql.shuffle.partitions", 200)),
                executor_cores=int(executor.get("cores", 1)),
                executor_instances=int(executor.get("instances", 1)),
                executor_memory_gb=_parse_memory_gb(str(executor.get("memory", "4g"))),
            )
            rationale = "Base config for first run."
        else:
            base_params = {
                "spark.sql.shuffle.partitions": int(
                    _get_spark_conf(base_manifest).get("spark.sql.shuffle.partitions", 200)
                ),
                "executor.cores": int(_get_executor_spec(base_manifest).get("cores", 1)),
                "executor.instances": int(_get_executor_spec(base_manifest).get("instances", 1)),
                "executor.memory_gb": _parse_memory_gb(str(_get_executor_spec(base_manifest).get("memory", "4g"))),
            }
            system, user = _build_tuning_prompt(history, base_params, constraints, args.history_url)
            response = llm.chat(system, user)
            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"LLM response was not valid JSON: {exc}") from exc
            variant = _apply_constraints(parsed.get("params", {}), constraints, driver_memory_gb)
            rationale = parsed.get("rationale", "")

        derived_dataversionid = _derive_dataversionid(dataversionid, iteration)
        manifest = _update_manifest(
            manifest=base_manifest,
            variant=variant,
            run_id=run_id,
            dataversionid=derived_dataversionid,
        )

        manifest_out = run_dir / f"manifest_{run_id}.yaml"
        manifest_masked_out = run_dir / f"manifest_{run_id}.masked.yaml"
        _write_yaml(manifest_out, manifest)
        _write_yaml(manifest_masked_out, _mask_manifest(manifest))

        transform_out = run_dir / transform_path.name
        shutil.copy2(transform_path, transform_out)

        requested_gb = driver_memory_gb + variant.executor_instances * variant.executor_memory_gb
        run_meta = {
            "run_id": run_id,
            "manifest_path": str(manifest_out),
            "manifest_masked_path": str(manifest_masked_out),
            "transform_path": str(transform_out),
            "transform_sha256": transform_hash,
            "dataversionid": derived_dataversionid,
            "params": {
                "spark.sql.shuffle.partitions": variant.shuffle_partitions,
                "executor.cores": variant.executor_cores,
                "executor.instances": variant.executor_instances,
                "executor.memory_gb": variant.executor_memory_gb,
                "driver.memory_gb": driver_memory_gb,
            },
            "rationale": rationale,
            "spark_ui": "",
            "runtime_seconds": None,
            "requested_gb": requested_gb,
            "requested_gb_seconds": None,
            "spill_gb": None,
            "output_files": None,
            "small_files": None,
            "app_id": "",
            "application_state": "",
            "history_api": "",
            "conclusion": {
                "good": "",
                "bad": "",
                "next_params": "",
            },
        }

        (run_dir / f"run_{run_id}.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )

        _run_kubectl(["apply", "-f", str(manifest_out)])
        status_obj = _wait_for_application(namespace, manifest["metadata"]["name"])
        final_state = (
            status_obj.get("status", {})
            .get("applicationState", {})
            .get("state", "")
        )

        app_id = _app_id_from_status(status_obj)
        if not app_id:
            driver_pod = _find_driver_pod(namespace, manifest["metadata"]["name"])
            log_args = ["logs", driver_pod, "-n", namespace]
            if args.driver_container:
                log_args.extend(["-c", args.driver_container])
            log_args.extend(["--tail", "2000"])
            logs = _run_kubectl(log_args)
            app_id = _extract_app_id(logs)

        stages_url = _history_stages_url(args.history_url, app_id)
        stages = _fetch_json(stages_url)
        metrics = _collect_metrics_from_stages(stages)

        run_meta.update(
            {
                "app_id": app_id,
                "application_state": final_state,
                "history_api": stages_url,
                "spark_ui": _history_ui_url(args.history_url, app_id),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "spill_gb": metrics.get("spill_gb"),
            }
        )
        if run_meta["runtime_seconds"] is not None:
            run_meta["requested_gb_seconds"] = run_meta["requested_gb"] * run_meta["runtime_seconds"]

        (run_dir / f"run_{run_id}.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )
        history.append(run_meta)

    summary = _summarize_runs()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _summarize_runs() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_path in sorted(RUNS_DIR.glob("run_*/run_*.json")):
        runs.append(json.loads(run_path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit("No runs found in output/runs.")

    scored: list[tuple[float, dict[str, Any]]] = []
    for run in runs:
        if run.get("requested_gb_seconds") is None:
            continue
        scored.append((float(run["requested_gb_seconds"]), run))

    if not scored:
        raise SystemExit("No runs with requested_gb_seconds recorded.")

    scored.sort(key=lambda item: item[0])
    best_score, best = scored[0]

    for run in runs:
        run["is_best"] = run.get("run_id") == best.get("run_id")
        run_path = Path(run["manifest_path"]).parent / f"run_{run['run_id']}.json"
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    return {
        "best_run_id": best["run_id"],
        "best_score": best_score,
        "best_params": best.get("params", {}),
        "runs_scored": [
            {
                "run_id": run["run_id"],
                "score": score,
                "spark_ui": run.get("spark_ui", ""),
                "params": run.get("params", {}),
                "conclusion": run.get("conclusion", {}),
            }
            for score, run in scored
        ],
    }


def cmd_summarize(args: argparse.Namespace) -> None:
    summary = _summarize_runs()
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark tuning experiment helper.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Generate N run manifests and metadata.")
    plan.add_argument("--manifest", required=True, help="Path to base SparkApplication YAML.")
    plan.add_argument("--transform", required=True, help="Path to transformation code.")
    plan.add_argument("--runs", type=int, required=True, help="Number of runs to generate.")
    plan.add_argument(
        "--max-total-memory-gb",
        type=int,
        default=500,
        help="Max total requested memory (driver + executors).",
    )
    plan.set_defaults(func=cmd_plan)

    record = sub.add_parser("record", help="Record results for a run.")
    record.add_argument("--run-id", required=True, help="Run index (e.g. 1).")
    record.add_argument("--spark-ui", default=None, help="Spark UI link.")
    record.add_argument("--runtime-seconds", type=float, default=None)
    record.add_argument("--requested-gb", type=float, default=None)
    record.add_argument("--spill-gb", type=float, default=None)
    record.add_argument("--output-files", type=int, default=None)
    record.add_argument("--small-files", type=int, default=None)
    record.add_argument("--good", default=None, help="What was good in this run.")
    record.add_argument("--bad", default=None, help="What was bad in this run.")
    record.add_argument("--next-params", default=None, help="Suggested next params.")
    record.set_defaults(func=cmd_record)

    propose = sub.add_parser("propose", help="Ask LLM to propose next run parameters.")
    propose.add_argument("--manifest", required=True, help="Path to base SparkApplication YAML.")
    propose.add_argument("--config", required=True, help="Path to config.yaml for LLM router.")
    propose.add_argument("--history-url", required=True, help="Spark History Server base URL.")
    propose.add_argument(
        "--max-total-memory-gb",
        type=int,
        default=500,
        help="Max total requested memory (driver + executors).",
    )
    propose.set_defaults(func=cmd_propose)

    iterate = sub.add_parser("iterate", help="Run full tuning loop with LLM.")
    iterate.add_argument("--manifest", required=True, help="Path to base SparkApplication YAML.")
    iterate.add_argument("--transform", required=True, help="Path to transformation code.")
    iterate.add_argument("--config", required=True, help="Path to config.yaml for LLM router.")
    iterate.add_argument("--history-url", required=True, help="Spark History Server base URL.")
    iterate.add_argument("--iterations", type=int, default=None, help="Number of iterations.")
    iterate.add_argument(
        "--max-total-memory-gb",
        type=int,
        default=500,
        help="Max total requested memory (driver + executors).",
    )
    iterate.add_argument(
        "--use-base-for-first",
        action="store_true",
        help="Use base manifest params for first run before LLM tuning.",
    )
    iterate.add_argument(
        "--driver-container",
        default=None,
        help="Driver container name for kubectl logs (optional).",
    )
    iterate.set_defaults(func=cmd_iterate)

    summarize = sub.add_parser("summarize", help="Summarize runs and pick best config.")
    summarize.add_argument("--scoring", default=None, help="Path to scoring JSON.")
    summarize.set_defaults(func=cmd_summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
