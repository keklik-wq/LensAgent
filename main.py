from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.agent_shell.config import AppConfig
from src.agent_shell.factory import (
    build_llm_client,
    build_spark_history_provider,
    build_spark_runtime,
)


OUTPUT_DIR = Path("output")
RUNS_DIR = OUTPUT_DIR / "runs"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_FILE = LOG_DIR / "agent.log"


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


def _update_manifest(
    manifest: dict[str, Any],
    variant: Variant,
    run_id: str,
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
    return data


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_tuning_prompt(
    history: list[dict[str, Any]],
    base_params: dict[str, Any],
    constraints: dict[str, Any],
    history_label: str,
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
        "history_url": history_label,
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


def _apply_constraints(
    params: dict[str, Any],
    constraints: dict[str, Any],
    driver_memory_gb: int,
) -> Variant:
    def _clamp(name: str, value: int) -> int:
        limits = constraints[name]
        return max(int(limits["min"]), min(int(limits["max"]), value))

    shuffle = _clamp("spark.sql.shuffle.partitions", int(params.get("spark.sql.shuffle.partitions", 200)))
    cores = _clamp("executor.cores", int(params.get("executor.cores", 1)))
    instances = _clamp("executor.instances", int(params.get("executor.instances", 1)))
    mem_gb = _clamp("executor.memory_gb", int(params.get("executor.memory_gb", 4)))

    max_total = int(constraints["total_memory_gb"]["max"])
    if driver_memory_gb > max_total:
        raise SystemExit(
            f"Driver memory ({driver_memory_gb} GB) exceeds total memory limit ({max_total} GB)."
        )

    total_gb = driver_memory_gb + instances * mem_gb
    if total_gb > max_total:
        max_executors_gb = max_total - driver_memory_gb
        if max_executors_gb < mem_gb:
            mem_gb = max(1, max_executors_gb)
            instances = max(1, int(max_executors_gb / max(mem_gb, 1)))
        else:
            instances = max(1, int(max_executors_gb / mem_gb))

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
            for key, value in node.items():
                if SENSITIVE_KEY_RE.search(str(key)):
                    masked[key] = _sanitize_value(str(value))
                else:
                    masked[key] = mask_node(value)
            return masked
        if isinstance(node, list):
            return [mask_node(item) for item in node]
        return node

    return mask_node(data)


def _build_run_id(index: int) -> str:
    return f"{index:03d}"


def _ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if not root_logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
    return logging.getLogger("lens-agent")


def _parse_stage_time(value: str | None) -> datetime | None:
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


def _load_stages_with_retry(
    app_id: str,
    history_provider: Any,
    poll_seconds: float = 2.0,
    timeout_seconds: int = 120,
) -> tuple[str, list[dict[str, Any]]]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    candidate_app_id = app_id
    while time.time() < deadline:
        try:
            stages = history_provider.get_stages(candidate_app_id)
        except Exception as exc:
            last_error = exc
            latest_app_id = history_provider.latest_app_id()
            if latest_app_id and latest_app_id != candidate_app_id:
                candidate_app_id = latest_app_id
            time.sleep(poll_seconds)
            continue
        if stages:
            return candidate_app_id, stages
        time.sleep(poll_seconds)
    if last_error is not None:
        raise SystemExit(f"Failed to load stages for {candidate_app_id}: {last_error}") from last_error
    raise SystemExit(f"Timed out waiting for stages for {candidate_app_id}")


def _generate_variants(
    base: Variant,
    max_total_memory_gb: int,
    driver_memory_gb: int,
    count: int,
) -> list[Variant]:
    cores_candidates = sorted(
        {
            max(1, base.executor_cores - 1),
            base.executor_cores,
            base.executor_cores + 1,
            base.executor_cores + 2,
        }
    )
    instances_candidates = sorted(
        {
            max(1, base.executor_instances - 2),
            base.executor_instances,
            base.executor_instances + 2,
            base.executor_instances + 4,
        }
    )
    memory_candidates = sorted(
        {
            max(1, base.executor_memory_gb - 4),
            base.executor_memory_gb,
            base.executor_memory_gb + 4,
            base.executor_memory_gb + 8,
        }
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


def run_loop(args: argparse.Namespace) -> None:
    logger = logging.getLogger("lens-agent")
    manifest_path = Path(args.manifest)
    transform_path = Path(args.transform)
    config_path = Path(args.config)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not transform_path.exists():
        raise SystemExit(f"Transform not found: {transform_path}")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    load_dotenv()
    app_config = AppConfig.load(config_path)
    logger.info("Loaded config from %s (runtime=%s, history=%s, llm=%s)", config_path, app_config.spark_runtime.backend, app_config.spark_history.backend, app_config.llm.backend)

    base_manifest = _read_yaml(manifest_path)
    driver = _get_driver_spec(base_manifest)
    driver_memory_gb = _parse_memory_gb(str(driver.get("memory", "4g")))

    namespace = args.namespace or base_manifest.get("metadata", {}).get("namespace", "default")
    base_manifest.setdefault("metadata", {})["namespace"] = namespace

    runtime = build_spark_runtime(app_config, kube_context=args.kube_context)
    history_provider = build_spark_history_provider(app_config, base_url_override=args.history_url)
    llm = build_llm_client(app_config)
    logger.info("Starting run loop (iterations=%s, namespace=%s)", args.iterations, namespace)

    constraints = {
        "spark.sql.shuffle.partitions": {"min": 200, "max": 10000},
        "executor.cores": {"min": 1, "max": 16},
        "executor.instances": {"min": 1, "max": 500},
        "executor.memory_gb": {"min": 1, "max": 256},
        "total_memory_gb": {"max": args.max_total_memory_gb},
    }

    transform_hash = _hash_file(transform_path)
    history: list[dict[str, Any]] = []
    base_params = {
        "spark.sql.shuffle.partitions": int(_get_spark_conf(base_manifest).get("spark.sql.shuffle.partitions", 200)),
        "executor.cores": int(_get_executor_spec(base_manifest).get("cores", 1)),
        "executor.instances": int(_get_executor_spec(base_manifest).get("instances", 1)),
        "executor.memory_gb": _parse_memory_gb(str(_get_executor_spec(base_manifest).get("memory", "4g"))),
    }

    for iteration in range(1, args.iterations + 1):
        run_id = _build_run_id(iteration)
        run_dir = RUNS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run %s: preparing manifest and inputs", run_id)

        if iteration == 1 and args.use_base_for_first:
            variant = Variant(
                shuffle_partitions=base_params["spark.sql.shuffle.partitions"],
                executor_cores=base_params["executor.cores"],
                executor_instances=base_params["executor.instances"],
                executor_memory_gb=base_params["executor.memory_gb"],
            )
            rationale = "Base config for first run."
        elif iteration == 1 and args.use_random_for_first:
            variant = _generate_variants(
                base=Variant(
                    shuffle_partitions=base_params["spark.sql.shuffle.partitions"],
                    executor_cores=base_params["executor.cores"],
                    executor_instances=base_params["executor.instances"],
                    executor_memory_gb=base_params["executor.memory_gb"],
                ),
                max_total_memory_gb=args.max_total_memory_gb,
                driver_memory_gb=driver_memory_gb,
                count=1,
            )[0]
            rationale = "Randomized candidate for first run."
        else:
            system, user = _build_tuning_prompt(
                history,
                base_params,
                constraints,
                history_provider.ui_url("latest"),
            )
            response = llm.chat(system, user)
            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"LLM response was not valid JSON: {exc}") from exc
            variant = _apply_constraints(parsed.get("params", {}), constraints, driver_memory_gb)
            rationale = str(parsed.get("rationale", ""))

        manifest = _update_manifest(manifest=base_manifest, variant=variant, run_id=run_id)
        logger.info(
            "Run %s: variant shuffle=%s cores=%s instances=%s mem_gb=%s",
            run_id,
            variant.shuffle_partitions,
            variant.executor_cores,
            variant.executor_instances,
            variant.executor_memory_gb,
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
            "driver_logs": "",
            "driver_logs_path": "",
        }
        (run_dir / f"run_{run_id}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        logger.info("Run %s: submitting Spark job", run_id)
        result = runtime.run_application(manifest, namespace, driver_container=args.driver_container)
        driver_logs_path = run_dir / "driver.log"
        driver_logs_path.write_text(result.driver_logs or "", encoding="utf-8")
        resolved_app_id, stages = _load_stages_with_retry(
            result.app_id,
            history_provider,
            poll_seconds=app_config.spark_history.poll_seconds,
            timeout_seconds=app_config.spark_history.timeout_seconds,
        )
        metrics = _collect_metrics_from_stages(stages)
        logger.info(
            "Run %s: completed app_id=%s state=%s runtime_seconds=%s",
            run_id,
            resolved_app_id,
            result.final_state,
            metrics.get("runtime_seconds"),
        )

        run_meta.update(
            {
                "app_id": resolved_app_id,
                "application_state": result.final_state,
                "history_api": history_provider.stages_url(resolved_app_id),
                "spark_ui": history_provider.ui_url(resolved_app_id),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "spill_gb": metrics.get("spill_gb"),
                "driver_logs": result.driver_logs,
                "driver_logs_path": str(driver_logs_path),
            }
        )
        if run_meta["runtime_seconds"] is not None:
            run_meta["requested_gb_seconds"] = run_meta["requested_gb"] * run_meta["runtime_seconds"]

        (run_dir / f"run_{run_id}.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        history.append(run_meta)

    summary = _summarize_runs()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", OUTPUT_DIR / "summary.json")


def _summarize_runs() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_path in sorted(RUNS_DIR.glob("run_*/run_*.json")):
        runs.append(json.loads(run_path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit("No runs found in output/runs.")

    scored: list[tuple[float, dict[str, Any]]] = []
    for run in runs:
        if run.get("requested_gb_seconds") is not None:
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
            }
            for score, run in scored
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark tuning loop with replaceable infrastructure backends.")
    parser.add_argument("--manifest", required=True, help="Path to SparkApplication YAML.")
    parser.add_argument("--transform", required=True, help="Path to transformation code file.")
    parser.add_argument("--config", required=True, help="Path to config.yaml.")
    parser.add_argument("--history-url", default=None, help="History base URL override.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations to run.")
    parser.add_argument(
        "--max-total-memory-gb",
        type=int,
        default=500,
        help="Max total requested memory (driver + executors).",
    )
    parser.add_argument(
        "--use-base-for-first",
        action="store_true",
        help="Use base manifest params for first run before tuning.",
    )
    parser.add_argument(
        "--use-random-for-first",
        action="store_true",
        help="Generate a randomized valid config for the first run.",
    )
    parser.add_argument(
        "--driver-container",
        default=None,
        help="Driver container name for Kubernetes runtime.",
    )
    parser.add_argument(
        "--kube-context",
        default=None,
        help="Kubernetes context override for kubernetes runtime.",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Namespace override.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.use_base_for_first and args.use_random_for_first:
        raise SystemExit("Choose only one of --use-base-for-first or --use-random-for-first.")
    _ensure_dirs()
    _setup_logging()
    run_loop(args)


if __name__ == "__main__":
    main()
