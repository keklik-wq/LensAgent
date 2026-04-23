from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import secrets
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.agent_shell.config import AppConfig, TuningParamConfig
from src.agent_shell.factory import (
    build_llm_client,
    build_spark_history_provider,
    build_spark_runtime,
)

OUTPUT_DIR = Path("output")
RUNS_DIR = OUTPUT_DIR / "runs"
LOG_DIR = OUTPUT_DIR / "logs"
LOG_FILE = LOG_DIR / "agent.log"


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


def _is_int_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    text = str(value).strip()
    return bool(re.fullmatch(r"[+-]?\d+", text))


def _is_float_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    try:
        float(text)
    except ValueError:
        return False
    return True


def _get_by_path(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_by_path(data: dict[str, Any], path: list[str], value: Any) -> None:
    current: dict[str, Any] = data
    for part in path[:-1]:
        next_node = current.get(part)
        if not isinstance(next_node, dict):
            next_node = {}
            current[part] = next_node
        current = next_node
    current[path[-1]] = value


def _coerce_param_value(value: Any, param_type: str) -> Any:
    if param_type == "int":
        return int(value)
    if param_type == "float":
        return float(value)
    if param_type == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y"}
    if param_type == "memory_gb":
        if isinstance(value, (int, float)):
            return int(value)
        return int(_parse_memory_gb(str(value)))
    if param_type == "enum":
        return str(value).strip()
    return str(value)


def _constraint_value_kind(spec: TuningParamConfig, value: Any = None) -> str | None:
    if spec.type == "int":
        return "int"
    if spec.type == "memory_gb":
        return "memory_gb"
    if spec.type == "float":
        return "float"
    if spec.type != "str":
        return None

    samples = [item for item in (value, spec.min, spec.max, spec.default) if item is not None]
    if not samples:
        return None
    if all(_is_int_like(item) for item in samples):
        return "int"
    if all(_is_float_like(item) for item in samples):
        return "float"
    return None


def _coerce_constraint_value(value: Any, spec: TuningParamConfig, *, field_name: str) -> Any:
    if spec.type == "enum":
        return _coerce_enum_value(value, spec, field_name=field_name)
    kind = _constraint_value_kind(spec, value)
    if kind == "int":
        if not _is_int_like(value):
            raise ValueError(
                f"{field_name} for tuning param expects an integer-like value, got {value!r}."
            )
        return int(str(value).strip())
    if kind == "float":
        if not _is_float_like(value):
            raise ValueError(
                f"{field_name} for tuning param expects a numeric value, got {value!r}."
            )
        return float(str(value).strip())
    if kind == "memory_gb":
        try:
            return _coerce_param_value(value, spec.type)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} for tuning param expects a memory value like '4g', got {value!r}."
            ) from exc
    return _coerce_param_value(value, spec.type)


def _restore_param_value(value: Any, spec: TuningParamConfig) -> Any:
    kind = _constraint_value_kind(spec, value)
    if spec.type == "str" and kind in {"int", "float"}:
        return str(value)
    return _coerce_param_value(value, spec.type)


def _coerce_enum_value(value: Any, spec: TuningParamConfig, *, field_name: str) -> str:
    if not spec.values:
        raise ValueError(f"{field_name} for tuning param has no configured enum values.")
    normalized = str(value).strip()
    if normalized not in spec.values:
        raise ValueError(
            f"{field_name} for tuning param expects one of {spec.values!r}, got {value!r}."
        )
    return normalized


def _format_param_value(value: Any, param_type: str) -> Any:
    if param_type == "memory_gb":
        return _format_memory_gb(int(value))
    if param_type == "int":
        return int(value)
    if param_type == "float":
        return float(value)
    if param_type == "bool":
        return bool(value)
    return value


def _build_constraints(
    tuning_params: dict[str, TuningParamConfig],
    max_total_memory_gb: int | None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    for name, spec in tuning_params.items():
        if spec.min is None and spec.max is None:
            continue
        constraints[name] = {
            "min": spec.min,
            "max": spec.max,
        }
    if max_total_memory_gb is not None:
        constraints["total_memory_gb"] = {"max": max_total_memory_gb}
    return constraints


def _build_response_schema(tuning_params: dict[str, TuningParamConfig]) -> dict[str, Any]:
    schema: dict[str, Any] = {"params": {}, "rationale": "string"}
    for name, spec in tuning_params.items():
        param_type = "string"
        if spec.type in {"int", "float", "bool"}:
            param_type = spec.type
        elif spec.type == "memory_gb":
            param_type = "int"
        elif spec.type == "enum":
            schema["params"][name] = {"type": "string", "enum": list(spec.values or [])}
            continue
        schema["params"][name] = param_type
    return schema


def _build_tunable_param_specs(
    tuning_params: dict[str, TuningParamConfig],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name, spec in tuning_params.items():
        specs[name] = {
            "type": spec.type,
            "min": spec.min,
            "max": spec.max,
            "values": spec.values,
            "default": spec.default,
            "path": spec.path,
        }
    return specs


def _build_base_params(
    manifest: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
) -> dict[str, Any]:
    base_params: dict[str, Any] = {}
    for name, spec in tuning_params.items():
        raw = _get_by_path(manifest, spec.path)
        if raw is None:
            if spec.default is None:
                raise ValueError(
                    f"Manifest missing value for tuning param {name} ({'.'.join(spec.path)})."
                )
            raw = spec.default
        base_params[name] = _coerce_param_value(raw, spec.type)
    return base_params


def _validate_params_within_bounds(
    params: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
    *,
    source_label: str,
) -> None:
    for name, spec in tuning_params.items():
        if name not in params:
            continue
        value = _coerce_constraint_value(params[name], spec, field_name=f"{source_label} value")
        if spec.type == "enum":
            continue
        if spec.min is not None:
            min_value = _coerce_constraint_value(spec.min, spec, field_name=f"{name}.min")
            if value < min_value:
                raise ValueError(
                    f"{source_label} for {name} is {params[name]!r}, below configured minimum {spec.min!r}."
                )
        if spec.max is not None:
            max_value = _coerce_constraint_value(spec.max, spec, field_name=f"{name}.max")
            if value > max_value:
                raise ValueError(
                    f"{source_label} for {name} is {params[name]!r}, above configured maximum {spec.max!r}."
                )


def _is_spark_conf_path(path: list[str]) -> bool:
    return len(path) >= 2 and path[0] == "spec" and path[1] == "sparkConf"


def _apply_params_to_manifest(
    manifest: dict[str, Any],
    params: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
) -> dict[str, Any]:
    data = _deep_copy(manifest)
    for name, spec in tuning_params.items():
        if name not in params:
            continue
        formatted = _format_param_value(params[name], spec.type)
        if _is_spark_conf_path(spec.path):
            formatted = str(formatted)
        _set_by_path(data, spec.path, formatted)
    return data


def _update_manifest_name(
    manifest: dict[str, Any],
    run_id: str,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    data = _deep_copy(manifest)
    data.setdefault("metadata", {})
    base_name = data["metadata"].get("name", "spark-app")
    suffix = f"-{campaign_id}" if campaign_id else ""
    data["metadata"]["name"] = f"{base_name}{suffix}-r{run_id}"
    return data


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_tuning_prompt(
    system_prompt: str,
    history: list[dict[str, Any]],
    base_params: dict[str, Any],
    tunable_param_specs: dict[str, dict[str, Any]],
    constraints: dict[str, Any],
    response_schema: dict[str, Any],
    history_label: str,
) -> tuple[str, str]:
    best_previous_run = _select_best_history_entry(history)
    latest_run = history[-1] if history else None
    user = {
        "task": "Propose next Spark config parameters for the next run.",
        "base_params": base_params,
        "tunable_params": tunable_param_specs,
        "history": history,
        "best_previous_run": best_previous_run,
        "latest_run": latest_run,
        "history_url": history_label,
        "constraints": constraints,
        "response_schema": response_schema,
    }
    return system_prompt, json.dumps(user, ensure_ascii=True)


def _build_retry_user_prompt(
    user_prompt: str,
    raw_response: str,
    error_message: str,
    attempt: int,
) -> str:
    payload = json.loads(user_prompt)
    payload["retry_feedback"] = {
        "attempt": attempt,
        "error": error_message,
        "invalid_response_excerpt": raw_response[:1000],
        "instruction": "Your previous response was not valid JSON. Return ONLY valid JSON that matches response_schema exactly.",
    }
    return json.dumps(payload, ensure_ascii=True)


def _select_best_history_entry(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [item for item in history if item.get("requested_gb_seconds") is not None]
    if not scored:
        return None
    return min(scored, key=lambda item: float(item["requested_gb_seconds"]))


def _build_llm_history_entry(run_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_meta.get("run_id"),
        "params": run_meta.get("params"),
        "rationale": run_meta.get("rationale"),
        "application_state": run_meta.get("application_state"),
        "runtime_seconds": run_meta.get("runtime_seconds"),
        "requested_gb": run_meta.get("requested_gb"),
        "requested_gb_seconds": run_meta.get("requested_gb_seconds"),
        "spill_gb": run_meta.get("spill_gb"),
        "output_files": run_meta.get("output_files"),
        "small_files": run_meta.get("small_files"),
        "spark_ui": run_meta.get("spark_ui"),
        "history_api": run_meta.get("history_api"),
    }


def _params_signature(
    params: dict[str, Any], names: list[str] | tuple[str, ...] | None = None
) -> tuple[tuple[str, Any], ...]:
    if names is None:
        items = params.items()
    else:
        items = ((name, params.get(name)) for name in names)
    return tuple(sorted(items))


def _history_param_signatures(
    history: list[dict[str, Any]],
    tuning_param_names: list[str],
) -> set[tuple[tuple[str, Any], ...]]:
    signatures: set[tuple[tuple[str, Any], ...]] = set()
    for item in history:
        params = item.get("params")
        if isinstance(params, dict):
            signatures.add(_params_signature(params, tuning_param_names))
    return signatures


def _next_candidate_values(current: Any, spec: TuningParamConfig) -> list[Any]:
    if spec.type == "enum":
        current_value = str(current).strip()
        return [value for value in (spec.values or []) if value != current_value]
    if spec.type == "bool":
        return [not bool(current)]
    if spec.type == "float":
        current_value = float(current)
        step = max(abs(current_value) * 0.1, 0.1)
        return [current_value - step, current_value + step]
    if spec.type == "memory_gb":
        current_value = int(current)
        return [current_value - 1, current_value + 1, current_value - 2, current_value + 2]
    current_value = int(current)
    return [current_value - 1, current_value + 1, current_value - 2, current_value + 2]


def _resolve_duplicate_params(
    params: dict[str, Any],
    history: list[dict[str, Any]],
    base_params: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
    driver_memory_gb: int,
    max_total_memory_gb: int | None,
) -> dict[str, Any]:
    tuning_param_names = list(tuning_params.keys())
    existing = _history_param_signatures(history, tuning_param_names)
    signature = _params_signature(params, tuning_param_names)
    if signature not in existing:
        return params

    for name, spec in tuning_params.items():
        if name not in params:
            continue
        for candidate_value in _next_candidate_values(params[name], spec):
            candidate = dict(params)
            candidate[name] = candidate_value
            try:
                resolved = _apply_constraints(
                    candidate,
                    base_params,
                    tuning_params,
                    driver_memory_gb,
                    max_total_memory_gb,
                )
            except (TypeError, ValueError):
                continue
            if _params_signature(resolved, tuning_param_names) not in existing:
                return resolved

    raise ValueError("Could not find a unique parameter configuration for the next run.")


def _apply_constraints(
    params: dict[str, Any],
    base_params: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
    driver_memory_gb: int,
    max_total_memory_gb: int | None,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name, spec in tuning_params.items():
        value = params.get(name, base_params.get(name, spec.default))
        if value is None:
            raise ValueError(f"Missing tuning parameter value for {name}.")
        coerced = _coerce_constraint_value(value, spec, field_name=f"{name} value")
        if spec.type != "enum" and spec.min is not None:
            coerced = max(_coerce_constraint_value(spec.min, spec, field_name=f"{name}.min"), coerced)
        if spec.type != "enum" and spec.max is not None:
            coerced = min(_coerce_constraint_value(spec.max, spec, field_name=f"{name}.max"), coerced)
        resolved[name] = _restore_param_value(coerced, spec)

    if max_total_memory_gb is not None:
        mem_gb = resolved.get("executor.memory_gb")
        instances = resolved.get("executor.instances")
        if isinstance(mem_gb, (int, float)) and isinstance(instances, (int, float)):
            if driver_memory_gb > max_total_memory_gb:
                raise ValueError(
                    f"Driver memory ({driver_memory_gb} GB) exceeds total memory limit ({max_total_memory_gb} GB)."
                )
            total_gb = driver_memory_gb + int(instances) * int(mem_gb)
            if total_gb > max_total_memory_gb:
                max_executors_gb = max_total_memory_gb - driver_memory_gb
                if max_executors_gb < mem_gb:
                    mem_gb = max(1, int(max_executors_gb))
                    instances = max(1, int(max_executors_gb / max(mem_gb, 1)))
                else:
                    instances = max(1, int(max_executors_gb / mem_gb))
                resolved["executor.memory_gb"] = int(mem_gb)
                resolved["executor.instances"] = int(instances)

    return resolved


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


def _build_campaign_id() -> str:
    return secrets.token_hex(2)


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
        "spill_gb": (memory_spill + disk_spill) / (1024**3),
    }


def _load_stages_with_retry(
    app_id: str,
    history_provider: Any,
    poll_seconds: float = 2.0,
    timeout_seconds: int = 120,
) -> tuple[str, list[dict[str, Any]]]:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            stages = history_provider.get_stages(app_id)
        except Exception as exc:
            last_error = exc
            time.sleep(poll_seconds)
            continue
        if stages:
            return app_id, stages
        time.sleep(poll_seconds)
    if last_error is not None:
        raise RuntimeError(f"Failed to load stages for {app_id}: {last_error}") from last_error
    raise RuntimeError(f"Timed out waiting for stages for {app_id}")


def _generate_random_params(
    base_params: dict[str, Any],
    tuning_params: dict[str, TuningParamConfig],
    count: int,
) -> list[dict[str, Any]]:
    random.seed(42)
    variants: list[dict[str, Any]] = []
    for _ in range(count):
        chosen: dict[str, Any] = {}
        for name, spec in tuning_params.items():
            if spec.type == "enum":
                if not spec.values:
                    raise ValueError(f"Enum tuning parameter {name} has no values.")
                chosen[name] = random.choice(spec.values)
                continue
            min_value = spec.min if spec.min is not None else base_params.get(name)
            max_value = spec.max if spec.max is not None else base_params.get(name)
            if min_value is None or max_value is None:
                chosen[name] = base_params.get(name)
                continue
            if spec.type == "float":
                chosen[name] = random.uniform(float(min_value), float(max_value))
            elif spec.type == "bool":
                chosen[name] = random.choice([True, False])
            else:
                chosen[name] = random.randint(int(min_value), int(max_value))
        variants.append(chosen)
    return variants


def _request_tuning_candidate(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    llm_json_retries: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    current_user_prompt = user_prompt
    last_error: Exception | None = None
    for attempt in range(1, llm_json_retries + 2):
        response = llm.chat(system_prompt, current_user_prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt > llm_json_retries:
                break
            logger.warning(
                "LLM returned invalid JSON on attempt %s/%s: %s",
                attempt,
                llm_json_retries + 1,
                exc,
            )
            current_user_prompt = _build_retry_user_prompt(
                current_user_prompt,
                response.content,
                str(exc),
                attempt + 1,
            )
    raise ValueError(f"LLM response was not valid JSON: {last_error}") from last_error


def run_loop(args: argparse.Namespace) -> None:
    logger = logging.getLogger("lens-agent")
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    load_dotenv()
    app_config = AppConfig.load(config_path)
    manifest_path = (app_config.config_dir / app_config.run.manifest).resolve()
    transform_path = (app_config.config_dir / app_config.run.transform).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not transform_path.exists():
        raise FileNotFoundError(f"Transform not found: {transform_path}")
    logger.info(
        "Loaded config from %s (runtime=%s, history=%s, llm=%s, first_run_mode=%s)",
        config_path,
        app_config.spark_runtime.backend,
        app_config.spark_history.backend,
        app_config.llm.backend,
        app_config.run.first_run_mode,
    )

    base_manifest = _read_yaml(manifest_path)
    driver_memory_raw = _get_by_path(base_manifest, ["spec", "driver", "memory"]) or "4g"
    driver_memory_gb = _parse_memory_gb(str(driver_memory_raw))

    namespace = args.namespace or base_manifest.get("metadata", {}).get("namespace", "default")
    base_manifest.setdefault("metadata", {})["namespace"] = namespace

    runtime = build_spark_runtime(app_config, kube_context=args.kube_context)
    history_provider = build_spark_history_provider(app_config, base_url_override=args.history_url)
    llm = build_llm_client(app_config)

    iterations = args.iterations if args.iterations is not None else app_config.tuning.iterations
    logger.info("Starting run loop (iterations=%s, namespace=%s)", iterations, namespace)

    tuning_params = app_config.tuning.params
    max_total_memory_gb = (
        args.max_total_memory_gb
        if args.max_total_memory_gb is not None
        else app_config.tuning.total_memory_gb_max
    )
    constraints = _build_constraints(tuning_params, max_total_memory_gb)
    response_schema = _build_response_schema(tuning_params)
    tunable_param_specs = _build_tunable_param_specs(tuning_params)

    transform_hash = _hash_file(transform_path)
    campaign_id = _build_campaign_id()
    history: list[dict[str, Any]] = []
    base_params = _build_base_params(base_manifest, tuning_params)
    _validate_params_within_bounds(
        base_params,
        tuning_params,
        source_label="Base manifest value",
    )

    for iteration in range(1, iterations + 1):
        run_id = _build_run_id(iteration)
        run_dir = RUNS_DIR / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run %s: preparing manifest and inputs", run_id)

        if iteration == 1 and app_config.run.first_run_mode == "base":
            chosen = _apply_constraints(
                {},
                base_params,
                tuning_params,
                driver_memory_gb,
                max_total_memory_gb,
            )
            rationale = "Base config for first run."
        elif iteration == 1 and app_config.run.first_run_mode == "random":
            candidate = _generate_random_params(base_params, tuning_params, 1)[0]
            chosen = _apply_constraints(
                candidate,
                base_params,
                tuning_params,
                driver_memory_gb,
                max_total_memory_gb,
            )
            rationale = "Randomized candidate for first run."
        else:
            system, user = _build_tuning_prompt(
                app_config.tuning.prompt,
                history,
                base_params,
                tunable_param_specs,
                constraints,
                response_schema,
                history_provider.ui_url("latest"),
            )
            parsed = _request_tuning_candidate(
                llm=llm,
                system_prompt=system,
                user_prompt=user,
                llm_json_retries=app_config.tuning.llm_json_retries,
                logger=logger,
            )
            chosen = _apply_constraints(
                parsed.get("params", {}),
                base_params,
                tuning_params,
                driver_memory_gb,
                max_total_memory_gb,
            )
            rationale = str(parsed.get("rationale", ""))
            chosen = _resolve_duplicate_params(
                chosen,
                history,
                base_params,
                tuning_params,
                driver_memory_gb,
                max_total_memory_gb,
            )

        manifest = _update_manifest_name(base_manifest, run_id=run_id, campaign_id=campaign_id)
        manifest = _apply_params_to_manifest(manifest, chosen, tuning_params)
        if "executor.cores" in chosen:
            _set_by_path(
                manifest,
                ["spec", "sparkConf", "spark.kubernetes.executor.request.cores"],
                str(int(chosen["executor.cores"])),
            )
        logger.info("Run %s: chosen params %s", run_id, chosen)

        manifest_out = run_dir / f"manifest_{run_id}.yaml"
        manifest_masked_out = run_dir / f"manifest_{run_id}.masked.yaml"
        _write_yaml(manifest_out, manifest)
        _write_yaml(manifest_masked_out, _mask_manifest(manifest))
        transform_out = run_dir / transform_path.name
        shutil.copy2(transform_path, transform_out)

        executor_instances = int(chosen.get("executor.instances", 1))
        executor_mem_gb = int(chosen.get("executor.memory_gb", 1))
        requested_gb = driver_memory_gb + executor_instances * executor_mem_gb
        run_params = dict(chosen)
        run_params["driver.memory_gb"] = driver_memory_gb
        run_meta = {
            "run_id": run_id,
            "manifest_path": str(manifest_out),
            "manifest_masked_path": str(manifest_masked_out),
            "transform_path": str(transform_out),
            "transform_sha256": transform_hash,
            "params": run_params,
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
        (run_dir / f"run_{run_id}.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )

        logger.info("Run %s: submitting Spark job", run_id)
        try:
            result = runtime.run_application(
                manifest, namespace, driver_container=args.driver_container
            )
        except KeyboardInterrupt as exc:
            logger.warning(
                "Interrupted during run %s, deleting SparkApplication %s",
                run_id,
                manifest.get("metadata", {}).get("name", ""),
            )
            runtime.delete_application(manifest, namespace)
            raise SystemExit("Interrupted, active SparkApplication deleted.") from exc
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
            run_meta["requested_gb_seconds"] = (
                run_meta["requested_gb"] * run_meta["runtime_seconds"]
            )

        (run_dir / f"run_{run_id}.json").write_text(
            json.dumps(run_meta, indent=2), encoding="utf-8"
        )
        history.append(_build_llm_history_entry(run_meta))

    summary = _summarize_runs()
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary to %s", OUTPUT_DIR / "summary.json")


def _summarize_runs() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for run_path in sorted(RUNS_DIR.glob("run_*/run_*.json")):
        runs.append(json.loads(run_path.read_text(encoding="utf-8")))
    if not runs:
        raise RuntimeError("No runs found in output/runs.")

    scored: list[tuple[float, dict[str, Any]]] = []
    for run in runs:
        if run.get("requested_gb_seconds") is not None:
            scored.append((float(run["requested_gb_seconds"]), run))
    if not scored:
        raise RuntimeError("No runs with requested_gb_seconds recorded.")

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
    parser = argparse.ArgumentParser(
        description="Spark tuning loop with replaceable infrastructure backends."
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml.")
    parser.add_argument("--history-url", default=None, help="History base URL override.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run. Overrides tuning.iterations when set.",
    )
    parser.add_argument(
        "--max-total-memory-gb",
        type=int,
        default=None,
        help="Max total requested memory (driver + executors). Overrides config when set.",
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
    _ensure_dirs()
    logger = _setup_logging()
    try:
        run_loop(args)
    except Exception:
        logger.exception("Unhandled error")
        raise


if __name__ == "__main__":
    main()
