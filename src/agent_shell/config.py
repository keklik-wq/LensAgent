from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RouterLlmConfig:
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: int
    allow_models: list[str]


@dataclass(frozen=True)
class LocalLlmConfig:
    strategy: str


@dataclass(frozen=True)
class LlmConfig:
    backend: str
    router: RouterLlmConfig | None
    local: LocalLlmConfig | None


@dataclass(frozen=True)
class KubernetesRuntimeConfig:
    kube_context: str | None


@dataclass(frozen=True)
class SparkSubmitRuntimeConfig:
    spark_submit_bin: str
    master_url: str
    deploy_mode: str
    event_log_dir: str
    poll_seconds: float
    timeout_seconds: int


@dataclass(frozen=True)
class LocalRuntimeConfig:
    app_id_prefix: str
    final_state: str
    driver_log_template: str


@dataclass(frozen=True)
class SparkRuntimeConfig:
    backend: str
    kubernetes: KubernetesRuntimeConfig | None
    spark_submit: SparkSubmitRuntimeConfig | None
    local: LocalRuntimeConfig | None


@dataclass(frozen=True)
class HttpHistoryConfig:
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class LocalHistoryConfig:
    base_url: str
    fixtures_path: str
    default_app_id: str


@dataclass(frozen=True)
class SparkHistoryConfig:
    backend: str
    http: HttpHistoryConfig | None
    local: LocalHistoryConfig | None
    poll_seconds: float
    timeout_seconds: int


@dataclass(frozen=True)
class AppConfig:
    llm: LlmConfig
    spark_runtime: SparkRuntimeConfig
    spark_history: SparkHistoryConfig

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise SystemExit(f"Config at {path} is empty or invalid YAML.")
        normalized = _normalize_legacy_config(raw)
        return AppConfig(
            llm=_coerce_llm(normalized.get("llm")),
            spark_runtime=_coerce_spark_runtime(normalized.get("spark_runtime")),
            spark_history=_coerce_spark_history(normalized.get("spark_history")),
        )


def _normalize_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    if "llm" not in data:
        router = data.get("llm_router")
        if not isinstance(router, dict):
            raise SystemExit("Config is missing required section: llm or llm_router")
        data["llm"] = {
            "backend": "router",
            "router": router,
        }
    if "spark_runtime" not in data:
        data["spark_runtime"] = {
            "backend": "kubernetes",
            "kubernetes": {
                "kube_context": None,
            },
        }
    if "spark_history" not in data:
        raise SystemExit("Config is missing required section: spark_history")
    return data


def _coerce_llm(raw: Any) -> LlmConfig:
    if not isinstance(raw, dict):
        raise SystemExit("Config section llm must be a mapping.")
    backend = str(raw.get("backend", "router"))
    router = raw.get("router")
    local = raw.get("local")
    if backend == "router":
        if not isinstance(router, dict):
            raise SystemExit("llm.backend=router requires llm.router.")
        return LlmConfig(
            backend=backend,
            router=_coerce_router_llm(router),
            local=None,
        )
    if backend == "local":
        return LlmConfig(
            backend=backend,
            router=None,
            local=_coerce_local_llm(local or {}),
        )
    raise SystemExit(f"Unsupported llm backend: {backend}")


def _coerce_router_llm(raw: dict[str, Any]) -> RouterLlmConfig:
    return RouterLlmConfig(
        base_url=str(raw["base_url"]),
        api_key_env=str(raw["api_key_env"]),
        model=str(raw["model"]),
        timeout_seconds=int(raw.get("timeout_seconds", 30)),
        allow_models=list(raw.get("allow_models", [])),
    )


def _coerce_local_llm(raw: dict[str, Any]) -> LocalLlmConfig:
    return LocalLlmConfig(
        strategy=str(raw.get("strategy", "best_previous")),
    )


def _coerce_spark_runtime(raw: Any) -> SparkRuntimeConfig:
    if not isinstance(raw, dict):
        raise SystemExit("Config section spark_runtime must be a mapping.")
    backend = str(raw.get("backend", "kubernetes"))
    if backend == "kubernetes":
        return SparkRuntimeConfig(
            backend=backend,
            kubernetes=KubernetesRuntimeConfig(
                kube_context=_coerce_optional_str((raw.get("kubernetes") or {}).get("kube_context")),
            ),
            spark_submit=None,
            local=None,
        )
    if backend == "spark_submit":
        return SparkRuntimeConfig(
            backend=backend,
            kubernetes=None,
            spark_submit=_coerce_spark_submit_runtime(raw.get("spark_submit") or {}),
            local=None,
        )
    if backend == "local":
        return SparkRuntimeConfig(
            backend=backend,
            kubernetes=None,
            spark_submit=None,
            local=_coerce_local_runtime(raw.get("local") or {}),
        )
    raise SystemExit(f"Unsupported spark_runtime backend: {backend}")


def _coerce_spark_submit_runtime(raw: dict[str, Any]) -> SparkSubmitRuntimeConfig:
    return SparkSubmitRuntimeConfig(
        spark_submit_bin=str(raw.get("spark_submit_bin", "spark-submit")),
        master_url=str(raw.get("master_url", "local[*]")),
        deploy_mode=str(raw.get("deploy_mode", "client")),
        event_log_dir=str(raw.get("event_log_dir", "/tmp/spark-events")),
        poll_seconds=float(raw.get("poll_seconds", 2.0)),
        timeout_seconds=int(raw.get("timeout_seconds", 300)),
    )


def _coerce_local_runtime(raw: dict[str, Any]) -> LocalRuntimeConfig:
    return LocalRuntimeConfig(
        app_id_prefix=str(raw.get("app_id_prefix", "local-app")),
        final_state=str(raw.get("final_state", "COMPLETED")),
        driver_log_template=str(
            raw.get(
                "driver_log_template",
                "Submitted application {app_id} for {app_name} in namespace {namespace}",
            )
        ),
    )


def _coerce_spark_history(raw: Any) -> SparkHistoryConfig:
    if not isinstance(raw, dict):
        raise SystemExit("Config section spark_history must be a mapping.")
    backend = str(raw.get("backend", "http"))
    if backend == "http":
        http = raw.get("http") or {}
        base_url = http.get("base_url")
        if not base_url:
            raise SystemExit("spark_history.backend=http requires spark_history.http.base_url.")
        return SparkHistoryConfig(
            backend=backend,
            http=HttpHistoryConfig(
                base_url=str(base_url),
                timeout_seconds=int(http.get("timeout_seconds", 30)),
            ),
            local=None,
            poll_seconds=float(raw.get("poll_seconds", 2.0)),
            timeout_seconds=int(raw.get("timeout_seconds", 120)),
        )
    if backend == "local":
        return SparkHistoryConfig(
            backend=backend,
            http=None,
            local=_coerce_local_history(raw.get("local") or {}),
            poll_seconds=float(raw.get("poll_seconds", 0.1)),
            timeout_seconds=int(raw.get("timeout_seconds", 5)),
        )
    raise SystemExit(f"Unsupported spark_history backend: {backend}")


def _coerce_local_history(raw: dict[str, Any]) -> LocalHistoryConfig:
    fixtures_path = raw.get("fixtures_path")
    if not fixtures_path:
        raise SystemExit("spark_history.backend=local requires spark_history.local.fixtures_path.")
    return LocalHistoryConfig(
        base_url=str(raw.get("base_url", "http://local-history")),
        fixtures_path=str(fixtures_path),
        default_app_id=str(raw.get("default_app_id", "local-app-001")),
    )


def _coerce_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
