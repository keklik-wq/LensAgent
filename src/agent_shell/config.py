from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouterLlmConfig(ConfigModel):
    base_url: str
    chat_path: str = "/v1/chat/completions"
    api_key_env: str
    model: str
    timeout_seconds: int = 30
    allow_models: list[str] = Field(default_factory=list)


class LocalLlmConfig(ConfigModel):
    strategy: str = "best_previous"


class OllamaLlmConfig(ConfigModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str
    timeout_seconds: int = 60
    keep_alive: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class LlmConfig(ConfigModel):
    backend: Literal["router", "local", "ollama"]
    router: RouterLlmConfig | None = None
    local: LocalLlmConfig | None = None
    ollama: OllamaLlmConfig | None = None

    @model_validator(mode="after")
    def validate_backend_config(self) -> LlmConfig:
        if self.backend == "router":
            if self.router is None:
                raise ValueError("llm.backend=router requires llm.router.")
            self.local = None
            self.ollama = None
        elif self.backend == "local":
            if self.local is None:
                self.local = LocalLlmConfig()
            self.router = None
            self.ollama = None
        elif self.backend == "ollama":
            if self.ollama is None:
                raise ValueError("llm.backend=ollama requires llm.ollama.")
            self.router = None
            self.local = None
        return self


class KubernetesRuntimeConfig(ConfigModel):
    kube_context: str | None = None
    kubeconfig_path: str | None = None


class SparkSubmitRuntimeConfig(ConfigModel):
    spark_submit_bin: str = "spark-submit"
    master_url: str = "local[*]"
    deploy_mode: str = "client"
    event_log_dir: str = "/tmp/spark-events"
    poll_seconds: float = 2.0
    timeout_seconds: int = 300


class LocalRuntimeConfig(ConfigModel):
    app_id_prefix: str = "local-app"
    final_state: str = "COMPLETED"
    driver_log_template: str = (
        "Submitted application {app_id} for {app_name} in namespace {namespace}"
    )


class SparkRuntimeConfig(ConfigModel):
    backend: Literal["kubernetes", "spark_submit", "local"]
    kubernetes: KubernetesRuntimeConfig | None = None
    spark_submit: SparkSubmitRuntimeConfig | None = None
    local: LocalRuntimeConfig | None = None

    @model_validator(mode="after")
    def validate_backend_config(self) -> SparkRuntimeConfig:
        if self.backend == "kubernetes":
            if self.kubernetes is None:
                self.kubernetes = KubernetesRuntimeConfig()
            self.spark_submit = None
            self.local = None
        elif self.backend == "spark_submit":
            if self.spark_submit is None:
                self.spark_submit = SparkSubmitRuntimeConfig()
            self.kubernetes = None
            self.local = None
        elif self.backend == "local":
            if self.local is None:
                self.local = LocalRuntimeConfig()
            self.kubernetes = None
            self.spark_submit = None
        return self


class HttpHistoryConfig(ConfigModel):
    base_url: str
    timeout_seconds: int = 30


class LocalHistoryConfig(ConfigModel):
    base_url: str = "http://local-history"
    fixtures_path: str
    default_app_id: str = "local-app-001"


class SparkHistoryConfig(ConfigModel):
    backend: Literal["http", "local"]
    http: HttpHistoryConfig | None = None
    local: LocalHistoryConfig | None = None
    poll_seconds: float = 2.0
    timeout_seconds: int = 120

    @model_validator(mode="after")
    def validate_backend_config(self) -> SparkHistoryConfig:
        if self.backend == "http":
            if self.http is None:
                raise ValueError("spark_history.backend=http requires spark_history.http.")
            self.local = None
        elif self.backend == "local":
            if self.local is None:
                raise ValueError("spark_history.backend=local requires spark_history.local.")
            self.http = None
        return self


class TuningParamConfig(ConfigModel):
    path: list[str]
    type: Literal["int", "float", "bool", "memory_gb", "str", "enum"] = "int"
    min: float | int | str | None = None
    max: float | int | str | None = None
    values: list[str] | None = None
    default: object | None = None

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path(cls, value: Any) -> list[str]:
        if isinstance(value, list):
            parts = [str(part) for part in value]
        else:
            parts = [part for part in str(value).split(".") if part]
        if not parts:
            raise ValueError("path must not be empty.")
        return parts

    @field_validator("values", mode="before")
    @classmethod
    def normalize_values(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
        else:
            raise ValueError("values must be a list or comma-separated string.")
        return values

    @model_validator(mode="after")
    def validate_enum_fields(self) -> TuningParamConfig:
        if self.type == "enum":
            if not self.values:
                raise ValueError("enum tuning param requires non-empty values.")
            if self.min is not None or self.max is not None:
                raise ValueError("enum tuning param does not support min/max.")
        elif self.values is not None:
            raise ValueError("values is only supported for enum tuning params.")
        return self


def _default_tuning_prompt() -> str:
    return (
        "You are a Spark tuning assistant. "
        "You must propose the next run parameters to minimize requested_gb_seconds, "
        "reduce spill_gb, and avoid too many small files. "
        "Use only the tunable parameters provided in the configuration payload. "
        "Do not assume any fixed Spark parameter names beyond what is provided for this run. "
        "Treat requested_gb_seconds as the primary optimization target. "
        "Continue tuning and exploring alternative configurations instead of freezing the configuration too early. "
        "Never return a parameter configuration that has already been run before. "
        "Across the tuning campaign, try to gather information about all tunable parameters so that each parameter changes in at least one run when it is feasible to do so. "
        "It is acceptable to make bold configuration changes when they help explore the search space and gather information about job behavior under different settings. "
        "Return ONLY valid JSON that matches the schema exactly."
    )


def _default_tuning_params() -> dict[str, dict[str, Any]]:
    return {
        "spark.sql.shuffle.partitions": {
            "path": "spec.sparkConf.spark.sql.shuffle.partitions",
            "type": "int",
            "min": 200,
            "max": 10000,
        },
        "executor.cores": {
            "path": "spec.executor.cores",
            "type": "int",
            "min": 1,
            "max": 16,
        },
        "executor.instances": {
            "path": "spec.executor.instances",
            "type": "int",
            "min": 1,
            "max": 500,
        },
        "executor.memory_gb": {
            "path": "spec.executor.memory",
            "type": "memory_gb",
            "min": 1,
            "max": 256,
        },
    }


class TuningConfig(ConfigModel):
    iterations: int = 2
    prompt: str = Field(default_factory=_default_tuning_prompt)
    llm_json_retries: int = 2
    params: dict[str, TuningParamConfig] = Field(default_factory=_default_tuning_params)
    constraints: dict[str, Any] = Field(
        default_factory=lambda: {"total_memory_gb": {"max": 500}}
    )

    @property
    def total_memory_gb_max(self) -> int | None:
        total_memory = (self.constraints.get("total_memory_gb") or {}).get("max")
        return int(total_memory) if total_memory is not None else None


class RunConfig(ConfigModel):
    manifest: str
    transform: str
    first_run_mode: Literal["llm", "base", "random"] = "llm"


class AppConfig(ConfigModel):
    config_path: Path
    config_dir: Path
    run: RunConfig
    llm: LlmConfig
    spark_runtime: SparkRuntimeConfig
    spark_history: SparkHistoryConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @classmethod
    def load(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit(f"Config at {path} is empty or invalid YAML.")
        try:
            return cls.model_validate(
                {
                    **raw,
                    "config_path": config_path,
                    "config_dir": config_path.parent,
                }
            )
        except ValidationError as exc:
            raise SystemExit(str(exc)) from exc
