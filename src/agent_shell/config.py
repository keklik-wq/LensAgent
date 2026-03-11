from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LlmRouterConfig:
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: int
    allow_models: list[str]


@dataclass(frozen=True)
class SparkHistoryConfig:
    base_url: str
    timeout_seconds: int


@dataclass(frozen=True)
class K8sConfig:
    kube_context: str
    namespace: str
    allow_pods_prefix: str
    max_log_bytes: int


@dataclass(frozen=True)
class OutputStorageConfig:
    type: str
    base_path: str
    max_list_files: int


@dataclass(frozen=True)
class PolicyConfig:
    max_actions: int
    allow_actions: list[str]


@dataclass(frozen=True)
class RuntimeConfig:
    dry_run: bool


@dataclass(frozen=True)
class ShellConfig:
    llm_router: LlmRouterConfig
    spark_history: SparkHistoryConfig
    k8s: K8sConfig
    output_storage: OutputStorageConfig
    policy: PolicyConfig
    runtime: RuntimeConfig

    @staticmethod
    def load(path: str | Path) -> "ShellConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return ShellConfig(
            llm_router=_coerce_llm_router(raw["llm_router"]),
            spark_history=_coerce_spark_history(raw["spark_history"]),
            k8s=_coerce_k8s(raw["k8s"]),
            output_storage=_coerce_output_storage(raw["output_storage"]),
            policy=_coerce_policy(raw["policy"]),
            runtime=_coerce_runtime(raw["runtime"]),
        )


def _coerce_llm_router(raw: dict[str, Any]) -> LlmRouterConfig:
    return LlmRouterConfig(
        base_url=raw["base_url"],
        api_key_env=raw["api_key_env"],
        model=raw["model"],
        timeout_seconds=int(raw["timeout_seconds"]),
        allow_models=list(raw["allow_models"]),
    )


def _coerce_spark_history(raw: dict[str, Any]) -> SparkHistoryConfig:
    return SparkHistoryConfig(
        base_url=raw["base_url"],
        timeout_seconds=int(raw["timeout_seconds"]),
    )


def _coerce_k8s(raw: dict[str, Any]) -> K8sConfig:
    return K8sConfig(
        kube_context=raw["kube_context"],
        namespace=raw["namespace"],
        allow_pods_prefix=raw["allow_pods_prefix"],
        max_log_bytes=int(raw["max_log_bytes"]),
    )


def _coerce_output_storage(raw: dict[str, Any]) -> OutputStorageConfig:
    return OutputStorageConfig(
        type=raw["type"],
        base_path=raw["base_path"],
        max_list_files=int(raw["max_list_files"]),
    )


def _coerce_policy(raw: dict[str, Any]) -> PolicyConfig:
    return PolicyConfig(
        max_actions=int(raw["max_actions"]),
        allow_actions=list(raw["allow_actions"]),
    )


def _coerce_runtime(raw: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        dry_run=bool(raw["dry_run"]),
    )
