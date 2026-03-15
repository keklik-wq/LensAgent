from __future__ import annotations

from .clients import LlmClient, LocalLlmClient
from .config import AppConfig
from .history import HttpSparkHistoryProvider, LocalSparkHistoryProvider, SparkHistoryProvider
from .llm_router import LlmRouterClient
from .runtime import KubernetesSparkRuntime, LocalSparkRuntime, SparkRuntime, SparkSubmitRuntime


def build_llm_client(config: AppConfig) -> LlmClient:
    if config.llm.backend == "router":
        if config.llm.router is None:
            raise SystemExit("llm.router config is required for router backend.")
        return LlmRouterClient(
            base_url=config.llm.router.base_url,
            api_key_env=config.llm.router.api_key_env,
            model=config.llm.router.model,
            timeout_seconds=config.llm.router.timeout_seconds,
            allow_models=config.llm.router.allow_models,
        )
    if config.llm.backend == "local":
        if config.llm.local is None:
            raise SystemExit("llm.local config is required for local backend.")
        return LocalLlmClient(strategy=config.llm.local.strategy)
    raise SystemExit(f"Unsupported llm backend: {config.llm.backend}")


def build_spark_runtime(config: AppConfig, kube_context: str | None = None) -> SparkRuntime:
    if config.spark_runtime.backend == "kubernetes":
        runtime_cfg = config.spark_runtime.kubernetes
        if runtime_cfg is None:
            raise SystemExit("spark_runtime.kubernetes config is required.")
        return KubernetesSparkRuntime(kube_context=kube_context or runtime_cfg.kube_context)
    if config.spark_runtime.backend == "spark_submit":
        runtime_cfg = config.spark_runtime.spark_submit
        if runtime_cfg is None:
            raise SystemExit("spark_runtime.spark_submit config is required.")
        return SparkSubmitRuntime(
            spark_submit_bin=runtime_cfg.spark_submit_bin,
            master_url=runtime_cfg.master_url,
            deploy_mode=runtime_cfg.deploy_mode,
            event_log_dir=runtime_cfg.event_log_dir,
            poll_seconds=runtime_cfg.poll_seconds,
            timeout_seconds=runtime_cfg.timeout_seconds,
        )
    if config.spark_runtime.backend == "local":
        runtime_cfg = config.spark_runtime.local
        if runtime_cfg is None:
            raise SystemExit("spark_runtime.local config is required.")
        return LocalSparkRuntime(
            app_id_prefix=runtime_cfg.app_id_prefix,
            final_state=runtime_cfg.final_state,
            driver_log_template=runtime_cfg.driver_log_template,
        )
    raise SystemExit(f"Unsupported spark runtime backend: {config.spark_runtime.backend}")


def build_spark_history_provider(
    config: AppConfig,
    base_url_override: str | None = None,
) -> SparkHistoryProvider:
    if config.spark_history.backend == "http":
        history_cfg = config.spark_history.http
        if history_cfg is None:
            raise SystemExit("spark_history.http config is required.")
        return HttpSparkHistoryProvider(
            base_url=base_url_override or history_cfg.base_url,
            timeout_seconds=history_cfg.timeout_seconds,
        )
    if config.spark_history.backend == "local":
        history_cfg = config.spark_history.local
        if history_cfg is None:
            raise SystemExit("spark_history.local config is required.")
        return LocalSparkHistoryProvider(
            fixtures_path=history_cfg.fixtures_path,
            base_url=base_url_override or history_cfg.base_url,
            default_app_id=history_cfg.default_app_id,
        )
    raise SystemExit(f"Unsupported spark history backend: {config.spark_history.backend}")
