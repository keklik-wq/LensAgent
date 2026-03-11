from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import ShellConfig
from .k8s_logs import K8sLogConfig, K8sLogReader
from .llm_router import LlmRouterClient
from .policy import Policy
from .prompts import build_system_prompt, build_user_prompt
from .spark_history import SparkHistoryClient
from .storage import OutputStorageInspector
from .types import AgentResult, AppContext


class AgentShell:
    def __init__(self, config: ShellConfig) -> None:
        self._config = config
        self._spark = SparkHistoryClient(
            config.spark_history.base_url,
            config.spark_history.timeout_seconds,
        )
        self._k8s = K8sLogReader(
            K8sLogConfig(
                kube_context=config.k8s.kube_context,
                namespace=config.k8s.namespace,
                allow_pods_prefix=config.k8s.allow_pods_prefix,
                max_log_bytes=config.k8s.max_log_bytes,
            )
        )
        self._storage = OutputStorageInspector(
            storage_type=config.output_storage.type,
            base_path=config.output_storage.base_path,
            max_list_files=config.output_storage.max_list_files,
        )
        self._llm = LlmRouterClient(
            base_url=config.llm_router.base_url,
            api_key_env=config.llm_router.api_key_env,
            model=config.llm_router.model,
            timeout_seconds=config.llm_router.timeout_seconds,
            allow_models=config.llm_router.allow_models,
        )
        self._policy = Policy(
            allow_actions=config.policy.allow_actions,
            max_actions=config.policy.max_actions,
        )

    def run(self, ctx: AppContext) -> AgentResult:
        context = self._collect_context(ctx)
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(context, self._policy.allow_actions)
        response = self._llm.chat(system_prompt, user_prompt)
        return self._policy.parse_and_validate(response.content)

    def _collect_context(self, ctx: AppContext) -> dict[str, Any]:
        app = self._spark.get_application(ctx.app_id)
        jobs = self._spark.get_jobs(ctx.app_id)
        stages = self._spark.get_stages(ctx.app_id)
        env = self._spark.get_environment(ctx.app_id)
        sql = self._spark.get_sql(ctx.app_id)
        executors = self._spark.get_executors(ctx.app_id)
        pods = self._k8s.list_pods()
        logs = {}
        for pod in pods:
            logs[pod] = self._k8s.read_logs(pod)
        output_stats = None
        if ctx.output_path:
            output_stats = asdict(self._storage.inspect(ctx.output_path))
        return {
            "app_id": ctx.app_id,
            "spark_history": {
                "application": app,
                "jobs": jobs,
                "stages": stages,
                "environment": env,
                "sql": sql,
                "executors": executors,
            },
            "k8s": {
                "namespace": ctx.namespace,
                "pods": pods,
                "logs": logs,
            },
            "output": output_stats,
        }
