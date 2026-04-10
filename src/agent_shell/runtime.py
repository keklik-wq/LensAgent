from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SparkRunResult:
    app_id: str
    final_state: str
    driver_logs: str


class SparkRuntime(Protocol):
    def run_application(
        self,
        manifest: dict[str, Any],
        namespace: str,
        driver_container: str | None = None,
    ) -> SparkRunResult: ...


class KubernetesSparkRuntime:
    def __init__(self, kube_context: str | None, kubeconfig_path: str | None = None) -> None:
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config

        self._k8s_client = k8s_client
        try:
            k8s_config.load_kube_config(config_file=kubeconfig_path, context=kube_context)
        except Exception:
            k8s_config.load_incluster_config()
        self._core_api = k8s_client.CoreV1Api()
        self._custom_api = k8s_client.CustomObjectsApi()

    def run_application(
        self,
        manifest: dict[str, Any],
        namespace: str,
        driver_container: str | None = None,
    ) -> SparkRunResult:
        api_version = manifest.get("apiVersion", "")
        kind = manifest.get("kind", "")
        if kind != "SparkApplication" or "/" not in api_version:
            raise SystemExit("Only SparkApplication manifests are supported.")
        group, version = api_version.split("/", 1)
        name = manifest.get("metadata", {}).get("name")
        if not name:
            raise SystemExit("Manifest metadata.name is required.")
        self._apply(group, version, namespace, name, manifest)
        status_obj = self._wait_for_completion(group, version, namespace, name)
        app_id = self._app_id_from_status(status_obj)
        logs = ""
        if not app_id:
            driver_pod = self._find_driver_pod(namespace, name)
            logs = self._core_api.read_namespaced_pod_log(
                driver_pod,
                namespace,
                container=driver_container,
                tail_lines=2000,
            )
            app_id = self._extract_app_id(logs)
        return SparkRunResult(
            app_id=app_id,
            final_state=status_obj.get("status", {}).get("applicationState", {}).get("state", ""),
            driver_logs=logs,
        )

    def _apply(
        self,
        group: str,
        version: str,
        namespace: str,
        name: str,
        manifest: dict[str, Any],
    ) -> None:
        plural = "sparkapplications"
        try:
            existing = self._custom_api.get_namespaced_custom_object(
                group, version, namespace, plural, name
            )
            metadata = existing.get("metadata", {})
            resource_version = metadata.get("resourceVersion")
            if resource_version:
                manifest.setdefault("metadata", {})
                manifest["metadata"]["resourceVersion"] = resource_version
            self._custom_api.replace_namespaced_custom_object(
                group, version, namespace, plural, name, manifest
            )
        except self._k8s_client.exceptions.ApiException as exc:
            if exc.status == 404:
                self._custom_api.create_namespaced_custom_object(
                    group, version, namespace, plural, manifest
                )
            else:
                raise

    def _wait_for_completion(
        self,
        group: str,
        version: str,
        namespace: str,
        name: str,
        poll_seconds: int = 20,
        timeout_seconds: int = 6 * 60 * 60,
    ) -> dict[str, Any]:
        start = time.time()
        while True:
            obj = self._custom_api.get_namespaced_custom_object(
                group,
                version,
                namespace,
                "sparkapplications",
                name,
            )
            state = obj.get("status", {}).get("applicationState", {}).get("state", "")
            if state in {"COMPLETED", "FAILED"}:
                return obj
            if time.time() - start > timeout_seconds:
                raise SystemExit(f"Timed out waiting for {name} to finish.")
            time.sleep(poll_seconds)

    def _find_driver_pod(self, namespace: str, app_name: str) -> str:
        labels = f"sparkoperator.k8s.io/app-name={app_name},spark-role=driver"
        pods = self._core_api.list_namespaced_pod(namespace, label_selector=labels)
        if pods.items:
            return pods.items[0].metadata.name
        pods = self._core_api.list_namespaced_pod(namespace)
        for item in pods.items:
            name = item.metadata.name or ""
            if name.startswith(f"{app_name}-driver"):
                return name
        raise SystemExit(f"Driver pod not found for app {app_name}")

    def _app_id_from_status(self, obj: dict[str, Any]) -> str:
        status = obj.get("status", {})
        
        for key in ("sparkApplicationId", "appId", "applicationId"):
            value = status.get(key)
            if value:
                return str(value)
        return ""

    def _extract_app_id(self, driver_logs: str) -> str:
        return _extract_app_id(driver_logs)


class SparkSubmitRuntime:
    def __init__(
        self,
        spark_submit_bin: str,
        master_url: str,
        deploy_mode: str,
        event_log_dir: str,
        poll_seconds: float,
        timeout_seconds: int,
    ) -> None:
        self._spark_submit_bin = spark_submit_bin
        self._master_url = master_url
        self._deploy_mode = deploy_mode
        self._event_log_dir = Path(event_log_dir)
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds

    def run_application(
        self,
        manifest: dict[str, Any],
        namespace: str,
        driver_container: str | None = None,
    ) -> SparkRunResult:
        del driver_container
        app_name = str(manifest.get("metadata", {}).get("name", "spark-app"))
        submit_cmd = self._build_submit_command(manifest, namespace)
        before = set(self._list_event_logs())
        result = subprocess.run(
            submit_cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        logs = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        if result.returncode != 0:
            raise SystemExit(f"spark-submit failed for {app_name}:\n{logs}")
        app_id = _try_extract_app_id(logs) or self._wait_for_event_log(before)
        return SparkRunResult(
            app_id=app_id,
            final_state="COMPLETED",
            driver_logs=logs,
        )

    def _build_submit_command(self, manifest: dict[str, Any], namespace: str) -> list[str]:
        del namespace
        spec = manifest.get("spec", {})
        command = [
            self._spark_submit_bin,
            "--master",
            self._master_url,
            "--deploy-mode",
            self._deploy_mode,
            "--name",
            str(manifest.get("metadata", {}).get("name", "spark-app")),
        ]

        for key, value in sorted((spec.get("sparkConf") or {}).items()):
            command.extend(["--conf", f"{key}={value}"])

        driver = spec.get("driver") or {}
        executor = spec.get("executor") or {}
        if driver.get("cores") is not None:
            command.extend(["--conf", f"spark.driver.cores={driver['cores']}"])
        if driver.get("memory"):
            command.extend(["--driver-memory", str(driver["memory"])])
        if executor.get("cores") is not None:
            command.extend(["--executor-cores", str(executor["cores"])])
        if executor.get("instances") is not None:
            command.extend(["--num-executors", str(executor["instances"])])
        if executor.get("memory"):
            command.extend(["--executor-memory", str(executor["memory"])])

        app_file = str(spec.get("mainApplicationFile", ""))
        if not app_file:
            raise SystemExit(
                "Manifest spec.mainApplicationFile is required for spark_submit runtime."
            )
        command.append(_normalize_app_path(app_file))
        for arg in spec.get("arguments", []) or []:
            command.append(str(arg))
        return command

    def _wait_for_event_log(self, before: set[str]) -> str:
        deadline = time.time() + self._timeout_seconds
        while time.time() < deadline:
            after = self._list_event_logs()
            for candidate in after:
                if candidate.name in before or candidate.name.endswith(".inprogress"):
                    continue
                match = _try_extract_app_id(candidate.name)
                if match:
                    return match
            time.sleep(self._poll_seconds)
        raise SystemExit(f"Timed out waiting for event log in {self._event_log_dir}")

    def _list_event_logs(self) -> list[Path]:
        if not self._event_log_dir.exists():
            return []
        return sorted(path for path in self._event_log_dir.iterdir() if path.is_file())


class LocalSparkRuntime:
    def __init__(
        self,
        app_id_prefix: str,
        final_state: str,
        driver_log_template: str,
    ) -> None:
        self._app_id_prefix = app_id_prefix
        self._final_state = final_state
        self._driver_log_template = driver_log_template

    def run_application(
        self,
        manifest: dict[str, Any],
        namespace: str,
        driver_container: str | None = None,
    ) -> SparkRunResult:
        del driver_container
        app_name = str(manifest.get("metadata", {}).get("name", "local-app"))
        suffix = app_name.rsplit("-r", 1)[-1] if "-r" in app_name else "001"
        app_id = f"{self._app_id_prefix}-{suffix}"
        logs = self._driver_log_template.format(
            app_id=app_id,
            app_name=app_name,
            namespace=namespace,
        )
        return SparkRunResult(
            app_id=app_id,
            final_state=self._final_state,
            driver_logs=logs,
        )


def _normalize_app_path(app_file: str) -> str:
    prefix = "local://"
    if app_file.startswith(prefix):
        return app_file[len(prefix) - 1 :]
    return app_file


def _extract_app_id(driver_logs: str) -> str:
    app_id = _try_extract_app_id(driver_logs)
    if app_id is None:
        raise SystemExit("Could not find appId in driver logs.")
    return app_id


def _try_extract_app_id(text: str) -> str | None:
    patterns = [
        r"app[-_]?id\s*[:=]\s*(app-\d{14}-\d{4})",
        r"(app-\d{14}-\d{4})",
        r"(local-\d+)",
        r"(spark-[a-f0-9]{32})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None
