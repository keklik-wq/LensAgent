from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class K8sLogConfig:
    kube_context: str
    namespace: str
    allow_pods_prefix: str
    max_log_bytes: int


class K8sLogReader:
    def __init__(self, config: K8sLogConfig) -> None:
        self._cfg = config

    def list_pods(self) -> list[str]:
        data = self._run_kubectl(
            [
                "get",
                "pods",
                "-n",
                self._cfg.namespace,
                "-o",
                "json",
            ]
        )
        obj = json.loads(data)
        names: list[str] = []
        for item in obj.get("items", []):
            name = item.get("metadata", {}).get("name", "")
            if name.startswith(self._cfg.allow_pods_prefix):
                names.append(name)
        return names

    def read_logs(self, pod: str, container: str | None = None, tail: int = 5000) -> str:
        if not pod.startswith(self._cfg.allow_pods_prefix):
            raise ValueError("Pod is not in allowlist")
        args = [
            "logs",
            pod,
            "-n",
            self._cfg.namespace,
            "--tail",
            str(tail),
        ]
        if container:
            args.extend(["-c", container])
        data = self._run_kubectl(args)
        return data[: self._cfg.max_log_bytes]

    def _run_kubectl(self, args: list[str]) -> str:
        cmd = ["kubectl", "--context", self._cfg.kube_context, *args]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
