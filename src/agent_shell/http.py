from __future__ import annotations

import json
import urllib.request
from typing import Any


class HttpClient:
    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def get_json(self, path: str, headers: dict[str, str] | None = None) -> Any:
        req = urllib.request.Request(
            f"{self._base_url}/{path.lstrip('/')}",
            headers=headers or {},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(
        self,
        path: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/{path.lstrip('/')}",
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
