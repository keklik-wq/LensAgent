from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from .http import HttpClient


class SparkHistoryProvider(Protocol):
    def get_stages(self, app_id: str) -> list[dict[str, Any]]:
        ...

    def latest_app_id(self) -> str | None:
        ...

    def stages_url(self, app_id: str) -> str:
        ...

    def ui_url(self, app_id: str) -> str:
        ...


class HttpSparkHistoryProvider:
    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self._client = HttpClient(base_url, timeout_seconds)
        self._base_url = base_url.rstrip("/")

    def get_stages(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(self._stages_path(app_id))

    def latest_app_id(self) -> str | None:
        applications = self._client.get_json("/api/v1/applications")
        if not applications:
            return None
        latest = applications[-1]
        app_id = latest.get("id")
        if not app_id:
            return None
        return str(app_id)

    def stages_url(self, app_id: str) -> str:
        return f"{self._base_url}{self._stages_path(app_id)}"

    def ui_url(self, app_id: str) -> str:
        return f"{self._base_url}/history/{app_id}/stages/"

    def _stages_path(self, app_id: str) -> str:
        return f"/api/v1/applications/{app_id}/stages"


class LocalSparkHistoryProvider:
    def __init__(self, fixtures_path: str, base_url: str, default_app_id: str) -> None:
        self._fixtures_dir = Path(fixtures_path)
        self._base_url = base_url.rstrip("/")
        self._default_app_id = default_app_id

    def get_stages(self, app_id: str) -> list[dict[str, Any]]:
        fixture_path = self._resolve_fixture(app_id)
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    def latest_app_id(self) -> str | None:
        return self._default_app_id

    def stages_url(self, app_id: str) -> str:
        return f"{self._base_url}/api/v1/applications/{app_id}/stages"

    def ui_url(self, app_id: str) -> str:
        return f"{self._base_url}/history/{app_id}/stages/"

    def _resolve_fixture(self, app_id: str) -> Path:
        candidates = [
            self._fixtures_dir / f"{app_id}-stages.json",
            self._fixtures_dir / f"{self._default_app_id}-stages.json",
            self._fixtures_dir / "stages.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SystemExit(f"No local Spark history fixture found in {self._fixtures_dir}")
