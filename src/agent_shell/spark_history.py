from __future__ import annotations

from typing import Any

from .http import HttpClient


class SparkHistoryClient:
    def __init__(self, base_url: str, timeout_seconds: int) -> None:
        self._client = HttpClient(base_url, timeout_seconds)

    def get_application(self, app_id: str) -> dict[str, Any]:
        return self._client.get_json(f"/api/v1/applications/{app_id}")

    def get_jobs(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(f"/api/v1/applications/{app_id}/jobs")

    def get_stages(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(f"/api/v1/applications/{app_id}/stages")

    def get_environment(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(f"/api/v1/applications/{app_id}/environment")

    def get_sql(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(f"/api/v1/applications/{app_id}/sql")

    def get_executors(self, app_id: str) -> list[dict[str, Any]]:
        return self._client.get_json(f"/api/v1/applications/{app_id}/executors")
