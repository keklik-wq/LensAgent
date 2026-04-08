from __future__ import annotations

import os

from .clients import LlmResponse
from .http import HttpClient


class LlmRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        model: str,
        timeout_seconds: int,
        allow_models: list[str],
    ) -> None:
        if model not in allow_models:
            raise ValueError(f"Model '{model}' is not in allowlist")
        self._client = HttpClient(base_url, timeout_seconds)
        self._api_key = os.getenv(api_key_env, "")
        self._model = model

    def chat(self, system: str, user: str, temperature: float = 0.2) -> LlmResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        raw = self._client.post_json("/v1/chat/completions", payload, headers=headers)
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LlmResponse(content=content, raw=raw)
