from __future__ import annotations

import logging
import socket
from typing import Any
from urllib.error import URLError

from .clients import LlmResponse
from .http import HttpClient

logger = logging.getLogger("lens-agent.ollama")


class OllamaLlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = HttpClient(base_url, timeout_seconds)
        self._model = model
        self._keep_alive = keep_alive
        self._options = dict(options or {})

    def chat(self, system: str, user: str, temperature: float = 0.2) -> LlmResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, **self._options},
        }
        if self._keep_alive is not None:
            payload["keep_alive"] = self._keep_alive

        logger.info("Sending request to Ollama model '%s' at %s", self._model, self._base_url)
        logger.info("Ollama request payload: %s", payload)
        logger.info(
            "Waiting for response from Ollama model '%s' (timeout=%ss)",
            self._model,
            self._timeout_seconds,
        )
        try:
            raw = self._client.post_json("/api/chat", payload)
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama request timed out for model '{self._model}'. "
                "Increase llm.ollama.timeout_seconds or use a smaller/faster model."
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise RuntimeError(
                    f"Ollama request timed out for model '{self._model}'. "
                    "Increase llm.ollama.timeout_seconds or use a smaller/faster model."
                ) from exc
            raise
        logger.info("Received response from Ollama model '%s'", self._model)
        message = raw.get("message", {})
        if not isinstance(message, dict):
            raise ValueError("Ollama response is missing message object.")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Ollama response content must be a string.")
        return LlmResponse(content=content, raw=raw)
