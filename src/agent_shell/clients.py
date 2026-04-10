from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LlmResponse:
    content: str
    raw: dict[str, object]


class LlmClient(Protocol):
    def chat(self, system: str, user: str, temperature: float = 0.2) -> LlmResponse: ...


class LocalLlmClient:
    def __init__(self, strategy: str = "best_previous") -> None:
        self._strategy = strategy

    def chat(self, system: str, user: str, temperature: float = 0.2) -> LlmResponse:
        del system
        del temperature
        payload = json.loads(user)
        base_params = payload.get("base_params", {})
        history = payload.get("history", [])
        params = self._pick_params(base_params, history)
        response = {
            "params": params,
            "rationale": f"Local LLM strategy '{self._strategy}' selected these parameters.",
        }
        return LlmResponse(content=json.dumps(response), raw={"backend": "local"})

    def _pick_params(
        self,
        base_params: dict[str, object],
        history: list[dict[str, object]],
    ) -> dict[str, object]:
        if self._strategy != "best_previous":
            return dict(base_params)

        completed = [
            item
            for item in history
            if item.get("requested_gb_seconds") is not None and isinstance(item.get("params"), dict)
        ]
        if not completed:
            return dict(base_params)
        best = min(completed, key=lambda item: float(item["requested_gb_seconds"]))
        chosen = dict(best["params"])
        chosen.pop("driver.memory_gb", None)
        return chosen
