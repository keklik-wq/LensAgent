from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LlmRouterConfig:
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: int
    allow_models: list[str]


@dataclass(frozen=True)
class AppConfig:
    llm_router: LlmRouterConfig

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise SystemExit(f"Config at {path} is empty or invalid YAML.")
        if "llm_router" not in raw:
            raise SystemExit("Config is missing required section: llm_router")
        return AppConfig(llm_router=_coerce_llm_router(raw["llm_router"]))


def _coerce_llm_router(raw: dict[str, Any]) -> LlmRouterConfig:
    return LlmRouterConfig(
        base_url=raw["base_url"],
        api_key_env=raw["api_key_env"],
        model=raw["model"],
        timeout_seconds=int(raw.get("timeout_seconds", 30)),
        allow_models=list(raw.get("allow_models", [])),
    )
