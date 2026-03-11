from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppContext:
    app_id: str
    namespace: str
    output_path: str | None = None


@dataclass(frozen=True)
class ActionProposal:
    kind: str
    payload: dict[str, Any]
    rationale: str


@dataclass(frozen=True)
class AgentResult:
    summary: str
    proposals: list[ActionProposal]
    diagnostics: dict[str, Any]
