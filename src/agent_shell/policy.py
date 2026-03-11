from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .types import ActionProposal, AgentResult


@dataclass(frozen=True)
class Policy:
    allow_actions: list[str]
    max_actions: int

    def parse_and_validate(self, content: str) -> AgentResult:
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            return AgentResult(
                summary="LLM response was not valid JSON.",
                proposals=[],
                diagnostics={"raw_response": content},
            )
        proposals = []
        for item in raw.get("proposals", [])[: self.max_actions]:
            kind = str(item.get("kind", ""))
            if kind not in self.allow_actions:
                continue
            payload = item.get("payload", {})
            rationale = str(item.get("rationale", ""))
            proposals.append(
                ActionProposal(kind=kind, payload=payload, rationale=rationale)
            )
        return AgentResult(
            summary=str(raw.get("summary", "")),
            proposals=proposals,
            diagnostics=dict(raw.get("diagnostics", {})),
        )
