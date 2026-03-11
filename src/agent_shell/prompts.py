from __future__ import annotations

import json
from typing import Any


def build_system_prompt() -> str:
    return (
        "You are a Spark optimization assistant. "
        "You must only propose actions from the allowlist. "
        "Return a JSON object that matches the required schema exactly. "
        "Do not include any extra keys or commentary."
    )


def build_user_prompt(context: dict[str, Any], allow_actions: list[str]) -> str:
    payload = {
        "task": "Analyze Spark app diagnostics and propose optimizations.",
        "allow_actions": allow_actions,
        "context": context,
        "response_schema": {
            "summary": "string",
            "proposals": [
                {
                    "kind": "string (must be in allow_actions)",
                    "payload": {"key": "value"},
                    "rationale": "string",
                }
            ],
            "diagnostics": {"key": "value"},
        },
    }
    return json.dumps(payload, ensure_ascii=True)
