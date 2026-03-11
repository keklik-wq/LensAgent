from src.agent_shell.policy import Policy


def test_policy_filters_actions_and_limits():
    policy = Policy(allow_actions=["a1", "a2"], max_actions=1)
    content = """
    {
      "summary": "ok",
      "proposals": [
        {"kind": "a1", "payload": {"x": 1}, "rationale": "r1"},
        {"kind": "a2", "payload": {"y": 2}, "rationale": "r2"},
        {"kind": "bad", "payload": {}, "rationale": "r3"}
      ],
      "diagnostics": {"k": "v"}
    }
    """
    result = policy.parse_and_validate(content)
    assert result.summary == "ok"
    assert len(result.proposals) == 1
    assert result.proposals[0].kind == "a1"
    assert result.diagnostics["k"] == "v"


def test_policy_handles_invalid_json():
    policy = Policy(allow_actions=["a1"], max_actions=2)
    result = policy.parse_and_validate("not-json")
    assert result.proposals == []
    assert "raw_response" in result.diagnostics
