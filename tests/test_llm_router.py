from src.agent_shell.llm_router import LlmRouterClient


def test_router_client_uses_configured_chat_path(monkeypatch) -> None:
    monkeypatch.setenv("TEST_ROUTER_KEY", "secret")
    client = LlmRouterClient(
        base_url="http://router",
        chat_path="/api/chat",
        api_key_env="TEST_ROUTER_KEY",
        model="m1",
        timeout_seconds=30,
        allow_models=["m1"],
    )

    captured: dict[str, object] = {}

    class FakeHttpClient:
        def post_json(
            self,
            path: str,
            payload: dict[str, object],
            headers: dict[str, str] | None = None,
        ) -> dict[str, object]:
            captured["path"] = path
            captured["payload"] = payload
            captured["headers"] = headers or {}
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"params":{"executor.instances":2},"rationale":"ok"}'
                        }
                    }
                ]
            }

    client._client = FakeHttpClient()  # type: ignore[attr-defined]

    response = client.chat("system", "user")

    assert response.content == '{"params":{"executor.instances":2},"rationale":"ok"}'
    assert captured["path"] == "/api/chat"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
