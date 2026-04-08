from pathlib import Path

from src.agent_shell.config import AppConfig
from src.agent_shell.factory import build_llm_client
from src.agent_shell.ollama import OllamaLlmClient


def test_loads_legacy_router_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm_router:
  base_url: "http://router"
  api_key_env: "KEY"
  model: "m1"
  timeout_seconds: 10
  allow_models: ["m1"]
spark_history:
  backend: "local"
  local:
    fixtures_path: "examples/local/history"
    base_url: "http://history"
    default_app_id: "local-app-001"
""",
        encoding="utf-8",
    )

    cfg = AppConfig.load(config_path)

    assert cfg.llm.backend == "router"
    assert cfg.llm.router is not None
    assert cfg.llm.router.model == "m1"
    assert cfg.spark_runtime.backend == "kubernetes"
    assert cfg.spark_runtime.kubernetes is not None
    assert cfg.spark_runtime.kubernetes.kubeconfig_path is None
    assert cfg.tuning.iterations == 2
    assert "requested_gb_seconds" in cfg.tuning.prompt
    assert cfg.tuning.llm_json_retries == 2


def test_loads_local_backends_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  backend: "local"
  local:
    strategy: "best_previous"
spark_runtime:
  backend: "local"
  local:
    app_id_prefix: "demo"
    final_state: "COMPLETED"
    driver_log_template: "{app_id}"
spark_history:
  backend: "local"
  local:
    fixtures_path: "examples/local/history"
    base_url: "http://history"
    default_app_id: "local-app-001"
tuning:
  iterations: 4
  prompt: "Custom prompt from yaml."
  llm_json_retries: 5
  params:
    executor.instances:
      path: "spec.executor.instances"
      type: "int"
      min: 1
      max: 10
""",
        encoding="utf-8",
    )

    cfg = AppConfig.load(config_path)

    assert cfg.llm.local is not None
    assert cfg.spark_runtime.local is not None
    assert cfg.spark_runtime.local.app_id_prefix == "demo"
    assert cfg.spark_history.local is not None
    assert cfg.tuning.iterations == 4
    assert cfg.tuning.prompt == "Custom prompt from yaml."
    assert cfg.tuning.llm_json_retries == 5


def test_loads_ollama_config_and_builds_client(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  backend: "ollama"
  ollama:
    base_url: "http://ollama:11434"
    model: "qwen2.5:3b"
    timeout_seconds: 45
    keep_alive: "10m"
    options:
      num_predict: 128
spark_runtime:
  backend: "local"
  local:
    app_id_prefix: "demo"
    final_state: "COMPLETED"
    driver_log_template: "{app_id}"
spark_history:
  backend: "local"
  local:
    fixtures_path: "examples/local/history"
    base_url: "http://history"
    default_app_id: "local-app-001"
""",
        encoding="utf-8",
    )

    cfg = AppConfig.load(config_path)
    client = build_llm_client(cfg)

    assert cfg.llm.ollama is not None
    assert cfg.llm.ollama.base_url == "http://ollama:11434"
    assert cfg.llm.ollama.options == {"num_predict": 128}
    assert isinstance(client, OllamaLlmClient)


def test_loads_kubernetes_kubeconfig_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  backend: "local"
  local:
    strategy: "best_previous"
spark_runtime:
  backend: "kubernetes"
  kubernetes:
    kube_context: "dev-cluster"
    kubeconfig_path: "/tmp/kubeconfig"
spark_history:
  backend: "local"
  local:
    fixtures_path: "examples/local/history"
    base_url: "http://history"
    default_app_id: "local-app-001"
""",
        encoding="utf-8",
    )

    cfg = AppConfig.load(config_path)

    assert cfg.spark_runtime.kubernetes is not None
    assert cfg.spark_runtime.kubernetes.kube_context == "dev-cluster"
    assert cfg.spark_runtime.kubernetes.kubeconfig_path == "/tmp/kubeconfig"


def test_ollama_client_parses_chat_response() -> None:
    client = OllamaLlmClient(
        base_url="http://ollama:11434",
        model="qwen2.5:3b",
        timeout_seconds=30,
        keep_alive="5m",
        options={"num_predict": 64},
    )

    class FakeHttpClient:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            self.payload = {"path": path, **payload}
            return {
                "model": "qwen2.5:3b",
                "message": {
                    "role": "assistant",
                    "content": '{"params":{"executor.instances":2},"rationale":"ok"}',
                },
            }

    fake_http = FakeHttpClient()
    client._client = fake_http  # type: ignore[attr-defined]

    response = client.chat("system", '{"history":[],"base_params":{}}', temperature=0.4)

    assert response.content == '{"params":{"executor.instances":2},"rationale":"ok"}'
    assert fake_http.payload is not None
    assert fake_http.payload["path"] == "/api/chat"
    assert fake_http.payload["model"] == "qwen2.5:3b"
    assert fake_http.payload["keep_alive"] == "5m"
    assert fake_http.payload["stream"] is False
    assert fake_http.payload["format"] == "json"
    assert fake_http.payload["options"] == {"temperature": 0.4, "num_predict": 64}
