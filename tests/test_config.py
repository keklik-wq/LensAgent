from pathlib import Path

from src.agent_shell.config import AppConfig


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
""",
        encoding="utf-8",
    )

    cfg = AppConfig.load(config_path)

    assert cfg.llm.local is not None
    assert cfg.spark_runtime.local is not None
    assert cfg.spark_runtime.local.app_id_prefix == "demo"
    assert cfg.spark_history.local is not None
