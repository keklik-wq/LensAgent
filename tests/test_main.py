import os

import main
from src.agent_shell.config import ShellConfig


def _dummy_config():
    return ShellConfig.load("config.yaml")


def test_main_requires_env(monkeypatch):
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.delenv("APP_ID", raising=False)
    try:
        main.main()
        raised = False
    except SystemExit:
        raised = True
    assert raised


def test_main_runs_with_env(monkeypatch):
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setenv("CONFIG_PATH", "config.yaml")
    monkeypatch.setenv("APP_ID", "app-1")
    monkeypatch.setenv("K8S_NAMESPACE", "ns")
    monkeypatch.setenv("OUTPUT_PATH", "/tmp/out")

    called = {}

    class DummyShell:
        def __init__(self, _cfg):
            pass

        def run(self, ctx):
            called["app_id"] = ctx.app_id
            called["namespace"] = ctx.namespace
            called["output_path"] = ctx.output_path

            class DummyResult:
                summary = "ok"
                proposals = []
                diagnostics = {}

            return DummyResult()

    monkeypatch.setattr(main, "ShellConfig", type("SC", (), {"load": lambda _p: _dummy_config()}))
    monkeypatch.setattr(main, "AgentShell", DummyShell)
    main.main()
    assert called["app_id"] == "app-1"
    assert called["namespace"] == "ns"
    assert called["output_path"] == "/tmp/out"
