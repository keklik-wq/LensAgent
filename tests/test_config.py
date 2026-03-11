from src.agent_shell.config import ShellConfig


def test_load_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm_router:
  base_url: "http://router"
  api_key_env: "KEY"
  model: "m1"
  timeout_seconds: 10
  allow_models: ["m1"]
spark_history:
  base_url: "http://spark"
  timeout_seconds: 20
k8s:
  kube_context: "ctx"
  namespace: "ns"
  allow_pods_prefix: "spark-"
  max_log_bytes: 100
output_storage:
  type: "local"
  base_path: "/tmp"
  max_list_files: 10
policy:
  max_actions: 2
  allow_actions: ["a1"]
runtime:
  dry_run: true
"""
    )
    loaded = ShellConfig.load(cfg)
    assert loaded.llm_router.model == "m1"
    assert loaded.k8s.namespace == "ns"
    assert loaded.policy.allow_actions == ["a1"]
