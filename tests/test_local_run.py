import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import main


def test_run_loop_with_local_backends(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(main, "RUNS_DIR", output_dir / "runs")
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    args = Namespace(
        manifest="examples/local/sparkapp.yaml",
        transform="examples/local/job.py",
        config="examples/local/config.local.yaml",
        history_url=None,
        iterations=None,
        max_total_memory_gb=32,
        use_base_for_first=True,
        use_random_for_first=False,
        driver_container=None,
        kube_context=None,
        namespace=None,
    )

    main._ensure_dirs()
    main.run_loop(args)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    first_run = json.loads(
        (output_dir / "runs" / "run_001" / "run_001.json").read_text(encoding="utf-8")
    )
    second_run = json.loads(
        (output_dir / "runs" / "run_002" / "run_002.json").read_text(encoding="utf-8")
    )

    assert summary["best_run_id"] == "001"
    assert first_run["application_state"] == "COMPLETED"
    assert second_run["app_id"] == "local-app-002"
    assert second_run["requested_gb_seconds"] is not None


def test_build_llm_history_entry_excludes_driver_logs() -> None:
    entry = main._build_llm_history_entry(
        {
            "run_id": "001",
            "params": {"executor.instances": 2},
            "rationale": "test",
            "application_state": "COMPLETED",
            "runtime_seconds": 1.5,
            "requested_gb": 4,
            "requested_gb_seconds": 6.0,
            "spill_gb": 0.0,
            "output_files": None,
            "small_files": None,
            "spark_ui": "http://spark-ui",
            "history_api": "http://history-api",
            "driver_logs": "very large logs",
            "driver_logs_path": "output/runs/run_001/driver.log",
            "manifest_path": "output/runs/run_001/manifest_001.yaml",
        }
    )

    assert entry["run_id"] == "001"
    assert entry["params"] == {"executor.instances": 2}
    assert "driver_logs" not in entry
    assert "driver_logs_path" not in entry
    assert "manifest_path" not in entry


def test_build_tuning_prompt_prefers_conservative_changes() -> None:
    history = [
        {
            "run_id": "001",
            "params": {
                "spark.sql.shuffle.partitions": 200,
                "executor.cores": 1,
                "executor.instances": 1,
                "executor.memory_gb": 1,
                "driver.memory_gb": 1,
            },
            "rationale": "Base config for first run.",
            "application_state": "COMPLETED",
            "runtime_seconds": 3.255,
            "requested_gb": 2,
            "requested_gb_seconds": 6.51,
            "spill_gb": 0.0,
            "output_files": None,
            "small_files": None,
            "spark_ui": "http://spark-ui",
            "history_api": "http://history-api",
        }
    ]

    system, user = main._build_tuning_prompt(
        system_prompt="Custom prompt from yaml.",
        history=history,
        base_params={
            "spark.sql.shuffle.partitions": 8,
            "executor.cores": 1,
            "executor.instances": 1,
            "executor.memory_gb": 1,
        },
        tunable_param_specs={
            "spark.sql.shuffle.partitions": {
                "type": "int",
                "min": 1,
                "max": 1000,
                "default": None,
                "path": ["spec", "sparkConf", "spark.sql.shuffle.partitions"],
            }
        },
        constraints={"total_memory_gb": {"max": 500}},
        response_schema={"params": {}, "rationale": "string"},
        history_label="http://history/latest",
    )
    payload = json.loads(user)

    assert system == "Custom prompt from yaml."
    assert "tunable_params" in payload
    assert payload["best_previous_run"]["run_id"] == "001"
    assert payload["latest_run"]["run_id"] == "001"


def test_resolve_duplicate_params_changes_candidate() -> None:
    history = [
        {
            "params": {
                "spark.sql.shuffle.partitions": 200,
                "executor.cores": 1,
                "executor.instances": 1,
                "executor.memory_gb": 1,
                "driver.memory_gb": 1,
            }
        }
    ]
    tuning_params = {
        "spark.sql.shuffle.partitions": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.shuffle.partitions"],
            type="int",
            min=1,
            max=1000,
            default=None,
        ),
        "executor.cores": main.TuningParamConfig(
            path=["spec", "executor", "cores"],
            type="int",
            min=1,
            max=4,
            default=None,
        ),
        "executor.instances": main.TuningParamConfig(
            path=["spec", "executor", "instances"],
            type="int",
            min=1,
            max=8,
            default=None,
        ),
        "executor.memory_gb": main.TuningParamConfig(
            path=["spec", "executor", "memory"],
            type="memory_gb",
            min=1,
            max=8,
            default=None,
        ),
    }
    params = {
        "spark.sql.shuffle.partitions": 200,
        "executor.cores": 1,
        "executor.instances": 1,
        "executor.memory_gb": 1,
    }

    resolved = main._resolve_duplicate_params(
        params=params,
        history=history,
        base_params=params,
        tuning_params=tuning_params,
        driver_memory_gb=1,
        max_total_memory_gb=8,
    )

    assert resolved != params
    assert resolved["spark.sql.shuffle.partitions"] != 200


def test_request_tuning_candidate_retries_invalid_json() -> None:
    class FakeLlm:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def chat(self, system: str, user: str, temperature: float = 0.2) -> SimpleNamespace:
            del temperature
            self.calls.append((system, user))
            if len(self.calls) == 1:
                return SimpleNamespace(
                    content='{"params":{"executor.instances":2},"rationale":"broken}'
                )
            return SimpleNamespace(
                content='{"params":{"executor.instances":2},"rationale":"ok"}'
            )

    llm = FakeLlm()
    logger = main.logging.getLogger("test")

    parsed = main._request_tuning_candidate(
        llm=llm,
        system_prompt="system",
        user_prompt='{"response_schema":{"params":{},"rationale":"string"}}',
        llm_json_retries=1,
        logger=logger,
    )

    assert parsed["params"]["executor.instances"] == 2
    assert len(llm.calls) == 2
    retry_payload = json.loads(llm.calls[1][1])
    assert retry_payload["retry_feedback"]["attempt"] == 2
    assert "invalid_response_excerpt" in retry_payload["retry_feedback"]
