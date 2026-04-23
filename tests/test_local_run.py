import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import main


def test_run_loop_with_local_backends(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(main, "RUNS_DIR", output_dir / "runs")
    monkeypatch.setattr(main, "load_dotenv", lambda: None)

    args = Namespace(
        config="examples/local/config.local.yaml",
        history_url=None,
        iterations=None,
        max_total_memory_gb=32,
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
            values=None,
            default=None,
        ),
        "executor.cores": main.TuningParamConfig(
            path=["spec", "executor", "cores"],
            type="int",
            min=1,
            max=4,
            values=None,
            default=None,
        ),
        "executor.instances": main.TuningParamConfig(
            path=["spec", "executor", "instances"],
            type="int",
            min=1,
            max=8,
            values=None,
            default=None,
        ),
        "executor.memory_gb": main.TuningParamConfig(
            path=["spec", "executor", "memory"],
            type="memory_gb",
            min=1,
            max=8,
            values=None,
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


def test_update_manifest_name_includes_campaign_id() -> None:
    manifest = {"metadata": {"name": "spark-app"}}

    updated = main._update_manifest_name(manifest, run_id="001", campaign_id="abcd")

    assert updated["metadata"]["name"] == "spark-app-abcd-r001"


def test_apply_params_to_manifest_serializes_spark_conf_values_as_strings() -> None:
    manifest = {
        "spec": {
            "sparkConf": {"spark.sql.shuffle.partitions": "4000"},
            "executor": {"cores": 4},
        }
    }
    tuning_params = {
        "spark.sql.shuffle.partitions": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.shuffle.partitions"],
            type="int",
            min=1,
            max=10000,
            values=None,
            default=None,
        ),
        "executor.cores": main.TuningParamConfig(
            path=["spec", "executor", "cores"],
            type="int",
            min=1,
            max=16,
            values=None,
            default=None,
        ),
    }

    updated = main._apply_params_to_manifest(
        manifest=manifest,
        params={"spark.sql.shuffle.partitions": 2500, "executor.cores": 6},
        tuning_params=tuning_params,
    )

    assert updated["spec"]["sparkConf"]["spark.sql.shuffle.partitions"] == "2500"
    assert updated["spec"]["executor"]["cores"] == 6


def test_apply_params_to_manifest_keeps_enum_spark_conf_values_as_strings() -> None:
    manifest = {
        "spec": {
            "sparkConf": {"spark.sql.parquet.compression.codec": "zstd"},
        }
    }
    tuning_params = {
        "spark.sql.parquet.compression.codec": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.parquet.compression.codec"],
            type="enum",
            min=None,
            max=None,
            values=["gzip", "zstd", "lz4"],
            default=None,
        ),
    }

    updated = main._apply_params_to_manifest(
        manifest=manifest,
        params={"spark.sql.parquet.compression.codec": "gzip"},
        tuning_params=tuning_params,
    )

    assert updated["spec"]["sparkConf"]["spark.sql.parquet.compression.codec"] == "gzip"


def test_apply_constraints_treats_numeric_string_bounds_as_numeric() -> None:
    tuning_params = {
        "spark.sql.shuffle.partitions": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.shuffle.partitions"],
            type="str",
            min="200",
            max="10000",
            values=None,
            default=None,
        )
    }

    resolved = main._apply_constraints(
        params={},
        base_params={"spark.sql.shuffle.partitions": "2500"},
        tuning_params=tuning_params,
        driver_memory_gb=1,
        max_total_memory_gb=None,
    )

    assert resolved["spark.sql.shuffle.partitions"] == "2500"


def test_apply_constraints_rejects_enum_value_outside_allowed_set() -> None:
    tuning_params = {
        "spark.sql.parquet.compression.codec": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.parquet.compression.codec"],
            type="enum",
            min=None,
            max=None,
            values=["gzip", "zstd", "lz4"],
            default=None,
        )
    }

    with pytest.raises(ValueError, match="expects one of"):
        main._apply_constraints(
            params={"spark.sql.parquet.compression.codec": "snappy"},
            base_params={"spark.sql.parquet.compression.codec": "gzip"},
            tuning_params=tuning_params,
            driver_memory_gb=1,
            max_total_memory_gb=None,
        )


def test_resolve_duplicate_params_switches_enum_value() -> None:
    tuning_params = {
        "spark.sql.parquet.compression.codec": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.parquet.compression.codec"],
            type="enum",
            min=None,
            max=None,
            values=["gzip", "zstd", "lz4"],
            default=None,
        )
    }

    resolved = main._resolve_duplicate_params(
        params={"spark.sql.parquet.compression.codec": "gzip"},
        history=[{"params": {"spark.sql.parquet.compression.codec": "gzip"}}],
        base_params={"spark.sql.parquet.compression.codec": "gzip"},
        tuning_params=tuning_params,
        driver_memory_gb=1,
        max_total_memory_gb=None,
    )

    assert resolved["spark.sql.parquet.compression.codec"] in {"zstd", "lz4"}


def test_validate_params_within_bounds_raises_for_base_manifest_out_of_range() -> None:
    tuning_params = {
        "spark.sql.shuffle.partitions": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.shuffle.partitions"],
            type="str",
            min="3000",
            max="10000",
            values=None,
            default=None,
        )
    }

    with pytest.raises(ValueError, match="Base manifest value for spark.sql.shuffle.partitions"):
        main._validate_params_within_bounds(
            {"spark.sql.shuffle.partitions": "2500"},
            tuning_params,
            source_label="Base manifest value",
        )


def test_validate_params_within_bounds_rejects_base_manifest_enum_value_outside_allowed_set() -> None:
    tuning_params = {
        "spark.sql.parquet.compression.codec": main.TuningParamConfig(
            path=["spec", "sparkConf", "spark.sql.parquet.compression.codec"],
            type="enum",
            min=None,
            max=None,
            values=["gzip", "zstd", "lz4"],
            default=None,
        )
    }

    with pytest.raises(
        ValueError,
        match="Base manifest value value for tuning param expects one of",
    ):
        main._validate_params_within_bounds(
            {"spark.sql.parquet.compression.codec": "snappy"},
            tuning_params,
            source_label="Base manifest value",
        )


def test_load_stages_with_retry_does_not_switch_to_latest_app_id() -> None:
    calls: list[str] = []

    class FakeHistoryProvider:
        def get_stages(self, app_id: str) -> list[dict[str, object]]:
            calls.append(app_id)
            raise RuntimeError("missing")

        def latest_app_id(self) -> str | None:
            return "spark-other"

    with pytest.raises(RuntimeError, match="spark-original"):
        main._load_stages_with_retry(
            app_id="spark-original",
            history_provider=FakeHistoryProvider(),
            poll_seconds=0,
            timeout_seconds=0.01,
        )

    assert calls
    assert all(app_id == "spark-original" for app_id in calls)


def test_run_loop_deletes_active_application_on_keyboard_interrupt(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    monkeypatch.setattr(main, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(main, "RUNS_DIR", output_dir / "runs")
    monkeypatch.setattr(main, "LOG_DIR", output_dir / "logs")
    monkeypatch.setattr(main, "load_dotenv", lambda: None)
    monkeypatch.setattr(main, "_build_campaign_id", lambda: "abcd")

    deleted: list[tuple[str, str]] = []

    class FakeRuntime:
        def run_application(
            self,
            manifest: dict[str, object],
            namespace: str,
            driver_container: str | None = None,
        ):
            del manifest, namespace, driver_container
            raise KeyboardInterrupt

        def delete_application(self, manifest: dict[str, object], namespace: str) -> None:
            deleted.append((str(manifest["metadata"]["name"]), namespace))  # type: ignore[index]

    class FakeHistoryProvider:
        def ui_url(self, app_id: str) -> str:
            return f"http://history/{app_id}"

    monkeypatch.setattr(main, "build_spark_runtime", lambda config, kube_context=None: FakeRuntime())
    monkeypatch.setattr(
        main,
        "build_spark_history_provider",
        lambda config, base_url_override=None: FakeHistoryProvider(),
    )
    monkeypatch.setattr(main, "build_llm_client", lambda config: object())

    args = Namespace(
        config="examples/local/config.local.yaml",
        history_url=None,
        iterations=1,
        max_total_memory_gb=32,
        driver_container=None,
        kube_context=None,
        namespace=None,
    )

    main._ensure_dirs()

    with pytest.raises(SystemExit, match="Interrupted, active SparkApplication deleted."):
        main.run_loop(args)

    assert deleted == [("local-spark-job-abcd-r001", "local-jobs")]
