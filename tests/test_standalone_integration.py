from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _docker_compose_available() -> bool:
    result = subprocess.run(
        ["docker", "compose", "version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run_compose(project_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-p", project_name, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _format_failure(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"exit_code={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_DOCKER_INTEGRATION") != "1",
    reason="Set RUN_DOCKER_INTEGRATION=1 to run Docker-based integration tests.",
)
def test_standalone_runtime_runs_job_via_spark_submit() -> None:
    if not _docker_compose_available():
        pytest.skip("docker compose is not available")

    project_name = f"lensagent-it-{uuid.uuid4().hex[:8]}"
    output_dir = PROJECT_ROOT / "output"
    backup_dir = PROJECT_ROOT / f"output.backup.{uuid.uuid4().hex[:8]}"

    had_output = output_dir.exists()
    if had_output:
        shutil.move(str(output_dir), str(backup_dir))

    try:
        result = _run_compose(
            project_name,
            "run",
            "--build",
            "--rm",
            "lens-agent-standalone",
            "python3",
            "main.py",
            "--config",
            "examples/docker/config.standalone.yaml",
            "--iterations",
            "1",
        )
        assert result.returncode == 0, _format_failure(result)

        summary_path = output_dir / "summary.json"
        run_meta_path = output_dir / "runs" / "run_001" / "run_001.json"
        assert summary_path.exists(), "summary.json was not created"
        assert run_meta_path.exists(), "run_001 metadata was not created"

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))

        assert summary["best_run_id"] == "001"
        assert run_meta["application_state"] == "COMPLETED"
        assert str(run_meta["app_id"]).startswith(("app-", "spark-"))
        assert run_meta["requested_gb_seconds"] is not None
        assert str(run_meta["history_api"]).endswith(f"/applications/{run_meta['app_id']}/stages")
    finally:
        down_result = _run_compose(project_name, "down", "-v", "--remove-orphans")
        if down_result.returncode != 0:
            print(_format_failure(down_result))

        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        if had_output and backup_dir.exists():
            shutil.move(str(backup_dir), str(output_dir))
