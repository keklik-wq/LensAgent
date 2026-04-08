import json
from argparse import Namespace
from pathlib import Path

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
        iterations=2,
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
