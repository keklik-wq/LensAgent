from pathlib import Path

from src.agent_shell.storage import OutputStorageInspector


def test_local_storage_inspection(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "part-000.parquet").write_bytes(b"0" * 1024)
    (data_dir / "part-001.parquet").write_bytes(b"0" * 1024)
    inspector = OutputStorageInspector("local", str(tmp_path), max_list_files=10)
    stats = inspector.inspect("data")
    assert stats.file_count == 2
    assert stats.total_bytes == 2048
    assert stats.format_hint == "parquet"
