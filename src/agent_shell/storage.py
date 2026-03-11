from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputStats:
    path: str
    file_count: int
    total_bytes: int
    small_files: int
    format_hint: str | None


class OutputStorageInspector:
    def __init__(self, storage_type: str, base_path: str, max_list_files: int) -> None:
        self._type = storage_type
        self._base_path = base_path.rstrip("/")
        self._max_list_files = max_list_files

    def inspect(self, output_path: str) -> OutputStats:
        if self._type == "local":
            return self._inspect_local(output_path)
        if self._type in {"hdfs", "s3"}:
            raise NotImplementedError(
                f"Storage type '{self._type}' requires a custom adapter"
            )
        raise ValueError(f"Unknown storage type '{self._type}'")

    def _inspect_local(self, output_path: str) -> OutputStats:
        full_path = output_path
        if not full_path.startswith("/"):
            full_path = f"{self._base_path}/{output_path.lstrip('/')}"
        file_count = 0
        total_bytes = 0
        small_files = 0
        format_hint = None
        for root, _, files in os.walk(full_path):
            for name in files:
                file_count += 1
                if file_count > self._max_list_files:
                    break
                path = os.path.join(root, name)
                size = os.path.getsize(path)
                total_bytes += size
                if size < 8 * 1024 * 1024:
                    small_files += 1
                format_hint = format_hint or _guess_format(name)
            if file_count > self._max_list_files:
                break
        return OutputStats(
            path=full_path,
            file_count=file_count,
            total_bytes=total_bytes,
            small_files=small_files,
            format_hint=format_hint,
        )


def _guess_format(filename: str) -> str | None:
    lower = filename.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".orc"):
        return "orc"
    if lower.endswith(".avro"):
        return "avro"
    if lower.endswith(".json"):
        return "json"
    if lower.endswith(".csv"):
        return "csv"
    return None
