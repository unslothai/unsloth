# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

import math

from core.data_recipe.export import (
    RecipeDatasetExportError,
    _json_dumps_row,
    _sanitize_json_value,
    build_dataset_download,
    build_in_memory_dataset_download,
)


def _write_parquet_rows(parquet_dir: Path, rows: list[dict]) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    import pandas as pd

    parquet_dir.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(rows).to_parquet(parquet_dir / "batch_00000.parquet", index = False)


def test_build_in_memory_dataset_download_writes_jsonl(tmp_path: Path):
    rows = [{"instruction": "Say hi", "output": "Hello"}]
    file_path, media_type, download_name = build_in_memory_dataset_download(
        rows,
        filename_stem = "preview-run",
    )
    try:
        assert media_type == "application/x-ndjson"
        assert download_name == "preview-run.jsonl"
        lines = file_path.read_text(encoding = "utf-8").strip().splitlines()
        assert json.loads(lines[0]) == rows[0]
    finally:
        file_path.unlink(missing_ok = True)


def test_build_dataset_download_jsonl_from_artifact(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "recipe-datasets"
    dataset_path = artifact_root / "job-123"
    parquet_dir = dataset_path / "parquet-files"
    _write_parquet_rows(parquet_dir, [{"question": "Q1", "answer": "A1"}])

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, media_type, download_name = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "my-run",
    )
    try:
        assert media_type == "application/x-ndjson"
        assert download_name == "my-run.jsonl"
        lines = file_path.read_text(encoding = "utf-8").strip().splitlines()
        assert json.loads(lines[0])["question"] == "Q1"
    finally:
        file_path.unlink(missing_ok = True)


def test_json_dumps_row_replaces_non_finite_numbers_with_null():
    row = {"score": float("nan"), "ratio": float("inf"), "label": "ok"}
    payload = json.loads(_json_dumps_row(row))
    assert payload == {"score": None, "ratio": None, "label": "ok"}


def test_sanitize_json_value_handles_numpy_nan():
    numpy = pytest.importorskip("numpy")
    assert _sanitize_json_value(numpy.float64("nan")) is None
    assert _sanitize_json_value(numpy.float64("inf")) is None
    assert math.isclose(_sanitize_json_value(numpy.float64(1.5)), 1.5)


def test_build_dataset_download_preserves_row_order_within_shard(tmp_path: Path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    import pandas as pd

    artifact_root = tmp_path / "recipe-datasets"
    dataset_path = artifact_root / "job-order"
    parquet_dir = dataset_path / "parquet-files"
    parquet_dir.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(
        [
            {"sequence": "first"},
            {"sequence": "second"},
            {"sequence": "third"},
        ]
    ).to_parquet(parquet_dir / "batch_00000.parquet", index = False)

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "ordered",
    )
    try:
        lines = file_path.read_text(encoding = "utf-8").strip().splitlines()
        sequences = [json.loads(line)["sequence"] for line in lines]
        assert sequences == ["first", "second", "third"]
    finally:
        file_path.unlink(missing_ok = True)


def test_build_dataset_download_parquet_zip_includes_images(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "recipe-datasets"
    dataset_path = artifact_root / "job-images"
    parquet_dir = dataset_path / "parquet-files"
    images_dir = dataset_path / "images" / "nested"
    _write_parquet_rows(parquet_dir, [{"image": "images/nested/pic.png"}])
    images_dir.mkdir(parents = True, exist_ok = True)
    (images_dir / "pic.png").write_bytes(b"png-bytes")

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "parquet",
        filename_stem = "with-images",
    )
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = archive.namelist()
            image_bytes = archive.read("images/nested/pic.png")
        assert "batch_00000.parquet" in names
        assert "images/nested/pic.png" in names
        assert image_bytes == b"png-bytes"
    finally:
        file_path.unlink(missing_ok = True)


def test_build_dataset_download_parquet_zip_from_artifact(tmp_path: Path, monkeypatch):
    artifact_root = tmp_path / "recipe-datasets"
    dataset_path = artifact_root / "job-456"
    parquet_dir = dataset_path / "parquet-files"
    _write_parquet_rows(parquet_dir, [{"col": "value"}])

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, media_type, download_name = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "parquet",
        filename_stem = "parquet-run",
    )
    try:
        assert media_type == "application/zip"
        assert download_name == "parquet-run.parquet.zip"
        with zipfile.ZipFile(file_path) as archive:
            names = archive.namelist()
        assert names == ["batch_00000.parquet"]
    finally:
        file_path.unlink(missing_ok = True)


def test_build_dataset_download_missing_parquet_raises(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "empty-artifact"
    dataset_path.mkdir()
    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    with pytest.raises(RecipeDatasetExportError, match = "parquet"):
        build_dataset_download(
            artifact_path = str(dataset_path),
            export_format = "jsonl",
            filename_stem = "missing",
        )


def test_download_job_dataset_route_uses_artifact_path(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi import BackgroundTasks

    jobs_route = pytest.importorskip(
        "routes.data_recipe.jobs",
        reason = "studio backend routes unavailable",
    )

    captured: dict[str, str] = {}

    def fake_build_dataset_download(**kwargs):
        captured.update(kwargs)
        jsonl_path = tmp_path / "out.jsonl"
        jsonl_path.write_text('{"ok": true}\n', encoding = "utf-8")
        return jsonl_path, "application/x-ndjson", "run.jsonl"

    class _FakeManager:
        def get_status(self, job_id: str):
            return {
                "status": "completed",
                "artifact_path": "/tmp/artifacts/job-1",
            }

    monkeypatch.setattr(jobs_route, "build_dataset_download", fake_build_dataset_download)
    monkeypatch.setattr(jobs_route, "get_job_manager", lambda: _FakeManager())

    response = jobs_route.download_job_dataset(
        "job-1",
        background_tasks = BackgroundTasks(),
        export_format = "jsonl",
        artifact_path = None,
        filename = "My Run",
    )
    assert captured["artifact_path"] == "/tmp/artifacts/job-1"
    assert captured["filename_stem"] == "My Run"
    assert response.filename == "run.jsonl"


def test_download_job_dataset_route_rejects_incomplete_run(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import BackgroundTasks

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")

    class _FakeManager:
        def get_status(self, job_id: str):
            return {"status": "running", "artifact_path": "/tmp/artifacts/job-1"}

    monkeypatch.setattr(jobs_route, "get_job_manager", lambda: _FakeManager())

    with pytest.raises(HTTPException) as exc_info:
        jobs_route.download_job_dataset(
            "job-1",
            background_tasks = BackgroundTasks(),
            export_format = "jsonl",
            artifact_path = None,
            filename = None,
        )
    assert exc_info.value.status_code == 409


def test_build_in_memory_job_dataset_download_pages_all_rows(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi import BackgroundTasks

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")

    class _FakeManager:
        def __init__(self):
            self.rows = [{"index": index} for index in range(12_500)]
            self.calls: list[tuple[int, int]] = []

        def get_dataset(
            self,
            job_id: str,
            *,
            limit: int,
            offset: int = 0,
        ):
            self.calls.append((limit, offset))
            page = self.rows[offset : offset + limit]
            return {"dataset": page, "total": len(self.rows)}

    fake_manager = _FakeManager()
    monkeypatch.setattr(jobs_route, "get_job_manager", lambda: fake_manager)

    response = jobs_route.download_job_dataset(
        "job-big",
        background_tasks = BackgroundTasks(),
        export_format = "jsonl",
        artifact_path = None,
        filename = "big-run",
    )
    assert response.filename == "big-run.jsonl"
    assert fake_manager.calls == [
        (jobs_route._IN_MEMORY_DOWNLOAD_PAGE_SIZE, 0),
        (jobs_route._IN_MEMORY_DOWNLOAD_PAGE_SIZE, 10_000),
    ]
    lines = Path(response.path).read_text(encoding = "utf-8").strip().splitlines()
    assert len(lines) == 12_500
    assert json.loads(lines[-1]) == {"index": 12_499}
