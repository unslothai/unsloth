# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import secrets
import tempfile
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
)


def _write_parquet_rows(parquet_dir: Path, rows: list[dict]) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    import pandas as pd

    parquet_dir.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(rows).to_parquet(parquet_dir / "batch_00000.parquet", index = False)


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


def test_build_dataset_download_exports_every_row_once_across_shards(tmp_path: Path, monkeypatch):
    """A dataset bigger than one fetch chunk still comes out whole, in artifact order.

    Reading it a page at a time re-derived the row order per page, and DuckDB's parallel parquet
    scan does not repeat it, so the pages overlapped and gapped: 120k rows exported as 72k
    distinct ones.
    """
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    import pandas as pd

    dataset_path = tmp_path / "recipe-datasets" / "job-big"
    parquet_dir = dataset_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    shard_size = 40_000
    for shard, start in enumerate(range(0, 2 * shard_size, shard_size)):
        pd.DataFrame({"i": range(start, start + shard_size)}).to_parquet(
            parquet_dir / f"batch_{shard:05d}.parquet",
            index = False,
            row_group_size = 5_000,
        )

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "big",
    )
    try:
        exported = [
            json.loads(line)["i"] for line in file_path.read_text(encoding = "utf-8").splitlines()
        ]
    finally:
        file_path.unlink(missing_ok = True)
    assert exported == list(range(2 * shard_size))


def test_build_dataset_download_leaves_no_temp_file_when_export_fails(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "recipe-datasets" / "job-fails"
    _write_parquet_rows(dataset_path / "parquet-files", [{"col": "value"}])
    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )
    monkeypatch.setattr(
        "core.data_recipe.export._write_jsonl_from_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("export blew up")),
    )

    before = set(Path(tempfile.gettempdir()).glob("*.jsonl"))
    with pytest.raises(RuntimeError):
        build_dataset_download(
            artifact_path = str(dataset_path),
            export_format = "jsonl",
            filename_stem = "leaky",
        )
    assert set(Path(tempfile.gettempdir()).glob("*.jsonl")) == before


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

        def get_status(self, job_id: str):
            # A preview run: completed, but with nothing persisted to export from.
            return {"status": "completed", "artifact_path": None}

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


def _download_app(monkeypatch, tmp_path: Path, jobs_route):
    """The real /api/data-recipe mount, with an auth database of its own."""
    from fastapi import FastAPI

    from auth import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    storage._reset_api_key_hash_cache()
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )

    from auth.authentication import create_access_token
    from routes.data_recipe import router as data_recipe_router

    def fake_build_dataset_download(**_kwargs):
        jsonl_path = tmp_path / "out.jsonl"
        jsonl_path.write_text('{"ok": true}\n', encoding = "utf-8")
        return jsonl_path, "application/x-ndjson", "run.jsonl"

    class _FakeManager:
        def get_status(self, job_id: str):
            return {"status": "completed", "artifact_path": "/tmp/artifacts/job-1"}

    monkeypatch.setattr(jobs_route, "build_dataset_download", fake_build_dataset_download)
    monkeypatch.setattr(jobs_route, "get_job_manager", lambda: _FakeManager())

    app = FastAPI()
    app.include_router(data_recipe_router, prefix = "/api/data-recipe")
    return app, create_access_token(storage.DEFAULT_ADMIN_USERNAME)


def test_download_route_accepts_the_bearer_from_the_query(monkeypatch, tmp_path: Path):
    """Neither an <a download> nor the native save command can set a header, so ?token= is the
    only credential this URL can carry. Behind the package's header-only guard it answered 401."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, token = _download_app(monkeypatch, tmp_path, jobs_route)
    client = TestClient(app)

    query = client.get("/api/data-recipe/jobs/job-1/download", params = {"token": token})
    assert query.status_code == 200

    header = client.get(
        "/api/data-recipe/jobs/job-1/download",
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert header.status_code == 200


def test_download_route_refuses_a_caller_with_no_token(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, _token = _download_app(monkeypatch, tmp_path, jobs_route)

    anonymous = TestClient(app).get("/api/data-recipe/jobs/job-1/download")
    assert anonymous.status_code == 401
    bad = TestClient(app).get(
        "/api/data-recipe/jobs/job-1/download",
        params = {"token": "not-a-real-token"},
    )
    assert bad.status_code == 401


def test_build_dataset_download_refuses_a_path_outside_the_dataset_roots():
    """An absolute path from outside every dataset root is refused the same way one merely outside
    the recipe root is. It used to escape as a bare ValueError, which the route answered 500 to."""
    with pytest.raises(RecipeDatasetExportError):
        build_dataset_download(
            artifact_path = "/etc",
            export_format = "jsonl",
            filename_stem = "escape",
        )


def test_build_dataset_download_falls_back_when_duckdb_cannot_read(tmp_path: Path, monkeypatch):
    """No duckdb (or a duckdb that refuses the query) still exports the whole dataset."""
    dataset_path = tmp_path / "recipe-datasets" / "job-fallback"
    _write_parquet_rows(dataset_path / "parquet-files", [{"i": 0}, {"i": 1}, {"i": 2}])
    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )
    monkeypatch.setattr(
        "core.data_recipe.export._stream_jsonl_from_parquet_with_duckdb",
        lambda **kwargs: False,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "fallback",
    )
    try:
        exported = [
            json.loads(line)["i"] for line in file_path.read_text(encoding = "utf-8").splitlines()
        ]
    finally:
        file_path.unlink(missing_ok = True)
    assert exported == [0, 1, 2]


def test_to_jsonable_maps_pandas_missing_sentinels_to_none():
    """NaT answers hasattr(isoformat) and isoformat()s to the string "NaT"; NA reaches the str()
    fallback as "<NA>". Either one writes a real value where the dataset had none."""
    pd = pytest.importorskip("pandas")
    from core.data_recipe.jsonable import to_jsonable, to_preview_jsonable

    for sentinel in (pd.NA, pd.NaT):
        assert to_jsonable(sentinel) is None
        assert to_preview_jsonable(sentinel) is None


def test_build_dataset_download_writes_a_missing_timestamp_as_null(tmp_path: Path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")

    dataset_path = tmp_path / "recipe-datasets" / "job-nat"
    parquet_dir = dataset_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    pd.DataFrame(
        {
            "seen_at": pd.to_datetime(["2020-01-01", None]),
            "score": pd.array([1, None], dtype = "Int64"),
        }
    ).to_parquet(parquet_dir / "batch_00000.parquet", index = False)

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "nat",
    )
    try:
        rows = [json.loads(line) for line in file_path.read_text(encoding = "utf-8").splitlines()]
    finally:
        file_path.unlink(missing_ok = True)
    assert rows[0]["seen_at"].startswith("2020-01-01")
    assert rows[1] == {"seen_at": None, "score": None}


def _write_appledouble_companion(path: Path) -> None:
    from utils.paths.path_utils import _MAGIC
    path.write_bytes(_MAGIC + b"\x00" * 60)


def test_build_dataset_download_ignores_appledouble_companions(tmp_path: Path, monkeypatch):
    """A macOS volume writes extended attributes to ._batch.parquet. DuckDB cannot parse one, so
    keeping it in the shard list took the whole streaming path down to the slowest fallback, and
    put a file no reader accepts inside the archive."""
    pytest.importorskip("duckdb")
    dataset_path = tmp_path / "recipe-datasets" / "job-macos"
    parquet_dir = dataset_path / "parquet-files"
    _write_parquet_rows(parquet_dir, [{"i": 0}, {"i": 1}])
    _write_appledouble_companion(parquet_dir / "._batch_00000.parquet")
    images_dir = dataset_path / "images"
    images_dir.mkdir(parents = True)
    (images_dir / "pic.png").write_bytes(b"png-bytes")
    _write_appledouble_companion(images_dir / "._pic.png")

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    from core.data_recipe.export import _stream_jsonl_from_parquet_with_duckdb

    streamed = tmp_path / "streamed.jsonl"
    assert _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = streamed,
    )
    assert [json.loads(line)["i"] for line in streamed.read_text().splitlines()] == [0, 1]

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "parquet",
        filename_stem = "macos",
    )
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = archive.namelist()
    finally:
        file_path.unlink(missing_ok = True)
    assert names == ["batch_00000.parquet", "images/pic.png"]


def test_build_dataset_download_keeps_a_real_file_named_like_a_companion(
    tmp_path: Path, monkeypatch
):
    """The name alone does not decide it: a genuine shard called ._batch.parquet is still a shard."""
    dataset_path = tmp_path / "recipe-datasets" / "job-dotunder"
    parquet_dir = dataset_path / "parquet-files"
    _write_parquet_rows(parquet_dir, [{"i": 0}])
    (parquet_dir / "batch_00000.parquet").rename(parquet_dir / "._batch_00000.parquet")

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    file_path, _, _ = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "dotunder",
    )
    try:
        assert [json.loads(line)["i"] for line in file_path.read_text().splitlines()] == [0]
    finally:
        file_path.unlink(missing_ok = True)
