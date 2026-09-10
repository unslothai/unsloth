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
    """Paging re-derived the row order per page and DuckDB's parallel scan does not repeat it, so
    the pages overlapped and gapped: 120k rows came out as 72k distinct ones."""
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
    # These tests are about who may fetch the link, not about what the export is called.
    monkeypatch.setattr(
        jobs_route,
        "download_filename",
        lambda *, artifact_path, export_format, stem: f"{stem}.jsonl",
    )

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
    # These tests are about who may fetch the link, not about what the export ends up called.
    monkeypatch.setattr(
        jobs_route,
        "download_filename",
        lambda *, artifact_path, export_format, stem: f"{stem}.jsonl",
    )

    app = FastAPI()
    app.include_router(data_recipe_router, prefix = "/api/data-recipe")
    return app, create_access_token(storage.DEFAULT_ADMIN_USERNAME)


def test_download_link_is_minted_over_the_bearer_and_used_without_one(monkeypatch, tmp_path: Path):
    """The URL is fetched without a header, so it carries a signed capability rather than the
    session token, which download history would keep."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, token = _download_app(monkeypatch, tmp_path, jobs_route)
    client = TestClient(app)

    minted = client.get(
        "/api/data-recipe/jobs/job-1/download-url",
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert minted.status_code == 200
    url = "/api/data-recipe" + minted.json()["path"]
    assert token not in url

    # No Authorization header at all, the way the browser fetches it.
    assert client.get(url).status_code == 200
    # And the header still works on its own, for an API client.
    assert (
        client.get(
            "/api/data-recipe/jobs/job-1/download",
            headers = {"Authorization": f"Bearer {token}"},
        ).status_code
        == 200
    )


def test_minting_a_download_link_needs_the_bearer(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, _token = _download_app(monkeypatch, tmp_path, jobs_route)

    assert TestClient(app).get("/api/data-recipe/jobs/job-1/download-url").status_code == 401


def test_download_route_refuses_an_unsigned_or_repointed_link(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from urllib.parse import parse_qs, urlparse

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, token = _download_app(monkeypatch, tmp_path, jobs_route)
    client = TestClient(app)

    assert client.get("/api/data-recipe/jobs/job-1/download").status_code == 401
    assert (
        client.get(
            "/api/data-recipe/jobs/job-1/download",
            params = {"token": "not-a-real-token"},
        ).status_code
        == 401
    )
    # A session JWT is not a download link, however valid it is as a bearer.
    assert (
        client.get(
            "/api/data-recipe/jobs/job-1/download",
            params = {"token": token},
        ).status_code
        == 401
    )

    url = client.get(
        "/api/data-recipe/jobs/job-1/download-url",
        params = {"artifact_path": "/recipes/mine"},
        headers = {"Authorization": f"Bearer {token}"},
    ).json()["path"]
    signed = parse_qs(urlparse(url).query)["token"][0]
    # Every parameter the export reads is signed, so the artifact cannot be swapped for another.
    assert (
        client.get(
            "/api/data-recipe/jobs/job-1/download",
            params = {"artifact_path": "/recipes/someone-else", "token": signed},
        ).status_code
        == 401
    )


def test_download_link_expires(monkeypatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, token = _download_app(monkeypatch, tmp_path, jobs_route)
    client = TestClient(app)

    url = (
        "/api/data-recipe"
        + client.get(
            "/api/data-recipe/jobs/job-1/download-url",
            headers = {"Authorization": f"Bearer {token}"},
        ).json()["path"]
    )
    assert client.get(url).status_code == 200

    real_time = jobs_route.time.time
    monkeypatch.setattr(
        jobs_route.time,
        "time",
        lambda: real_time() + jobs_route._DOWNLOAD_LINK_TTL + 1,
    )
    assert client.get(url).status_code == 401


def _write_appledouble_companion(path: Path) -> None:
    from utils.paths.path_utils import _MAGIC
    path.write_bytes(_MAGIC + b"\x00" * 60)


def test_build_dataset_download_ignores_appledouble_companions(tmp_path: Path, monkeypatch):
    """DuckDB cannot parse a ._batch.parquet companion, so keeping it in the shard list dropped the
    export to the slowest fallback and put an unreadable file in the archive."""
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


def test_both_readers_export_a_decimal_column_as_the_same_number(tmp_path: Path, monkeypatch):
    """The same artifact exported as 1.2 or as "1.20" depending on which reader was available."""
    pytest.importorskip("duckdb")
    pyarrow = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pyarrow_parquet
    from decimal import Decimal

    from core.data_recipe.export import (
        _stream_jsonl_from_parquet_with_duckdb,
        _write_jsonl_with_pyarrow,
    )

    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    pyarrow_parquet.write_table(
        pyarrow.table({"price": pyarrow.array([Decimal("1.20")], type = pyarrow.decimal128(10, 2))}),
        parquet_dir / "batch_00000.parquet",
    )

    streamed = tmp_path / "streamed.jsonl"
    assert _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = streamed,
    )
    from_pyarrow = tmp_path / "pyarrow.jsonl"
    assert _write_jsonl_with_pyarrow(parquet_dir, from_pyarrow)

    assert json.loads(streamed.read_text().strip()) == {"price": 1.2}
    assert json.loads(from_pyarrow.read_text().strip()) == {"price": 1.2}


def test_to_jsonable_maps_pandas_missing_sentinels_to_none():
    """NaT isoformat()s to "NaT" and NA hits the str() fallback as "<NA>", either of which writes
    a real value where the dataset had none."""
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


def test_jsonl_export_ships_the_images_its_rows_reference(tmp_path: Path, monkeypatch):
    """Rows carry relative image paths, so a bare JSONL loses every image."""
    dataset_path = tmp_path / "recipe-datasets" / "job-multimodal"
    _write_parquet_rows(dataset_path / "parquet-files", [{"image": "images/nested/pic.png"}])
    nested = dataset_path / "images" / "nested"
    nested.mkdir(parents = True)
    (nested / "pic.png").write_bytes(b"png-bytes")

    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    from core.data_recipe.export import download_filename

    assert (
        download_filename(artifact_path = str(dataset_path), export_format = "jsonl", stem = "run")
        == "run.jsonl.zip"
    )

    file_path, media_type, download_name = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "run",
    )
    try:
        assert media_type == "application/zip"
        assert download_name == "run.jsonl.zip"
        with zipfile.ZipFile(file_path) as archive:
            names = sorted(archive.namelist())
            rows = archive.read("run.jsonl").decode("utf-8")
            assert archive.read("images/nested/pic.png") == b"png-bytes"
    finally:
        file_path.unlink(missing_ok = True)
    assert names == ["images/nested/pic.png", "run.jsonl"]
    assert json.loads(rows.strip()) == {"image": "images/nested/pic.png"}


def test_jsonl_export_stays_a_plain_file_without_images(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "recipe-datasets" / "job-text-only"
    _write_parquet_rows(dataset_path / "parquet-files", [{"text": "hello"}])
    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    from core.data_recipe.export import download_filename

    assert (
        download_filename(artifact_path = str(dataset_path), export_format = "jsonl", stem = "run")
        == "run.jsonl"
    )
    file_path, media_type, download_name = build_dataset_download(
        artifact_path = str(dataset_path),
        export_format = "jsonl",
        filename_stem = "run",
    )
    try:
        assert (media_type, download_name) == ("application/x-ndjson", "run.jsonl")
    finally:
        file_path.unlink(missing_ok = True)


def test_pandas_fallback_exports_a_schema_duckdb_will_not_take(tmp_path: Path):
    """DuckDB refuses a dataset carrying its own `filename`, since read_parquet wants that name."""
    pytest.importorskip("duckdb")
    from core.data_recipe.export import (
        _stream_jsonl_from_parquet_with_duckdb,
        _write_jsonl_from_parquet,
    )

    parquet_dir = tmp_path / "parquet-files"
    _write_parquet_rows(parquet_dir, [{"filename": "a.png", "text": "x"}])

    assert not _stream_jsonl_from_parquet_with_duckdb(
        parquet_dir = parquet_dir,
        destination = tmp_path / "unused.jsonl",
    )
    destination = tmp_path / "out.jsonl"
    _write_jsonl_from_parquet(parquet_dir, destination)
    assert json.loads(destination.read_text().strip()) == {"filename": "a.png", "text": "x"}


def test_minting_refuses_a_run_whose_shards_are_gone(tmp_path: Path, monkeypatch):
    """A historical run's artifact path comes from the client, and nothing confirmed it still held
    anything: the link minted fine and the browser then failed invisibly against it."""
    dataset_path = tmp_path / "recipe-datasets" / "job-swept"
    (dataset_path / "parquet-files").mkdir(parents = True)
    monkeypatch.setattr(
        "core.data_recipe.export._resolve_recipe_artifact_path",
        lambda artifact_path: dataset_path,
    )

    from core.data_recipe.export import download_filename

    with pytest.raises(RecipeDatasetExportError, match = "parquet"):
        download_filename(artifact_path = str(dataset_path), export_format = "jsonl", stem = "swept")


def test_minting_is_refused_for_a_keyless_caller(monkeypatch, tmp_path: Path):
    """The capability outlives the setting that admitted the caller, so a keyless request may not
    mint one. Same refusal the signed gallery-video links make."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    jobs_route = pytest.importorskip("routes.data_recipe.jobs")
    app, token = _download_app(monkeypatch, tmp_path, jobs_route)

    app.dependency_overrides[jobs_route.request_admitted_without_credential] = lambda: True
    refused = TestClient(app).get(
        "/api/data-recipe/jobs/job-1/download-url",
        headers = {"Authorization": f"Bearer {token}"},
    )
    assert refused.status_code == 403
    app.dependency_overrides.clear()


def test_download_link_outlasts_the_native_save_dialog():
    """The chooser opens before the request, so a link expiring while it sits open 401s after the
    destination was picked."""
    jobs_route = pytest.importorskip("routes.data_recipe.jobs")

    assert jobs_route._DOWNLOAD_LINK_TTL >= 15 * 60


def test_pyarrow_fallback_streams_a_merged_shard_by_row_group(tmp_path: Path):
    """merge_batches collapses a whole run into one file, so a shard is not a safe unit to read."""
    pytest.importorskip("pandas")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    import pandas as pd

    from core.data_recipe.export import _JSONL_EXPORT_BATCH_ROWS, _write_jsonl_with_pyarrow

    parquet_dir = tmp_path / "parquet-files"
    parquet_dir.mkdir(parents = True)
    rows = _JSONL_EXPORT_BATCH_ROWS * 2 + 5
    pd.DataFrame({"i": range(rows)}).to_parquet(
        parquet_dir / "batch_00000.parquet",
        index = False,
        row_group_size = 1_000,
    )
    assert pyarrow_parquet.ParquetFile(parquet_dir / "batch_00000.parquet").num_row_groups > 1

    destination = tmp_path / "out.jsonl"
    assert _write_jsonl_with_pyarrow(parquet_dir, destination)
    exported = [json.loads(line)["i"] for line in destination.read_text().splitlines()]
    assert exported == list(range(rows))
