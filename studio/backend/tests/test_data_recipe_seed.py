# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
import tempfile
import types
from pathlib import Path

import pytest
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel


def test_seed_inspect_load_kwargs_disables_remote_code_execution():
    seed_route = (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text()

    assert '"trust_remote_code": False' in seed_route


def _load_seed_module(monkeypatch: pytest.MonkeyPatch):
    """Load the data-recipe seed route with lightweight stubs.

    The production module depends on several backend packages that are not
    needed for this regression test. Loading the file directly keeps the test
    focused on the quota logic we care about.
    """
    root = Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"

    core_pkg = types.ModuleType("core")
    core_data_recipe_pkg = types.ModuleType("core.data_recipe")
    jsonable = types.ModuleType("core.data_recipe.jsonable")
    jsonable.to_preview_jsonable = lambda value: value

    utils_pkg = types.ModuleType("utils")
    utils_paths = types.ModuleType("utils.paths")

    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    utils_paths.ensure_dir = ensure_dir
    utils_paths.seed_uploads_root = lambda: Path(tempfile.gettempdir()) / "seed-uploads"
    utils_paths.unstructured_uploads_root = (
        lambda: Path(tempfile.gettempdir()) / "unstructured-uploads"
    )

    models_pkg = types.ModuleType("models")
    models_data_recipe = types.ModuleType("models.data_recipe")

    class _SeedInspectRequest(BaseModel):
        pass

    class _SeedInspectResponse(BaseModel):
        dataset_name: str | None = None
        resolved_path: str | None = None
        columns: list[str] = []
        preview_rows: list[dict] = []
        split: str | None = None
        subset: str | None = None
        resolved_paths: list[str] | None = None

    class _SeedInspectUploadRequest(BaseModel):
        pass

    class _UnstructuredFileUploadResponse(BaseModel):
        file_id: str
        filename: str
        size_bytes: int
        status: str
        error: str | None = None

    models_data_recipe.SeedInspectRequest = _SeedInspectRequest
    models_data_recipe.SeedInspectResponse = _SeedInspectResponse
    models_data_recipe.SeedInspectUploadRequest = _SeedInspectUploadRequest
    models_data_recipe.UnstructuredFileUploadResponse = _UnstructuredFileUploadResponse

    monkeypatch.setitem(sys.modules, "core", core_pkg)
    monkeypatch.setitem(sys.modules, "core.data_recipe", core_data_recipe_pkg)
    monkeypatch.setitem(sys.modules, "core.data_recipe.jsonable", jsonable)
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.paths", utils_paths)
    monkeypatch.setitem(sys.modules, "models", models_pkg)
    monkeypatch.setitem(sys.modules, "models.data_recipe", models_data_recipe)

    spec = importlib.util.spec_from_file_location("seed_repro", root)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unstructured_multi_file_upload_quota_is_server_side(monkeypatch):
    seed = _load_seed_module(monkeypatch)

    with tempfile.TemporaryDirectory() as td:
        seed.UNSTRUCTURED_UPLOAD_ROOT = Path(td) / "uploads"
        seed.MAX_FILE_SIZE = 10
        seed.MAX_TOTAL_SIZE = 15
        seed._extract_text_from_file = lambda file_path, ext: "ok"

        async def do_upload(name: str, payload: bytes, existing_ids: str):
            upload = UploadFile(filename=name, file=io.BytesIO(payload))
            return await seed.upload_unstructured_file(
                file=upload,
                block_id="block1",
                existing_file_ids=existing_ids,
            )

        first = asyncio.run(do_upload("a.txt", b"A" * 9, ""))
        assert first.status == "ok"

        with pytest.raises(HTTPException, match="Total upload limit"):
            asyncio.run(do_upload("b.txt", b"B" * 9, ""))

        raw_total = sum(
            p.stat().st_size
            for p in (seed.UNSTRUCTURED_UPLOAD_ROOT / "block1").iterdir()
            if p.is_file()
            and not p.name.endswith(".extracted.txt")
            and not p.name.endswith(".meta.json")
        )
        assert raw_total == 9
        assert seed._get_block_total_size(seed.UNSTRUCTURED_UPLOAD_ROOT / "block1") == 9
