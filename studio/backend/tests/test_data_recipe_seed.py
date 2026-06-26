# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
from pathlib import Path

import pytest


def _seed_route_source() -> str:
    return (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text()


def test_seed_inspect_load_kwargs_disables_remote_code_execution():
    assert '"trust_remote_code": False' in _seed_route_source()


class _FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


def _load_seed_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    pytest.importorskip("structlog")

    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    route_path = backend_root / "routes" / "data_recipe" / "seed.py"
    spec = importlib.util.spec_from_file_location("seed_under_test", route_path)
    assert spec is not None and spec.loader is not None
    seed_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_route)
    seed_route.UNSTRUCTURED_UPLOAD_ROOT = tmp_path / "unstructured-uploads"
    return seed_route


def _run_upload(
    seed_route,
    filename: str,
    content: bytes,
    block_id: str = "block",
):
    return asyncio.run(
        seed_route.upload_unstructured_file(_FakeUpload(filename, content), block_id)
    )


def _block_files(seed_route, block_id: str = "block") -> list[str]:
    block_dir = seed_route.UNSTRUCTURED_UPLOAD_ROOT / block_id
    if not block_dir.exists():
        return []
    return sorted(path.name for path in block_dir.iterdir())


def _raise(exc: BaseException):
    def raise_exc(*args, **kwargs):
        raise exc

    return raise_exc


@pytest.mark.parametrize(
    ("filename", "package"),
    [
        ("paper.pdf", "pymupdf4llm"),
        ("notes.docx", "mammoth"),
    ],
)
def test_unstructured_upload_names_missing_extractor_dependency(
    monkeypatch, tmp_path, filename, package
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(
        seed_route,
        "_extract_text_from_file",
        _raise(ModuleNotFoundError(f"No module named {package!r}", name = package)),
    )

    result = _run_upload(seed_route, filename, b"%PDF-1.7")

    assert result.status == "error"
    assert (
        result.error
        == f"Cannot read {Path(filename).suffix} files: the '{package}' package is not installed."
    )
    assert _block_files(seed_route) == []


def test_unstructured_upload_keeps_txt_path_working(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    result = _run_upload(seed_route, "notes.txt", b"hello")

    assert result.status == "ok"
    assert result.error is None
    assert any(name.endswith(".txt") for name in _block_files(seed_route))
    assert any(name.endswith(".extracted.txt") for name in _block_files(seed_route))


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("cannot import internal symbol"),
        ModuleNotFoundError(
            "No module named 'missing_transitive_pkg'",
            name = "missing_transitive_pkg",
        ),
    ],
)
def test_unstructured_upload_import_errors_stay_generic(monkeypatch, tmp_path, exc):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "_extract_text_from_file", _raise(exc))
    result = _run_upload(seed_route, "paper.pdf", b"%PDF-1.7")

    assert result.status == "error"
    assert result.error == "Text extraction failed."
    assert _block_files(seed_route) == []
