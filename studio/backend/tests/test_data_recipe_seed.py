# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
from pathlib import Path

import pytest


def _seed_route_source() -> str:
    return (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text(encoding = "utf-8")


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


_TEST_UPLOAD_UID = "0f" * 16


def test_remove_unstructured_block_deletes_directory(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _run_upload(seed_route, "notes.txt", b"hello", block_id = _TEST_UPLOAD_UID)
    assert _block_files(seed_route, _TEST_UPLOAD_UID) != []

    result = asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert result == {"status": "ok", "deleted": True}
    assert not (seed_route.UNSTRUCTURED_UPLOAD_ROOT / _TEST_UPLOAD_UID).exists()


def test_remove_unstructured_block_missing_directory_is_ok(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    result = asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert result == {"status": "ok", "deleted": False}


def test_remove_unstructured_block_rejects_unsafe_ids(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block("../escape"))

    assert exc.value.status_code == 400


def test_remove_unstructured_block_rejects_legacy_node_ids(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _run_upload(seed_route, "notes.txt", b"hello", block_id = "n1")
    assert _block_files(seed_route, "n1") != []

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block("n1"))

    assert exc.value.status_code == 400
    assert _block_files(seed_route, "n1") != []


def test_remove_unstructured_block_rejects_symlink_escape(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("keep me")
    root = seed_route.UNSTRUCTURED_UPLOAD_ROOT
    root.mkdir(parents = True)
    (root / _TEST_UPLOAD_UID).symlink_to(outside)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert exc.value.status_code == 400
    assert (outside / "victim.txt").exists()


def test_remove_unstructured_block_fails_if_directory_remains(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    root = seed_route.UNSTRUCTURED_UPLOAD_ROOT
    block_dir = root / _TEST_UPLOAD_UID
    block_dir.mkdir(parents = True)
    (block_dir / "victim.txt").write_text("keep me")

    calls = []

    def noop_rmtree(path, *args, **kwargs):
        calls.append((path, args, kwargs))

    monkeypatch.setattr(seed_route.shutil, "rmtree", noop_rmtree)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert calls
    assert exc.value.status_code == 500
    assert block_dir.exists()


def test_total_upload_quota_is_scoped_per_block(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES", 10)

    first = _run_upload(seed_route, "a.txt", b"123456789")
    assert first.status == "ok"

    with pytest.raises(seed_route.HTTPException) as exc:
        _run_upload(seed_route, "b.txt", b"123")
    assert exc.value.status_code == 413

    # Another block starts with its own untouched budget.
    other = _run_upload(seed_route, "c.txt", b"123", block_id = "other")
    assert other.status == "ok"


class _BlockPlugin:
    """Meta path finder making the optional seed plugin look uninstalled."""

    def __init__(self, name: str = "data_designer_unstructured_seed"):
        self.name = name
        self.attempts = 0

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname == self.name or fullname.startswith(self.name + "."):
            self.attempts += 1
            raise ModuleNotFoundError(f"No module named {fullname!r}", name = fullname)
        return None


def _without_plugin(monkeypatch, seed_route):
    import sys

    blocker = _BlockPlugin()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for name in [m for m in sys.modules if m.split(".")[0] == blocker.name]:
        monkeypatch.delitem(sys.modules, name)
    seed_route._CHUNKING = None
    return blocker


def test_unstructured_preview_reports_unavailable_without_the_plugin(monkeypatch, tmp_path):
    """Deferring the plugin import must not change what a missing plugin looks like."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _without_plugin(monkeypatch, seed_route)

    assert seed_route._chunking() is None

    with pytest.raises(seed_route.HTTPException) as exc:
        seed_route._read_preview_rows_from_unstructured_file(
            path = tmp_path / "a.txt", preview_size = 5, chunk_size = None, chunk_overlap = None
        )
    assert exc.value.status_code == 500
    assert "Unstructured seed support not available" in exc.value.detail

    with pytest.raises(seed_route.HTTPException) as exc:
        seed_route._read_preview_rows_from_multi_files(
            block_id = "block",
            file_ids = ["a"],
            file_names = ["a.txt"],
            preview_size = 5,
            chunk_size = None,
            chunk_overlap = None,
        )
    assert exc.value.status_code == 500
    assert "Unstructured seed support not available" in exc.value.detail


def test_missing_plugin_is_probed_once(monkeypatch, tmp_path):
    """A failed probe is remembered, so previews do not retry the import every time."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    blocker = _without_plugin(monkeypatch, seed_route)

    assert seed_route._chunking() is None
    assert seed_route._chunking() is None
    assert blocker.attempts == 1


def test_text_extraction_falls_back_to_raw_without_the_plugin(monkeypatch, tmp_path):
    """normalize_unstructured_text lives in the plugin; without it raw text stands."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _without_plugin(monkeypatch, seed_route)
    source = tmp_path / "notes.txt"
    source.write_text("a\n\n\n\nb", encoding = "utf-8")

    # The plugin is what collapses the run of blank lines.
    assert seed_route._extract_text_from_file(source, ".txt") == "a\n\n\n\nb"


def test_plugin_resolution_survives_a_reload_and_normalizes(monkeypatch, tmp_path):
    """With the plugin installed the same call sites still go through it."""
    pytest.importorskip("data_designer_unstructured_seed")
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    seed_route._CHUNKING = None

    chunking = seed_route._chunking()
    assert chunking is not None
    assert chunking.resolve_chunking(0, 0)[0] == 1
    source = tmp_path / "notes.txt"
    source.write_text("a\n\n\n\nb", encoding = "utf-8")
    assert seed_route._extract_text_from_file(source, ".txt") == "a\n\nb"
