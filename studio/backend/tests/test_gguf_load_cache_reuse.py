# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for cached GGUF reuse and load/download exclusion.

No GPU, network, or subprocesses are required.
"""

from __future__ import annotations

import asyncio
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub optional dependencies before importing the modules under test.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
        "HTTPStatusError",
    ):
        setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Request = type("Request", (), {})

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    _httpx_stub.Timeout = _FakeTimeout
    _httpx_stub.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    sys.modules.setdefault("httpx", _httpx_stub)


from huggingface_hub import constants as hf_constants

from core.inference.llama_cpp import (
    LlamaCppBackend,
    cached_gguf_for_load,
    gguf_load_in_flight,
    hf_gguf_load_in_flight,
)


REPO = "unsloth/gemma-test-GGUF"
VARIANT = "UD-Q4_K_XL"
MAIN = f"gemma-test-{VARIANT}.gguf"


def _build_cache(
    root: Path,
    repo_id: str,
    files: dict[str, int],
    *,
    snapshot_sha: str = "a" * 40,
) -> Path:
    """Create ``$root/models--<repo>/snapshots/<sha>/<rel>`` for each entry."""
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    snap = repo_dir / "snapshots" / snapshot_sha
    snap.mkdir(parents = True, exist_ok = True)
    for rel, size in files.items():
        full = snap / rel
        full.parent.mkdir(parents = True, exist_ok = True)
        full.write_bytes(b"\0" * size)
    return snap


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    return tmp_path


def _fail_download(*_args, **_kwargs):
    raise AssertionError("must reuse the cached GGUF instead of downloading")


def _fail_get_paths_info(*_args, **_kwargs):
    raise AssertionError("cached reuse must return before the sizing preflight")


class TestLoadReusesCachedCopy:
    def test_online_reuse_after_revision_bump(self, hf_cache):
        """A new repo revision does not replace a complete cached model."""
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", _fail_get_paths_info),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / MAIN)

    def test_online_reuse_when_reupload_renamed_the_file(self, hf_cache):
        """A renamed variant still reuses its cached file."""
        backend = LlamaCppBackend()
        old_name = f"gemma-test-old-{VARIANT}.gguf"
        snap = _build_cache(hf_cache, REPO, {old_name: 4})

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", _fail_get_paths_info),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / old_name)

    def test_downloads_when_nothing_cached(self, hf_cache):
        backend = LlamaCppBackend()
        downloaded: list[str] = []

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = p, size = 1) for p in paths if p is not None]

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert downloaded == [MAIN]
        assert out == f"/fake/{REPO}/{MAIN}"

    def test_force_redownloads_despite_cache(self, hf_cache):
        """A forced download ignores a complete cached copy."""
        backend = LlamaCppBackend()
        _build_cache(hf_cache, REPO, {MAIN: 4})
        downloaded: list[str] = []

        def fake_download(
            repo_id,
            filename,
            token = None,
            **kwargs,
        ):
            assert kwargs.get("force_download") is True
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = p, size = 1) for p in paths if p is not None]

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT, force = True)

        assert downloaded == [MAIN]
        assert out == f"/fake/{REPO}/{MAIN}"

    def test_split_reused_only_when_colocated(self, hf_cache):
        backend = LlamaCppBackend()
        shard1 = f"gemma-test-{VARIANT}-00001-of-00002.gguf"
        shard2 = f"gemma-test-{VARIANT}-00002-of-00002.gguf"
        snap = _build_cache(hf_cache, REPO, {shard1: 4, shard2: 4})

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [shard1, shard2]),
            patch("huggingface_hub.get_paths_info", _fail_get_paths_info),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / shard1)

    def test_partial_split_set_downloads(self, hf_cache):
        """A partial split set is not reused."""
        backend = LlamaCppBackend()
        shard1 = f"gemma-test-{VARIANT}-00001-of-00002.gguf"
        shard2 = f"gemma-test-{VARIANT}-00002-of-00002.gguf"
        _build_cache(hf_cache, REPO, {shard1: 4})
        downloaded: list[str] = []

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = p, size = 4) for p in paths if p is not None]

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [shard1, shard2]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert downloaded == [shard1, shard2]
        assert out == f"/fake/{REPO}/{shard1}"

    def test_reuse_prefers_newest_snapshot_after_update(self, hf_cache):
        """Loads prefer the newest complete snapshot."""
        import os

        backend = LlamaCppBackend()
        old_snap = _build_cache(hf_cache, REPO, {MAIN: 4}, snapshot_sha = "a" * 40)
        new_snap = _build_cache(hf_cache, REPO, {MAIN: 6}, snapshot_sha = "b" * 40)
        os.utime(old_snap, (1_000_000, 1_000_000))
        os.utime(new_snap, (2_000_000, 2_000_000))

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", _fail_get_paths_info),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(new_snap / MAIN)

    def test_companion_prefers_main_snapshot_sibling(self, hf_cache):
        """A cached mmproj is reused from the main model's snapshot."""
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4, "mmproj-F16.gguf": 2})

        def _fail_list(*_args, **_kwargs):
            raise AssertionError("snapshot sibling must resolve without a repo listing")

        with patch("huggingface_hub.list_repo_files", _fail_list):
            out = backend._download_mmproj(hf_repo = REPO, near_path = str(snap / MAIN))

        assert out == str(snap / "mmproj-F16.gguf")


class TestCachedGgufForLoadProbe:
    def test_complete_copy_found(self, hf_cache):
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})
        assert cached_gguf_for_load(REPO, VARIANT) == str(snap / MAIN)

    def test_absent_copy_is_none(self, hf_cache):
        assert cached_gguf_for_load(REPO, VARIANT) is None

    def test_partial_split_is_none(self, hf_cache):
        shard1 = f"gemma-test-{VARIANT}-00001-of-00002.gguf"
        _build_cache(hf_cache, REPO, {shard1: 4})
        assert cached_gguf_for_load(REPO, VARIANT) is None


class TestLoadHubDownloadExclusion:
    def test_in_flight_marker_counts_and_normalizes_case(self):
        assert not hf_gguf_load_in_flight(REPO)
        with gguf_load_in_flight(REPO):
            assert hf_gguf_load_in_flight(REPO.upper())
            with gguf_load_in_flight(REPO.lower()):
                assert hf_gguf_load_in_flight(REPO)
            assert hf_gguf_load_in_flight(REPO)
        assert not hf_gguf_load_in_flight(REPO)

    def test_marker_noops_for_local_loads(self):
        with gguf_load_in_flight(None):
            assert not hf_gguf_load_in_flight("")

    def test_marker_cleared_on_exception(self):
        with pytest.raises(RuntimeError):
            with gguf_load_in_flight(REPO):
                raise RuntimeError("boom")
        assert not hf_gguf_load_in_flight(REPO)

    def test_hub_download_refused_while_load_in_flight(self):
        from fastapi import HTTPException

        from hub.schemas.downloads import DownloadModelRequest
        from hub.services.models import downloads as dl

        body = DownloadModelRequest(repo_id = REPO, gguf_variant = VARIANT)
        with (
            patch.object(dl, "resolve_cached_repo_id_case", lambda repo_id, repo_type: repo_id),
            gguf_load_in_flight(REPO),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(dl.download_model_response(body))

        assert exc_info.value.status_code == 409
        assert "load" in exc_info.value.detail.lower()
