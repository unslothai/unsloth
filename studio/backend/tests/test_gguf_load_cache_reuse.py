# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for cached GGUF reuse and load/download exclusion.

No GPU, network, or subprocesses are required.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import threading
import types as _types
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub optional dependencies before importing the modules under test.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
# routes/inference.py binds structlog.get_logger at import time, and setdefault
# keeps a bare stub an earlier test left behind: repair it rather than rely on order.
_structlog_stub.get_logger = lambda *_args, **_kwargs: logging.getLogger("structlog_stub")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

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
    GgufLoadIntent,
    LlamaCppBackend,
    cached_gguf_for_load,
    gguf_load_in_flight,
    hf_gguf_load_in_flight,
)
from core.inference.openai_auto_download import _DISK_RESERVE_BYTES


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
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: _types.SimpleNamespace(hub_cache = tmp_path),
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    return tmp_path


def _fail_download(*_args, **_kwargs):
    raise AssertionError("must reuse the cached GGUF instead of downloading")


def _fail_get_paths_info(*_args, **_kwargs):
    raise AssertionError("cached reuse must return before the sizing preflight")


GIB = 1024**3
_LOW_DISK_SIZES = {
    "gemma-test-UD-IQ1_S.gguf": GIB,
    "gemma-test-Q4_K_M.gguf": 2 * GIB,
    "gemma-test-Q6_K.gguf": 6 * GIB,
    "gemma-test-Q8_0.gguf": 8 * GIB,
}


@contextmanager
def _low_disk_hub(
    free: int,
    *,
    snap: Path | None = None,
    served: str | None = None,
    sizes: dict[str, int] | None = None,
):
    sizes = sizes or _LOW_DISK_SIZES
    downloaded: list[str] = []

    def fake_get_paths_info(
        _repo,
        paths,
        *,
        revision = None,
        token = None,
    ):
        on_disk = snap is not None and revision == snap.name
        return [
            _types.SimpleNamespace(path = path, size = 4 if on_disk else sizes[path]) for path in paths
        ]

    def fake_download(_repo, filename, *_args, **_kwargs):
        downloaded.append(filename)
        return served or f"/fake/{REPO}/{filename}"

    usage = _types.SimpleNamespace(total = 100 * GIB, used = 100 * GIB - free, free = free)
    with (
        patch("huggingface_hub.list_repo_files", lambda *_a, **_k: list(sizes)),
        patch("huggingface_hub.get_paths_info", fake_get_paths_info),
        patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
        patch("shutil.disk_usage", lambda *_a, **_k: usage),
        patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
    ):
        yield downloaded


def _load_route_module(name: str, relative_path: str):
    """Import a route module under a private name so patches can't leak."""
    spec = importlib.util.spec_from_file_location(name, Path(_BACKEND_DIR) / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_route_load(route, request):
    fastapi_request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    return asyncio.run(
        route._load_model_impl(request, fastapi_request, current_subject = "test-user")
    )


@contextmanager
def _reuse_route(route, backend, response):
    with (
        patch.object(
            route,
            "_resolve_model_identifier_for_request",
            return_value = (REPO, REPO, False),
        ),
        patch.object(route, "resolve_effective_chat_template_override", return_value = None),
        patch.object(route, "_gguf_load_response", return_value = response) as response_mock,
        patch.object(route, "get_llama_cpp_backend", return_value = backend),
        patch.object(
            route,
            "get_inference_backend",
            return_value = SimpleNamespace(active_model_name = None),
        ),
    ):
        yield response_mock


async def _inline_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


async def _no_gguf_gpu_ids(*_args, **_kwargs):
    # Mirrors the resolver's no-gpu_ids early return: no ids, not Vulkan ordinals.
    return None, False


class TestLoadReusesCachedCopy:
    def test_download_uses_selected_cache_for_lookup_preflight_and_write(
        self, tmp_path, monkeypatch
    ):
        backend = LlamaCppBackend()
        selected = tmp_path / "selected" / "hub"
        startup = tmp_path / "startup" / "hub"
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(startup))
        monkeypatch.setattr(
            "utils.hf_cache_settings.get_hf_cache_paths",
            lambda: _types.SimpleNamespace(hub_cache = selected),
        )
        seen = {"lookups": [], "disk": [], "downloads": []}

        def cached_lookup(
            repo_id,
            filename,
            *,
            cache_dir = None,
            **_kwargs,
        ):
            seen["lookups"].append((repo_id, filename, cache_dir))
            return None

        def disk_usage(path):
            seen["disk"].append(str(path))
            return _types.SimpleNamespace(free = 1024)

        def download(repo_id, filename, _token, **kwargs):
            seen["downloads"].append((repo_id, filename, kwargs.get("cache_dir")))
            return str(selected / filename)

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch(
                "huggingface_hub.get_paths_info",
                lambda _repo, paths, **_kwargs: [
                    _types.SimpleNamespace(path = path, size = 4) for path in paths
                ],
            ),
            patch("huggingface_hub.try_to_load_from_cache", cached_lookup),
            patch("core.inference.llama_cpp.shutil.disk_usage", disk_usage),
            patch(
                "core.inference.llama_cpp.hf_hub_download_with_xet_fallback",
                download,
            ),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(selected / MAIN)
        assert seen == {
            "lookups": [(REPO, MAIN, str(selected))],
            "disk": [str(selected)],
            "downloads": [(REPO, MAIN, str(selected))],
        }

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

    def test_reuse_size_check_uses_cached_snapshot_revision(self, hf_cache):
        """Current-revision size changes do not invalidate an older complete copy."""
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})
        revisions: list[str | None] = []

        def fake_get_paths_info(
            _repo,
            paths,
            *,
            revision = None,
            token = None,
        ):
            revisions.append(revision)
            size = 4 if revision == snap.name else 8
            return [_types.SimpleNamespace(path = path, size = size) for path in paths]

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / MAIN)
        assert revisions == [snap.name]

    def test_reuse_when_cached_revision_vanished_from_hub(self, hf_cache):
        """The Hub answers an unknown revision with an empty result, not an error."""
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", lambda *_a, **_k: []),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / MAIN)

    def test_truncated_cached_file_is_not_reused(self, hf_cache):
        backend = LlamaCppBackend()
        _build_cache(hf_cache, REPO, {MAIN: 4})
        downloaded: list[str] = []

        def fake_get_paths_info(
            _repo,
            paths,
            *,
            revision = None,
            token = None,
        ):
            return [_types.SimpleNamespace(path = path, size = 8) for path in paths]

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert downloaded == [MAIN]
        assert out == f"/fake/{REPO}/{MAIN}"

    def test_truncated_cached_split_shard_is_not_reused(self, hf_cache):
        backend = LlamaCppBackend()
        shard1 = f"gemma-test-{VARIANT}-00001-of-00002.gguf"
        shard2 = f"gemma-test-{VARIANT}-00002-of-00002.gguf"
        _build_cache(hf_cache, REPO, {shard1: 8, shard2: 4})
        downloaded: list[str] = []

        def fake_get_paths_info(
            _repo,
            paths,
            *,
            revision = None,
            token = None,
        ):
            return [_types.SimpleNamespace(path = path, size = 8) for path in paths]

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [shard1, shard2]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert downloaded == [shard1, shard2]
        assert out == f"/fake/{REPO}/{shard1}"

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

    def test_low_disk_fallback_reuses_cached_copy(self, hf_cache):
        backend = LlamaCppBackend()
        fallback = "gemma-test-Q2_K.gguf"
        snap = _build_cache(hf_cache, REPO, {fallback: 4})

        def fake_get_paths_info(
            _repo,
            paths,
            *,
            revision = None,
            token = None,
        ):
            size = 4 if revision == snap.name else 100
            return [_types.SimpleNamespace(path = path, size = size) for path in paths]

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: [MAIN]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("shutil.disk_usage", lambda *_a, **_k: _types.SimpleNamespace(free = 10)),
            patch.object(
                backend,
                "_find_fitting_variant",
                lambda *_a, **_k: (fallback, 4, []),
            ),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / fallback)

    def test_low_disk_fallback_picks_largest_variant_leaving_reserve(self, hf_cache):
        backend = LlamaCppBackend()
        with _low_disk_hub(_DISK_RESERVE_BYTES + 5 * GIB // 2) as downloaded:
            out = backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == ["gemma-test-Q4_K_M.gguf"]
        assert out == f"/fake/{REPO}/gemma-test-Q4_K_M.gguf"
        variant, notice = backend._pending_variant_fallback
        assert variant == "Q4_K_M"
        assert notice == (
            "Not enough disk space to download Q8_0 (8.6 GB needed, 8.1 GB free), "
            "so Q4_K_M (2.1 GB) was loaded instead."
        )

    def test_low_disk_fallback_keeps_smallest_when_none_leaves_reserve(self, hf_cache):
        backend = LlamaCppBackend()
        with _low_disk_hub(_DISK_RESERVE_BYTES + GIB // 2) as downloaded:
            backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == ["gemma-test-UD-IQ1_S.gguf"]
        assert backend._pending_variant_fallback[0] == "UD-IQ1_S"

    def test_low_disk_fallback_reuses_cached_variant_before_downloading(self, hf_cache):
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {"gemma-test-Q4_K_M.gguf": 4})
        with _low_disk_hub(_DISK_RESERVE_BYTES + GIB // 2, snap = snap) as downloaded:
            out = backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == []
        assert out == str(snap / "gemma-test-Q4_K_M.gguf")

    def test_low_disk_fallback_prefers_the_largest_cached_variant_under_the_reserve(self, hf_cache):
        backend = LlamaCppBackend()
        snap = _build_cache(
            hf_cache, REPO, {"gemma-test-Q4_K_M.gguf": 4, "gemma-test-Q6_K.gguf": 4}
        )
        with _low_disk_hub(GIB // 2, snap = snap) as downloaded:
            out = backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == []
        assert out == str(snap / "gemma-test-Q6_K.gguf")

    def test_low_disk_fallback_never_picks_a_larger_cached_variant(self, hf_cache):
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {"gemma-test-Q8_0.gguf": 4})
        with _low_disk_hub(_DISK_RESERVE_BYTES + GIB // 2, snap = snap) as downloaded:
            backend._download_gguf(hf_repo = REPO, hf_variant = "Q6_K")

        assert downloaded == ["gemma-test-UD-IQ1_S.gguf"]

    def test_low_disk_fallback_stays_on_the_requested_checkpoint(self, hf_cache):
        backend = LlamaCppBackend()
        sizes = {
            "gemma-test-Q4_K_M.gguf": 2 * GIB,
            "gemma-test-Q8_0.gguf": 16 * GIB,
            "distilled/gemma-test-distilled-Q6_K.gguf": 6 * GIB,
        }
        with _low_disk_hub(_DISK_RESERVE_BYTES + 13 * GIB // 2, sizes = sizes) as downloaded:
            backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == ["gemma-test-Q4_K_M.gguf"]
        assert backend._pending_variant_fallback[0] == "Q4_K_M"

    def test_low_disk_fallback_ignores_a_truncated_cached_variant(self, hf_cache):
        backend = LlamaCppBackend()
        _build_cache(hf_cache, REPO, {"gemma-test-Q6_K.gguf": 4})
        with _low_disk_hub(_DISK_RESERVE_BYTES + GIB // 2) as downloaded:
            backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

        assert downloaded == ["gemma-test-UD-IQ1_S.gguf"]

    def test_low_disk_without_any_fitting_variant_refuses(self, hf_cache):
        backend = LlamaCppBackend()
        with _low_disk_hub(GIB // 2), pytest.raises(RuntimeError, match = "any variant"):
            backend._download_gguf(hf_repo = REPO, hf_variant = "Q8_0")

    def test_a_fallback_and_a_memory_notice_are_both_reported(self):
        backend = LlamaCppBackend()
        backend._variant_fallback_warning = "Q4_K_M was loaded instead."
        backend._record_load_warning("The model does not fit in GPU memory.")

        assert backend.last_load_warning == (
            "Q4_K_M was loaded instead. The model does not fit in GPU memory."
        )

        # The arch-crash retry calls this again after the download; the teardown clears both.
        backend._begin_load_warnings()
        assert backend.last_load_warning == "Q4_K_M was loaded instead."

    def test_low_disk_fallback_load_records_the_served_variant(self, hf_cache, tmp_path):
        backend = LlamaCppBackend()
        served = tmp_path / "gemma-test-Q4_K_M.gguf"
        served.write_bytes(b"GGUF" + b"\0" * 4096)
        backend._find_llama_server_binary = lambda *_a, **_k: "/usr/bin/true"
        backend._remote_non_chat_gguf_refusal = lambda **_k: None
        backend._non_chat_gguf_refusal_for_path = lambda *_a, **_k: None
        backend._non_chat_gguf_refusal = lambda *_a, **_k: None
        backend._kill_process = lambda *_a, **_k: None
        backend._wait_for_health = lambda timeout = 600.0, interval = 0.5, cancelled = None: True

        def _start(cmd, env, **_kwargs):
            backend._process = Mock(pid = 424242, returncode = None)
            backend._process.poll.return_value = None
            backend._stdout_lines = ["build: 6543", "main: loading model"]
            return backend._process

        backend._start_llama_process = _start
        requested = GgufLoadIntent(
            model_identifier = REPO,
            hf_repo = REPO,
            hf_variant = "Q8_0",
            speculative_type = "none",
            n_ctx = 4096,
        )
        with _low_disk_hub(_DISK_RESERVE_BYTES + 5 * GIB // 2, served = str(served)):
            assert backend.load_model(requested) is True

        assert backend.hf_variant == "Q4_K_M"
        assert "Q4_K_M" in (backend.last_load_warning or "")
        assert not backend.matches_load_source(requested)
        assert backend.matches_load_source(replace(requested, hf_variant = "Q4_K_M"))
        # Replayed by the crash respawn, so it must name the quant that is running.
        assert backend.last_load_intent.hf_variant == "Q4_K_M"

        with _low_disk_hub(100 * GIB, served = str(served)):
            assert backend.load_model(requested) is True

        assert backend.hf_variant == "Q8_0"
        assert backend.last_load_warning is None

    def test_companion_prefers_main_snapshot_sibling(self, hf_cache):
        """A cached mmproj is reused from the main model's snapshot."""
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4, "mmproj-F16.gguf": 2})

        def _fail_list(*_args, **_kwargs):
            raise AssertionError("snapshot sibling must resolve without a repo listing")

        with patch("huggingface_hub.list_repo_files", _fail_list):
            out = backend._download_mmproj(hf_repo = REPO, near_path = str(snap / MAIN))

        assert out == str(snap / "mmproj-F16.gguf")

    def test_companion_finds_snapshot_through_hf_symlink(self, hf_cache):
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {})
        blobs = snap.parent.parent / "blobs"
        main_blob = blobs / "main"
        mmproj_blob = blobs / "mmproj"
        main_blob.write_bytes(b"main")
        mmproj_blob.write_bytes(b"mmproj")
        try:
            (snap / MAIN).symlink_to(main_blob)
            (snap / "mmproj-F16.gguf").symlink_to(mmproj_blob)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")

        with patch("huggingface_hub.list_repo_files", _fail_download):
            out = backend._download_mmproj(hf_repo = REPO, near_path = str(snap / MAIN))

        assert out == str(snap / "mmproj-F16.gguf")

    def test_companion_does_not_download_during_hub_job(self, hf_cache):
        backend = LlamaCppBackend()
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})
        registry = _types.SimpleNamespace(active_job_refs = lambda _repo: [object()])

        with (
            patch("huggingface_hub.list_repo_files", _fail_download),
            patch("hub.utils.download_registry.get_models_registry", lambda: registry),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_mmproj(hf_repo = REPO, near_path = str(snap / MAIN))

        assert out is None


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

    def test_partial_new_snapshot_does_not_hide_complete_split(self, hf_cache):
        import os

        shard1 = f"gemma-test-{VARIANT}-00001-of-00002.gguf"
        shard2 = f"gemma-test-{VARIANT}-00002-of-00002.gguf"
        old = _build_cache(
            hf_cache,
            REPO,
            {shard1: 4, shard2: 4},
            snapshot_sha = "a" * 40,
        )
        new = _build_cache(hf_cache, REPO, {shard1: 4}, snapshot_sha = "b" * 40)
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))

        assert cached_gguf_for_load(REPO, VARIANT) == str(old / shard1)

    def test_split_requires_every_declared_shard(self, hf_cache):
        shard1 = f"gemma-test-{VARIANT}-00001-of-00003.gguf"
        shard2 = f"gemma-test-{VARIANT}-00002-of-00003.gguf"
        _build_cache(hf_cache, REPO, {shard1: 4, shard2: 4})

        assert cached_gguf_for_load(REPO, VARIANT) is None

    def test_required_mmproj_must_share_main_snapshot(self, hf_cache):
        snap = _build_cache(hf_cache, REPO, {MAIN: 4})
        assert cached_gguf_for_load(REPO, VARIANT) == str(snap / MAIN)
        assert cached_gguf_for_load(REPO, VARIANT, require_mmproj = True) is None

        (snap / "mmproj-F16.gguf").write_bytes(b"mmproj")
        assert cached_gguf_for_load(REPO, VARIANT, require_mmproj = True) == str(snap / MAIN)

    def test_required_mmproj_scans_past_newer_main_only_snapshot(self, hf_cache):
        import os

        old = _build_cache(
            hf_cache,
            REPO,
            {MAIN: 4, "mmproj-F16.gguf": 2},
            snapshot_sha = "a" * 40,
        )
        new = _build_cache(hf_cache, REPO, {MAIN: 4}, snapshot_sha = "b" * 40)
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))

        assert cached_gguf_for_load(REPO, VARIANT, require_mmproj = True) == str(old / MAIN)


class TestLoadHubDownloadExclusion:
    def test_remote_intent_carries_the_verified_cache_hint(self):
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_verified_cache_hint",
            "routes/inference.py",
        )
        verified = (REPO, VARIANT, "/cached/model.gguf", ((MAIN, 123),))
        config = SimpleNamespace(
            identifier = REPO,
            gguf_hf_repo = REPO,
            gguf_variant = VARIANT,
            gguf_verified = verified,
            is_vision = False,
        )

        intent = route._resolve_gguf_load_intent(
            config,
            LoadRequest(model_path = REPO, gguf_variant = VARIANT),
            native_grant_backed = False,
            chat_template_override = None,
            extra_args = None,
            placement = SimpleNamespace(
                resolved_gpu_ids = None,
                gpu_ids_are_vulkan_ordinals = None,
            ),
            n_parallel = 1,
        )

        assert intent.verified_gguf == verified

    def test_resident_local_directory_intent_uses_variant_until_path_is_resolved(self):
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_local_variant_identity",
            "routes/inference.py",
        )
        model_identifier = "/models/local-quants"
        loaded_path = "/models/model.gguf"
        backend = SimpleNamespace(
            extra_args = None,
            gguf_path = loaded_path,
            layer_preserves_tensor_intent = False,
            last_load_intent = GgufLoadIntent(
                model_identifier = model_identifier,
                gguf_path = loaded_path,
                hf_variant = VARIANT,
            ),
        )

        with patch.object(route, "_mtp_draft_for_path", return_value = None):
            intent = route._active_gguf_intent(
                LoadRequest(model_path = model_identifier, gguf_variant = "Q8_0"),
                backend,
                model_identifier = model_identifier,
                chat_template_override = None,
                n_parallel = 1,
                native_grant_backed = False,
            )

        assert intent.gguf_path is None
        assert intent.hf_variant == "Q8_0"

    @pytest.mark.parametrize("reuse_case", ["unchanged", "new_roots", "completed_file"])
    def test_resident_gguf_reuse_precedes_model_metadata_resolution(self, reuse_case, tmp_path):
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_resident_fast_path_test",
            "routes/inference.py",
        )
        response = object()
        backend = SimpleNamespace(
            is_loaded = True,
            model_identifier = REPO,
            adopt_load_intent_if_matched = lambda _intent: True,
            _audio_probed = True,
            # The reuse fast path consults this before asserting CHAT ownership; a real
            # LlamaCppBackend exposes it as a property, so the double has to carry it too.
            holds_no_vram = False,
        )
        request = LoadRequest(model_path = REPO, gguf_variant = VARIANT)
        if reuse_case == "new_roots":
            request._gguf_companion_roots = ("weights-revision", "companion-revision")
        elif reuse_case == "completed_file":
            request._gguf_companion_roots = (str(tmp_path),)
            backend._openai_gguf_companion_roots = request._gguf_companion_roots
            backend._openai_gguf_companion_state = ()
            (tmp_path / "mmproj-F16.gguf").write_bytes(b"GGUF companion")

        class MetadataReached(BaseException):
            pass

        with (
            _reuse_route(route, backend, response),
            patch.object(
                route,
                "ModelConfig",
                SimpleNamespace(
                    from_identifier = lambda **_kwargs: (_ for _ in ()).throw(MetadataReached())
                ),
            ),
            patch.object(route, "_active_gguf_intent", return_value = object()),
        ):
            if reuse_case != "unchanged":
                with pytest.raises(MetadataReached):
                    _run_route_load(route, request)
                return
            result = _run_route_load(route, request)

        assert result is response

    def test_post_config_reuse_preserves_resolved_display_name(self):
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_resolved_display_name_test",
            "routes/inference.py",
        )
        response = object()
        intent = GgufLoadIntent(model_identifier = REPO, gguf_path = "/cached/model.gguf")
        backend = SimpleNamespace(
            is_loaded = True,
            model_identifier = REPO,
            gguf_path = None,
            matches_load_source = lambda _intent: True,
            adopt_load_intent_if_matched = lambda _intent: True,
            _audio_probed = True,
            # The reuse fast path consults this before asserting CHAT ownership; a real
            # LlamaCppBackend exposes it as a property, so the double has to carry it too.
            holds_no_vram = False,
        )
        config = SimpleNamespace(
            identifier = REPO,
            display_name = "Gemma Test (UD-Q4_K_XL)",
            is_gguf = True,
            is_lora = False,
            is_vision = False,
            gguf_hf_repo = None,
        )

        with (
            _reuse_route(route, backend, response) as response_mock,
            patch.object(
                route,
                "ModelConfig",
                SimpleNamespace(from_identifier = lambda **_kwargs: config),
            ),
            patch.object(route, "_resolve_inherited_extra_args", return_value = None),
            patch.object(
                route,
                "_prepare_load_placement",
                return_value = route._LoadPlacement(None, None, False, False),
            ),
            patch.object(route, "_resolve_gguf_load_intent", return_value = intent),
            patch.object(route, "_loaded_is_local_model", return_value = False),
        ):
            result = _run_route_load(route, LoadRequest(model_path = REPO))

        assert result is response
        assert response_mock.call_args.args[1] == "already_loaded"
        assert response_mock.call_args.kwargs["display_name"] == config.display_name

    def test_local_intent_does_not_retain_hf_token(self):
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_local_intent_token_test",
            "routes/inference.py",
        )
        request = LoadRequest(model_path = "/models/local.gguf", hf_token = "secret")
        common = dict(
            chat_template_override = None,
            extra_args = None,
            gpu_ids = None,
            n_parallel = 1,
        )

        local = route._gguf_request_intent(
            GgufLoadIntent(model_identifier = request.model_path, gguf_path = request.model_path),
            request,
            **common,
        )
        remote = route._gguf_request_intent(
            GgufLoadIntent(model_identifier = REPO, hf_repo = REPO),
            request,
            **common,
        )

        assert local.hf_token is None
        assert remote.hf_token == "secret"

    def test_runtime_response_projection_fails_loudly_on_backend_drift(self):
        route = _load_route_module(
            "inference_route_module_for_runtime_projection_test",
            "routes/inference.py",
        )
        incomplete_backend = SimpleNamespace(
            requested_spec_mode = "auto",
            is_diffusion = False,
            requested_parallel_slots = 1,
            effective_parallel_slots = 1,
        )

        with pytest.raises(AttributeError, match = "runtime response fields"):
            route._llama_runtime_fields(incomplete_backend)

    def test_real_backend_resolves_every_runtime_response_field(self):
        # The shipped class, not a fake: an unresolved field 500s every load and poll.
        from core.inference.llama_cpp import LlamaCppBackend
        from models.inference import _InferenceRuntimeFields

        route = _load_route_module(
            "inference_route_module_for_real_backend_projection_test",
            "routes/inference.py",
        )

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend.__init__()
        supplied = {
            "requires_trust_remote_code",
            "speculative_type",
            "requested_parallel_slots",
            "parallel_slots",
            "is_mlx",
            "mlx_kv_bits",
            "mlx_kv_bits_requested",
            "mlx_kv_quant_eligibility",
            "mlx_kv_quant_reason",
            "mlx_kv_quant_note",
            "chat_template_override_reason",
            # Constant True: llama.cpp allocates the window it reports.
            "context_length_enforced",
            # Read from requested_extra_args, which is what the load was invoked
            # with rather than the rewritten launch list.
            "requested_llama_extra_args",
        }
        unresolved = sorted(
            name
            for name in _InferenceRuntimeFields.model_fields
            if name not in supplied and not (hasattr(backend, name) or hasattr(backend, f"_{name}"))
        )
        assert unresolved == []

        fields = route._llama_runtime_fields(backend)
        assert fields["is_mlx"] is False
        assert fields["mlx_kv_bits_requested"] is None

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

    def test_chat_load_marker_is_repo_agnostic_and_nests(self):
        # The GPU arbiter needs to know a chat load exists before llama-server is spawned, for local paths and safetensors too, so this marker carries no repo key.
        from core.inference.llama_cpp import chat_load_active, chat_load_in_flight

        assert not chat_load_active()
        with chat_load_in_flight():
            assert chat_load_active()
            with chat_load_in_flight():
                assert chat_load_active()
            assert chat_load_active()
        assert not chat_load_active()

        with pytest.raises(RuntimeError):
            with chat_load_in_flight():
                raise RuntimeError("boom")
        assert not chat_load_active()

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

    def test_hub_download_rechecks_marker_before_claim(self):
        from fastapi import HTTPException

        from hub.schemas.downloads import DownloadModelRequest
        from hub.services.models import downloads as dl

        scope = None

        def mark_load(*_args, **_kwargs):
            nonlocal scope
            if scope is None:
                scope = gguf_load_in_flight(REPO)
                scope.__enter__()
            return frozenset()

        class _Registry:
            def claim(self, *_args, admission_check, **_kwargs):
                assert admission_check() is False
                return False, "admission_blocked"

            def current_generation(self, _key):
                return 0

        registry = _Registry()
        body = DownloadModelRequest(repo_id = REPO, gguf_variant = VARIANT)
        try:
            with (
                patch.object(dl, "resolve_cached_repo_id_case", lambda repo_id, repo_type: repo_id),
                patch.object(dl.gguf_variants, "gguf_variant_blob_hashes", mark_load),
                patch.object(dl, "_registry", registry),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    asyncio.run(dl.download_model_response(body))
        finally:
            if scope is not None:
                scope.__exit__(None, None, None)

        assert exc_info.value.status_code == 409

    def test_registry_admission_check_prevents_claim(self):
        from hub.utils.download_registry import DownloadRegistry, TRANSPORT_HTTP

        registry = DownloadRegistry()
        claimed, state = registry.claim(
            f"{REPO}::{VARIANT}",
            TRANSPORT_HTTP,
            repo_type = "model",
            repo_id = REPO,
            variant = VARIANT,
            admission_check = lambda: False,
        )

        assert claimed is False
        assert state == "admission_blocked"
        assert registry.active_jobs(REPO) == {}

    def test_same_variant_job_stays_visible_during_retry_handoff(self):
        from hub.utils.download_registry import DownloadRegistry, TRANSPORT_XET
        from core.inference.llama_cpp import _hub_download_blocks_gguf_load

        registry = DownloadRegistry()
        key = f"{REPO}::{VARIANT}"
        claimed, _ = registry.claim(
            key,
            TRANSPORT_XET,
            repo_type = "model",
            repo_id = REPO,
            variant = VARIANT,
        )
        assert claimed is True
        assert registry.has_active_variant(REPO, VARIANT.lower()) is True

        registry.release_active_slot(key)

        assert registry.active_jobs(REPO) == {}
        assert registry.active_job_refs(REPO)
        assert registry.has_active_variant(REPO, VARIANT) is True
        with (
            patch("hub.utils.download_registry.get_models_registry", lambda: registry),
            patch(
                "core.inference.llama_cpp.cached_gguf_for_load",
                side_effect = AssertionError("same-variant jobs must block before cache reuse"),
            ),
        ):
            assert _hub_download_blocks_gguf_load(REPO, VARIANT) is True

        registry.set_job(key, "complete")
        assert registry.has_active_variant(REPO, VARIANT) is False

    def test_other_variant_job_still_allows_complete_cached_load(self):
        from core.inference.llama_cpp import _hub_download_blocks_gguf_load
        from hub.utils.download_registry import DownloadRegistry, TRANSPORT_HTTP

        registry = DownloadRegistry()
        registry.claim(
            f"{REPO}::Q8_0",
            TRANSPORT_HTTP,
            repo_type = "model",
            repo_id = REPO,
            variant = "Q8_0",
        )
        with (
            patch("hub.utils.download_registry.get_models_registry", lambda: registry),
            patch(
                "core.inference.llama_cpp.cached_gguf_for_load",
                return_value = "/cached/model.gguf",
            ) as cached_probe,
        ):
            assert _hub_download_blocks_gguf_load(REPO, VARIANT) is False

        cached_probe.assert_called_once_with(
            REPO,
            VARIANT,
            require_mmproj = False,
            verify_sizes = True,
            hf_token = None,
        )

    def test_cancelled_request_keeps_marker_until_load_thread_finishes(self):
        from core.inference.llama_cpp import _with_gguf_load_marker

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        class FakeBackend:
            @_with_gguf_load_marker
            # The marker forwards the scoped cancel event, so a loader must accept it.
            def load_model(
                self,
                intent,
                load_cancel_event = None,
            ):
                started.set()
                release.wait(timeout = 2)
                finished.set()
                return True

        async def scenario():
            with patch(
                "core.inference.llama_cpp._hub_download_blocks_gguf_load",
                return_value = False,
            ):
                task = asyncio.create_task(
                    asyncio.to_thread(
                        FakeBackend().load_model,
                        GgufLoadIntent(model_identifier = REPO, hf_repo = REPO),
                    )
                )
                assert await asyncio.to_thread(started.wait, 1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert hf_gguf_load_in_flight(REPO)

                release.set()
                assert await asyncio.to_thread(finished.wait, 1)
                for _ in range(100):
                    if not hf_gguf_load_in_flight(REPO):
                        break
                    await asyncio.sleep(0.001)
                assert not hf_gguf_load_in_flight(REPO)

        asyncio.run(scenario())

    def test_load_marker_precedes_hub_guard_which_precedes_the_gpu_handoff(self):
        source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text(
            encoding = "utf-8"
        )
        # Anchor on the enclosing function, not a nearby `if config.is_gguf:`: the old anchor took the last such line before the load marker, which held
        # only while _resolve_inherited_extra_args sat above every one of them. The ordering is a property of _load_model_impl.
        marker = source.index("enter_context(gguf_load_in_flight")
        gguf_branch_start = source.rindex("async def _load_model_impl", 0, marker)
        gguf_branch = source[gguf_branch_start:]

        # One chain, in this order:
        # - _resolve_inherited_extra_args first: the inherited value (e.g. a carried --no-mmproj) shapes the guard's require_mmproj.
        # - the gguf_load_in_flight marker before the hub-download guard: that handshake keeps a load and the download manager off the same files.
        # - both before the CHAT handoff: the guard's 409 loads nothing, so checking it later destroyed a resident pipeline for a load that could never start.
        # - the resident unload last. Anchored on call forms so each assertion pins a call site, not a definition.
        assert (
            gguf_branch.index("= _resolve_inherited_extra_args(")
            < gguf_branch.index("enter_context(gguf_load_in_flight")
            < gguf_branch.index("_hub_download_blocks_gguf_load")
            < gguf_branch.index("enter_context(chat_load_in_flight")
            < gguf_branch.index("unsloth_backend.unload_model")
        )
        llama_source = (
            Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        ).read_text(encoding = "utf-8")
        assert "@_with_gguf_load_marker\n    def load_model(" in llama_source

    def _capture_hub_guard_require_mmproj(
        self,
        stored_extra_args,
        request_extra_args = None,
    ):
        """Drive /load's GGUF path and return the hub guard's require_mmproj.

        The guard reports a conflicting download, so the 409 is the observation
        point and no llama-server ever starts.
        """
        import core.inference.llama_cpp as llama_cpp_module

        from fastapi import HTTPException
        from models.inference import LoadRequest

        route = _load_route_module(
            "inference_route_module_for_inherited_extra_args_test",
            "routes/inference.py",
        )
        captured = {}

        def _fake_blocks(
            repo,
            variant,
            *,
            require_mmproj,
            hf_token = None,
        ):
            captured["repo"] = repo
            captured["variant"] = variant
            captured["require_mmproj"] = require_mmproj
            return True

        # A vision GGUF: require_mmproj is True unless the extras say --no-mmproj.
        config = SimpleNamespace(
            is_gguf = True,
            is_lora = False,
            is_vision = True,
            is_audio = False,
            audio_type = None,
            has_audio_input = False,
            gguf_hf_repo = REPO,
            gguf_variant = VARIANT,
            gguf_file = None,
            gguf_mmproj_file = None,
            identifier = REPO,
            display_name = REPO,
        )
        # Pass-through extras the running backend recorded for the last load.
        llama_backend = SimpleNamespace(
            is_loaded = False,
            extra_args = list(stored_extra_args),
            extra_args_source = (REPO, VARIANT),
            hf_variant = VARIANT,
            model_identifier = REPO,
            matches_load_source = lambda _intent: False,
            adopt_load_intent_if_matched = lambda _intent: False,
        )
        request = LoadRequest(
            model_path = REPO,
            gguf_variant = VARIANT,
            llama_extra_args = request_extra_args,
        )

        with (
            patch.object(
                route,
                "ModelConfig",
                SimpleNamespace(from_identifier = lambda **_kwargs: config),
            ),
            patch.object(route, "get_llama_cpp_backend", lambda: llama_backend),
            patch.object(
                route,
                "get_inference_backend",
                lambda: SimpleNamespace(active_model_name = None),
            ),
            patch.object(route, "_resolve_gguf_gpu_ids_for_request", _no_gguf_gpu_ids),
            patch.object(route, "_guard_chat_load_against_training", return_value = None),
            patch.object(route, "_effective_load_in_4bit", return_value = False),
            patch.object(route, "_hf_offline_if_unreachable", nullcontext),
            patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
            patch.object(llama_cpp_module, "_hub_download_blocks_gguf_load", _fake_blocks),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    route._load_model_impl(
                        request,
                        SimpleNamespace(
                            app = SimpleNamespace(
                                state = SimpleNamespace(llama_parallel_slots = 1),
                            ),
                        ),
                        current_subject = "test-user",
                    )
                )

        assert exc_info.value.status_code == 409
        assert captured["repo"] == REPO
        return captured["require_mmproj"]

    def test_inherited_extra_args_shape_hub_guard_require_mmproj(self):
        # Inheritance must resolve before the hub-download guard: an inherited
        # --no-mmproj decides require_mmproj, so resolving later rejects a load
        # over a download the effective arguments disable (#7251).
        assert self._capture_hub_guard_require_mmproj(["--no-mmproj"]) is False
        # Control: nothing to inherit, so a vision GGUF still needs its mmproj.
        assert self._capture_hub_guard_require_mmproj([]) is True
        # An explicit request list wins over the stored one, both ways.
        assert (
            self._capture_hub_guard_require_mmproj([], request_extra_args = ["--no-mmproj"]) is False
        )
        assert (
            self._capture_hub_guard_require_mmproj(["--no-mmproj"], request_extra_args = []) is True
        )


class _CompanionMetadataReached(BaseException):
    """Raised from a stubbed from_identifier so the load stops with its kwargs captured."""


def _companion_cache_repo(tmp_path, *, weights: str = "weights-revision"):
    """An HF cache repo whose weights snapshot predates the one holding the companions.

    The shape #10599 reports: the MTP head and the mmproj were published after the
    quant, so ``refs/main`` names a revision that holds no weights at all and the
    companions sit beside it, in a snapshot the weights' own directory does not reach.
    """
    repo = tmp_path / "models--org--Vision-GGUF"
    selected = repo / "snapshots" / weights
    companions = repo / "snapshots" / "companion-revision"
    selected.mkdir(parents = True)
    companions.mkdir(parents = True)
    (repo / "blobs").mkdir(exist_ok = True)
    (repo / "refs").mkdir(exist_ok = True)
    (repo / "refs" / "main").write_text("companion-revision")
    (selected / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    (companions / "mtp-vision-model-Q4_0.gguf").write_bytes(b"GGUF drafter")
    (companions / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
    os.utime(selected, (1_000, 1_000))
    os.utime(companions, (2_000, 2_000))
    return repo, selected, companions


@contextmanager
def _capture_load_config(
    route,
    seen,
    *,
    native_grant_backed = False,
    model_identifier = None,
):
    backend = SimpleNamespace(
        is_loaded = False,
        model_identifier = None,
        holds_no_vram = False,
        active_model_name = None,
        _audio_probed = True,
        adopt_load_intent_if_matched = lambda _intent: False,
    )

    def _from_identifier(**kwargs):
        seen.update(kwargs)
        raise _CompanionMetadataReached()

    with (
        patch.object(
            route,
            "_resolve_model_identifier_for_request",
            return_value = (model_identifier, model_identifier, native_grant_backed),
        ),
        patch.object(route, "resolve_effective_chat_template_override", return_value = None),
        patch.object(route, "get_llama_cpp_backend", return_value = backend),
        patch.object(route, "get_inference_backend", return_value = backend),
        patch.object(route, "ModelConfig", SimpleNamespace(from_identifier = _from_identifier)),
    ):
        yield


class TestPathLoadCompanionRoots:
    """#10599: a model picked in chat loads by PATH, so the auto-switch route's
    repo-level widening never ran for it and an MTP head or mmproj published into a
    later revision of the same repo dir stayed invisible until the user deleted and
    refetched the whole repo."""

    def _roots_for(self, tmp_path, name, **kwargs):
        from models.inference import LoadRequest

        route = _load_route_module(name, "routes/inference.py")
        load_path = kwargs.pop("load_path")
        preset = kwargs.pop("preset_roots", None)
        request = LoadRequest(model_path = load_path, gguf_variant = "Q4_K_M")
        if preset is not None:
            request._gguf_companion_roots = preset
        seen = {}
        with _capture_load_config(route, seen, model_identifier = load_path, **kwargs):
            with pytest.raises(_CompanionMetadataReached):
                _run_route_load(route, request)
        return seen.get("gguf_companion_roots")

    def test_path_load_widens_to_the_sibling_snapshot_holding_the_companions(self, tmp_path):
        _repo, selected, companions = _companion_cache_repo(tmp_path)
        roots = self._roots_for(
            tmp_path,
            "inference_route_module_for_path_load_widening",
            load_path = str(selected),
        )
        assert tuple(map(Path, roots or ())) == (selected, companions)

    def test_path_load_resolves_both_the_mtp_head_and_the_mmproj(self, tmp_path):
        """The user-visible symptom, through the real ModelConfig: both companions."""
        from utils.models.model_config import ModelConfig
        from core.inference.local_model_resolver import local_path_gguf_companion_roots

        _repo, selected, companions = _companion_cache_repo(tmp_path)
        assert (
            ModelConfig.from_identifier(str(selected), gguf_variant = "Q4_K_M").gguf_mtp_file is None
        )
        config = ModelConfig.from_identifier(
            str(selected),
            gguf_variant = "Q4_K_M",
            gguf_companion_roots = local_path_gguf_companion_roots(str(selected)),
        )
        assert config.gguf_mtp_file == str((companions / "mtp-vision-model-Q4_0.gguf").resolve())
        assert config.gguf_mmproj_file == str(
            (companions / "mmproj-vision-model-F16.gguf").resolve()
        )

    def test_a_pinned_non_selected_revision_is_not_widened(self, tmp_path):
        """Naming a revision the repo-level selection would not hand out pins it."""
        repo, pinned, _companions = _companion_cache_repo(tmp_path)
        newer_weights = repo / "snapshots" / "newer-weights-revision"
        newer_weights.mkdir(parents = True)
        (newer_weights / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
        os.utime(newer_weights, (3_000, 3_000))
        assert not self._roots_for(
            tmp_path,
            "inference_route_module_for_pinned_revision",
            load_path = str(pinned),
        )

    def test_a_native_grant_backed_load_is_not_widened(self, tmp_path):
        """A native grant covers one directory; a sibling revision is outside it."""
        _repo, selected, _companions = _companion_cache_repo(tmp_path)
        assert not self._roots_for(
            tmp_path,
            "inference_route_module_for_native_grant_widening",
            load_path = str(selected),
            native_grant_backed = True,
        )

    def test_roots_the_caller_already_set_are_left_alone(self, tmp_path):
        """Auto-switch resolved these from the repo, and the idle stash restores what
        the last load actually used: neither may be recomputed from the path."""
        _repo, selected, _companions = _companion_cache_repo(tmp_path)
        roots = self._roots_for(
            tmp_path,
            "inference_route_module_for_preset_roots",
            load_path = str(selected),
            preset_roots = (str(selected),),
        )
        assert tuple(map(Path, roots or ())) == (selected,)

    def test_a_bare_repo_id_is_not_widened(self, tmp_path):
        assert not self._roots_for(
            tmp_path,
            "inference_route_module_for_repo_id_load",
            load_path = REPO,
        )


def test_a_symlinked_sibling_snapshot_is_not_a_trusted_root(tmp_path):
    """`is_dir()` follows symlinks, and `follow_symlinks=False` is 3.13+.

    Access is validated for the snapshot the caller named. A directory symlink placed
    under `snapshots/` would otherwise be handed back as a trusted disjoint search root,
    so `ModelConfig.from_identifier` would read GGUF companions from wherever it points.
    """
    from core.inference.local_model_resolver import local_gguf_companion_roots

    repo = tmp_path / "cache" / "models--a--b"
    snapshots = repo / "snapshots"
    selected = snapshots / "rev1"
    selected.mkdir(parents = True)
    (selected / "model.gguf").touch()
    real_sibling = snapshots / "rev2"
    real_sibling.mkdir()
    (real_sibling / "mmproj.gguf").touch()
    outside = tmp_path / "other_tenant"
    outside.mkdir()
    (outside / "leaked.gguf").touch()
    (snapshots / "rev3").symlink_to(outside, target_is_directory = True)

    roots = local_gguf_companion_roots(str(selected), repo_level = True)
    names = [os.path.basename(root) for root in roots]
    assert "rev1" in names and "rev2" in names
    assert "rev3" not in names, f"escaped symlink returned as a trusted root: {roots}"
    assert all(str(outside) not in root for root in roots)


def test_a_repo_with_no_sibling_snapshot_widens_nothing(tmp_path):
    """One root is the caller's own snapshot, so there is nothing to widen to.

    Callers read a non-None roots tuple as `allow_disjoint_search_root`, which makes the
    mmproj scan recursive, so returning a bare single root silently relaxes a guard
    instead of being inert.
    """
    from core.inference.local_model_resolver import local_path_gguf_companion_roots

    repo = tmp_path / "cache" / "models--a--b"
    snapshots = repo / "snapshots"
    selected = snapshots / "only"
    selected.mkdir(parents = True)
    (selected / "model.gguf").touch()

    assert local_path_gguf_companion_roots(str(selected)) == ()


def test_an_explicitly_empty_companion_scope_is_not_recomputed():
    """`()` means two different things and only one of them may be widened.

    It is the default, meaning nobody has decided yet, and it is also what the
    auto-switch route and the idle stash assign when they resolved an exact revision
    and deliberately want no sibling widening. A truthiness test conflates them, and
    the pinned request then picks up a projector or drafter from another revision.
    """
    from models.inference import LoadRequest

    unset = LoadRequest(model_path = "/models/x")
    assert unset._gguf_companion_roots == ()
    assert unset._gguf_companion_roots_set is False

    pinned = LoadRequest(model_path = "/models/x")
    pinned._gguf_companion_roots = ()
    pinned._gguf_companion_roots_set = True
    # The path route's own guard, spelled exactly as it is at routes/inference.py.
    assert not (not pinned._gguf_companion_roots and not pinned._gguf_companion_roots_set)
    assert not unset._gguf_companion_roots and not unset._gguf_companion_roots_set
