# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for cached GGUF reuse and load/download exclusion.

No GPU, network, or subprocesses are required.
"""

from __future__ import annotations

import asyncio
import sys
import threading
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
    _hub_download_blocks_gguf_load,
    _supplied_mmproj_for_loaded_model,
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

    def test_three_digit_split_reused_only_when_colocated(self, hf_cache):
        backend = LlamaCppBackend()
        shard1 = f"gemma-test-{VARIANT}-001-of-002.gguf"
        shard2 = f"gemma-test-{VARIANT}-002-of-002.gguf"
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
                "_find_smallest_fitting_variant",
                lambda *_a, **_k: (fallback, 4, []),
            ),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", _fail_download),
        ):
            out = backend._download_gguf(hf_repo = REPO, hf_variant = VARIANT)

        assert out == str(snap / fallback)

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

    def test_partial_three_digit_split_is_none(self, hf_cache):
        shard1 = f"gemma-test-{VARIANT}-001-of-002.gguf"
        _build_cache(hf_cache, REPO, {shard1: 4})
        assert cached_gguf_for_load(REPO, VARIANT) is None

    def test_complete_three_digit_split_is_found(self, hf_cache):
        shard1 = f"gemma-test-{VARIANT}-001-of-002.gguf"
        shard2 = f"gemma-test-{VARIANT}-002-of-002.gguf"
        snap = _build_cache(hf_cache, REPO, {shard1: 4, shard2: 4})
        assert cached_gguf_for_load(REPO, VARIANT) == str(snap / shard1)

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

    def test_other_variant_job_allows_compatible_cross_snapshot_projector(
        self, tmp_path
    ):
        from hub.utils.download_registry import DownloadRegistry, TRANSPORT_HTTP

        main = tmp_path / "gemma-main.gguf"
        mmproj = tmp_path / "gemma-mmproj.gguf"
        main.touch()
        mmproj.touch()
        registry = DownloadRegistry()
        registry.claim(
            f"{REPO}::Q8_0",
            TRANSPORT_HTTP,
            repo_type = "model",
            repo_id = REPO,
            variant = "Q8_0",
        )

        def cached_probe(*_args, require_mmproj = False, **_kwargs):
            return None if require_mmproj else str(main)

        with (
            patch("hub.utils.download_registry.get_models_registry", lambda: registry),
            patch(
                "core.inference.llama_cpp.cached_gguf_for_load",
                side_effect = cached_probe,
            ),
            patch(
                "utils.models.model_config.mmproj_matches_model_family",
                return_value = True,
            ),
        ):
            assert (
                _hub_download_blocks_gguf_load(
                    REPO,
                    VARIANT,
                    require_mmproj = True,
                    mmproj_path = str(mmproj),
                )
                is False
            )

    def test_supplied_projector_is_kept_only_for_the_reused_cached_main(self, tmp_path):
        cached_main = tmp_path / "old" / MAIN
        downloaded_main = tmp_path / "new" / MAIN
        mmproj = tmp_path / "projector" / "gemma-mmproj.gguf"
        for path in (cached_main, downloaded_main, mmproj):
            path.parent.mkdir(parents = True, exist_ok = True)
            path.touch()

        assert _supplied_mmproj_for_loaded_model(
            cached_model_path = str(cached_main),
            loaded_model_path = str(cached_main),
            mmproj_path = str(mmproj),
        ) == str(mmproj)
        assert (
            _supplied_mmproj_for_loaded_model(
                cached_model_path = str(cached_main),
                loaded_model_path = str(downloaded_main),
                mmproj_path = str(mmproj),
            )
            is None
        )

    def test_cancelled_request_keeps_marker_until_load_thread_finishes(self):
        from core.inference.llama_cpp import _with_gguf_load_marker

        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        class FakeBackend:
            @_with_gguf_load_marker
            def load_model(self, *, hf_repo):
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
                    asyncio.to_thread(FakeBackend().load_model, hf_repo = REPO)
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

    def test_audio_input_requires_projector_in_decorator_conflict_probe(self):
        from core.inference.llama_cpp import _with_gguf_load_marker

        class FakeBackend:
            @_with_gguf_load_marker
            def load_model(
                self,
                *,
                hf_repo,
                hf_variant,
                has_audio_input = False,
            ):
                return True

        with patch(
            "core.inference.llama_cpp._hub_download_blocks_gguf_load",
            return_value = False,
        ) as blocked:
            assert FakeBackend().load_model(
                hf_repo = REPO,
                hf_variant = VARIANT,
                has_audio_input = True,
            )

        assert blocked.call_args.kwargs["require_mmproj"] is True

    def test_load_marker_precedes_hub_guard_and_unload(self):
        source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text()
        gguf_branch = source[source.index("if config.is_gguf:") :]

        # The gguf_load_in_flight marker must be entered before the hub-download
        # guard and the unload so a concurrent load can't race the download
        # manager. The llama_extra_args inheritance that used to sit between the
        # marker and the guard now runs in _guard_chat_load_against_training, ahead
        # of the GGUF branch, so it is no longer a landmark inside this slice.
        assert (
            gguf_branch.index("enter_context(gguf_load_in_flight")
            < gguf_branch.index("_hub_download_blocks_gguf_load")
            < gguf_branch.index("unsloth_backend.unload_model")
        )
        llama_source = (
            Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        ).read_text()
        assert "@_with_gguf_load_marker\n    def load_model(" in llama_source
