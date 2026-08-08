# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the offline GGUF cache fallback path (#5505).

When ``huggingface.co`` is unreachable but the repo is cached, three failures
hit: ``list_gguf_variants`` 500'd (empty dropdown), ``detect_gguf_model_remote``
returned None (GGUF-only repo misrouted), and ``_download_gguf`` synthesised a
name absent from cache. Follow-ups: the cache filter matches the snapshot-relative
path (subdir layouts findable), and DNS auto-detect scopes ``HF_HUB_OFFLINE`` to
one load so a transient hiccup can't pin the singleton offline.

No GPU, no network, no subprocess. Linux/macOS/Windows compatible.
"""

from __future__ import annotations

import contextlib
import importlib.util as _importlib_util
import os
import socket
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# Stub heavy/unavailable external deps before importing the modules under
# test (same pattern as other studio backend tests).
def _module_available(name: str) -> bool:
    """True if the real module can be imported. Probed rather than imported: these stubs
    land in sys.modules for the whole session, so an empty one breaks anything imported
    later that actually uses the module."""
    try:
        return _importlib_util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not _module_available("loggers"):
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("loggers", _loggers_stub)

if not _module_available("structlog"):
    sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

# Prefer real httpx if installed (CI installs it). Stub only as fallback.
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
    _cached_colocated_split_main,
    _gguf_files_for_variant,
    _hf_offline_if_unreachable,
    _probe_dns_dead,
    _resolve_repo_id_casing,
)
from utils.models.model_config import (
    _detect_gguf_from_hf_cache,
    _extract_quant_label,
    _iter_hf_cache_snapshots,
    _list_gguf_variants_from_hf_cache,
    detect_gguf_model_remote,
    list_gguf_variants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """Point ``huggingface_hub.constants.HF_HUB_CACHE`` at a temp dir."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: _types.SimpleNamespace(hub_cache = tmp_path),
    )
    return tmp_path


@pytest.fixture
def clean_offline_env(monkeypatch):
    """Strip ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` for the test."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


@pytest.fixture
def clean_proxy_env(monkeypatch):
    """Keep direct-urlopen probe tests independent of the runner's proxy settings."""
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(key, raising = False)


class TestGgufVariantFileResolution:
    def test_prefers_exact_unknown_variant_over_big_endian_sibling(self):
        files = [
            "tinyllamas/stories260K-be.gguf",
            "tinyllamas/stories260K-infill.gguf",
            "tinyllamas/stories260K.gguf",
        ]

        assert _gguf_files_for_variant(files, "stories260K") == ["tinyllamas/stories260K.gguf"]

    @pytest.mark.parametrize(
        "big_endian_path",
        [
            "model-Q4_K_M-be.gguf",
            "model-Q4_K_M_be.gguf",
            "model-Q4_K_M_be_infill.gguf",
            r"nested\model-Q4_K_M_be.gguf",
        ],
    )
    def test_filters_big_endian_known_quant_before_exact_match(self, big_endian_path):
        files = [
            big_endian_path,
            "model-Q4_K_M.gguf",
        ]

        assert _gguf_files_for_variant(files, "Q4_K_M") == ["model-Q4_K_M.gguf"]

    def test_keeps_model_name_be_token_before_quant(self):
        files = [
            "foo-be-Q4_K_M.gguf",
        ]

        assert _gguf_files_for_variant(files, "Q4_K_M") == ["foo-be-Q4_K_M.gguf"]

    def test_keeps_model_name_be_token_with_quant_subdir(self):
        files = [
            "Q4_K_M/foo-be.gguf",
        ]

        assert _gguf_files_for_variant(files, "Q4_K_M") == ["Q4_K_M/foo-be.gguf"]

    def test_empty_variant_filters_big_endian_files(self):
        files = [
            "model-Q4_K_M-be.gguf",
            "model-Q4_K_M.gguf",
        ]

        assert _gguf_files_for_variant(files, "") == ["model-Q4_K_M.gguf"]

    def test_remote_listing_skips_big_endian_quant_sibling(self, monkeypatch, clean_offline_env):
        siblings = [
            _types.SimpleNamespace(rfilename = "model-Q4_K_M-be.gguf", size = 100),
            _types.SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10),
        ]
        monkeypatch.setattr(
            "huggingface_hub.model_info",
            lambda *_args, **_kwargs: _types.SimpleNamespace(siblings = siblings),
        )

        variants, has_vision = list_gguf_variants("org/repo")

        assert has_vision is False
        assert [(v.quant, v.filename, v.size_bytes) for v in variants] == [
            ("Q4_K_M", "model-Q4_K_M.gguf", 10)
        ]

    def test_download_uses_exact_variant_label(self, monkeypatch, tmp_path):
        backend = LlamaCppBackend()
        downloaded: list[str] = []

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = path, size = 1) for path in paths if path is not None]

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(
            "utils.hf_cache_settings.get_hf_cache_paths",
            lambda: _types.SimpleNamespace(hub_cache = tmp_path),
        )
        with (
            patch(
                "huggingface_hub.list_repo_files",
                lambda *_a, **_k: [
                    "tinyllamas/stories260K-be.gguf",
                    "tinyllamas/stories260K-infill.gguf",
                    "tinyllamas/stories260K.gguf",
                ],
            ),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(
                hf_repo = "ggml-org/models",
                hf_variant = "stories260K",
            )

        assert downloaded == ["tinyllamas/stories260K.gguf"]
        assert out == "/fake/ggml-org/models/tinyllamas/stories260K.gguf"

    def test_download_reuses_older_snapshot_when_current_ref_snapshot_is_partial(
        self, monkeypatch, hf_cache
    ):
        # Keep coverage for offline reuse; online reuse is tested separately.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        backend = LlamaCppBackend()
        repo = "unsloth/vision-GGUF"
        old = _build_cache(
            hf_cache,
            repo,
            {"model-UD-Q4_K_XL.gguf": 4},
            snapshot_sha = "a" * 40,
        )
        _build_cache(
            hf_cache,
            repo,
            {"mtp-model.gguf": 1},
            snapshot_sha = "b" * 40,
        )

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = path, size = 4) for path in paths if path]

        def fail_download(*_args, **_kwargs):
            raise AssertionError("should reuse the cached GGUF instead of downloading")

        with (
            patch(
                "huggingface_hub.list_repo_files",
                lambda *_a, **_k: ["model-UD-Q4_K_XL.gguf", "mtp-model.gguf"],
            ),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fail_download),
        ):
            out = backend._download_gguf(
                hf_repo = repo,
                hf_variant = "UD-Q4_K_XL",
            )

        assert out == str(old / "model-UD-Q4_K_XL.gguf")

    def test_download_reuses_cached_gguf_when_lowercase_partial_cache_shadows_it(
        self, monkeypatch, hf_cache
    ):
        # Keep coverage for case-insensitive offline cache lookup.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        backend = LlamaCppBackend()
        canonical_repo = "unsloth/gemma-4-E2B-it-GGUF"
        requested_repo = "unsloth/gemma-4-e2b-it-gguf"
        gguf_file = "gemma-4-E2B-it-UD-Q4_K_XL.gguf"
        snap = _build_cache(
            hf_cache,
            canonical_repo,
            {gguf_file: 4},
            snapshot_sha = "a" * 40,
        )
        lower_snap = _build_cache(
            hf_cache,
            requested_repo,
            {"mtp-gemma-4-E2B-it.gguf": 1},
            snapshot_sha = "b" * 40,
        )
        os.utime(lower_snap, (2000, 2000))
        os.utime(snap, (1000, 1000))
        seen_repos: list[str] = []

        def fake_list_repo_files(repo_id, token = None):
            seen_repos.append(repo_id)
            return [gguf_file]

        def fake_get_paths_info(
            repo_id,
            paths,
            token = None,
        ):
            seen_repos.append(repo_id)
            return [_types.SimpleNamespace(path = path, size = 4) for path in paths if path]

        def fake_cache(repo_id, filename, *args, **kwargs):
            seen_repos.append(repo_id)
            return str(snap / filename) if repo_id == canonical_repo else None

        def fail_download(*_args, **_kwargs):
            raise AssertionError("should reuse the cached GGUF instead of downloading")

        with (
            patch("huggingface_hub.list_repo_files", fake_list_repo_files),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", fake_cache),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fail_download),
        ):
            out = backend._download_gguf(
                hf_repo = requested_repo,
                hf_variant = "UD-Q4_K_XL",
            )

        assert out == str(snap / gguf_file)
        assert seen_repos

    def test_download_online_reuses_complete_cached_snapshot(self, monkeypatch, hf_cache):
        # Loads reuse complete cached models across repo revisions.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        backend = LlamaCppBackend()
        repo = "unsloth/vision-GGUF"
        snap = _build_cache(hf_cache, repo, {"model-UD-Q4_K_XL.gguf": 4}, snapshot_sha = "a" * 40)

        def fail_download(*_args, **_kwargs):
            raise AssertionError("must reuse the cached GGUF instead of downloading")

        with (
            patch(
                "huggingface_hub.list_repo_files",
                lambda *_a, **_k: ["model-UD-Q4_K_XL.gguf"],
            ),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fail_download),
        ):
            out = backend._download_gguf(hf_repo = repo, hf_variant = "UD-Q4_K_XL")

        assert out == str(snap / "model-UD-Q4_K_XL.gguf")

    def test_download_reuses_older_snapshot_when_offline_env_is_true(self, monkeypatch, hf_cache):
        # HF_HUB_OFFLINE accepts truthy spellings beyond "1" (true/yes/on); the offline
        # cache reuse must trigger for those too, otherwise the earlier Hub calls run
        # offline while this branch still attempts hf_hub_download and the cached GGUF
        # cannot load.
        monkeypatch.setenv("HF_HUB_OFFLINE", "true")
        backend = LlamaCppBackend()
        repo = "unsloth/vision-GGUF"
        old = _build_cache(hf_cache, repo, {"model-UD-Q4_K_XL.gguf": 4}, snapshot_sha = "a" * 40)

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = p, size = 4) for p in paths if p]

        def fail_download(*_args, **_kwargs):
            raise AssertionError("should reuse the cached GGUF instead of downloading")

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: ["model-UD-Q4_K_XL.gguf"]),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fail_download),
        ):
            out = backend._download_gguf(hf_repo = repo, hf_variant = "UD-Q4_K_XL")

        assert out == str(old / "model-UD-Q4_K_XL.gguf")

    def test_download_companion_resolves_from_case_variant_snapshot_offline(
        self, monkeypatch, hf_cache
    ):
        # Offline, resolve_cached_repo_id_case can keep a partial lower-case spelling,
        # so the companion (mmproj) must resolve from whichever case-variant snapshot
        # actually holds it rather than being dropped by an hf_hub_download on the
        # wrong casing.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        backend = LlamaCppBackend()
        canonical_repo = "unsloth/gemma-4-E2B-it-GGUF"
        requested_repo = "unsloth/gemma-4-e2b-it-gguf"
        snap = _build_cache(hf_cache, canonical_repo, {"mmproj-F16.gguf": 4}, snapshot_sha = "a" * 40)
        # A partial lower-case dir exists so casing resolution keeps the requested spelling.
        _build_cache(hf_cache, requested_repo, {"config.json": 1}, snapshot_sha = "b" * 40)

        _offline_exc = type("OfflineModeIsEnabled", (Exception,), {})

        def fake_list_repo_files(repo_id, token = None):
            raise _offline_exc("offline")

        def fail_download(*_args, **_kwargs):
            raise AssertionError("should resolve the companion from cache, not download")

        with (
            patch("huggingface_hub.list_repo_files", fake_list_repo_files),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fail_download),
        ):
            out = backend._download_mmproj(hf_repo = requested_repo)

        assert out == str(snap / "mmproj-F16.gguf")

    def test_download_companion_uses_selected_cache_not_import_time_default(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        import_time_cache = tmp_path / "import-time-cache"
        selected_cache = tmp_path / "selected-cache"
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(import_time_cache))
        monkeypatch.setattr(
            "utils.hf_cache_settings.get_hf_cache_paths",
            lambda: _types.SimpleNamespace(hub_cache = selected_cache),
        )
        repo = "unsloth/vision-GGUF"
        snap = _build_cache(selected_cache, repo, {"mmproj-F16.gguf": 4})
        backend = LlamaCppBackend()

        offline_error = type("OfflineModeIsEnabled", (Exception,), {})

        def fail_list(*_args, **_kwargs):
            raise offline_error("offline")

        def fail_download(*_args, **_kwargs):
            raise AssertionError("selected-cache companion must not download")

        with (
            patch("huggingface_hub.list_repo_files", fail_list),
            patch(
                "core.inference.llama_cpp.hf_hub_download_with_xet_fallback",
                fail_download,
            ),
        ):
            out = backend._download_mmproj(hf_repo = repo)

        assert out == str(snap / "mmproj-F16.gguf")

    def test_download_includes_uppercase_split_gguf_shards(self, monkeypatch, tmp_path):
        backend = LlamaCppBackend()
        downloaded: list[str] = []

        files = [
            "model-Q4_K_M-00001-of-00002.GGUF",
            "model-Q4_K_M-00002-of-00002.GGUF",
        ]

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = path, size = 1) for path in paths if path is not None]

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(
            "utils.hf_cache_settings.get_hf_cache_paths",
            lambda: _types.SimpleNamespace(hub_cache = tmp_path),
        )
        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: files),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(
                hf_repo = "org/repo",
                hf_variant = "Q4_K_M",
            )

        assert downloaded == files
        assert out == "/fake/org/repo/model-Q4_K_M-00001-of-00002.GGUF"

    def test_download_refetches_split_gguf_when_shards_span_snapshots(self, monkeypatch, hf_cache):
        # The cached main shard lives in an older snapshot; its sibling shard is only
        # in a newer, separate snapshot. Reusing the main shard alone would leave
        # llama.cpp unable to resolve the sibling, so the whole set must be re-fetched
        # together (co-located) rather than served split across snapshot dirs.
        backend = LlamaCppBackend()
        repo = "org/split"
        files = [
            "model-Q4_K_M-00001-of-00002.gguf",
            "model-Q4_K_M-00002-of-00002.gguf",
        ]
        _build_cache(hf_cache, repo, {files[0]: 4}, snapshot_sha = "a" * 40)
        _build_cache(hf_cache, repo, {files[1]: 4}, snapshot_sha = "b" * 40)
        downloaded: list[str] = []

        def fake_get_paths_info(
            _repo_id,
            paths,
            token = None,
        ):
            return [_types.SimpleNamespace(path = p, size = 4) for p in paths if p]

        def fake_download(
            repo_id,
            filename,
            token = None,
            **_kwargs,
        ):
            downloaded.append(filename)
            return f"/fake/{repo_id}/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", lambda *_a, **_k: files),
            patch("huggingface_hub.get_paths_info", fake_get_paths_info),
            patch("huggingface_hub.try_to_load_from_cache", lambda *_a, **_k: None),
            patch("core.inference.llama_cpp.hf_hub_download_with_xet_fallback", fake_download),
        ):
            out = backend._download_gguf(hf_repo = repo, hf_variant = "Q4_K_M")

        assert downloaded == files
        assert out == f"/fake/{repo}/{files[0]}"


def _siblings(items: dict[str, int]):
    """Mock ``hf_model_info(...).siblings`` payload."""
    return _types.SimpleNamespace(
        siblings = [
            _types.SimpleNamespace(rfilename = name, size = size) for name, size in items.items()
        ],
    )


# ---------------------------------------------------------------------------
# _iter_hf_cache_snapshots
# ---------------------------------------------------------------------------


class TestIterHfCacheSnapshots:
    def test_returns_empty_when_cache_dir_missing(self, monkeypatch):
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", "/no/such/dir")
        assert list(_iter_hf_cache_snapshots("unsloth/foo")) == []

    def test_returns_empty_when_repo_not_cached(self, hf_cache):
        assert list(_iter_hf_cache_snapshots("unsloth/not-here")) == []

    def test_returns_empty_when_snapshots_dir_missing(self, hf_cache):
        # Repo dir exists but no snapshots/ inside.
        (hf_cache / "models--unsloth--bare").mkdir()
        assert list(_iter_hf_cache_snapshots("unsloth/bare")) == []

    def test_yields_newest_first(self, hf_cache):
        old = _build_cache(hf_cache, "unsloth/multi", {"x.gguf": 1}, snapshot_sha = "a" * 40)
        new = _build_cache(hf_cache, "unsloth/multi", {"y.gguf": 1}, snapshot_sha = "b" * 40)
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        out = list(_iter_hf_cache_snapshots("unsloth/multi"))
        assert [p.name for p in out] == ["b" * 40, "a" * 40]

    def test_skips_snapshot_when_mtime_is_unavailable(self, hf_cache, monkeypatch):
        stale = _build_cache(hf_cache, "unsloth/multi", {"x.gguf": 1}, snapshot_sha = "a" * 40)
        good = _build_cache(hf_cache, "unsloth/multi", {"y.gguf": 1}, snapshot_sha = "b" * 40)
        original_stat = Path.stat

        def flaky_stat(self, *args, **kwargs):
            if self == stale:
                raise FileNotFoundError(str(self))
            return original_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", flaky_stat)

        out = list(_iter_hf_cache_snapshots("unsloth/multi"))
        assert out == [good]

    def test_repo_id_match_is_case_insensitive(self, hf_cache):
        _build_cache(hf_cache, "unsloth/Foo-GGUF", {"Foo-Q4_K_M.gguf": 1})
        # Lookup with different org/name casing still resolves
        out = list(_iter_hf_cache_snapshots("UNSLOTH/foo-gguf"))
        assert len(out) == 1


# ---------------------------------------------------------------------------
# _list_gguf_variants_from_hf_cache / list_gguf_variants
# ---------------------------------------------------------------------------


class TestListGgufVariantsFromCache:
    def test_returns_variants_when_cached(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/Qwen3.5-4B-GGUF",
            {
                "Qwen3.5-4B-UD-Q4_K_XL.gguf": 100,
                "Qwen3.5-4B-Q2_K.gguf": 50,
            },
        )
        out = _list_gguf_variants_from_hf_cache("unsloth/Qwen3.5-4B-GGUF")
        assert out is not None
        variants, has_vision = out
        assert sorted(v.quant for v in variants) == ["Q2_K", "UD-Q4_K_XL"]
        assert has_vision is False

    def test_returns_none_when_not_cached(self, hf_cache):
        assert _list_gguf_variants_from_hf_cache("unsloth/absent") is None


class TestCachedColocatedSplitMain:
    def test_prefers_older_complete_snapshot_over_newer_partial(self, hf_cache):
        # Newer snapshot has only shard 1; older snapshot has the complete set. The
        # complete older snapshot must win so the split GGUF can load co-located.
        shard1 = "m-00001-of-00002.gguf"
        shard2 = "m-00002-of-00002.gguf"
        old = _build_cache(
            hf_cache, "unsloth/split-GGUF", {shard1: 100, shard2: 100}, snapshot_sha = "a" * 40
        )
        new = _build_cache(hf_cache, "unsloth/split-GGUF", {shard1: 100}, snapshot_sha = "b" * 40)
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))

        main = _cached_colocated_split_main("unsloth/split-GGUF", shard1, [shard2], {})
        assert main is not None
        assert main.startswith(str(old))

    def test_returns_none_when_shards_span_snapshots(self, hf_cache):
        shard1 = "m-00001-of-00002.gguf"
        shard2 = "m-00002-of-00002.gguf"
        a = _build_cache(hf_cache, "unsloth/split-GGUF", {shard1: 100}, snapshot_sha = "a" * 40)
        b = _build_cache(hf_cache, "unsloth/split-GGUF", {shard2: 100}, snapshot_sha = "b" * 40)
        os.utime(a, (1000, 1000))
        os.utime(b, (2000, 2000))

        assert _cached_colocated_split_main("unsloth/split-GGUF", shard1, [shard2], {}) is None


class TestResolveRepoIdCasing:
    def test_maps_to_canonical_casing(self, monkeypatch):
        monkeypatch.setattr(
            "utils.paths.resolve_cached_repo_id_case",
            lambda repo: "unsloth/Gemma-4-GGUF" if repo.lower() == "unsloth/gemma-4-gguf" else repo,
        )
        # A companion download passed the resolved id reads the same cache entry
        # as the main GGUF instead of missing it under the requested casing.
        assert _resolve_repo_id_casing("unsloth/gemma-4-gguf") == "unsloth/Gemma-4-GGUF"

    def test_passthrough_on_resolver_error(self, monkeypatch):
        def boom(_repo):
            raise RuntimeError("resolver unavailable")

        monkeypatch.setattr("utils.paths.resolve_cached_repo_id_case", boom)
        assert _resolve_repo_id_casing("unsloth/gemma-4-gguf") == "unsloth/gemma-4-gguf"

    def test_companion_only_newer_snapshot_does_not_shadow_real_variants(self, hf_cache):
        # A newer snapshot holds only a vision projector fetched on demand,
        # while the quant files live in an older snapshot. The newer snapshot
        # must not shadow the real variants; the vision flag carries over.
        old = _build_cache(
            hf_cache,
            "unsloth/vision-GGUF",
            {"vision-Q4_K_M.gguf": 100},
            snapshot_sha = "a" * 40,
        )
        new = _build_cache(
            hf_cache,
            "unsloth/vision-GGUF",
            {"mmproj-vision-F16.gguf": 10},
            snapshot_sha = "b" * 40,
        )
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))

        out = _list_gguf_variants_from_hf_cache("unsloth/vision-GGUF")
        assert out is not None
        variants, has_vision = out
        assert [v.quant for v in variants] == ["Q4_K_M"]
        assert has_vision is True

    def test_companion_only_cache_returns_empty_variants_with_vision(self, hf_cache):
        # Only a vision projector is cached anywhere: report the vision flag
        # with an empty variant list rather than None.
        _build_cache(hf_cache, "unsloth/vision-GGUF", {"mmproj-vision-F16.gguf": 10})
        out = _list_gguf_variants_from_hf_cache("unsloth/vision-GGUF")
        assert out is not None
        variants, has_vision = out
        assert variants == []
        assert has_vision is True


class TestListGgufVariantsOffline:
    def test_offline_env_short_circuits_api(self, hf_cache, clean_offline_env, monkeypatch):
        _build_cache(hf_cache, "unsloth/a", {"a-UD-Q4_K_XL.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            variants, _has = list_gguf_variants("unsloth/a")
        assert len(variants) == 1
        assert variants[0].quant == "UD-Q4_K_XL"

    @pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_cache_miss_does_not_call_api(
        self, hf_cache, clean_offline_env, monkeypatch, offline_variable
    ):
        monkeypatch.setenv(offline_variable, "yes")

        def boom(*_args, **_kwargs):
            raise AssertionError("API must not be called on an offline cache miss")

        with patch("huggingface_hub.model_info", boom):
            variants, has_vision = list_gguf_variants("unsloth/not-cached")

        assert variants == []
        assert has_vision is False

    def test_api_exception_falls_back_to_cache(self, hf_cache, clean_offline_env):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            variants, _has = list_gguf_variants("unsloth/a")
        assert len(variants) == 1
        assert variants[0].quant == "Q4_K_M"

    def test_api_exception_with_no_cache_reraises(self, hf_cache, clean_offline_env):
        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            with pytest.raises(OSError, match = "network down"):
                list_gguf_variants("unsloth/never-cached")

    def test_online_path_unaffected(self, hf_cache, clean_offline_env):
        # When the API succeeds, cache is not consulted.
        api_payload = _siblings({"a-UD-Q4_K_XL.gguf": 5, "a-Q2_K.gguf": 3})

        def hf_info(*a, **k):
            return api_payload

        with patch("huggingface_hub.model_info", hf_info):
            variants, _has = list_gguf_variants("unsloth/a")
        assert sorted(v.quant for v in variants) == ["Q2_K", "UD-Q4_K_XL"]


# ---------------------------------------------------------------------------
# _detect_gguf_from_hf_cache / detect_gguf_model_remote
# ---------------------------------------------------------------------------


class TestDetectGgufFromCache:
    def test_picks_best_quant(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/a",
            {"a-Q2_K.gguf": 1, "a-UD-Q4_K_XL.gguf": 1},
        )
        assert _detect_gguf_from_hf_cache("unsloth/a") == "a-UD-Q4_K_XL.gguf"

    def test_subdir_only_quant_resolves(self, hf_cache):
        """Regression: ``BF16/foo.gguf`` (quant only in directory). The pre-fix
        cache scan matched on basename and missed this layout."""
        _build_cache(
            hf_cache,
            "unsloth/gpt-oss-20b-BF16",
            {"BF16/foo.gguf": 1},
        )
        out = _detect_gguf_from_hf_cache("unsloth/gpt-oss-20b-BF16")
        assert (
            out == "BF16/foo.gguf"
        ), f"subdir-only layout must resolve to relative path, got {out}"

    def test_subdir_quant_keeps_be_model_name_token(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/a",
            {"Q4_K_M/foo-be.gguf": 1},
        )
        assert _detect_gguf_from_hf_cache("unsloth/a") == "Q4_K_M/foo-be.gguf"

    def test_big_endian_only_cache_is_not_detected(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/a",
            {"model-Q4_K_M-be.gguf": 1},
        )
        assert _detect_gguf_from_hf_cache("unsloth/a") is None

    def test_returns_none_when_no_gguf(self, hf_cache):
        _build_cache(hf_cache, "unsloth/a", {"README.md": 10})
        assert _detect_gguf_from_hf_cache("unsloth/a") is None


class TestDetectGgufModelRemoteOffline:
    def test_offline_env_short_circuits_retries(self, hf_cache, clean_offline_env, monkeypatch):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            assert detect_gguf_model_remote("unsloth/a") == "a-Q4_K_M.gguf"

    @pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_offline_cache_miss_does_not_call_api(
        self, hf_cache, clean_offline_env, monkeypatch, offline_variable
    ):
        monkeypatch.setenv(offline_variable, "on")

        def boom(*_args, **_kwargs):
            raise AssertionError("API must not be called on an offline cache miss")

        with patch("huggingface_hub.model_info", boom):
            assert detect_gguf_model_remote("unsloth/not-cached") is None

    def test_api_3x_failure_then_cache(self, hf_cache, clean_offline_env):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("hub down")

        # Patch time.sleep so the 1s/2s/4s backoff doesn't slow the test.
        with (
            patch("huggingface_hub.model_info", boom),
            patch("time.sleep", lambda *_: None),
        ):
            out = detect_gguf_model_remote("unsloth/a")
        assert out == "a-Q4_K_M.gguf"

    def test_remote_big_endian_only_repo_is_not_detected(self, clean_offline_env, monkeypatch):
        siblings = [
            _types.SimpleNamespace(rfilename = "model-Q4_K_M-be.gguf"),
        ]
        monkeypatch.setattr(
            "huggingface_hub.model_info",
            lambda *_args, **_kwargs: _types.SimpleNamespace(siblings = siblings),
        )

        assert detect_gguf_model_remote("unsloth/a") is None

    def test_repository_not_found_does_not_consult_cache(self, hf_cache, clean_offline_env):
        # Cache has a file but the API says the repo is gone.
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})

        class RepositoryNotFoundError(Exception):
            pass

        def gone(*a, **k):
            raise RepositoryNotFoundError("404")

        with patch("huggingface_hub.model_info", gone):
            out = detect_gguf_model_remote("unsloth/a")
        # Early-return semantics preserved: 404 wins over a stale cache.
        assert out is None


# ---------------------------------------------------------------------------
# _probe_dns_dead / _hf_offline_if_unreachable
# ---------------------------------------------------------------------------


class _DnsState:
    """Toggles resolver failure. The probe uses ``getaddrinfo`` (IPv4 + IPv6), so patch
    both entry points."""

    def __init__(self, monkeypatch):
        self._mp = monkeypatch
        self._real = socket.gethostbyname
        self._real_addr = socket.getaddrinfo

    def fail(self):
        def _fail(*a, **k):
            raise socket.gaierror(-2, "Name or service not known")

        self._mp.setattr(socket, "gethostbyname", _fail)
        self._mp.setattr(socket, "getaddrinfo", _fail)

    def ok(self):
        self._mp.setattr(socket, "gethostbyname", lambda *a, **k: "127.0.0.1")
        self._mp.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
        )

    def restore(self):
        self._mp.setattr(socket, "gethostbyname", self._real)
        self._mp.setattr(socket, "getaddrinfo", self._real_addr)


@pytest.fixture(autouse = True)
def _drop_reachability_memo():
    """_hf_unreachable reuses a fresh verdict before re-running the DNS shortcut, so a
    verdict left by a neighbouring test would short-circuit this one's stubs. Process-global
    state, so clear it either side rather than relying on collection order."""
    from utils.utils import reset_hf_reachability_cache

    reset_hf_reachability_cache()
    yield
    reset_hf_reachability_cache()


@pytest.fixture
def dns(monkeypatch):
    return _DnsState(monkeypatch)


@pytest.fixture
def reachable(monkeypatch):
    """Control the endpoint reachability probe; defaults to reachable so no test hits the network."""
    import utils.utils as _utils

    def _set(unreachable: bool):
        monkeypatch.setattr(_utils, "hf_unreachable", lambda *a, **k: unreachable)

    _set(False)
    return _set


class TestProbeDnsDead:
    def test_returns_false_on_success(self, dns):
        dns.ok()
        assert _probe_dns_dead() is False

    def test_returns_true_on_failure(self, dns):
        dns.fail()
        assert _probe_dns_dead() is True

    def test_restores_prior_socket_timeout(self, dns):
        dns.ok()
        socket.setdefaulttimeout(7.5)
        try:
            _probe_dns_dead()
            assert socket.getdefaulttimeout() == 7.5
        finally:
            socket.setdefaulttimeout(None)


class TestHfOfflineIfUnreachable:
    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env):
        """hf_dns_dead stands down when a proxy is configured, so on a runner with
        HTTP_PROXY set the patched resolver is bypassed and the DNS-failure scenario these
        tests mean to exercise never happens."""

    def test_dns_fail_sets_env_inside_block_only(self, dns, reachable, clean_offline_env):
        dns.fail()
        assert "HF_HUB_OFFLINE" not in os.environ
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is True
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # P1 #2: env must be restored after the block
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    def test_dns_ok_and_reachable_is_noop(self, dns, reachable, clean_offline_env):
        dns.ok()
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_dns_ok_but_endpoint_unreachable_engages(self, dns, reachable, clean_offline_env):
        # WAN down behind a live router: DNS answers, egress does not.
        dns.ok()
        reachable(True)
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is True
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert "HF_HUB_OFFLINE" not in os.environ

    def test_probe_opt_out_keeps_dns_only_behaviour(self, dns, clean_offline_env, monkeypatch):
        monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "0")
        dns.ok()
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_dns_recovers_between_calls(self, dns, reachable, clean_offline_env):
        # First call: DNS dead -> env set inside, cleared on exit.
        dns.fail()
        with _hf_offline_if_unreachable():
            pass
        assert "HF_HUB_OFFLINE" not in os.environ
        # Second call: DNS healthy -> no env mutation.
        dns.ok()
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_user_set_hf_hub_offline_is_preserved(
        self, dns, reachable, clean_offline_env, monkeypatch
    ):
        # User explicitly set offline before launching Unsloth.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_unreachable() as did_set:
            assert did_set is False
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # Helper must not pop a variable it did not set.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_user_set_transformers_offline_is_preserved(
        self, dns, reachable, clean_offline_env, monkeypatch
    ):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_unreachable():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # HF_HUB_OFFLINE was set by helper -> removed.
        assert "HF_HUB_OFFLINE" not in os.environ
        # TRANSFORMERS_OFFLINE pre-existed -> preserved.
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_exception_inside_block_still_restores_env(self, dns, reachable, clean_offline_env):
        dns.fail()
        with pytest.raises(RuntimeError, match = "boom"):
            with _hf_offline_if_unreachable():
                raise RuntimeError("boom")
        # Cleanup must happen on exception as well.
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ


class TestEndpointAwareOfflineDetection:
    """A reachable HF_ENDPOINT mirror must not be declared offline just because
    huggingface.co does not resolve (air-gapped / corporate networks)."""

    @pytest.fixture
    def no_upstream_dns(self, monkeypatch):
        def _host(h, *a, **k):
            if "huggingface.co" in str(h):
                raise socket.gaierror(-2, "Name or service not known")
            return "127.0.0.1"

        def _addr(h, *a, **k):
            if "huggingface.co" in str(h):
                raise socket.gaierror(-2, "Name or service not known")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(socket, "gethostbyname", _host)
        monkeypatch.setattr(socket, "getaddrinfo", _addr)

    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("https://hf-mirror.com", "hf-mirror.com"),
            ("hf-mirror.com", "hf-mirror.com"),
            ("https://hf-mirror.com:8443", "hf-mirror.com"),
            ("https://hf-mirror.com/path", "hf-mirror.com"),
            ("", "huggingface.co"),
        ],
    )
    def test_endpoint_host_parsing(self, monkeypatch, endpoint, expected):
        from core.inference.llama_cpp import _hf_endpoint_host
        monkeypatch.setenv("HF_ENDPOINT", endpoint)
        assert _hf_endpoint_host() == expected

    def test_dns_precheck_follows_endpoint(self, monkeypatch, no_upstream_dns):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        assert _probe_dns_dead() is False

    def test_default_endpoint_still_probes_huggingface(self, monkeypatch, no_upstream_dns):
        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        assert _probe_dns_dead() is True


class TestProxyOnlyEgress:
    """With a proxy, the proxy resolves the hub host, so local DNS proves nothing and
    must not be used to declare the hub offline."""

    @pytest.fixture
    def dns_all_dead(self, monkeypatch):
        def _fail(*a, **k):
            raise socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(socket, "gethostbyname", _fail)
        monkeypatch.setattr(socket, "getaddrinfo", _fail)

    def test_dns_shortcut_stands_down_when_proxy_configured(self, monkeypatch, dns_all_dead):
        from utils.utils import hf_dns_dead

        monkeypatch.delenv("HF_ENDPOINT", raising = False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.internal:3128")
        assert hf_dns_dead() is False

    def test_dns_shortcut_applies_without_a_proxy(self, monkeypatch, dns_all_dead):
        from utils.utils import hf_dns_dead

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
            monkeypatch.delenv(key, raising = False)
        monkeypatch.setattr("utils.utils.hf_proxy_configured", lambda: False)
        assert hf_dns_dead() is True

    def test_no_proxy_bypass_restores_the_shortcut(self, monkeypatch, dns_all_dead):
        """A host listed in NO_PROXY does not go through the proxy, so DNS matters again."""
        from utils.utils import hf_proxy_configured

        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "huggingface.co")
        assert hf_proxy_configured() is False


class TestAllProxyIsHonoured:
    """The Hub client (requests) honours all_proxy but urllib does not, so a proxy-only
    setup would fail the probe's direct lookup and be called offline while real hub calls
    succeed. Resolution must match requests' select_proxy."""

    @pytest.fixture(autouse = True)
    def _clean_proxy_env(self, monkeypatch):
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        ):
            monkeypatch.delenv(key, raising = False)
        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")

    def test_all_proxy_alone_resolves(self, monkeypatch):
        from utils.utils import hf_proxy_configured, hf_proxy_for_endpoint

        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        assert hf_proxy_for_endpoint() == "http://proxy.internal:3128"
        assert hf_proxy_configured() is True

    def test_scheme_specific_beats_all_proxy(self, monkeypatch):
        from utils.utils import hf_proxy_for_endpoint

        monkeypatch.setenv("ALL_PROXY", "http://catchall:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://specific:3128")
        assert hf_proxy_for_endpoint() == "http://specific:3128"

    def test_no_proxy_wins_over_all_proxy(self, monkeypatch):
        from utils.utils import hf_proxy_for_endpoint

        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "huggingface.co")
        assert hf_proxy_for_endpoint() is None

    def test_no_proxy_cidr_matches_requests(self, monkeypatch):
        """The probe and Hub client must both bypass a proxy for an IP in a CIDR."""
        from utils.utils import hf_proxy_for_endpoint

        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")
        assert hf_proxy_for_endpoint("https://10.23.4.5") is None

    def test_no_proxy_host_with_port_matches_requests(self, monkeypatch):
        """requests includes an explicit endpoint port when matching NO_PROXY."""
        from utils.utils import hf_proxy_for_endpoint

        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "huggingface.co:443")
        assert hf_proxy_for_endpoint("https://huggingface.co:443") is None

    def test_direct_egress_resolves_to_none(self):
        from utils.utils import hf_proxy_for_endpoint
        assert hf_proxy_for_endpoint() is None

    def test_connect_target_follows_all_proxy(self, monkeypatch):
        from utils.utils import hf_connect_target
        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        assert hf_connect_target() == ("proxy.internal", 3128)

    def test_probe_opens_through_all_proxy(self, monkeypatch):
        """The HEAD probe must be issued through the proxy, not attempted directly."""
        import urllib.request

        from utils.transformers_version import hf_endpoint_unreachable

        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        seen = {}
        real_build = urllib.request.build_opener

        def _spy(*handlers):
            for h in handlers:
                if isinstance(h, urllib.request.ProxyHandler):
                    seen["proxies"] = dict(h.proxies)
            opener = real_build(*handlers)
            opener.open = lambda req, timeout = None: _Resp()  # noqa: ARG005
            return opener

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "build_opener", _spy)
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("probe bypassed the proxy"),
        )
        assert hf_endpoint_unreachable(timeout = 1) is False
        assert seen["proxies"] == {"https": "http://proxy.internal:3128"}

    def test_probe_forces_direct_opener_when_no_proxy_bypasses(self, monkeypatch):
        import urllib.request

        from utils.transformers_version import hf_endpoint_unreachable

        monkeypatch.setenv("HF_ENDPOINT", "https://10.23.4.5")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "10.0.0.0/8")
        seen = {}
        real_build = urllib.request.build_opener

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _spy(*handlers):
            for handler in handlers:
                if isinstance(handler, urllib.request.ProxyHandler):
                    seen["proxies"] = dict(handler.proxies)
            opener = real_build(*handlers)
            opener.open = lambda req, timeout = None: _Resp()  # noqa: ARG005
            return opener

        monkeypatch.setattr(urllib.request, "build_opener", _spy)
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: pytest.fail("default opener re-evaluated proxy settings"),
        )
        assert hf_endpoint_unreachable(timeout = 1) is False
        assert seen["proxies"] == {}


class TestEnvOfflineSkipsTheProbe:
    """An explicitly offline process must emit no network traffic. The hub reads only
    HF_HUB_OFFLINE, so TRANSFORMERS_OFFLINE alone still has to engage the guard."""

    def test_transformers_offline_engages_without_probing(self, monkeypatch):
        llama_cpp = pytest.importorskip("core.inference.llama_cpp")

        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        monkeypatch.setattr(
            llama_cpp,
            "_hf_unreachable",
            lambda: pytest.fail("probed the network in an explicitly offline process"),
        )
        with llama_cpp._hf_offline_if_unreachable() as engaged:
            assert engaged is True
            assert hf_constants.HF_HUB_OFFLINE is True

    def test_hub_offline_zero_still_opts_out(self, monkeypatch):
        """HF_HUB_OFFLINE=0 is an explicit "stay online" and outranks TRANSFORMERS_OFFLINE."""
        llama_cpp = pytest.importorskip("core.inference.llama_cpp")

        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        monkeypatch.setattr(
            llama_cpp,
            "_hf_unreachable",
            lambda: pytest.fail("probed despite an opt-out"),
        )
        with llama_cpp._hf_offline_if_unreachable() as engaged:
            assert engaged is False

    def test_falsey_transformers_offline_still_probes(self, monkeypatch):
        llama_cpp = pytest.importorskip("core.inference.llama_cpp")

        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
        calls = []
        monkeypatch.setattr(
            llama_cpp,
            "_hf_unreachable",
            lambda: (calls.append(1), False)[1],
        )
        with llama_cpp._hf_offline_if_unreachable() as engaged:
            assert engaged is False
        assert calls == [1]


class TestSlowLinkIsNotOffline:
    """A slow but reachable endpoint must not be quarantined: that would fail an uncached
    load that would otherwise merely be slow. A blackholed route must still be offline."""

    @pytest.fixture(autouse = True)
    def _direct_probe(self, clean_proxy_env):
        pass

    def _probe_raising(self, monkeypatch, exc):
        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(exc))
        from utils.transformers_version import hf_endpoint_unreachable

        return hf_endpoint_unreachable

    def test_timeout_with_tcp_egress_is_reachable(self, monkeypatch):
        import urllib.error

        probe = self._probe_raising(monkeypatch, urllib.error.URLError(TimeoutError("slow")))
        monkeypatch.setattr("utils.utils.hf_tcp_reachable", lambda *a, **k: True)
        assert probe(timeout = 1) is False

    def test_timeout_without_tcp_egress_is_offline(self, monkeypatch):
        import urllib.error

        probe = self._probe_raising(monkeypatch, urllib.error.URLError(TimeoutError("dead")))
        monkeypatch.setattr("utils.utils.hf_tcp_reachable", lambda *a, **k: False)
        assert probe(timeout = 1) is True

    def test_bare_timeout_error_takes_the_same_path(self, monkeypatch):
        probe = self._probe_raising(monkeypatch, TimeoutError("slow"))
        monkeypatch.setattr("utils.utils.hf_tcp_reachable", lambda *a, **k: True)
        assert probe(timeout = 1) is False

    def test_refused_connection_counts_as_egress(self, monkeypatch):
        """Something answered, so the network works even though nothing is listening."""
        import socket as _socket

        from utils.utils import hf_tcp_reachable

        def _refuse(*a, **k):
            raise ConnectionRefusedError()

        monkeypatch.setattr(_socket, "create_connection", _refuse)
        assert hf_tcp_reachable(1, "http://127.0.0.1:9") is True


class TestConcurrentGuardsHoldTheirOwnReference:
    """Overlapping requests must each hold a reference. If a later guard no-ops because an
    earlier one already set HF_HUB_OFFLINE, the earlier guard's exit restores the network
    while the later request is still resolving hub files."""

    def test_second_guard_engages_and_offline_survives_first_exit(
        self, monkeypatch, clean_offline_env
    ):
        import threading

        import core.inference.llama_cpp as lc

        monkeypatch.setattr(lc, "_hf_unreachable", lambda: True)

        a_in, b_in, a_done = threading.Event(), threading.Event(), threading.Event()
        seen: dict = {}

        def worker_a():
            with lc._hf_offline_if_unreachable():
                a_in.set()
                b_in.wait(5)
            a_done.set()

        def worker_b():
            a_in.wait(5)
            with lc._hf_offline_if_unreachable() as engaged:
                seen["engaged"] = engaged
                b_in.set()
                a_done.wait(5)
                seen["offline_after_a_exit"] = hf_constants.HF_HUB_OFFLINE
                seen["env_after_a_exit"] = os.environ.get("HF_HUB_OFFLINE")

        ta, tb = threading.Thread(target = worker_a), threading.Thread(target = worker_b)
        ta.start()
        tb.start()
        ta.join(20)
        tb.join(20)

        assert seen.get("engaged") is True, "second guard no-opped instead of taking a reference"
        assert seen.get("offline_after_a_exit") is True
        assert seen.get("env_after_a_exit") == "1"
        # Both windows closed -> fully restored.
        assert hf_constants.HF_HUB_OFFLINE is False
        assert "HF_HUB_OFFLINE" not in os.environ

    def test_user_set_offline_is_still_a_noop(self, monkeypatch, clean_offline_env):
        import core.inference.llama_cpp as lc

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(lc, "_hf_unreachable", lambda: True)
        with lc._hf_offline_if_unreachable() as engaged:
            assert engaged is False
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    def test_guard_ownership_cannot_race_the_environment_check(
        self, monkeypatch, clean_offline_env
    ):
        """A guard that starts just before another sets the env must retain its window."""
        import threading

        import core.inference.llama_cpp as lc
        import utils.utils as uu

        monkeypatch.setattr(lc, "_hf_unreachable", lambda: True)
        original_active = uu.force_hf_offline_active
        original_state = uu.force_hf_offline_state
        owner_entered = threading.Event()
        release_owner = threading.Event()
        owner_thread = None

        def owner():
            with uu.force_hf_offline():
                owner_entered.set()
                release_owner.wait(5)

        def start_owner_after(snapshot):
            nonlocal owner_thread
            if threading.current_thread() is threading.main_thread() and owner_thread is None:
                owner_thread = threading.Thread(target = owner)
                owner_thread.start()
                assert owner_entered.wait(5)
            return snapshot

        def stale_active_read():
            # Old implementation: depth is read before the owner sets the env.
            return start_owner_after(original_active())

        def atomic_state_then_owner_enters():
            # New implementation: ownership + env are one consistent snapshot. Even if
            # another owner enters immediately after it, hf_env_offline catches that and
            # this guard takes a reference instead of treating the env as user-owned.
            return start_owner_after(original_state())

        monkeypatch.setattr(uu, "force_hf_offline_active", stale_active_read)
        monkeypatch.setattr(uu, "force_hf_offline_state", atomic_state_then_owner_enters)
        try:
            with lc._hf_offline_if_unreachable() as engaged:
                assert engaged is True
        finally:
            release_owner.set()
            if owner_thread is not None:
                owner_thread.join(10)
        assert owner_thread is not None


class TestSpawnEnvironmentDoesNotInheritScopedOffline:
    """A scoped parent guard must not permanently quarantine a newly spawned worker."""

    def test_multiprocessing_spawn_window_uses_pre_guard_values(
        self, monkeypatch, clean_offline_env
    ):
        from utils.hf_cache_settings import child_environment_for_spawn
        from utils.utils import force_hf_offline
        with force_hf_offline():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            with child_environment_for_spawn({}):
                assert "HF_HUB_OFFLINE" not in os.environ
                assert "TRANSFORMERS_OFFLINE" not in os.environ
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_explicit_subprocess_environment_uses_pre_guard_values(
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        from utils.hf_cache_settings import HuggingFaceCachePaths
        from utils.utils import force_hf_offline

        paths = HuggingFaceCachePaths(tmp_path, tmp_path / "hub", tmp_path / "xet", "studio")
        with force_hf_offline():
            child_env = paths.child_env()
        assert "HF_HUB_OFFLINE" not in child_env
        assert "TRANSFORMERS_OFFLINE" not in child_env

    def test_nested_spawn_contexts_remain_reentrant(self, monkeypatch, clean_offline_env):
        from utils.hf_cache_settings import child_environment_for_spawn
        from utils.utils import force_hf_offline
        with force_hf_offline():
            with child_environment_for_spawn({}):
                with child_environment_for_spawn({}):
                    assert "HF_HUB_OFFLINE" not in os.environ
            assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_user_transformers_offline_keeps_child_hub_offline(
        self, monkeypatch, clean_offline_env
    ):
        from utils.hf_cache_settings import child_environment_for_spawn
        from utils.utils import force_hf_offline

        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        with force_hf_offline():
            with child_environment_for_spawn({}):
                assert os.environ.get("HF_HUB_OFFLINE") == "1"
                assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        assert "HF_HUB_OFFLINE" not in os.environ
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


class TestSpawnWindowKeepsTheParentOffline:
    """Restoring the user's env for a spawn must not un-offline the guarded parent."""

    def test_env_offline_still_true_inside_the_spawn_window(self, monkeypatch, clean_offline_env):
        from utils.hf_cache_settings import child_environment_for_spawn
        from utils.transformers_version import _env_offline
        from utils.utils import force_hf_offline, hf_env_offline

        with force_hf_offline():
            assert _env_offline() is True and hf_env_offline() is True
            with child_environment_for_spawn({}):
                # The child must inherit the user's own (online) intent ...
                assert "HF_HUB_OFFLINE" not in os.environ
                # ... while the parent's own gates stay closed.
                assert _env_offline() is True
                assert hf_env_offline() is True
            assert _env_offline() is True

    def test_guarded_metadata_fetch_stays_local_during_a_concurrent_spawn(
        self, monkeypatch, clean_offline_env
    ):
        """Guard on one thread, spawn window on a second, raw metadata read on a third.
        No request may leave the process."""
        import threading

        from utils.hf_cache_settings import child_environment_for_spawn
        from utils.utils import force_hf_offline

        from utils import transformers_version as tv

        calls: list = []
        monkeypatch.setattr(tv, "_adapter_base_from_hf_cache", lambda name: "cached/base")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("network")),
        )

        guard_open = threading.Event()
        release_guard = threading.Event()
        in_window = threading.Event()
        release_window = threading.Event()

        def _guard():
            with force_hf_offline():
                guard_open.set()
                release_guard.wait(30)

        def _spawn():
            guard_open.wait(30)
            with child_environment_for_spawn({}):
                in_window.set()
                release_window.wait(30)

        threads = [
            threading.Thread(target = _guard, daemon = True),
            threading.Thread(target = _spawn, daemon = True),
        ]
        for thread in threads:
            thread.start()
        try:
            assert in_window.wait(30)
            assert tv._remote_lora_base("some-org/some-lora") == "cached/base"
            assert calls == []
        finally:
            release_window.set()
            release_guard.set()
            for thread in threads:
                thread.join(30)


class TestProxyTimeoutIsNotExcused:
    """A live proxy proves the proxy is up, not that it can reach the hub."""

    @pytest.fixture(autouse = True)
    def _direct_probe(self, clean_proxy_env):
        pass

    def _probe_timing_out(self, monkeypatch):
        import urllib.error
        import urllib.request

        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError(TimeoutError("slow"))),
        )
        from utils.transformers_version import hf_endpoint_unreachable

        return hf_endpoint_unreachable

    def test_timeout_through_a_proxy_stays_unreachable(self, monkeypatch):
        probe = self._probe_timing_out(monkeypatch)
        monkeypatch.setattr("utils.utils.hf_proxy_configured", lambda: True)
        # Even if the proxy itself accepts TCP, the hub behind it may be blackholed.
        monkeypatch.setattr("utils.utils.hf_tcp_reachable", lambda *a, **k: True)
        assert probe(timeout = 1) is True

    def test_timeout_without_a_proxy_still_uses_tcp(self, monkeypatch):
        probe = self._probe_timing_out(monkeypatch)
        monkeypatch.setattr("utils.utils.hf_proxy_configured", lambda: False)
        monkeypatch.setattr("utils.utils.hf_tcp_reachable", lambda *a, **k: True)
        assert probe(timeout = 1) is False

    def test_lifetime_mode_fails_open_on_proxy_timeout(self, monkeypatch):
        probe = self._probe_timing_out(monkeypatch)
        monkeypatch.setattr("utils.utils.hf_proxy_configured", lambda: True)
        assert probe(timeout = 1, proxy_timeouts_offline = False) is False


class TestEndpointNormalisation:
    """An empty or whitespace HF_ENDPOINT must fall back to the default hub in BOTH the
    DNS shortcut and the HTTP probe, or the two stages disagree."""

    @pytest.fixture(autouse = True)
    def _direct_probe(self, clean_proxy_env):
        pass

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_blank_endpoint_falls_back(self, monkeypatch, value):
        from utils.utils import hf_endpoint_host, hf_endpoint_url

        monkeypatch.setenv("HF_ENDPOINT", value)
        assert hf_endpoint_host() == "huggingface.co"
        assert hf_endpoint_url() == "https://huggingface.co"

    def test_probe_uses_the_same_normalised_url(self, monkeypatch):
        seen: list = []
        import urllib.request

        def _capture(req, *a, **k):
            seen.append(req.full_url)
            raise urllib.error.URLError("stop")

        import urllib.error

        monkeypatch.setenv("HF_ENDPOINT", "   ")
        monkeypatch.setattr(urllib.request, "urlopen", _capture)
        from utils.transformers_version import hf_endpoint_unreachable

        hf_endpoint_unreachable(timeout = 1)
        assert seen and seen[0].startswith("https://huggingface.co"), seen


class TestIpv6Endpoint:
    def test_ipv6_literal_resolves(self):
        """gethostbyname is IPv4-only and would call an AAAA-only mirror dead."""
        from utils.utils import dns_host_dead
        assert dns_host_dead("::1", timeout = 2.0) is False

    def test_unresolvable_host_still_dead(self, monkeypatch):
        # Mock the resolver rather than trusting the runner's: an ISP or captive
        # portal that hijacks NXDOMAIN resolves .invalid and would fail this test on
        # a perfectly good build. Also saves a real 2s lookup per run.
        import socket as _socket

        def _nxdomain(*a, **k):
            raise _socket.gaierror(-2, "Name or service not known")

        monkeypatch.setattr(_socket, "getaddrinfo", _nxdomain)
        from utils.utils import dns_host_dead

        assert dns_host_dead("no-such-host.invalid", timeout = 2.0) is True


class TestGuardSkipsLocalPaths:
    """A local path never reaches the hub, so probing costs time and prevents nothing."""

    def test_local_path_is_a_noop(self, tmp_path, monkeypatch):
        from core.inference.llama_cpp import _hf_offline_if_unreachable_for

        called: list = []
        monkeypatch.setattr("utils.utils.hf_unreachable", lambda *a, **k: called.append(1) or True)
        with _hf_offline_if_unreachable_for(str(tmp_path / "model.gguf")) as engaged:
            assert engaged is None  # nullcontext yields None
        assert called == [], "probed the hub for a local path"

    def test_remote_id_still_guarded(self, monkeypatch, clean_offline_env):
        import core.inference.llama_cpp as lc
        monkeypatch.setattr(lc, "_hf_unreachable", lambda: True)
        with lc._hf_offline_if_unreachable_for("unsloth/Qwen3.5-4B-GGUF") as engaged:
            assert engaged is True
            assert os.environ.get("HF_HUB_OFFLINE") == "1"


class TestGatewayErrorsAreNotConnectionFailures:
    """Lifetime offline flags must not be set by a momentary 502/503/504."""

    @pytest.fixture(autouse = True)
    def _direct_probe(self, clean_proxy_env):
        pass

    def _probe_with(self, monkeypatch, exc):
        import urllib.request

        def _urlopen(*a, **k):
            raise exc

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        from utils.transformers_version import hf_endpoint_unreachable

        return hf_endpoint_unreachable

    @pytest.mark.parametrize("code", [502, 503, 504])
    def test_strict_mode_treats_gateway_error_as_reachable(self, monkeypatch, code):
        import urllib.error

        exc = urllib.error.HTTPError("u", code, "err", {}, None)
        probe = self._probe_with(monkeypatch, exc)
        assert probe(timeout = 1, gateway_errors_offline = False) is False
        # Default (scoped callers) keeps treating a downed hub as offline.
        assert probe(timeout = 1) is True

    @pytest.mark.parametrize("code", [401, 403, 404, 429])
    def test_other_http_errors_always_reachable(self, monkeypatch, code):
        import urllib.error

        exc = urllib.error.HTTPError("u", code, "err", {}, None)
        probe = self._probe_with(monkeypatch, exc)
        assert probe(timeout = 1) is False
        assert probe(timeout = 1, gateway_errors_offline = False) is False


class TestHfUnreachableProbe:
    """``utils.utils.hf_unreachable``: memoised, opt-outable, fails open."""

    @pytest.fixture(autouse = True)
    def _reset(self):
        from utils.utils import reset_hf_reachability_cache

        reset_hf_reachability_cache()
        yield
        reset_hf_reachability_cache()

    def _patch_probe(self, monkeypatch, result, calls):
        import utils.transformers_version as tv
        def _probe(*_a, **_k):
            calls.append(1)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(tv, "hf_endpoint_unreachable", _probe)

    def test_probes_once_then_memoises(self, monkeypatch, clean_offline_env):
        from utils.utils import hf_unreachable

        calls: list = []
        self._patch_probe(monkeypatch, True, calls)
        assert hf_unreachable() is True
        assert hf_unreachable() is True
        assert len(calls) == 1

    def test_shared_probe_keeps_gateway_errors_online(self, monkeypatch, clean_offline_env):
        from utils.utils import hf_unreachable
        import utils.transformers_version as tv

        seen = {}

        # **_kwargs so the sibling ambiguity flag does not turn this into a TypeError
        # that the guard's fail-open would swallow. Both flags are asserted by
        # TestSlowProxyDoesNotForceOffline.
        def _probe(
            timeout,
            *,
            gateway_errors_offline = True,
            **_kwargs,
        ):
            seen["gateway_errors_offline"] = gateway_errors_offline
            return gateway_errors_offline

        monkeypatch.setattr(tv, "hf_endpoint_unreachable", _probe)
        assert hf_unreachable() is False
        assert seen["gateway_errors_offline"] is False

    def test_opt_out_skips_probe(self, monkeypatch, clean_offline_env):
        from utils.utils import hf_unreachable

        calls: list = []
        self._patch_probe(monkeypatch, True, calls)
        monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "0")
        assert hf_unreachable() is False
        assert calls == []

    def test_probe_failure_reports_reachable(self, monkeypatch, clean_offline_env):
        from utils.utils import hf_unreachable

        calls: list = []
        self._patch_probe(monkeypatch, RuntimeError("boom"), calls)
        # Fail open: a broken probe must not strand a working install offline.
        assert hf_unreachable() is False

    def test_reset_forces_reprobe(self, monkeypatch, clean_offline_env):
        from utils.utils import hf_unreachable, reset_hf_reachability_cache

        calls: list = []
        self._patch_probe(monkeypatch, True, calls)
        assert hf_unreachable() is True
        reset_hf_reachability_cache()
        assert hf_unreachable() is True
        assert len(calls) == 2

    def test_memo_window_is_short_in_both_directions(self):
        """Stale either way is a bug: a stale 'reachable' hides the plug being pulled,
        a stale 'unreachable' fails a download after the user reconnects."""
        import utils.utils as uu
        assert uu._HF_REACHABILITY_TTL_S <= 10.0

    def test_verdict_expires_so_a_disconnect_is_noticed(self, monkeypatch, clean_offline_env):
        import time as _time

        import utils.utils as uu
        from utils.utils import hf_unreachable

        monkeypatch.setattr(uu, "_HF_REACHABILITY_TTL_S", 0.2)
        verdict = {"value": False}
        monkeypatch.setattr(
            __import__("utils.transformers_version", fromlist = ["x"]),
            "hf_endpoint_unreachable",
            lambda *a, **k: verdict["value"],
        )
        assert hf_unreachable() is False  # online during the download
        verdict["value"] = True  # plug pulled
        _time.sleep(0.3)
        assert hf_unreachable() is True


class TestExtractQuantLabelSubdir:
    """``_extract_quant_label`` must consider parent dirs when the basename has
    no quant token (subdir layouts like ``BF16/foo.gguf``)."""

    def test_quant_in_basename_unchanged(self):
        assert _extract_quant_label("BF16/foo-BF16.gguf") == "BF16"
        assert _extract_quant_label("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_quant_only_in_parent_dir(self):
        assert _extract_quant_label("BF16/foo.gguf") == "BF16"

    def test_ud_prefix_in_parent_dir(self):
        assert _extract_quant_label("UD-Q4_K_XL/weight.gguf") == "UD-Q4_K_XL"

    def test_deeper_nesting_picks_nearest_quant_dir(self):
        # Multiple matching parents: prefer the innermost (closest to the file).
        assert _extract_quant_label("models/MXFP4_MOE/foo.gguf") == "MXFP4_MOE"


class TestDownloadMmprojOfflineCacheFallback:
    """``_download_mmproj`` must resolve cached mmproj GGUFs offline, like
    ``_download_gguf``; else the offline vision load returns None despite a cache hit."""

    def test_cache_lookup_returns_cached_mmproj_when_list_repo_files_fails(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/vision-GGUF",
            {
                "vision-Q4_K_M.gguf": 1,
                "mmproj-vision-F16.gguf": 1,
            },
        )
        backend = LlamaCppBackend()

        def boom_list(*a, **k):
            raise OSError("offline")

        def fake_download(
            repo_id,
            filename,
            token = None,
            **kwargs,
        ):
            # Echo back so the test can verify the cache-resolved filename
            return f"/fake/cache/{repo_id}/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", boom_list),
            patch(
                "core.inference.llama_cpp.hf_hub_download_with_xet_fallback",
                fake_download,
            ),
        ):
            out = backend._download_mmproj(
                hf_repo = "unsloth/vision-GGUF",
                hf_token = None,
            )
        assert out is not None, "mmproj must resolve from cache when offline"
        assert "mmproj-vision-F16.gguf" in out

    def test_prefers_f16_variant_when_multiple_mmproj_in_cache(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/vision-GGUF",
            {
                "mmproj-vision-BF16.gguf": 1,
                "mmproj-vision-F16.gguf": 1,
            },
        )
        backend = LlamaCppBackend()

        def boom_list(*a, **k):
            raise OSError("offline")

        captured = {}

        def fake_download(
            repo_id,
            filename,
            token = None,
            **kwargs,
        ):
            captured["filename"] = filename
            return f"/fake/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", boom_list),
            patch(
                "core.inference.llama_cpp.hf_hub_download_with_xet_fallback",
                fake_download,
            ),
        ):
            backend._download_mmproj(
                hf_repo = "unsloth/vision-GGUF",
                hf_token = None,
            )
        assert captured.get("filename") == "mmproj-vision-F16.gguf"

    def test_no_mmproj_in_cache_returns_none(self, hf_cache):
        _build_cache(
            hf_cache,
            "unsloth/text-only-GGUF",
            {"text-Q4_K_M.gguf": 1},
        )
        backend = LlamaCppBackend()

        def boom_list(*a, **k):
            raise OSError("offline")

        with patch("huggingface_hub.list_repo_files", boom_list):
            out = backend._download_mmproj(
                hf_repo = "unsloth/text-only-GGUF",
                hf_token = None,
            )
        assert out is None


class TestListLocalGgufVariantsSubdir:
    """Subdir layouts like ``BF16/foo.gguf`` and ``Q4_K_M/foo.gguf`` must
    yield distinct quant labels, not collapse on basename."""

    def test_two_subdir_variants_do_not_collapse(self, tmp_path):
        from utils.models.model_config import list_local_gguf_variants

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "BF16").mkdir()
        (tmp_path / "BF16" / "foo.gguf").write_bytes(b"\0" * 100)
        (tmp_path / "Q4_K_M").mkdir()
        (tmp_path / "Q4_K_M" / "foo.gguf").write_bytes(b"\0" * 50)

        variants, _ = list_local_gguf_variants(str(tmp_path))
        quants = {v.quant for v in variants}
        assert "BF16" in quants, f"BF16 missing from {quants}"
        assert "Q4_K_M" in quants, f"Q4_K_M missing from {quants}"
        assert len(variants) == 2

    def test_find_local_gguf_by_variant_locates_subdir(self, tmp_path):
        from utils.models.model_config import _find_local_gguf_by_variant

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "BF16").mkdir()
        target = tmp_path / "BF16" / "foo.gguf"
        target.write_bytes(b"\0" * 10)

        out = _find_local_gguf_by_variant(str(tmp_path), "BF16")
        assert out is not None
        assert Path(out).name == "foo.gguf"

    def test_find_local_gguf_by_variant_ignores_big_endian_sibling(self, tmp_path):
        from utils.models.model_config import _find_local_gguf_by_variant

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model-Q4_K_M-be.gguf").write_bytes(b"\0" * 10)
        target = tmp_path / "model-Q4_K_M.gguf"
        target.write_bytes(b"\0" * 20)

        out = _find_local_gguf_by_variant(str(tmp_path), "Q4_K_M")
        assert out == str(target.absolute())

    def test_find_local_gguf_by_variant_skips_big_endian_only_match(self, tmp_path):
        from utils.models.model_config import _find_local_gguf_by_variant

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model-Q4_K_M-be.gguf").write_bytes(b"\0" * 10)

        assert _find_local_gguf_by_variant(str(tmp_path), "Q4_K_M") is None

    def test_find_local_gguf_by_variant_keeps_split_symlink_name(self, tmp_path):
        from utils.models.model_config import _find_local_gguf_by_variant

        blobs = tmp_path / "blobs"
        blobs.mkdir()
        snap = tmp_path / "snapshots" / "rev" / "BF16"
        snap.mkdir(parents = True)
        (tmp_path / "snapshots" / "rev" / "config.json").write_text("{}")
        for i, sha in enumerate(("aa" * 32, "bb" * 32), start = 1):
            (blobs / sha).write_bytes(b"\0" * 10)
            _symlink_or_skip(snap / f"model-BF16-0000{i}-of-00002.gguf", blobs / sha)

        out = _find_local_gguf_by_variant(str(tmp_path / "snapshots" / "rev"), "BF16")
        assert out is not None
        assert Path(out).name == "model-BF16-00001-of-00002.gguf"

    def test_detect_gguf_model_keeps_split_symlink_name(self, tmp_path):
        from utils.models.model_config import detect_gguf_model

        blobs = tmp_path / "blobs"
        blobs.mkdir()
        snap = tmp_path / "snapshots" / "rev"
        snap.mkdir(parents = True)
        for i, (sha, size) in enumerate((("cc" * 32, 10), ("dd" * 32, 20)), start = 1):
            (blobs / sha).write_bytes(b"\0" * size)
            _symlink_or_skip(snap / f"model-BF16-0000{i}-of-00002.gguf", blobs / sha)

        out = detect_gguf_model(str(snap))
        assert out is not None
        assert Path(out).name == "model-BF16-00001-of-00002.gguf"

    def test_lone_split_symlink_uses_colocated_target_shards(self, tmp_path):
        from utils.models.model_config import _find_local_gguf_by_variant, detect_gguf_model

        target_dir = tmp_path / "external" / "BF16"
        target_dir.mkdir(parents = True)
        target = target_dir / "model-BF16-00001-of-00002.gguf"
        target.write_bytes(b"\0" * 10)
        (target_dir / "model-BF16-00002-of-00002.gguf").write_bytes(b"\0" * 10)

        local = tmp_path / "local"
        local.mkdir()
        (local / "config.json").write_text("{}")
        link = local / target.name
        _symlink_or_skip(link, target)

        expected = str(target.absolute())
        assert _find_local_gguf_by_variant(str(local), "BF16") == expected
        assert detect_gguf_model(str(local)) == expected
        assert detect_gguf_model(str(link)) == expected

    def test_model_config_variant_ignores_big_endian_sibling(self, tmp_path):
        from utils.models.model_config import ModelConfig

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model-Q4_K_M-be.gguf").write_bytes(b"\0" * 10)
        target = tmp_path / "model-Q4_K_M.gguf"
        target.write_bytes(b"\0" * 20)

        config = ModelConfig.from_identifier(str(tmp_path), gguf_variant = "Q4_K_M")
        assert config is not None
        assert config.gguf_file == str(target.resolve())

    def test_model_config_direct_split_gguf_keeps_file_when_variant_is_echoed(self, tmp_path):
        """A settings reload sends status.gguf_variant with the direct file path."""
        from utils.models.model_config import ModelConfig

        first = tmp_path / "Model-UD-Q3_K_M-00001-of-00002.gguf"
        second = tmp_path / "Model-UD-Q3_K_M-00002-of-00002.gguf"
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        config = ModelConfig.from_identifier(str(first), gguf_variant = "UD-Q3_K_M")

        assert config is not None
        assert config.is_gguf is True
        assert config.gguf_file == str(first.resolve())

    def test_local_variant_listing_keeps_subdir_be_model_name_token(self, tmp_path):
        from utils.models.model_config import list_local_gguf_variants

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "Q4_K_M").mkdir()
        (tmp_path / "Q4_K_M" / "foo-be.gguf").write_bytes(b"\0" * 10)

        variants, _ = list_local_gguf_variants(str(tmp_path))
        assert [(v.quant, v.filename, v.size_bytes) for v in variants] == [
            ("Q4_K_M", "Q4_K_M/foo-be.gguf", 10)
        ]


class TestListGgufVariantsPermanentErrors:
    """Permanent HF errors must surface; cache fallback only on transient."""

    def test_repository_not_found_re_raises(self, hf_cache, clean_offline_env):
        from utils.models.model_config import list_gguf_variants

        _build_cache(hf_cache, "u/repo-gguf", {"foo-Q4_K_M.gguf": 1})

        class _RepoNotFound(Exception):
            pass

        _RepoNotFound.__name__ = "RepositoryNotFoundError"

        def boom(*a, **k):
            raise _RepoNotFound("repo deleted")

        with patch("huggingface_hub.model_info", boom):
            with pytest.raises(Exception) as exc_info:
                list_gguf_variants("u/repo-gguf")
        assert type(exc_info.value).__name__ == "RepositoryNotFoundError"

    def test_gated_repo_re_raises(self, hf_cache, clean_offline_env):
        from utils.models.model_config import list_gguf_variants

        _build_cache(hf_cache, "u/gated-gguf", {"foo-Q4_K_M.gguf": 1})

        class _GatedRepo(Exception):
            pass

        _GatedRepo.__name__ = "GatedRepoError"

        def boom(*a, **k):
            raise _GatedRepo("auth required")

        with patch("huggingface_hub.model_info", boom):
            with pytest.raises(Exception) as exc_info:
                list_gguf_variants("u/gated-gguf")
        assert type(exc_info.value).__name__ == "GatedRepoError"

    def test_transient_error_still_falls_back_to_cache(self, hf_cache, clean_offline_env):
        from utils.models.model_config import list_gguf_variants

        _build_cache(hf_cache, "u/transient-gguf", {"foo-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            variants, _ = list_gguf_variants("u/transient-gguf")
        assert any(v.quant == "Q4_K_M" for v in variants)


class TestDetectGgufFromCacheExcludesMmproj:
    """A partial cache with only a vision projector must not route it as
    the main model."""

    def test_mmproj_only_returns_none(self, hf_cache):
        from utils.models.model_config import _detect_gguf_from_hf_cache
        _build_cache(
            hf_cache,
            "u/vision-only-mmproj",
            {"mmproj-vision-F16.gguf": 1},
        )
        assert _detect_gguf_from_hf_cache("u/vision-only-mmproj") is None

    def test_main_plus_mmproj_returns_main(self, hf_cache):
        from utils.models.model_config import _detect_gguf_from_hf_cache

        _build_cache(
            hf_cache,
            "u/vision-full",
            {
                "model-Q4_K_M.gguf": 1,
                "mmproj-vision-F16.gguf": 1,
            },
        )
        out = _detect_gguf_from_hf_cache("u/vision-full")
        assert out is not None
        assert "mmproj" not in out.lower()


class TestProbeDnsDeadNoGlobalTimeoutMutation:
    """``_probe_dns_dead`` must not change ``socket.setdefaulttimeout`` process-wide;
    concurrent sockets would inherit it during the probe window."""

    def test_default_timeout_unchanged_when_dns_up(self, monkeypatch):
        import socket as _socket
        from core.inference.llama_cpp import _probe_dns_dead

        prev = _socket.getdefaulttimeout()
        set_calls = []

        original_set = _socket.setdefaulttimeout

        def tracking_set(value):
            set_calls.append(value)
            original_set(value)

        monkeypatch.setattr(_socket, "setdefaulttimeout", tracking_set)
        # The probe resolves with getaddrinfo, not the IPv4-only gethostbyname, so
        # patching the latter left this test doing a real lookup and never exercising
        # the "DNS up" branch it is named for.
        monkeypatch.setattr(
            _socket,
            "getaddrinfo",
            lambda *a, **k: [(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
        )

        try:
            _probe_dns_dead("example.invalid", timeout = 0.5)
        finally:
            # Restore exact state regardless of test-side mutation.
            original_set(prev)

        assert set_calls == [], (
            f"_probe_dns_dead mutated socket.setdefaulttimeout {set_calls}; "
            "must isolate timeout to the probe thread"
        )

    def test_wedged_resolver_is_inconclusive_not_dead(self, monkeypatch):
        """A missed deadline must not take the offline shortcut.

        Slow-but-working DNS resolves past 2s, and this shortcut skips the fail-open
        reachability probe, so calling it dead strands a working machine offline for a
        whole job. A genuinely wedged resolver is still caught downstream, by the HEAD
        probe hanging on the same lookup.
        """
        import socket as _socket
        from core.inference.llama_cpp import _probe_dns_dead

        # Patch getaddrinfo, which is what the probe calls; patching gethostbyname made
        # this pass vacuously off the real NXDOMAIN for .invalid.
        def wedged(*a, **k):
            import threading
            threading.Event().wait()

        monkeypatch.setattr(_socket, "getaddrinfo", wedged)
        assert _probe_dns_dead("example.invalid", timeout = 0.1) is False

    def test_slow_but_resolving_dns_is_not_dead(self, monkeypatch):
        """The case the deadline used to misread: an answer that arrives after it."""
        import socket as _socket
        import time as _time
        from utils.utils import dns_host_dead

        def slow(*a, **k):
            _time.sleep(0.3)
            return [(2, 1, 6, "", ("93.184.216.34", 443))]

        monkeypatch.setattr(_socket, "getaddrinfo", slow)
        assert dns_host_dead("slow.example.test", timeout = 0.1) is False

    def test_nxdomain_is_still_dead(self, monkeypatch):
        """The reported bug's shortcut must keep working: a real resolver error."""
        import socket as _socket
        from utils.utils import dns_host_dead

        def nxdomain(*a, **k):
            raise _socket.gaierror("Name or service not known")

        monkeypatch.setattr(_socket, "getaddrinfo", nxdomain)
        assert dns_host_dead("no-such-host.invalid", timeout = 2.0) is True


class TestWaitForHealthRetriesOnReadError:
    """A TCP RST mid-read while llama-server is still binding (Windows: WinError
    10054) must not abort the health-poll loop and mask warmup as a fatal load."""

    def test_read_error_then_success(self, monkeypatch):
        import httpx

        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._port = 65500

        class _FakeProc:
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout = None):
                return 0

        backend._process = _FakeProc()
        backend._stdout_thread = None
        backend._stdout_lines = []

        calls = {"n": 0}

        def fake_get(
            url,
            timeout = None,
            trust_env = None,
        ):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadError("WinError 10054")
            if calls["n"] == 2:
                raise httpx.RemoteProtocolError("short read")
            if calls["n"] == 3:
                raise httpx.WriteError("peer dropped")

            class _OK:
                status_code = 200

            return _OK()

        monkeypatch.setattr("core.inference.llama_cpp.httpx.get", fake_get)
        assert backend._wait_for_health(timeout = 5.0, interval = 0.01) is True
        assert calls["n"] == 4, (
            f"_wait_for_health should retry past ReadError/RemoteProtocol/Write; "
            f"saw {calls['n']} attempts"
        )

    def test_real_process_exit_still_short_circuits(self, monkeypatch):
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend()
        backend._port = 65501

        class _DeadProc:
            returncode = 137

            def poll(self):
                return 137

            def terminate(self):
                pass

            def kill(self):
                pass

            def wait(self, timeout = None):
                return 137

        backend._process = _DeadProc()
        backend._stdout_thread = None
        backend._stdout_lines = ["fatal: out of memory"]
        assert backend._wait_for_health(timeout = 5.0, interval = 0.01) is False


class TestProxyDetectionWithoutRequests:
    """huggingface_hub 1.x moved to httpx and dropped requests. Proxy selection must
    not silently answer "no proxy" there: hf_dns_dead stands down for proxies, so
    going blind forces a working proxy-only machine offline."""

    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env):
        pass

    def _without_requests(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def _fail(name, *a, **k):
            if name == "requests.utils" or name.split(".")[0] == "requests":
                raise ImportError("No module named 'requests'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _fail)

    def test_env_proxy_still_found_without_requests(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        self._without_requests(monkeypatch)
        from utils.utils import hf_proxy_for_endpoint

        assert hf_proxy_for_endpoint("https://huggingface.co") == "http://proxy.internal:3128"

    def test_all_proxy_still_found_without_requests(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        self._without_requests(monkeypatch)
        from utils.utils import hf_proxy_for_endpoint

        assert hf_proxy_for_endpoint("https://huggingface.co") == "http://proxy.internal:3128"

    def test_no_proxy_still_bypasses_without_requests(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", "huggingface.co")
        self._without_requests(monkeypatch)
        from utils.utils import hf_proxy_for_endpoint

        assert hf_proxy_for_endpoint("https://huggingface.co") is None

    def test_dns_shortcut_stands_down_without_requests(self, monkeypatch):
        monkeypatch.setenv("ALL_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("HF_ENDPOINT", "https://does-not-resolve.invalid")
        self._without_requests(monkeypatch)
        from utils.utils import hf_dns_dead

        assert hf_dns_dead(timeout = 1.0) is False


class TestSocksProxyIsNotEgressEvidence:
    """urllib cannot route through socks5://, so its instant failure is not proof the
    hub is unreachable -- the Hub client reaches it through that proxy fine."""

    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env):
        pass

    def test_socks_scheme_is_not_usable_by_urllib(self):
        from utils.utils import hf_proxy_usable_by_urllib

        assert hf_proxy_usable_by_urllib("socks5://127.0.0.1:1080") is False
        assert hf_proxy_usable_by_urllib("socks5h://127.0.0.1:1080") is False
        assert hf_proxy_usable_by_urllib("http://127.0.0.1:3128") is True
        assert hf_proxy_usable_by_urllib("https://127.0.0.1:3128") is True
        assert hf_proxy_usable_by_urllib(None) is True

    def test_socks_proxy_reports_reachable(self, monkeypatch):
        import urllib.request

        monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")

        def _boom(*a, **k):
            raise AssertionError("must not attempt urlopen through a socks proxy")

        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        from utils.transformers_version import hf_endpoint_unreachable

        assert hf_endpoint_unreachable(1) is False


class TestProbeFailsOpenOnNonNetworkErrors:
    """A malformed endpoint or a bug in the probe is not an answer about the network.
    Classifying it as offline silently quarantines every load."""

    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env):
        pass

    @pytest.mark.parametrize("exc", [ValueError("bad url"), TypeError("bad port"), MemoryError()])
    def test_non_socket_exception_is_inconclusive(self, monkeypatch, exc):
        import urllib.request

        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(exc),
        )
        from utils.transformers_version import hf_endpoint_unreachable

        assert hf_endpoint_unreachable(1) is False

    def test_client_side_urlerror_is_inconclusive(self, monkeypatch):
        """urllib reports client-side problems as URLError with a plain string reason
        ("no host given", "unknown url type"). That is not an answer about egress."""
        import urllib.error
        import urllib.request

        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("no host given")),
        )
        from utils.transformers_version import hf_endpoint_unreachable

        assert hf_endpoint_unreachable(1) is False

    def test_socket_reason_urlerror_is_still_offline(self, monkeypatch):
        import socket
        import urllib.error
        import urllib.request

        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(
                urllib.error.URLError(socket.gaierror(-2, "Name or service not known"))
            ),
        )
        from utils.transformers_version import hf_endpoint_unreachable

        assert hf_endpoint_unreachable(1) is True

    def test_socket_error_is_still_offline(self, monkeypatch):
        import urllib.request

        monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("network unreachable")),
        )
        from utils.transformers_version import hf_endpoint_unreachable

        assert hf_endpoint_unreachable(1) is True

    def test_undeterminable_connect_target_fails_open(self, monkeypatch):
        from utils.utils import hf_tcp_reachable
        monkeypatch.setenv("HF_ENDPOINT", "https://")
        assert hf_tcp_reachable(1.0, "https://") is True


class TestExplicitBaseEnvIsAlsoScrubbed:
    """child_env(base=...) is what the vision sidecar builds, from a raw os.environ
    copy, so it has to lose the scoped offline flags too."""

    def test_explicit_base_loses_scoped_offline(self, monkeypatch, tmp_path):
        from utils.utils import force_hf_offline
        from utils.hf_cache_settings import get_hf_cache_paths
        from utils.native_path_leases import child_env_without_native_path_secret

        monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
        paths = get_hf_cache_paths()
        with force_hf_offline():
            env = paths.child_env(child_env_without_native_path_secret())
            assert env.get("HF_HUB_OFFLINE") is None
            assert env.get("TRANSFORMERS_OFFLINE") is None
            assert paths.child_env(dict(os.environ)).get("HF_HUB_OFFLINE") is None

    def test_user_set_offline_still_reaches_the_child(self, monkeypatch):
        from utils.utils import force_hf_offline
        from utils.hf_cache_settings import get_hf_cache_paths

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        paths = get_hf_cache_paths()
        with force_hf_offline():
            assert paths.child_env(dict(os.environ)).get("HF_HUB_OFFLINE") == "1"


class TestHttpsProxyDefaultPort:
    """An https:// proxy with no explicit port listens on 443, not 80."""

    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env):
        pass

    def test_https_proxy_defaults_to_443(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "https://proxy.internal")
        from utils.utils import hf_connect_target
        assert list(hf_connect_target("https://huggingface.co")) == ["proxy.internal", 443]

    def test_http_proxy_defaults_to_80(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal")
        from utils.utils import hf_connect_target
        assert list(hf_connect_target("https://huggingface.co")) == ["proxy.internal", 80]


class TestMetadataReadsUseTheHubProxy:
    """The reachability probe and the metadata reads must agree about egress.

    urllib's default opener ignores ALL_PROXY: ``getproxies`` reports it under an ``all``
    key, but ``ProxyHandler`` only ever dispatches ``<scheme>_open``. huggingface_hub 0.36
    is built on requests, which does honour it. So on a proxy-only machine the probe
    reported "online" through the proxy while every raw config.json / tokenizer_config.json
    / adapter_config.json read went direct and silently failed, dropping the sidecar tier
    decision back to name matching. Served by a real loopback proxy, not a mock.
    """

    @pytest.fixture(autouse = True)
    def _no_ambient_proxy(self, clean_proxy_env, clean_offline_env):
        pass

    @pytest.fixture
    def stub_proxy(self, monkeypatch, tmp_path):
        """A real HTTP proxy on loopback; yields (seen_requests, payload)."""
        import json as _json
        import threading as _threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        seen: list[tuple[str, str]] = []
        payload = {"architectures": ["MinistralForCausalLM"], "model_type": "ministral"}

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802
                seen.append((self.command, self.path))
                body = _json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _threading.Thread(target = srv.serve_forever, daemon = True).start()
        # Proxy-only egress: the hub name never resolves locally (.invalid, RFC 2606).
        monkeypatch.setenv("HF_ENDPOINT", "http://hub.invalid")
        monkeypatch.setenv("ALL_PROXY", f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        try:
            yield seen, payload
        finally:
            srv.shutdown()

    def _clear_caches(self, monkeypatch):
        import urllib.request

        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "_config_json_cache", {})
        monkeypatch.setattr(tv, "_tokenizer_class_cache", {})
        # urlopen builds its default opener once per process and caches it in _opener, so
        # an earlier test's proxy env would otherwise decide this one's routing.
        monkeypatch.setattr(urllib.request, "_opener", None)

    def test_all_proxy_is_ignored_by_the_default_opener(self, monkeypatch):
        """Guards the premise: urllib alone would never use this proxy."""
        import urllib.request

        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:3128")
        assert urllib.request.getproxies().get("all") == "http://127.0.0.1:3128"
        handler = urllib.request.ProxyHandler(urllib.request.getproxies())
        assert not hasattr(handler, "https_open")
        assert not hasattr(handler, "http_open")

    def test_config_json_read_goes_through_the_proxy(self, stub_proxy, monkeypatch):
        seen, payload = stub_proxy
        self._clear_caches(monkeypatch)
        from utils.transformers_version import _load_config_json

        assert _load_config_json("acme/ministral-3b") == payload
        assert seen == [("GET", "http://hub.invalid/acme/ministral-3b/resolve/main/config.json")]

    def test_tokenizer_config_read_goes_through_the_proxy(self, stub_proxy, monkeypatch):
        seen, _ = stub_proxy
        self._clear_caches(monkeypatch)
        from utils.transformers_version import _check_tokenizer_config_needs_v5

        _check_tokenizer_config_needs_v5("acme/ministral-3b")
        assert seen == [
            ("GET", "http://hub.invalid/acme/ministral-3b/resolve/main/tokenizer_config.json"),
        ]

    def test_adapter_config_read_goes_through_the_proxy(self, stub_proxy, monkeypatch):
        seen, _ = stub_proxy
        self._clear_caches(monkeypatch)
        from utils.transformers_version import _remote_lora_base

        _remote_lora_base("acme/ministral-3b")
        assert seen == [
            ("GET", "http://hub.invalid/acme/ministral-3b/resolve/main/adapter_config.json"),
        ]

    def test_no_proxy_bypass_keeps_the_metadata_read_direct(self, monkeypatch, tmp_path):
        """A CIDR NO_PROXY entry requests honours and urllib does not must not be
        re-routed through the proxy by the metadata reads either."""
        import json as _json
        import threading as _threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        proxy_seen: list = []
        hub_seen: list = []
        payload = {"model_type": "ministral"}

        def _make(log, body_bytes):
            class _H(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.0"

                def do_GET(self):  # noqa: N802
                    log.append((self.command, self.path))
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body_bytes)))
                    self.end_headers()
                    self.wfile.write(body_bytes)

                def log_message(self, *args):
                    pass

            return _H

        body = _json.dumps(payload).encode()
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), _make(proxy_seen, body))
        hub = ThreadingHTTPServer(("127.0.0.1", 0), _make(hub_seen, body))
        for srv in (proxy, hub):
            _threading.Thread(target = srv.serve_forever, daemon = True).start()
        try:
            monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hub.server_address[1]}")
            monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_address[1]}")
            monkeypatch.setenv("NO_PROXY", "127.0.0.0/8")
            monkeypatch.setenv("HF_HOME", str(tmp_path))
            self._clear_caches(monkeypatch)
            from utils.transformers_version import _load_config_json

            assert _load_config_json("acme/ministral-3b") == payload
            assert proxy_seen == []
            assert hub_seen == [("GET", "/acme/ministral-3b/resolve/main/config.json")]
        finally:
            proxy.shutdown()
            hub.shutdown()


class TestSlowProxyDoesNotForceOffline:
    """A functional-but-slow proxy is ambiguous, so the shared guard must fail open.

    Otherwise an uncached load through a corporate proxy whose HEAD exceeds the probe
    timeout is turned cache-only, and the hub client's longer request never runs. The
    worker already passes proxy_timeouts_offline=False; the route guard must agree.
    """

    @pytest.fixture(autouse = True)
    def _fresh(self, monkeypatch):
        from utils.utils import reset_hf_reachability_cache

        reset_hf_reachability_cache()
        monkeypatch.delenv("UNSLOTH_OFFLINE_PROBE", raising = False)
        yield
        reset_hf_reachability_cache()

    def test_shared_guard_passes_both_ambiguity_flags_off(self, monkeypatch):
        seen = {}

        def _probe(timeout, **kwargs):
            seen.update(kwargs)
            return False

        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "hf_endpoint_unreachable", _probe)
        from utils.utils import hf_unreachable

        assert hf_unreachable(timeout = 1) is False
        assert seen == {"gateway_errors_offline": False, "proxy_timeouts_offline": False}

    def test_slow_proxy_reads_reachable_not_offline(self, monkeypatch):
        """End to end through the real classifier: a clean timeout behind a proxy."""
        import utils.transformers_version as tv

        monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising = False)

        class _SlowOpener:
            def open(
                self,
                req,
                timeout = None,
            ):
                raise TimeoutError("proxy is up, upstream is slow")

        monkeypatch.setattr(tv, "_hf_proxy_opener", lambda _url: _SlowOpener())
        # Sanity: the same input IS offline with the flag on, so the flag is what moves it.
        assert tv.hf_endpoint_unreachable(1, proxy_timeouts_offline = True) is True

        from utils.utils import hf_unreachable

        assert hf_unreachable(timeout = 1) is False


class TestValidateGuardCoversMetadataPreflights:
    """/validate resolves the config under the guard, then runs more remote reads.

    Those preflights (upgrade check, trust-remote-code, sizing, training guard) each
    fetch raw metadata, so leaving them outside the window just moves the stall.
    """

    def test_every_remote_preflight_on_validate_is_wrapped(self):
        """AST check: no bare await asyncio.to_thread(<remote preflight>, ...) remains."""
        import ast
        import pathlib

        # Anchored on __file__: CI runs pytest from the repo root, not studio/backend.
        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
        tree = ast.parse(src)
        remote = {
            "check_upgrade_for_model",
            "latest_tier_active_for",
            "get_base_model_from_lora_identifier",
            "_guard_chat_load_against_training",
        }
        bare = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "to_thread"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            name = getattr(first, "id", None) or getattr(first, "attr", None)
            if name in remote:
                bare.append((name, node.lineno))
        assert bare == [], f"unguarded remote preflight(s): {bare}"


class TestGuardIsKeyedOnWhatIsRead:
    """A LOCAL adapter path can resolve to a REMOTE base, and the base is what gets
    fetched. Keying the guard on the outer request would hand back a null context while
    the preflight reads the hub."""

    def test_predicate_contract(self):
        """_any_remote: local-only is False, anything unknown or remote is True."""
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
        tree = ast.parse(src)
        fn = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_any_remote"
        )
        ns: dict = {}
        # is_local_path is imported inside the body; give it a stub module to import from.
        exec(compile(ast.Module([fn], []), "<any_remote>", "exec"), ns)
        import sys
        import types

        stub = types.ModuleType("utils.paths")
        stub.is_local_path = lambda p: p.startswith("/local")
        saved = sys.modules.get("utils.paths")
        sys.modules["utils.paths"] = stub
        try:
            any_remote = ns["_any_remote"]
            assert any_remote("/local/adapter") is False
            assert any_remote("org/model") is True
            assert any_remote(("/local/adapter", "org/base")) is True
            assert any_remote(("/local/a", "/local/b")) is False
            assert any_remote(None) is False
            assert any_remote(()) is False
            # A falsy entry means there is no base to read, not an unknown one: a local
            # model whose config carries base_model=None must not pay the probe.
            assert any_remote((None,)) is False
            assert any_remote(("/local/adapter", None)) is False
            assert any_remote(("/local/adapter", "", "org/base")) is True
            assert any_remote((123,)) is True  # unresolvable: guard anyway
        finally:
            if saved is None:
                del sys.modules["utils.paths"]
            else:
                sys.modules["utils.paths"] = saved

    def test_every_guarded_call_passes_what_it_reads(self):
        """AST check: no _offline_guarded call reads config.identifier or a security
        target while keying the guard on the bare outer identifier."""
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
        tree = ast.parse(src)
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "to_thread"):
                continue
            names = [getattr(a, "id", None) or getattr(a, "attr", None) for a in node.args]
            if names[:1] != ["_offline_guarded"] or len(node.args) < 3:
                continue
            targets, read = node.args[1], node.args[2]
            read_name = getattr(read, "id", None) or getattr(read, "attr", None)
            # A read of the resolved config must not be keyed on model_identifier alone.
            if read_name in ("latest_tier_active_for", "_guard_chat_load_against_training"):
                if isinstance(targets, ast.Name):
                    bad.append((read_name, node.lineno))
            if read_name == "check_upgrade_for_model":
                if getattr(targets, "id", None) == "model_identifier":
                    bad.append((read_name, node.lineno))
        assert bad == [], f"guard keyed on the wrong target: {bad}"


class TestTransformersOfflineDoesNotSilenceDatasets:
    """TRANSFORMERS_OFFLINE=1 asks for cached MODEL files. Deriving HF_DATASETS_OFFLINE
    from it fails an uncached hf_dataset for the whole job on a machine with egress."""

    def _worker_block(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "core" / "training" / "worker.py").read_text(encoding = "utf-8")
        start = src.index("# Offline auto-detect:")
        return src[start : start + 2200]

    def test_datasets_offline_is_gated_on_the_network_verdict(self):
        block = self._worker_block()
        assert "if _network_offline:" in block
        assert block.index("if _network_offline:") < block.index("HF_DATASETS_OFFLINE")

    def test_env_requested_offline_does_not_set_network_offline(self):
        block = self._worker_block()
        # hf_env_offline() feeds _offline only; the two probes feed _network_offline too.
        assert "_offline = hf_env_offline()" in block
        assert "_offline = _network_offline = hf_dns_dead()" in block
        assert "_offline = _network_offline = hf_endpoint_unreachable(" in block

    def test_hub_and_transformers_offline_are_still_set_either_way(self):
        block = self._worker_block()
        hub = block.index('os.environ["HF_HUB_OFFLINE"] = "1"')
        gate = block.index("if _network_offline:")
        assert hub < gate, "HF_HUB_OFFLINE must not be gated on the network verdict"


class TestModelConfigPredicateHonoursTheWindow:
    """model_config._env_offline gates detect_audio_type's raw requests.get, which the
    patched hub constant does not cover. During hf_environment_restored_for_spawn the env
    is back to the user's values, so an env-only predicate would fire that 15s request."""

    def test_predicate_is_true_inside_the_spawn_window(self, monkeypatch):
        import threading

        from utils.models.model_config import _env_offline
        from utils.utils import force_hf_offline, hf_environment_restored_for_spawn

        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            monkeypatch.delenv(key, raising = False)

        seen = {}
        in_window = threading.Event()
        checked = threading.Event()

        def _spawn_window():
            with hf_environment_restored_for_spawn():
                seen["env_during"] = os.environ.get("HF_HUB_OFFLINE")
                in_window.set()
                checked.wait(5)

        with force_hf_offline():
            assert _env_offline() is True
            t = threading.Thread(target = _spawn_window)
            t.start()
            assert in_window.wait(5)
            seen["predicate_during"] = _env_offline()
            checked.set()
            t.join(5)
            assert _env_offline() is True

        assert seen["env_during"] is None  # the user's value really was restored
        assert seen["predicate_during"] is True  # but the predicate still reads offline
        assert _env_offline() is False  # and it lets go afterwards

    def test_raw_fetch_is_skipped_during_the_window(self, monkeypatch, tmp_path):
        """The stall this prevents: _detect_audio_from_tokenizer's own requests.get is the
        call the gate protects, so drive that function rather than a caller that could
        return earlier for unrelated reasons."""
        import threading

        import utils.models.model_config as mc
        from utils.utils import force_hf_offline, hf_environment_restored_for_spawn

        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            monkeypatch.delenv(key, raising = False)
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        calls = []

        class _FakeRequests:
            @staticmethod
            def get(url, **kwargs):
                calls.append(url)
                raise AssertionError("no request may leave during an offline window")

        monkeypatch.setitem(sys.modules, "requests", _FakeRequests)

        # Control: outside any window the gate is open and the fetch is attempted.
        mc._detect_audio_from_tokenizer("org/some-audio-model", None, local_files_only = False)
        assert calls, "the raw fetch is not reached at all, so the gate is untested"
        calls.clear()

        in_window = threading.Event()
        release = threading.Event()

        def _spawn_window():
            with hf_environment_restored_for_spawn():
                in_window.set()
                release.wait(5)

        with force_hf_offline():
            t = threading.Thread(target = _spawn_window)
            t.start()
            assert in_window.wait(5)
            mc._detect_audio_from_tokenizer(
                "org/some-audio-model",
                None,
                local_files_only = False,
            )
            release.set()
            t.join(5)

        assert calls == []


class TestLocalModelWithRemoteBaseIsGuarded:
    """A local LoRA or GGUF is served from disk, so the load guard stands down for it. Its
    base can still be a hub repo, and resolving the config dereferences that base."""

    def test_helper_engages_for_a_remote_target(self, monkeypatch):
        import utils.models.model_config as mc
        import core.inference.llama_cpp as llama_cpp

        opened = []

        @contextlib.contextmanager
        def _fake_for(target):
            opened.append(target)
            yield

        monkeypatch.setattr(llama_cpp, "_hf_offline_if_unreachable_for", _fake_for)
        with mc._offline_while_reading("org/base"):
            pass
        assert opened == ["org/base"]

    def test_helper_is_a_noop_when_the_guard_is_unavailable(self, monkeypatch):
        """Worker contexts may not be able to import the route-side guard."""
        import builtins

        import utils.models.model_config as mc

        real_import = builtins.__import__

        def _boom(name, *a, **k):
            if name == "core.inference.llama_cpp":
                raise ImportError("not available here")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _boom)
        with mc._offline_while_reading("org/base"):
            pass  # must not raise

    def test_local_lora_base_lookup_runs_inside_the_window(self):
        """AST check: the base-derived vision/audio probes sit under the guard."""
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "utils" / "models" / "model_config.py").read_text(
            encoding = "utf-8",
        )
        tree = ast.parse(src)

        guarded = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            if not any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id", None) == "_offline_while_reading"
                for i in node.items
            ):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    name = getattr(inner.func, "id", None)
                    if name:
                        guarded.add(name)

        for name in ("is_vision_model", "detect_audio_type"):
            assert name in guarded, f"{name} is not inside an _offline_while_reading window"

    def test_no_unguarded_base_vision_probe_remains(self):
        """Every is_vision_model call on a resolved base must be under the guard."""
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "utils" / "models" / "model_config.py").read_text(
            encoding = "utf-8",
        )
        tree = ast.parse(src)

        guarded_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call)
                and getattr(i.context_expr.func, "id", None) == "_offline_while_reading"
                for i in node.items
            ):
                guarded_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

        bare = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and getattr(node.func, "id", None) == "is_vision_model"
            ):
                continue
            # Only calls on a name that came from resolved metadata, not on the identifier
            # the caller already guarded.
            arg = node.args[0] if node.args else None
            if (
                getattr(arg, "id", None) in ("base", "check_model")
                and node.lineno not in guarded_lines
            ):
                bare.append(node.lineno)
        assert bare == [], f"unguarded base vision probe at line(s) {bare}"


class TestMetadataUrlsUseTheDownloadRoute:
    """A Hub-compatible mirror implements /{repo}/resolve/{rev}/{file}, the route
    hf_hub_url builds. /raw is a huggingface.co web route a mirror need not serve, so
    reads against a mirror 404 and the tier falls back to name matching."""

    def test_url_matches_what_the_hub_client_would_build(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf.mirror.internal")
        from utils.transformers_version import _hf_raw_url

        ours = _hf_raw_url("acme/ministral-3b", "config.json")
        assert ours == "https://hf.mirror.internal/acme/ministral-3b/resolve/main/config.json"

        try:
            from huggingface_hub import hf_hub_url
        except Exception:
            return
        theirs = hf_hub_url(
            "acme/ministral-3b",
            "config.json",
            endpoint = "https://hf.mirror.internal",
        )
        assert ours == theirs, f"diverged from the hub client: {ours} vs {theirs}"

    def test_trailing_slash_endpoint_does_not_double_up(self, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf.mirror.internal/")
        from utils.transformers_version import _hf_raw_url
        assert "//acme" not in _hf_raw_url("acme/m", "config.json").removeprefix("https://")

    def test_no_raw_route_remains_in_the_metadata_readers(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "utils" / "transformers_version.py").read_text(
            encoding = "utf-8",
        )
        assert "/raw/main" not in src


class TestLocalGgufWithoutABaseSkipsTheProbe:
    """The exporter writes export_metadata.json with a null base_model for a non-LoRA
    checkpoint. Guarding a lookup that never happens would make a wholly local load pay
    the reachability probe."""

    def test_guard_is_not_entered_without_a_base(self):
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "utils" / "models" / "model_config.py").read_text(
            encoding = "utf-8",
        )
        tree = ast.parse(src)

        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            for item in node.items:
                ctx = item.context_expr
                if not (
                    isinstance(ctx, ast.Call)
                    and getattr(ctx.func, "id", None) == "_offline_while_reading"
                ):
                    continue
                arg = ctx.args[0] if ctx.args else None
                if getattr(arg, "id", None) != "base":
                    continue
                # The base variant must sit under an `if base:` truthiness check.
                guarded_by_if = any(
                    isinstance(anc, ast.If)
                    and getattr(anc.test, "id", None) == "base"
                    and any(node is n or node in ast.walk(n) for n in anc.body)
                    for anc in ast.walk(tree)
                    if isinstance(anc, ast.If)
                )
                if not guarded_by_if:
                    bad.append(node.lineno)
        assert bad == [], f"_offline_while_reading(base) entered unconditionally at {bad}"

    def test_none_target_would_otherwise_be_treated_as_remote(self):
        """Why the check is needed: a null target is classified remote, not local."""
        from utils.paths import is_local_path
        assert is_local_path(None or "") is False


class TestLocalLoraRemoteBaseIsInTheGuardTargets:
    """A local LoRA's adapter_config points at a remote base. latest_tier_active_for and
    the training guard resolve that base internally, so both local paths passed alone
    would select a null context while the base's metadata is fetched."""

    def test_every_config_keyed_guard_passes_the_base(self):
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
        tree = ast.parse(src)

        checked = 0
        bad = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_thread"
            ):
                continue
            if not node.args or getattr(node.args[0], "id", None) != "_offline_guarded":
                continue
            targets = node.args[1]
            if not isinstance(targets, ast.Tuple):
                continue
            names = []
            for el in targets.elts:
                if isinstance(el, ast.Attribute):
                    names.append(f"{getattr(el.value, 'id', '?')}.{el.attr}")
                elif isinstance(el, ast.Call):
                    names.append(getattr(el.func, "id", "?"))
                else:
                    names.append(getattr(el, "id", "?"))
            # A tuple keyed on the resolved config must also carry its base.
            if "config.identifier" in names:
                checked += 1
                if not any(n == "getattr" for n in names):
                    bad.append((node.lineno, names))
        assert checked >= 4, f"expected the four config-keyed guards, saw {checked}"
        assert bad == [], f"guard target tuple omits the resolved base: {bad}"


class TestHungProbeHonoursFailOpenBehindAProxy:
    """A thread still alive at the join is ambiguous behind a proxy: connect, TLS and the
    response can each stay under `timeout` while the total runs past it. Direct, a hang
    still means the real hub calls would hang too, so that verdict is unchanged."""

    class _Hang:
        def open(
            self,
            req,
            timeout = None,
        ):
            import threading as _t
            _t.Event().wait()

    def test_behind_a_proxy_the_flag_decides(self, monkeypatch):
        import utils.transformers_version as tv

        monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
        monkeypatch.delenv("NO_PROXY", raising = False)
        monkeypatch.setattr(tv, "_hf_proxy_opener", lambda _u: self._Hang())

        assert tv.hf_endpoint_unreachable(1, proxy_timeouts_offline = True) is True
        assert tv.hf_endpoint_unreachable(1, proxy_timeouts_offline = False) is False

    def test_direct_hang_still_reads_unreachable(self, monkeypatch, clean_proxy_env):
        """Unchanged: a direct hang means the load would hang the same way."""
        import utils.transformers_version as tv

        monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
        monkeypatch.setattr(tv, "_hf_proxy_opener", lambda _u: self._Hang())
        assert tv.hf_endpoint_unreachable(1, proxy_timeouts_offline = False) is True

    def test_it_stays_bounded(self, monkeypatch, clean_proxy_env):
        import time as _time

        import utils.transformers_version as tv

        monkeypatch.setenv("HF_ENDPOINT", "https://hub.example.test")
        monkeypatch.setattr(tv, "_hf_proxy_opener", lambda _u: self._Hang())
        t0 = _time.monotonic()
        assert tv.hf_endpoint_unreachable(1) is True
        assert _time.monotonic() - t0 < 5.0


class TestGuardsShareOneDnsLookup:
    """/validate opens several guards per request. The DNS shortcut is cheap only when
    DNS is fast; on a slow resolver it costs its full timeout per guard."""

    @pytest.fixture(autouse = True)
    def _fresh(self):
        from utils.utils import reset_hf_reachability_cache

        reset_hf_reachability_cache()
        yield
        reset_hf_reachability_cache()

    def test_slow_dns_costs_one_lookup_across_sibling_guards(self, monkeypatch, clean_offline_env):
        """The case that accumulates: DNS answers past its deadline, so the shortcut is
        inconclusive and the probe decides. That probe memoises, so the siblings reuse it."""
        from core.inference.llama_cpp import _hf_unreachable

        dns = []
        probe = []
        monkeypatch.setattr("utils.utils.hf_dns_dead", lambda *a, **k: (dns.append(1), False)[1])
        import utils.transformers_version as tv

        monkeypatch.setattr(
            tv, "hf_endpoint_unreachable", lambda *a, **k: (probe.append(1), True)[1]
        )

        assert _hf_unreachable() is True
        for _ in range(5):
            assert _hf_unreachable() is True
        assert dns == [1], f"DNS shortcut ran {len(dns)} times, expected once"
        assert probe == [1], f"probe ran {len(probe)} times, expected once"

    def test_a_reachable_verdict_is_reused_too(self, monkeypatch, clean_offline_env):
        from core.inference.llama_cpp import _hf_unreachable

        dns = []
        probe = []
        monkeypatch.setattr("utils.utils.hf_dns_dead", lambda *a, **k: (dns.append(1), False)[1])
        import utils.transformers_version as tv

        monkeypatch.setattr(
            tv, "hf_endpoint_unreachable", lambda *a, **k: (probe.append(1), False)[1]
        )

        assert _hf_unreachable() is False
        for _ in range(5):
            assert _hf_unreachable() is False
        assert dns == [1] and probe == [1]

    def test_a_dead_lookup_is_not_recorded_so_recovery_is_immediate(
        self, monkeypatch, clean_offline_env
    ):
        """A dead lookup fails fast, so caching it would only delay recovery by the TTL."""
        from core.inference.llama_cpp import _hf_unreachable

        state = {"dead": True}
        monkeypatch.setattr("utils.utils.hf_dns_dead", lambda *a, **k: state["dead"])
        import utils.transformers_version as tv

        monkeypatch.setattr(tv, "hf_endpoint_unreachable", lambda *a, **k: False)

        assert _hf_unreachable() is True
        state["dead"] = False
        assert _hf_unreachable() is False  # no waiting out the TTL
