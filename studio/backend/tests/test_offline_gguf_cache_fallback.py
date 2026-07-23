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
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

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
    _hf_offline_if_dns_dead,
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
    monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "0")


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
    def test_unreachable_endpoint_short_circuits_retries(
        self, hf_cache, clean_offline_env, monkeypatch
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})
        monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "1")
        monkeypatch.setattr(
            "utils.transformers_version.hf_endpoint_unreachable",
            lambda timeout = 3: True,
        )

        def boom(*args, **kwargs):
            raise AssertionError("API must not be called when endpoint is unreachable")

        with patch("huggingface_hub.model_info", boom):
            assert detect_gguf_model_remote("unsloth/a") == "a-Q4_K_M.gguf"

    def test_unreachable_endpoint_lists_cached_variants_without_api(
        self, hf_cache, clean_offline_env, monkeypatch
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})
        monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "1")
        monkeypatch.setattr(
            "utils.transformers_version.hf_endpoint_unreachable",
            lambda timeout = 3: True,
        )

        def boom(*args, **kwargs):
            raise AssertionError("API must not be called when endpoint is unreachable")

        with patch("huggingface_hub.model_info", boom):
            variants, has_vision = list_gguf_variants("unsloth/a")
        assert [variant.filename for variant in variants] == ["a-Q4_K_M.gguf"]
        assert has_vision is False

    def test_offline_env_short_circuits_retries(self, hf_cache, clean_offline_env, monkeypatch):
        _build_cache(hf_cache, "unsloth/a", {"a-Q4_K_M.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            assert detect_gguf_model_remote("unsloth/a") == "a-Q4_K_M.gguf"

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

    def test_transient_timeout_retries_before_success(self, clean_offline_env):
        calls = 0

        class ReadTimeout(Exception):
            pass

        def flaky(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ReadTimeout("temporary timeout")
            return _types.SimpleNamespace(
                siblings = [_types.SimpleNamespace(rfilename = "a-Q4_K_M.gguf")]
            )

        with (
            patch("huggingface_hub.model_info", flaky),
            patch("time.sleep", lambda *_: None),
        ):
            assert detect_gguf_model_remote("unsloth/a") == "a-Q4_K_M.gguf"
        assert calls == 3

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
# _probe_dns_dead / _hf_offline_if_dns_dead
# ---------------------------------------------------------------------------


class _DnsState:
    """Tiny helper that toggles ``socket.gethostbyname`` failure mode."""

    def __init__(self, monkeypatch):
        self._mp = monkeypatch
        self._real = socket.gethostbyname

    def fail(self):
        def _fail(*a, **k):
            raise socket.gaierror(-2, "Name or service not known")

        self._mp.setattr(socket, "gethostbyname", _fail)

    def ok(self):
        self._mp.setattr(socket, "gethostbyname", lambda *a, **k: "127.0.0.1")

    def restore(self):
        self._mp.setattr(socket, "gethostbyname", self._real)


@pytest.fixture
def dns(monkeypatch):
    return _DnsState(monkeypatch)


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


class TestHfOfflineIfDnsDead:
    def test_dns_fail_sets_env_inside_block_only(self, dns, clean_offline_env):
        dns.fail()
        assert "HF_HUB_OFFLINE" not in os.environ
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is True
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # P1 #2: env must be restored after the block
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ

    def test_dns_ok_is_noop(self, dns, clean_offline_env):
        dns.ok()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_dns_recovers_between_calls(self, dns, clean_offline_env):
        # First call: DNS dead -> env set inside, cleared on exit.
        dns.fail()
        with _hf_offline_if_dns_dead():
            pass
        assert "HF_HUB_OFFLINE" not in os.environ
        # Second call: DNS healthy -> no env mutation.
        dns.ok()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert "HF_HUB_OFFLINE" not in os.environ

    def test_user_set_hf_hub_offline_is_preserved(self, dns, clean_offline_env, monkeypatch):
        # User explicitly set offline before launching Unsloth.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # Helper must not pop a variable it did not set.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_user_set_transformers_offline_is_preserved(self, dns, clean_offline_env, monkeypatch):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # HF_HUB_OFFLINE was set by helper -> removed.
        assert "HF_HUB_OFFLINE" not in os.environ
        # TRANSFORMERS_OFFLINE pre-existed -> preserved.
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_exception_inside_block_still_restores_env(self, dns, clean_offline_env):
        dns.fail()
        with pytest.raises(RuntimeError, match = "boom"):
            with _hf_offline_if_dns_dead():
                raise RuntimeError("boom")
        # Cleanup must happen on exception as well.
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ


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
        monkeypatch.setattr(_socket, "gethostbyname", lambda h: "127.0.0.1")

        try:
            _probe_dns_dead("example.invalid", timeout = 0.5)
        finally:
            # Restore exact state regardless of test-side mutation.
            original_set(prev)

        assert set_calls == [], (
            f"_probe_dns_dead mutated socket.setdefaulttimeout {set_calls}; "
            "must isolate timeout to the probe thread"
        )

    def test_returns_dead_when_resolver_wedges(self, monkeypatch):
        import socket as _socket
        from core.inference.llama_cpp import _probe_dns_dead

        # Simulate a wedged resolver: thread blocks forever.
        def wedged(host):
            import threading
            threading.Event().wait()

        monkeypatch.setattr(_socket, "gethostbyname", wedged)
        assert _probe_dns_dead("example.invalid", timeout = 0.1) is True


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
