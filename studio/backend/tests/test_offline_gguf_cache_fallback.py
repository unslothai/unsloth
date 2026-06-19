# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the offline GGUF cache fallback path (#5505).

Three failure modes hit users when ``huggingface.co`` is unreachable
but the requested GGUF repo is fully cached locally:

* ``list_gguf_variants`` raised through ``HTTPException(500)`` so the
  variant dropdown sat empty.
* ``detect_gguf_model_remote`` returned ``None`` so a GGUF-only repo
  was misrouted into the transformers/Unsloth backend (on macOS this
  surfaced as a hardware error).
* ``_download_gguf`` fell back to a synthetic ``{repo}-{variant}.gguf``
  name that did not exist in cache when the in-repo filename did not
  echo the repo name (e.g. ``unsloth/Qwen3.6-27B-MTP-GGUF`` ships
  ``Qwen3.6-27B-UD-Q4_K_XL.gguf`` with no ``MTP`` token).

Two follow-up regressions covered here:

* P1 #1: the cache-side variant filter must match the snapshot-relative
  path, not just the basename, so subdir layouts like
  ``BF16/foo.gguf`` are findable.
* P1 #2: the DNS auto-detect must scope ``HF_HUB_OFFLINE`` to one load
  via try/finally so a transient resolver hiccup cannot lock the
  long-lived ``LlamaCppBackend`` singleton offline forever.

No GPU, no network, no subprocess. Linux, macOS, Windows compatible.
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

# Stub heavy/unavailable external deps before importing the modules
# under test (same pattern as other studio backend tests).
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
    _hf_offline_if_dns_dead,
    _probe_dns_dead,
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


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """Point ``huggingface_hub.constants.HF_HUB_CACHE`` at a temp dir."""
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    return tmp_path


@pytest.fixture
def clean_offline_env(monkeypatch):
    """Strip ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` for the test."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


def _siblings(items: dict[str, int]):
    """Mock ``hf_model_info(...).siblings`` payload."""
    return _types.SimpleNamespace(
        siblings = [
            _types.SimpleNamespace(rfilename = name, size = size)
            for name, size in items.items()
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
        old = _build_cache(
            hf_cache, "unsloth/multi", {"x.gguf": 1}, snapshot_sha = "a" * 40
        )
        new = _build_cache(
            hf_cache, "unsloth/multi", {"y.gguf": 1}, snapshot_sha = "b" * 40
        )
        os.utime(old, (1000, 1000))
        os.utime(new, (2000, 2000))
        out = list(_iter_hf_cache_snapshots("unsloth/multi"))
        assert [p.name for p in out] == ["b" * 40, "a" * 40]

    def test_repo_id_match_is_case_insensitive(self, hf_cache):
        _build_cache(hf_cache, "unsloth/Foo-GGUF", {"Foo-Q4_K_M.gguf": 1})
        # Lookup with a different casing of the org/name still resolves
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


class TestListGgufVariantsOffline:
    def test_offline_env_short_circuits_api(
        self, hf_cache, clean_offline_env, monkeypatch
    ):
        _build_cache(hf_cache, "unsloth/a", {"a-UD-Q4_K_XL.gguf": 1})
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise AssertionError("API must not be called when offline env set")

        with patch("huggingface_hub.model_info", boom):
            variants, _has = list_gguf_variants("unsloth/a")
        assert len(variants) == 1
        assert variants[0].quant == "UD-Q4_K_XL"

    def test_api_exception_falls_back_to_cache(
        self,
        hf_cache,
        clean_offline_env,
    ):
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
        """P1 #1 regression: ``BF16/foo.gguf`` (quant only in directory).
        Before the fix, the offline cache scan matched on basename and
        missed this layout, falling through to the synthetic
        ``{repo}-{variant}.gguf`` heuristic."""
        _build_cache(
            hf_cache,
            "unsloth/gpt-oss-20b-BF16",
            {"BF16/foo.gguf": 1},
        )
        out = _detect_gguf_from_hf_cache("unsloth/gpt-oss-20b-BF16")
        assert (
            out == "BF16/foo.gguf"
        ), f"subdir-only layout must resolve to relative path, got {out}"

    def test_returns_none_when_no_gguf(self, hf_cache):
        _build_cache(hf_cache, "unsloth/a", {"README.md": 10})
        assert _detect_gguf_from_hf_cache("unsloth/a") is None


class TestDetectGgufModelRemoteOffline:
    def test_offline_env_short_circuits_retries(
        self,
        hf_cache,
        clean_offline_env,
        monkeypatch,
    ):
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

    def test_repository_not_found_does_not_consult_cache(
        self,
        hf_cache,
        clean_offline_env,
    ):
        # Cache has a file but the API explicitly says repo is gone.
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

    def test_user_set_hf_hub_offline_is_preserved(
        self,
        dns,
        clean_offline_env,
        monkeypatch,
    ):
        # User explicitly set offline before launching Studio.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead() as did_set:
            assert did_set is False
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
        # Helper must not pop a variable it did not set.
        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    def test_user_set_transformers_offline_is_preserved(
        self,
        dns,
        clean_offline_env,
        monkeypatch,
    ):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        dns.fail()
        with _hf_offline_if_dns_dead():
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        # HF_HUB_OFFLINE was set by helper -> removed.
        assert "HF_HUB_OFFLINE" not in os.environ
        # TRANSFORMERS_OFFLINE pre-existed -> preserved.
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    def test_exception_inside_block_still_restores_env(
        self,
        dns,
        clean_offline_env,
    ):
        dns.fail()
        with pytest.raises(RuntimeError, match = "boom"):
            with _hf_offline_if_dns_dead():
                raise RuntimeError("boom")
        # Cleanup must happen on exception as well.
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ


class TestExtractQuantLabelSubdir:
    """``_extract_quant_label`` must consider the parent directories when
    the basename has no quant token. Subdir layouts like ``BF16/foo.gguf``
    are documented in this codebase and surface through the cache scan."""

    def test_quant_in_basename_unchanged(self):
        assert _extract_quant_label("BF16/foo-BF16.gguf") == "BF16"
        assert _extract_quant_label("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_quant_only_in_parent_dir(self):
        assert _extract_quant_label("BF16/foo.gguf") == "BF16"

    def test_ud_prefix_in_parent_dir(self):
        assert _extract_quant_label("UD-Q4_K_XL/weight.gguf") == "UD-Q4_K_XL"

    def test_deeper_nesting_picks_nearest_quant_dir(self):
        # When multiple parent segments could match, prefer the one closest
        # to the file (innermost). This matches how repos like
        # ``models/MXFP4_MOE/foo.gguf`` are laid out.
        assert _extract_quant_label("models/MXFP4_MOE/foo.gguf") == "MXFP4_MOE"


class TestDownloadMmprojOfflineCacheFallback:
    """``LlamaCppBackend._download_mmproj`` must resolve cached mmproj
    GGUFs offline, same shape as ``_download_gguf``. Without this the
    offline vision GGUF load path returns ``None`` even when the mmproj
    is present in cache."""

    def test_cache_lookup_returns_cached_mmproj_when_list_repo_files_fails(
        self,
        hf_cache,
    ):
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

        def fake_download(*, repo_id, filename, token = None):
            # Echo back so the test can verify the cache-resolved filename
            return f"/fake/cache/{repo_id}/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", boom_list),
            patch("huggingface_hub.hf_hub_download", fake_download),
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

        def fake_download(*, repo_id, filename, token = None):
            captured["filename"] = filename
            return f"/fake/{filename}"

        with (
            patch("huggingface_hub.list_repo_files", boom_list),
            patch("huggingface_hub.hf_hub_download", fake_download),
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
    produce distinct quant labels, not collapse on basename."""

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

    def test_transient_error_still_falls_back_to_cache(
        self, hf_cache, clean_offline_env
    ):
        from utils.models.model_config import list_gguf_variants

        _build_cache(hf_cache, "u/transient-gguf", {"foo-Q4_K_M.gguf": 1})

        def boom(*a, **k):
            raise OSError("network down")

        with patch("huggingface_hub.model_info", boom):
            variants, _ = list_gguf_variants("u/transient-gguf")
        assert any(v.quant == "Q4_K_M" for v in variants)


class TestDetectGgufFromCacheExcludesMmproj:
    """A partial cache with only a vision projector must not route the
    projector as the main model."""

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
    """``_probe_dns_dead`` must not change ``socket.setdefaulttimeout``
    process-wide -- concurrent sockets without explicit timeout would
    inherit it for the probe window."""

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
            # Restore exact state regardless of any test-side mutation.
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
    """A TCP RST mid-read while llama-server is still binding the port
    (Windows: WinError 10054) must not abort the health-poll loop --
    that masks a legitimate 'still warming up' state as a fatal load."""

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

        def fake_get(url, timeout = None):
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
