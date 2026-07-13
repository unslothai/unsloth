# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the latest-transformers support check and the consented sidecar install."""

import ast
import json
import os
import textwrap
import time
import pytest
from pathlib import Path


# The backend uses "from utils..." imports; ensure the backend dir is on sys.path.
import sys

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub the custom logger before importing the modules under test.
import types as _types

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.transformers_latest as tl
import utils.transformers_version as tv
from utils.transformers_latest import (
    check_upgrade_for_model,
    install_latest_transformers,
    latest_transformers_supports,
    _fetch_remote_model_types,
    _model_types_from_config,
)
from utils.transformers_version import (
    _config_mapping_cache,
    _config_json_cache,
    _higher_tier,
    _is_valid_version_string,
    _model_types_from_source,
    _tier_from_config_mapping,
    _venv_t5_latest_packages,
    activate_transformers_for_subprocess,
    ensure_latest_transformers_venv,
    get_transformers_tier,
    latest_venv_pinned_version,
)


# A CONFIG_MAPPING_NAMES source exercising every construct the AST extractor supports.
_MAPPING_SOURCE = """
from collections import OrderedDict
CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("llama", "LlamaConfig"),
        ("gemma4", "Gemma4Config"),
    ],
    **{"qwen3_moe": "Qwen3MoeConfig"},
)
CONFIG_MAPPING_NAMES.update({"brandnew_arch": "BrandNewConfig"})
"""

_MAIN_ONLY_SOURCE = """
CONFIG_MAPPING_NAMES = {
    "llama": "LlamaConfig",
    "gemma4": "Gemma4Config",
    "qwen3_moe": "Qwen3MoeConfig",
    "brandnew_arch": "BrandNewConfig",
    "dev_only_arch": "DevOnlyConfig",
}
"""


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_urlopen_factory(counter: dict):
    """urlopen stub serving the PyPI JSON and both refs' mapping sources."""

    def _fake_urlopen(req, timeout = None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        counter[url] = counter.get(url, 0) + 1
        counter["__total__"] = counter.get("__total__", 0) + 1
        if url == tl._PYPI_JSON_URL:
            return _FakeResponse(json.dumps({"info": {"version": "5.13.0"}}).encode())
        if "/v5.13.0/" in url and url.endswith("auto_mappings.py"):
            return _FakeResponse(_MAPPING_SOURCE.encode())
        if "/v5.13.0/" in url and url.endswith("configuration_auto.py"):
            return _FakeResponse(b"CONFIG_MAPPING_NAMES = {}\n")
        if "/main/" in url and url.endswith("auto_mappings.py"):
            return _FakeResponse(_MAIN_ONLY_SOURCE.encode())
        if "/main/" in url and url.endswith("configuration_auto.py"):
            return _FakeResponse(b"CONFIG_MAPPING_NAMES = {}\n")
        raise AssertionError(f"unexpected URL fetched: {url}")

    return _fake_urlopen


@pytest.fixture(autouse = True)
def _isolated_caches(tmp_path: Path, monkeypatch):
    """Fresh in-memory + on-disk caches per test; no accidental real studio_root writes."""
    tl.clear_caches()
    monkeypatch.setattr(tl, "_cache_file", lambda: tmp_path / "transformers_latest_check.json")
    # The sidecar swap reservation writes a lock file next to the venv dir;
    # point it at tmp so tests never touch the real studio root.
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
    monkeypatch.delenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    yield
    tl.clear_caches()


def _no_network(monkeypatch, exc = None):
    """Fail every urlopen and return a counter; tests assert n == 0 to prove no fetch
    happened (check_upgrade_for_model swallows exceptions, so a raising stub alone
    cannot prove the negative)."""
    calls = {"n": 0}

    def _raise(*args, **kwargs):
        calls["n"] += 1
        raise (exc or OSError("network fetch attempted"))

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    return calls


# --- AST extraction shared with the static router ---


class TestModelTypesFromSource:
    def test_ordereddict_update_and_unpacking(self):
        keys = _model_types_from_source(_MAPPING_SOURCE)
        assert keys == {"llama", "gemma4", "qwen3_moe", "brandnew_arch"}

    def test_plain_dict_literal(self):
        keys = _model_types_from_source(_MAIN_ONLY_SOURCE)
        assert "dev_only_arch" in keys and "llama" in keys

    def test_syntax_error_raises_for_caller_to_handle(self):
        with pytest.raises(SyntaxError):
            _model_types_from_source("def broken(:\n")


class TestFetchRemoteModelTypes:
    def test_merges_both_auto_files(self, monkeypatch):
        counter = {}
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(counter))
        keys = _fetch_remote_model_types("v5.13.0")
        assert keys is not None and "brandnew_arch" in keys

    def test_all_fetches_failing_returns_none(self, monkeypatch):
        _no_network(monkeypatch, exc = OSError("no route"))
        assert _fetch_remote_model_types("main") is None

    def test_empty_mapping_treated_as_failure(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout = None: _FakeResponse(b"CONFIG_MAPPING_NAMES = {}\n"),
        )
        assert _fetch_remote_model_types("main") is None

    def test_transient_failure_of_one_file_fails_whole_lookup(self, monkeypatch):
        # One file times out: the partial map must not be returned and cached.
        def _fake(req, timeout = None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.endswith("configuration_auto.py"):
                return _FakeResponse(_MAPPING_SOURCE.encode())
            raise OSError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        assert _fetch_remote_model_types("main") is None

    def test_missing_auto_mappings_404_still_succeeds(self, monkeypatch):
        # Pre-5.10 tags have no auto_mappings.py; a 404 must not fail the lookup.
        import urllib.error

        def _fake(req, timeout = None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.endswith("configuration_auto.py"):
                return _FakeResponse(_MAPPING_SOURCE.encode())
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        keys = _fetch_remote_model_types("v5.9.0")
        assert keys is not None and "brandnew_arch" in keys

    def test_unparseable_file_fails_whole_lookup(self, monkeypatch):
        def _fake(req, timeout = None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if url.endswith("configuration_auto.py"):
                return _FakeResponse(_MAPPING_SOURCE.encode())
            return _FakeResponse(b"def broken(:\n")

        monkeypatch.setattr("urllib.request.urlopen", _fake)
        assert _fetch_remote_model_types("main") is None


# --- latest_transformers_supports: snapshot, cache, offline, kill switch ---


class TestLatestTransformersSupports:
    def test_supported_in_pypi(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        result = latest_transformers_supports("brandnew_arch")
        assert result == {
            "pypi_version": "5.13.0",
            "supported_in_pypi": True,
            "supported_in_main": True,
        }

    def test_dev_only_arch_reported_main_only(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        result = latest_transformers_supports("dev_only_arch")
        assert result["supported_in_pypi"] is False
        assert result["supported_in_main"] is True

    def test_unknown_everywhere(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        result = latest_transformers_supports("no_such_arch")
        assert result["supported_in_pypi"] is False and result["supported_in_main"] is False

    def test_network_failure_returns_none(self, monkeypatch):
        _no_network(monkeypatch, exc = OSError("down"))
        assert latest_transformers_supports("brandnew_arch") is None

    def test_offline_returns_none_without_fetch(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        calls = _no_network(monkeypatch)
        assert latest_transformers_supports("brandnew_arch") is None
        assert calls["n"] == 0

    def test_kill_switch_returns_none_without_fetch(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
        calls = _no_network(monkeypatch)
        assert latest_transformers_supports("brandnew_arch") is None
        assert calls["n"] == 0

    def test_memory_cache_hit_avoids_refetch(self, monkeypatch):
        counter = {}
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(counter))
        latest_transformers_supports("brandnew_arch")
        first_total = counter["__total__"]
        latest_transformers_supports("some_other_arch")
        assert counter["__total__"] == first_total

    def test_disk_cache_survives_restart(self, monkeypatch):
        counter = {}
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(counter))
        latest_transformers_supports("brandnew_arch")
        # Simulate a restart: memory gone, disk snapshot stays, network unavailable.
        tl.clear_caches()
        _no_network(monkeypatch)
        result = latest_transformers_supports("brandnew_arch")
        assert result is not None and result["supported_in_pypi"] is True

    def test_expired_snapshot_refetches(self, monkeypatch):
        counter = {}
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(counter))
        latest_transformers_supports("brandnew_arch")
        stale = dict(tl._memory_snapshot, fetched_at = time.time() - tl._CACHE_TTL_SECONDS - 1)
        tl.clear_caches()
        tl._save_snapshot_file(stale)
        first_total = counter["__total__"]
        latest_transformers_supports("brandnew_arch")
        assert counter["__total__"] > first_total

    def test_corrupt_disk_cache_ignored(self, monkeypatch, tmp_path: Path):
        counter = {}
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory(counter))
        tl._cache_file().write_text("{not json", encoding = "utf-8")
        result = latest_transformers_supports("brandnew_arch")
        assert result is not None and counter["__total__"] > 0

    def test_failure_backoff_skips_immediate_retry(self, monkeypatch):
        calls = {"n": 0}

        def _fail(*args, **kwargs):
            calls["n"] += 1
            raise OSError("down")

        monkeypatch.setattr("urllib.request.urlopen", _fail)
        assert latest_transformers_supports("brandnew_arch") is None
        first = calls["n"]
        assert latest_transformers_supports("brandnew_arch") is None
        assert calls["n"] == first  # backed off, no second network attempt


# --- check_upgrade_for_model: the tier hook ---


def _local_model(tmp_path: Path, model_type: str) -> str:
    d = tmp_path / f"model_{model_type}"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


_FAKE_OVERLAYS = {
    "default": frozenset({"llama", "bert", "gpt2"}),
    "530": frozenset({"qwen3_moe", "qwen3_next"}),
    "550": frozenset({"gemma4"}),
    "510": frozenset({"gemma4_unified"}),
    "latest": frozenset(),
}


def _fake_overlays(monkeypatch, overlays = None):
    overlays = overlays or _FAKE_OVERLAYS
    fake = lambda tier: overlays.get(tier, frozenset())
    monkeypatch.setattr(tv, "_config_model_types", fake)
    monkeypatch.setattr(tl, "_config_model_types", fake)


class TestCheckUpgradeForModel:
    def test_unknown_type_supported_in_pypi_signals(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        result = check_upgrade_for_model(_local_model(tmp_path, "brandnew_arch"))
        assert result == {
            "model_type": "brandnew_arch",
            "pypi_version": "5.13.0",
            "supported_in_pypi": True,
            "supported_in_main": True,
        }

    def test_dev_only_type_signals_main_only(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        result = check_upgrade_for_model(_local_model(tmp_path, "dev_only_arch"))
        assert result["supported_in_pypi"] is False and result["supported_in_main"] is True

    def test_unknown_everywhere_falls_through(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        assert check_upgrade_for_model(_local_model(tmp_path, "no_such_arch")) is None

    def test_offline_falls_through_without_fetch(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        calls = _no_network(monkeypatch)
        assert check_upgrade_for_model(_local_model(tmp_path, "brandnew_arch")) is None
        assert calls["n"] == 0

    def test_network_failure_falls_through(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        _no_network(monkeypatch, exc = OSError("down"))
        assert check_upgrade_for_model(_local_model(tmp_path, "brandnew_arch")) is None

    def test_known_default_type_never_fetches(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        calls = _no_network(monkeypatch)
        assert check_upgrade_for_model(_local_model(tmp_path, "llama")) is None
        assert calls["n"] == 0

    def test_known_sidecar_type_never_fetches(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        calls = _no_network(monkeypatch)
        assert check_upgrade_for_model(_local_model(tmp_path, "gemma4_unified")) is None
        assert calls["n"] == 0

    def test_hardcoded_tier_type_never_fetches_even_without_overlays(
        self, tmp_path: Path, monkeypatch
    ):
        # Sidecar overlays unreadable, but the hardcoded tables route it.
        _fake_overlays(
            monkeypatch,
            {"default": frozenset({"llama"})},
        )
        calls = _no_network(monkeypatch)
        assert check_upgrade_for_model(_local_model(tmp_path, "qwen3_5_moe")) is None
        assert calls["n"] == 0

    def test_unreadable_default_overlay_bails_out(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch, {"default": frozenset()})
        calls = _no_network(monkeypatch)
        assert check_upgrade_for_model(_local_model(tmp_path, "brandnew_arch")) is None
        assert calls["n"] == 0

    def test_no_model_type_falls_through(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        _no_network(monkeypatch)
        d = tmp_path / "no_type"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"architectures": ["Whatever"]}))
        assert check_upgrade_for_model(str(d)) is None

    def test_nested_model_type_is_used(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        d = tmp_path / "nested"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"text_config": {"model_type": "brandnew_arch"}}))
        result = check_upgrade_for_model(str(d))
        assert result is not None and result["model_type"] == "brandnew_arch"

    def test_never_raises_on_internal_error(self, monkeypatch):
        monkeypatch.setattr(
            tl, "_load_config_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        assert check_upgrade_for_model("some/model") is None


class TestNestedModelTypeExtraction:
    def test_top_level_wins(self):
        assert _model_types_from_config(
            {"model_type": "a", "text_config": {"model_type": "b"}}
        ) == ["a", "b"]

    def test_nested_fallback(self):
        assert _model_types_from_config({"llm_config": {"model_type": "b"}}) == ["b"]

    def test_missing_returns_none(self):
        assert _model_types_from_config({}) == []


# --- Routing parity: overlay-shipped model_types route as before, never remote-check ---


class TestRoutingParity:
    def test_all_overlay_types_route_identically_and_never_check(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        calls = _no_network(monkeypatch)
        expected_tier = {
            "llama": "default",
            "bert": "default",
            "gpt2": "default",
            "qwen3_moe": "530",
            "qwen3_next": "530",
            "gemma4": "550",
            "gemma4_unified": "510",
        }
        for model_type, tier in expected_tier.items():
            cfg = {"model_type": model_type}
            assert _tier_from_config_mapping(cfg) == tier, model_type
            assert check_upgrade_for_model(_local_model(tmp_path, model_type)) is None
        assert calls["n"] == 0

    def test_real_installed_mappings_route_without_checker(self, monkeypatch, tmp_path: Path):
        """Parity over the REAL installed overlays (base + any provisioned sidecar):
        every shipped model_type resolves statically, so the remote checker never
        fires and routing is byte-identical with the feature enabled."""
        _no_network(monkeypatch)
        seen = 0
        for tier in ("default", "530", "550", "510"):
            types = tv._config_model_types(tier)
            if not types:
                continue  # overlay not provisioned in this environment
            for model_type in types:
                assert _tier_from_config_mapping({"model_type": model_type}) is not None
                seen += 1
        if seen == 0:
            pytest.skip("no transformers overlay available in this environment")

    def test_get_tier_unchanged_by_kill_switch(self, tmp_path: Path, monkeypatch):
        _fake_overlays(monkeypatch)
        _no_network(monkeypatch)
        path = _local_model(tmp_path, "no_such_arch")
        _config_json_cache.clear()
        tier_default = get_transformers_tier(path, probe = False)
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
        _config_json_cache.clear()
        assert get_transformers_tier(path, probe = False) == tier_default == "default"


# --- .venv_t5_latest provisioning and routing participation ---


class TestLatestVenvProvisioning:
    def test_version_string_validation(self):
        assert _is_valid_version_string("5.13.0")
        assert _is_valid_version_string("5.14.0rc1")
        assert not _is_valid_version_string("5.13.0; rm -rf /")
        assert not _is_valid_version_string("git+https://evil")
        assert not _is_valid_version_string("")

    def test_packages_pin_exact_version(self):
        pkgs = _venv_t5_latest_packages("5.13.0")
        assert pkgs[0] == "transformers==5.13.0"
        assert any(p.startswith("huggingface_hub==") for p in pkgs)

    def test_ensure_latest_writes_pin_and_invalidates_cache(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        recorded = {}

        def _fake_ensure(dir_, packages, label):
            recorded["dir"] = dir_
            recorded["packages"] = packages
            Path(dir_).mkdir(parents = True, exist_ok = True)
            return True

        monkeypatch.setattr(tv, "_ensure_venv_dir", _fake_ensure)
        _config_mapping_cache["latest"] = frozenset({"stale"})
        assert ensure_latest_transformers_venv("5.13.0") is True
        # Stage-and-swap: pip installs into staging, the live dir is the swap result.
        assert recorded["dir"] == str(venv_dir) + ".staging"
        assert "transformers==5.13.0" in recorded["packages"]
        assert venv_dir.is_dir()
        assert not Path(str(venv_dir) + ".staging").exists()
        assert latest_venv_pinned_version() == "5.13.0"
        assert "latest" not in _config_mapping_cache

    def test_ensure_latest_upgrade_failure_keeps_old_sidecar(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        venv_dir.mkdir(parents = True)
        (venv_dir / tv._LATEST_PIN_MARKER).write_text(
            json.dumps({"version": "5.12.0", "packages": ["transformers==5.12.0"]})
        )
        (venv_dir / "transformers").mkdir()
        monkeypatch.setattr(tv, "_venv_dir_is_valid", lambda *a, **k: True)
        # Install fails mid-flight: the previous sidecar and pin survive.
        monkeypatch.setattr(tv, "_ensure_venv_dir", lambda *a, **k: False)
        assert ensure_latest_transformers_venv("5.13.0") is False
        assert latest_venv_pinned_version() == "5.12.0"
        assert (venv_dir / "transformers").is_dir()
        assert not Path(str(venv_dir) + ".staging").exists()

    def test_ensure_latest_rejects_bad_version(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / ".venv_t5_latest"))
        monkeypatch.setattr(
            tv,
            "_ensure_venv_dir",
            lambda *a: (_ for _ in ()).throw(AssertionError("must not install")),
        )
        assert ensure_latest_transformers_venv("5.13.0 && curl evil") is False

    def test_ensure_latest_offline_refuses(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / ".venv_t5_latest"))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(
            tv,
            "_ensure_venv_dir",
            lambda *a: (_ for _ in ()).throw(AssertionError("must not install")),
        )
        assert ensure_latest_transformers_venv("5.13.0") is False

    def test_unpinned_sidecar_never_installs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / ".venv_t5_latest"))
        monkeypatch.setattr(
            tv,
            "_ensure_venv_dir",
            lambda *a: (_ for _ in ()).throw(AssertionError("must not install")),
        )
        assert tv._ensure_venv_t5_latest_exists() is False

    def test_pinned_sidecar_repairs_with_same_version(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        venv_dir.mkdir()
        (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        monkeypatch.setattr(tv, "_venv_dir_is_valid", lambda *a: False)
        recorded = {}

        def _fake_ensure(dir_, packages, label):
            recorded["dir"] = dir_
            recorded["packages"] = packages
            Path(dir_).mkdir(parents = True, exist_ok = True)
            return True

        monkeypatch.setattr(tv, "_ensure_venv_dir", _fake_ensure)
        assert tv._ensure_venv_t5_latest_exists() is True
        # Repair also stage-and-swaps, never installing into the live dir.
        assert recorded["dir"] == str(venv_dir) + ".staging"
        assert "transformers==5.13.0" in recorded["packages"]
        assert latest_venv_pinned_version() == "5.13.0"


class TestLatestTierRouting:
    def test_latest_outranks_510(self):
        assert _higher_tier("latest", "510") == "latest"
        assert _higher_tier("510", "latest") == "latest"

    def test_tier_from_mapping_prefers_lowest_but_reaches_latest(self, monkeypatch):
        overlays = dict(_FAKE_OVERLAYS)
        overlays["latest"] = frozenset({"brandnew_arch"})
        _fake_overlays(monkeypatch, overlays)
        assert _tier_from_config_mapping({"model_type": "brandnew_arch"}) == "latest"
        # Anything a lower tier ships stays on the lower tier.
        assert _tier_from_config_mapping({"model_type": "qwen3_moe"}) == "530"

    def test_overlay_dir_for_latest(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        (venv_dir / "transformers").mkdir(parents = True)
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        # Unpinned dir is ignored: activation refuses an unpinned sidecar.
        assert tv._overlay_transformers_dir("latest") is None
        (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
        assert tv._overlay_transformers_dir("latest") == str(venv_dir / "transformers")

    def test_probe_order_excludes_unprovisioned_latest(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / ".venv_t5_latest"))
        assert tv._probe_tier_order() == tv._PROBE_TIER_ORDER

    def test_probe_order_includes_provisioned_latest(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        venv_dir.mkdir()
        (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        assert tv._probe_tier_order() == tv._PROBE_TIER_ORDER + ("latest",)

    def test_activation_prepends_latest_dir(self, tmp_path: Path, monkeypatch):
        venv_dir = tmp_path / ".venv_t5_latest"
        venv_dir.mkdir()
        (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
        monkeypatch.setattr(tv, "get_transformers_tier", lambda *a, **k: "latest")
        monkeypatch.setattr(tv, "_ensure_venv_t5_latest_exists", lambda: True)
        old_sys_path = list(sys.path)
        old_pp = os.environ.get("PYTHONPATH")
        try:
            activate_transformers_for_subprocess("some/brand-new-model")
            assert sys.path[0] == str(venv_dir)
            assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(venv_dir)
        finally:
            sys.path[:] = old_sys_path
            if old_pp is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = old_pp

    def test_activation_raises_when_latest_missing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / ".venv_t5_latest"))
        monkeypatch.setattr(tv, "get_transformers_tier", lambda *a, **k: "latest")
        with pytest.raises(RuntimeError, match = "venv_t5_latest"):
            activate_transformers_for_subprocess("some/brand-new-model")


# --- install_latest_transformers: the consent endpoint helper ---


class TestInstallLatestTransformers:
    def test_success_path(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        monkeypatch.setattr(tl, "compat_plan", lambda v: ((), []))
        recorded = {}

        def _fake_ensure(
            version,
            extra_packages = (),
            before_swap = None,
        ):
            recorded["args"] = (version, extra_packages)
            return True

        monkeypatch.setattr(tl, "ensure_latest_transformers_venv", _fake_ensure)
        monkeypatch.setattr(tl, "latest_venv_pinned_version", lambda: "5.13.0")
        result = install_latest_transformers("5.13.0")
        assert result["success"] is True and result["version"] == "5.13.0"
        assert recorded["args"] == ("5.13.0", ())

    def test_version_mismatch_rejected(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        monkeypatch.setattr(
            tl,
            "ensure_latest_transformers_venv",
            lambda v, extra_packages = (): (_ for _ in ()).throw(AssertionError("must not install")),
        )
        result = install_latest_transformers("4.99.0")
        assert result["success"] is False and "not the latest" in result["message"]

    def test_offline_rejected(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        _no_network(monkeypatch)
        result = install_latest_transformers("5.13.0")
        assert result["success"] is False and "offline" in result["message"].lower()

    def test_kill_switch_rejected(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
        _no_network(monkeypatch)
        result = install_latest_transformers("5.13.0")
        assert result["success"] is False

    def test_install_failure_reported(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        monkeypatch.setattr(tl, "compat_plan", lambda v: ((), []))
        monkeypatch.setattr(
            tl,
            "ensure_latest_transformers_venv",
            lambda v, extra_packages = (), before_swap = None: False,
        )
        result = install_latest_transformers("5.13.0")
        assert result["success"] is False and "failed" in result["message"]

    def test_blocked_by_incompatible_deps(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        monkeypatch.setattr(tl, "compat_plan", lambda v: ((), ["numpy>=99.0"]))
        monkeypatch.setattr(
            tl,
            "ensure_latest_transformers_venv",
            lambda v, extra_packages = (): (_ for _ in ()).throw(AssertionError("must not install")),
        )
        result = install_latest_transformers("5.13.0")
        assert result["success"] is False and "numpy>=99.0" in result["message"]

    def test_compat_shadows_passed_to_installer(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
        monkeypatch.setattr(tl, "compat_plan", lambda v: (("tokenizers==0.23.0",), []))
        recorded = {}

        def _fake_ensure(
            version,
            extra_packages = (),
            before_swap = None,
        ):
            recorded["extras"] = extra_packages
            return True

        monkeypatch.setattr(tl, "ensure_latest_transformers_venv", _fake_ensure)
        monkeypatch.setattr(tl, "latest_venv_pinned_version", lambda: "5.13.0")
        result = install_latest_transformers("5.13.0")
        assert result["success"] is True
        assert recorded["extras"] == ("tokenizers==0.23.0",)


class TestCompatPlan:
    def _patch_env(self, monkeypatch, requires, installed):
        monkeypatch.setattr(tl, "_fetch_requires_dist", lambda v: requires)

        def _ver(name):
            from importlib.metadata import PackageNotFoundError

            key = name.lower().replace("_", "-")
            if key not in installed:
                raise PackageNotFoundError(name)
            return installed[key]

        monkeypatch.setattr("importlib.metadata.version", _ver)

    def test_satisfied_env_needs_nothing(self, monkeypatch):
        self._patch_env(
            monkeypatch,
            ["tokenizers<=0.23.0,>=0.22.0", "safetensors>=0.8.0", "numpy>=1.17"],
            {"tokenizers": "0.22.2", "safetensors": "0.8.0", "numpy": "2.4.4"},
        )
        extras, blockers = tl.compat_plan("5.13.0")
        assert extras == () and blockers == []

    def test_unsatisfied_shadowable_dep_pinned(self, monkeypatch):
        self._patch_env(
            monkeypatch,
            ["tokenizers>=0.24.0"],
            {"tokenizers": "0.22.2"},
        )
        monkeypatch.setattr(tl, "_resolve_exact_version", lambda name, spec: "0.24.1")
        extras, blockers = tl.compat_plan("5.99.0")
        assert extras == ("tokenizers==0.24.1",) and blockers == []

    def test_unsatisfied_non_shadowable_dep_blocks(self, monkeypatch):
        self._patch_env(monkeypatch, ["numpy>=99.0"], {"numpy": "2.4.4"})
        extras, blockers = tl.compat_plan("5.99.0")
        assert extras == () and blockers == ["numpy>=99.0"]

    def test_cli_only_dep_ignored(self, monkeypatch):
        self._patch_env(monkeypatch, ["typer"], {})
        extras, blockers = tl.compat_plan("5.13.0")
        assert extras == () and blockers == []

    def test_sidecar_provided_hub_checked_against_recipe_pin(self, monkeypatch):
        self._patch_env(monkeypatch, ["huggingface-hub<2.0,>=1.5.0"], {"huggingface-hub": "0.36.2"})
        extras, blockers = tl.compat_plan("5.13.0")
        assert extras == () and blockers == []  # 1.8.0 sidecar pin satisfies it

    def test_sidecar_provided_hub_out_of_range_blocks(self, monkeypatch):
        self._patch_env(monkeypatch, ["huggingface-hub>=2.1"], {"huggingface-hub": "0.36.2"})
        extras, blockers = tl.compat_plan("5.99.0")
        assert blockers == ["huggingface-hub>=2.1"]

    def test_unfetchable_requires_dist_blocks_install(self, monkeypatch):
        # Proceeding unverified could pin a sidecar whose imports crash workers.
        monkeypatch.setattr(tl, "_fetch_requires_dist", lambda v: None)
        extras, blockers = tl.compat_plan("5.13.0")
        assert extras == () and len(blockers) == 1 and "retry" in blockers[0]

    def test_extra_marker_requirements_skipped(self, monkeypatch):
        self._patch_env(
            monkeypatch,
            ['torch>=99.0; extra == "torch"', 'pytest; python_version < "3.0"'],
            {},
        )
        extras, blockers = tl.compat_plan("5.13.0")
        assert extras == () and blockers == []


def test_get_snapshot_dedupes_concurrent_fetch(monkeypatch):
    """While one thread is fetching, other callers return None instead of stacking fetches."""
    with tl._lock:
        tl._is_fetching = True
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise AssertionError("must not fetch while another fetch is in flight")

    monkeypatch.setattr(tl, "_refresh_snapshot", boom)
    assert tl._get_snapshot() is None
    assert calls["n"] == 0
    tl.clear_caches()


def test_install_serialized():
    """A second install call while one is in progress gets a structured refusal."""
    from utils.transformers_version import try_begin_sidecar_swap

    assert try_begin_sidecar_swap() is True
    out = tl.install_latest_transformers("5.13.0")
    assert out["success"] is False
    assert "already in progress" in out["message"]
    tl.clear_caches()


def test_install_in_progress_reflects_reservation():
    """is_install_in_progress mirrors the shared sidecar swap reservation, so a
    lazy repair (which takes the same reservation) also blocks worker starts."""
    from utils.transformers_version import end_sidecar_swap, try_begin_sidecar_swap

    assert tl.is_install_in_progress() is False
    assert try_begin_sidecar_swap() is True
    try:
        assert tl.is_install_in_progress() is True
    finally:
        end_sidecar_swap()
    assert tl.is_install_in_progress() is False


def test_upgrade_check_sees_nested_model_types(monkeypatch):
    """A supported wrapper with a brand-new nested backbone must still signal."""
    cfg = {
        "model_type": "llava",  # in every installed overlay
        "text_config": {"model_type": "zz_brand_new_llm"},
    }
    monkeypatch.setattr(tl, "_load_config_json", lambda *a, **k: cfg)
    monkeypatch.setattr(
        tl,
        "latest_transformers_supports",
        lambda mt: {
            "pypi_version": "5.13.0",
            "supported_in_pypi": mt == "zz_brand_new_llm",
            "supported_in_main": mt == "zz_brand_new_llm",
        },
    )
    out = tl.check_upgrade_for_model("some-org/wrapped-new-backbone")
    assert out is not None
    assert out["model_type"] == "zz_brand_new_llm"


def test_upgrade_check_ignores_nested_known_types(monkeypatch):
    """All nested types known to installed overlays -> no signal, no remote call."""
    cfg = {
        "model_type": "llava",
        "text_config": {"model_type": "llama"},
        "vision_config": {"model_type": "clip_vision_model"},
    }
    monkeypatch.setattr(tl, "_load_config_json", lambda *a, **k: cfg)
    calls = []
    monkeypatch.setattr(tl, "latest_transformers_supports", lambda mt: calls.append(mt) or None)
    assert tl.check_upgrade_for_model("some-org/normal-vlm") is None
    assert calls == []


def test_upgrade_check_requires_primary_supported(monkeypatch):
    """Latest supporting only a nested type must not prompt: routing still
    cannot load the primary, so the install would not fix the model."""
    cfg = {
        "model_type": "zz_new_wrapper",
        "text_config": {"model_type": "zz_new_llm"},
    }
    monkeypatch.setattr(tl, "_load_config_json", lambda *a, **k: cfg)
    monkeypatch.setattr(
        tl,
        "latest_transformers_supports",
        lambda mt: {
            "pypi_version": "5.13.0",
            "supported_in_pypi": mt == "zz_new_llm",
            "supported_in_main": mt == "zz_new_llm",
        },
    )
    assert tl.check_upgrade_for_model("some-org/half-supported") is None


def test_upgrade_check_requires_every_missing_type(monkeypatch):
    """Primary supported but a nested backbone missing from latest -> no prompt
    (CONFIG_MAPPING would still fail on the sub-config); all supported -> signal
    carries the primary type."""
    cfg = {
        "model_type": "zz_new_wrapper",
        "text_config": {"model_type": "zz_new_llm"},
    }
    monkeypatch.setattr(tl, "_load_config_json", lambda *a, **k: cfg)
    monkeypatch.setattr(
        tl,
        "latest_transformers_supports",
        lambda mt: {
            "pypi_version": "5.13.0",
            "supported_in_pypi": mt == "zz_new_wrapper",
            "supported_in_main": mt == "zz_new_wrapper",
        },
    )
    assert tl.check_upgrade_for_model("some-org/half-supported") is None

    monkeypatch.setattr(
        tl,
        "latest_transformers_supports",
        lambda mt: {
            "pypi_version": "5.13.0",
            "supported_in_pypi": True,
            "supported_in_main": True,
        },
    )
    out = tl.check_upgrade_for_model("some-org/fully-supported")
    assert out is not None and out["model_type"] == "zz_new_wrapper"


def test_install_success_invalidates_capability_caches(monkeypatch):
    """A successful install must drop tier probes, the latest mapping, and the
    vision-detection cache so the new sidecar takes effect without a restart."""
    from utils.models import model_config as mc

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen_factory({}))
    monkeypatch.setattr(tl, "compat_plan", lambda v: ((), []))
    monkeypatch.setattr(
        tl, "ensure_latest_transformers_venv", lambda v, extra_packages = (), before_swap = None: True
    )
    monkeypatch.setattr(tl, "latest_venv_pinned_version", lambda: "5.13.0")

    tv._probe_tier_cache["stale/model"] = "default"
    tv._config_mapping_cache["latest"] = frozenset({"stale_type"})
    tv._config_mapping_cache["default"] = frozenset({"llama"})
    mc._vision_detection_cache[("stale/model", None, False)] = False

    result = install_latest_transformers("5.13.0")
    assert result["success"] is True
    assert tv._probe_tier_cache == {}
    assert "latest" not in tv._config_mapping_cache
    assert tv._config_mapping_cache.get("default") == frozenset({"llama"})  # untouched
    assert mc._vision_detection_cache == {}

    tv._probe_tier_cache.clear()
    tv._config_mapping_cache.clear()
    tl.clear_caches()


def test_vision_subprocess_unions_sidecar_registry():
    """The embedded vision-check script must extend the inlined parent sets with
    the ACTIVE sidecar's registry so sidecar-only architectures classify."""
    from utils.models import model_config as mc

    script = mc._VISION_CHECK_SCRIPT
    ast.parse(script)
    stub_registry = {
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES": {
            "zz_sidecar_vlm": "ZzSidecarForConditionalGeneration"
        },
    }
    ns = {}
    # Exec only the registry-union block against a stubbed sidecar registry.
    body = script.split("from transformers import AutoConfig", 1)[1]
    body = body.split("kwargs = {", 1)[0]
    helpers = script.split("sys.path.insert(0, backend_dir)", 1)[1]
    helpers = helpers.split("try:", 1)[0]
    exec(helpers, ns)

    class _FakeMa:
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = stub_registry[
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES"
        ]

    import sys as _sys
    import types as _types

    fake_pkg = _types.ModuleType("transformers.models.auto")
    fake_pkg.modeling_auto = _FakeMa
    saved = {
        k: _sys.modules.get(k)
        for k in ("transformers.models.auto", "transformers.models.auto.modeling_auto")
    }
    _sys.modules["transformers.models.auto"] = fake_pkg
    _sys.modules["transformers.models.auto.modeling_auto"] = _FakeMa
    try:
        exec(textwrap.dedent(body), ns)
    finally:
        for k, v in saved.items():
            if v is None:
                _sys.modules.pop(k, None)
            else:
                _sys.modules[k] = v

    assert "zz_sidecar_vlm" in ns["_VLM_MODEL_TYPES"]
    assert "ZzSidecarForConditionalGeneration" in ns["_VLM_CLASS_NAMES"]

    class _Cfg:
        architectures = ["ZzSidecarForConditionalGeneration"]
        model_type = "zz_sidecar_vlm"

    assert ns["_is_vlm"](_Cfg()) is True


def test_upgrade_check_mixed_pypi_main_reports_dev_only(monkeypatch):
    """Primary in the PyPI release but a nested type only on main: no install
    may be offered (CONFIG_MAPPING would fail on the nested sub-config), so the
    aggregate must read as main-only."""
    cfg = {
        "model_type": "zz_new_wrapper",
        "text_config": {"model_type": "zz_new_llm"},
    }
    monkeypatch.setattr(tl, "_load_config_json", lambda *a, **k: cfg)
    monkeypatch.setattr(
        tl,
        "latest_transformers_supports",
        lambda mt: {
            "pypi_version": "5.13.0",
            "supported_in_pypi": mt == "zz_new_wrapper",
            "supported_in_main": True,
        },
    )
    out = tl.check_upgrade_for_model("some-org/mixed-support")
    assert out is not None
    assert out["model_type"] == "zz_new_wrapper"
    assert out["supported_in_pypi"] is False  # no install offered
    assert out["supported_in_main"] is True


def test_install_endpoint_not_mounted_on_v1():
    """The consented pip-install endpoint is a Studio admin action; it must live
    on studio_router (kept off the OpenAI-compatible /v1 mount), not router."""
    from routes import inference as ri

    path = "/install-latest-transformers"
    assert path in [r.path for r in ri.studio_router.routes]
    assert path not in [r.path for r in ri.router.routes]


def test_kill_switch_removes_provisioned_latest_from_routing(tmp_path, monkeypatch):
    """UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS must roll back a provisioned latest
    sidecar: no overlay mapping, no probe participation, no file deletion needed."""
    venv_dir = tmp_path / ".venv_t5_latest"
    (venv_dir / "transformers").mkdir(parents = True)
    (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))

    assert tv._overlay_transformers_dir("latest") == str(venv_dir / "transformers")
    assert tv._probe_tier_order() == tv._PROBE_TIER_ORDER + ("latest",)

    monkeypatch.setenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "1")
    tv._config_mapping_cache.pop("latest", None)
    assert tv._overlay_transformers_dir("latest") is None
    assert tv._probe_tier_order() == tv._PROBE_TIER_ORDER
    tv._config_mapping_cache.pop("latest", None)


def test_repair_failure_preserves_pin_and_live_dir(tmp_path, monkeypatch):
    """A failed lazy repair must not delete the incomplete-but-pinned live
    sidecar: the pin survives so a later attempt can still repair it."""
    venv_dir = tmp_path / ".venv_t5_latest"
    venv_dir.mkdir()
    (venv_dir / tv._LATEST_PIN_MARKER).write_text("5.13.0")
    (venv_dir / "partial_file").write_text("x")
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))
    monkeypatch.setattr(tv, "_venv_dir_is_valid", lambda *a: False)
    monkeypatch.setattr(tv, "_ensure_venv_dir", lambda *a, **k: False)

    from utils.transformers_version import latest_venv_pinned_version

    assert tv._ensure_venv_t5_latest_exists() is False
    assert venv_dir.is_dir()
    assert (venv_dir / "partial_file").exists()
    assert latest_venv_pinned_version() == "5.13.0"
    assert not (tmp_path / ".venv_t5_latest.staging").exists()


def test_failed_staging_install_removes_staging_dir(tmp_path, monkeypatch):
    """A pip failure inside _ensure_venv_dir returns False without raising, so
    the except cleanup never runs; the partial staging dir must still go."""
    venv_dir = tmp_path / ".venv_t5_latest"
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(venv_dir))

    def _fake_ensure(dir_, packages, label):
        Path(dir_).mkdir(parents = True, exist_ok = True)
        (Path(dir_) / "partial").write_text("x")
        return False

    monkeypatch.setattr(tv, "_ensure_venv_dir", _fake_ensure)
    assert ensure_latest_transformers_venv("5.13.0") is False
    assert not Path(str(venv_dir) + ".staging").exists()
