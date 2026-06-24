# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio shim over the shared unsloth_zoo Xet -> HTTP stall fallback.

The watchdog and transport-policy matrix (cached short-circuit, cancel, error
propagation, the single Xet -> HTTP retry, the snapshot variant, the knobs) is
tested once in unsloth_zoo (tests/test_hf_xet_fallback.py). Here we assert only
the Studio-specific seam: the shim re-exports the shared API and injects Studio's
marker-aware prepare_cache_for_transport on the HTTP retry. CPU-only, no network,
no real subprocess (the per-attempt download seam is monkeypatched).
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy/unavailable deps before importing the module under test. Use the
# real structlog when present; a bare stub left in sys.modules would break later
# modules that log at import time.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules["structlog"] = _types.ModuleType("structlog")

import huggingface_hub

import unsloth_zoo.hf_xet_fallback as shared
import utils.hf_xet_fallback as xf


DL_REPO, FILE = "ztest/xet-dl", "model-Q4_K_XL.gguf"


def test_shim_reexports_shared_api():
    assert xf.DownloadStallError is shared.DownloadStallError
    for name in (
        "start_watchdog",
        "get_hf_download_state",
        "child_should_disable_xet",
        "hf_hub_download_with_xet_fallback",
        "snapshot_download_with_xet_fallback",
    ):
        assert hasattr(xf, name), f"shim missing {name}"


def test_child_should_disable_xet_truth_table():
    assert xf.child_should_disable_xet({"disable_xet": True}) is True
    assert xf.child_should_disable_xet({"disable_xet": False}) is False
    assert xf.child_should_disable_xet({}) is False


def test_shim_injects_studio_prepare_on_http_retry(monkeypatch):
    """A stall on Xet retries over HTTP, and the shim runs Studio's marker-aware
    ``prepare_cache_for_transport(..., 'http')`` before the retry (not the generic
    delete-incompletes default)."""
    for var in ("UNSLOTH_DISABLE_XET", "UNSLOTH_STABLE_DOWNLOADS", "HF_HUB_DISABLE_XET"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)

    seen_disable_xet = []

    def fake_attempt(
        repo_id,
        *,
        kind,
        params,
        token,
        repo_type,
        disable_xet,
        cancel_event,
        stall_timeout,
        interval,
        grace_period,
        on_status,
    ):
        seen_disable_xet.append(disable_xet)
        return ("ok", "/cache/model.gguf") if disable_xet else ("stall", None)

    monkeypatch.setattr(shared, "_run_download_attempt", fake_attempt)

    prepared = []
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport",
        lambda repo_type, repo_id, mode, *a, **k: prepared.append((repo_type, repo_id, mode)),
    )

    out = xf.hf_hub_download_with_xet_fallback(DL_REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert seen_disable_xet == [False, True]  # Xet first, then HTTP
    assert prepared == [("model", DL_REPO, "http")], "shim must run Studio's marker-aware prep"
