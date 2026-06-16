# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for hf_hub_download_with_xet_fallback's transport policy.

The per-attempt download (``_run_download_attempt``) is monkeypatched to a
synchronous fake so no real process spawns: we exercise only the decision logic
(cached short-circuit, cancel, deterministic-error propagation, and the single
Xet->HTTP fallback on a stall). No GPU, no network, no subprocess.
"""

from __future__ import annotations

import sys
import threading
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import huggingface_hub

import utils.hf_xet_fallback as xf

REPO, FILE = "ztest/xet-dl", "model-Q4_K_XL.gguf"


@pytest.fixture(autouse = True)
def _no_real_cache_hit(monkeypatch):
    """Default: the cached probe misses, so the download path is exercised.
    Individual tests override this when they want a cache hit."""
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)


class _FakeAttempt:
    """Records calls to the download seam and returns scripted results."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []  # list of (disable_xet,)

    def __call__(self, repo_id, filename, token, *, repo_type, disable_xet,
                 cancel_event, stall_timeout, interval, grace_period, on_status):
        self.calls.append(_types.SimpleNamespace(
            repo_id = repo_id, filename = filename, disable_xet = disable_xet,
            repo_type = repo_type,
        ))
        return self._results[len(self.calls) - 1]


def _install(monkeypatch, results):
    fake = _FakeAttempt(results)
    monkeypatch.setattr(xf, "_run_download_attempt", fake)
    return fake


def test_cached_file_short_circuits(monkeypatch, tmp_path):
    cached = tmp_path / "cached.gguf"
    cached.write_bytes(b"\0" * 8)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached))
    fake = _install(monkeypatch, [])  # must not be called

    out = xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert out == str(cached)
    assert fake.calls == [], "spawned a download for an already-cached file"


def test_cancel_before_start_raises_no_attempt(monkeypatch):
    fake = _install(monkeypatch, [])
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(REPO, FILE, None, cancel_event = ev)
    assert fake.calls == []


def test_nonstall_error_propagates_without_fallback(monkeypatch):
    fake = _install(monkeypatch, [("error", "RepositoryNotFoundError: 404 not found")])
    with pytest.raises(RuntimeError, match = "RepositoryNotFoundError"):
        xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert len(fake.calls) == 1, "deterministic error must not trigger an HTTP fallback"
    assert fake.calls[0].disable_xet is False


def test_immediate_success_uses_xet_only(monkeypatch):
    prepared = []
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport",
        lambda *a, **k: prepared.append(a),
    )
    fake = _install(monkeypatch, [("ok", "/cache/model.gguf")])
    out = xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 1 and fake.calls[0].disable_xet is False
    assert prepared == [], "no cache prep should run when Xet succeeds first try"


def test_stall_then_http_fallback_succeeds(monkeypatch):
    prepared = []
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport",
        lambda repo_type, repo_id, mode, *a, **k: prepared.append((repo_type, repo_id, mode)),
    )
    fake = _install(monkeypatch, [("stall", None), ("ok", "/cache/model.gguf")])

    out = xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert out == "/cache/model.gguf"
    assert len(fake.calls) == 2
    assert fake.calls[0].disable_xet is False  # Xet first
    assert fake.calls[1].disable_xet is True   # HTTP fallback
    assert prepared == [("model", REPO, "http")], "must prep cache for HTTP before the retry"


def test_second_stall_raises_download_stall_error(monkeypatch):
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport", lambda *a, **k: None
    )
    fake = _install(monkeypatch, [("stall", None), ("stall", None)])
    with pytest.raises(xf.DownloadStallError):
        xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert len(fake.calls) == 2


def test_cancelled_midattempt_raises_no_fallback(monkeypatch):
    fake = _install(monkeypatch, [("cancelled", None)])
    with pytest.raises(RuntimeError, match = "Cancelled"):
        xf.hf_hub_download_with_xet_fallback(REPO, FILE, None)
    assert len(fake.calls) == 1


def test_per_file_independent_fallback(monkeypatch):
    """A stalled shard falls back; a sibling shard that succeeds does not."""
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport", lambda *a, **k: None
    )
    # shard A: immediate ok (no fallback). shard B: stall then ok (one fallback).
    fake = _install(monkeypatch, [("ok", "/a"), ("stall", None), ("ok", "/b")])
    assert xf.hf_hub_download_with_xet_fallback(REPO, "shardA.gguf", None) == "/a"
    assert xf.hf_hub_download_with_xet_fallback(REPO, "shardB.gguf", None) == "/b"
    assert [c.disable_xet for c in fake.calls] == [False, False, True]
