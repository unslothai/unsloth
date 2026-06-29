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

# Stub heavy/unavailable deps before importing the module under test. Use the real structlog when
# present; a bare stub would break later modules that log at import time.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules["structlog"] = _types.ModuleType("structlog")

import huggingface_hub

try:
    import unsloth_zoo.hf_xet_fallback as _shared_mod
    shared = _shared_mod
except Exception:  # noqa: BLE001
    # Still collect the degraded-path test when unsloth_zoo is missing / too old / torch-less /
    # GPU-less -- the same failure cases the shim itself degrades for.
    shared = None

import utils.hf_xet_fallback as xf


DL_REPO, FILE = "ztest/xet-dl", "model-Q4_K_XL.gguf"


def _requires_shared():
    if shared is None:
        pytest.skip("unsloth_zoo.hf_xet_fallback is not installed in this environment")


def test_shim_reexports_shared_api():
    _requires_shared()
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
    _requires_shared()
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


def test_shim_snapshot_injects_studio_prepare(monkeypatch):
    """The snapshot wrapper forwards Studio's marker-aware prep, like the file wrapper."""
    captured = {}

    def fake_snapshot(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["prepare_for_http_fn"] = kwargs.get("prepare_for_http_fn")
        return "/tmp/snap-dir"

    monkeypatch.setattr(xf, "_shared_snapshot_download_with_xet_fallback", fake_snapshot)
    out = xf.snapshot_download_with_xet_fallback("org/model")
    assert out == "/tmp/snap-dir"
    assert captured["repo_id"] == "org/model"
    assert captured["prepare_for_http_fn"] is xf._studio_prepare_for_http


def test_degrades_gracefully_without_shared_helper(monkeypatch):
    """On an older unsloth_zoo that lacks the shared helper, the shim must still
    import (Studio boots) and provide stub API that does plain HF downloads with
    the watchdog disabled, instead of crashing at startup."""
    import importlib

    class _BlockShared:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo.hf_xet_fallback":
                raise ModuleNotFoundError(f"No module named '{name}'", name = name)
            return None

    finder = _BlockShared()
    saved_shared = sys.modules.pop("unsloth_zoo.hf_xet_fallback", None)
    saved_shim = sys.modules.pop("utils.hf_xet_fallback", None)
    sys.meta_path.insert(0, finder)
    try:
        degraded = importlib.import_module("utils.hf_xet_fallback")

        # Boots without raising and mirrors the shared API surface.
        assert issubclass(degraded.DownloadStallError, RuntimeError)
        assert degraded.child_should_disable_xet({"disable_xet": True}) is True
        assert degraded.get_hf_download_state(["x"]) is None  # unmeasurable
        event = degraded.start_watchdog(repo_ids = ["x"], on_stall = lambda m: None)
        assert hasattr(event, "set") and not event.is_set()  # never fires

        # Degraded mode still emits heartbeats so the orchestrator's inactivity
        # deadline is not tripped during a long load.
        import time as _time

        beats = []
        hb_stop = degraded.start_watchdog(
            repo_ids = ["x"],
            on_stall = lambda m: None,
            on_heartbeat = beats.append,
            interval = 0.02,
        )
        try:
            deadline = _time.monotonic() + 2.0
            while not beats and _time.monotonic() < deadline:
                _time.sleep(0.02)
            assert beats, "degraded watchdog emitted no heartbeat"
        finally:
            hb_stop.set()

        # Downloads fall back to plain huggingface_hub (no watchdog, no crash).
        called = {}

        def _fake_snapshot(repo_id, **kwargs):
            called["repo_id"] = repo_id
            return "/snap-dir"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
        assert degraded.snapshot_download_with_xet_fallback("org/model") == "/snap-dir"
        assert called["repo_id"] == "org/model"

        # Cancellation contract is preserved even in degraded mode: an already-set
        # cancel_event must abort before starting the plain HF download.
        import threading as _threading

        cancelled = _threading.Event()
        cancelled.set()
        called.clear()
        with pytest.raises(RuntimeError, match = "Cancelled"):
            degraded.snapshot_download_with_xet_fallback("org/model", cancel_event = cancelled)
        assert "repo_id" not in called, "degraded download ran despite cancellation"
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        if saved_shared is not None:
            sys.modules["unsloth_zoo.hf_xet_fallback"] = saved_shared
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim


def test_degrades_when_unsloth_zoo_entirely_absent():
    """Studio-only test/build environments do not install unsloth_zoo at all. The
    shim must degrade for that case too: ``import unsloth_zoo.hf_xet_fallback``
    then raises ModuleNotFoundError(name='unsloth_zoo') (the top-level package),
    not name='unsloth_zoo.hf_xet_fallback'. Regression guard for the shim re-raising
    and breaking every Studio import that transitively pulls it in."""
    import importlib

    class _BlockZoo:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            # Simulate the whole package being absent: the first missing component
            # is 'unsloth_zoo', so that is what ModuleNotFoundError.name carries.
            if name == "unsloth_zoo" or name.startswith("unsloth_zoo."):
                raise ModuleNotFoundError("No module named 'unsloth_zoo'", name = "unsloth_zoo")
            return None

    finder = _BlockZoo()
    saved = {
        k: v
        for k, v in list(sys.modules.items())
        if k == "unsloth_zoo" or k.startswith("unsloth_zoo.")
    }
    for k in saved:
        del sys.modules[k]
    saved_shim = sys.modules.pop("utils.hf_xet_fallback", None)
    sys.meta_path.insert(0, finder)
    try:
        degraded = importlib.import_module("utils.hf_xet_fallback")
        # Boots without raising and exposes the stub API.
        assert issubclass(degraded.DownloadStallError, RuntimeError)
        assert degraded.get_hf_download_state(["x"]) is None
        event = degraded.start_watchdog(repo_ids = ["x"], on_stall = lambda m: None)
        assert hasattr(event, "set") and not event.is_set()
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        sys.modules.update(saved)
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim


def test_degrades_when_shared_helper_import_raises_importerror():
    """unsloth_zoo can be installed yet fail to import because a heavy dependency
    it initializes at package import (torch) is missing -- a llama.cpp/GGUF-only
    Studio install has no torch. That raises ImportError (not ModuleNotFoundError),
    so the shim must degrade for it too rather than crash the Studio server at
    startup. Regression guard for the inference E2E jobs."""
    import importlib

    class _BlockWithImportError:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo.hf_xet_fallback":
                # Mirror unsloth_zoo/__init__ on a torch-less install: a plain ImportError with no .name.
                raise ImportError("Unsloth: Pytorch is not installed.")
            return None

    finder = _BlockWithImportError()
    saved_shared = sys.modules.pop("unsloth_zoo.hf_xet_fallback", None)
    saved_zoo = sys.modules.pop("unsloth_zoo", None)
    saved_shim = sys.modules.pop("utils.hf_xet_fallback", None)
    sys.meta_path.insert(0, finder)
    try:
        degraded = importlib.import_module("utils.hf_xet_fallback")
        assert issubclass(degraded.DownloadStallError, RuntimeError)
        assert degraded.get_hf_download_state(["x"]) is None
        event = degraded.start_watchdog(repo_ids = ["x"], on_stall = lambda m: None)
        assert hasattr(event, "set") and not event.is_set()
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        if saved_shared is not None:
            sys.modules["unsloth_zoo.hf_xet_fallback"] = saved_shared
        if saved_zoo is not None:
            sys.modules["unsloth_zoo"] = saved_zoo
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim


def test_retries_under_light_gpu_init_when_import_fails(monkeypatch):
    """unsloth_zoo's __init__ runs GPU detection that raises NotImplementedError on a GPU-less host
    (CPU GGUF Studio). The shim must retry the import under UNSLOTH_ZOO_DISABLE_GPU_INIT=1 (its light
    path), then restore the env; if even the retry fails, it degrades instead of crashing."""
    import importlib
    import os

    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)
    seen_env = []

    class _GpuGatedBlocker:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            # The crash is in unsloth_zoo's __init__ (the PARENT import fails before the submodule), so
            # intercept "unsloth_zoo" itself (works whether or not it is installed; finder is first).
            if name == "unsloth_zoo":
                # Record the env each attempt sees; raise the no-GPU error both times so the shim
                # degrades (the recovery-succeeds path is covered by real unsloth_zoo on a CPU host).
                seen_env.append(os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT"))
                raise NotImplementedError("Unsloth cannot find any torch accelerator")
            return None

    finder = _GpuGatedBlocker()
    saved = {
        k: v
        for k, v in list(sys.modules.items())
        if k == "unsloth_zoo" or k.startswith("unsloth_zoo.")
    }
    for k in saved:
        del sys.modules[k]
    saved_shim = sys.modules.pop("utils.hf_xet_fallback", None)
    sys.meta_path.insert(0, finder)
    try:
        degraded = importlib.import_module("utils.hf_xet_fallback")
        # First attempt without the light env, then a retry with it set.
        assert seen_env == [None, "1"], seen_env
        # Both attempts raised -> Studio still boots in degraded mode.
        assert issubclass(degraded.DownloadStallError, RuntimeError)
        # The env override must not leak past the import.
        assert os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") is None
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        sys.modules.update(saved)
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim
