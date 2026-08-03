# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Unsloth shim over the shared unsloth_zoo Xet -> HTTP fallback.

The transport-policy matrix is tested once in unsloth_zoo; here we assert only the
Unsloth seam: re-exporting the shared API and injecting the marker-aware
prepare_cache_for_transport on the HTTP retry. CPU-only, no network, no real subprocess.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy/unavailable deps before importing the module under test. Use real structlog when present;
# a bare stub would break later modules that log at import time.
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
except Exception:  # noqa: BLE001 - still collect degraded-path tests when unsloth_zoo is unavailable
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
    """A Xet stall retries over HTTP and the shim runs Unsloth's marker-aware
    ``prepare_cache_for_transport(..., 'http')`` before the retry."""
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
        force_download = False,
    ):
        seen_disable_xet.append(disable_xet)
        return ("ok", "/cache/model.gguf") if disable_xet else ("stall", None)

    monkeypatch.setattr(shared, "_run_download_attempt", fake_attempt)

    prepared = []
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport",
        lambda repo_type, repo_id, mode, *a, **k: prepared.append(
            (repo_type, repo_id, mode, k.get("root"))
        ),
    )

    selected_cache = "/captured/hub"
    out = xf.hf_hub_download_with_xet_fallback(
        DL_REPO,
        FILE,
        None,
        cache_dir = selected_cache,
    )
    assert out == "/cache/model.gguf"
    assert seen_disable_xet == [False, True]  # Xet first, then HTTP
    assert prepared == [
        ("model", DL_REPO, "http", Path(selected_cache))
    ], "shim must prepare the cache captured by the download"


def test_shim_snapshot_injects_studio_prepare(monkeypatch):
    """The snapshot wrapper forwards Unsloth's marker-aware prep, like the file wrapper."""
    captured = {}

    def fake_snapshot(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["prepare_for_http_fn"] = kwargs.get("prepare_for_http_fn")
        return "/tmp/snap-dir"

    monkeypatch.setattr(xf, "_shared_snapshot_download_with_xet_fallback", fake_snapshot)
    selected_cache = "/captured/hub"
    out = xf.snapshot_download_with_xet_fallback(
        "org/model",
        cache_dir = selected_cache,
    )
    assert out == "/tmp/snap-dir"
    assert captured["repo_id"] == "org/model"
    prepared = []
    monkeypatch.setattr(
        "hub.utils.download_registry.prepare_cache_for_transport",
        lambda repo_type, repo_id, mode, *a, **k: prepared.append(
            (repo_type, repo_id, mode, k.get("root"))
        ),
    )
    captured["prepare_for_http_fn"]("model", "org/model")
    assert prepared == [("model", "org/model", "http", Path(selected_cache))]


def test_degrades_gracefully_without_shared_helper(monkeypatch):
    """On an older unsloth_zoo lacking the shared helper, the shim still imports (Unsloth
    boots) and exposes stub API doing plain HF downloads with the watchdog disabled."""
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

        # Degraded mode still emits heartbeats so the inactivity deadline is not tripped.
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

        # Cancellation still holds: an already-set cancel_event aborts before the HF download.
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
    """When unsloth_zoo is absent entirely, the import raises
    ModuleNotFoundError(name='unsloth_zoo') (top-level package). Guard that the shim still
    degrades and does not re-raise, breaking every Unsloth import that pulls it in."""
    import importlib

    class _BlockZoo:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            # Whole package absent, so ModuleNotFoundError.name is the top-level 'unsloth_zoo'.
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
    """unsloth_zoo can be installed yet fail to import when torch is missing (llama.cpp/GGUF-only
    Unsloth), raising ImportError not ModuleNotFoundError. The shim must degrade for that too."""
    import importlib

    class _BlockWithImportError:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo.hf_xet_fallback":
                # Mirror a torch-less install: a plain ImportError with no .name.
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
    """GPU detection in unsloth_zoo's __init__ raises NotImplementedError on a GPU-less host. The shim
    retries under UNSLOTH_ZOO_DISABLE_GPU_INIT=1, restores the env, and degrades if the retry fails.
    The backend loads lazily (first use of a heavy helper), so this triggers the load explicitly
    before asserting the retry/degrade behavior."""
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
            # Crash is in unsloth_zoo's __init__, so intercept "unsloth_zoo" itself (the parent).
            if name == "unsloth_zoo":
                # Record the env each attempt sees; raise the no-GPU error both times so the shim
                # degrades.
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
        # Import is light (lazy backend); unsloth_zoo not loaded yet.
        assert seen_env == [], seen_env
        # First use of a heavy helper triggers the load (attempt without the light env, then a retry
        # with it set); accessing DownloadStallError drives it via __getattr__.
        stall_error = degraded.DownloadStallError
        assert seen_env == [None, "1"], seen_env
        # Both attempts raised -> Unsloth still boots in degraded mode.
        assert issubclass(stall_error, RuntimeError)
        # The env override must not leak past the load.
        assert os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") is None
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        sys.modules.update(saved)
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim


def test_a_worker_spawned_during_the_gpu_init_retry_does_not_inherit_the_override(monkeypatch):
    """The shim sets UNSLOTH_ZOO_DISABLE_GPU_INIT=1 process-wide while it retries an optional
    import, and unsloth_zoo answers that flag with STUB triton and bitsandbytes, so a child that
    inherited it would run for life against no-ops and never clear it."""
    import importlib
    import os

    from utils.child_stdio import utf8_child_env

    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)
    child_envs = []

    class _SpawnsAWorkerMidImport:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo":
                # A concurrent request lands mid-retry and spawns its worker right here.
                child_envs.append(utf8_child_env())
                raise NotImplementedError("Unsloth cannot find any torch accelerator")
            return None

    finder = _SpawnsAWorkerMidImport()
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
        shim = importlib.import_module("utils.hf_xet_fallback")
        shim.DownloadStallError  # drives the load: plain attempt, then the retry
        assert len(child_envs) == 2, child_envs
        # The retry is the attempt that sets it; neither child may see it.
        assert os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") is None
        for env in child_envs:
            assert "UNSLOTH_ZOO_DISABLE_GPU_INIT" not in env, env
            assert env["PYTHONIOENCODING"] == "utf-8"
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        sys.modules.update(saved)
        if saved_shim is not None:
            sys.modules["utils.hf_xet_fallback"] = saved_shim


def test_a_spawn_cannot_overlap_the_loader_env_override_window():
    """multiprocessing spawn copies the parent's LIVE os.environ and takes no env argument, so the
    only way to keep the shim's transient UNSLOTH_ZOO_DISABLE_GPU_INIT out of a worker is that a
    spawn cannot start while a loader holds it; a child that inherits it silently trains against
    unsloth_zoo's stub triton and bitsandbytes. Structural on purpose: it asserts the two share one
    lock rather than trying to hit a microsecond window by timing."""
    import threading

    from utils.hf_cache_settings import child_environment_for_spawn

    loader_holds = threading.Event()
    release_loader = threading.Event()
    spawn_started = threading.Event()
    spawn_entered = threading.Event()

    def _loader():
        with xf.env_override_barrier():
            loader_holds.set()
            release_loader.wait(5.0)

    def _spawner():
        spawn_started.set()
        with child_environment_for_spawn({}):
            spawn_entered.set()

    t_loader = threading.Thread(target = _loader, daemon = True)
    t_loader.start()
    assert loader_holds.wait(5.0), "loader never took the barrier"

    t_spawn = threading.Thread(target = _spawner, daemon = True)
    t_spawn.start()
    assert spawn_started.wait(5.0)
    try:
        assert not spawn_entered.wait(0.5), "a spawn started inside the loader override window"
    finally:
        release_loader.set()
    assert spawn_entered.wait(5.0), "the spawn never proceeded after the loader let go"
    t_loader.join(5.0)
    t_spawn.join(5.0)


def test_the_spawn_barrier_is_reentrant():
    """child_environment_for_spawn nests (an inference respawn inside a training start), which its
    RLock _spawn_env_lock allows, so the barrier it now takes alongside must be reentrant too or the
    inner enter deadlocks. Bounded on purpose: a plain Lock hangs the suite rather than failing."""
    import threading

    from utils.hf_cache_settings import child_environment_for_spawn

    done = threading.Event()

    def _nest():
        with child_environment_for_spawn({"HF_HUB_CACHE": "outer"}):
            with child_environment_for_spawn({"HF_HUB_CACHE": "inner"}):
                pass
        done.set()

    worker = threading.Thread(target = _nest, daemon = True)
    worker.start()
    assert done.wait(10.0), "a nested spawn deadlocked on the loader barrier"
    worker.join(5.0)


def test_an_operator_set_gpu_init_override_still_reaches_the_child(monkeypatch):
    """Only the loader's own transient value is dropped. Someone who exported the flag themselves
    on a GPU-less box meant it, and their children must keep it."""
    from utils.child_stdio import utf8_child_env

    monkeypatch.setenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", "1")
    monkeypatch.setattr(xf, "_gpu_init_override_depth", 0, raising = False)
    assert utf8_child_env()["UNSLOTH_ZOO_DISABLE_GPU_INIT"] == "1"


def test_importing_child_should_disable_xet_stays_light(monkeypatch):
    """Regression guard for the stale-transformers-sidecar bug: importing the shim (and
    ``child_should_disable_xet``) must NOT pull in ``transformers``/``unsloth_zoo``. The worker calls
    this at startup to decide the Xet env flip BEFORE activating the sidecar; an eager import here
    would cache the default transformers 4.57.x in sys.modules, defeating the sidecar sys.path prepend
    and breaking 5.x models (Qwen3.5/GLM/gemma-4)."""
    import importlib

    for name in [
        m
        for m in list(sys.modules)
        if m == "transformers"
        or m.startswith("transformers.")
        or m == "unsloth_zoo"
        or m.startswith("unsloth_zoo.")
        or m == "utils.hf_xet_fallback"
    ]:
        monkeypatch.delitem(sys.modules, name, raising = False)

    mod = importlib.import_module("utils.hf_xet_fallback")
    # The lightweight decision works without the heavy backend.
    assert mod.child_should_disable_xet({"disable_xet": True}) is True
    assert mod.child_should_disable_xet({}) is False
    # And nothing heavy was imported as a side effect.
    assert "transformers" not in sys.modules, "importing the shim must not import transformers"
    assert "unsloth_zoo" not in sys.modules, "importing the shim must not import unsloth_zoo"


def test_start_watchdog_drops_kwargs_the_installed_zoo_cannot_take(monkeypatch):
    """Version-skew adapter, and load-bearing: the floor's start_watchdog is keyword-only with no
    **kwargs and no connect_timeout, so passing one raises TypeError into the caller's
    `except Exception` and the watchdog silently never starts. That is the feature entirely off.
    """
    import threading

    import utils.hf_xet_fallback as shim

    seen = {}

    def _old_signature_watchdog(
        *,
        repo_ids,
        on_stall,
        repo_type = "model",
        cache_dir = None,
        interval = 30.0,
        stall_timeout = 180.0,
        xet_disabled = False,
        on_heartbeat = None,
        watch_new_partials_only = False,
        baseline_incomplete_blobs = None,
        child_pid = None,
    ):
        seen.update(locals())
        return threading.Event()

    class _FakeShared:
        start_watchdog = staticmethod(_old_signature_watchdog)

    monkeypatch.setattr(shim, "_shared", _FakeShared, raising = False)
    monkeypatch.setattr(shim, "_shared_available", True, raising = False)

    stop = shim.start_watchdog(
        repo_ids = ["a/b"],
        on_stall = lambda _m: None,
        watch_new_partials_only = True,
        child_pid = 1234,
        connect_timeout = 600.0,  # only on the unreleased zoo
    )
    assert stop is not None, "the watchdog did not start"
    assert seen["watch_new_partials_only"] is True, "a SUPPORTED kwarg was dropped"
    assert seen["child_pid"] == 1234


def test_start_watchdog_passes_everything_to_a_zoo_that_accepts_it(monkeypatch):
    """A newer zoo must still receive the newer knobs."""
    import threading

    import utils.hf_xet_fallback as shim

    seen = {}

    def _new_signature_watchdog(**kwargs):
        seen.update(kwargs)
        return threading.Event()

    class _FakeShared:
        start_watchdog = staticmethod(_new_signature_watchdog)

    monkeypatch.setattr(shim, "_shared", _FakeShared, raising = False)
    monkeypatch.setattr(shim, "_shared_available", True, raising = False)

    shim.start_watchdog(repo_ids = ["a/b"], on_stall = lambda _m: None, connect_timeout = 600.0)
    assert seen["connect_timeout"] == 600.0, "a newer zoo lost the kwarg it supports"
