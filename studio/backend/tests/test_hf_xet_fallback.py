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


@pytest.fixture(autouse = True)
def _restore_shim_module_identity():
    """Put BOTH bindings of the shim back after every test in this file.

    The degraded-path tests below drop ``utils.hf_xet_fallback`` from ``sys.modules`` and import a
    throwaway copy. Restoring only the ``sys.modules`` entry is not enough: the import machinery
    also rebinds the module as an attribute of the ``utils`` PACKAGE, and that binding keeps
    pointing at the throwaway. The two then disagree, and a later test in the same process
    monkeypatches one copy (pytest resolves a dotted target through the package attribute) while
    the code under test imports the other, so the patch silently does nothing and the real
    downloader runs against the network. Caught by
    tests/test_video_backend.py::test_fetch_te_prequant_only_reports_what_it_downloaded, which
    reached the Hub and got a 401 when it ran after this file."""
    import utils as _utils_pkg

    original = sys.modules.get("utils.hf_xet_fallback")
    original_attr = getattr(_utils_pkg, "hf_xet_fallback", None)
    try:
        yield
    finally:
        if original is not None:
            sys.modules["utils.hf_xet_fallback"] = original
        if original_attr is not None:
            _utils_pkg.hf_xet_fallback = original_attr


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
    # This seam checks the Xet -> HTTP transition, not the independently configurable
    # number of Xet retries (newer Zoo releases default to two).
    monkeypatch.setenv("UNSLOTH_XET_ATTEMPTS", "1")
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


def test_no_light_gpu_init_retry_on_an_accelerator_host(monkeypatch):
    """The UNSLOTH_ZOO_DISABLE_GPU_INIT retry makes unsloth_zoo take its MLX/CPU path, which injects
    triton and bitsandbytes STUBS into sys.modules for the whole process. On a GPU host whose
    unsloth_zoo import failed for an unrelated reason (a bitsandbytes/CUDA mismatch, say), those
    stubs raise "called on Apple Silicon / MLX" from the first CUDA-only kernel a later GGUF or
    compiled diffusion generation touches, so a healthy GPU starts 500ing. The shim must degrade
    instead of retrying there."""
    import importlib
    import os

    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)
    attempts = []

    class _Blocker:
        def find_spec(
            self,
            name,
            path = None,
            target = None,
        ):
            if name == "unsloth_zoo":
                attempts.append(os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT"))
                raise RuntimeError("CUDA Setup failed despite GPU being available")
            return None

    finder = _Blocker()
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
        monkeypatch.setattr(shim, "_gpu_present", lambda: True)
        assert shim._load_shared() is False
        # Exactly ONE attempt, made without the light-init flag.
        assert attempts == [None], attempts
        assert os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT") is None
    finally:
        sys.meta_path.remove(finder)
        sys.modules.pop("utils.hf_xet_fallback", None)
        sys.modules.update(saved)
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
        # The retry only applies to a host with no accelerator (see _gpu_present); pin that on the freshly imported module.
        monkeypatch.setattr(degraded, "_gpu_present", lambda: False)
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
        # There is only a retry to spawn into on a host with no accelerator (see _gpu_present),
        # so pin that rather than letting the runner's hardware decide the assertion.
        monkeypatch.setattr(shim, "_gpu_present", lambda: False)
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


def test_first_download_dispatch_loads_zoo_once(monkeypatch):
    """Import stays light, but a real download dispatch activates the shared Zoo helper."""
    import utils.hf_xet_fallback as shim

    calls: list[str] = []

    class _FakeShared:
        @staticmethod
        def hf_hub_download_with_xet_fallback(*args, **kwargs):
            calls.append("download")
            return "/cache/model.bin"

    monkeypatch.setattr(shim, "_shared", _FakeShared, raising = False)
    monkeypatch.setattr(shim, "_load_shared", lambda: calls.append("load") or True, raising = True)

    assert (
        shim.hf_hub_download_with_xet_fallback("org/model", "model.bin", None, cache_dir = "/cache")
        == "/cache/model.bin"
    )
    assert calls == ["load", "download"]


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


def test_apply_xet_env_delegates_to_the_zoo(monkeypatch):
    """One rule, in one place: Unsloth asks the zoo to size the worker rather than sizing it too."""
    import types

    import utils.hf_xet_fallback as shim

    seen = {}

    def _apply(env, **kwargs):
        seen["env"] = env
        seen["kwargs"] = kwargs
        env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] = "123"
        return {"HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": "123"}

    monkeypatch.setattr(
        shim, "_load_optional", lambda _name: types.SimpleNamespace(apply_xet_env = _apply)
    )
    env = {"HF_HUB_DISABLE_XET": "0"}
    assert shim.apply_xet_env(env) == {"HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": "123"}
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"] == "123"
    assert seen["env"] is env, "the worker's own env has to be the one that gets sized"
    # Short Xet timeouts suit a child our ladder supervises; process-wide they would not.
    assert seen["kwargs"]["fail_fast"] is True


def test_apply_xet_env_returns_none_when_the_zoo_cannot_size(monkeypatch):
    """None, not {}: an empty write is a legitimate result, so the caller needs the two apart to
    know whether to fall back to clearing the high-performance flag itself."""
    import types

    import utils.hf_xet_fallback as shim

    monkeypatch.setattr(shim, "_load_optional", lambda _name: None)
    assert shim.apply_xet_env({}) is None

    monkeypatch.setattr(shim, "_load_optional", lambda _name: types.SimpleNamespace())
    assert shim.apply_xet_env({}) is None, "an older zoo without apply_xet_env must read as absent"

    def _boom(env, **kwargs):
        raise RuntimeError("no")

    monkeypatch.setattr(
        shim, "_load_optional", lambda _name: types.SimpleNamespace(apply_xet_env = _boom)
    )
    assert shim.apply_xet_env({}) is None, "a raising zoo must degrade, not crash the download"


def test_a_zoo_that_can_resize_is_asked_for_the_workers_own_cache(monkeypatch):
    """``env`` is a copy of ours and already carries the zoo's import-time sizing, which its
    setdefault apply would keep. A zoo that can resize gets the worker's cache instead, so a backend
    whose cache moved after startup does not size the worker for the volume it left behind. An older
    zoo, with no resize to call, keeps the previous behaviour."""
    import types

    import utils.hf_xet_fallback as shim

    seen = {}

    def _resize(env, cache_dir, **kwargs):
        seen["cache_dir"] = cache_dir
        env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] = "1000000000"
        return {"HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "1000000000"}

    def _apply(env, **kwargs):
        seen["applied"] = True
        return {}

    monkeypatch.setattr(
        shim,
        "_load_optional",
        lambda _name: types.SimpleNamespace(
            apply_xet_env = _apply,
            resize_for_cache_dir = _resize,
        ),
    )
    env = {
        "HF_HUB_CACHE": "/new/volume/hub",
        "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "64000000000",
    }
    written = shim.apply_xet_env(env, env["HF_HUB_CACHE"])
    assert seen["cache_dir"] == "/new/volume/hub"
    assert "applied" not in seen, "resizing and applying both would size the worker twice"
    assert written["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "1000000000"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "1000000000"

    monkeypatch.setattr(
        shim, "_load_optional", lambda _name: types.SimpleNamespace(apply_xet_env = _apply)
    )
    assert shim.apply_xet_env({}, "/new/volume/hub") == {}
    assert seen["applied"] is True


# --- free-RAM clamp (issue #9032) ---------------------------------------------------------------
# The zoo sizes Xet's buffers from TOTAL RAM, which cannot see a loaded model. Unsloth clamps the
# result to what is free. The bar: shrink under pressure, change nothing otherwise.

_GB = 1_000_000_000
_LIMIT = "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"


def _fake_profile_cls():
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class _Profile:
        total_ram_bytes: int
        available_ram_bytes: int
        cpu_count: int = 16
        ram_source: str = "psutil"
        cpu_source: str = "affinity"
        free_disk_bytes: int = 500 * _GB
        disk_source: str = "statvfs"

    return _Profile


def _fake_tuning(
    total,
    available,
    *,
    calls = None,
):
    """Stand-in zoo sized like the real one (an eighth of total RAM), recording the profile it was
    asked about so a test can prove the clamp re-asks rather than editing numbers itself."""
    import types

    profile_cls = _fake_profile_cls()

    def _overrides(profile, **kwargs):
        if calls is not None:
            calls.append(profile)
        limit = max(1 * _GB, profile.total_ram_bytes // 8)
        return {
            _LIMIT: str(limit),
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(limit // 2),
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(limit // 32),
            "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "8",
            "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": "32",
            "HF_XET_CLIENT_READ_TIMEOUT": "60s",
        }

    return types.SimpleNamespace(
        xet_env_overrides = _overrides,
        system_profile = lambda cache_dir = None: profile_cls(total, available),
        _MIN_BUFFER_LIMIT = 1 * _GB,
        _RAM_FRACTION = 8,
    )


def test_clamp_is_a_no_op_when_the_machine_has_room(monkeypatch):
    """The design rests on this: an eighth of TOTAL cannot exceed a quarter of AVAILABLE unless RAM
    is already held, so an idle machine keeps the zoo's numbers and no download gets slower."""
    import utils.hf_xet_fallback as shim

    calls = []
    module = _fake_tuning(32 * _GB, 30 * _GB, calls = calls)
    sized = module.xet_env_overrides(module.system_profile())
    calls.clear()

    env = dict(sized)
    written = shim.clamp_to_available_ram(env, dict(sized), module = module)

    assert written == sized, "an affordable budget must come back untouched"
    assert env == sized
    assert calls == [], "re-sizing a machine that fits would burn the zero-cost guarantee"


def test_clamp_shrinks_a_budget_free_ram_cannot_afford(monkeypatch):
    """Issue #9032: 32GB box, 27B GGUF resident, 8GB free. The zoo still hands out a 4GB buffer
    because total RAM has not changed, and that on top of the loaded weights is the swap."""
    import utils.hf_xet_fallback as shim

    calls = []
    module = _fake_tuning(32 * _GB, 8 * _GB, calls = calls)
    unclamped = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))
    calls.clear()

    env = dict(unclamped)
    written = shim.clamp_to_available_ram(env, dict(unclamped), module = module)

    budget = 8 * _GB // 4
    assert int(unclamped[_LIMIT]) > budget, "precondition: the unclamped budget overshoots"
    assert int(written[_LIMIT]) <= budget, "the clamp has to bring it inside what is free"
    assert env[_LIMIT] == written[_LIMIT], "the worker's env is what actually ships"
    assert calls, "the zoo must be the one re-sizing, so its formulas stay the single source"
    assert calls[0].total_ram_bytes < 32 * _GB, "it should be asked about a smaller machine"
    # Every derived number moves together; a limit shrunk on its own would leave the per-file and
    # concurrency values describing a budget that no longer exists.
    assert int(written["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) < int(
        unclamped["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]
    )


def test_clamp_bottoms_out_at_the_zoos_own_floor(monkeypatch):
    """Below the floor the answer is a different transport, not a tinier buffer: Xet has a minimum
    it can work in, and ``_memory_pressure_reason`` routes a machine this tight to HTTP."""
    import utils.hf_xet_fallback as shim

    module = _fake_tuning(32 * _GB, 1 * _GB)
    unclamped = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 1 * _GB))

    written = shim.clamp_to_available_ram({}, dict(unclamped), module = module)
    assert int(written[_LIMIT]) == module._MIN_BUFFER_LIMIT
    assert int(written[_LIMIT]) < int(unclamped[_LIMIT])


def test_clamp_never_writes_a_key_the_user_set(monkeypatch):
    """The zoo's apply is setdefault, so a user-set variable never reaches ``sized`` and must not be
    reintroduced here. Same mechanism covers HF_XET_HIGH_PERFORMANCE: the zoo drops its caps, no
    budget key arrives, and the clamp stands down with it."""
    import utils.hf_xet_fallback as shim

    calls = []
    module = _fake_tuning(32 * _GB, 1 * _GB, calls = calls)

    # High-performance stand-down: no budget key in what the zoo wrote.
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    assert shim.clamp_to_available_ram(env, {}, module = module) == {}
    assert env == {"HF_XET_HIGH_PERFORMANCE": "1"}
    assert calls == [], "with no caps to clamp there is nothing to re-size"

    # A user-pinned per-file size is absent from `sized` for the same reason; clamping the rest must
    # not write it back at our number.
    sized = {_LIMIT: str(8 * _GB)}
    env = {_LIMIT: str(8 * _GB), "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": "999"}
    written = shim.clamp_to_available_ram(env, sized, module = module)
    assert set(written) == {_LIMIT}, "only keys the zoo wrote are ours to rewrite"
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"] == "999"


def test_clamp_stands_down_when_ram_cannot_be_measured(monkeypatch):
    """No psutil, no cgroup, or a zoo too old to expose a profile: absence of evidence is not
    evidence of pressure, so the download runs as before."""
    import types

    import utils.hf_xet_fallback as shim

    sized = {_LIMIT: str(8 * _GB)}

    unmeasurable = _fake_tuning(0, 0)
    assert shim.clamp_to_available_ram({}, dict(sized), module = unmeasurable) == sized

    old_zoo = types.SimpleNamespace(apply_xet_env = lambda *a, **k: {})
    assert shim.clamp_to_available_ram({}, dict(sized), module = old_zoo) == sized

    def _boom(*args, **kwargs):
        raise RuntimeError("no")

    raising = types.SimpleNamespace(xet_env_overrides = _boom, system_profile = _boom)
    assert (
        shim.clamp_to_available_ram({}, dict(sized), module = raising) == sized
    ), "a clamp must never be the thing that breaks a download"


def test_clamp_holds_against_the_real_zoo_formulas():
    """The fakes above pin the seam; this pins the arithmetic, so a zoo that changes how it sizes
    cannot quietly reintroduce a budget bigger than free RAM."""
    tuning = pytest.importorskip("unsloth_zoo.hf_xet_tuning")

    idle = tuning.SystemProfile(
        total_ram_bytes = 32 * 1024**3,
        available_ram_bytes = 30 * _GB,
        cpu_count = 24,
        ram_source = "psutil",
        cpu_source = "affinity",
        free_disk_bytes = 500 * _GB,
        disk_source = "statvfs",
    )
    import dataclasses

    loaded = dataclasses.replace(idle, available_ram_bytes = 8 * _GB)

    import types

    import utils.hf_xet_fallback as shim

    sized = dict(tuning.xet_env_overrides(idle, fail_fast = True))

    def _module_for(profile):
        return types.SimpleNamespace(
            xet_env_overrides = tuning.xet_env_overrides,
            system_profile = lambda cache_dir = None: profile,
            _MIN_BUFFER_LIMIT = tuning._MIN_BUFFER_LIMIT,
            _RAM_FRACTION = tuning._RAM_FRACTION,
        )

    assert (
        shim.clamp_to_available_ram({}, dict(sized), module = _module_for(idle)) == sized
    ), "30GB free must still buy the zoo's own 32GB-machine budget"

    clamped = shim.clamp_to_available_ram({}, dict(sized), module = _module_for(loaded))
    assert int(clamped[_LIMIT]) <= 8 * _GB // 4, "a 27B model resident must shrink the budget"
    assert int(clamped[_LIMIT]) < int(sized[_LIMIT])


def test_clamp_never_raises_a_value_the_zoo_had_lowered():
    """The recompute runs without the throttled flag that a 429 backoff sets, so the clamp must take
    the smaller of the two per key. Otherwise shrinking buffers would restore the stream ceiling."""
    import utils.hf_xet_fallback as shim

    module = _fake_tuning(32 * _GB, 8 * _GB)
    unclamped = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))
    # As if a 429 had halved the ceiling on the way in.
    throttled = dict(unclamped, HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY = "4")

    written = shim.clamp_to_available_ram({}, dict(throttled), module = module)
    assert int(written[_LIMIT]) < int(throttled[_LIMIT]), "the budget still has to shrink"
    assert (
        written["HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY"] == "4"
    ), "a clamp that raises anything is not a clamp"


def test_free_ram_pressure_reason_applies_the_zoos_own_floor(monkeypatch):
    """The threshold logic itself, now that both transport callers share this one helper."""
    import utils.hf_xet_fallback as shim

    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (2 * _GB, 4 * _GB))
    reason = shim.free_ram_pressure_reason()
    assert reason is not None and "2.0GB RAM free" in reason

    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (4 * _GB, 4 * _GB))
    assert shim.free_ram_pressure_reason() is None, "at the floor is not below it"

    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (30 * _GB, 4 * _GB))
    assert shim.free_ram_pressure_reason() is None

    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (None, 4 * _GB))
    assert shim.free_ram_pressure_reason() is None, "unmeasurable is not pressure"

    def _boom():
        raise RuntimeError("no")

    monkeypatch.setattr(shim, "available_ram_bytes", _boom)
    assert shim.free_ram_pressure_reason() is None


# --- concurrent-worker reservations --------------------------------------------------------------
# A worker allocates in the child, after Popen returns, so free RAM does not move until well after
# sizing. Without reservations, downloads starting together each read the same untouched number.


@pytest.fixture(autouse = True)
def clean_ledger():
    """Autouse: sizing reserves RAM, so any clamp test leaves a reservation that would otherwise
    follow the process into the next test and shrink its budget."""
    import utils.hf_xet_fallback as shim

    shim._budget_reservations.clear()
    shim._pending_reservation.token = None
    yield shim
    shim._budget_reservations.clear()
    shim._pending_reservation.token = None


def test_workers_starting_together_do_not_promise_the_same_ram_twice(clean_ledger):
    """Four downloads queued at once each used to take a quarter of the same snapshot, promising the
    whole machine before any of them had allocated a byte."""
    import os

    shim = clean_ledger
    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))

    promised, budgets = 0, []
    for _ in range(4):
        written = shim.clamp_to_available_ram({}, dict(sized), module = module)
        shim.bind_worker_budget(os.getpid())  # this process stands in for a live worker
        budgets.append(int(written[_LIMIT]))
        promised += budgets[-1]

    # Strictly under, not merely equal: without the ledger these four land on exactly 8GB, a
    # quarter each of the same snapshot, which is the whole bug.
    assert promised < 8 * _GB, f"four workers promised {promised / _GB:.2f}GB of 8GB free"
    assert budgets[1] < budgets[0], "the second worker ignored what the first was already promised"


def test_a_reservation_frees_when_its_worker_exits(clean_ledger):
    """Held forever, one finished download would shrink every later one on the machine."""
    import subprocess
    import sys

    shim = clean_ledger
    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))

    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait()
    shim.clamp_to_available_ram({}, dict(sized), module = module)
    shim.bind_worker_budget(dead.pid)

    with shim._budget_lock:
        assert shim._live_reserved_locked() == 0, "a finished worker still held RAM"


def test_a_spawn_that_never_happened_does_not_leak(clean_ledger):
    """Popen raising must not strand the reservation its sizing took."""
    shim = clean_ledger
    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))

    shim.clamp_to_available_ram({}, dict(sized), module = module)
    shim.bind_worker_budget(None)
    with shim._budget_lock:
        assert shim._live_reserved_locked() == 0

    # And one never bound at all ages out rather than pinning RAM for the process's life.
    shim.clamp_to_available_ram({}, dict(sized), module = module)
    for entry in shim._budget_reservations.values():
        entry[2] -= shim._UNBOUND_RESERVATION_TTL + 1
    with shim._budget_lock:
        assert shim._live_reserved_locked() == 0


def test_one_download_on_an_idle_machine_is_still_untouched(clean_ledger):
    """The ledger must not cost the common case: nothing is reserved yet, so sizing is the zoo's."""
    shim = clean_ledger
    module = _fake_tuning(32 * _GB, 30 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 30 * _GB))

    assert shim.clamp_to_available_ram({}, dict(sized), module = module) == sized


def test_the_transport_gate_counts_ram_promised_to_running_downloads(clean_ledger, monkeypatch):
    """The clamp bottoms out at Xet's floor, so enough simultaneous workers would still add up past
    free RAM. Subtracting reservations sends the next one to HTTP instead.

    Both halves of that guard are asserted, because the promise and the RAM reading cover different
    moments: three just-admitted workers have taken nothing yet, so only the ledger can stop the
    fourth; three that have finished allocating are already missing from `available`, which stops it
    without the ledger. `os.getpid()` stands in for all three, so its own RSS is stubbed out rather
    than credited three times against promises it has nothing to do with."""
    import os

    shim = clean_ledger
    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (8 * _GB, 4 * _GB))
    assert shim.free_ram_pressure_reason() is None, "8GB free is not pressure on its own"

    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: 0)  # spawned, nothing allocated yet
    promised = 0
    for _ in range(3):
        written = shim.clamp_to_available_ram({}, dict(sized), module = module)
        shim.bind_worker_budget(os.getpid())
        promised += int(written[_LIMIT])

    reason = shim.free_ram_pressure_reason()
    assert reason is not None, "three running downloads left too little RAM for a fourth on Xet"
    assert "RAM free" in reason

    # Same three workers once their buffers are resident: the promises are spent, and the RAM
    # reading they have already moved is what refuses the fourth.
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: promised)
    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (8 * _GB - promised, 4 * _GB))
    assert (
        shim.free_ram_pressure_reason() is not None
    ), "three allocated downloads left too little RAM for a fourth on Xet"


def test_a_resident_promise_is_not_charged_against_free_ram_twice(clean_ledger, monkeypatch):
    """Once a worker's buffers are resident, `available` has already dropped by them.

    The reservation exists to cover the gap between sizing and allocation, so charging the whole
    promise on top of a reading that already reflects it counts the same bytes twice for the
    worker's entire lifetime. On an 8GB-free host that is the difference between the next Auto
    download getting Xet and being told "only 2.0GB RAM free" while 4GB genuinely is."""
    import os

    shim = clean_ledger
    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))

    written = shim.clamp_to_available_ram({}, dict(sized), module = module)
    promise = int(written[_LIMIT])
    shim.bind_worker_budget(os.getpid())

    # Not yet allocated: the promise is the only thing standing between the sibling and this RAM.
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: 0)
    with shim._budget_lock:
        assert shim._live_reserved_locked() == promise

    # Allocated: `available` fell by `promise`, so the ledger must stop asking for it again.
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: promise)
    with shim._budget_lock:
        assert shim._live_reserved_locked() == 0, "a resident promise was subtracted a second time"

    # Half in flight leaves exactly the unmaterialized half reserved.
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: promise // 2)
    with shim._budget_lock:
        assert shim._live_reserved_locked() == promise - promise // 2

    # And the gate follows: 8GB free with one fully resident 2GB worker is 6GB, not 4GB, so Auto
    # for the next download stays on Xet.
    monkeypatch.setattr(shim, "available_ram_bytes", lambda: (8 * _GB - promise, 4 * _GB))
    monkeypatch.setattr(shim, "_worker_rss", lambda pid: promise)
    assert (
        shim.free_ram_pressure_reason() is None
    ), "the next Auto download was demoted to HTTP over RAM its sibling never took"


def test_the_ledger_reads_a_real_workers_rss(clean_ledger):
    """The credit above is only correct if the psutil read actually works on a live child."""
    import subprocess
    import sys

    shim = clean_ledger
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin = subprocess.PIPE,
    )
    try:
        rss = shim._worker_rss(child.pid)
        assert rss > 0, "a running interpreter reported no resident memory"
    finally:
        child.stdin.close()
        child.wait(timeout = 30)

    # An exited worker cannot be read, and an unreadable one keeps its whole promise reserved.
    assert shim._worker_rss(child.pid) == 0 or not shim._pid_alive(child.pid)


def test_concurrent_sizings_cannot_all_claim_the_same_free_ram(clean_ledger):
    """The reservation tests above start workers one after another, which never exercises the race:
    read the ledger, then reserve, with a gap in between. These threads sit in that gap together.

    The barrier is in system_profile, which the clamp reads OUTSIDE the lock, so all four arrive at
    the decision at once; the sleep in xet_env_overrides widens the read-to-reserve window that a
    split critical section would leave open."""
    import os
    import threading
    import time

    shim = clean_ledger
    workers = 4
    barrier = threading.Barrier(workers)
    profile_cls = _fake_profile_cls()
    profile = profile_cls(32 * _GB, 8 * _GB)

    def _overrides(prof, **kwargs):
        time.sleep(0.02)
        limit = max(1 * _GB, prof.total_ram_bytes // 8)
        return {
            _LIMIT: str(limit),
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE": str(limit // 2),
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE": str(limit // 32),
            "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "8",
        }

    def _profile_of(cache_dir = None):
        barrier.wait(timeout = 10)
        return profile

    module = _types.SimpleNamespace(
        xet_env_overrides = _overrides,
        system_profile = _profile_of,
        _MIN_BUFFER_LIMIT = 1 * _GB,
        _RAM_FRACTION = 8,
    )
    sized = _overrides(profile)

    granted: list[int] = []
    lock = threading.Lock()

    def _size():
        written = shim.clamp_to_available_ram({}, dict(sized), module = module)
        shim.bind_worker_budget(os.getpid())
        with lock:
            granted.append(int(written[_LIMIT]))

    threads = [threading.Thread(target = _size) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 30)

    assert len(granted) == workers, "a sizing thread never finished"
    total = sum(granted)
    assert total < 8 * _GB, (
        f"{workers} concurrent sizings promised {total / _GB:.2f}GB of 8GB free; "
        "the ledger read and the reservation are not one decision"
    )
    assert len(set(granted)) > 1, "identical budgets means they all read the same snapshot"


def test_the_ledgers_liveness_probe_never_signals_on_windows(clean_ledger, monkeypatch):
    """`os.kill(pid, 0)` is not a probe on Windows.

    CPython's `os_kill_impl` routes every signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT into
    `OpenProcess(PROCESS_ALL_ACCESS)` + `TerminateProcess(handle, sig)`, so signal 0 KILLS the
    target. The ledger prunes dead reservations on every sizing and on every capability probe, so
    the old probe would terminate a running download merely because a second one was considered."""
    import os as _os

    import utils.process_lifetime as pl

    shim = clean_ledger
    monkeypatch.setattr(pl, "_is_windows", lambda: True)

    signalled: list[tuple] = []

    def _forbidden(pid, sig):
        signalled.append((pid, sig))
        raise AssertionError("os.kill must never be reached on Windows")

    monkeypatch.setattr(_os, "kill", _forbidden)

    shim._pid_alive(_os.getpid())
    assert signalled == [], "the liveness probe signalled the worker it was asking about"


def test_a_running_worker_keeps_its_reservation_on_windows(clean_ledger, monkeypatch):
    """The Windows probe must also answer correctly, or every reservation is pruned on sight and
    concurrent workers go back to promising the same free RAM."""
    import ctypes
    import os as _os

    import utils.process_lifetime as pl

    shim = clean_ledger
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(_os, "kill", _fail_on_kill)

    WAIT_TIMEOUT = 0x102

    class _FakeKernel32:
        def __init__(self, *_args, **_kwargs):
            self.OpenProcess = _FakeFn(0xBEEF)
            self.WaitForSingleObject = _FakeFn(WAIT_TIMEOUT)  # still running
            self.CloseHandle = _FakeFn(1)

    monkeypatch.setattr(ctypes, "WinDLL", _FakeKernel32, raising = False)

    assert shim._pid_alive(4321) is True

    module = _fake_tuning(32 * _GB, 8 * _GB)
    sized = module.xet_env_overrides(_fake_profile_cls()(32 * _GB, 8 * _GB))
    shim.clamp_to_available_ram({}, dict(sized), module = module)
    shim.bind_worker_budget(4321)
    with shim._budget_lock:
        assert shim._live_reserved_locked() > 0, "a live Windows worker's reservation was pruned"

    # And an exited worker (handle signalled) frees its reservation.
    class _FakeKernel32Dead(_FakeKernel32):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.WaitForSingleObject = _FakeFn(0)  # WAIT_OBJECT_0: exited

    monkeypatch.setattr(ctypes, "WinDLL", _FakeKernel32Dead, raising = False)
    with shim._budget_lock:
        assert shim._live_reserved_locked() == 0, "an exited Windows worker still held RAM"


class _FakeFn:
    """A stand-in for a ctypes function pointer: assignable argtypes/restype, fixed return."""

    def __init__(self, result):
        self._result = result
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args, **_kwargs):
        return self._result


def _fail_on_kill(pid, sig):
    raise AssertionError("os.kill must never be reached on Windows")
