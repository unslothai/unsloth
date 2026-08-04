# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the hub download path's transport selection, RAM caps, and stall -> HTTP recovery.

The model-hub page is how most users download, and it was the ONE download path with no stall
detection: it relied on the worker's exit code, and a Xet transfer that hangs with no progress and
no error never produces one. These tests pin the three pieces that close that gap.

CPU-only, no network, no real worker subprocess.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
try:
    import structlog  # noqa: F401
except ImportError:
    sys.modules["structlog"] = _types.ModuleType("structlog")

from hub.services import download_lifecycle as dl
from hub.utils import download_registry


# --------------------------------------------------------------------------------------------
# Transport selection
# --------------------------------------------------------------------------------------------


def test_explicit_modes_are_honoured(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    assert dl.resolve_requested_use_xet("http", True)[0] is False
    assert dl.resolve_requested_use_xet("xet", False)[0] is True


def test_explicit_xet_beats_an_unhealthy_verdict(monkeypatch):
    """An explicit choice is not overruled by the health verdict; it still gets the memory caps and
    the stall fallback."""
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    monkeypatch.setattr(dl, "resolve_auto_use_xet", lambda: (False, "demoted"))
    assert dl.resolve_requested_use_xet("xet", True)[0] is True


def test_auto_defers_to_the_health_verdict(monkeypatch):
    monkeypatch.setattr(dl, "resolve_auto_use_xet", lambda: (False, "Xet stalled twice"))
    use_xet, reason = dl.resolve_requested_use_xet("auto", True)
    assert use_xet is False
    assert reason == "Xet stalled twice"


def test_legacy_use_xet_still_works(monkeypatch):
    """An older frontend, or a scripted API caller, sends no transport_mode at all."""
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    assert dl.resolve_requested_use_xet(None, True)[0] is True
    assert dl.resolve_requested_use_xet(None, False)[0] is False


def test_auto_falls_back_to_xet_when_health_is_unavailable(monkeypatch):
    """A missing or broken health module means "no opinion", never "downgrade"."""
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    fake = _types.ModuleType("utils.hf_xet_fallback")
    fake.xet_health = lambda **kw: None
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    assert dl.resolve_auto_use_xet()[0] is True


def test_auto_reports_http_when_hf_xet_is_missing(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: False)
    use_xet, reason = dl.resolve_auto_use_xet()
    assert use_xet is False
    assert "hf_xet" in reason


def test_auto_is_not_a_real_transport():
    """ "auto" is a request preference: the .transport marker must keep naming the writer that
    produced a partial, or a resume picks the wrong strategy."""
    assert download_registry.TRANSPORT_AUTO not in download_registry.VALID_TRANSPORTS
    assert download_registry.TRANSPORT_AUTO in download_registry.VALID_TRANSPORT_MODES


# --------------------------------------------------------------------------------------------
# RAM caps reach the worker environment
# --------------------------------------------------------------------------------------------


class _FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.pid = 4242
        self.stderr = None
        self.returncode = 0


def _spawn_env(
    monkeypatch,
    *,
    use_xet: bool,
    parent_env: dict | None = None,
) -> dict:
    captured = {}

    def _fake_popen(
        cmd,
        env = None,
        **kwargs,
    ):
        captured.update(env or {})
        return _FakePopen()

    paths = _types.SimpleNamespace(child_env = lambda *a, **k: dict(parent_env or {}))
    fake_settings = _types.ModuleType("utils.hf_cache_settings")
    fake_settings.get_hf_cache_paths = lambda: paths
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", fake_settings)
    monkeypatch.setattr(dl.subprocess, "Popen", _fake_popen)
    dl.spawn_worker(["--repo-id", "a/b"], None, use_xet = use_xet)
    return captured


def _tuning_available() -> bool:
    try:
        from utils.hf_xet_fallback import xet_env_overrides
        return bool(xet_env_overrides())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(
    not _tuning_available(),
    reason = "the installed unsloth_zoo predates hf_xet_tuning, so there are no caps to apply",
)
def test_xet_worker_is_sized_from_the_machine(monkeypatch):
    """The budget scales with the host, so pin the invariant rather than a number: what hf_xet can
    hold (buffer + files * per-file) must fit the limit the same call set."""
    env = _spawn_env(monkeypatch, use_xet = True)
    limit = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"])
    worst = int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE"]) + int(
        env["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"]
    ) * int(env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE"])
    assert 0 < worst <= limit
    assert env["HF_HUB_DISABLE_XET"] == "0"


def test_the_zoo_decides_and_studio_does_not_second_guess_it(monkeypatch):
    """Studio used to force the flag off here. Two copies of one rule drifted, and on a 2TB host the
    worker ended up with a 24GB laptop's buffer, 3.4x slower than the machine's own setting."""
    import utils.hf_xet_fallback as shim

    seen = {}

    def _apply(env):
        seen.update(env)
        env["HF_XET_SENTINEL"] = "sized-by-the-zoo"
        return {"HF_XET_SENTINEL": "sized-by-the-zoo"}

    monkeypatch.setattr(shim, "apply_xet_env", _apply)
    env = _spawn_env(monkeypatch, use_xet = True, parent_env = {"HF_XET_HIGH_PERFORMANCE": "1"})
    assert env["HF_XET_SENTINEL"] == "sized-by-the-zoo"
    # The worker's own env is what gets sized, and the flag is left exactly as the zoo left it.
    assert seen["HF_HUB_DISABLE_XET"] == "0"
    assert env["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_high_performance_is_cleared_even_without_the_tuning_module(monkeypatch):
    """An unsloth_zoo with no `hf_xet_tuning` is exactly the version that sets
    HF_XET_HIGH_PERFORMANCE=1 at import, so routing the clear through the (then empty) overrides
    would hand the worker a 64GB buffer ceiling on the installs Studio alone cannot fix."""
    import utils.hf_xet_fallback as shim

    monkeypatch.setattr(shim, "apply_xet_env", lambda *a, **k: None)
    env = _spawn_env(monkeypatch, use_xet = True, parent_env = {"HF_XET_HIGH_PERFORMANCE": "1"})
    assert env["HF_XET_HIGH_PERFORMANCE"] == "0"
    assert env["HF_XET_HP"] == "0"


def test_the_legacy_opt_in_still_works_without_the_tuning_module(monkeypatch):
    """Newer zoos honour the flag on their own, but this is the escape hatch on installs that
    cannot, so it has to keep working there."""
    import utils.hf_xet_fallback as shim

    monkeypatch.setattr(shim, "apply_xet_env", lambda *a, **k: None)
    monkeypatch.setenv("UNSLOTH_XET_ALLOW_HIGH_PERFORMANCE", "1")
    env = _spawn_env(monkeypatch, use_xet = True, parent_env = {"HF_XET_HIGH_PERFORMANCE": "1"})
    assert env["HF_XET_HIGH_PERFORMANCE"] == "1"


@pytest.mark.skipif(
    not _tuning_available(),
    reason = "the installed unsloth_zoo predates hf_xet_tuning, so there are no caps to preserve",
)
def test_explicit_cap_from_the_operator_is_preserved(monkeypatch):
    env = _spawn_env(
        monkeypatch,
        use_xet = True,
        parent_env = {
            "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT": "7777777",
        },
    )
    assert env["HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT"] == "7777777"


def test_http_worker_gets_no_xet_caps(monkeypatch):
    env = _spawn_env(monkeypatch, use_xet = False)
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT" not in env


# --------------------------------------------------------------------------------------------
# Stall -> kill -> HTTP retry
# --------------------------------------------------------------------------------------------


class _KillablePopen:
    def __init__(self):
        self.pid = 999
        self.killed = threading.Event()

    def kill(self):
        self.killed.set()


def _registry_stub():
    return _types.SimpleNamespace(get_job_metadata = lambda key: None)


def test_stall_watchdog_kills_the_worker(monkeypatch):
    """The kill converts an invisible hang into an "error" exit, the state the HTTP-retry keys on."""
    proc = _KillablePopen()
    seen: list[str] = []
    started = {}

    fake = _types.ModuleType("utils.hf_xet_fallback")

    def _start_watchdog(*, on_stall, **kwargs):
        started.update(kwargs)
        on_stall("Download appears stalled (xet transport) -- no progress for 30s")
        return threading.Event()

    fake.start_watchdog = _start_watchdog
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)

    stop = dl._start_stall_watchdog(
        _registry_stub(),
        "models--a--b",
        proc,
        repo_type = "model",
        repo_id = "a/b",
        label = "a/b",
        log_prefix = "[hub]",
        logger = dl.logger,
        on_stall = seen.append,
    )
    assert stop is not None
    assert proc.killed.is_set(), "a stalled worker was not killed"
    assert seen and "stalled" in seen[0]
    assert started["repo_ids"] == ["a/b"]
    assert started["child_pid"] == 999


def test_stall_watchdog_survives_an_already_exited_worker(monkeypatch):
    """The worker can exit between the stall verdict and the kill; that is a race, not an error."""

    class _Gone:
        pid = 1

        def kill(self):
            raise ProcessLookupError

    fake = _types.ModuleType("utils.hf_xet_fallback")
    fake.start_watchdog = lambda *, on_stall, **kw: (on_stall("stalled"), threading.Event())[1]
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)

    stop = dl._start_stall_watchdog(
        _registry_stub(),
        "k",
        _Gone(),
        repo_type = "model",
        repo_id = "a/b",
        label = "a/b",
        log_prefix = "[hub]",
        logger = dl.logger,
        on_stall = lambda _m: None,
    )
    assert stop is not None


def test_missing_watchdog_degrades_quietly(monkeypatch):
    """An older unsloth_zoo without start_watchdog must not break downloads, only stall detection."""
    fake = _types.ModuleType("utils.hf_xet_fallback")
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    assert (
        dl._start_stall_watchdog(
            _registry_stub(),
            "k",
            _KillablePopen(),
            repo_type = "model",
            repo_id = "a/b",
            label = "a/b",
            log_prefix = "[hub]",
            logger = dl.logger,
            on_stall = lambda _m: None,
        )
        is None
    )


def test_stall_is_recorded_against_the_machine(monkeypatch):
    recorded: list = []
    fake = _types.ModuleType("utils.hf_xet_fallback")
    fake.record_xet_outcome = lambda ok, reason: recorded.append((ok, reason))
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    dl._record_xet_failure("Xet stalled", dl.logger)
    assert recorded == [(False, "Xet stalled")]


def test_recording_a_failure_never_raises(monkeypatch):
    fake = _types.ModuleType("utils.hf_xet_fallback")

    def _boom(ok, reason):
        raise RuntimeError("state file is read-only")

    fake.record_xet_outcome = _boom
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    dl._record_xet_failure("Xet stalled", dl.logger)  # must not propagate


# --------------------------------------------------------------------------------------------
# Capabilities endpoint
# --------------------------------------------------------------------------------------------


def test_capabilities_report_what_auto_resolves_to(monkeypatch):
    fake = _types.ModuleType("utils.hf_xet_fallback")
    fake.xet_health = lambda **kw: _types.SimpleNamespace(
        use_xet = False,
        reason = "Xet failed 2 times in a row on this machine",
    )
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    caps = download_registry.get_download_transport_capabilities()
    if not caps.xet.available:
        pytest.skip("hf_xet is not installed in this environment")
    assert caps.auto_resolves_to == download_registry.TRANSPORT_HTTP
    assert "2 times" in caps.auto_reason


def test_capabilities_stay_optimistic_when_health_raises(monkeypatch):
    fake = _types.ModuleType("utils.hf_xet_fallback")

    def _boom(**kw):
        raise RuntimeError("no")

    fake.xet_health = _boom
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake)
    caps = download_registry.get_download_transport_capabilities()
    if not caps.xet.available:
        pytest.skip("hf_xet is not installed in this environment")
    # The download-time ladder still recovers, so an unknown verdict should not cost the fast path.
    assert caps.auto_resolves_to == download_registry.TRANSPORT_XET


# --------------------------------------------------------------------------------------------
# CPU-only hosts
# --------------------------------------------------------------------------------------------


def test_optional_loader_retries_with_gpu_init_disabled(monkeypatch):
    """unsloth_zoo.__init__ runs torch accelerator detection and raises on a CPU-only host, which is
    exactly the small machine these caps protect, so without the retry they switch off where they
    are needed."""
    import importlib

    import utils.hf_xet_fallback as shim

    attempts: list[str | None] = []
    sentinel = _types.ModuleType("fake_zoo_module")

    def _fake_import(name):
        import os

        seen = os.environ.get("UNSLOTH_ZOO_DISABLE_GPU_INIT")
        attempts.append(seen)
        if seen != "1":
            raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
        return sentinel

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)

    assert shim._load_optional("unsloth_zoo.hf_xet_tuning") is sentinel
    assert attempts == [None, "1"]
    # The flag is scoped to the retry: it must not leak into unrelated later imports.
    import os

    assert "UNSLOTH_ZOO_DISABLE_GPU_INIT" not in os.environ


def test_optional_loader_returns_none_when_truly_absent(monkeypatch):
    import importlib

    import utils.hf_xet_fallback as shim

    def _always_fail(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _always_fail)
    assert shim._load_optional("unsloth_zoo.hf_xet_tuning") is None
    # A missing module means "no opinion", never a hard failure.
    assert shim.xet_env_overrides() == {}
    assert shim.xet_health() is None
    shim.record_xet_outcome(False, "x")


def test_capabilities_probe_is_opt_in(monkeypatch):
    """The UI polls this endpoint on render, so it must stay cheap by default. The download-start
    path opts in: a host with an unreachable CAS and no recorded failure yet would otherwise learn
    by stalling."""
    from hub.utils import download_registry

    seen: list[bool] = []

    class _Health:
        use_xet = False
        reason = "probed: CAS unreachable"

    def _fake_health(*, probe = True):
        seen.append(probe)
        return _Health()

    # Patch the sys.modules entry, not an imported alias: the endpoint does a local
    # `from utils.hf_xet_fallback import xet_health`, and test_hf_xet_fallback.py swaps that
    # sys.modules entry in and out, so an alias captured here can be a different module object.
    import sys
    import types

    stub = types.ModuleType("utils.hf_xet_fallback")
    stub.xet_health = _fake_health
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", stub)

    caps = download_registry.get_download_transport_capabilities()
    if not caps.xet.available:
        # The health lookup sits behind an hf_xet availability check, so with no hf_xet installed
        # neither call reaches it.
        pytest.skip("hf_xet is not installed in this environment")
    download_registry.get_download_transport_capabilities(probe = True)

    assert seen == [False, True]


def test_gpu_init_override_is_serialized(monkeypatch):
    """The optional-module retry must not leak the process-wide GPU-init override: a leaked
    UNSLOTH_ZOO_DISABLE_GPU_INIT=1 is inherited by every spawned worker for the life of the process.

    Scope: this races the loader against itself. The cross-loader interleave with _load_shared is
    not reproducible by thread timing, and is established by construction instead (both take the
    same `_load_lock` around their save/set/restore) -- see the test below.
    """
    import importlib
    import os
    import threading

    import utils.hf_xet_fallback as shim

    monkeypatch.delenv("UNSLOTH_ZOO_DISABLE_GPU_INIT", raising = False)

    def _always_fail(name):
        time.sleep(0.005)  # widen the window the lock has to close
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _always_fail)

    threads = [
        threading.Thread(target = shim._load_optional, args = ("unsloth_zoo.hf_xet_tuning",))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert "UNSLOTH_ZOO_DISABLE_GPU_INIT" not in os.environ


def test_both_loaders_share_one_env_lock():
    """The cross-loader guarantee, checked structurally. Two separate locks would each be correct in
    isolation and still allow the interleave that leaves the override set permanently."""
    import inspect

    import utils.hf_xet_fallback as shim

    for fn in (shim._load_shared, shim._load_optional):
        source = inspect.getsource(fn)
        assert "UNSLOTH_ZOO_DISABLE_GPU_INIT" in source
        assert (
            "with _load_lock:" in source
        ), f"{fn.__name__} mutates the GPU-init override outside the shared _load_lock"


def test_the_worker_never_gets_the_flag_and_our_caps_together(monkeypatch):
    """End to end against whichever unsloth_zoo is installed, with nothing stubbed. Which of the
    two the zoo picks is its call and changes with the version; what must never happen either way
    is both at once, because xet-core applies the preset after reading the environment, so it
    voids the limit while still honouring the smaller per-file and concurrency numbers."""
    env = _spawn_env(monkeypatch, use_xet = True, parent_env = {"HF_XET_HIGH_PERFORMANCE": "1"})
    flag_on = env.get("HF_XET_HIGH_PERFORMANCE", "0").strip().lower() in ("1", "true", "yes", "on")
    sized = "HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE" in env
    assert not (flag_on and sized), f"worst of both: flag on with our sizing still applied ({env})"
