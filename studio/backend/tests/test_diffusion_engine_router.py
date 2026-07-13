# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the diffusion engine router (diffusers vs native sd.cpp selection)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

_ENVS = (
    "UNSLOTH_DIFFUSION_ENGINE",
    "UNSLOTH_DIFFUSION_SD_CPP",
    "UNSLOTH_DIFFUSION_SD_CPP_MPS",
    "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
)


@pytest.fixture(autouse = True)
def _clean_env_and_state(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    # A light status-capable stub so neither selection nor active_status() imports the
    # heavy diffusers/sd.cpp backends; the active engine NAME comes from module state.
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(status = lambda: {"loaded": False, "repo_id": None}),
    )
    # Default: no resident sd-server (so existing tests exercise the sd-cli path only) and
    # a stubbed runnability probe, so neither reaches the real install/exec path.
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    yield


def _set_device(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = backend, device = backend),
    )


def _set_binary(monkeypatch, path):
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: path)


def _set_runnable(monkeypatch, version = "sd-cli v0"):
    """Stub the runnability probe so a stubbed binary path is treated as executable
    (the router now probes ``SdCppEngine(...).version()`` before committing to native)."""
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: version))


def _select(fam_name = "z-image"):
    """Activate the engine for a family and return which engine was chosen."""
    r.select_and_activate_engine(detect_family(fam_name))
    return r.active_engine_name()


# ── core selection matrix ─────────────────────────────────────────────────────


def test_cpu_with_binary_and_supported_family_picks_sd_cpp(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    assert _select() == ENGINE_SD_CPP
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_cpu_with_only_sd_server_picks_sd_cpp(monkeypatch):
    # An sd-server-only install (no runnable sd-cli) must still route to native: the
    # backend prefers the resident server, so a runnable sd-server is native availability.
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)  # no sd-cli
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: None))
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: "/usr/bin/sd-server")
    assert _select() == ENGINE_SD_CPP


def test_present_but_not_runnable_binary_falls_back(monkeypatch):
    # A binary that exists but cannot run (version() -> None) must fall back to
    # diffusers at selection, not commit native and fail inside the load.
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: None))
    assert _select() == ENGINE_DIFFUSERS
    assert "binary unavailable" in (r.active_status()["fallback_reason"] or "")


@pytest.mark.parametrize("gpu", ["cuda", "rocm", "xpu"])
def test_gpu_backends_use_diffusers(monkeypatch, gpu):
    _set_device(monkeypatch, gpu)
    _set_binary(monkeypatch, "/usr/bin/sd-cli")  # even with a binary, GPU stays diffusers
    assert _select() == ENGINE_DIFFUSERS
    assert "uses diffusers" in (r.active_status()["fallback_reason"] or "")


def test_forced_diffusers_overrides_cpu(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "diffusers")
    assert _select() == ENGINE_DIFFUSERS
    assert "forced" in (r.active_status()["fallback_reason"] or "")


def test_sd_cpp_disabled_uses_diffusers(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP", "0")
    assert _select() == ENGINE_DIFFUSERS
    assert "disabled" in (r.active_status()["fallback_reason"] or "")


def test_mps_default_diffusers_but_optin_sd_cpp(monkeypatch):
    _set_device(monkeypatch, "mps")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    # Default: MPS is not native-eligible -> diffusers.
    assert _select() == ENGINE_DIFFUSERS
    # Opt in: MPS routes to sd.cpp.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_MPS", "1")
    assert _select() == ENGINE_SD_CPP


def test_unsupported_family_falls_back(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    monkeypatch.setattr(r, "family_sd_cpp_supported", lambda fam: False)
    assert _select() == ENGINE_DIFFUSERS
    assert "no native sd.cpp asset mapping" in (r.active_status()["fallback_reason"] or "")


def test_missing_binary_falls_back(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)  # install unavailable
    assert _select() == ENGINE_DIFFUSERS
    assert "binary unavailable" in (r.active_status()["fallback_reason"] or "")


def test_force_sd_cpp_on_gpu_when_binary_present(monkeypatch):
    _set_device(monkeypatch, "cuda")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    assert _select() == ENGINE_SD_CPP


def test_force_sd_cpp_without_binary_falls_back(monkeypatch):
    _set_device(monkeypatch, "cuda")
    _set_binary(monkeypatch, None)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    assert _select() == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "backend, expected",
    [("rocm", "rocm"), ("cuda", "cuda"), ("xpu", "vulkan"), ("cpu", "auto"), ("mps", "auto")],
)
def test_install_accelerator_maps_backend(backend, expected):
    assert r._install_accelerator_for(backend) == expected


def test_force_native_install_uses_gpu_accelerator(monkeypatch):
    # Forcing sd_cpp on a ROCm host with no binary must install the ROCm build, not the
    # default CPU one -- otherwise the forced-native generation silently runs on CPU.
    _set_device(monkeypatch, "rocm")
    _set_runnable(monkeypatch)
    seen = {}

    def _fake_ensure(**kwargs):
        seen.update(kwargs)
        return "/usr/bin/sd-cli"

    monkeypatch.setattr(r, "ensure_sd_cpp_binary", _fake_ensure)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    assert _select() == ENGINE_SD_CPP
    assert seen.get("accelerator") == "rocm"


# ── active_status annotation ──────────────────────────────────────────────────


def test_active_status_injects_engine_and_reason(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)
    _select()  # -> diffusers fallback (no binary)
    st = r.active_status()
    assert st["engine"] == ENGINE_DIFFUSERS
    assert st["fallback_reason"] and "binary unavailable" in st["fallback_reason"]


# ── engine-switch eviction ordering ───────────────────────────────────────────


def test_switch_unloads_old_engine_before_publishing_new(monkeypatch):
    # The arbiter's diffusion evictor unloads get_active_diffusion_engine(); if the router
    # published the new (empty) engine BEFORE the old one finished unloading, a concurrent
    # chat/video acquire_for could evict the empty engine and take the GPU while the old
    # model was still resident (two large models briefly co-resident -> OOM). Assert the
    # OLD engine stays the published active target until its unload() completes, then the
    # new engine is published.
    seen = {}

    def _fake_engine():
        return SimpleNamespace(
            unload = lambda: seen.__setitem__("active_during_unload", r.active_engine_name()),
            status = lambda: {"loaded": False, "repo_id": None},
        )

    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: _fake_engine())
    r._active_engine_name = ENGINE_SD_CPP
    r._activate(ENGINE_DIFFUSERS, "switch test")
    assert seen["active_during_unload"] == ENGINE_SD_CPP
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_no_switch_keeps_engine_and_refreshes_reason(monkeypatch):
    # When the engine does not change, _activate must not spuriously unload anything and must
    # still refresh the recorded fallback reason (the diffusers-only steady state).
    calls = {"unload": 0}

    def _fake_engine():
        return SimpleNamespace(
            unload = lambda: calls.__setitem__("unload", calls["unload"] + 1),
            status = lambda: {"loaded": False, "repo_id": None},
        )

    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: _fake_engine())
    r._active_engine_name = ENGINE_DIFFUSERS
    r._activate(ENGINE_DIFFUSERS, "still diffusers")
    assert calls["unload"] == 0
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert r.active_status()["fallback_reason"] == "still diffusers"


def test_activate_serializes_switch_and_concurrent_query(monkeypatch):
    # Regression for the check->unload->publish race: _activate releases _lock during the slow
    # unload(), so without the transition lock a second _activate could observe the not-yet-updated
    # active engine, take the "no change" branch, and return the engine the first call is
    # concurrently unloading. Drive both paths on threads and assert the concurrent query is blocked
    # until the switch completes (i.e. the whole transition is serialized).
    import threading

    r._active_engine_name = ENGINE_DIFFUSERS
    r._fallback_reason = None

    release_unload = threading.Event()
    unload_started = threading.Event()

    def _slow_unload():
        unload_started.set()
        release_unload.wait(2.0)

    engine = SimpleNamespace(status = lambda: {"loaded": False, "repo_id": None}, unload = _slow_unload)
    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: engine)

    switch_done = threading.Event()

    def _switch():
        r._activate(ENGINE_SD_CPP, None)  # diffusers -> sd_cpp: unloads the old engine (blocks)
        switch_done.set()

    t = threading.Thread(target = _switch)
    t.start()
    assert unload_started.wait(2.0)  # switch is mid-unload, holding the transition lock

    query_done = threading.Event()

    def _query():
        r._activate(ENGINE_DIFFUSERS, None)  # would hit the "no change" branch pre-fix
        query_done.set()

    q = threading.Thread(target = _query)
    q.start()
    # Serialized: while the switch holds the transition lock the query cannot complete. Pre-fix it
    # would return immediately (active is still diffusers), setting query_done at once.
    assert not query_done.wait(0.4)

    release_unload.set()
    t.join(2.0)
    q.join(2.0)
    assert switch_done.is_set() and query_done.is_set()
