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
    # A light status-capable stub so selection / active_status() never import the heavy diffusers or sd.cpp
    # backends; the active engine NAME is module state. unload() is part of the engine contract, so it is honoured.
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    # Default: no resident sd-server (tests exercise the sd-cli path) and a stubbed runnability probe, so neither reaches a real install/exec.
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    # The selection is module state and several tests below set it by plain assignment, so monkeypatch cannot undo it.
    # Restore it here: a leaked ENGINE_SD_CPP left get_active_diffusion_engine() returning the sd.cpp backend for the rest
    # of the process, and eight tests in test_openai_images_generations_route.py 503'd in a full-suite run.
    saved_engine = r._active_engine_name
    saved_reason = r._fallback_reason
    try:
        yield
    finally:
        r._active_engine_name = saved_engine
        r._fallback_reason = saved_reason


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
    # An sd-server-only install (no runnable sd-cli) still routes to native: the backend prefers the resident server.
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)  # no sd-cli
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: None))
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: "/usr/bin/sd-server")
    assert _select() == ENGINE_SD_CPP


def test_present_but_not_runnable_binary_falls_back(monkeypatch):
    # A binary that exists but cannot run falls back to diffusers at selection, not commit native and fail inside the load.
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
    # Forcing sd_cpp on a ROCm host with no binary must install the ROCm build, else the forced-native generation silently runs on CPU.
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
    # The arbiter's diffusion evictor unloads get_active_diffusion_engine(): publishing the new (empty) engine before the old
    # one finished unloading would let a concurrent acquire take the GPU with the old model still resident.
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
    # When the engine does not change, _activate must not unload anything but must still refresh the recorded fallback reason.
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
    # Regression: without the transition lock a second _activate during a slow unload() reads the stale active engine, so the query must block until the switch ends.
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
    # Serialized: the query cannot complete while the switch holds the transition lock.
    assert not query_done.wait(0.4)

    release_unload.set()
    t.join(2.0)
    q.join(2.0)
    assert switch_done.is_set() and query_done.is_set()


def test_begin_load_on_refuses_an_engine_that_was_switched_away(monkeypatch):
    # A load selects its engine, then yields (device probe, arbiter acquire) before registering. A second load choosing the
    # OTHER engine unloads the captured engine in that gap, so registering there strands a model nothing can reach.
    diffusers = SimpleNamespace(name = "diffusers")
    sd_cpp = SimpleNamespace(name = "sd_cpp")
    active = {"engine": diffusers}
    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: active["engine"])

    started: list[str] = []
    assert r.begin_load_on(diffusers, lambda: started.append("ok") or "status") == "status"
    assert started == ["ok"]

    # A competing request switched the active engine after this one captured `diffusers`.
    active["engine"] = sd_cpp
    with pytest.raises(RuntimeError, match = "engine changed"):
        r.begin_load_on(diffusers, lambda: started.append("leaked"))
    assert started == ["ok"]


def test_begin_load_on_holds_the_transition_lock_while_registering(monkeypatch):
    # The check and the registration must be one operation under the lock a switch takes, so no _activate can slip between them.
    import threading

    engine = SimpleNamespace(name = "diffusers")
    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: engine)

    inside = threading.Event()
    release = threading.Event()

    def _slow_start():
        inside.set()
        release.wait(2.0)
        return "status"

    t = threading.Thread(target = lambda: r.begin_load_on(engine, _slow_start))
    t.start()
    assert inside.wait(2.0)
    assert r._transition_lock.locked()
    release.set()
    t.join(2.0)


def test_switch_aborts_when_the_old_engine_fails_to_unload(monkeypatch):
    # A swallowed teardown failure published the new engine anyway and stranded the old model: the evictor, /images/unload and
    # the next load all resolve through get_active_diffusion_engine(), so nothing could reclaim the resident pipeline (or a live
    # sd-server) and the next load allocated on top of it. Keep the old engine published and fail the switch.
    def _fake_engine():
        def _boom():
            raise RuntimeError("sd-server would not die")

        return SimpleNamespace(unload = _boom, status = lambda: {"loaded": True, "repo_id": "x"})

    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: _fake_engine())
    r._active_engine_name = ENGINE_SD_CPP
    with pytest.raises(RuntimeError, match = "Could not switch the diffusion engine"):
        r._activate(ENGINE_DIFFUSERS, "switch test")
    # Still the old engine, so the resident model remains reachable and reclaimable.
    assert r.active_engine_name() == ENGINE_SD_CPP


# ── predict_engine: the download plan's read-only twin of the selection ───────


def test_predict_engine_matches_the_selection_without_activating(monkeypatch):
    # The download plan is built from this prediction, so it must agree with the selection the load will make, but staging a
    # download must not unload a resident model, so it activates nothing.
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    r._active_engine_name = ENGINE_DIFFUSERS
    assert r.predict_engine(detect_family("z-image"), model_kind = "gguf") == ENGINE_SD_CPP
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_predict_engine_counts_an_installable_binary_as_available(monkeypatch):
    # The first native load on a fresh CPU host installs the binary, so predicting diffusers just because nothing is on disk
    # yet would stage components the load never opens. It must not install anything itself.
    _set_device(monkeypatch, "cpu")
    installs: list[dict] = []

    def _ensure(**kwargs):
        installs.append(kwargs)
        return None

    monkeypatch.setattr(r, "ensure_sd_cpp_binary", _ensure)
    monkeypatch.setattr(r, "ensure_sd_server_binary", _ensure)
    assert r.predict_engine(detect_family("z-image"), model_kind = "gguf") == ENGINE_SD_CPP
    assert installs and all(k.get("allow_install") is False for k in installs)


def test_predict_engine_falls_back_when_install_is_disabled_and_nothing_is_installed(monkeypatch):
    # With installs off and no binary present, the load really will fall back to diffusers.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    assert r.predict_engine(detect_family("z-image"), model_kind = "gguf") == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "kwargs, device",
    [
        ({"model_kind": "pipeline"}, "cpu"),  # native is GGUF-only
        ({"model_kind": "single_file"}, "cpu"),
        ({"model_kind": "gguf"}, "cuda"),  # a usable GPU always means diffusers
    ],
)
def test_predict_engine_returns_diffusers_where_the_load_would(monkeypatch, kwargs, device):
    _set_device(monkeypatch, device)
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    assert r.predict_engine(detect_family("z-image"), **kwargs) == ENGINE_DIFFUSERS


def test_predict_engine_returns_diffusers_for_a_family_without_native_assets(monkeypatch):
    # sdxl has no single-file VAE / text-encoder mapping, so sd-cli cannot serve it.
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    _set_runnable(monkeypatch)
    assert r.predict_engine(detect_family("sdxl"), model_kind = "gguf") == ENGINE_DIFFUSERS
