# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF on a CPU-only torch stays on sd.cpp: a native failure is refused, not swapped for a diffusers load
that downloads the dense encoder the Hub plan never staged (FLUX.3 Klein: 7.2 GB staged, 11.8 GB loaded)."""

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
    r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV,
)


@pytest.fixture(autouse = True)
def _host(monkeypatch):
    r._supported_family_capabilities.cache_clear()
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "off_torch_sd_cpp_device", lambda *_a, **_k: None)
    monkeypatch.setattr(r, "image_install_accelerator", lambda backend: "auto")
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "sd_cpp_binary_runs_family", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "usable_or_recorded_failure", lambda b, *_a, **_k: b)
    # FLUX.3 needs a newer diffusers than some test venvs carry; the pick refusal is not under test here.
    monkeypatch.setattr(r, "family_pipeline_available", lambda fam: True)
    saved = (r._active_engine_name, r._fallback_reason, r._engine_pick, r._pending_engine_pick)
    try:
        yield
    finally:
        r._active_engine_name, r._fallback_reason, r._engine_pick, r._pending_engine_pick = saved
        r._supported_family_capabilities.cache_clear()


def _device(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda **_: SimpleNamespace(backend = backend, device = backend),
    )


def _binaries(monkeypatch, path):
    """No sd.cpp build at all (``path`` None), as after a failed install or an unrunnable one."""
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: path)
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: path)
    monkeypatch.setattr(
        r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "v" if path else None)
    )


FLUX3 = "flux.3-klein"


def test_a_failed_native_route_refuses_the_cpu_gguf_load(monkeypatch):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    unloads = []
    with pytest.raises(ValueError, match = "Choose the Diffusers engine"):
        r.select_and_activate_engine(
            detect_family(FLUX3), model_kind = "gguf", before_fallback = lambda: unloads.append(1)
        )
    assert unloads == []  # refused before anything unloads


def test_the_plan_refuses_what_the_load_refuses(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    with pytest.raises(ValueError, match = "native sd.cpp engine cannot load"):
        r.predict_engine(detect_family(FLUX3), model_kind = "gguf")


def test_a_runnable_build_still_routes_native(monkeypatch):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, "/opt/sd-cli")
    fam = detect_family(FLUX3)
    assert r.predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_an_explicit_diffusers_pick_still_loads_on_torch_cpu(monkeypatch):
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    fam = detect_family(FLUX3)
    assert r.predict_engine(fam, model_kind = "gguf", engine = "diffusers") == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, model_kind = "gguf", engine = "diffusers")
    assert r.active_engine_name() == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "env, value",
    [
        ("UNSLOTH_DIFFUSION_ENGINE", "diffusers"),
        ("UNSLOTH_DIFFUSION_SD_CPP", "0"),
        (r.CPU_GGUF_DIFFUSERS_FALLBACK_ENV, "1"),
    ],
)
def test_the_operator_switches_keep_the_diffusers_route(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    r.select_and_activate_engine(detect_family(FLUX3), model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "backend, kind, fam_name",
    [
        ("cuda", "gguf", FLUX3),  # a GPU torch keeps diffusers for a GGUF
        ("cpu", "pipeline", FLUX3),  # native is GGUF-only
        ("cpu", "gguf", "sdxl"),  # no native asset mapping: diffusers is the only route
    ],
)
def test_other_loads_are_unchanged(monkeypatch, backend, kind, fam_name):
    _device(monkeypatch, backend)
    _binaries(monkeypatch, None)
    fam = detect_family(fam_name)
    assert r.predict_engine(fam, model_kind = kind) == ENGINE_DIFFUSERS
    r.select_and_activate_engine(fam, model_kind = kind)
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_every_natively_mapped_image_family_is_covered(monkeypatch):
    from core.inference.diffusion_families import _FAMILIES, family_sd_cpp_supported

    _device(monkeypatch, "cpu")
    _binaries(monkeypatch, None)
    mapped = [f for f in _FAMILIES if family_sd_cpp_supported(f)]
    assert mapped
    for fam in mapped:
        with pytest.raises(ValueError):
            r.select_and_activate_engine(fam, model_kind = "gguf")
