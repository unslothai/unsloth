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
    yield


def _set_device(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = backend, device = backend),
    )


def _set_binary(monkeypatch, path):
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: path)


def _select(fam_name = "z-image"):
    """Activate the engine for a family and return which engine was chosen."""
    r.select_and_activate_engine(detect_family(fam_name))
    return r.active_engine_name()


# ── core selection matrix ─────────────────────────────────────────────────────


def test_cpu_with_binary_and_supported_family_picks_sd_cpp(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, "/usr/bin/sd-cli")
    assert _select() == ENGINE_SD_CPP
    assert r.active_engine_name() == ENGINE_SD_CPP


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
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    assert _select() == ENGINE_SD_CPP


def test_force_sd_cpp_without_binary_falls_back(monkeypatch):
    _set_device(monkeypatch, "cuda")
    _set_binary(monkeypatch, None)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    assert _select() == ENGINE_DIFFUSERS


# ── active_status annotation ──────────────────────────────────────────────────


def test_active_status_injects_engine_and_reason(monkeypatch):
    _set_device(monkeypatch, "cpu")
    _set_binary(monkeypatch, None)
    _select()  # -> diffusers fallback (no binary)
    st = r.active_status()
    assert st["engine"] == ENGINE_DIFFUSERS
    assert st["fallback_reason"] and "binary unavailable" in st["fallback_reason"]
