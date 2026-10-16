# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) FLUX.3 Klein and Klein Base: the single-file assets sd-cli is handed, the text-encoder flag,
the companion-repo classification, and the architecture gate that keeps a pre-FLUX.3 build off the native route."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import (
    detect_family,
    family_sd_cpp_supported,
    sd_cpp_companion_only_repo_ids,
    sd_cpp_text_encoders_for,
)
from core.inference.sd_cpp_args import text_encoder_flags_for_family
from core.inference.sd_cpp_backend import SdCppDiffusionBackend, sd_cpp_binary_runs_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

from .test_sd_cpp_backend import _FakeEngine

_TE = ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q8_0.gguf", "llm")
_VAE_FILE = "vae/diffusion_pytorch_model.safetensors"
# (GGUF repo, GGUF file, family name, the FP8 repo its VAE comes from)
_CASES = (
    (
        "unsloth/flux-3-image-klein-GGUF",
        "flux-3-image-klein-UD-Q4_K_XL.gguf",
        "flux.3-klein",
        "unsloth/flux-3-image-klein-FP8",
    ),
    (
        "unsloth/flux-3-image-klein-base-GGUF",
        "flux-3-image-klein-base-UD-Q4_K_XL.gguf",
        "flux.3-klein-base",
        "unsloth/flux-3-image-klein-base-FP8",
    ),
)


@pytest.mark.parametrize("repo, gguf, name, fp8_repo", _CASES)
def test_the_gguf_repo_maps_to_its_native_assets(repo, gguf, name, fp8_repo):
    fam = detect_family(repo)
    assert fam is not None and fam.name == name
    assert family_sd_cpp_supported(fam)
    assert fam.sd_cpp_vae == (fp8_repo, _VAE_FILE)
    # sd.cpp detects the FLUX.2 VAE layout for FLUX.3 itself, and applies FLUX.3's own 6.93 shift.
    assert fam.sd_cpp_vae_format is None
    assert fam.sd_cpp_flow_shift is None
    assert fam.sd_cpp_sampling_method == "euler"
    assert sd_cpp_text_encoders_for(fam, repo, gguf) == (_TE,)
    # References go through the VAE: no projector is planned, so text-to-image and editing fetch the same files.
    assert not any(kind == "llm_vision" for _r, _f, kind in fam.sd_cpp_text_encoders)
    assert text_encoder_flags_for_family(name) == ("--llm",)


@pytest.mark.parametrize("repo, gguf, name, fp8_repo", _CASES)
def test_the_load_plans_the_vae_and_the_encoder(repo, gguf, name, fp8_repo):
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(repo, gguf, detect_family(repo))
    assert (fp8_repo, _VAE_FILE, "vae") in specs
    assert _TE in specs


def test_the_encoder_repo_is_a_companion_and_the_fp8_repos_stay_loadable():
    companions = sd_cpp_companion_only_repo_ids()
    assert "unsloth/qwen3-vl-4b-instruct-gguf" in companions
    # The VAE lives in the prequant repos, which carry real denoisers and must stay in the catalog.
    assert "unsloth/flux-3-image-klein-fp8" not in companions
    assert "unsloth/flux-3-image-klein-base-fp8" not in companions


def _binary(tmp_path, name, *, marker):
    path = tmp_path / name
    body = b"ELF-ish padding " * 64 + (b"Flux.3 Image Klein" if marker else b"Flux.2 Klein") + b" tail"
    path.write_bytes(body)
    return str(path)


@pytest.mark.parametrize("name", ["flux.3-klein", "flux.3-klein-base"])
def test_a_build_without_flux3_cannot_run_the_family(tmp_path, name):
    fam = detect_family(name)
    assert fam.sd_cpp_arch_marker == "Flux.3 Image Klein"
    assert not sd_cpp_binary_runs_family(_binary(tmp_path, "old", marker = False), fam)
    assert sd_cpp_binary_runs_family(_binary(tmp_path, "new", marker = True), fam)


@pytest.fixture
def _router(monkeypatch):
    r._supported_family_capabilities.cache_clear()
    for e in (
        "UNSLOTH_DIFFUSION_ENGINE",
        "UNSLOTH_DIFFUSION_SD_CPP",
        "UNSLOTH_DIFFUSION_SD_CPP_MPS",
        "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    ):
        monkeypatch.delenv(e, raising = False)
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = "cpu", device = "cpu"),
    )
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0"))
    saved = (r._active_engine_name, r._fallback_reason)
    try:
        yield
    finally:
        r._active_engine_name, r._fallback_reason = saved
        r._supported_family_capabilities.cache_clear()


@pytest.mark.parametrize("name", ["flux.3-klein", "flux.3-klein-base"])
def test_no_gpu_host_routes_flux3_gguf_to_sd_cpp_only_on_a_capable_build(_router, monkeypatch, tmp_path, name):
    fam = detect_family(name)
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-old", marker = False))
    r.select_and_activate_engine(fam)
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert "predates" in (r._fallback_reason or "")

    monkeypatch.setattr(r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-new", marker = True))
    r.select_and_activate_engine(fam)
    assert r.active_engine_name() == ENGINE_SD_CPP
