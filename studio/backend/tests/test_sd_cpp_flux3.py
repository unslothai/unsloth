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
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(
        repo, gguf, detect_family(repo)
    )
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
    body = (
        b"ELF-ish padding " * 64 + (b"Flux.3 Image Klein" if marker else b"Flux.2 Klein") + b" tail"
    )
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
def test_no_gpu_host_routes_flux3_gguf_to_sd_cpp_only_on_a_capable_build(
    _router, monkeypatch, tmp_path, name
):
    fam = detect_family(name)
    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-old", marker = False)
    )
    r.select_and_activate_engine(fam)
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert "predates" in (r._fallback_reason or "")

    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-new", marker = True)
    )
    r.select_and_activate_engine(fam)
    assert r.active_engine_name() == ENGINE_SD_CPP


def _flux3_native_state(name):
    import core.inference.sd_cpp_backend as bk
    from core.inference.sd_cpp_engine import SdCppModelFiles

    fam = detect_family("", override = name)
    files = SdCppModelFiles(
        diffusion_model = "/m/flux3.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )
    return bk._SdState(
        repo_id = "unsloth/flux-3-image-klein-GGUF",
        base_repo = fam.base_repo,
        family = fam,
        device = "cpu",
        files = files,
        sampling_method = fam.sd_cpp_sampling_method,
        mode = "oneshot",
        server = None,
    )


def _ref_png(color):
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("name", ["flux.3-klein", "flux.3-klein-base"])
def test_flux3_references_run_natively_without_a_projector(monkeypatch, name):
    # sd.cpp feeds FLUX.3 references through the VAE: no vision projector, no edit marker. Edit is that same call.
    from PIL import Image

    engine = _FakeEngine()
    b = SdCppDiffusionBackend(engine = engine)
    b._state = _flux3_native_state(name)
    status = b.status()
    assert status["workflows"] == ["txt2img", "reference", "edit"]
    assert status["conditioning"]["unified_edit"] is True
    assert status["conditioning"]["localized_edit_modes"] == []
    assert status["conditioning"]["notes"] == []
    seen = {}

    def _capture(files, params, output_path, **kw):
        seen["refs"] = len(params.ref_images)
        Image.new("RGB", (1, 1)).save(output_path)

    monkeypatch.setattr(engine, "generate", _capture)
    out = b.generate(
        prompt = "p",
        steps = 2,
        workflow = "reference",
        width = 512,
        height = 512,
        init_image = _ref_png((10, 0, 0)),
        reference_images = [_ref_png((20, 0, 0))],
    )
    assert seen["refs"] == 2 and out["workflow"] == "reference"
    seen.clear()
    out = b.generate(
        prompt = "p",
        steps = 2,
        workflow = "edit",
        width = 768,
        height = 512,
        init_image = _ref_png((1, 1, 1)),
        reference_images = [_ref_png((2, 2, 2))],
    )
    assert seen["refs"] == 2 and out["workflow"] == "edit"
    from core.inference.diffusion_conditioning import LocalizedEdit

    with pytest.raises(ValueError, match = "localized_edit"):
        b.generate(
            prompt = "p",
            workflow = "edit",
            init_image = _ref_png((1, 1, 1)),
            width = 512,
            height = 512,
            localized_edit = LocalizedEdit("paint", _ref_png((255, 255, 255))),
        )


def test_the_native_load_refuses_the_other_flux3_variant():
    # The diffusers loader's variant refusal runs on the native path too, right after the assets resolve and before
    # the sigmas and the native state are built.
    import inspect

    import core.inference.sd_cpp_backend as bk

    src = inspect.getsource(bk.SdCppDiffusionBackend._run_load)
    fetch = src.index("paths = self._fetch_assets(")
    check = src.index('assert_flux3_checkpoint_matches_family(fam, paths["diffusion_model"])')
    sigmas = src.index("sample_sigmas, grid_base = _base_sample_sigmas(")
    assert fetch < check < sigmas
