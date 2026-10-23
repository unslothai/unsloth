# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) Krea 2: assets, guidance mapping, companion repos, and the pre-master-721 build gate."""

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
from core.inference.sd_cpp_backend import (
    SdCppDiffusionBackend,
    _map_guidance,
    _negative_prompt_guidance_threshold,
    sd_cpp_binary_runs_family,
)
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

from .test_sd_cpp_backend import _FakeEngine

_TE = ("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q8_0.gguf", "llm")
_VAE = ("unsloth/Krea-2-Turbo", "vae/diffusion_pytorch_model.safetensors")
# Third-party Krea 2 GGUF repos (Turbo and Raw); no unsloth GGUF exists yet.
_GGUFS = (
    ("realrebelai/KREA-2_GGUFs", "TURBO/Krea-2-Turbo-Q4_K_M.gguf"),
    ("realrebelai/KREA-2_GGUFs", "BASE/Krea-2-Base-Q4_K_M.gguf"),
    ("vantagewithai/Krea-2-Turbo-GGUF", "krea2_turbo-Q4_K_M.gguf"),
    ("stduhpf/Krea-2-Turbo-GGUF", "krea2-turbo-Q5_0.gguf"),
)


@pytest.mark.parametrize("repo, gguf", _GGUFS)
def test_the_gguf_repo_maps_to_its_native_assets(repo, gguf):
    fam = detect_family(repo)
    assert fam is not None and fam.name == "krea-2"
    assert family_sd_cpp_supported(fam)
    assert fam.sd_cpp_vae == _VAE
    # sd.cpp detects the Wan 2.1 VAE layout and applies Krea2's own 1.15 flow shift.
    assert fam.sd_cpp_vae_format is None
    assert fam.sd_cpp_flow_shift is None
    assert fam.sd_cpp_sampling_method == "euler"
    assert sd_cpp_text_encoders_for(fam, repo, gguf) == (_TE,)
    assert text_encoder_flags_for_family("krea-2") == ("--llm",)
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(repo, gguf, fam)
    assert (*_VAE, "vae") in specs
    assert _TE in specs


def test_the_vae_mirror_stays_a_loadable_pipeline():
    companions = sd_cpp_companion_only_repo_ids()
    # unsloth/Krea-2-Turbo is the full ungated mirror of the base, not a VAE-only repack.
    assert "unsloth/krea-2-turbo" not in companions
    assert "unsloth/qwen3-vl-4b-instruct-gguf" in companions


def test_guidance_follows_the_krea2_convention():
    fam = detect_family("krea/Krea-2-Turbo")
    assert _map_guidance(fam, 0.0) == (1.0, None)
    assert _map_guidance(fam, None) == (1.0, None)
    assert _map_guidance(fam, 3.5) == (4.5, None)
    assert _map_guidance(fam, 0.5) == (1.5, None)
    assert _negative_prompt_guidance_threshold(fam) == 0.0


def _binary(tmp_path, name, *, marker):
    path = tmp_path / name
    path.write_bytes(b"ELF-ish padding " * 64 + (b"Krea2" if marker else b"Ideogram 4") + b" tail")
    return str(path)


def test_a_build_without_krea2_cannot_run_the_family(tmp_path):
    fam = detect_family("krea-2")
    assert fam.sd_cpp_arch_marker == "Krea2"
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
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0"))
    saved = (r._active_engine_name, r._fallback_reason)
    try:
        yield
    finally:
        r._active_engine_name, r._fallback_reason = saved
        r._supported_family_capabilities.cache_clear()


def _on(monkeypatch, backend):
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = backend, device = backend),
    )


def test_no_gpu_host_routes_krea2_gguf_to_sd_cpp_only_on_a_capable_build(
    _router, monkeypatch, tmp_path
):
    _on(monkeypatch, "cpu")
    fam = detect_family("realrebelai/KREA-2_GGUFs")
    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-old", marker = False)
    )
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert r._fallback_reason == "the installed sd.cpp build predates 'krea-2' support"

    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-new", marker = True)
    )
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_an_old_sd_server_falls_back_to_a_capable_sd_cli(_router, monkeypatch, tmp_path):
    _on(monkeypatch, "cpu")
    fam = detect_family("krea-2")
    monkeypatch.setattr(
        r, "ensure_sd_server_binary", lambda **_: _binary(tmp_path, "sd-server-old", marker = False)
    )
    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **_: _binary(tmp_path, "sd-cli-new", marker = True)
    )
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_a_gpu_host_keeps_krea2_on_diffusers(_router, monkeypatch, tmp_path):
    _on(monkeypatch, "cuda")
    calls = []
    monkeypatch.setattr(
        r,
        "ensure_sd_cpp_binary",
        lambda **_: calls.append(1) or _binary(tmp_path, "sd-cli-new", marker = True),
    )
    r.select_and_activate_engine(detect_family("krea-2"), model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert r._fallback_reason == "GPU backend 'cuda' uses diffusers"
    assert calls == []


_RAW_IDS = (
    ("BASE/Krea-2-Base-Q4_K_M.gguf", "realrebelai/KREA-2_GGUFs", "krea/Krea-2-Turbo"),
    ("krea2_raw-Q4_K_M.gguf", "vantagewithai/Krea-2-Raw-GGUF", "krea/Krea-2-Turbo"),
    ("Krea-2-Raw-Q8_0.gguf", "someone/whatever-GGUF", "krea/Krea-2-Turbo"),
    (None, "krea/Krea-2-Raw", "krea/Krea-2-Raw"),
)
_TURBO_IDS = (
    ("TURBO/Krea-2-Turbo-Q4_K_M.gguf", "realrebelai/KREA-2_GGUFs", "krea/Krea-2-Turbo"),
    ("krea2_turbo-Q4_K_M.gguf", "vantagewithai/Krea-2-Turbo-GGUF", "krea/Krea-2-Turbo"),
    (None, "krea/Krea-2-Turbo", "krea/Krea-2-Turbo"),
)


@pytest.mark.parametrize("ids", _RAW_IDS)
def test_raw_and_base_named_checkpoints_get_raw_defaults(ids):
    from core.inference.diffusion_families import default_generation_params
    assert default_generation_params(*ids) == (52, 3.5)


@pytest.mark.parametrize("ids", _TURBO_IDS)
def test_turbo_checkpoints_keep_turbo_defaults(ids):
    from core.inference.diffusion_families import default_generation_params
    assert default_generation_params(*ids) == (8, 0.0)
    # FLUX.1 Krea dev is not a Krea 2.
    assert default_generation_params("flux1-krea-dev-Q4_K_M.gguf") == (20, 3.5)


def _state(gguf, repo, base):
    return SimpleNamespace(
        family = detect_family("krea-2"),
        gguf_filename = gguf,
        display_repo_id = None,
        repo_id = repo,
        base_repo = base,
        flow_shift = detect_family("krea-2").sd_cpp_flow_shift,
    )


@pytest.mark.parametrize("ids", _RAW_IDS)
def test_raw_passes_the_resolution_derived_mu(ids):
    from core.inference.sd_cpp_backend import _request_flow_shift

    assert _request_flow_shift(_state(*ids), 256, 256) == pytest.approx(0.5)
    assert _request_flow_shift(_state(*ids), 1024, 1024) == pytest.approx(0.90625)
    assert _request_flow_shift(_state(*ids), 1280, 1280) == pytest.approx(1.15)


@pytest.mark.parametrize("ids", _TURBO_IDS)
def test_turbo_keeps_sd_cpp_fixed_mu(ids):
    from core.inference.sd_cpp_backend import _request_flow_shift

    # None omits --flow-shift, so sd.cpp applies its Krea2 default mu = 1.15, which is what Turbo uses.
    assert _request_flow_shift(_state(*ids), 1024, 1024) is None


def test_other_families_keep_their_fixed_shift():
    from core.inference.sd_cpp_backend import _request_flow_shift

    qwen = detect_family("Qwen/Qwen-Image")
    state = SimpleNamespace(
        family = qwen,
        gguf_filename = "Krea-2-Base-Q4_K_M.gguf",
        display_repo_id = None,
        repo_id = "unsloth/Qwen-Image-GGUF",
        base_repo = "Qwen/Qwen-Image",
        flow_shift = qwen.sd_cpp_flow_shift,
    )
    assert _request_flow_shift(state, 1024, 1024) == qwen.sd_cpp_flow_shift
