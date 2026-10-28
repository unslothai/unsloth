# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native (sd.cpp) MOSS-Image: the assets sd-cli is handed (FLUX.2 small-decoder VAE, Qwen3.5-9B, and the
tokenizer.json sd.cpp's MOSS conditioner requires on --tokenizer), the generic optional --tokenizer plumbing, and
the routing rule that a MOSS GGUF loads only natively, from a build that carries the architecture."""

from __future__ import annotations

import pytest

from core.inference import diffusion_engine_router as r
from core.inference.diffusion_families import (
    _FAMILIES,
    detect_family,
    family_gguf_loadable,
    family_sd_cpp_supported,
    sd_cpp_text_encoders_for,
)
from core.inference.sd_cpp_args import (
    SdCppGenParams,
    SdCppModelFiles,
    SdCppVideoGenParams,
    build_sd_cpp_command,
    build_sd_cpp_server_command,
    build_sd_cpp_video_command,
)
from core.inference.sd_cpp_backend import SdCppDiffusionBackend, sd_cpp_binary_runs_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

from .test_diffusion_engine_router import (  # noqa: F401 -- the autouse fixture resets router state
    _clean_env_and_state,
    _set_binary,
    _set_device,
    _set_runnable,
)
from .test_sd_cpp_backend import _FakeEngine

_REPO = "unsloth/MOSS-Image-SFT-GGUF"
_GGUF = "MOSS-Image-SFT-Q4_K_M.gguf"
_SNAP = "unsloth/MOSS-Image-SFT-FP8"
_TE = ("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q8_0.gguf", "llm")
_TOKENIZER = (_SNAP, "tokenizer/tokenizer.json")


def _moss():
    fam = detect_family(_REPO)
    assert fam is not None and fam.name == "moss-image"
    return fam


def test_the_family_row_carries_the_native_assets():
    fam = _moss()
    assert family_sd_cpp_supported(fam) and family_gguf_loadable(fam)
    assert not fam.pipeline_only and fam.native_transformer_files_only
    # MOSS's own VAE (the FLUX.2 small decoder), not FLUX.2-dev's; sd.cpp detects its layout itself.
    assert fam.sd_cpp_vae == (_SNAP, "vae/diffusion_pytorch_model.safetensors")
    assert fam.sd_cpp_vae_format is None
    # sd.cpp's MOSS default shift (5) is the reference recipe, so no override.
    assert fam.sd_cpp_flow_shift is None
    assert sd_cpp_text_encoders_for(fam, _REPO, _GGUF) == (_TE,)
    assert fam.sd_cpp_tokenizer == _TOKENIZER
    assert fam.sd_cpp_arch_marker == "MOSS-Image"
    # The Base GGUF resolves to the same row and fetches the same (identical) companions.
    assert detect_family("unsloth/MOSS-Image-Base-GGUF") is fam


def test_only_moss_declares_a_tokenizer_or_native_only_transformer_files():
    assert [f.name for f in _FAMILIES if f.sd_cpp_tokenizer] == ["moss-image"]
    assert [f.name for f in _FAMILIES if f.native_transformer_files_only] == ["moss-image"]


def test_the_asset_specs_fetch_the_tokenizer():
    specs = SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(_REPO, _GGUF, _moss())
    assert specs == [
        (_REPO, _GGUF, "diffusion_model"),
        (_SNAP, "vae/diffusion_pytorch_model.safetensors", "vae"),
        _TE,
        (*_TOKENIZER, "tokenizer"),
    ]
    z = detect_family("z-image")
    assert not any(
        kind == "tokenizer"
        for _r, _f, kind in SdCppDiffusionBackend(engine = _FakeEngine())._asset_specs(
            "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", z
        )
    )


def test_the_tokenizer_is_a_protected_companion_file():
    from hub.utils.companion_assets import _sd_cpp_component_specs
    assert _TOKENIZER in _sd_cpp_component_specs(_moss(), _REPO, _GGUF)


_FILES = dict(diffusion_model = "/m/moss.gguf", vae = "/m/vae.safetensors", llm = "/m/qwen.gguf")


def _tokenizer_arg(cmd):
    return [cmd[i + 1] for i, flag in enumerate(cmd) if flag == "--tokenizer"]


@pytest.mark.parametrize("tokenizer", [None, "/m/tokenizer.json"])
def test_every_builder_passes_the_optional_tokenizer(tokenizer):
    files = SdCppModelFiles(**_FILES, tokenizer = tokenizer)
    expected = [tokenizer] if tokenizer else []
    cli = build_sd_cpp_command(
        binary = "sd-cli", files = files, params = SdCppGenParams(prompt = "x"), output_path = "/o/x.png"
    )
    server = build_sd_cpp_server_command(
        binary = "sd-server", files = files, host = "127.0.0.1", port = 1234
    )
    video = build_sd_cpp_video_command(
        binary = "sd-cli",
        files = files,
        params = SdCppVideoGenParams(prompt = "x", width = 448, height = 256, num_frames = 17, fps = 24),
        output_path = "/o/x.webm",
    )
    for cmd in (cli, server, video):
        assert _tokenizer_arg(cmd) == expected


def test_the_tokenizer_defaults_to_none():
    assert SdCppModelFiles(**_FILES).tokenizer is None


def _binary(tmp_path, name, *, marker):
    path = tmp_path / name
    path.write_bytes(b"ELF-ish padding " * 64 + (b"MOSS-Image" if marker else b"") + b" trailing")
    return str(path)


def test_the_marker_is_the_model_version_string(tmp_path):
    fam = _moss()
    assert sd_cpp_binary_runs_family(_binary(tmp_path, "new", marker = True), fam)
    assert not sd_cpp_binary_runs_family(_binary(tmp_path, "old", marker = False), fam)


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_a_moss_gguf_goes_native_even_on_a_gpu_host(monkeypatch, tmp_path, backend):
    _set_device(monkeypatch, backend)
    _set_runnable(monkeypatch)
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli", marker = True))
    fam = _moss()
    assert r.predict_engine(fam, model_kind = "gguf") == ENGINE_SD_CPP
    r.select_and_activate_engine(fam, model_kind = "gguf")
    assert r.active_engine_name() == ENGINE_SD_CPP
    if backend == "cuda":
        assert r.predict_engine(fam, model_kind = "pipeline") == ENGINE_DIFFUSERS


def test_a_build_without_moss_is_refused_not_sent_to_diffusers(monkeypatch, tmp_path):
    _set_device(monkeypatch, "cuda")
    _set_runnable(monkeypatch)
    # Installs off: the resident pre-975 build is what this host has.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli-old", marker = False))
    fam = _moss()
    fallbacks = []
    with pytest.raises(ValueError, match = "no usable sd.cpp build here supports 'moss-image'"):
        r.predict_engine(fam, model_kind = "gguf")
    with pytest.raises(ValueError, match = "predates 'moss-image' support"):
        r.select_and_activate_engine(
            fam, model_kind = "gguf", before_fallback = lambda: fallbacks.append(1)
        )
    assert fallbacks == []


def test_an_explicit_diffusers_pick_of_a_moss_gguf_is_refused(monkeypatch):
    fam = _moss()
    with pytest.raises(ValueError, match = "Diffusers cannot load a moss-image GGUF"):
        r.refuse_engine_pick(fam, engine = "diffusers", model_kind = "gguf")
    # The pipeline repo on diffusers is fine where diffusers can build it (transformers >= 5.3).
    monkeypatch.setattr(r, "family_pipeline_available", lambda _fam: True)
    r.refuse_engine_pick(fam, engine = "diffusers", model_kind = "pipeline")


@pytest.mark.parametrize(
    "env, value",
    [("UNSLOTH_DIFFUSION_SD_CPP", "0"), ("UNSLOTH_DIFFUSION_ENGINE", "diffusers")],
)
def test_the_native_disable_wins_and_diffusers_refuses_the_gguf(monkeypatch, env, value):
    # The operator's switch is honoured as everywhere else; the diffusers loader then refuses the GGUF before eviction.
    monkeypatch.setenv(env, value)
    _set_device(monkeypatch, "cuda")
    assert r.predict_engine(_moss(), model_kind = "gguf") == ENGINE_DIFFUSERS
    from core.inference.diffusion import DiffusionBackend

    with pytest.raises(ValueError, match = "only on the native stable-diffusion.cpp engine"):
        DiffusionBackend().validate_load_request(_REPO, gguf_filename = _GGUF, model_kind = "gguf")


def test_the_route_validation_passes_a_gguf_native_can_take(monkeypatch, tmp_path):
    # Diffusers validation runs before routing: pass a GGUF bound for sd.cpp, refuse it once native cannot take it.
    from core.inference.diffusion import DiffusionBackend

    _set_device(monkeypatch, "cuda")
    _set_runnable(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "0")
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli", marker = True))
    fam = DiffusionBackend().validate_load_request(_REPO, gguf_filename = _GGUF, model_kind = "gguf")
    assert fam.name == "moss-image"
    assert r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _GGUF)

    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli-old", marker = False))
    assert not r.family_buildable_here(fam, model_kind = "gguf", gguf_filename = _GGUF)
    with pytest.raises(ValueError, match = "only on the native stable-diffusion.cpp engine"):
        DiffusionBackend().validate_load_request(_REPO, gguf_filename = _GGUF, model_kind = "gguf")
    assert r.family_buildable_here(fam, model_kind = "pipeline") == r.family_pipeline_available(fam)


def test_diffusers_refuses_a_moss_single_file_transformer(monkeypatch, tmp_path):
    from core.inference.diffusion import DiffusionBackend

    # Even with a capable native build: sd.cpp takes only the GGUF, so a loose transformer file has no engine.
    _set_device(monkeypatch, "cuda")
    _set_runnable(monkeypatch)
    _set_binary(monkeypatch, _binary(tmp_path, "sd-cli", marker = True))
    with pytest.raises(ValueError, match = "single-file transformer only on the native"):
        DiffusionBackend().validate_load_request(
            _REPO, gguf_filename = "MOSS-Image-SFT-INT8.safetensors", model_kind = "single_file"
        )


def test_a_loose_moss_safetensors_is_not_offered():
    from hub.utils.comfy_models import _single_file_loadable
    assert not _single_file_loadable(_moss(), None, "MOSS-Image-SFT-INT8.safetensors")
    assert _single_file_loadable(detect_family("z-image"), None, "z_image_turbo_bf16.safetensors")
