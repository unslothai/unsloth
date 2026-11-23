# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp text-encoder precision: the default keeps the precision the released bf16 encoders had (and ComfyUI's
templates load) wherever a quant measurably moves the render, and UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT trades it down."""

import pytest

from core.inference import diffusion_families as df


def _fam(name):
    return next(f for f in df._FAMILIES if f.name == name)


def _llm(encoders):
    return [(repo, filename) for repo, filename, kind in encoders if kind == "llm"]


@pytest.fixture(autouse = True)
def _no_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT", raising = False)


def test_klein_4b_and_z_image_default_to_the_bf16_encoder():
    for name in ("flux.2-klein", "z-image"):
        fam = _fam(name)
        assert _llm(
            df.sd_cpp_text_encoders_for(fam, "unsloth/FLUX.2-klein-4B-GGUF", "x-4b-Q8_0.gguf")
        ) == [("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf")], name


def test_klein_9b_defaults_to_the_bf16_encoder():
    fam = _fam("flux.2-klein")
    for kwargs in ({"inner_dim": 4096}, {"gguf_filename": "flux-2-klein-9b-Q4_K_M.gguf"}):
        got = df.sd_cpp_text_encoders_for(fam, "unsloth/FLUX.2-klein-9B-GGUF", **kwargs)
        assert _llm(got) == [("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf")], kwargs


def test_flux2_dev_never_defaults_below_8_bit():
    fam = _fam("flux.2-dev")
    files = [f for _r, f in _llm(df.sd_cpp_text_encoders_for(fam))]
    assert files == ["Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf"]


@pytest.mark.parametrize(
    "value, tag",
    [("q8_0", "Q8_0"), ("Q4_K_M", "Q4_K_M"), ("bf16", "BF16"), ("bogus", None), ("", None)],
)
def test_env_trades_the_llm_encoder_down(monkeypatch, value, tag):
    monkeypatch.setenv("UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT", value)
    fam = _fam("flux.2-klein")
    want4 = f"Qwen3-4B-{tag or 'BF16'}.gguf"
    want9 = f"Qwen3-8B-{tag or 'BF16'}.gguf"
    assert _llm(df.sd_cpp_text_encoders_for(fam, inner_dim = 3072)) == [
        ("unsloth/Qwen3-4B-GGUF", want4)
    ]
    assert _llm(df.sd_cpp_text_encoders_for(fam, inner_dim = 4096)) == [
        ("unsloth/Qwen3-8B-GGUF", want9)
    ]
    dev = [f for _r, f in _llm(df.sd_cpp_text_encoders_for(_fam("flux.2-dev")))]
    assert dev == [f"Mistral-Small-3.2-24B-Instruct-2506-{tag or 'Q8_0'}.gguf"]


def test_env_leaves_other_encoders_alone(monkeypatch):
    """Only the unsloth LLM GGUFs that ship every precision are rewritten: the FLUX.1 CLIP / T5 and the Qwen-Image
    VL encoder with its projector keep their table entries."""
    monkeypatch.setenv("UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT", "q4_k_m")
    for fam in df._FAMILIES:
        if fam.name in ("flux.2-klein", "flux.2-dev", "z-image"):
            continue
        assert df.sd_cpp_text_encoders_for(fam) == tuple(fam.sd_cpp_text_encoders or ()), fam.name


def test_delete_guard_candidates_cover_every_precision():
    """A load made under another UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT fetched that file, so the guard names them all."""
    cands = {f for _r, f, _k in df.sd_cpp_text_encoder_candidates(_fam("flux.2-klein"))}
    for tag in ("BF16", "Q8_0", "Q4_K_M"):
        assert f"Qwen3-4B-{tag}.gguf" in cands and f"Qwen3-8B-{tag}.gguf" in cands
