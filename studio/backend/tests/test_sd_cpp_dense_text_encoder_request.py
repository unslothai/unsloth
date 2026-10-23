# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit dense text encoder (text_encoder_quant "none" / "off") on the native engine. The diffusers backend
loads the released bf16 encoder for it; sd.cpp reads its own GGUF encoder, so a pick whose native encoder is
quantised (FLUX.3's Q8_0 Qwen3-VL) is refused with the reason instead of silently loading the quant."""

import pytest

from core.inference import diffusion_families as df
from core.inference.diffusion_families import detect_family


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(df.SD_CPP_TEXT_ENCODER_QUANT_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", raising = False)


_FLUX3 = ("unsloth/flux-3-image-klein-GGUF", "flux-3-image-klein-UD-Q4_K_XL.gguf")


def _gate(**kwargs):
    from routes.inference import _assert_native_precision_unset
    return _assert_native_precision_unset(**kwargs)


@pytest.mark.parametrize("value", ["none", "off", " None "])
def test_flux3_native_refuses_a_dense_encoder_request(value):
    fam = detect_family(_FLUX3[0])
    assert fam is not None and fam.name == "flux.3-klein"
    with pytest.raises(RuntimeError, match = r"could not be used.*Qwen3-VL-4B-Instruct-Q8_0\.gguf"):
        _gate(
            text_encoder_quant = value,
            fam = fam,
            repo_id = _FLUX3[0],
            gguf_filename = _FLUX3[1],
        )


def test_auto_and_unset_still_pass_on_flux3():
    fam = detect_family(_FLUX3[0])
    for value in (None, "", "auto"):
        _gate(text_encoder_quant = value, fam = fam, repo_id = _FLUX3[0], gguf_filename = _FLUX3[1])


def test_a_bf16_native_encoder_honours_the_dense_request():
    """Klein 4B / 9B and Z-Image read the BF16 encoder natively: nothing to refuse."""
    for repo, gguf in (
        ("unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"),
        ("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf"),
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
    ):
        fam = detect_family(repo)
        assert df.sd_cpp_quantized_text_encoders(fam, repo, gguf) == (), repo
        _gate(text_encoder_quant = "none", fam = fam, repo_id = repo, gguf_filename = gguf)


def test_the_override_that_quantises_klein_is_refused_too(monkeypatch):
    monkeypatch.setenv(df.SD_CPP_TEXT_ENCODER_QUANT_ENV, "q8_0")
    repo, gguf = "unsloth/FLUX.2-klein-4B-GGUF", "flux-2-klein-4b-Q4_K_M.gguf"
    fam = detect_family(repo)
    with pytest.raises(RuntimeError, match = "Qwen3-4B-Q8_0.gguf"):
        _gate(text_encoder_quant = "off", fam = fam, repo_id = repo, gguf_filename = gguf)


def test_the_opt_in_precision_fallback_keeps_loading(monkeypatch):
    from core.inference import diffusion_auto_policy

    monkeypatch.setattr(diffusion_auto_policy, "precision_fallback_allowed", lambda: True)
    fam = detect_family(_FLUX3[0])
    _gate(text_encoder_quant = "none", fam = fam, repo_id = _FLUX3[0], gguf_filename = _FLUX3[1])


def test_flux3_native_encoder_is_flagged_quantised():
    fam = detect_family(_FLUX3[0])
    assert df.sd_cpp_quantized_text_encoders(fam, *_FLUX3) == ("Qwen3-VL-4B-Instruct-Q8_0.gguf",)
