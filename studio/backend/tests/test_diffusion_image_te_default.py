# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image families default to a hosted pre-quantized text encoder; ``none`` still pins the bf16 one.

The scheme per family is the ComfyUI template's where ours holds up against the bf16 encoder, else the best hosted
scheme that does (measured numbers in each family's comment). FLUX.2 klein, HiDream-I1 and Ideogram 4 keep bf16 by
default (nothing hosted came close enough), SDXL's CLIPs are too small to bother.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_te_prequant as tpq  # noqa: E402
from core.inference.diffusion_families import _FAMILIES  # noqa: E402
from core.inference.diffusion_precision import (  # noqa: E402
    effective_te_quant,
    resolve_te_quant_request,
)

CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16)

_T5_INT8 = ("unsloth/FLUX.1-schnell-FP8", "FLUX.1-schnell-text_encoder_2-INT8-ConvRot.safetensors")
_T5_FP8 = ("unsloth/FLUX.1-schnell-FP8", "FLUX.1-schnell-text_encoder_2-FP8.safetensors")
_QWEN25VL_INT8 = ("unsloth/Qwen-Image-FP8", "Qwen-Image-text_encoder-INT8-ConvRot.safetensors")
_QWEN25VL_FP8 = ("unsloth/Qwen-Image-FP8", "Qwen-Image-text_encoder-FP8.safetensors")

# family -> (scheme an unset request resolves to, component, hosted (repo, filename), fp8 (repo, filename) behind it)
EXPECTED = {
    "flux.1": ("int8", "text_encoder_2", _T5_INT8, _T5_FP8),
    "flux.1-kontext": ("fp8", "text_encoder_2", _T5_FP8, None),
    "flux.2-dev": (
        "int8",
        "text_encoder",
        ("unsloth/FLUX.2-dev-FP8", "FLUX.2-dev-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/FLUX.2-dev-FP8", "FLUX.2-dev-text_encoder-FP8.safetensors"),
    ),
    "qwen-image": ("int8", "text_encoder", _QWEN25VL_INT8, _QWEN25VL_FP8),
    "qwen-image-edit": ("int8", "text_encoder", _QWEN25VL_INT8, _QWEN25VL_FP8),
    "qwen-image-layered": ("fp8", "text_encoder", _QWEN25VL_FP8, None),
    "hunyuanimage-2.1": ("int8", "text_encoder", _QWEN25VL_INT8, _QWEN25VL_FP8),
    "z-image": (
        "int8",
        "text_encoder",
        ("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/Z-Image-Turbo-FP8", "Z-Image-Turbo-text_encoder-FP8.safetensors"),
    ),
    "krea-2": (
        "int8",
        "text_encoder",
        ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/Krea-2-Turbo-FP8", "Krea-2-Turbo-text_encoder-FP8.safetensors"),
    ),
    "lumina-2": (
        "int8",
        "text_encoder",
        ("unsloth/Lumina-Image-2.0-FP8", "Lumina-Image-2.0-text_encoder-INT8-ConvRot.safetensors"),
        ("unsloth/Lumina-Image-2.0-FP8", "Lumina-Image-2.0-text_encoder-FP8.safetensors"),
    ),
}
# Measured and deliberately left on the bf16 encoder by default.
KEEP_BF16 = ("flux.2-klein", "hidream-i1", "ideogram-4", "sdxl")


def _fam(name):
    return next(f for f in _FAMILIES if f.name == name)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_unset_request_resolves_to_the_hosted_scheme(name):
    fam = _fam(name)
    scheme, component, hosted, fp8 = EXPECTED[name]
    for value in (None, "", "auto"):
        assert resolve_te_quant_request(value, fam.te_quant_auto) == (scheme, True)
    source = tpq.te_prequant_sources_for_base(
        fam, fam.base_repo, te_quant_mode = scheme, target = CUDA_BF16
    )[component]
    assert (source.location, source.filename) == hosted
    if fp8 is not None:
        # The int8 file's fallback is the fp8 file in the same repo, never the dense shards.
        assert source.fallback_filenames[0] == fp8[1]
    # The pre-load gate asks about fp8 support only: neither hosted file needs torchao.
    assert effective_te_quant(scheme, name) == "fp8"


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("value", ["none", "off", "None", "OFF"])
def test_none_still_pins_the_bf16_encoder(name, value):
    fam = _fam(name)
    # Paired with the default: the same family resolves an unset request to a hosted scheme.
    assert resolve_te_quant_request(None, fam.te_quant_auto)[0] is not None
    assert resolve_te_quant_request(value, fam.te_quant_auto) == (None, False)
    assert tpq.te_prequant_sources_for_base(fam, fam.base_repo, te_quant_mode = None, target = CUDA_BF16) == {}


def test_every_image_family_is_either_defaulted_or_measured_and_kept_bf16():
    undefaulted = {f.name for f in _FAMILIES if f.te_quant_auto is None}
    # FLUX.3 Klein and Qwen-Image-2.1 default to int8 ConvRot on their own; everything else is listed above.
    assert undefaulted == set(KEEP_BF16)
    for name in KEEP_BF16:
        assert resolve_te_quant_request(None, _fam(name).te_quant_auto) == (None, False)


@pytest.mark.parametrize(
    "name, base",
    [
        ("qwen-image", "Qwen/Qwen-Image-2512"),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit"),
        ("qwen-image-edit", "Qwen/Qwen-Image-Edit-2509"),
        ("flux.1", "black-forest-labs/FLUX.1-dev"),
        ("flux.1", "black-forest-labs/FLUX.1-Krea-dev"),
        ("z-image", "Tongyi-MAI/Z-Image"),
        ("krea-2", "krea/Krea-2-Raw"),
    ],
)
def test_every_release_of_the_same_encoder_gets_the_default_file(name, base):
    fam = _fam(name)
    scheme, component, hosted, _fp8 = EXPECTED[name]
    source = tpq.te_prequant_sources_for_base(fam, base, te_quant_mode = scheme, target = CUDA_BF16)[component]
    assert (source.location, source.filename) == hosted
