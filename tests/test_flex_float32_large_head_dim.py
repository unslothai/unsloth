# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A float32 load above head_dim 128 must not pick flex attention: Inductor needs minutes per flex
kernel there (B200, torch 2.15, head_dim 256: 174 s against 7 s in bf16), so a float32 Gemma 3
FastModel load sat compiling for over 20 minutes."""

from types import SimpleNamespace

import pytest
import torch

import unsloth  # noqa: F401
from transformers.utils import import_utils

from unsloth.models import _utils


class SupportsFlex:
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True


B200 = ((10, 0), 232448)
H100 = ((9, 0), 232448)
A100 = ((8, 0), 166912)


@pytest.fixture
def cards(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.delenv("UNSLOTH_FORCE_FLOAT32", raising = False)
    monkeypatch.delenv(_utils._FLEX_LARGE_HEAD_DIM_ENV_VAR, raising = False)
    monkeypatch.setattr(import_utils, "is_torch_flex_attn_available", lambda: True, raising = False)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(_utils, "_flex_attention_gpu_is_supported", lambda: True)
    monkeypatch.setattr(_utils.torch.version, "hip", None, raising = False)
    monkeypatch.setattr(_utils.torch, "__version__", "2.15.0+cu132")

    def install(*devices):
        monkeypatch.setattr(_utils.torch.cuda, "is_available", lambda: bool(devices))
        monkeypatch.setattr(_utils.torch.cuda, "device_count", lambda: len(devices))
        monkeypatch.setattr(
            _utils.torch.cuda, "get_device_capability", lambda index = 0: devices[index][0]
        )
        monkeypatch.setattr(
            _utils.torch.cuda,
            "get_device_properties",
            lambda index = 0: SimpleNamespace(shared_memory_per_block_optin = devices[index][1]),
        )

    return install


def _attention(model_type, head_dim):
    return SimpleNamespace(
        model_type = model_type,
        head_dim = head_dim,
        num_attention_heads = 4,
        attention_dropout = 0,
    )


def _gemma3_vlm(head_dim = 256):
    # Gemma3ForConditionalGeneration: the text tower decides, the vision tower is ignored.
    return SimpleNamespace(
        model_type = "gemma3",
        text_config = _attention("gemma3_text", head_dim),
        vision_config = _attention("siglip_vision_model", 72),
        attention_dropout = 0,
    )


def _resolve(
    config,
    dtype,
    supports_sdpa = False,
    requested = None,
):
    return _utils.resolve_attention_implementation(
        SupportsFlex,
        config,
        requested_attn_implementation = requested,
        supports_sdpa = supports_sdpa,
        dtype = dtype,
    )


@pytest.mark.parametrize("device", [B200, H100, A100])
def test_float32_gemma3_head_dim_256_leaves_flex(cards, device):
    cards(device)
    assert _resolve(_gemma3_vlm(), torch.float32) == "eager"
    assert _resolve(_attention("gemma3_text", 256), torch.float32) == "eager"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, None])
def test_half_precision_gemma3_keeps_flex(cards, dtype):
    cards(B200)
    assert _resolve(_gemma3_vlm(), dtype) == "flex_attention"


def test_forced_float32_gemma3_leaves_flex(cards, monkeypatch):
    # Forced float32 loads bf16 but runs Gemma 3 attention in float32.
    cards(A100)
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    assert _resolve(_gemma3_vlm(), torch.bfloat16) == "eager"


def test_float32_head_dim_128_keeps_flex(cards):
    # Gemma 3 27B: head_dim 128 compiles in under a minute, unchanged.
    cards(B200)
    assert _resolve(_gemma3_vlm(head_dim = 128), torch.float32) == "flex_attention"


def test_float32_large_head_dim_route_falls_back_to_sdpa(cards):
    # A model routed to flex for head_dim > 256 takes SDPA in float32 and keeps flex in bf16.
    cards(B200)
    config = _attention("llama", 512)
    assert _resolve(config, torch.bfloat16, supports_sdpa = True) == "flex_attention"
    config = _attention("llama", 512)
    assert _resolve(config, torch.float32, supports_sdpa = True) == "sdpa"


def test_explicit_flex_and_override_keep_flex(cards, monkeypatch):
    cards(B200)
    assert _resolve(_gemma3_vlm(), torch.float32, requested = "flex_attention") == "flex_attention"
    # Above head_dim 256 the explicit request goes through the flash fallback, which must still keep flex.
    config = _attention("llama", 512)
    assert (
        _resolve(config, torch.float32, supports_sdpa = True, requested = "flex_attention")
        == "flex_attention"
    )
    monkeypatch.setenv(_utils._FLEX_LARGE_HEAD_DIM_ENV_VAR, "1")
    assert _resolve(_gemma3_vlm(), torch.float32) == "flex_attention"
