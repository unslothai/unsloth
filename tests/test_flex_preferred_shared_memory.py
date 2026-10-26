# SPDX-License-Identifier: AGPL-3.0-only
"""Gemma 3 keeps flex attention only where its head_dim 256 template fits the card (#14439)."""

from types import SimpleNamespace
from unittest import mock

import pytest

import unsloth  # noqa: F401
from transformers.utils import import_utils

from unsloth.models import _utils


class SupportsFlex:
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_sdpa = True


L4 = 101376
A100 = 166912
B200 = 232448
# capability per shared memory class, so the sm8x small-tile gate sees a real-looking card
CAPABILITY = {L4: (8, 9), A100: (8, 0), B200: (10, 0)}


def _gemma3(head_dim = 256, model_type = "gemma3_text"):
    return SimpleNamespace(
        model_type = model_type, head_dim = head_dim, num_attention_heads = 4, attention_dropout = 0
    )


@pytest.fixture
def cards(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ENABLE_FLEX_ATTENTION", "1")
    monkeypatch.setattr(import_utils, "is_torch_flex_attn_available", lambda: True, raising = False)
    monkeypatch.setattr(_utils, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(_utils, "_flex_attention_gpu_is_supported", lambda: True)
    monkeypatch.setattr(_utils.torch.version, "hip", None, raising = False)

    def install(*shared_memory):
        monkeypatch.setattr(_utils.torch.cuda, "is_available", lambda: bool(shared_memory))
        monkeypatch.setattr(_utils.torch.cuda, "device_count", lambda: len(shared_memory))
        monkeypatch.setattr(
            _utils.torch.cuda,
            "get_device_properties",
            lambda index = 0: SimpleNamespace(shared_memory_per_block_optin = shared_memory[index]),
        )
        monkeypatch.setattr(
            _utils.torch.cuda,
            "get_device_capability",
            lambda index = 0: CAPABILITY[shared_memory[index]],
        )

    return install


def _resolve(config):
    return _utils.resolve_attention_implementation(SupportsFlex, config, supports_sdpa = False)


@pytest.mark.parametrize("torch_version,env", [("2.9.1", ""), ("2.14.0", "0")])
def test_gemma3_leaves_flex_where_the_template_does_not_fit(cards, monkeypatch, torch_version, env):
    # Without the sm8x small tiles (torch < 2.10, or switched off) the default template must not run.
    monkeypatch.setattr(_utils.torch, "__version__", torch_version)
    monkeypatch.setenv("UNSLOTH_FLEX_SMALL_SMEM_TILES", env)
    cards(L4)
    config = _gemma3()
    assert _resolve(config) == "eager"
    assert config._attn_implementation == "eager"


@pytest.mark.parametrize("shared_memory", [A100, B200])
def test_gemma3_keeps_flex_where_the_template_fits(cards, shared_memory):
    cards(shared_memory)
    assert _resolve(_gemma3()) == "flex_attention"


def test_l4_keeps_flex_with_small_tiles(cards, monkeypatch):
    monkeypatch.setattr(_utils.torch, "__version__", "2.10.0+cu130")
    monkeypatch.delenv("UNSLOTH_FLEX_SMALL_SMEM_TILES", raising = False)
    cards(L4)
    assert _resolve(_gemma3()) == "flex_attention"


def test_a_mixed_box_follows_its_smallest_card(cards):
    cards(A100, L4)
    assert _resolve(_gemma3()) == "eager"


def test_gemma3_27b_head_dim_128_keeps_flex(cards):
    cards(L4)
    assert _resolve(_gemma3(head_dim = 128)) == "flex_attention"


def test_models_that_do_not_prefer_flex_are_untouched(cards):
    cards(L4)
    config = SimpleNamespace(
        model_type = "llama", head_dim = 256, num_attention_heads = 4, attention_dropout = 0
    )
    assert _utils._supports_flex_attention(SupportsFlex, config, "llama") is True


def test_rocm_cpu_and_a_failing_probe_keep_flex(cards, monkeypatch):
    cards()  # no CUDA device
    assert _resolve(_gemma3()) == "flex_attention"
    cards(L4)
    monkeypatch.setattr(_utils.torch.version, "hip", "6.2.0", raising = False)
    assert _resolve(_gemma3()) == "flex_attention"
    monkeypatch.setattr(_utils.torch.version, "hip", None, raising = False)
    with mock.patch.object(
        _utils.torch.cuda, "get_device_properties", side_effect = RuntimeError("probe")
    ):
        assert _resolve(_gemma3()) == "flex_attention"


def test_l4_float32_gemma3_does_not_get_the_small_tiles(cards, monkeypatch):
    # fp32 small tiles overflow L4 smem ("No valid triton configs"), so fp32 keeps eager.
    import torch

    monkeypatch.setattr(_utils.torch, "__version__", "2.10.0+cu130")
    monkeypatch.delenv("UNSLOTH_FLEX_SMALL_SMEM_TILES", raising = False)
    cards(L4)
    resolve = lambda dtype: _utils.resolve_attention_implementation(
        SupportsFlex, _gemma3(), supports_sdpa = False, dtype = dtype
    )
    assert resolve(torch.float32) == "eager"
    assert resolve(torch.bfloat16) == "flex_attention"
    assert resolve(torch.float16) == "flex_attention"
    assert resolve(None) == "flex_attention"
    cards(A100)
    assert resolve(torch.float32) == "flex_attention"
