# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import copy
import types

import pytest
import torch

from real_accelerator import has_real_cuda

if not has_real_cuda():
    pytest.skip("needs a CUDA device", allow_module_level = True)

import unsloth  # noqa: F401
import unsloth.models._utils as U
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

B, H, KV, T, D = 2, 8, 2, 256, 64


class Causal(torch.nn.Module):
    is_causal = True
    num_key_value_groups = H // KV
    training = True


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D, device = "cuda", dtype = torch.bfloat16)
    k = torch.randn(B, KV, T, D, device = "cuda", dtype = torch.bfloat16)
    v = torch.randn(B, KV, T, D, device = "cuda", dtype = torch.bfloat16)
    return q, k, v


def test_the_wrapper_advertises_the_reroute():
    U.patch_flex_attention_kernel_options()
    function = ALL_ATTENTION_FUNCTIONS["flex_attention"]
    assert (
        getattr(function, "_unsloth_maskless_causal_sdpa", False) is U._FLEX_MASKLESS_SDPA_ENABLED
    )


def _flex():
    U.patch_flex_attention_kernel_options()
    return ALL_ATTENTION_FUNCTIONS["flex_attention"]


def _causal_reference(q, k, v, scale):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        is_causal = True,
        scale = scale,
        enable_gqa = True,
    ).transpose(1, 2)


def test_a_dropped_mask_runs_sdpa_is_causal(qkv):
    q, k, v = qkv
    before = dict(U.FLEX_MASKLESS_SDPA_STATS)
    out, _ = _flex()(Causal(), q, k, v, None, scaling = D**-0.5)
    assert U.FLEX_MASKLESS_SDPA_STATS["sdpa"] == before["sdpa"] + 1
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal = True,
        scale = D**-0.5,
        enable_gqa = True,
    ).transpose(1, 2)
    assert torch.equal(out, reference)


@pytest.mark.parametrize("case", ["softcap", "head_dim_512"])
def test_a_dropped_mask_that_stays_on_flex_is_still_causal(case):
    # Never hand flex a None mask for a causal call: it would attend to future tokens.
    torch.manual_seed(0)
    head_dim = 512 if case == "head_dim_512" else D
    q = torch.randn(1, H, 128, head_dim, device = "cuda", dtype = torch.bfloat16)
    k = torch.randn(1, KV, 128, head_dim, device = "cuda", dtype = torch.bfloat16)
    v = torch.randn(1, KV, 128, head_dim, device = "cuda", dtype = torch.bfloat16)
    kwargs = {"softcap": 1e4} if case == "softcap" else {}  # a softcap this large is a no-op
    before = dict(U.FLEX_MASKLESS_SDPA_STATS)
    out, _ = _flex()(Causal(), q, k, v, None, scaling = head_dim**-0.5, **kwargs)
    assert U.FLEX_MASKLESS_SDPA_STATS["sdpa"] == before["sdpa"]
    reference = _causal_reference(q, k, v, head_dim**-0.5)
    assert (out.float() - reference).abs().max() < 2e-2


def test_counters_do_not_mutate_while_compiling(qkv, monkeypatch):
    q, k, v = qkv
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    before = dict(U.FLEX_MASKLESS_SDPA_STATS)
    _flex()(Causal(), q, k, v, None, scaling = D**-0.5)
    assert U.FLEX_MASKLESS_SDPA_STATS == before


@pytest.mark.parametrize(
    "case",
    [
        "mask",
        "softcap",
        "s_aux",
        "not_causal",
        "no_is_causal",
        "is_causal_false",
        "head_dim_512",
        "decode",
        "cache",
        "sdpa_excluded",
    ],
)
def test_everything_else_stays_on_flex(qkv, case):
    q, k, v = qkv
    module, args, kwargs = Causal(), (None,), {}
    if case == "mask":
        args = (torch.ones(B, 1, T, T, device = "cuda", dtype = torch.bool),)
    if case == "softcap":
        kwargs = {"softcap": 30.0}
    if case == "s_aux":
        kwargs = {"s_aux": torch.zeros(H, device = "cuda")}
    if case == "not_causal":
        module.is_causal = False
    if case == "no_is_causal":
        module = torch.nn.Module()  # vision callers: None means bidirectional
    if case == "is_causal_false":
        kwargs = {"is_causal": False}  # per-call bidirectional override
    if case == "head_dim_512":
        q, k, v = (torch.randn(*t.shape[:3], 512, device = "cuda", dtype = t.dtype) for t in (q, k, v))
    if case == "decode":
        q = q[:, :, :1]
    if case == "cache":
        q = q[:, :, : T // 2]
    if case == "sdpa_excluded":
        module.config = types.SimpleNamespace(model_type = "gemma3_text")
    assert U._maskless_causal_sdpa_forward(module, q, k, v, args, kwargs) is None


def _softcap_call(T, requires_grad = False):
    q, k, v = (torch.randn(1, n, T, D, device = "cuda", dtype = torch.bfloat16) for n in (H, KV, KV))
    q.requires_grad_(requires_grad)
    return _flex()(Causal(), q, k, v, None, scaling = D**-0.5, softcap = 1e4), q


def test_a_fallback_mask_first_built_under_inference_mode_still_trains():
    U._CAUSAL_BLOCK_MASKS.clear()
    with torch.inference_mode():
        _softcap_call(128)
    (out, _), q = _softcap_call(128, requires_grad = True)
    out.float().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_the_fallback_mask_cache_is_bounded():
    U._CAUSAL_BLOCK_MASKS.clear()
    for T in range(128, 128 * 13, 128):
        _softcap_call(T)
    assert 0 < len(U._CAUSAL_BLOCK_MASKS) <= 8


def test_the_wrapper_vetoes_configs_whose_layers_cannot_take_sdpa():
    from transformers import Gemma2Config, Gemma3TextConfig, LlamaConfig, Qwen3Config

    accepts = _flex()._unsloth_maskless_causal_sdpa_accepts
    assert accepts(LlamaConfig()) and accepts(Qwen3Config(head_dim = 256))
    assert not accepts(Qwen3Config(head_dim = 512))
    assert not accepts(Gemma3TextConfig())  # SDPA disabled for Gemma 3
    assert not accepts(Gemma2Config())  # softcap


def test_a_class_level_sdpa_opt_out_keeps_the_mask():
    from transformers import LlamaConfig, LlamaForCausalLM

    no_sdpa = type("NoSdpaLlama", (LlamaForCausalLM,), {"_supports_sdpa": False})
    config = LlamaConfig()
    U.resolve_attention_implementation(no_sdpa, config)
    accepts = _flex()._unsloth_maskless_causal_sdpa_accepts
    assert not accepts(copy.deepcopy(config))  # from_pretrained copies the config it is given
    U._NO_SDPA_MODEL_TYPES.discard("llama")
    assert accepts(LlamaConfig())
