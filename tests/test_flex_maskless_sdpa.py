# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""A flex call whose causal mask unsloth_zoo dropped runs SDPA is_causal; everything else stays on flex."""

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
    assert getattr(function, "_unsloth_maskless_causal_sdpa", False) is U._FLEX_MASKLESS_SDPA_ENABLED


def test_a_dropped_mask_runs_sdpa_is_causal(qkv):
    q, k, v = qkv
    before = dict(U.FLEX_MASKLESS_SDPA_STATS)
    out = U._maskless_causal_sdpa_forward(Causal(), q, k, v, (None,), {"scaling": D ** -0.5})
    assert out is not None and U.FLEX_MASKLESS_SDPA_STATS["sdpa"] == before["sdpa"] + 1
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal = True, scale = D ** -0.5, enable_gqa = True,
    ).transpose(1, 2)
    assert torch.equal(out[0], reference)


@pytest.mark.parametrize("case", ["mask", "softcap", "s_aux", "not_causal", "no_is_causal", "decode", "cache"])
def test_everything_else_stays_on_flex(qkv, case):
    q, k, v = qkv
    module, args, kwargs = Causal(), (None,), {}
    if case == "mask": args = (torch.ones(B, 1, T, T, device = "cuda", dtype = torch.bool),)
    if case == "softcap": kwargs = {"softcap": 30.0}
    if case == "s_aux": kwargs = {"s_aux": torch.zeros(H, device = "cuda")}
    if case == "not_causal": module.is_causal = False
    if case == "no_is_causal": module = torch.nn.Module()  # vision callers: None means bidirectional
    if case == "decode": q = q[:, :, :1]
    if case == "cache": q = q[:, :, : T // 2]
    assert U._maskless_causal_sdpa_forward(module, q, k, v, args, kwargs) is None
