# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Maxwell / Pascal (sm_5x / sm_6x): ptxas rejects ld eviction hints below sm_70, so the indexed
RoPE kernel crashed a GTX 10xx / Tesla P4 with "Modifier '.evict_first' on 'ld' requires .target
sm_70 or higher", and the fused NF4 dequant failed to compile before falling back to bitsandbytes.
The RoPE cases compile for the old target with the ptxas Triton ships: no old GPU needed."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs an NVIDIA CUDA device", allow_module_level = True)

triton = pytest.importorskip("triton")
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from unsloth.kernels import utils as utils_mod
from unsloth.kernels import rope_embedding as rope_mod


def _compile(fn, signature, constexprs, cc):
    signature = {**signature, **{k: "constexpr" for k in constexprs}}
    return triton.compile(
        ASTSource(fn, signature = signature, constexprs = constexprs),
        target = GPUTarget("cuda", cc, 32),
    )


@pytest.mark.parametrize(
    "capabilities, supported",
    [
        ([(6, 1)], False),
        ([(5, 2)], False),
        ([(7, 0)], True),
        ([(7, 5)], True),
        ([(10, 0)], True),
        ([(8, 6), (6, 1)], False),
        ([(9, 0), (12, 0)], True),
    ],
)
def test_fused_nf4_kernels_only_from_volta(capabilities, supported, monkeypatch):
    # Below sm_70 the bitsandbytes kernels dequantize instead of the fused Triton ones.
    monkeypatch.setattr(utils_mod, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(utils_mod, "DEVICE_COUNT", len(capabilities))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capabilities[index])
    assert utils_mod._nf4_kernels_supported() is supported


def _rope_qk_constexprs(capability, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
    rope_mod._eviction_hints_ok_at.cache_clear()
    try:
        evict = rope_mod._eviction_hints_ok(torch.device("cuda", 0))
    finally:
        rope_mod._eviction_hints_ok_at.cache_clear()
    return dict(
        head_dim = 64,
        n_heads_K = 4,
        BACKWARD_PASS = False,
        HAS_ROPE_INDICES = True,
        BLOCK_SIZE = 64,
        EVICT_INDICES = evict,
    )


_ROPE_SIGNATURE = {
    "Q": "*fp16",
    "Q_batch_stride": "i32",
    "Q_head_stride": "i32",
    "Q_seq_stride": "i32",
    "K": "*fp16",
    "K_batch_stride": "i32",
    "K_head_stride": "i32",
    "K_seq_stride": "i32",
    "cos": "*fp16",
    "cos_row_stride": "i32",
    "sin": "*fp16",
    "sin_row_stride": "i32",
    "rope_embedding_indices": "*i32",
    "seqlen": "i32",
}


@pytest.mark.parametrize("capability", [(5, 2), (6, 0), (6, 1)])
def test_indexed_rope_assembles_before_volta(capability, monkeypatch):
    constexprs = _rope_qk_constexprs(capability, monkeypatch)
    assert constexprs["EVICT_INDICES"] is False
    cc = capability[0] * 10 + capability[1]
    kernel = rope_mod._rope_embedding_QK.fn
    assert _compile(kernel, _ROPE_SIGNATURE, constexprs, cc).asm["cubin"]


@pytest.mark.parametrize("capability", [(7, 0), (7, 5), (8, 0), (9, 0), (10, 0), (12, 0)])
def test_volta_and_newer_keep_the_eviction_hint(capability, monkeypatch):
    constexprs = _rope_qk_constexprs(capability, monkeypatch)
    assert constexprs["EVICT_INDICES"] is True
    ptx = _compile(rope_mod._rope_embedding_QK.fn, _ROPE_SIGNATURE, constexprs, 75).asm["ptx"]
    assert "evict_first" in ptx
