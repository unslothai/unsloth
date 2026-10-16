# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The chunked (vocab > 65536) cross entropy forward streams its logits chunk with evict_first from
Volta on, and drops the hint below sm_70 where ptxas rejects it. Compile-only with Triton's own
ptxas: no launch."""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is not None:
    pytest.skip("needs an NVIDIA CUDA device", allow_module_level = True)

triton = pytest.importorskip("triton")
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import inspect

if "constexprs" not in inspect.signature(ASTSource.__init__).parameters:
    pytest.skip("ASTSource takes constexprs from Triton 3.3", allow_module_level = True)

from unsloth.kernels import cross_entropy_loss as ce_mod
from unsloth.kernels import rope_embedding as rope_mod

_SIGNATURE = {
    "logits_ptr": "*bf16",
    "loss_ptr": "*fp32",
    "logsumexp_ptr": "*fp32",
    "labels_ptr": "*i64",
}

_COMMON = dict(DO_SOFTCAPPING = False, SOFTCAP = 0.0, DO_LOGIT_SCALING = False, LOGIT_SCALE = 0.0)

_CHUNKED = dict(logits_row_stride = 128256, VOCAB_SIZE = 128256, N_CHUNKS = 2, BLOCK_SIZE = 65536)


def _policy(capability, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: capability)
    rope_mod._eviction_hints_ok_at.cache_clear()
    try:
        ce_mod._chunked_evict_at.cache_clear()
        return ce_mod._chunked_evict_policy(torch.device("cuda", 0))
    finally:
        rope_mod._eviction_hints_ok_at.cache_clear()
        ce_mod._chunked_evict_at.cache_clear()


def _compile(evict, cc):
    fn = ce_mod._chunked_cross_entropy_forward.fn
    signature = {name: _SIGNATURE.get(name, "constexpr") for name in fn.arg_names}
    constexprs = dict(_CHUNKED, EVICT = evict, **_COMMON)
    return triton.compile(
        ASTSource(fn, signature = signature, constexprs = constexprs),
        target = GPUTarget("cuda", cc, 32),
        options = {"num_warps": 32},
    )


@pytest.mark.parametrize("capability", [(5, 2), (6, 1)])
def test_chunked_cross_entropy_assembles_before_volta(capability, monkeypatch):
    evict = _policy(capability, monkeypatch)
    assert evict == ""
    cc = capability[0] * 10 + capability[1]
    assert _compile(evict, cc).asm["cubin"]


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
def test_chunked_cross_entropy_skips_the_hint_on_sm100(capability, monkeypatch):
    # Measured 1.25% slower on B200 with the hint.
    assert _policy(capability, monkeypatch) == ""


@pytest.mark.parametrize("capability", [(7, 0), (7, 5), (8, 0), (9, 0), (12, 0)])
def test_chunked_cross_entropy_streams_logits_from_volta(capability, monkeypatch):
    evict = _policy(capability, monkeypatch)
    assert evict == "evict_first"
    # Each target's own lowering: sm_80+ adds an L2 cache policy to the load.
    assert "ld.global.L1::evict_first" in _compile(evict, capability[0] * 10 + capability[1]).asm["ptx"]
