# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The RoPE kernel's constexpr strides and head counts are fixed per model: a compiled forward and
backward over changing batch sizes and sequence lengths stays one graph and matches eager."""

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

if not has_real_cuda():
    pytest.skip(reason = "needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")

import unsloth  # noqa: F401
from unsloth.kernels import rope_embedding as RE

# Batch never 1 or equal to a head count, seq never head_dim: torch's own 0/1 and duck-size
# specializations would recompile on any kernel.
N_HEADS, N_KV, HEAD_DIM = 8, 4, 64


def _qk(batch, seq):
    torch.manual_seed(batch * 1000 + seq)
    width = (N_HEADS + 2 * N_KV) * HEAD_DIM
    qkv = torch.randn(batch, seq, width, device = "cuda", dtype = torch.float16, requires_grad = True)
    Q = qkv[..., : N_HEADS * HEAD_DIM].view(batch, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
    K = qkv[..., N_HEADS * HEAD_DIM : (N_HEADS + N_KV) * HEAD_DIM]
    K = K.view(batch, seq, N_KV, HEAD_DIM).transpose(1, 2)
    return qkv, Q, K


@pytest.mark.gpu
@pytest.mark.parametrize("with_indices", [False, True])
def test_compiled_rope_keeps_one_graph_across_batch_and_seq(with_indices):
    if not RE._TRACEABLE:
        pytest.skip(reason = "RoPE ops are not traceable on this torch / Triton")
    from torch._dynamo.utils import counters

    freqs = torch.outer(
        torch.arange(4096.0, device = "cuda"), torch.rand(HEAD_DIM // 2, device = "cuda")
    )
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos().half(), emb.sin().half()

    def rope(Q, K, idx):
        q, k = RE.fast_rope_embedding(Q, K, cos, sin, idx)
        return q.float().square().sum() + k.float().sum()

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(rope, dynamic = True, fullgraph = True)
    for batch, seq in [(3, 37), (5, 96), (6, 129), (7, 13), (3, 300)]:
        idx = torch.randint(0, 4096, (batch, seq), device = "cuda") if with_indices else None
        qkv, Q, K = _qk(batch, seq)
        compiled(Q, K, idx).backward()
        grad = qkv.grad
        qkv, Q, K = _qk(batch, seq)
        rope(Q, K, idx).backward()
        torch.testing.assert_close(grad, qkv.grad, rtol = 1e-3, atol = 1e-3)
    assert counters["stats"]["unique_graphs"] == 1
