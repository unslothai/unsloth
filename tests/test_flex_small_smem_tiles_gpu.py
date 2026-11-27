# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GPU: the sm86 / sm89 tiles reach Inductor through the registered flex_attention.

The gate is forced on (any CUDA card), then the compiled flex forward kernel's shared memory
must fit 101376 bytes, against the default tile that does not; output and grads match SDPA.
"""

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._utils as u

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or getattr(torch.version, "hip", None)
    or torch.cuda.get_device_capability(0) < (8, 0)
    or u.Version(torch.__version__.split("+")[0]) < u.Version("2.10"),
    reason = "needs an Ampere+ CUDA card and torch >= 2.10",
)

L4_SHARED_MEMORY = 101376


def _flex_kernels_shared(monkeypatch, small_ok):
    import triton.compiler.compiler as tc
    from transformers.integrations.flex_attention import flex_attention_forward
    from torch.nn.attention.flex_attention import create_block_mask

    shared = []
    orig = tc.CompiledKernel.__init__

    def rec(self, *a, **k):
        orig(self, *a, **k)
        if "flex_attention" in self.name:
            shared.append((self.name, int(self.metadata.shared)))

    monkeypatch.setattr(tc.CompiledKernel, "__init__", rec)
    monkeypatch.setattr(torch._inductor.config, "compile_threads", 1)
    monkeypatch.setattr(torch._inductor.config, "force_disable_caches", True)
    monkeypatch.setattr(u, "_flex_small_smem_device_ok", lambda: small_ok)
    monkeypatch.setattr(u, "_flex_tuned_bwd_device_ok", lambda: False)
    torch._dynamo.reset()
    # unsloth may already have wrapped the module function at import: wrap the original
    base = getattr(flex_attention_forward, "_unsloth_original_forward", flex_attention_forward)
    wrapped = u._wrap_flex_attention_forward(base)

    torch.manual_seed(0)
    B, H, KV, L, D = 1, 8, 4, 512, 256
    q = torch.randn(B, H, L, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(B, KV, L, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    v = torch.randn(B, KV, L, D, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    mask = create_block_mask(lambda b, h, qi, ki: qi >= ki, None, None, L, L, device = "cuda")

    class Module(torch.nn.Module):
        is_causal = True
        training = True
        num_key_value_groups = H // KV

    out, _ = wrapped(Module(), q, k, v, mask, scaling = D**-0.5)
    out.float().square().sum().backward()
    grads = [t.grad.float().clone() for t in (q, k, v)]
    torch._dynamo.reset()
    return list(shared), out.detach().float(), grads, (q, k, v)


def _sdpa_reference(q, k, v):
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    rep = qr.shape[1] // kr.shape[1]
    o = torch.nn.functional.scaled_dot_product_attention(
        qr.float(),
        kr.float().repeat_interleave(rep, 1),
        vr.float().repeat_interleave(rep, 1),
        is_causal = True,
        scale = q.shape[-1] ** -0.5,
    )
    o.transpose(1, 2).square().sum().backward()
    return o.transpose(1, 2).detach(), [t.grad.float() for t in (qr, kr, vr)]


def test_small_tiles_reach_the_compiled_flex_kernel(monkeypatch):
    on, out_on, grads_on, qkv = _flex_kernels_shared(monkeypatch, True)
    fwd_on = [s for n, s in on if "backward" not in n]
    bwd_on = [s for n, s in on if "backward" in n]
    assert fwd_on, on
    # engagement: with the gate on, every flex forward kernel fits an L4 / A10
    assert max(fwd_on) <= L4_SHARED_MEMORY, on
    assert bwd_on and max(bwd_on) <= L4_SHARED_MEMORY, on
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    if smem >= u._FLEX_TEMPLATE_HEAD_DIM_256_SHARED_MEMORY:
        # control (only where the default tiles fit): the kill-switch path compiles another tile
        off, _, _, _ = _flex_kernels_shared(monkeypatch, False)
        fwd_off = [s for n, s in off if "backward" not in n]
        assert fwd_off and sorted(fwd_on) != sorted(fwd_off), (on, off)
    ref_out, ref_grads = _sdpa_reference(*qkv)
    assert torch.nn.functional.cosine_similarity(out_on.flatten(), ref_out.flatten(), dim = 0) > 0.999
    for g, r in zip(grads_on, ref_grads):
        assert torch.nn.functional.cosine_similarity(g.flatten(), r.flatten(), dim = 0) > 0.999
