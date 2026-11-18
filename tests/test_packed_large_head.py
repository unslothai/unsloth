# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Packed rows of head dim 512 layers (Gemma-4 global) run the large head kernel, one causal interval
per document, compiled and eager, instead of masked SDPA / the efficient kernel (CUDA sm80+)."""

import pytest

torch = pytest.importorskip("torch")
# ROCm reports gfx capabilities >= 8, but the large head kernel and the traceable varlen path are CUDA only.
if (
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 8
):
    pytest.skip("needs CUDA sm80+", allow_module_level = True)
hf = pytest.importorskip("unsloth.utils.hf_packed_attention")
lh = pytest.importorskip("unsloth.kernels.large_head_attention")

LENGTHS = [37, 1, 64, 90, 8]


class _Global(torch.nn.Module):
    is_causal = True
    sliding_window = None


def _inputs():
    g = torch.Generator(device = "cuda").manual_seed(0)
    T = sum(LENGTHS)
    q, k, v = (
        torch.randn(1, h, T, 512, device = "cuda", dtype = torch.bfloat16, generator = g)
        for h in (8, 1, 1)
    )
    cu = torch.tensor([0, *torch.tensor(LENGTHS).cumsum(0).tolist()], device = "cuda").int()
    doc = torch.repeat_interleave(torch.arange(len(LENGTHS), device = "cuda"), cu.diff())
    mask = (doc[:, None] == doc[None, :]) & torch.ones(T, T, dtype = torch.bool, device = "cuda").tril()
    return [x.requires_grad_() for x in (q, k, v)], cu, mask[None, None]


def _reference(q, k, v):
    outs, start = [], 0
    for L in LENGTHS:
        s = slice(start, start + L)
        outs.append(
            torch.nn.functional.scaled_dot_product_attention(
                *(x[:, :, s].float() for x in (q, k.expand_as(q), v.expand_as(q))), is_causal = True
            )
        )
        start += L
    return torch.cat(outs, 2).transpose(1, 2)


def _check(out, q, k, v):
    ref = _reference(q, k, v)
    grads = torch.autograd.grad(out.float().square().sum(), (q, k, v))
    ref_grads = torch.autograd.grad(ref.square().sum(), (q, k, v))
    torch.testing.assert_close(out.float(), ref, atol = 2e-2, rtol = 2e-2)
    for a, b in zip(grads, ref_grads):
        assert ((a.float() - b).norm() / b.norm()).item() < 2e-2


@pytest.fixture
def wrapper():
    orig, previous = [], hf._ORIG_SDPA[0]
    hf._ORIG_SDPA[0] = lambda *a, **k: orig.append(1) or (None, None)
    assert lh._self_check(torch.device("cuda", torch.cuda.current_device()), torch.bfloat16)
    yield orig
    hf._ORIG_SDPA[0] = previous


def _kernels(fn):
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    return out, {e.name for e in prof.events()}


def test_compiled_block_runs_the_large_head_kernel_per_document(wrapper):
    (q, k, v), cu, mask = _inputs()
    module = _Global()

    def attend(q, k, v, cu, mask):
        return hf._sdpa_packed_varlen(
            module,
            q,
            k,
            v,
            mask,
            packed_seq_lengths = cu.diff(),
            **{hf.BLOCK_VARLEN_KWARG: (cu, max(LENGTHS))},
        )[0]

    torch._dynamo.reset()
    compiled = torch.compile(attend, backend = "aot_eager", fullgraph = True)
    out, names = _kernels(lambda: compiled(q, k, v, cu, mask))
    assert not wrapper, "packed hd512 rows fell back to masked SDPA"
    assert any("_lh_fwd" in n for n in names)
    _check(out, q, k, v)


def test_eager_packed_rows_run_the_large_head_kernel(wrapper):
    (q, k, v), cu, mask = _inputs()
    before = hf.HF_PACKED_ATTENTION_STATS.get("large_head", 0)
    out = hf._sdpa_packed_varlen(_Global(), q, k, v, mask, packed_seq_lengths = cu.diff())[0]
    assert not wrapper
    assert hf.HF_PACKED_ATTENTION_STATS.get("large_head", 0) == before + 1
    _check(out, q, k, v)


class _PerLayerConfig:
    """Gemma-4 unified: the global kv head count is per layer and refuses a global read."""

    _attn_implementation = "sdpa"
    num_attention_heads = 8

    @property
    def num_key_value_heads(self):
        raise ValueError("'num_key_value_heads' is a per-layer attribute")


def test_per_layer_kv_heads_keep_the_packed_block_compiled(wrapper):
    (q, _, _), cu, mask = _inputs()
    attention = _Global()
    attention.config, attention.head_dim, attention.num_key_value_groups = _PerLayerConfig(), 512, 8
    layer = torch.nn.Module()
    layer.self_attn = attention
    hidden = torch.zeros(1, q.shape[2], 64, device = "cuda", dtype = torch.bfloat16)
    kwargs = {"packed_seq_lengths": cu.diff(), "attention_mask": mask}
    varlen = hf.block_varlen_kwargs(layer, hidden, kwargs)
    assert isinstance(varlen, tuple) and varlen[0].tolist() == cu.tolist()
