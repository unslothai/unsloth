# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HiDream grouped MoE (diffusion_regional_compile): each routed expert runs only on its own tokens through
torch._grouped_mm (static shapes, device offsets). CPU here (torch._grouped_mm has a CPU kernel): the gate is opened by
hand. The grouped result equals per-token expert math summed in the dense path's order (fp32: to 1e-6; bf16: rounding),
traces fullgraph with no break, and every non-plain expert (LoRA, quantized subclass, hooks, fp32) keeps the dense
path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(torch, "_grouped_mm"):
    pytest.skip("torch without _grouped_mm", allow_module_level = True)
from core.inference import diffusion_regional_compile as rc  # noqa: E402


def _moe(dtype = torch.bfloat16):
    from diffusers.models.transformers.transformer_hidream_image import MOEFeedForwardSwiGLU

    torch.manual_seed(0)
    m = MOEFeedForwardSwiGLU(32, 64, 4, 2, _force_inference_output = True).to(dtype).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.2)
    return m


def _route(n = 37, seed = 1):
    g = torch.Generator().manual_seed(seed)
    w, idx = torch.randn(n, 4, generator = g).softmax(-1).topk(2)
    return idx.reshape(-1), w.reshape(-1, 1)


def _reference(m, x, idx, w):
    """Per token: its two experts, each weighted, summed lower expert index first (the dense path's order)."""
    idx2, w2 = idx.view(-1, 2), w.view(-1, 2).to(x.dtype)
    outs = torch.stack([e(x) for e in m.experts])  # [E, N, D]
    rows = torch.arange(x.shape[0])
    a = outs[idx2[:, 0], rows] * w2[:, :1]
    b = outs[idx2[:, 1], rows] * w2[:, 1:]
    lo = idx2[:, :1] <= idx2[:, 1:]
    first, second = torch.where(lo, a, b), torch.where(lo, b, a)
    return (torch.zeros_like(x) + first) + second


@pytest.fixture()
def opened(monkeypatch):
    monkeypatch.setattr(rc, "_grouped_supported", lambda *a: True)
    monkeypatch.delenv(rc.GROUPED_MOE_ENV, raising = False)
    # CPU tensors: open the device / dtype part of the gate, keep the expert checks
    monkeypatch.setattr(
        rc,
        "_grouped_ready",
        lambda self, x: self.__dict__.get(rc._GROUPED_ATTR, False)
        and all(rc._plain_expert(e) for e in self.experts),  # noqa: SLF001
    )


@pytest.mark.parametrize("dtype, tol", [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)])
def test_grouped_matches_per_token_math(opened, dtype, tol):
    m = _moe(dtype)
    if dtype is torch.float32:
        # the plain-expert gate is bf16 only; fp32 checks the math itself
        out_fn = lambda x, i, w: rc._moe_infer_grouped(m, x, i, w)  # noqa: SLF001
    else:
        m.__dict__[rc._GROUPED_ATTR] = True  # noqa: SLF001
        out_fn = lambda x, i, w: rc._moe_infer_dense(m, x, i, w)  # noqa: SLF001
    x = torch.randn(37, 32).to(dtype)
    idx, w = _route()
    with torch.no_grad():
        before = rc._GROUPED_CALLS[0]  # noqa: SLF001
        got = out_fn(x, idx, w)
        ref = _reference(m, x, idx, w)
    assert rc._GROUPED_CALLS[0] == before + 1  # noqa: SLF001
    torch.testing.assert_close(got, ref, rtol = tol, atol = tol)


def test_install_marks_the_layers_and_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setattr(rc, "_grouped_supported", lambda *a: True)
    m = torch.nn.Module()
    moe = _moe()
    m.add_module("ff", moe)
    assert rc.install_traceable_moe(m) == 1 and moe.__dict__[rc._GROUPED_ATTR] is True  # noqa: SLF001
    monkeypatch.setenv(rc.GROUPED_MOE_ENV, "0")
    assert rc.install_traceable_moe(m) == 1 and moe.__dict__[rc._GROUPED_ATTR] is False  # noqa: SLF001
    monkeypatch.delenv(rc.GROUPED_MOE_ENV)
    monkeypatch.setattr(rc, "_grouped_supported", lambda *a: False)
    rc.install_traceable_moe(m)
    assert moe.__dict__[rc._GROUPED_ATTR] is False  # noqa: SLF001


def test_cpu_and_fp32_keep_the_dense_path():
    m = _moe()
    m.__dict__[rc._GROUPED_ATTR] = True  # noqa: SLF001
    assert not rc._grouped_ready(m, torch.randn(4, 32).to(torch.bfloat16))  # noqa: SLF001 - CPU tensor
    assert not rc._plain_expert(_moe(torch.float32).experts[0])  # noqa: SLF001


def test_lora_quantized_or_hooked_experts_keep_the_dense_path():
    m = _moe()
    e = m.experts[0]
    assert rc._plain_expert(e)  # noqa: SLF001
    e.w1.bias = torch.nn.Parameter(torch.zeros(e.w1.out_features, dtype = torch.bfloat16))
    assert not rc._plain_expert(e)  # noqa: SLF001
    e.w1.bias = None
    h = e.w2.register_forward_hook(lambda *a: None)
    assert not rc._plain_expert(e)  # noqa: SLF001
    h.remove()
    e.w3 = type("LoraLinear", (torch.nn.Linear,), {})(32, e.w3.out_features, bias = False).to(
        torch.bfloat16
    )
    assert not rc._plain_expert(e)  # noqa: SLF001


def test_grouped_traces_fullgraph_without_breaks(opened):
    m = _moe()
    m.__dict__[rc._GROUPED_ATTR] = True  # noqa: SLF001
    x = torch.randn(37, 32).to(torch.bfloat16)
    idx, w = _route()
    f = lambda x, i, w: rc._moe_infer_grouped(m, x, i, w)  # noqa: SLF001
    with torch.no_grad():
        eager = f(x, idx, w)
        torch._dynamo.reset()
        traced = torch.compile(f, backend = "eager", fullgraph = True)(x, idx, w)
        torch._dynamo.reset()
    assert torch.equal(eager, traced)


def test_whole_tiny_hidream_matches_dense(opened):
    model = diffusers.HiDreamImageTransformer2DModel(
        patch_size = 2, in_channels = 4, out_channels = 4, num_layers = 2, num_single_layers = 2,
        attention_head_dim = 8, num_attention_heads = 2, caption_channels = [8, 8], text_emb_dim = 8,
        num_routed_experts = 4, num_activated_experts = 2, axes_dims_rope = (4, 2, 2), max_resolution = (8, 8),
        llama_layers = [0, 1, 2, 3],
    ).eval().to(torch.bfloat16)  # fmt: skip
    g = torch.Generator().manual_seed(0)
    inputs = dict(
        hidden_states = torch.randn(2, 4, 8, 8, generator = g).to(torch.bfloat16),
        timesteps = torch.tensor([500] * 2),
        encoder_hidden_states_t5 = torch.randn(2, 5, 8, generator = g).to(torch.bfloat16),
        encoder_hidden_states_llama3 = torch.randn(4, 2, 7, 8, generator = g).to(torch.bfloat16),
        pooled_embeds = torch.randn(2, 8, generator = g).to(torch.bfloat16),
        return_dict = False,
    )
    rc.install_traceable_moe(model)
    moes = [mm for mm in model.modules() if type(mm).__name__ == "MOEFeedForwardSwiGLU"]
    with torch.no_grad():
        for mm in moes:
            mm.__dict__[rc._GROUPED_ATTR] = False  # noqa: SLF001
        dense = model(**inputs)[0]
        for mm in moes:
            mm.__dict__[rc._GROUPED_ATTR] = True  # noqa: SLF001
        before = rc._GROUPED_CALLS[0]  # noqa: SLF001
        grouped = model(**inputs)[0]
    assert rc._GROUPED_CALLS[0] - before == len(moes)  # noqa: SLF001
    torch.testing.assert_close(grouped, dense, rtol = 5e-2, atol = 5e-2)


def test_a_grouped_mm_that_raises_keeps_the_dense_experts(monkeypatch):
    """torch 2.8 ships torch._grouped_mm but raises off sm90: the probe must turn the grouped path off."""
    import core.inference.diffusion_regional_compile as rc

    def boom(*a, **k):
        raise RuntimeError(
            "torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0"
        )

    monkeypatch.setattr(torch, "_grouped_mm", boom, raising = False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda *a: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = 0: (8, 9))
    monkeypatch.setattr(rc, "_GROUPED_RUNS", {})
    assert rc._grouped_supported() is False  # noqa: SLF001
    assert rc._GROUPED_RUNS == {0: False}  # noqa: SLF001


def test_the_probe_touches_only_the_denoisers_card(monkeypatch):
    import core.inference.diffusion_regional_compile as rc

    monkeypatch.setattr(torch, "_grouped_mm", lambda *a, **k: None, raising = False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda *a: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i = 0: (8, 9))
    probed = []
    monkeypatch.setattr(rc, "_grouped_runs", lambda i: probed.append(i) or True)
    assert rc._grouped_supported(torch.nn.Linear(2, 2)) is True  # noqa: SLF001
    assert probed == [3]  # a CPU-resident model: the current card only, never all eight
