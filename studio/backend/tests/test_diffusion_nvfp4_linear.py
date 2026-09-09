# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the FlashInfer NVFP4 Linear and the torchao -> flashinfer conversion.

The CUDA-gated group comes first because it is the go/no-go for the whole backend: if torchao's
payload is not FlashInfer's payload then the hosted artifact cannot be re-expressed and would have
to be requantized, which is a different model. Those tests import torch inside the body and skip
without a Blackwell card plus flashinfer. The hermetic group below them runs on CPU torch and
covers the refusals, the walk and the state dict.
"""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_nvfp4_linear as nl
from core.inference import diffusion_nvfp4_ops as ops

# Real DiT shapes: z-image / flux attention projections, and the (256, 15360) modulation
# projection the image policy admits below min_features.
REAL_SHAPES = ((3072, 3072), (12288, 3072), (18432, 3072), (15360, 256))

# Measured on a B200 with flashinfer 0.6.6 over 8 seeds x 4 token counts: the converted layer
# differs from torchao's own NVFP4 module by 0.003 to 0.032 relative (worst case M = 4096), which
# is two independent roundings of the SAME quantization -- the two encoders break e2m1 ties
# differently and the two GEMMs accumulate in a different order. The bound below is set above that
# worst case. The stronger claim, and the one that says the flashinfer arm is not a second, worse
# quantization, is DENSE_GAP_BOUND: each arm's distance to the dense bf16 Linear (0.131) agrees to
# 0.0005 or better at every token count.
FORWARD_REL_BOUND = 0.05
DENSE_GAP_BOUND = 0.002


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("sm_%d%d has no flashinfer NVFP4 kernels" % capability)
    pytest.importorskip("flashinfer")
    pytest.importorskip("torchao")
    return torch


def _torchao_linear(
    torch,
    out_features: int,
    in_features: int,
    *,
    bias: bool = True,
    seed: int = 0,
):
    """A torchao-quantized ``nn.Linear``, the way a hosted NVFP4 checkpoint arrives."""
    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias = bias).to("cuda", torch.bfloat16)
    dense = nn.Linear(in_features, out_features, bias = bias).to("cuda", torch.bfloat16)
    dense.load_state_dict(linear.state_dict())
    quantize_(linear, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    return linear, dense


def _rel(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


# ── T-CUDA-3: payload byte equivalence, the go/no-go ──────────────────────────────────────────


@pytest.mark.parametrize("out_features,in_features", REAL_SHAPES)
def test_torchao_and_flashinfer_pack_the_same_payload(out_features, in_features):
    """torchao's ``qdata``/``scale`` ARE FlashInfer's ``wq``/``w_sf``, which is what makes the
    conversion a re-expression rather than a requantization.

    Block scales are bit-identical, always. The 4-bit codes agree on 99.8 percent of nibbles and
    every disagreement is one e2m1 step in magnitude with the sign preserved -- the two quantisers
    round the same ties differently in fp32, which is a property of the two ENCODERS and not of the
    format. Nothing in the shipped path depends on the two agreeing: the checkpoint carries
    torchao's bytes and the GEMM reads exactly those bytes. What this test guards is that they are
    the same LAYOUT, so that reading them as FlashInfer operands is legitimate.
    """
    torch = _cuda_or_skip()
    import flashinfer
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    torch.manual_seed(0)
    with torch.cuda.device(0):
        w = torch.randn(out_features, in_features, device = "cuda", dtype = torch.bfloat16) * 0.02
        amax = w.float().abs().amax().clamp(min = 1e-8)
        w_gsf = (6.0 * 448.0 / amax).reshape(1)
        per_tensor_scale = (amax / (6.0 * 448.0)).reshape(1)
        tensor = NVFP4Tensor.to_nvfp4(w, per_tensor_scale = per_tensor_scale, is_swizzled_scales = True)
        fi_q, fi_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)

    ao_q = tensor.qdata.view(torch.uint8)
    assert tuple(ao_q.shape) == tuple(fi_q.shape) == (out_features, in_features // 2)
    # The block scales: same bytes, same order, no repack anywhere.
    assert torch.equal(tensor.scale.reshape(-1).view(torch.uint8), fi_sf.reshape(-1))
    assert tensor.scale.numel() == ops._swizzled_sf_numel(out_features, in_features // 16, 128)

    codes = lambda t: torch.stack(((t & 0xF).long(), (t >> 4).long()), -1).reshape(t.shape[0], -1)
    ca, cb = codes(ao_q), codes(fi_q)
    differing = ca != cb
    fraction = float(differing.float().mean())
    assert fraction < 0.005, f"{fraction:.4%} of nibbles differ, which is more than tie noise"
    if fraction:
        # Every disagreement is one magnitude step, same sign: a round-to-nearest tie, not a
        # different encoding.
        assert torch.equal((ca >= 8)[differing], (cb >= 8)[differing])
        assert int(((ca & 7).int() - (cb & 7).int()).abs()[differing].max()) == 1


# ── T-CUDA-4: the converted layer against torchao's own module ────────────────────────────────


@pytest.mark.parametrize("m", [1, 512, 4096, 16384])
def test_converted_layer_matches_the_torchao_module(m):
    torch = _cuda_or_skip()

    linear, dense = _torchao_linear(torch, 18432, 3072)
    x = torch.randn(m, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))
    with torch.inference_mode():
        want = linear(x)
        got = converted(x)
        reference = dense(x)

    assert bool(torch.isfinite(got).all())
    assert tuple(got.shape) == (m, 18432)
    against_torchao = _rel(got, want)
    # Both quantized paths sit the same distance from the dense Linear: the flashinfer arm is not
    # a second, worse quantization, it is the same one on a different kernel.
    assert against_torchao < FORWARD_REL_BOUND, f"M={m} rel={against_torchao:.5f}"
    assert abs(_rel(got, reference) - _rel(want, reference)) < DENSE_GAP_BOUND


def test_converted_layer_keeps_the_leading_dimensions_and_the_bias():
    torch = _cuda_or_skip()

    linear, _ = _torchao_linear(torch, 3072, 3072)
    converted = nl.nvfp4_linear_from_torchao(
        linear, torch.tensor([448.0 * 6.0 / 0.2], device = "cuda")
    )
    assert converted.bias is not None
    x = torch.randn(2, 77, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    with torch.inference_mode():
        assert tuple(converted(x).shape) == (2, 77, 3072)
        # An empty batch is what an attention trim can hand a quantized Linear.
        empty = converted(torch.zeros(0, 3072, device = "cuda", dtype = torch.bfloat16))
    assert tuple(empty.shape) == (0, 3072)


# ── T-CUDA-7: the M = 1 GEMM, never exercised before ──────────────────────────────────────────


@pytest.mark.parametrize("out_features,in_features", [(18432, 3072), (15360, 256)])
def test_m1_gemm_is_finite_on_both_backends(out_features, in_features, capsys):
    """The image policies quantize modulation projections that run at M = 1. An FP4 tensor-core
    GEMM with a single row is the shape furthest from what these kernels are tuned for, so it gets
    its own test rather than being assumed to work because M = 512 does."""
    torch = _cuda_or_skip()
    import time

    linear, _ = _torchao_linear(torch, out_features, in_features)
    x = torch.randn(1, in_features, device = "cuda", dtype = torch.bfloat16) * 0.05
    converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))

    timings = {}
    for name, module in (("torchao", linear), ("flashinfer", converted)):
        with torch.inference_mode():
            for _ in range(3):
                y = module(x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                y = module(x)
            torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - start) / 20 * 1e3
        assert bool(torch.isfinite(y).all()), f"{name} produced a non-finite M=1 result"
        assert tuple(y.shape) == (1, out_features)
    with capsys.disabled():
        print(
            f"\n  M=1 ({in_features} -> {out_features}) per call: "
            + ", ".join(f"{k} {v:.3f} ms" for k, v in timings.items())
        )


# ── T-CUDA-5: one graph, no breaks ────────────────────────────────────────────────────────────


def test_a_two_layer_block_compiles_fullgraph():
    torch = _cuda_or_skip()
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, first, second):
            super().__init__()
            self.first, self.second = first, second

        def forward(self, x):
            return self.second(torch.nn.functional.silu(self.first(x)))

    linear_a, _ = _torchao_linear(torch, 3072, 3072, seed = 1)
    linear_b, _ = _torchao_linear(torch, 3072, 3072, seed = 2)
    x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    gsf = ops.global_scale(x)
    block = Block(
        nl.nvfp4_linear_from_torchao(linear_a, gsf), nl.nvfp4_linear_from_torchao(linear_b, gsf)
    ).eval()

    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(block, fullgraph = True)
    with torch.inference_mode():
        got = compiled(x)
        want = block(x)

    assert not counters["graph_break"], dict(counters["graph_break"])
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
    assert bool(torch.isfinite(got).all())
    # Compiled against eager over a TWO layer chain: inductor fuses the silu and the bias add
    # differently, the intermediate moves by a bf16 ulp or two, and the second layer's 4-bit
    # activation quantiser turns some of those into a one step code change. Measured 0.018.
    assert _rel(got, want) < 0.03
    torch._dynamo.reset()


# ── T-CUDA-6: capture and replay ──────────────────────────────────────────────────────────────


def test_graphed_forward_captures_and_replays_a_converted_block():
    torch = _cuda_or_skip()
    import torch.nn as nn

    from core.inference import diffusion_cuda_graph as cg

    class TinyDiT(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(
            self,
            hidden_states,
            timestep,
            return_dict = True,
        ):
            out = self.layer(hidden_states) + timestep
            if return_dict:
                return types.SimpleNamespace(sample = out)
            return (out,)

    linear, _ = _torchao_linear(torch, 3072, 3072, seed = 3)
    x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    module = TinyDiT(nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))).eval()
    handle = cg.GraphedForward(module).enable()
    try:
        timestep = torch.zeros(1, 1, device = "cuda", dtype = torch.bfloat16)
        with torch.inference_mode():
            eager = handle.orig(hidden_states = x, timestep = timestep, return_dict = False)[0].clone()
            first = None
            for _ in range(20):
                replayed = module(hidden_states = x, timestep = timestep, return_dict = False)[0]
                assert _rel(replayed, eager) < FORWARD_REL_BOUND
                # Every replay is the same kernels on the same input: a replay that drifts is a
                # captured buffer being written from outside the graph.
                first = replayed.clone() if first is None else first
                assert torch.equal(replayed, first)
        assert handle.stats["captures"] == 1
        assert handle.stats["replays"] == 20
        assert bool(torch.isfinite(replayed).all())
    finally:
        cg.uninstall_all([handle])


# ── CUDA: the whole-model conversion and the prewarm ──────────────────────────────────────────


def test_convert_nvfp4_backend_moves_a_real_quantized_tree_and_prewarms_it():
    torch = _cuda_or_skip()
    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    torch.manual_seed(7)
    tree = nn.Sequential(nn.Linear(3072, 3072), nn.SiLU(), nn.Linear(3072, 3072))
    tree = tree.to("cuda", torch.bfloat16).eval()
    quantize_(tree, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    # What the builder bakes: each layer's OWN input amax, not the model input's. A scale taken
    # from the wrong activation is not an error anywhere, it is just a worse model, which is why
    # the builder measures it per fqn.
    with torch.inference_mode():
        hidden = tree[1](tree[0](x))
        before = tree(x)
    metadata = {
        "nvfp4_policy": {"activation_scales_baked": True},
        "act_global_scales": {
            "0": float(ops.global_scale(x)),
            "2": float(ops.global_scale(hidden)),
        },
    }
    assert nl.convert_nvfp4_backend(tree, metadata, "flashinfer") == 2
    assert nl.is_nvfp4_flashinfer_linear(tree[0]) and nl.is_nvfp4_flashinfer_linear(tree[2])
    with torch.inference_mode():
        after = tree(x)
    assert bool(torch.isfinite(after).all())
    assert _rel(after, before) < FORWARD_REL_BOUND

    nl.reset_tuned_shapes()
    assert nl.nvfp4_prewarm(tree, (512,)) == 1  # one distinct (M, K, N) across both layers
    assert tree[0]._tuned is True and tree[2]._tuned is True
    assert nl.nvfp4_prewarm(tree, (512,)) == 0  # the shape cache is process wide
    with torch.inference_mode():
        tuned = tree(x)
    assert bool(torch.isfinite(tuned).all())
    nl.reset_tuned_shapes()


# ── hermetic: the walk, the refusals and the state dict ───────────────────────────────────────


class _FakeNVFP4Tensor:
    """Duck-typed like torchao's tensor subclass for the walk, without needing a GPU to build one.

    The name matters: ``is_nvfp4_tensor`` keys on it rather than importing torchao, so that the
    prequant loader can ask the question on a host that has no torchao at all.
    """

    __name__ = "NVFP4Tensor"

    def __init__(self):
        self.qdata = object()
        self.scale = object()
        self.per_tensor_scale = None


_FakeNVFP4Tensor.__name__ = "NVFP4Tensor"
_FakeNVFP4Tensor.__qualname__ = "NVFP4Tensor"


def _quantized_tree():
    import torch.nn as nn

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 64
            self.out_features = 64
            self.weight = _FakeNVFP4Tensor()

    class Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.to_q = Fake()
            self.attn.to_k = Fake()
            self.plain = nn.Linear(8, 8)

    return Tree()


def test_the_walk_finds_the_quantized_linears_and_leaves_the_plain_one_alone():
    pytest.importorskip("torch")
    tree = _quantized_tree()
    names = [
        name
        for name, mod in nl._iter_linears(tree)
        if nl.is_nvfp4_tensor(getattr(mod, "weight", None))
    ]
    assert names == ["attn.to_q", "attn.to_k"]


class _RecordingLogger:
    """Just enough logger to read back the refusal, with no logging plugin in the way."""

    def __init__(self):
        self.lines: list[str] = []

    def __getattr__(self, _level):
        return self.lines.append

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def test_conversion_refuses_without_baked_activation_scales():
    pytest.importorskip("torch")

    tree = _quantized_tree()
    logger = _RecordingLogger()
    assert nl.convert_nvfp4_backend(tree, {"scheme": "nvfp4"}, "flashinfer", logger = logger) == 0
    assert "bakes no activation scales" in logger.text
    assert type(tree.attn.to_q).__name__ != "NVFP4FlashInferLinear"


def test_conversion_refuses_when_one_layer_has_no_scale():
    pytest.importorskip("torch")

    tree = _quantized_tree()
    logger = _RecordingLogger()
    metadata = {
        "nvfp4_policy": {"activation_scales_baked": True},
        "act_global_scales": {"attn.to_q": 1234.0},
    }
    assert nl.convert_nvfp4_backend(tree, metadata, "flashinfer", logger = logger) == 0
    assert "attn.to_k" in logger.text
    assert "1 of 2" in logger.text
    assert type(tree.attn.to_q).__name__ != "NVFP4FlashInferLinear"


def test_a_declared_but_empty_scale_block_is_refused_as_loudly_as_a_missing_one():
    pytest.importorskip("torch")

    logger = _RecordingLogger()
    metadata = {"nvfp4_policy": {"activation_scales_baked": True}, "act_global_scales": {}}
    assert nl.convert_nvfp4_backend(_quantized_tree(), metadata, "flashinfer", logger = logger) == 0
    assert "flag set, scales missing" in logger.text


def test_conversion_is_a_no_op_for_the_torchao_backend():
    pytest.importorskip("torch")
    tree = _quantized_tree()
    metadata = {"act_global_scales": {"attn.to_q": 1.0, "attn.to_k": 1.0}}
    assert nl.convert_nvfp4_backend(tree, metadata, "torchao") == 0


def test_conversion_skips_a_model_with_no_nvfp4_weights():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tree = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
    metadata = {"act_global_scales": {"0": 1.0}}
    assert nl.convert_nvfp4_backend(tree, metadata, "flashinfer") == 0
    assert isinstance(tree[0], nn.Linear)
    assert torch is not None


def test_state_dict_round_trips_with_and_without_a_bias():
    torch = pytest.importorskip("torch")

    cls = nl.nvfp4_linear_class()
    assert cls is nl.nvfp4_linear_class()

    def _build(bias: bool):
        return cls(
            64,
            32,
            wq = torch.arange(32 * 32, dtype = torch.uint8).reshape(32, 32),
            w_sf = torch.ones(128, 4, dtype = torch.uint8),
            alpha = torch.tensor([0.25]),
            a_gsf = torch.tensor([1344.0]),
            bias = torch.arange(32, dtype = torch.bfloat16) if bias else None,
        )

    with_bias = _build(True)
    assert sorted(with_bias.state_dict()) == ["a_gsf", "alpha", "bias", "w_sf", "wq"]
    reloaded = _build(True)
    reloaded.load_state_dict(with_bias.state_dict(), strict = True)
    for key, value in with_bias.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], value)

    without_bias = _build(False)
    assert "bias" not in without_bias.state_dict()
    assert nl.is_nvfp4_flashinfer_linear(without_bias)
    assert without_bias._tuned is False


def test_prewarm_without_flashinfer_is_a_no_op(monkeypatch):
    pytest.importorskip("torch")
    import builtins

    real_import = builtins.__import__

    def _no_flashinfer(name, *args, **kwargs):
        if name == "flashinfer":
            raise ImportError("no flashinfer here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_flashinfer)
    assert nl.nvfp4_prewarm(_quantized_tree(), (1, 512)) == 0


def test_a_whole_model_artifact_converts_without_a_policy_block():
    """PR 1's video artifacts quantise EVERY admitted linear and declare no policy at all. The
    backend keys on the baked scales, not on a policy, so those artifacts get the fast layer too;
    a conversion that required a policy block would silently leave every video family on torchao."""
    torch = _cuda_or_skip()
    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    torch.manual_seed(11)
    tree = nn.Sequential(nn.Linear(3072, 3072), nn.SiLU(), nn.Linear(3072, 3072))
    tree = tree.to("cuda", torch.bfloat16).eval()
    quantize_(tree, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
    with torch.inference_mode():
        hidden = tree[1](tree[0](x))
        before = tree(x)
    metadata = {
        "scheme": "nvfp4",
        "family": "wan2.2-ti2v-5b",
        # Top level, because a whole-model artifact has no policy block to carry it.
        "activation_scales_baked": True,
        "act_global_scales": {
            "0": float(ops.global_scale(x)),
            "2": float(ops.global_scale(hidden)),
        },
    }
    assert nl.convert_nvfp4_backend(tree, metadata, "flashinfer") == 2
    assert nl.is_nvfp4_flashinfer_linear(tree[0]) and nl.is_nvfp4_flashinfer_linear(tree[2])
    with torch.inference_mode():
        after = tree(x)
    assert bool(torch.isfinite(after).all())
    assert _rel(after, before) < FORWARD_REL_BOUND


def test_a_whole_model_artifact_that_baked_nothing_says_the_flag_is_set():
    """The same distinction the policy artifacts get: a build whose bake produced no scales is a
    build to rerun, and a checkpoint that never asked for one is a backend to stop asking."""
    pytest.importorskip("torch")

    logger = _RecordingLogger()
    metadata = {"scheme": "nvfp4", "activation_scales_baked": True, "act_global_scales": {}}
    assert nl.convert_nvfp4_backend(_quantized_tree(), metadata, "flashinfer", logger = logger) == 0
    assert "flag set, scales missing" in logger.text
