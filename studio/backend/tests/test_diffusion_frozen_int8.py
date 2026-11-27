# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frozen W8A8 int8 LoRA base (the int8 default; UNSLOTH_DIFFUSION_INT8_TRAIN=weight-only keeps torchao): numerics
against the dense layer, quantize-once, streamed conversion, and the trainer gates."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_frozen_int8 as fi8


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim = 0).item()


@pytest.mark.parametrize("rows", [40, 8])
def test_forward_and_input_grad_track_the_dense_layer(rows):
    torch.manual_seed(0)
    lin = nn.Linear(64, 96).to(torch.bfloat16).requires_grad_(False)
    m = fi8.FrozenInt8Linear(lin)
    x = torch.randn(2, rows, 64, dtype = torch.bfloat16, requires_grad = True)
    m(x).float().pow(2).sum().backward()
    ref_x = x.detach().clone().requires_grad_(True)
    F.linear(ref_x.float(), lin.weight.float(), lin.bias.float()).pow(2).sum().backward()
    assert _cos(m(x), F.linear(x.float(), lin.weight.float(), lin.bias.float())) > 0.9995
    assert _cos(x.grad, ref_x.grad) > 0.999
    assert x.grad.dtype == torch.bfloat16
    # Frozen: no parameter of the int8 layer takes a gradient, and the weight is int8 codes + fp32 row scales.
    assert m.weight.dtype == torch.int8 and m.weight_scale.dtype == torch.float32
    assert all(not p.requires_grad for p in m.parameters())


def _rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


def _vs_fp64(
    layer_cls,
    rows,
    k = 512,
    n = 256,
    scale_mutation = None,
):
    """Relative L2 error of forward and grad_x against a float64 dense oracle (cosine alone cannot see a scale error)."""
    torch.manual_seed(0)
    lin = nn.Linear(k, n, bias = False).to(torch.bfloat16).requires_grad_(False)
    lin.weight.data.mul_(0.02 / lin.weight.float().std())
    m = layer_cls(lin)
    if scale_mutation is not None:
        m.weight_scale.mul_(scale_mutation)
    x = torch.randn(rows, k, dtype = torch.bfloat16, requires_grad = True)
    g = torch.randn(rows, n, dtype = torch.bfloat16)
    y = m(x)
    y.backward(g)
    x64 = x.detach().double().requires_grad_(True)
    y64 = x64 @ lin.weight.double().t()
    y64.backward(g.double())
    return _rel(y.detach(), y64.detach()), _rel(x.grad, x64.grad)


# Measured (K=512, N=256, bf16): int8 ~0.86e-2 fwd / 0.81e-2 grad_x, fp8 e4m3 3.7e-2.
_INT8_REL_BOUND = 2e-2
_FP8_REL_BOUND = 6e-2


@pytest.mark.parametrize("rows", [64, 512, 8])
def test_int8_relative_error_against_fp64(rows):
    fwd, gx = _vs_fp64(fi8.FrozenInt8Linear, rows)
    assert fwd < _INT8_REL_BOUND and gx < _INT8_REL_BOUND, (fwd, gx)


def test_relative_bound_sees_a_scale_error_cosine_misses():
    # A 5% weight-scale error leaves cosine at ~1 but moves the relative error past the bound.
    fwd, gx = _vs_fp64(fi8.FrozenInt8Linear, 64, scale_mutation = 1.05)
    assert fwd > _INT8_REL_BOUND and gx > _INT8_REL_BOUND


def _cpu_fp8_mm_ok() -> bool:
    try:
        a = torch.zeros(16, 16).to(torch.float8_e4m3fn)
        one = torch.ones((), dtype = torch.float32)
        torch._scaled_mm(a, a.t(), scale_a = one, scale_b = one, out_dtype = torch.bfloat16)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _cpu_fp8_mm_ok(), reason = "no CPU fp8 GEMM here")
@pytest.mark.parametrize("rows", [64, 512])
def test_fp8_relative_error_against_fp64(rows):
    from core.training import diffusion_frozen_fp8 as ff8

    ff8.BACKWARD_WEIGHTS.active = False
    fwd, gx = _vs_fp64(ff8.FrozenFp8Linear, rows)
    assert fwd < _FP8_REL_BOUND and gx < _FP8_REL_BOUND, (fwd, gx)


def test_weight_is_quantized_once_not_per_call(monkeypatch):
    lin = nn.Linear(32, 32, bias = False).to(torch.bfloat16)
    m = fi8.FrozenInt8Linear(lin)
    calls = []
    monkeypatch.setattr(fi8, "quantize_weight", lambda *a, **k: calls.append(1))
    x = torch.randn(24, 32, dtype = torch.bfloat16, requires_grad = True)
    for _ in range(3):
        m(x).sum().backward()
    assert calls == []


def test_rowwise_codes_round_trip():
    w = torch.randn(48, 64)
    q, s = fi8.quantize_weight(w, rows = 16)
    assert q.abs().max() <= 127
    assert (q.float() * s - w).abs().max() <= s.max() * 0.5 + 1e-6


def test_convert_streams_and_skips_unaligned():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.b = nn.Linear(30, 32)
            self.to(torch.bfloat16).requires_grad_(False)

    m = _M()
    done, failed = fi8.convert_frozen_int8(m, {"a", "b"}, device = "meta")
    assert (done, failed) == (1, 1)
    assert isinstance(m.a, fi8.FrozenInt8Linear) and m.a.weight.device.type == "meta"
    # An unaligned layer keeps its bf16 weight (already moved; the final move would take it anyway).
    assert type(m.b) is nn.Linear and m.b.weight.dtype == torch.bfloat16


def test_compiles_without_graph_breaks():
    torch._dynamo.reset()
    lin = nn.Linear(64, 64).to(torch.bfloat16).requires_grad_(False)
    m = fi8.FrozenInt8Linear(lin)
    from torch._dynamo.utils import counters

    counters.clear()
    f = torch.compile(m, backend = "aot_eager", fullgraph = True)
    x = torch.randn(32, 64, dtype = torch.bfloat16, requires_grad = True)
    f(x).float().sum().backward()
    assert x.grad is not None and sum(counters["graph_break"].values()) == 0


def test_trainer_gates(monkeypatch):
    """W8A8 is the int8 default and compiles; the weight-only opt-out runs eager."""
    cfg = type("C", (), {"compile_transformer": "auto", "train_steps": 10})()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    assert dit._int8_w8a8() is True
    assert dit._should_compile(cfg, False, "cuda", "int8") is True
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "weight-only")
    assert dit._int8_w8a8() is False
    assert dit._should_compile(cfg, False, "cuda", "int8") is False
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    assert dit._should_compile(cfg, False, "cuda", "int8") is True
    cfg.compile_transformer = "off"
    assert dit._should_compile(cfg, False, "cuda", "int8") is False


@pytest.mark.parametrize(
    "raw, want",
    [
        (None, True),
        ("", True),
        ("w8a8", True),
        ("W8A8", True),
        ("1", True),
        ("weight-only", False),
        ("weight_only", False),
        ("WO", False),
        ("torchao", False),
        ("0", False),
        ("off", False),
    ],
)
def test_int8_kernel_env(monkeypatch, raw, want):
    from core.training import diffusion_block_swap as bs

    if raw is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raw)
    assert bs.int8_w8a8_requested() is want
    assert dit._int8_w8a8() is want
    assert bs.swap_lowp_mode("int8") == ("int8_w8a8" if want else None)


@pytest.mark.parametrize(
    "kernel, env, want",
    [
        ("w8a8", None, True),
        ("weight-only", None, False),
        (None, None, True),  # not an int8 bundle (or unreadable): the default
        (
            "weight-only",
            "w8a8",
            True,
        ),  # explicit env wins (the identity check then refuses the switch)
        ("w8a8", "weight-only", False),
    ],
)
def test_resume_continues_the_recorded_int8_kernel(monkeypatch, kernel, env, want):
    """Real bundles go through test_diffusion_int8_resume.py; this pins the precedence."""
    from core.training import diffusion_checkpoint as ck

    if env is None:
        monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", env)
    monkeypatch.setattr(ck, "recorded_int8_resume_kernel", lambda p: kernel)
    cfg = type("C", (), {"resume_from_checkpoint": "/some/run"})()
    assert dit._int8_w8a8(cfg) is want
    fresh = type("C", (), {"resume_from_checkpoint": None})()
    assert dit._int8_w8a8(fresh) is (env != "weight-only")


def test_recorded_int8_resume_kernel_never_raises(tmp_path):
    from core.training import diffusion_checkpoint as ck
    assert ck.recorded_int8_resume_kernel(tmp_path / "missing") is None
    assert ck.recorded_int8_resume_kernel(None) is None


def test_the_int8_probe_does_not_cache_an_out_of_memory(monkeypatch):
    """A full card says nothing about the int8 GEMM: the next probe runs again instead of reading a cached no."""
    calls = []

    def flaky(a, b):
        calls.append(1)
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return torch.full((32, 32), 32, dtype = torch.int32)

    real_ones = torch.ones
    monkeypatch.setattr(dit, "_INT8_MM_PROBE_OK", {})
    monkeypatch.setattr(torch, "_int_mm", flaky)
    monkeypatch.setattr(torch, "ones", lambda *a, **k: real_ones(*a, **{**k, "device": "cpu"}))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    dev = torch.device("cuda", 0)
    assert dit._int8_mm_probe_ok(dev) is False
    assert dit._INT8_MM_PROBE_OK == {}
    assert dit._int8_mm_probe_ok(dev) is True and len(calls) == 2
    assert dit._INT8_MM_PROBE_OK == {"cuda:0": True}


def test_apply_lands_the_base_and_reports(monkeypatch):
    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32, bias = False)
            self.norm = nn.LayerNorm(32)
            self.to(torch.bfloat16).requires_grad_(False)

    model = nn.Module()
    model.blocks = nn.ModuleList([_B(), _B()])
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None, device = "meta") is True
    assert all(p.device.type == "meta" for p in model.parameters())
    assert isinstance(model.blocks[0].to_q, fi8.FrozenInt8Linear)
    assert "W8A8" in events[-1]["message"]


def test_fp16_preflight_allows_int8_only_as_w8a8(monkeypatch):
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "weight-only")
    assert "fp16" in (tc.training_precision_preflight_error("flux.2-klein", "int8") or "")
    for env in (None, "w8a8"):
        if env is None:
            monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", env)
        reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
        assert reason is None or "fp16" not in reason
    assert "fp16" in (tc.training_precision_preflight_error("flux.2-klein", "bf16") or "")


@pytest.mark.parametrize("cap, refused", [((7, 0), True), ((7, 5), False), ((8, 9), False)])
def test_fp16_preflight_refuses_w8a8_without_int8_tensor_cores(monkeypatch, cap, refused):
    """sm70 has no int8 GEMM: the trainer's probe would raise after residents were evicted, so the preflight refuses."""
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
    assert ("7.5+" in (reason or "")) is refused


@pytest.mark.parametrize(
    "name, cap, declined",
    [
        ("NVIDIA GeForce GTX 1660 SUPER", (7, 5), True),
        ("NVIDIA GeForce GTX 1650", (7, 5), True),
        ("Tesla T4", (7, 5), False),
        ("NVIDIA GeForce RTX 2080", (7, 5), False),
        ("Tesla V100-SXM2-16GB", (7, 0), True),
        ("NVIDIA L4", (8, 9), False),
    ],
)
def test_fp16_preflight_and_trainer_decline_cards_without_int8_tensor_cores(
    monkeypatch, name, cap, declined
):
    """GTX 16xx report sm75 like a T4 but have no tensor cores: explicit int8 (W8A8 on an fp16 card) is declined
    with a clear reason before eviction, and the trainer's GEMM check declines the same cards."""
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: True)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_TRAIN", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8") or ""
    assert ("tensor cores" in reason) is declined
    if declined:
        assert "nf4" in reason
    decline = dit._int8_gemm_decline(torch.device("cuda", 0))
    assert (decline is not None) is declined
    if declined:
        assert name in decline


def test_int8_probe_runs_on_the_operands_card_and_skips_cards_without_tensor_cores(monkeypatch):
    seen = []

    def fake_int_mm(a, b):
        seen.append(str(a.device))
        return torch.full((32, 32), 32, dtype = torch.int32)

    monkeypatch.setattr(dit, "_INT8_MM_PROBE_OK", {})
    monkeypatch.setattr(torch, "_int_mm", fake_int_mm)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    real_ones = torch.ones
    # The probe allocates on the card it names; record that, then build the tensor on the host.
    monkeypatch.setattr(
        torch, "ones", lambda *a, **k: real_ones(*a, **{**k, "device": "cpu"}).requires_grad_(False)
    )
    # "cuda" resolves to the current card (cuda:1), and the cache keys on it.
    assert str(dit._concrete_device("cuda")) == "cuda:1"
    assert dit._int8_mm_probe_ok("cuda") is True
    assert list(dit._INT8_MM_PROBE_OK) == ["cuda:1"]
    assert (
        dit._int8_mm_probe_ok(torch.device("cuda", 1)) is True and len(seen) == 1
    )  # cached per card
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce GTX 1660 Ti")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    assert (
        dit._int8_mm_probe_ok(torch.device("cuda", 0)) is False and len(seen) == 1
    )  # declined, never run


def test_a_card_without_int8_tensor_cores_falls_back_with_the_reason(monkeypatch):
    model = nn.Module()
    model.lin = nn.Linear(32, 32, bias = False).to(torch.bfloat16).requires_grad_(False)
    monkeypatch.setattr(
        dit, "_int8_gemm_decline", lambda dev: "int8 W8A8 needs int8 tensor cores, which X lacks"
    )
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None, device = "cuda:0") is False
    assert "tensor cores" in events[-1]["message"] and "weight-only" in events[-1]["message"]
    assert type(model.lin) is nn.Linear


def test_conversion_keeps_the_fp32_island_on_an_untargeted_layer():
    """fp16 path: islands are installed before the W8A8 conversion and live on the module's forward. A converted
    Z-Image feed_forward.w2 (not a LoRA target) must still run fp32 with autocast off, not fp16 / bf16."""
    from core.training.diffusion_dit_fp16 import install_fp32_islands

    class _FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.w2 = nn.Linear(64, 32)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.feed_forward = _FF()
            self.attention = _Attn()

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    torch.manual_seed(0)
    model = _M().to(torch.bfloat16).requires_grad_(False)
    islands = install_fp32_islands(model, "z-image", "cpu")
    assert islands == ["layers.0.feed_forward.w2"]
    done, failed = fi8.convert_frozen_int8(
        model, {"layers.0.feed_forward.w2", "layers.0.attention.to_q"}
    )
    assert (done, failed) == (2, 0)
    w2, q = model.layers[0].feed_forward.w2, model.layers[0].attention.to_q
    assert isinstance(w2, fi8.FrozenInt8Linear) and isinstance(q, fi8.FrozenInt8Linear)
    assert getattr(w2, "_unsloth_fp32_island", False)
    assert not getattr(q, "_unsloth_fp32_island", False)
    x64 = torch.randn(24, 64, dtype = torch.bfloat16, requires_grad = True)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        y = w2(x64)
        yq = q(torch.randn(24, 32, dtype = torch.bfloat16))
    assert y.dtype == torch.float32 and yq.dtype == torch.bfloat16
    y.sum().backward()
    assert x64.grad is not None and torch.isfinite(x64.grad).all()


@pytest.mark.parametrize("fp16, refused", [(True, False), (False, True)])
def test_preflight_needs_torchao_for_int8_only_where_the_trainer_does(monkeypatch, fp16, refused):
    """The fp16 path's W8A8 base runs torch._int_mm and raises instead of falling back to torchao, so a missing
    torchao must not refuse it; the bf16 path keeps torchao weight-only int8 as its fallback and still needs it."""
    import core.training.diffusion_train_common as tc

    monkeypatch.setattr(tc, "bf16_unsupported_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_accelerator_missing_reason", lambda fam: None)
    monkeypatch.setattr(tc, "dit_fp16_fallback", lambda fam: fp16)
    monkeypatch.setattr(tc, "has_functional_torchao", lambda: False)
    monkeypatch.setattr(tc, "torch_is_rocm", lambda: False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_TRAIN", "w8a8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    reason = tc.training_precision_preflight_error("flux.2-klein", "int8")
    assert ("torchao" in (reason or "")) is refused
    if not refused:
        assert reason is None


def test_a_partial_w8a8_conversion_does_not_fall_back_to_a_mixed_base(monkeypatch):
    """A device move that fails after some layers converted must not hand the half-W8A8 base to the torchao
    weight-only fallback (the trainer would compile, cache and record it as weight-only int8)."""

    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32, bias = False)
            self.to_k = nn.Linear(32, 32, bias = False)
            self.to(torch.bfloat16).requires_grad_(False)

    model = nn.Module()
    model.blocks = nn.ModuleList([_B()])
    moves = []
    real_to = nn.Module.to

    def to(self, *a, **k):
        if type(self) is nn.Linear:
            moves.append(1)
            if len(moves) == 2:
                raise torch.OutOfMemoryError("injected OOM on the second move")
        return real_to(self, *a, **k)

    monkeypatch.setattr(nn.Module, "to", to)
    events = []
    with pytest.raises(ValueError, match = "partway|part of the base"):
        dit._apply_int8_w8a8(model, events.append, None, device = "meta")
    kinds = {type(m) for m in model.blocks[0].children()}
    assert kinds == {
        fi8.FrozenInt8Linear,
        nn.Linear,
    }  # the mixed state the fallback would have trained


def test_a_w8a8_failure_before_any_layer_still_falls_back(monkeypatch):
    model = nn.Module()
    model.lin = nn.Linear(32, 32, bias = False).to(torch.bfloat16).requires_grad_(False)

    def boom(*a, **k):
        raise RuntimeError("no int8 GEMM")

    monkeypatch.setattr(fi8, "convert_frozen_int8", boom)
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None, device = "meta") is False
    assert "weight-only" in events[-1]["message"]


def test_a_resume_cannot_switch_between_w8a8_and_weight_only_int8():
    """Both int8 bases record base_precision_effective='int8'; flipping UNSLOTH_DIFFUSION_INT8_TRAIN across a resume
    would continue the adapter and optimizer on different base math. Older bundles (no record) stay resumable."""
    import dataclasses

    from core.training import diffusion_checkpoint as ck

    base = ck.CheckpointIdentity(
        family = "flux.2-klein",
        base_model = "m",
        lora_target_modules = ("to_q",),
        lora_rank = 4,
        lora_alpha = 4,
        precision = "bf16",
        base_precision = "int8",
        resolution = 512,
        base_precision_effective = "int8",
    )
    record = getattr(ck, "with_int8_kernel", None)
    assert record is not None
    w8a8 = ck.CheckpointIdentity.from_dict(record(base, True).as_dict())
    weight_only = record(base, False)
    assert "int8 base kernel" in (w8a8.mismatch_reason(weight_only) or "")
    assert weight_only.mismatch_reason(record(base, False)) is None
    legacy = dataclasses.replace(base, base_int8_kernel = None)
    assert legacy.mismatch_reason(w8a8) is None and legacy.mismatch_reason(weight_only) is None


def _two_linears():
    class _B(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(32, 32, bias = False)
            self.to_k = nn.Linear(32, 32, bias = False)
            self.to(torch.bfloat16).requires_grad_(False)

    model = nn.Module()
    model.blocks = nn.ModuleList([_B()])
    return model


def _fail_second_quantize(monkeypatch, module, attr, exc):
    real = getattr(module, attr)
    calls = []

    def quantize(w):
        calls.append(1)
        if len(calls) == 2:
            raise exc
        return real(w)

    monkeypatch.setattr(module, attr, quantize)
    return calls


def test_an_oom_inside_w8a8_quantization_is_not_a_kept_layer(monkeypatch):
    """An OOM inside FrozenInt8Linear (not the device move) used to be counted as a kept layer, so a partly W8A8
    base trained and recorded the same base_int8_kernel='w8a8' identity as a full one."""
    model = _two_linears()
    _fail_second_quantize(
        monkeypatch, fi8, "quantize_weight", torch.cuda.OutOfMemoryError("CUDA out of memory")
    )
    with pytest.raises(ValueError, match = "part of the base"):
        dit._apply_int8_w8a8(model, [].append, None)
    assert {type(m) for m in model.blocks[0].children()} == {fi8.FrozenInt8Linear, nn.Linear}


def test_a_non_oom_w8a8_layer_failure_is_still_kept(monkeypatch):
    model = _two_linears()
    _fail_second_quantize(monkeypatch, fi8, "quantize_weight", RuntimeError("unsupported layer"))
    events = []
    assert dit._apply_int8_w8a8(model, events.append, None) is True
    assert "(1 kept)" in events[-1]["message"]


def test_an_oom_inside_fp8_quantization_is_not_recorded_as_fp8_or_bf16(monkeypatch):
    """fp8: an OOM inside FrozenFp8Linear was a silent per-layer skip (a partly fp8 base recorded as fp8), and one
    escaping between layers left the converted layers in a base the caller records as bf16. Both now refuse."""
    from core.training import diffusion_frozen_fp8 as f8

    model = _two_linears()
    _fail_second_quantize(
        monkeypatch, f8, "_quantize_weight", torch.cuda.OutOfMemoryError("CUDA out of memory")
    )
    with pytest.raises(ValueError, match = "part of the base"):
        dit._apply_fp8_training(model, [].append, None)
    assert {type(m) for m in model.blocks[0].children()} == {f8.FrozenFp8Linear, nn.Linear}

    model = _two_linears()
    _fail_second_quantize(monkeypatch, f8, "_quantize_weight", RuntimeError("unsupported layer"))
    events = []
    assert dit._apply_fp8_training(model, events.append, None) is True
    assert any("1 layer(s) could not be converted" in e["message"] for e in events)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_declares_the_dense_compute_dtype_not_its_int8_storage(dtype):
    # Callers that cast a projection's input to the dtype the layer computes in read ``compute_dtype``; the int8
    # weight buffer is storage, and reading it truncated MOSS-Image's activations to integers.
    m = fi8.FrozenInt8Linear(nn.Linear(64, 32, bias = False).to(dtype))
    assert m.weight.dtype == torch.int8
    assert m.compute_dtype == dtype


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_moss_casts_a_frozen_int8_projection_input_to_the_dense_dtype(dtype):
    pytest.importorskip("diffusers")
    from core.inference.diffusion_moss_image import _compute_dtype

    m = fi8.FrozenInt8Linear(nn.Linear(64, 32, bias = False).to(dtype))
    assert _compute_dtype(m) == dtype
