# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the DiT fp16 training path on non-bf16 GPUs (T4)."""

from __future__ import annotations

import json
import random

import pytest
import torch

import core.inference.rocm_bf16 as rocm_bf16
import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_train_common as dtc
from core.training.diffusion_train_common import DiffusionLoraConfig

FP16_FAMILIES = ("flux.2-klein", "z-image", "wan2.2-ti2v-5b")
BASES = {
    "flux.2-klein": "black-forest-labs/FLUX.2-klein-base-4B",
    "z-image": "Tongyi-MAI/Z-Image-Turbo",
    "wan2.2-ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "qwen-image": "Qwen/Qwen-Image",
}


@pytest.fixture
def card(monkeypatch):
    """Pretend an NVIDIA CUDA card at a capability; bf16 'supported' as torch reports it (emulation counts)."""
    monkeypatch.delenv(getattr(dtc, "DIT_FP16_ENV", "UNSLOTH_STUDIO_DIT_FP16"), raising = False)
    for mod in (dtc, dit):
        monkeypatch.setattr(mod, "torch_is_rocm", lambda: False)
    # native_bf16_supported reads the torch build too: on a ROCm host it would answer for the AMD card.
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda *a, **k: False)

    def set_cap(cap):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: cap)

    return set_cap


def _cfg(family, tmp_path, **kw):
    return DiffusionLoraConfig(
        base_model = BASES[family],
        model_family = family,
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "o"),
        **kw,
    )


def _route(monkeypatch, tmp_path, family):
    """Run run_dit_lora_training up to _train_dit and capture what it would train with."""
    (tmp_path / "a.png").write_bytes(b"")
    seen = {}

    def fake_train(cfg, spec, pairs, rng, device, weight_dtype, *rest, **kw):
        seen.update(
            device = device, weight_dtype = weight_dtype, fp16_amp = kw.get("fp16_amp", False), kw = kw
        )
        return "ok"

    monkeypatch.setattr(dit, "_train_dit", fake_train)
    monkeypatch.setattr(dit, "resolve_train_device", lambda: "cuda")
    monkeypatch.setattr(dit, "_assert_trusted_base_model", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_assert_gated_access", lambda *a, **k: None)
    monkeypatch.setattr(
        dit, "discover_image_caption_pairs", lambda *a, **k: [(str(tmp_path / "a.png"), "c")]
    )
    monkeypatch.setattr(dit, "resolve_train_steps", lambda cfg, *a, **k: 3)
    # The real checkpoint identity: the trainer rewrites it with dataclasses.replace (world size, cache mode), so a None
    # stand-in raises before the dtype route this file checks.
    monkeypatch.setattr(dit, "_apply_perf_flags", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_restore_perf_flags", lambda *a, **k: None)
    monkeypatch.setattr(dit, "training_fetch_repo", lambda *a, **k: "x")
    out = dit.run_dit_lora_training(_cfg(family, tmp_path))
    assert out == "ok"
    return seen


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
@pytest.mark.parametrize("family", FP16_FAMILIES)
def test_bf16_card_takes_the_bf16_path(card, monkeypatch, tmp_path, cap, family):
    card(cap)
    seen = _route(monkeypatch, tmp_path, family)
    assert seen["weight_dtype"] is torch.bfloat16
    # The old call carried no fp16 flag at all; the new one passes False, never True, on a bf16 card.
    assert seen["fp16_amp"] is False
    assert dtc.bf16_unsupported_reason(family) is None
    monkeypatch.setattr(dtc, "resolve_train_device", lambda: "cuda")
    assert dtc.effective_mixed_precision(_cfg(family, tmp_path).normalized()) == "bf16"


@pytest.mark.parametrize("family", FP16_FAMILIES)
def test_t4_takes_the_fp16_path(card, monkeypatch, tmp_path, family):
    card((7, 5))
    # Fails on main: the T4 was refused with "requires a bfloat16-capable GPU".
    seen = _route(monkeypatch, tmp_path, family)
    assert seen["fp16_amp"] is True
    # fp32 data / non-quantized weights, fp16 only inside autocast.
    assert seen["weight_dtype"] is torch.float32
    assert dtc.bf16_unsupported_reason(family) is None
    assert dtc.training_precision_preflight_error(family, "nf4") is None


@pytest.mark.parametrize("family", FP16_FAMILIES)
def test_t4_fp16_resume_identity_records_fp16(card, monkeypatch, tmp_path, family):
    card((7, 5))
    monkeypatch.setattr(dtc, "resolve_train_device", lambda: "cuda")
    cfg = _cfg(family, tmp_path).normalized()
    assert dtc.effective_mixed_precision(cfg) == "fp16"


def test_t4_still_refuses_families_without_an_fp16_path(card, monkeypatch, tmp_path):
    card((7, 5))
    assert "bfloat16" in (dtc.bf16_unsupported_reason("qwen-image") or "")
    with pytest.raises(ValueError, match = "bfloat16"):
        _route(monkeypatch, tmp_path, "qwen-image")


@pytest.mark.parametrize("mode", ["bf16", "int8", "fp8", "mxfp8"])
def test_t4_refuses_dense_bases_before_eviction(card, mode):
    card((7, 5))
    reason = dtc.training_precision_preflight_error("z-image", mode)
    assert reason and "nf4" in reason


def test_t4_dense_base_refused_in_the_trainer_too(card, monkeypatch, tmp_path):
    card((7, 5))
    monkeypatch.setattr(dit, "_train_dit", lambda *a, **k: "ok")
    monkeypatch.setattr(dit, "resolve_train_device", lambda: "cuda")
    (tmp_path / "a.png").write_bytes(b"")
    cfg = _cfg("z-image", tmp_path, base_precision = "bf16")
    with pytest.raises(ValueError, match = "nf4"):
        dit.run_dit_lora_training(cfg)


def test_kill_switch_restores_the_t4_refusal(card, monkeypatch, tmp_path):
    card((7, 5))
    monkeypatch.setenv(dtc.DIT_FP16_ENV, "0")
    assert "bfloat16" in (dtc.bf16_unsupported_reason("z-image") or "")
    with pytest.raises(ValueError, match = "bfloat16"):
        _route(monkeypatch, tmp_path, "z-image")


def test_t4_family_infos_offer_nf4_only(card, monkeypatch):
    card((7, 5))
    monkeypatch.setattr(dtc, "dit_accelerator_missing_reason", lambda *a, **k: None)
    infos = {i["name"]: i for i in dtc.family_train_infos()}
    for fam in FP16_FAMILIES:
        if fam in infos:
            assert infos[fam]["precision_modes"] == ["nf4"], fam
    if "qwen-image" in infos:
        assert infos["qwen-image"]["precision_modes"] == []


def test_t4_family_infos_hide_compile_the_fp16_path_never_runs(card, monkeypatch):
    # The fp16 trainer always runs eager, so the panel must not offer a compile switch it ignores.
    monkeypatch.setattr(dtc, "dit_accelerator_missing_reason", lambda *a, **k: None)
    monkeypatch.setattr(dtc, "has_functional_torchao", lambda: False)
    card((7, 5))
    t4 = {i["name"]: i for i in dtc.family_train_infos()}
    card((8, 9))
    l4 = {i["name"]: i for i in dtc.family_train_infos()}
    present = [fam for fam in FP16_FAMILIES if fam in t4]
    assert present, "no fp16 family is listed, so this test would prove nothing"
    for fam in present:
        assert t4[fam]["precision_modes"] == ["nf4"], fam
        assert t4[fam]["supports_compile"] is False, fam
        # bf16 cards keep the compile switch.
        assert l4[fam]["supports_compile"] is True, fam


def _toy(lr = 0.1):
    w = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.SGD([w], lr = lr)
    from torch.optim.lr_scheduler import LambdaLR

    sched = LambdaLR(opt, lambda s: (s + 1) / 10)
    return w, opt, sched


def test_bf16_update_never_touches_the_scaler():
    w, opt, sched = _toy()

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"bf16 path touched scaler.{name}")

    w.grad = torch.ones(4)
    assert dit._optimizer_update(opt, sched, Boom(), False) is True
    assert sched.last_epoch == 1 and torch.all(w < 1)


def test_inf_grad_skips_update_and_schedule_then_recovers():
    w, opt, sched = _toy()
    scaler = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    before = w.detach().clone()
    lr0 = sched.get_last_lr()[0]
    scaler.scale((w * torch.tensor([1.0, float("inf"), 1.0, 1.0])).sum()).backward()
    scaler.unscale_(opt)
    assert dit._optimizer_update(opt, sched, scaler, True) is False
    assert torch.equal(w.detach(), before), "a skipped step must not move the weights"
    assert (
        sched.last_epoch == 0 and sched.get_last_lr()[0] == lr0
    ), "the LR schedule must not tick on a skip"
    assert scaler.get_scale() == 2.0**15
    opt.zero_grad(set_to_none = True)
    scaler.scale(w.sum()).backward()
    scaler.unscale_(opt)
    assert dit._optimizer_update(opt, sched, scaler, True) is True
    assert sched.last_epoch == 1 and not torch.equal(w.detach(), before)
    assert torch.allclose(w.detach(), before - lr0 * 1.0)


def test_scaled_grad_accumulation_matches_unscaled():
    torch.manual_seed(0)
    x = [torch.randn(3, 4) for _ in range(3)]
    ref = torch.nn.Linear(4, 2)
    got = torch.nn.Linear(4, 2)
    got.load_state_dict(ref.state_dict())
    opt = torch.optim.SGD(got.parameters(), lr = 0.0)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    for xi in x:
        (ref(xi).pow(2).mean() / 3).backward()
        scaler.scale(got(xi).pow(2).mean() / 3).backward()
    scaler.unscale_(opt)
    for a, b in zip(ref.parameters(), got.parameters()):
        assert torch.allclose(a.grad, b.grad, rtol = 1e-5, atol = 1e-7)


def test_scaler_state_round_trips_through_the_checkpoint_progress():
    scaler = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    w, opt, sched = _toy()
    for _ in range(3):
        opt.zero_grad(set_to_none = True)
        scaler.scale((w * float("inf")).sum()).backward()
        scaler.unscale_(opt)
        dit._optimizer_update(opt, sched, scaler, True)
    base = {"running_loss": 1.5, "resolved_base_precision": "nf4"}
    assert (
        dit._resume_progress(base, scaler, 3, False) is base
    ), "bf16 checkpoints keep the old progress block"
    prog = json.loads(json.dumps(dit._resume_progress(base, scaler, 3, True)))
    fresh = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    assert dit._restore_scaler(prog, fresh) == 3
    assert fresh.get_scale() == scaler.get_scale() == 2.0**13
    # A checkpoint from before the fp16 path (no scaler block) leaves the default scale.
    other = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    assert dit._restore_scaler(base, other) == 0 and other.get_scale() == 2.0**16


class Linear4bit(torch.nn.Linear):
    """Stands in for bitsandbytes' Linear4bit (matched by class name): carries compute_dtype."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.compute_dtype = torch.bfloat16


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.proj = Linear4bit(8, 8)
        self.seen = []

    def forward(self, x):
        self.seen.append((x.dtype, torch.is_autocast_enabled("cpu")))
        return x + self.proj(self.norm(x))


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = Linear4bit(8, 8)
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.proj_out = torch.nn.Linear(8, 8)

    def forward(self, x):
        x = x + self.time_embed(x)
        for b in self.blocks:
            x = b(x)
        return self.proj_out(x)


def test_prepare_fp16_base_sets_compute_dtype_upcasts_and_islands(monkeypatch):
    from core.training import diffusion_dit_fp16 as f16

    monkeypatch.setitem(f16.FP16_FP32_MODULES, "toy", (r"^blocks\.1$", r"^proj_out$"))
    m = _Toy().to(torch.bfloat16)
    rep = f16.prepare_fp16_base(m)
    assert rep["linear4bit_fp16"] == 3
    assert f16.install_fp32_islands(m, "toy", "cpu") == ["blocks.1", "proj_out"]
    assert (
        m.time_embed.compute_dtype is torch.float16
        and m.blocks[0].proj.compute_dtype is torch.float16
    )
    assert m.blocks[1].proj.compute_dtype is torch.float32
    assert all(p.dtype is torch.float32 for p in m.parameters())
    with torch.autocast("cpu", dtype = torch.bfloat16):
        m(torch.randn(2, 8))
    # Block 0 runs under autocast, block 1 (the island) with autocast off and fp32 input.
    assert m.blocks[0].seen[-1] == (torch.float32, True)
    assert m.blocks[1].seen[-1] == (torch.float32, False)


def test_islands_recompute_identically_under_gradient_checkpointing(monkeypatch):
    from torch.utils.checkpoint import checkpoint

    from core.training import diffusion_dit_fp16 as f16

    monkeypatch.setitem(f16.FP16_FP32_MODULES, "toy", (r"^blocks\.1$",))
    torch.manual_seed(0)
    a = _Toy()
    b = _Toy()
    b.load_state_dict(a.state_dict())
    for m in (a, b):
        f16.prepare_fp16_base(m)
        f16.install_fp32_islands(m, "toy", "cpu")
    x = torch.randn(2, 8)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        ya = a(x)
        h = b.time_embed(x) + x
        for blk in b.blocks:
            h = checkpoint(blk, h, use_reentrant = False)
        yb = b.proj_out(h)
    ya.float().sum().backward()
    yb.float().sum().backward()
    assert torch.equal(ya, yb)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa.grad, pb.grad)
    # The recompute in backward re-entered the island with autocast still off.
    assert b.blocks[1].seen[-1] == (torch.float32, False) and len(b.blocks[1].seen) == 2


def test_island_wraps_the_peft_layer_not_just_its_base(monkeypatch):
    """Installed after the adapter, the island covers the LoRA branch too (it overflowed in fp16 on Z-Image)."""
    from core.training import diffusion_dit_fp16 as f16

    class LoraWrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = Linear4bit(8, 8)
            self.lora_A = torch.nn.Linear(8, 2, bias = False)
            self.lora_B = torch.nn.Linear(2, 8, bias = False)
            self.seen = None

        def forward(self, x):
            self.seen = torch.is_autocast_enabled("cpu")
            return self.base_layer(x) + self.lora_B(self.lora_A(x))

    m = torch.nn.Module()
    m.ff = torch.nn.Module()
    m.ff.w2 = LoraWrapped()
    monkeypatch.setitem(f16.FP16_FP32_MODULES, "toy", (r"\.w2$",))
    assert f16.install_fp32_islands(m, "toy", "cpu") == ["ff.w2"]
    assert m.ff.w2.base_layer.compute_dtype is torch.float32
    with torch.autocast("cpu", dtype = torch.bfloat16):
        y = m.ff.w2(torch.randn(2, 8))
    assert m.ff.w2.seen is False and y.dtype is torch.float32


def test_zimage_islands_cover_every_block_output_projection():
    from core.training import diffusion_dit_fp16 as f16

    names = [
        "noise_refiner.0.attention.to_out.0",
        "noise_refiner.0.attention.to_out.0.base_layer",
        "context_refiner.1.feed_forward.w2",
        "layers.3.feed_forward.w2",
        "layers.3.attention.to_out.0",
        "t_embedder",
    ]

    class Named(torch.nn.Module):
        def named_modules(self, *a, **k):
            return [(n, None) for n in names]

    got = f16.island_names(Named(), "z-image")
    assert got == [
        "noise_refiner.0.attention.to_out.0",
        "context_refiner.1.feed_forward.w2",
        "layers.3.feed_forward.w2",
        "layers.3.attention.to_out.0",
        "t_embedder",
    ]


def test_overflow_probe_names_the_overflowing_module():
    from core.training import diffusion_dit_fp16 as f16

    class Big(torch.nn.Module):
        def forward(self, x):
            return (x.half() * 6.0e4) * 4.0  # past fp16 max from a finite input

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), Big(), torch.nn.Identity())
    rec = {}
    with f16.overflow_probe(m, rec):
        m(torch.ones(1, 4))
    assert rec["first_overflow"] == "1"
    assert rec["overflows"] == {
        "1": 1
    }, "the Identity after it saw a non-finite INPUT, so it is not blamed"
    assert rec["max_abs"]["0"] > 0
    rec2 = {}
    m(torch.ones(1, 4))
    assert rec2 == {}


def test_every_fp16_family_has_an_island_table_and_a_spec():
    from core.training import diffusion_dit_fp16 as f16
    for fam in dtc.DIT_FP16_TRAIN_FAMILIES:
        assert fam in dit._SPECS
        assert f16.FP16_FP32_MODULES.get(fam), fam


class Params4bit(torch.nn.Parameter):
    """Stands in for bitsandbytes' Params4bit (matched by class name and quant_state)."""

    def __new__(cls, data):
        p = super().__new__(cls, data, requires_grad = False)
        p.quant_state = type("QS", (), {"shape": data.shape})()
        return p


class _Nf4Linear(Linear4bit):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.weight = Params4bit(self.weight.data)
        self.calls = []

    def forward(self, x):
        self.calls.append("orig")
        return torch.nn.functional.linear(
            x.to(self.compute_dtype), self.weight.to(self.compute_dtype)
        ).to(x.dtype)


# Matched by class name, like bitsandbytes' own.
_Nf4Linear.__name__ = "Linear4bit"


class _FakeDequant:
    """Records the dtype each nf4 linear asks the dequant cache for."""

    def __init__(self):
        self.layers = 0
        self.asked = {}

    def get(self, base, dtype):
        self.asked[id(base)] = dtype
        return base.weight.detach().to(dtype)


class _FastToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.proj_out = torch.nn.Linear(8, 8)
        self.embed = torch.nn.Module()
        self.embed.fc = _Nf4Linear(8, 8, bias = False)
        self.head = _Nf4Linear(8, 8, bias = False)

    def forward(self, x):
        x = self.lin(x + self.embed(x))
        return self.head(self.proj_out(x))


def _fast_toy(monkeypatch):
    from peft import LoraConfig, inject_adapter_in_model

    from core.training import diffusion_dit_fp16 as f16

    torch.manual_seed(0)
    m = _FastToy()
    m.embed.forward = lambda x: m.embed.fc(x)
    m = inject_adapter_in_model(
        LoraConfig(
            r = 2, lora_alpha = 2, target_modules = ["lin", "proj_out"], init_lora_weights = "gaussian"
        ),
        m,
    )
    for name in ("lin", "proj_out"):
        torch.nn.init.normal_(getattr(m, name).lora_B["default"].weight)
    f16.prepare_fp16_base(m)
    monkeypatch.setitem(f16.FP16_FP32_MODULES, "toy", (r"^proj_out$", r"^embed$", r"^head$"))
    assert f16.install_fp32_islands(m, "toy", "cpu") == ["proj_out", "embed", "head"]
    return m


def test_fast_lora_leaves_fp32_islands_in_fp32(monkeypatch):
    """Fast LoRA and the dequant cache install after the islands: a patched island root would run under fp16 autocast."""
    from core.training import diffusion_lora_fast as fl

    m = _fast_toy(monkeypatch)
    seen = {}
    for name in ("lin", "proj_out"):
        mod = getattr(m, name)
        mod.lora_A["default"].register_forward_pre_hook(
            lambda _m, args, name = name: seen.__setitem__(
                name, (args[0].dtype, torch.is_autocast_enabled("cpu"))
            )
        )
    cache = _FakeDequant()
    calls = fl.STATS["calls"]
    # proj_out is an island and a LoRA layer: skipped; lin: patched.
    assert fl.install_fast_lora(m, cache) == 1
    assert (
        "_unsloth_orig_forward" not in m.proj_out.__dict__
        and "_unsloth_orig_forward" in m.lin.__dict__
    )
    # head is an island and a bare nf4 linear: skipped; embed.fc sits inside an island: cached.
    assert "_unsloth_orig_forward" not in m.head.__dict__ and cache.layers == 1
    out = []
    m.proj_out.register_forward_hook(lambda _m, _a, o: out.append(o.dtype))
    with torch.autocast("cpu", dtype = torch.float16):
        y = m(torch.randn(3, 8))
    # The patched layer took the fast path in fp16; the island's LoRA branch saw fp32 with autocast off.
    assert fl.STATS["calls"] == calls + 1
    assert seen["proj_out"] == (torch.float32, False) and out == [torch.float32]
    # The island's nf4 linear dequantises in fp32 (its compute dtype), the island root runs its own forward.
    assert cache.asked == {id(m.embed.fc): torch.float32} and m.head.calls == ["orig"]
    assert y.dtype is torch.float32 and torch.isfinite(y).all()
    y.sum().backward()
    assert all(
        p.grad is not None and p.grad.dtype is torch.float32
        for p in m.parameters()
        if p.requires_grad
    )
    fl.uninstall_fast_lora(m)
    assert m.proj_out._unsloth_fp32_island and "forward" in m.proj_out.__dict__
    assert "forward" not in m.lin.__dict__


def test_fast_lora_matches_peft_under_fp16_autocast(monkeypatch):
    """Off the islands, the cast-once path gives PEFT's fp16 forward and fp32 master grads."""
    from core.training import diffusion_lora_fast as fl

    ref = _fast_toy(monkeypatch)
    fast = _fast_toy(monkeypatch)
    fast.load_state_dict(ref.state_dict())
    fl.install_fast_lora(fast)
    x = torch.randn(3, 8)
    ys = []
    for m in (ref, fast):
        with torch.autocast("cpu", dtype = torch.float16):
            y = m(x)
        (y.float().square().mean() * 1024.0).backward()
        ys.append(y)
    assert torch.equal(ys[0], ys[1])
    for (n, a), b in zip(ref.named_parameters(), fast.parameters()):
        if a.requires_grad:
            assert b.grad.dtype is torch.float32
            torch.testing.assert_close(a.grad, b.grad, rtol = 1e-3, atol = 1e-5, msg = n)


def test_dequant_cache_stays_off_on_the_fp16_path(monkeypatch):
    """bf16 nf4 runs keep main's dequant cache; the fp16 path leaves the card's room to the checkpointing budget."""
    from core.training import diffusion_lora_fast as fl

    monkeypatch.setattr(dit, "torch_is_rocm", lambda: False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", raising = False)
    assert dit._dequant_cache_wanted(True, "cuda", False)
    assert not dit._dequant_cache_wanted(True, "cuda", True)
    assert not dit._dequant_cache_wanted(False, "cuda", False)
    assert not dit._dequant_cache_wanted(True, "xpu", False)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", "0")
    assert not fl.dequant_cache_enabled() and not dit._dequant_cache_wanted(True, "cuda", False)
    # ROCm keeps the cache too (gfx1151 nf4 klein LoRA 1.68x Linux / 1.63x Windows); the fp16 path still declines it
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: True)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE")
    assert dit._dequant_cache_wanted(True, "cuda", False)
    assert not dit._dequant_cache_wanted(True, "cuda", True)


@pytest.mark.parametrize(
    "mode, warned", [("on", True), ("ON ", True), ("auto", False), ("off", False), (None, False)]
)
def test_fp16_path_reports_a_declined_compile_request(mode, warned, caplog):
    # Old clients may still send compile_transformer='on'; the fp16 path runs eager, so it must say so, not raise.
    events = []
    cfg = type("Cfg", (), {"compile_transformer": mode})()
    with caplog.at_level("WARNING", logger = dit.__name__):
        assert dit._warn_fp16_compile_declined(cfg, events.append) is warned
    msgs = [e["message"] for e in events if e["type"] == "warning"]
    logged = [r.getMessage() for r in caplog.records if r.name == dit.__name__]
    if warned:
        assert len(msgs) == 1 and "compile_transformer='on' is ignored" in msgs[0]
        assert logged == msgs
    else:
        assert msgs == [] and logged == []


def test_declined_compile_warning_runs_only_on_the_fp16_path():
    # bf16 cards never reach it: the call sits under `if fp16_amp:` right before the compile decision.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(dit._train_dit).lstrip())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "fp16_amp":
            hits += [
                ast.unparse(n)
                for n in node.body
                if isinstance(n, ast.Expr) and "_warn_fp16_compile_declined" in ast.unparse(n)
            ]
    assert hits == ["_warn_fp16_compile_declined(cfg, on_event)"]
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_warn_fp16_compile_declined"
    ]
    assert len(calls) == 1
    src = inspect.getsource(dit._train_dit)
    assert src.index("_warn_fp16_compile_declined(cfg, on_event)") < src.index(
        "_maybe_compile_transformer("
    )


# ---- Z-Image to_out / w2: fp16 operands, fp32 accumulate and output, power-of-two scaled inputs


class _LoraW(torch.nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.base_layer = _Nf4Linear(k, n, bias = False)
        self.lora_A = torch.nn.Linear(k, 2, bias = False)
        self.lora_B = torch.nn.Linear(2, n, bias = False)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


def _zimage_toy():
    torch.manual_seed(0)
    blk = torch.nn.Module()
    blk.attention = torch.nn.Module()
    blk.attention.to_out = torch.nn.ModuleList([_LoraW(32, 16)])
    blk.feed_forward = torch.nn.Module()
    blk.feed_forward.w2 = _LoraW(64, 16)
    blk.feed_forward.w1 = _LoraW(16, 64)
    m = torch.nn.Module()
    m.layers = torch.nn.ModuleList([blk])
    for lw in (blk.attention.to_out[0], blk.feed_forward.w2, blk.feed_forward.w1):
        # nf4 dequant gives bf16 values: keep the fake weight on that grid.
        lw.base_layer.weight.data = lw.base_layer.weight.data.to(torch.bfloat16).float()
    return m


@pytest.fixture
def scaled_mm(monkeypatch):
    """CPU stand-ins: exact fp16-operand / fp32-accumulate GEMM, the fake nf4 weight as its own dequant."""
    from core.training import diffusion_dit_fp16 as f16

    calls = []

    def mm(a, b):
        assert a.dtype is torch.float16 and b.dtype is torch.float16
        calls.append(a.shape)
        return a.float() @ b.float()

    monkeypatch.setattr(f16, "scaled_mm_supported", lambda device_type: True)
    monkeypatch.setattr(f16, "_mm_fp32_out", mm)
    monkeypatch.setattr(
        f16, "_dequant_4bit", lambda layer: layer.weight.detach().to(torch.bfloat16)
    )
    return calls


def _ref(lw, x):
    return torch.nn.functional.linear(x, lw.base_layer.weight.float()) + lw.lora_B(lw.lora_A(x))


def test_zimage_scaled_mm_installs_on_to_out_and_w2_only(scaled_mm):
    from core.training import diffusion_dit_fp16 as f16

    m = _zimage_toy()
    rep = {}
    names = f16.install_fp32_islands(m, "z-image", "cpu", report = rep)
    assert names == ["layers.0.attention.to_out.0", "layers.0.feed_forward.w2"]
    assert rep["fp16_scaled_mm"] == 2
    blk = m.layers[0]
    assert "_unsloth_scaled_mm_orig" in blk.feed_forward.w2.base_layer.__dict__
    assert "_unsloth_scaled_mm_orig" in blk.attention.to_out[0].base_layer.__dict__
    assert "_unsloth_scaled_mm_orig" not in blk.feed_forward.w1.base_layer.__dict__
    # klein / Wan islands keep the fp32 path.
    rep = {}
    f16.install_fp32_islands(_zimage_toy(), "flux.2-klein", "cpu", report = rep)
    assert rep["fp16_scaled_mm"] == 0


def test_zimage_scaled_mm_stays_finite_past_fp16_range_and_matches_fp32(scaled_mm):
    """w2's input reaches 8.7e5 on a T4: a plain fp16 cast is inf, the scaled fp16 GEMM matches the fp32 island."""
    from core.training import diffusion_dit_fp16 as f16

    m = _zimage_toy()
    f16.install_fp32_islands(m, "z-image", "cpu")
    w2 = m.layers[0].feed_forward.w2
    x = torch.randn(8, 64) * 1.0e4
    x[3, 5] = 8.7e5
    x.requires_grad_(True)
    assert not torch.isfinite(x.half()).all()
    with torch.autocast("cpu", dtype = torch.float16):
        y = w2(x)
    assert y.dtype is torch.float32 and torch.isfinite(y).all()
    assert w2.base_layer.calls == [], "the fp32 nf4 forward did not run"
    xr = x.detach().clone().requires_grad_(True)
    ref = _ref(w2, xr)
    assert ((y - ref).norm() / ref.norm()) < 2e-3
    # Grads scaled like GradScaler's 65536 go past fp16 range too.
    g = torch.randn_like(y) * 65536.0 * 4
    y.backward(g)
    ref.backward(g)
    assert torch.isfinite(x.grad).all()
    assert ((x.grad - xr.grad).norm() / xr.grad.norm()) < 2e-3
    assert w2.lora_A.weight.grad is not None and w2.lora_A.weight.grad.dtype is torch.float32
    assert len(scaled_mm) == 2  # forward + dx


def test_scaled_mm_forward_split_keeps_fp32_input_precision(scaled_mm):
    """One fp16 rounding of the input costs ~1e-4; the hi + lo split must land at fp32 level, the dx term at fp16."""
    from core.training import diffusion_dit_fp16 as f16

    torch.manual_seed(1)
    x = torch.randn(16, 64, dtype = torch.float64) * torch.logspace(-3, 3, 64, dtype = torch.float64)
    w = torch.randn(32, 64).to(torch.bfloat16).double()
    ref = x @ w.t()
    w16t = w.t().to(torch.float16)
    one = f16._scaled_fp32_mm(x.float(), w16t)
    two = f16._scaled_fp32_mm(x.float(), w16t, split = True)
    err = lambda y: float((y.double() - ref).norm() / ref.norm())  # noqa: E731
    assert err(two) < 1e-6 < 1e-5 < err(one)
    assert two.dtype is torch.float32 and two.shape == (16, 32)


def test_pow2_scale_is_exact_and_lands_under_fp16_max():
    from core.training import diffusion_dit_fp16 as f16

    for amax in (3.0e-6, 0.7, 1.0, 65504.0, 8.7e5, 3.0e9):
        t = torch.tensor([amax, -amax / 3])
        s = f16._pow2_scale(t)
        assert 2.0**14 < float((t * s).abs().max()) <= 2.0**15
        assert float(torch.log2(s)).is_integer()
    assert float(f16._pow2_scale(torch.zeros(4))) > 0
    for bad in (float("inf"), float("nan")):
        assert not torch.isfinite(
            torch.tensor([1.0, bad]) * f16._pow2_scale(torch.tensor([1.0, bad]))
        ).all()


def test_scaled_mm_keeps_non_finite_inputs_non_finite(scaled_mm):
    """GradScaler must still see an overflow upstream: inf in, non-finite out, never silently finite."""
    from core.training import diffusion_dit_fp16 as f16

    m = _zimage_toy()
    f16.install_fp32_islands(m, "z-image", "cpu")
    x = torch.randn(4, 64)
    x[0, 0] = float("inf")
    y = m.layers[0].feed_forward.w2(x)
    assert not torch.isfinite(y).all()


def test_scaled_mm_skips_weights_past_fp16_and_honours_the_kill_switch(scaled_mm, monkeypatch):
    from core.training import diffusion_dit_fp16 as f16

    m = _zimage_toy()
    m.layers[0].feed_forward.w2.base_layer.weight.data[0, 0] = 1.0e5
    rep = {}
    f16.install_fp32_islands(m, "z-image", "cpu", report = rep)
    assert rep["fp16_scaled_mm"] == 1
    assert "_unsloth_scaled_mm_orig" not in m.layers[0].feed_forward.w2.base_layer.__dict__
    y = m.layers[0].feed_forward.w2(torch.randn(2, 64))
    assert m.layers[0].feed_forward.w2.base_layer.calls == ["orig"] and y.dtype is torch.float32
    monkeypatch.undo()
    monkeypatch.setenv(f16.SCALED_MM_ENV, "0")
    assert f16.scaled_mm_supported("cuda") is False
    monkeypatch.delenv(f16.SCALED_MM_ENV)
    assert f16.scaled_mm_supported("cpu") is False


def test_scaled_mm_recomputes_identically_under_gradient_checkpointing(scaled_mm):
    from torch.utils.checkpoint import checkpoint

    from core.training import diffusion_dit_fp16 as f16

    m = _zimage_toy()
    f16.install_fp32_islands(m, "z-image", "cpu")
    w2 = m.layers[0].feed_forward.w2
    x = (torch.randn(6, 64) * 3.0e4).requires_grad_(True)
    y1 = w2(x)
    y1.sum().backward()
    g1 = x.grad.clone()
    x.grad = None
    y2 = checkpoint(w2, x, use_reentrant = False)
    y2.sum().backward()
    assert torch.equal(y1, y2) and torch.equal(g1, x.grad)


def test_trainer_reports_scaled_mm_layers_in_model_load_event():
    import inspect

    src = inspect.getsource(dit._train_dit)
    assert "install_fp32_islands(transformer, spec.family, device, report = fp16_report)" in src
    assert '"fp16_scaled_mm": fp16_report.get("fp16_scaled_mm", 0)' in src
