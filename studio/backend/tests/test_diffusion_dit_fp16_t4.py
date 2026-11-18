# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for DiT LoRA training on GPUs without bf16 (T4 / sm75): which cards take the fp16 path, that bf16 cards
take exactly the old one, and the fp16 mechanics (loss-scale skips, LR schedule, scaler resume, fp32 islands)."""

from __future__ import annotations

import json
import random

import pytest
import torch

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
    monkeypatch.setattr(dit, "identity_for_config", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_apply_perf_flags", lambda *a, **k: None)
    monkeypatch.setattr(dit, "_restore_perf_flags", lambda *a, **k: None)
    monkeypatch.setattr(dit, "training_fetch_repo", lambda *a, **k: "x")
    out = dit.run_dit_lora_training(_cfg(family, tmp_path))
    assert out == "ok"
    return seen


# ---- routing: bf16 cards keep the old path exactly --------------------------------------------------------------


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


# ---- fp16 mechanics ------------------------------------------------------------------------------------------------


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
    # Overflow step: inf in the (scaled) gradient.
    scaler.scale((w * torch.tensor([1.0, float("inf"), 1.0, 1.0])).sum()).backward()
    scaler.unscale_(opt)
    assert dit._optimizer_update(opt, sched, scaler, True) is False
    assert torch.equal(w.detach(), before), "a skipped step must not move the weights"
    assert (
        sched.last_epoch == 0 and sched.get_last_lr()[0] == lr0
    ), "the LR schedule must not tick on a skip"
    assert scaler.get_scale() == 2.0**15
    # A finite step applies, ticks the schedule once, keeps the scale.
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
    for _ in range(3):  # back off three times
        opt.zero_grad(set_to_none = True)
        scaler.scale((w * float("inf")).sum()).backward()
        scaler.unscale_(opt)
        dit._optimizer_update(opt, sched, scaler, True)
    base = {"running_loss": 1.5, "resolved_base_precision": "nf4"}
    assert (
        dit._resume_progress(base, scaler, 3, False) is base
    ), "bf16 checkpoints keep the old progress block"
    prog = json.loads(
        json.dumps(dit._resume_progress(base, scaler, 3, True))
    )  # the manifest is JSON
    fresh = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    assert dit._restore_scaler(prog, fresh) == 3
    assert fresh.get_scale() == scaler.get_scale() == 2.0**13
    # A checkpoint from before the fp16 path (no scaler block) leaves the default scale.
    other = torch.amp.GradScaler("cpu", init_scale = 2.0**16)
    assert dit._restore_scaler(base, other) == 0 and other.get_scale() == 2.0**16


# ---- fp32 islands --------------------------------------------------------------------------------------------------


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
    # The island's quantized matmul computes fp32.
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
    # Hooks are gone after the block.
    rec2 = {}
    m(torch.ones(1, 4))
    assert rec2 == {}


def test_every_fp16_family_has_an_island_table_and_a_spec():
    from core.training import diffusion_dit_fp16 as f16
    for fam in dtc.DIT_FP16_TRAIN_FAMILIES:
        assert fam in dit._SPECS
        assert f16.FP16_FP32_MODULES.get(fam), fam
