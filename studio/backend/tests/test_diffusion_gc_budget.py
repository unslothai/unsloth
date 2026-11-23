# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for diffusion_gc_budget: modes, engagement, plan arithmetic, OOM fall back, gradient parity."""

from __future__ import annotations

import pytest
import torch

import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_gc_budget as gb

GIB = 1 << 30


@pytest.mark.parametrize(
    "raw, mode",
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("plain", "plain"),
        ("PARTIAL", "partial"),
        ("junk", "auto"),
    ],
)
def test_mode_parsing(raw, mode, monkeypatch):
    if raw is None:
        monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    else:
        monkeypatch.setenv(gb.GC_MODE_ENV, raw)
    assert gb.gc_budget_mode() == mode


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = None

    def enable_gradient_checkpointing(self, gradient_checkpointing_func = None):
        self.func = gradient_checkpointing_func


def test_engages_only_on_cuda_auto_and_diffusers_models(monkeypatch):
    monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    model = _Model()
    budget = gb.enable_budgeted_checkpointing(model, "cuda")
    assert isinstance(budget, gb.BudgetedCheckpoint) and model.func is budget
    assert gb.enable_budgeted_checkpointing(_Model(), "cpu") is None
    assert gb.enable_budgeted_checkpointing(_Model(), "xpu") is None
    assert gb.enable_budgeted_checkpointing(torch.nn.Linear(2, 2), "cuda") is None


@pytest.mark.parametrize("mode", ["plain", "partial"])
def test_kill_switch_keeps_the_old_path(mode, monkeypatch):
    monkeypatch.setenv(gb.GC_MODE_ENV, mode)
    model = _Model()
    assert gb.enable_budgeted_checkpointing(model, "cuda") is None
    assert model.func is None


def test_rocm_uses_the_budget_too(monkeypatch):
    # gfx1151 Linux + Windows: 1.40x / 2.18x bf16 klein LoRA steps, Linux losses bitwise equal to every block checkpointed
    monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.13.0")
    model = _Model()
    budget = gb.enable_budgeted_checkpointing(model, "cuda")
    assert isinstance(budget, gb.BudgetedCheckpoint) and model.func is budget
    monkeypatch.setenv(gb.GC_MODE_ENV, "plain")
    assert gb.enable_budgeted_checkpointing(_Model(), "cuda") is None


def test_budget_reads_trusted_free(monkeypatch):
    # Windows ROCm's WDDM free over-report is capped (utils.hardware.trusted_mem_get_info) before any plan
    import utils.hardware as hw
    monkeypatch.setattr(hw, "trusted_mem_get_info", lambda *a, **k: (7, 9))
    assert gb._mem_get_info() == (7, 9)


@pytest.mark.parametrize(
    "platform, hip, device, capped",
    [
        ("win32", "6.4.0", None, True),  # the spawned trainer: detection never ran, a ROCm torch
        ("win32", None, None, False),  # Windows NVIDIA
        ("linux", "6.4.0", None, False),  # Linux ROCm: the driver figure is trusted
        ("win32", "6.4.0", "cuda", False),  # detection ran and did not say ROCm
    ],
)
def test_trainer_child_caps_windows_rocm_free(platform, hip, device, capped, monkeypatch):
    # The trainer is a spawned process that never runs detect_hardware(), so IS_ROCM stays False there
    import types

    import utils.hardware.hardware as hwmod
    import core.training.diffusion_lora_fast as fl

    gib = 1 << 30
    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(hip = hip)
    fake.__version__ = "2.9.1"
    fake.cuda = types.SimpleNamespace(
        mem_get_info = lambda *a: (16 * gib, 16 * gib), memory_reserved = lambda *a: 6 * gib
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    monkeypatch.setattr(hwmod.sys, "platform", platform)
    monkeypatch.setattr(hwmod, "IS_ROCM", False)
    monkeypatch.setattr(hwmod, "DEVICE", device)
    want = (10 * gib if capped else 16 * gib, 16 * gib)
    assert gb._mem_get_info() == want and fl._mem_get_info() == want


def _planned(
    class_cost,
    blocks,
    probe_area = None,
):
    budget = gb.BudgetedCheckpoint()
    budget.class_cost = dict(class_cost)
    for idx, (cls, proxy) in enumerate(blocks):
        budget.block_class[idx] = cls
        budget.block_proxy[idx] = proxy
    budget.probe_area = probe_area
    return budget


def test_plan_keeps_blocks_in_order_until_the_headroom_is_spent():
    # 40 GiB card, 36 GiB free, 4 GiB projected peak, 4 GiB margin: (36 - 4 - 4) * 0.85 = 23.8 GiB for kept blocks.
    budget = _planned({"B": 1.0}, [("B", GIB)] * 10)
    assert budget.plan(4 * GIB, 36 * GIB, 0, 40 * GIB) == 10
    budget = _planned({"B": 4.0}, [("B", GIB)] * 10)
    assert budget.plan(4 * GIB, 36 * GIB, 0, 40 * GIB) == 5
    assert budget.keep == frozenset(range(5))
    assert budget.kept_bytes == 20 * GIB
    assert budget.planned and not budget.probing


def test_plan_counts_reserved_cache_and_keeps_the_margin():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 10)
    # 24 GiB card: margin 2.4 GiB; free 2 + cached 6 - peak 6 leaves nothing once the margin is taken.
    assert budget.plan(6 * GIB, 2 * GIB, 6 * GIB, 24 * GIB) == 0
    assert budget.keep == frozenset()


def test_plan_scales_to_the_largest_bucket():
    blocks = [("B", GIB)] * 10
    plain = _planned({"B": 2.0}, blocks).plan(4 * GIB, 36 * GIB, 0, 40 * GIB)
    scaled = _planned({"B": 2.0}, blocks, probe_area = 100).plan(
        4 * GIB, 36 * GIB, 0, 40 * GIB, max_canvas_area = 200
    )
    assert (plain, scaled) == (10, 5)


def test_plan_skips_an_unmeasured_class():
    budget = _planned({"A": 1.0}, [("A", GIB), ("A", GIB), ("B", GIB), ("A", GIB)])
    assert budget.plan(0, 80 * GIB, 0, 80 * GIB) == 3
    assert budget.keep == frozenset({0, 1, 3})


def test_spare_bytes_now_gives_up_what_another_load_took(monkeypatch):
    budget = _planned({"B": 1.0}, [("B", GIB)] * 4)
    # 80 GiB card, margin 8 GiB: headroom (80 - 8) * 0.85 = 61.2 GiB, 4 GiB kept.
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    spare = budget.spare_bytes
    assert spare > 50 * GIB
    state = {"free": 80 * GIB, "reserved": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (state["free"], 80 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: state["reserved"])
    assert budget.spare_bytes_now() == spare
    # This process emptying its cache moves bytes from reserved to free: nothing lost.
    state.update(free = 70 * GIB, reserved = 10 * GIB)
    assert budget.spare_bytes_now() == spare
    # Another load took 20 GiB since the plan.
    state.update(free = 50 * GIB, reserved = 10 * GIB)
    assert budget.spare_bytes_now() == spare - 20 * GIB
    state.update(free = 0, reserved = 0)
    assert budget.spare_bytes_now() == 0
    budget.step_down()
    state.update(free = 80 * GIB, reserved = 0)
    assert budget.spare_bytes_now() == 0


def test_fall_back_clears_the_plan():
    budget = _planned({"B": 1.0}, [("B", GIB)] * 4)
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    assert budget.fall_back() is True
    assert budget.disabled and budget.keep == frozenset()
    assert budget.fall_back() is False
    assert budget.plan(0, 80 * GIB, 0, 80 * GIB) == 0


def test_a_caller_first_seen_after_the_plan_is_checkpointed_and_not_recorded():
    budget = gb.BudgetedCheckpoint(base = lambda fn, *a, **k: ("ckpt", fn(*a, **k)))
    budget.probing = False
    budget.plan(0, 80 * GIB, 0, 80 * GIB)
    assert budget(lambda x: x + 1, 1) == ("ckpt", 2)
    assert budget.index == {}


def test_is_out_of_memory():
    assert gb.is_out_of_memory(torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate"))
    assert gb.is_out_of_memory(RuntimeError("CUDA error: out of memory"))
    assert not gb.is_out_of_memory(RuntimeError("shape mismatch"))
    assert not gb.is_out_of_memory(ValueError("out of memory"))


def _tiny_zimage():
    diffusers = pytest.importorskip("diffusers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = diffusers.ZImageTransformer2DModel(
        dim = 64,
        n_layers = 4,
        n_refiner_layers = 1,
        n_heads = 2,
        n_kv_heads = 2,
        cap_feat_dim = 32,
        axes_dims = [8, 12, 12],
        axes_lens = [64, 32, 32],
    )
    model.requires_grad_(False)
    model.add_adapter(
        peft.LoraConfig(r = 4, lora_alpha = 4, init_lora_weights = False, target_modules = ["to_q", "to_v"])
    )
    return model


def _lora_grads(model):
    torch.manual_seed(1)
    x, cap, t = [torch.randn(16, 1, 8, 8)], [torch.randn(6, 32)], torch.tensor([0.3])
    model.train()
    model.zero_grad(set_to_none = True)
    out = model(x, t, cap, return_dict = False)[0]
    sum((o**2).mean() for o in out).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def test_every_state_matches_uncheckpointed_grads(monkeypatch):
    monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    # CPU stand-in for the CUDA allocator counter: grows by 1 MiB per read, so a measured block keeps ~1 MiB.
    counter = iter(range(0, 1 << 40, 1 << 20))
    monkeypatch.setattr(gb, "_memory_allocated", lambda: next(counter))
    reference = _lora_grads(_tiny_zimage())
    model = _tiny_zimage()
    calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        lambda fn, *a, **k: calls.append(k.get("use_reentrant")) or real_checkpoint(fn, *a, **k),
    )
    budget = gb.enable_budgeted_checkpointing(model, "cuda")
    assert budget is not None

    def check(n_checkpointed):
        calls.clear()
        grads = _lora_grads(model)
        assert grads.keys() == reference.keys()
        for name in reference:
            torch.testing.assert_close(grads[name], reference[name], rtol = 1e-5, atol = 1e-6)
        assert set(calls) <= {False}
        assert len(calls) == n_checkpointed

    # Step 1: 6 ZImageTransformerBlocks (noise refiner, context refiner, 4 layers), one class: the first is measured.
    check(5)
    assert budget.total_blocks == 6 and set(budget.class_cost) == {"ZImageTransformerBlock"}
    assert budget.class_cost["ZImageTransformerBlock"] > 0
    assert budget.plan(0, 80 * GIB, 0, 80 * GIB) == 6
    check(0)
    # Room for two: the first two keep their activations.
    budget.planned = False
    budget.keep = frozenset({0, 1})
    check(4)
    budget.fall_back()
    check(6)


def test_plan_failure_leaves_every_block_checkpointed(monkeypatch):
    budget = _planned({"B": 1.0}, [("B", GIB)] * 4)

    def boom(*_a, **_k):
        raise RuntimeError("no device")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    dit._plan_gc_budget(budget, 0, 0)
    assert budget.disabled and budget.keep == frozenset()
