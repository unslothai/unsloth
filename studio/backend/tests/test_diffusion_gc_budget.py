# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for the budgeted DiT gradient checkpointing (diffusion_gc_budget): mode parsing and the kill switch,
which devices and models it engages on, the first-step measurement, the plan arithmetic, the out-of-memory fall back,
and LoRA gradients equal to an uncheckpointed run in every state."""

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
    monkeypatch.setattr(gb, "_is_rocm", lambda: False)
    model = _Model()
    budget = gb.enable_budgeted_checkpointing(model, "cuda")
    assert isinstance(budget, gb.BudgetedCheckpoint) and model.func is budget
    assert gb.enable_budgeted_checkpointing(_Model(), "cpu") is None
    assert gb.enable_budgeted_checkpointing(_Model(), "xpu") is None
    assert gb.enable_budgeted_checkpointing(torch.nn.Linear(2, 2), "cuda") is None


@pytest.mark.parametrize("mode", ["plain", "partial"])
def test_kill_switch_keeps_the_old_path(mode, monkeypatch):
    monkeypatch.setenv(gb.GC_MODE_ENV, mode)
    monkeypatch.setattr(gb, "_is_rocm", lambda: False)
    model = _Model()
    assert gb.enable_budgeted_checkpointing(model, "cuda") is None
    assert model.func is None


def test_rocm_keeps_the_old_path(monkeypatch):
    monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    monkeypatch.setattr(gb, "_is_rocm", lambda: True)
    assert gb.enable_budgeted_checkpointing(_Model(), "cuda") is None


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


def test_plan_stops_at_an_unmeasured_class():
    budget = _planned({"A": 1.0}, [("A", GIB), ("A", GIB), ("B", GIB), ("A", GIB)])
    assert budget.plan(0, 80 * GIB, 0, 80 * GIB) == 2


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
    monkeypatch.setattr(gb, "_is_rocm", lambda: False)
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
    # Room for every block: nothing recomputed.
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
