# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the checkpointing budget runs alongside DiT block swap. Swapped blocks stay checkpointed and are never
planned, resident blocks are planned into the room left, the pool's auto growth is held back, and an out-of-memory
rerun starts from a clean slot pool."""

from __future__ import annotations

import inspect
import sys
import types

import torch

import core.training.diffusion_block_swap as bs
import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_gc_budget as gb

GIB = 1 << 30


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Model(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self, n = 4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(n)])


class _Base:
    def __init__(self):
        self.calls = []
        self.context_fn = None

    def __call__(self, function, *args, **kwargs):
        self.calls.append(gb._owner(function))
        return function(*args, **kwargs)


def _step(budget, model, x):
    for block in model.blocks:
        x = budget(block, x)
    return x


def test_swapped_blocks_are_never_measured_or_kept(monkeypatch):
    counter = iter(range(0, 10**9, 1000))
    monkeypatch.setattr(gb, "_memory_allocated", lambda: next(counter))
    model = _Model(4)
    swapped = (model.blocks[0], model.blocks[3])
    for block in swapped:
        setattr(block, gb.SWAPPED_ATTR, True)
    base, plain = _Base(), _Base()
    budget = gb.BudgetedCheckpoint(base = base, swapped = plain)
    x = torch.randn(1, 2, 4)
    _step(budget, model, x)
    # Probing measured the first resident block, not the swapped first block; swapped blocks went through the
    # plain checkpoint every time, resident ones never.
    assert sorted(budget.block_class) == [0, 1]
    assert plain.calls == [model.blocks[0], model.blocks[3]]
    assert model.blocks[0] not in base.calls and model.blocks[3] not in base.calls
    budget.plan(projected_peak = 1 * GIB, free_bytes = 70 * GIB, reserved_bytes = 0, total_bytes = 80 * GIB)
    assert budget.planned and len(budget.keep) == 2
    base.calls.clear()
    plain.calls.clear()
    _step(budget, model, x)
    # After the plan the resident blocks run unchecked; the swapped ones are still checkpointed.
    assert base.calls == [] and plain.calls == [model.blocks[0], model.blocks[3]]


def test_swapped_blocks_use_a_checkpoint_without_the_selective_policy():
    # A swapped block's hooks copy weights inside the region (and the other way round in the recompute): a selective
    # checkpoint would fail the backward with a copy_ it has no storage for.
    budget = gb.BudgetedCheckpoint()
    assert budget.swapped.context_fn is None
    block = _Block()
    setattr(block, gb.SWAPPED_ATTR, True)
    x = torch.randn(1, 2, 4, requires_grad = True)
    budget(block, x).sum().backward()
    assert budget.swapped.saved_attention == 0 and x.grad is not None


def _install_fake_zoo(monkeypatch, made):
    class FakeSwap:
        def __init__(
            self,
            layers,
            indices,
            prefetch_depth = 2,
            device = None,
            model = None,
        ):
            self.blocks = [layers[i] for i in indices]
            made.append(self)

        def host_bytes(self):
            return 0

        def remove(self):
            pass

    mod = types.ModuleType("unsloth_zoo.block_swap")
    mod.BlockSwap = FakeSwap
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap", mod)
    monkeypatch.setattr(bs, "_place_rest", lambda *a, **k: None)


def test_install_marks_only_the_swapped_blocks(monkeypatch):
    made = []
    _install_fake_zoo(monkeypatch, made)
    model = _Model(4)
    info = bs.install_dit_block_swap(
        model, "cuda:0", 2, tokens = 16, free_bytes = 1 * GIB, total_bytes = 8 * GIB
    )
    assert info["swapped_blocks"] == 2
    marked = [i for i, b in enumerate(model.blocks) if getattr(b, gb.SWAPPED_ATTR, False)]
    assert marked == [i for i in info["swapped"]["blocks"]] and len(marked) == 2


class _FakeBlock:
    def __init__(self, nbytes):
        self._n = nbytes

    def nbytes(self):
        return self._n


class _FakeSwapper:
    def __init__(
        self,
        depth,
        settled,
        sizes,
        sigs = 1,
    ):
        self._auto = True
        self._settled = settled
        self.depth = depth
        self.blocks = [_FakeBlock(n) for n in sizes]
        self.free = {i: [] for i in range(sigs)}
        self.resets = 0

    def reset(self):
        self.resets += 1


def test_pool_growth_is_reserved_until_auto_depth_settles(monkeypatch):
    zoo = types.ModuleType("unsloth_zoo.block_swap")
    zoo._AUTO_MAX_DEPTH = 4
    pkg = sys.modules.get("unsloth_zoo") or types.ModuleType("unsloth_zoo")
    monkeypatch.setitem(sys.modules, "unsloth_zoo", pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap", zoo)
    monkeypatch.setattr(pkg, "block_swap", zoo, raising = False)
    model = _Model(2)
    model._unsloth_block_swaps = [
        _FakeSwapper(2, False, [GIB // 2, GIB // 4]),
        _FakeSwapper(3, True, [GIB]),
    ]
    assert bs.swap_pool_growth_bytes(model) == 2 * (GIB // 2)
    assert bs.swap_pool_growth_bytes(_Model(2)) == 0


def test_rerun_resets_the_swap_pools():
    model = _Model(2)
    swappers = [_FakeSwapper(2, True, [1]), _FakeSwapper(2, True, [1])]
    model._unsloth_block_swaps = swappers
    assert bs.reset_block_swaps(model) == 2
    assert [s.resets for s in swappers] == [1, 1]


def test_plan_holds_back_the_swap_pool_growth(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        dit, "_plan_gc_budget", lambda *a, held_bytes = 0, **k: seen.setdefault("held", held_bytes)
    )
    dit._plan_after_first_step(object(), None, 0, 0, swap_pool_bytes = 3 * GIB)
    assert seen["held"] == 3 * GIB


def test_trainer_keeps_the_budget_with_block_swap():
    src = inspect.getsource(dit._train_dit)
    assert "None if swapping else enable_budgeted_checkpointing" not in src
    assert "gc_budget = enable_budgeted_checkpointing(transformer, device)" in src
    retry = src[src.index("def _retry_fully_checkpointed") :]
    assert "reset_block_swaps(transformer)" in retry[: retry.index("replay.replay(")]
