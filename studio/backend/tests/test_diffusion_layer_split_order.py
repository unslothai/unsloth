# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer split vs the load-time speed setup, CPU only: Studio's First-Block-Cache is installed before placement, so a
split must remove it instead of refusing the denoiser, the load must re-read the cache state, and a decline must name
the engine's real reason."""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion as D
from core.inference import diffusion_cache as dc
from core.inference import diffusion_layer_split as ls
from core.inference import diffusion_layer_split_plan as lsp
from core.inference import diffusion_multigpu as mg

from .test_diffusion_layer_split import FakeDiT, Pipe, _two_card_budget
from .test_diffusion_layer_split_plan import _cards, _plan, _refine, engine  # noqa: F401 - fixture


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(ls.LAYER_SPLIT_ENV, raising = False)
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    yield


class _Registry:
    def __init__(self, *names):
        self.hooks = {n: object() for n in names}


def _hook(module, *names):
    module._diffusers_hook = _Registry(*names)


def _fbcache(dit):
    """What apply_step_cache leaves: FBC hooks on the leader and the other blocks, and the marker."""
    blocks = [b for _, _, b in ls.find_blocks(dit)]
    _hook(blocks[0], "fbc_leader_block_hook")
    for block in blocks[1:]:
        _hook(block, "fbc_block_hook")
    dit._unsloth_step_cache = dc.TC_FBCACHE


def _fake_toggle(monkeypatch, removes = True):
    calls = []

    def toggle(pipe, *, steps, **kw):
        calls.append(steps)
        t = pipe.transformer
        if removes:
            for m in t.modules():
                m.__dict__.pop("_diffusers_hook", None)
            t._unsloth_step_cache = None
        return None

    monkeypatch.setattr(dc, "maybe_toggle_step_cache", toggle)
    return calls


def test_studio_step_cache_does_not_refuse_a_split_but_other_hooks_do():
    dit = FakeDiT()
    _fbcache(dit)
    assert ls.step_cache_hooked(dit)
    assert "carries diffusers hooks" in ls.layer_split_refusal(dit)
    assert ls.layer_split_refusal(dit, step_cache_ok = True) is None
    # a foreign hook next to the cache (group offload, layerwise casting) still refuses
    _hook(dit.proj_out, "group_offloading")
    assert not ls.step_cache_hooked(dit)
    assert "carries diffusers hooks" in ls.layer_split_refusal(dit, step_cache_ok = True)
    assert not ls.step_cache_hooked(FakeDiT())


def test_placement_removes_the_step_cache_first(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    _fbcache(dit)
    pipe = Pipe(dit)
    calls = _fake_toggle(monkeypatch)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    assert calls == [0]  # disengaged through the cache's own teardown
    assert ls.is_layer_split(dit) and not ls.step_cache_hooked(dit)
    assert D._step_cache_dropped_by_layer_split(pipe)
    ls.undo_layer_split(pipe)


def test_a_cache_that_will_not_come_off_refuses_the_split(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    _fbcache(dit)
    _fake_toggle(monkeypatch, removes = False)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    with pytest.raises(ls.LayerSplitError, match = "step cache"):
        ls.place_layer_split(Pipe(dit), primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    assert not ls.is_layer_split(dit)


def test_a_split_that_does_not_fit_leaves_the_cache_alone(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    _fbcache(dit)
    calls = _fake_toggle(monkeypatch)
    with pytest.raises(ls.LayerSplitError):
        ls.place_layer_split(Pipe(dit), primary = 0, cards = [1], budgets_mib = {0: 1, 1: 1})
    assert calls == [] and ls.step_cache_hooked(dit)


def test_uncached_or_still_cached_pipes_are_not_reported_dropped():
    dit = FakeDiT()
    pipe = Pipe(dit)
    dit._unsloth_step_cache = dc.TC_FBCACHE
    assert not D._step_cache_dropped_by_layer_split(pipe)
    dit._unsloth_step_cache = None
    assert D._step_cache_dropped_by_layer_split(pipe)


def test_the_planner_plans_through_the_step_cache():
    dit = FakeDiT()
    _fbcache(dit)
    blocks = lsp.split_blocks(dit)
    assert blocks is not None and len(blocks) == 6


def test_a_decline_names_the_engines_reason(monkeypatch, engine):  # noqa: F811
    _cards(monkeypatch)
    dit = FakeDiT()
    _hook(dit.transformer_blocks[0], "group_offloading")

    class HookedPipe:
        transformer = dit
        components: dict = {}

    pipe = HookedPipe()
    monkeypatch.setattr(lsp, "_denoiser_layouts", lambda p: None)
    plan = _plan()
    refined, _ = _refine(plan, pipe = pipe)
    assert refined is plan
    declined = lsp.layer_split_status(pipe)["declined"]
    assert declined.startswith("denoiser cannot be split (transformer: ")
    assert "group_offloading" in declined

