# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer split vs the load-time speed setup, CPU only: Studio's First-Block-Cache is installed before placement, so a
split must remove it instead of refusing the denoiser, the load must re-read the cache state, and a decline must name
the engine's real reason."""

from __future__ import annotations

import types

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
    # pin the host so the fake [0, 1] cards plan on any machine (ROCm included)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(mg, "_capability", lambda o: None)
    monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
    monkeypatch.setattr(mg, "_rocm_torch", lambda: False)
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


def test_the_speed_layer_is_set_up_for_a_split_the_placement_will_make():
    plan = types.SimpleNamespace(offload_policy = "group")
    split = types.SimpleNamespace(offload_policy = "none", layer_split = object())
    assert D._layer_split_expected(None, plan, lambda p, pl, lg: split)
    assert not D._layer_split_expected(
        None, plan, lambda p, pl, lg: plan
    )  # the one-card plan stands
    resident = types.SimpleNamespace(offload_policy = "none")
    assert not D._layer_split_expected(
        None, resident, lambda p, pl, lg: split
    )  # already resident: nothing to change
    assert not D._layer_split_expected(
        None, plan, lambda p, pl, lg: 1 / 0
    )  # a failing preview keeps the plan


def test_the_step_cache_goes_back_on_after_the_split_placed_the_blocks(monkeypatch):
    seen = []
    monkeypatch.setattr(
        D,
        "apply_step_cache",
        lambda view, mode = None, **kw: seen.append((view, mode, kw)) or f"m{len(seen)}",
    )
    pipe, second = object(), object()
    assert D._restore_step_cache_after_layer_split(pipe, (second, pipe), threshold = 0.1) == "m2"
    assert [v for v, _m, _k in seen] == [second, pipe] and {m for _v, m, _k in seen} == {
        dc.TC_FBCACHE
    }
    assert seen[0][2] == {"threshold": 0.1}


def test_split_loads_bind_speed_records_after_placement():
    import inspect

    from core.inference import video as V

    image = inspect.getsource(D.DiffusionBackend)
    video = inspect.getsource(V.VideoBackend)
    for src in (image, video):
        assert "split_expected = _layer_split_expected(" in src
        assert "offload_active = speed_offload" in src
        assert "_restore_step_cache_after_layer_split(" in src
        assert "engage_pinned_denoisers(" in src
    assert "diffusion_cuda_graph.arm_block_graphs(" in video


def test_a_resident_torchao_free_int8_dit_is_compiled_like_a_torchao_one(monkeypatch):
    monkeypatch.setattr(D, "native_offload_host", lambda target: True)
    monkeypatch.setattr(D, "native_quant_host", lambda target: False)
    assert D._native_quant_needs_compile(object(), "int8", resident = True)
    assert not D._native_quant_needs_compile(
        object(), "int8", resident = False
    )  # streamed: the offload path decides
    assert not D._native_quant_needs_compile(
        object(), None, resident = True
    )  # torchao: the existing rule
    monkeypatch.setattr(
        D, "native_quant_host", lambda target: True
    )  # ROCm / Apple keep their native policy
    assert not D._native_quant_needs_compile(object(), "int8", resident = True)
    monkeypatch.setattr(D, "native_quant_host", lambda target: False)
    monkeypatch.setattr(D, "native_offload_host", lambda target: False)
    assert not D._native_quant_needs_compile(object(), "int8", resident = True)


def test_a_companion_split_preview_counts_as_a_resident_denoiser():
    plan = types.SimpleNamespace(offload_policy = "group")
    companion = types.SimpleNamespace(offload_policy = "none", secondary_ordinal = 1, layer_split = None)
    assert D._companion_split_expected(None, plan, lambda p, pl, lg: companion)
    assert not D._companion_split_expected(
        None, plan, lambda p, pl, lg: plan
    )  # the one-card plan stands
    layered = types.SimpleNamespace(
        offload_policy = "none", secondary_ordinal = None, layer_split = object()
    )
    assert not D._companion_split_expected(
        None, plan, lambda p, pl, lg: layered
    )  # _layer_split_expected's case
    resident = types.SimpleNamespace(offload_policy = "none")
    assert not D._companion_split_expected(
        None, resident, lambda p, pl, lg: companion
    )  # already resident
    assert not D._companion_split_expected(
        None, plan, lambda p, pl, lg: 1 / 0
    )  # a failing preview keeps the plan


def test_both_backends_price_a_companion_split_as_resident_for_the_quant_rule():
    import inspect

    from core.inference import video as V
    for src in (inspect.getsource(D.DiffusionBackend), inspect.getsource(V.VideoBackend)):
        rule = src[src.index("quant_forces_compile = ") :]
        rule = rule[: rule.index("if quant_forces_compile and effective_speed == SPEED_OFF:")]
        assert (
            "_native_quant_needs_compile(" in rule
            and "_companion_split_expected(pipe, plan" in rule
        )


def test_both_backends_force_compile_through_the_one_quant_rule():
    import inspect

    from core.inference import video as V
    for src in (inspect.getsource(D.DiffusionBackend), inspect.getsource(V.VideoBackend)):
        assert "quant_forces_compile = " in src and "_native_quant_needs_compile(" in src
        assert "if quant_forces_compile and effective_speed == SPEED_OFF:" in src
        assert src.index("split_expected = _layer_split_expected(") < src.index(
            "quant_forces_compile = "
        )
