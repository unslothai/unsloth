# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer split planner (core/inference/diffusion_layer_split_plan.py), CPU only, fake card memory.

The pure packer is driven with card limits directly; the post-load refinement through ``refine_plan_for_multigpu``
with patched loaded sizes, block layouts and secondary limits; the engine is a recording fake."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from core.inference import diffusion_layer_split_plan as lsp
from core.inference import diffusion_memory as dm
from core.inference import diffusion_multigpu as mg

L = lsp.DenoiserLayout


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    monkeypatch.delenv(lsp.LAYER_SPLIT_ENV, raising = False)
    # make_selection filters the fake [0, 1] cards through the real host (torch's device count, the primary's
    # capability) and the planner declines on a HIP torch: pinned, so a ROCm host (a one-GPU APU included) plans
    # these fake cards like any other. The ROCm decline has its own test.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(mg, "_capability", lambda o: None)
    monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
    yield


def _check(split, limits):
    """Every card within its limit, every block placed exactly once in contiguous ranges."""
    for o, req in split.card_required_mib:
        assert req <= limits[o], (o, req, limits[o])
    for name in dict.fromkeys(r.denoiser for r in split.ranges):
        rs = [r for r in split.ranges if r.denoiser == name]
        assert rs[0].start == 0
        for a, b in zip(rs, rs[1:]):
            assert a.stop == b.start
    return split


def test_packs_a_dit_too_big_for_one_card_over_two():
    limits = {0: 18000, 1: 18000}
    split, reason = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 23)},
        companions = {"text_encoder": (4000, "text_encoder"), "vae": (300, "other")},
        primary = 0,
        card_limits = limits,
        headroom_mib = 2000,
        overhead_mib = 500,
    )
    assert split is not None, reason
    _check(split, limits)
    assert split.cards == (0, 1)
    assert split.ranges[-1].stop == 23
    assert split.text_encoders_card == 1
    weights = dict(split.card_weights_mib)
    assert sum(weights.values()) == 1000 + 23000 + 4000 + 300
    assert weights[0] == 1000 + 300 + sum(r.mib for r in split.ranges if r.card == 0)
    assert split.boundaries == 2
    assert "layer split" in reason


def test_text_encoders_too_big_for_any_card_stream_and_the_blocks_still_split():
    kw = dict(
        denoisers = {"transformer": L(1000, (1100,) * 54)},
        companions = {"text_encoder": (45000, "text_encoder"), "vae": (300, "other")},
        primary = 0,
        card_limits = {0: 55000, 1: 55000},
        headroom_mib = 6000,
        overhead_mib = 1000,
    )
    split, reason = lsp.plan_layer_split(**kw)
    assert split is not None, reason
    assert split.text_encoders_streamed and split.text_encoders_card is None
    assert split.as_public_dict()["text_encoders_streamed"] is True
    assert {r.card for r in split.ranges} == {0, 1}
    _check(split, kw["card_limits"])
    kw["card_limits"] = {0: 55000, 1: 110000}
    split, reason = lsp.plan_layer_split(**kw)
    assert split is not None, reason
    assert not split.text_encoders_streamed and split.text_encoders_card == 1


def test_streamed_encoders_are_paged_not_moved_and_undo_removes_their_hooks(monkeypatch):
    calls = {"stream": [], "engine": [], "removed": []}

    class Comp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.moved = []

        def to(self, *a, **k):
            self.moved.append(a[0] if a else k.get("device"))
            return self

    class Pipe:
        def __init__(self):
            self.transformer = Comp()
            self.text_encoder = Comp()
            self.vae = Comp()
            self.components = {
                "transformer": self.transformer,
                "text_encoder": self.text_encoder,
                "vae": self.vae,
            }

    pipe = Pipe()
    split = lsp.LayerSplit(
        primary = 0,
        cards = (0, 1),
        ranges = (
            lsp.BlockRange("transformer", 0, 1, 0, 1),
            lsp.BlockRange("transformer", 1, 2, 1, 1),
        ),
        text_encoders_card = None,
        card_required_mib = ((0, 1), (1, 1)),
        card_limit_mib = ((0, 9), (1, 9)),
        text_encoders_streamed = True,
    )
    from core.inference import diffusion_layer_split as engine

    monkeypatch.setattr(
        dm,
        "stream_text_encoders_onto",
        lambda p, d, logger = None: calls["stream"].append(str(d)) or ["text_encoder"],
    )
    monkeypatch.setattr(dm, "_remove_group_offload_hooks", lambda m: calls["removed"].append(m))
    monkeypatch.setattr(lsp, "split_blocks", lambda m: [("b.0", None), ("b.1", None)])
    monkeypatch.setattr(
        engine, "place_layer_split", lambda p, **kw: calls["engine"].append(kw["assignment"]) or {}
    )
    monkeypatch.setattr(engine, "undo_layer_split", lambda p, attr = "transformer": False)
    import contextlib

    from core.inference import diffusion_fast_load

    monkeypatch.setattr(
        diffusion_fast_load, "fast_upload", lambda mods, dev, logger = None: contextlib.nullcontext()
    )
    lsp.place_layer_split(pipe, primary = "cuda:0", split = split)
    assert pipe.text_encoder.moved == []  # never moved onto a card: paged by its hooks
    assert pipe.vae.moved == [torch.device("cuda", 0)]
    assert calls["stream"] == ["cuda:0"]
    assert calls["engine"] == [{"b.0": 0, "b.1": 1}]
    record = mg._PLACEMENTS[pipe]
    assert record["streamed"] == ["text_encoder"] and record["moved"] == []
    assert record["placement"]["text_encoder"].startswith("cpu (streamed")
    lsp.undo_layer_split(pipe)
    assert calls["removed"] == [pipe.text_encoder]
    assert pipe not in mg._PLACEMENTS


def test_every_denoiser_keeps_a_block_off_the_primary():
    limits = {0: 100000, 1: 6000}
    kw = dict(
        companions = {"text_encoder": (2000, "text_encoder"), "vae": (300, "other")},
        primary = 0,
        card_limits = limits,
        headroom_mib = 1000,
        overhead_mib = 0,
        working_set_mib = 1000,
        secondary_overhead_mib = 0,
    )
    for denoisers in (
        {"transformer": L(0, (1000,) * 10)},
        {"transformer": L(0, (1000,) * 10), "transformer_2": L(0, (1000,) * 10)},
    ):
        split, reason = lsp.plan_layer_split(denoisers = denoisers, **kw)
        assert split is not None, reason
        _check(split, limits)
        for name in denoisers:
            assert any(r.card != 0 for r in split.ranges if r.denoiser == name), (
                name,
                split.ranges,
            )


def test_too_small_for_the_combined_cards_declines():
    split, reason = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 40)},
        companions = {"vae": (300, "other")},
        primary = 0,
        card_limits = {0: 18000, 1: 18000},
        headroom_mib = 2000,
        overhead_mib = 500,
    )
    assert split is None
    assert "does not fit" in reason


def test_no_secondary_or_unknown_blocks_declines():
    assert (
        lsp.plan_layer_split(
            denoisers = {"transformer": L(0, (1000,))},
            companions = {},
            primary = 0,
            card_limits = {0: 18000},
            headroom_mib = 0,
            overhead_mib = 0,
        )[0]
        is None
    )
    assert (
        lsp.plan_layer_split(
            denoisers = {"transformer": L(1000, ())},
            companions = {},
            primary = 0,
            card_limits = {0: 18000, 1: 18000},
            headroom_mib = 0,
            overhead_mib = 0,
        )[0]
        is None
    )


def test_mixed_card_sizes_cut_in_proportion_to_free_capacity():
    limits = {0: 20000, 1: 68000}
    split, _ = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 60)},
        companions = {},
        primary = 0,
        card_limits = limits,
        headroom_mib = 3000,
        overhead_mib = 500,
    )
    _check(split, limits)
    on = {o: sum(r.mib for r in split.ranges if r.card == o) for o in (0, 1)}
    reserve = lsp.secondary_reserve_mib()
    caps = {0: 20000 - 1000 - 3000 - max(500, reserve), 1: 68000 - 2000 - reserve}
    for o in (0, 1):
        assert abs(on[o] / caps[o] - 60000 / sum(caps.values())) < 1000 / caps[o] + 1e-9


def test_uses_a_third_card_only_when_two_are_not_enough():
    limits = {0: 18000, 1: 18000, 2: 18000}
    two, _ = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 25)},
        companions = {},
        primary = 0,
        card_limits = limits,
        headroom_mib = 2000,
        overhead_mib = 500,
    )
    assert two.cards == (0, 1)
    three, _ = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 40)},
        companions = {},
        primary = 0,
        card_limits = limits,
        headroom_mib = 2000,
        overhead_mib = 500,
    )
    _check(three, limits)
    assert three.cards == (0, 1, 2)
    assert three.boundaries == 3


def test_largest_secondary_is_used_first():
    split, _ = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 25)},
        companions = {},
        primary = 0,
        card_limits = {0: 18000, 1: 9000, 2: 30000},
        headroom_mib = 2000,
        overhead_mib = 500,
    )
    assert split.cards == (0, 2)


def test_two_experts_are_each_cut_with_the_headroom_charged_once():
    limits = {0: 40000, 1: 40000}
    split, reason = lsp.plan_layer_split(
        denoisers = {"transformer": L(500, (700,) * 40), "transformer_2": L(500, (700,) * 40)},
        companions = {"text_encoder": (6000, "text_encoder"), "vae": (300, "other")},
        primary = 0,
        card_limits = limits,
        headroom_mib = 4000,
        overhead_mib = 500,
    )
    assert split is not None, reason
    _check(split, limits)
    assert {r.denoiser for r in split.ranges if r.card == 1} == {"transformer", "transformer_2"}
    weights = dict(split.card_weights_mib)
    required = dict(split.card_required_mib)
    reserve = lsp.secondary_reserve_mib()
    assert required[0] == weights[0] + 4000 + max(500, reserve)
    assert required[1] == weights[1] + max(1400, 1024) + reserve


def test_phase_headroom_prices_decode_on_the_primary_and_encode_on_the_te_card():
    split, _ = lsp.plan_layer_split(
        denoisers = {"transformer": L(1000, (1000,) * 23)},
        companions = {"text_encoder": (4000, "text_encoder"), "vae": (300, "other")},
        primary = 0,
        card_limits = {0: 20000, 1: 20000},
        headroom_mib = 0,
        overhead_mib = 500,
        phase_headroom_mib = (3000, 1500, 2500),
    )
    weights = dict(split.card_weights_mib)
    required = dict(split.card_required_mib)
    reserve = lsp.secondary_reserve_mib()
    assert split.text_encoders_card == 1
    assert required[0] == weights[0] + max(1500, 2500) + max(500, reserve)
    assert required[1] == weights[1] + 3000 + reserve


def test_secondary_is_charged_a_block_working_set_not_the_primary_activation_peak():
    denoisers = {"transformer": L(1500, (1070,) * 56)}
    limits = {0: 80000, 1: 80000}
    split, reason = lsp.plan_layer_split(
        denoisers = denoisers,
        companions = {"vae": (200, "other")},
        primary = 0,
        card_limits = limits,
        headroom_mib = 30000,
        overhead_mib = 2048,
    )
    assert split is not None, reason
    weights = dict(split.card_weights_mib)
    required = dict(split.card_required_mib)
    assert lsp.block_working_set_mib(denoisers, 30000) == 2140
    assert required[1] == weights[1] + 2140 + lsp.secondary_reserve_mib()
    assert required[0] == weights[0] + 30000 + 2048
    assert lsp.block_working_set_mib(denoisers, 1500) == 1500
    assert lsp.block_working_set_mib({"transformer": L(0, (100,))}, 0) == 1024


def test_reserve_follows_comfyui_minimum_inference_memory(monkeypatch):
    monkeypatch.setattr(lsp.sys, "platform", "linux")
    assert lsp.secondary_reserve_mib() == 1200
    monkeypatch.setattr(lsp.sys, "platform", "win32")
    assert lsp.secondary_reserve_mib() == 1500


def test_layouts_count_storage_bytes_so_quantized_blocks_are_smaller():
    class Block(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.w = torch.nn.Parameter(torch.empty(4096, 4096, dtype = dtype), requires_grad = False)

    class Den(torch.nn.Module):
        _no_split_modules = ["Block"]

        def __init__(self, dtype):
            super().__init__()
            self.proj_out = torch.nn.Linear(4096, 1024, dtype = torch.bfloat16)
            self.transformer_blocks = torch.nn.ModuleList(Block(dtype) for _ in range(3))
            self.single_transformer_blocks = torch.nn.ModuleList(Block(dtype) for _ in range(2))

    class Pipe:
        transformer = Den(torch.bfloat16)

    class Quant:
        transformer = Den(torch.int8)

    bf16 = lsp._denoiser_layouts(Pipe())["transformer"]
    int8 = lsp._denoiser_layouts(Quant())["transformer"]
    assert bf16.block_mib == (32,) * 5
    assert int8.block_mib == (16,) * 5
    assert bf16.stem_mib == 9


def test_a_denoiser_without_a_block_list_has_no_layout():
    class Pipe:
        transformer = torch.nn.Linear(4, 4)

    assert lsp._denoiser_layouts(Pipe()) is None


@dataclasses.dataclass
class _Target:
    supports_model_cpu_offload: bool = True
    ordinal: int = 0
    device: str = "cuda"
    backend: str = "cuda"
    is_cuda_torch_device: bool = True


def _memory(free = 24000, total = 24576):
    return dm.DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "discrete_vram", free_mib = free, total_mib = total
    )


def _plan(**kw):
    args = dict(
        target = _Target(),
        device_memory = _memory(),
        model_dense_mib = 24000,
        companion_dense_mib = 4300,
        text_encoder_dense_mib = 4000,
        runtime_headroom_mib = 2000,
    )
    args.update(kw)
    return dm.plan_diffusion_memory(**args)


_BIG = {
    "transformer": (24000, "dit"),
    "text_encoder": (4000, "text_encoder"),
    "vae": (300, "other"),
}


class _Engine:
    def __init__(self, fail = False):
        self.fail = fail
        self.placed = []
        self.undone = 0

    def place(
        self,
        pipe,
        *,
        primary,
        split,
        logger = None,
    ):
        self.placed.append((primary, split))
        if self.fail:
            raise RuntimeError("boom")
        return {}

    def undo(self, pipe):
        self.undone += 1


@pytest.fixture
def engine(monkeypatch):
    eng = _Engine()
    monkeypatch.setattr(lsp, "engine_available", lambda: True)
    monkeypatch.setattr(lsp, "place_layer_split", eng.place)
    monkeypatch.setattr(lsp, "undo_layer_split", eng.undo)
    return eng


def _cards(
    monkeypatch,
    sizes = _BIG,
    sec_limit = 20000,
    layouts = None,
):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: sizes)
    monkeypatch.setattr(mg, "secondary_resident_limit_mib", lambda o, mode: sec_limit)
    monkeypatch.setattr(
        mg, "best_secondary", lambda selection, probes = None: selection.secondaries[0]
    )
    monkeypatch.setattr(
        lsp,
        "_denoiser_layouts",
        lambda pipe: layouts if layouts is not None else {"transformer": L(1000, (1000,) * 23)},
    )


class _Pipe:
    """Weak-referenceable stand-in; the refinement only reads it through the patched size / layout readers."""

    components: dict = {}

    def to(self, *a, **k):
        return self


def _refine(
    plan,
    pipe = None,
    ordinals = (0, 1),
):
    pipe = pipe if pipe is not None else _Pipe()
    with mg.selection_scope(mg.make_selection(0, list(ordinals))):
        return mg.refine_plan_for_multigpu(pipe, plan), pipe


def test_one_card_offload_becomes_a_layer_split(monkeypatch, engine):
    plan = _plan()
    assert lsp.plan_offloads_denoiser(plan)
    _cards(monkeypatch)
    refined, pipe = _refine(plan)
    assert refined is not plan
    assert refined.offload_policy == dm.OFFLOAD_NONE
    assert refined.layer_split is not None and refined.secondary_ordinal is None
    assert refined.fallback_plan is plan
    assert refined.layer_split.cards == (0, 1)
    public = refined.as_public_dict()
    assert public["layer_split"]["cards"] == [0, 1]
    assert "layer_split" not in plan.as_public_dict()
    assert any("layer split" in r for r in refined.reasons)


def test_no_engine_keeps_the_one_card_plan(monkeypatch):
    monkeypatch.setattr(lsp, "engine_available", lambda: False)
    _cards(monkeypatch)
    plan = _plan()
    assert _refine(plan)[0] is plan


def test_no_selection_or_kill_switches_keep_the_plan(monkeypatch, engine):
    _cards(monkeypatch)
    plan = _plan()
    assert mg.refine_plan_for_multigpu(_Pipe(), plan) is plan
    assert _refine(plan, ordinals = (0,))[0] is plan
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "0")
    assert _refine(plan)[0] is plan
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "auto")
    monkeypatch.setenv(mg.MULTIGPU_ENV, "0")
    assert _refine(plan)[0] is plan
    assert engine.placed == []


def test_keep_multiple_models_loaded_gets_no_selection(monkeypatch, engine):
    _cards(monkeypatch)
    plan = _plan()
    monkeypatch.setattr(mg, "models_kept_loaded", lambda: True)
    seen = []

    class Backend:
        @mg.with_load_selection
        def _run_load(self, **kwargs):
            seen.append(mg.current_selection())
            return mg.refine_plan_for_multigpu(_Pipe(), plan)

    assert Backend()._run_load(gpu_ordinal = 0, _gpu_ordinals = [0, 1]) is plan
    assert seen == [None]


def test_a_dit_that_fits_one_card_keeps_todays_plan(monkeypatch, engine):
    small = {
        "transformer": (8000, "dit"),
        "text_encoder": (4000, "text_encoder"),
        "vae": (300, "other"),
    }
    _cards(monkeypatch, sizes = small)
    plan = _plan(model_dense_mib = 8000)
    assert plan.offload_policy == dm.OFFLOAD_NONE
    assert _refine(plan)[0] is plan
    streamed_te = dataclasses.replace(
        _plan(),
        offload_policy = dm.OFFLOAD_GROUP,
        stream_transformer = False,
        stream_text_encoders = True,
    )
    _cards(monkeypatch, sec_limit = 1000)  # no companion assignment either
    assert _refine(streamed_te)[0] is streamed_te
    assert engine.placed == []


def test_quantized_storage_that_fits_one_card_is_a_companion_split_not_a_layer_split(
    monkeypatch, engine
):
    quant = {
        "transformer": (12000, "dit"),
        "text_encoder": (4000, "text_encoder"),
        "vae": (300, "other"),
    }
    _cards(monkeypatch, sizes = quant)
    refined, _ = _refine(_plan())
    assert refined.layer_split is None
    assert refined.secondary_ordinal == 1


def test_companion_split_is_preferred_over_a_layer_split(monkeypatch, engine):
    fits = {
        "transformer": (14000, "dit"),
        "text_encoder": (8000, "text_encoder"),
        "vae": (300, "other"),
    }
    _cards(monkeypatch, sizes = fits)
    refined, _ = _refine(_plan())
    assert refined.secondary_ordinal == 1 and refined.layer_split is None


def test_cards_too_small_keep_the_offload_plan_and_record_why(monkeypatch, engine):
    _cards(monkeypatch, sec_limit = 6000)
    plan = _plan()
    refined, pipe = _refine(plan)
    assert refined is plan
    status = lsp.layer_split_status(pipe)
    assert "does not fit" in status["declined"]
    assert mg.placement_status(pipe)["layer_split"] == status


def test_rocm_explicit_modes_and_legacy_cpu_offload_keep_the_plan(monkeypatch, engine):
    _cards(monkeypatch)
    plan = _plan()
    monkeypatch.setattr(lsp, "_is_rocm", lambda: True)
    assert _refine(plan)[0] is plan
    monkeypatch.setattr(lsp, "_is_rocm", lambda: False)
    balanced = _plan(requested_mode = "balanced")
    assert _refine(balanced)[0] is balanced
    legacy = _plan(explicit_offload = True)
    assert legacy.estimates.get("explicit_cpu_offload")
    assert _refine(legacy)[0] is legacy
    assert engine.placed == []


def test_force_cuts_a_denoiser_one_card_would_hold(monkeypatch, engine):
    small = {"transformer": (8000, "dit"), "vae": (300, "other")}
    _cards(monkeypatch, sizes = small, layouts = {"transformer": L(500, (500,) * 15)})
    plan = _plan(model_dense_mib = 8000, companion_dense_mib = 300, text_encoder_dense_mib = 0)
    assert plan.offload_policy == dm.OFFLOAD_NONE
    monkeypatch.setenv(lsp.LAYER_SPLIT_ENV, "force")
    refined, _ = _refine(plan)
    assert refined.layer_split is not None and refined.layer_split.cards == (0, 1)


def test_video_two_expert_cut_when_neither_expert_fits_a_card(monkeypatch, engine):
    sizes = {
        "transformer": (28500, "dit"),
        "transformer_2": (28500, "dit"),
        "text_encoder": (6000, "text_encoder"),
        "vae": (300, "other"),
    }
    layouts = {"transformer": L(500, (700,) * 40), "transformer_2": L(500, (700,) * 40)}
    _cards(monkeypatch, sizes = sizes, sec_limit = 34000, layouts = layouts)
    plan = _plan(
        device_memory = _memory(free = 40000, total = 40960),
        model_dense_mib = 57000,
        companion_dense_mib = 6300,
        text_encoder_dense_mib = 6000,
    )
    refined, _ = _refine(plan, ordinals = (0, 1, 2))
    split = refined.layer_split
    assert split is not None
    assert {r.denoiser for r in split.ranges} == {"transformer", "transformer_2"}


def _layer_plan(monkeypatch, engine):
    _cards(monkeypatch)
    refined, pipe = _refine(_plan())
    assert refined.layer_split is not None
    return refined, pipe


def test_apply_records_the_split_and_cfg_parallel_declines(monkeypatch, engine):
    plan, pipe = _layer_plan(monkeypatch, engine)
    assert lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    assert engine.placed[0][0] == "cuda:0" and engine.placed[0][1] is plan.layer_split
    assert lsp.is_layer_split(pipe) and mg.is_split(pipe)
    status = mg.placement_status(pipe)
    assert status["layer_split"]["cards"] == [0, 1]
    assert status["layer_split"]["boundaries"] == 2
    with mg.selection_scope(mg.make_selection(0, [0, 1])):
        assert (
            mg.cfg_parallel_decline(pipe, offload_policy = "none", selection = mg.current_selection())
            == "denoiser is split across cards (layer split)"
        )
    lsp.forget_layer_split(pipe)
    assert not lsp.is_layer_split(pipe)


def test_failed_placement_applies_the_one_card_fallback(monkeypatch, engine):
    plan, pipe = _layer_plan(monkeypatch, engine)
    engine.fail = True
    applied = []
    real = dm.apply_memory_plan

    def spy(p, pl, **kw):
        applied.append(pl)
        if pl is plan:
            return real(p, pl, **kw)
        return pl.offload_policy, False

    monkeypatch.setattr(dm, "apply_memory_plan", spy)
    policy, _ = dm.apply_memory_plan(pipe, plan, device = "cuda", placement_device = "cuda:0")
    assert engine.undone == 1
    assert applied == [plan, plan.fallback_plan]
    assert policy == plan.fallback_plan.offload_policy
    assert dm.applied_plan(pipe, plan) is plan.fallback_plan
    assert not lsp.is_layer_split(pipe)
    assert "placement failed" in lsp.layer_split_status(pipe)["declined"]


def test_video_guard_reads_per_card_floors_from_the_layer_split(monkeypatch, engine):
    from core.inference import video as V

    plan, pipe = _layer_plan(monkeypatch, engine)
    assert V._video_split_card_floors(pipe) is None
    lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    floors = V._video_split_card_floors(pipe)
    split = plan.layer_split
    weights = dict(split.card_weights_mib)
    required = dict(split.card_required_mib)
    reserve = dict(split.card_reserve_mib)
    assert floors == (
        (0, weights[0], True, True, True, None, None),
        (1, weights[1], False, False, True, required[1] - weights[1] - reserve[1], reserve[1]),
    )
    assert reserve[1] > 0


def test_video_guard_prices_a_block_card_by_its_planned_headroom():
    from core.inference import video as V

    cards = ((0, 13300, True, True, True, None), (1, 20000, False, False, True, 3200))
    free = {0: 60000, 1: 23500}
    kw = dict(family = "wan", available_mib = lambda o: free[o], planned_shape = (832, 480, 33))
    assert V.video_split_shortfall_message(cards, width = 832, height = 480, frames = 33, **kw) is None
    message = V.video_split_shortfall_message(cards, width = 832, height = 480, frames = 66, **kw)
    assert message is not None and "cuda:1" in message
    assert V.video_split_shortfall_message(cards, width = 416, height = 240, frames = 17, **kw) is None


def test_video_guard_scales_only_the_working_set_not_the_fixed_reserve():
    from core.inference import video as V

    cards = ((0, 13300, True, True, True, None, None), (1, 20000, False, False, True, 700, 2500))
    free = {0: 60000, 1: 24000}
    kw = dict(family = "wan", available_mib = lambda o: free[o], planned_shape = (832, 480, 33))
    assert V.video_split_shortfall_message(cards, width = 832, height = 480, frames = 66, **kw) is None
    free[1] = 20000 + 700 + 2400
    message = V.video_split_shortfall_message(cards, width = 832, height = 480, frames = 33, **kw)
    assert message is not None and "cuda:1" in message


def test_a_from_pipe_sibling_sees_the_split_through_its_shared_denoiser(monkeypatch, engine):
    from core.inference import diffusion_layer_split as engine_mod

    plan, pipe = _layer_plan(monkeypatch, engine)
    shared = torch.nn.Linear(2, 2)
    pipe.transformer = shared
    lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    sibling = type("Sibling", (), {"transformer": shared})()
    monkeypatch.setattr(engine_mod, "is_layer_split", lambda m: m is shared)
    assert lsp.applied_layer_split(sibling) is plan.layer_split
    assert lsp.secondary_block_cards(sibling) == (1,)
    assert mg.is_split(sibling)
    mg.undo_split(pipe)
    assert lsp.applied_layer_split(pipe) is None and lsp.applied_layer_split(sibling) is None


def test_a_failed_replacement_does_not_report_the_previous_split(monkeypatch, engine):
    plan, pipe = _layer_plan(monkeypatch, engine)
    assert lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    assert lsp.is_layer_split(pipe)
    engine.fail = True
    assert not lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    assert not lsp.is_layer_split(pipe) and not mg.is_split(pipe)
    assert "placement failed" in lsp.layer_split_status(pipe)["declined"]


def test_an_expert_wholly_on_the_primary_is_placed_plainly(monkeypatch):
    import contextlib

    from core.inference import diffusion_fast_load
    from core.inference import diffusion_layer_split as engine_mod

    class Comp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.moved = []

        def to(self, *a, **k):
            self.moved.append(a[0] if a else k.get("device"))
            return self

    class Pipe:
        def __init__(self):
            self.transformer = Comp()
            self.transformer_2 = Comp()
            self.vae = Comp()
            self.components = {
                "transformer": self.transformer,
                "transformer_2": self.transformer_2,
                "vae": self.vae,
            }

    pipe = Pipe()
    split = lsp.LayerSplit(
        primary = 0,
        cards = (0, 1),
        ranges = (
            lsp.BlockRange("transformer", 0, 1, 0, 1),
            lsp.BlockRange("transformer", 1, 2, 1, 1),
            lsp.BlockRange("transformer_2", 0, 2, 0, 2),
        ),
        text_encoders_card = None,
        card_required_mib = ((0, 1), (1, 1)),
        card_limit_mib = ((0, 9), (1, 9)),
    )
    placed = []
    monkeypatch.setattr(lsp, "split_blocks", lambda m: [("b.0", None), ("b.1", None)])
    monkeypatch.setattr(
        engine_mod,
        "place_layer_split",
        lambda p, **kw: placed.append((kw["attr"], kw["assignment"])) or {},
    )
    monkeypatch.setattr(
        diffusion_fast_load, "fast_upload", lambda mods, dev, logger = None: contextlib.nullcontext()
    )
    out = lsp.place_layer_split(pipe, primary = "cuda:0", split = split)
    assert placed == [
        ("transformer", {"b.0": 0, "b.1": 1})
    ]  # only the expert that is cut reaches the engine
    assert out["transformer_2"]["whole_on_primary"] is True
    assert pipe.transformer_2.moved == [torch.device("cuda", 0)]


def test_offload_installers_refuse_or_replace_a_split(monkeypatch):
    from core.inference import diffusion_layer_split as engine_mod

    dit = torch.nn.Linear(2, 2)
    pipe = type("P", (), {"transformer": dit, "components": {"transformer": dit}})()
    monkeypatch.setattr(engine_mod, "is_layer_split", lambda m: m is dit)
    monkeypatch.setattr(dm, "_offload_groups", lambda m: [])
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda p: True)
    assert dm.hook_resident_denoiser(pipe, "cuda:0") is False
    undone = []
    monkeypatch.setattr(lsp, "undo_layer_split", lambda p: undone.append(p))
    dm._apply_group_offload(pipe, "cpu", None)
    assert undone == [pipe]


def test_image_guard_checks_every_other_card_running_blocks(monkeypatch, engine):
    from core.inference import diffusion as D

    plan, pipe = _layer_plan(monkeypatch, engine)
    free = {"mib": 100000}
    monkeypatch.setattr(
        D,
        "reclaimable_snapshot_device_memory",
        lambda target: dm.DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "discrete_vram",
            free_mib = free["mib"],
            total_mib = 102400,
        ),
    )
    kw = {"width": 1024, "height": 1024, "family": "flux2"}
    assert D._guard_layer_cards(pipe, object(), kw) == 0  # not placed: nothing to check
    lsp.apply_layer_split_plan(pipe, plan, placement = "cuda:0")
    assert D._guard_layer_cards(pipe, object(), kw) == 1
    free["mib"] = 10
    with pytest.raises(ValueError, match = "cuda:1"):
        D._guard_layer_cards(pipe, object(), kw)


def test_two_denoisers_that_fit_only_jointly_are_cut_jointly():
    kw = dict(
        companions = {},
        primary = 0,
        card_limits = {0: 3, 1: 3},
        headroom_mib = 0,
        overhead_mib = 0,
        working_set_mib = 0,
        secondary_overhead_mib = 0,
    )
    denoisers = {"transformer": L(0, (1, 1)), "transformer_2": L(0, (1, 2, 1))}
    split, reason = lsp.plan_layer_split(denoisers = denoisers, **kw)
    assert split is not None, reason
    per_card = {}
    for r in split.ranges:
        per_card[r.card] = per_card.get(r.card, 0) + r.mib
    assert per_card[0] <= 3 and per_card[1] <= 3
    for name, layout in denoisers.items():
        mine = sorted((r.start, r.stop) for r in split.ranges if r.denoiser == name)
        assert mine[0][0] == 0 and mine[-1][1] == len(layout.block_mib)  # every block placed once
        assert any(r.card != 0 for r in split.ranges if r.denoiser == name)
    assert lsp._search_cuts([(1, 1), (1, 2, 1)], [2, 2]) is None  # truly infeasible stays declined


def test_dual_denoisers_may_keep_one_wholly_on_the_primary():
    # A=(2,2) fits only wholly on the 4 MiB primary, B=(3,3) only wholly on the 6 MiB secondary: forcing both
    # to cut onto the secondary would decline a placement that fits
    assert lsp._search_cuts([(2, 2), (3, 3)], [4, 6]) == [[0, 2, 2], [0, 0, 2]]
    assert lsp._search_cuts([(2, 2), (3, 3)], [10, 0]) is None  # nothing off the primary: no split
    split, reason = lsp.plan_layer_split(
        denoisers = {"transformer": L(0, (2, 2)), "transformer_2": L(0, (3, 3))},
        companions = {},
        primary = 0,
        card_limits = {0: 4, 1: 6},
        headroom_mib = 0,
        overhead_mib = 0,
        phase_headroom_mib = (0, 0, 0),
        working_set_mib = 0,
        secondary_overhead_mib = 0,
    )
    assert split is not None, reason
    _check(split, {0: 4, 1: 6})
    assert {(r.denoiser, r.card) for r in split.ranges} == {
        ("transformer", 0),
        ("transformer_2", 1),
    }


def test_failed_placement_drops_fused_records_left_on_another_card(monkeypatch):
    # the fused QKV owned the shared allocation on the card it was built on; once its parts moved, pipe.to("cpu")
    # cannot reach it, so the one-card fallback would OOM beside it
    from types import SimpleNamespace

    from torch import nn

    def attn(fused_device):
        a = nn.Module()
        a.to_q, a.to_k, a.to_v = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)
        a.__dict__["_unsloth_fused_qkv"] = (
            nn.Linear(4, 12, device = fused_device),
            a.to_q,
            a.to_k,
            a.to_v,
        )
        return a

    dit = nn.Module()
    dit.stale, dit.intact = attn("meta"), attn("cpu")
    pipe = SimpleNamespace(transformer = dit, to = lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(lsp, "place_layer_split", boom)
    monkeypatch.setattr(lsp, "undo_layer_split", lambda p: None)
    assert (
        lsp.apply_layer_split_plan(pipe, SimpleNamespace(layer_split = object()), placement = "cuda:0")
        is False
    )
    assert "_unsloth_fused_qkv" not in dit.stale.__dict__
    assert "_unsloth_fused_qkv" in dit.intact.__dict__  # still on its parts' device: kept


def test_an_arch_pinned_attention_backend_keeps_blocks_off_other_architectures(monkeypatch, engine):
    # flash3 is Hopper only and was gated on the H100 primary; a B200 secondary passes make_selection (higher
    # capability) and would reject the kernel on its first block
    from core.inference.diffusion_attention import ATTENTION_BACKEND_ATTR

    caps = {0: (9, 0), 1: (10, 0), 2: (9, 0)}
    monkeypatch.setattr(mg, "_capability", lambda o: caps.get(int(o)))
    _cards(monkeypatch)
    pipe = _Pipe()
    pipe.transformer = torch.nn.Linear(2, 2)
    setattr(pipe.transformer, ATTENTION_BACKEND_ATTR, "_flash_3_hub")
    plan = _plan()
    refined, _ = _refine(plan, pipe = pipe, ordinals = (0, 1))
    assert refined is plan  # the only secondary is another architecture: no split
    assert "attention backend _flash_3_hub" in lsp._DECLINES[pipe]
    refined, _ = _refine(plan, pipe = pipe, ordinals = (0, 1, 2))
    assert refined.layer_split is not None and 1 not in refined.layer_split.cards
    setattr(
        pipe.transformer, ATTENTION_BACKEND_ATTR, "_native_cudnn"
    )  # runs on every admitted card
    refined, _ = _refine(plan, pipe = pipe, ordinals = (0, 1))
    assert refined.layer_split is not None and 1 in refined.layer_split.cards
