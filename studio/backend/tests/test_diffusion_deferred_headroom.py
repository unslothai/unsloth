# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The deferred speed profile declines when the eager peak plus the fused QKV copies would not fit on the card."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion as dmod  # noqa: E402
from core.inference import diffusion_speed as dspeed  # noqa: E402

MIB = 1 << 20
HEADROOM_ENV = "UNSLOTH_DIFFUSION_DEFERRED_HEADROOM"


def _tiny_unet():
    from diffusers import UNet2DConditionModel
    torch.manual_seed(0)
    return UNet2DConditionModel(
        sample_size = 8,
        in_channels = 4,
        out_channels = 4,
        layers_per_block = 1,
        block_out_channels = (32, 64),
        down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim = 24,
        attention_head_dim = 8,
        norm_num_groups = 8,
    )


def _bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _keep_copies(monkeypatch):
    monkeypatch.setenv(dspeed.DROP_UNFUSED_QKV_ENV, "0")


def test_added_bytes_equal_what_diffusers_fuse_adds(monkeypatch):
    _keep_copies(monkeypatch)
    unet = _tiny_unet()
    pipe = types.SimpleNamespace(unet = unet)
    added = dspeed.fused_qkv_added_bytes(pipe)
    before = _bytes(unet)
    unet.fuse_qkv_projections()
    assert added > 0
    assert _bytes(unet) - before == added
    # already fused: nothing more to add
    assert dspeed.fused_qkv_added_bytes(pipe) == 0


def test_no_unet_adds_nothing():
    assert dspeed.fused_qkv_added_bytes(types.SimpleNamespace(transformer = object())) == 0


@pytest.fixture
def card(monkeypatch):
    """A fake CUDA card: ``peak`` / ``free`` / ``reserved`` in MiB."""
    reading = {"peak": 0, "free": 0, "reserved": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev = None: reading["peak"] * MIB)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev = None: (reading["free"] * MIB, 24576 * MIB)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev = None: reading["reserved"] * MIB)
    monkeypatch.delenv(HEADROOM_ENV, raising = False)
    # the fused copies stay beside the originals (the case the gate guards); test_drop_* cover the dropped case
    monkeypatch.setenv(dspeed.DROP_UNFUSED_QKV_ENV, "0")
    return reading


@pytest.fixture
def engage(monkeypatch):
    ran = []
    monkeypatch.setattr(
        dmod.DiffusionBackend, "_engage_deferred_profile", lambda self, state: ran.append(state)
    )
    monkeypatch.setattr(dmod, "invalidate_cfg_replica", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "drop_displaced_cfg_router", lambda *a, **k: None)
    backend = object.__new__(dmod.DiffusionBackend)

    def make(offload_policy = dmod.OFFLOAD_MODEL):
        return types.SimpleNamespace(
            pipe = types.SimpleNamespace(unet = _tiny_unet()),
            speed_deferred = True,
            deferred_static_plan = {"every": 3, "min_steps": 20},
            resolved = {"speed_mode": {"value": "deferred", "reason": "auto"}},
            transformer_cache = None,
            offload_policy = offload_policy,
            device = "cuda:0",
        )

    def run(offload_policy = dmod.OFFLOAD_MODEL, state = None):
        state = state if state is not None else make(offload_policy)
        dmod.DiffusionBackend._engage_deferred_speed(backend, state)
        return state, bool(ran and ran[-1] is state)

    run.make = make
    return run


def test_profile_declined_when_the_render_would_not_fit(card, engage):
    # SDXL 2048^2 at 8 GiB on an L4: eager peak 6197 MiB, 544 MiB free beside a 7234 MiB cache
    # (the tiny U-Net's fused copies are KiB: the peak sits 300 MiB inside capacity, short of the margin)
    card.update(peak = 7778 - 300, free = 544, reserved = 7234)
    state, engaged = engage()
    assert not engaged
    assert state.speed_deferred is False  # never retried on this load
    assert (
        state.deferred_static_plan is None
    )  # no static skip either: renders stay as speed_mode=off
    assert "declined" in state.resolved["speed_mode"]["reason"]


def test_profile_engages_when_the_render_fits(card, engage):
    # SDXL 1024^2 at 8 GiB: eager peak 5232 MiB leaves room for the fused copies
    card.update(peak = 5232, free = 1322, reserved = 6456)
    _state, engaged = engage()
    assert engaged


def test_header_names_the_kill_switch():
    assert dmod.DEFERRED_HEADROOM_ENV == HEADROOM_ENV


def test_kill_switch_engages_regardless(card, engage, monkeypatch):
    card.update(peak = 7700, free = 50, reserved = 7728)
    monkeypatch.setenv(HEADROOM_ENV, "0")
    _state, engaged = engage()
    assert engaged


def test_grouped_offload_is_not_gated(card, engage):
    card.update(peak = 7700, free = 50, reserved = 7728)
    _state, engaged = engage(offload_policy = dmod.OFFLOAD_GROUP)
    assert engaged


def test_unreadable_card_engages(card, engage, monkeypatch):
    def boom(dev = None):
        raise RuntimeError("no reading")

    monkeypatch.setattr(torch.cuda, "mem_get_info", boom)
    card.update(peak = 7700)
    _state, engaged = engage()
    assert engaged


def test_eager_peak_read_at_render_end_survives_a_counter_reset(card, engage):
    # a caller resets the peak counters before the 3rd generate: the eager peak noted as render 2 ended still counts
    state = engage.make()
    card.update(peak = 7778 - 300, free = 544, reserved = 7234)
    dmod._note_deferred_eager_peak(state)
    card["peak"] = 10
    _state, engaged = engage(state = state)
    assert not engaged


def test_render_start_opens_a_fresh_peak_window_without_the_learned_probe(
    card, engage, monkeypatch
):
    # a previous model left a 23 GiB peak in the counters and no learned-peak probe reset them (LEARNED_PEAKS=0)
    from core.inference import diffusion_stable_arena as arena

    resets = []
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda dev = None: card.update(peak = 0))
    monkeypatch.setattr(arena, "reset_peak_mapped", lambda index: resets.append(index))
    state = engage.make()
    card.update(peak = 23 * 1024, free = 16384, reserved = 1024)
    dmod._open_deferred_peak_window(state)
    assert card["peak"] == 0 and resets == [0]
    card["peak"] = 5232  # this load's own eager render
    dmod._note_deferred_eager_peak(state)
    _state, engaged = engage(state = state)
    assert engaged


def test_hires_refinement_peaks_are_noted_too():
    # The refinement pass does not count as a generation, but its (larger) peak must still be noted before the next
    # pass resets the counters, so the note may not sit under the generation-count branch.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(dmod))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "generation_count" in ast.unparse(node.body[0]):
            assert "_note_deferred_eager_peak" not in ast.unparse(node), ast.unparse(node)[:200]
            break
    else:
        pytest.fail("generation_count branch not found")
    assert "_note_deferred_eager_peak(state)" in inspect.getsource(dmod)


def test_drop_makes_the_fuse_add_nothing(monkeypatch):
    # with the unfused copies dropped after the fuse, the fused weights replace the originals
    monkeypatch.delenv(dspeed.DROP_UNFUSED_QKV_ENV, raising = False)
    pipe = types.SimpleNamespace(unet = _tiny_unet())
    assert dspeed.unfused_projections_will_drop(pipe.unet)
    assert dspeed.fused_qkv_added_bytes(pipe) == 0


def test_a_lora_keeps_the_copies_so_they_count(monkeypatch):
    monkeypatch.delenv(dspeed.DROP_UNFUSED_QKV_ENV, raising = False)
    unet = _tiny_unet()
    attn = next(m for m in unet.modules() if callable(getattr(m, "fuse_projections", None)))
    attn.to_q.lora_A = (
        torch.nn.ModuleDict()
    )  # a PEFT adapter on the projections: the drop is skipped
    pipe = types.SimpleNamespace(unet = unet)
    assert not dspeed.unfused_projections_will_drop(unet)
    assert dspeed.fused_qkv_added_bytes(pipe) > 0


def test_drop_lets_the_profile_engage_where_the_copies_would_not_fit(card, engage, monkeypatch):
    # the reading test_profile_declined_when_the_render_would_not_fit declines with the copies kept engages once
    # they are dropped: the fuse then adds nothing to the card
    monkeypatch.delenv(dspeed.DROP_UNFUSED_QKV_ENV, raising = False)
    card.update(peak = 7778 - 300, free = 544, reserved = 7234)
    _state, engaged = engage()
    assert engaged
