# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image text encoder partial pinning (CPU, the pinned streamer faked)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for _p in (_BACKEND, _BACKEND / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from core.inference import diffusion_minimax_h3_image as h3i  # noqa: E402

GB = 10**9


def test_partial_pin_count_stops_at_the_budget_less_the_slack():
    sizes = [GB] * 10
    assert h3i.partial_pin_count(sizes, 6 * GB, slack_bytes = GB) == 5
    assert h3i.partial_pin_count(sizes, 6 * GB, reserve_bytes = 2 * GB, slack_bytes = GB) == 3
    assert h3i.partial_pin_count(sizes, None) == 0
    assert h3i.partial_pin_count(sizes, GB // 2, slack_bytes = 0) == 0


def test_partial_pin_count_honours_the_plan_room_unless_pins_replace_counted_copies():
    sizes = [GB] * 10
    assert h3i.partial_pin_count(sizes, 9 * GB, max_pin_bytes = 4 * GB, slack_bytes = GB) == 3
    assert (
        h3i.partial_pin_count(sizes, 9 * GB, max_pin_bytes = 4 * GB, replaced = True, slack_bytes = GB)
        == 8
    )


class _FakePinned:
    def __init__(self, members, resident):
        self.members = list(members)[resident:]
        self.pinned_bytes = sum(m.weight.numel() * 4 for m in self.members)
        self.replaced_bytes = self.pinned_bytes
        self._host = {}
        self.removed = False

    def stats(self):
        return {"groups": len(self.members)}

    def remove(self):
        self.removed = True


def _setup(monkeypatch, budget_mib, seen):
    from core.inference import diffusion_memory, diffusion_offload_prefetch

    monkeypatch.setattr(diffusion_memory, "_pin_budget_mib", lambda: budget_mib)
    monkeypatch.setattr(h3i, "TE_PARTIAL_PIN_SLACK_BYTES", 0)

    def fake_stream(owner, blocks, device, resident, **kw):
        blocks = list(blocks)
        need = sum(m.weight.numel() * 4 for m in blocks[resident:])
        if need + kw.get("reserve_bytes", 0) > budget_mib << 20:
            return None
        seen.append(len(blocks) - resident)
        return _FakePinned(blocks, resident)

    monkeypatch.setattr(diffusion_offload_prefetch, "stream_module_list", fake_stream)


def _layers(n = 8, mib = 1):
    return torch.nn.ModuleList([torch.nn.Linear(256, mib * 1024, bias = False) for _ in range(n)])


def test_an_encoder_over_the_budget_pins_its_leading_layers_and_streams_the_rest(monkeypatch):
    seen = []
    _setup(monkeypatch, budget_mib = 5, seen = seen)
    layers = _layers()
    s = h3i.make_list_streamer(torch.nn.Module(), layers, "cpu", 0, "text_encoder")
    assert isinstance(s, h3i.SplitListStreamer)
    assert seen == [8, 5] or seen == [5]
    assert len(s.parts[0].members) == 5 and s.streamed == 3
    assert s.stats() == {"groups": 5, "pageable": 3}
    s.remove()
    assert s.parts[0].removed


def test_the_kill_switch_and_the_denoiser_keep_all_or_nothing(monkeypatch):
    seen = []
    _setup(monkeypatch, budget_mib = 5, seen = seen)
    assert isinstance(
        h3i.make_list_streamer(torch.nn.Module(), _layers(), "cpu", 0, "transformer"),
        h3i.ModuleListStreamer,
    )
    monkeypatch.setenv(h3i.TE_PARTIAL_PIN_ENV, "0")
    assert isinstance(
        h3i.make_list_streamer(torch.nn.Module(), _layers(), "cpu", 0, "text_encoder"),
        h3i.ModuleListStreamer,
    )


def test_an_encoder_that_fits_is_pinned_whole(monkeypatch):
    seen = []
    _setup(monkeypatch, budget_mib = 64, seen = seen)
    s = h3i.make_list_streamer(torch.nn.Module(), _layers(), "cpu", 0, "text_encoder")
    assert isinstance(s, _FakePinned) and len(s.members) == 8


def test_the_split_leaves_room_for_what_the_load_parks_after_the_encoder(monkeypatch):
    seen = []
    _setup(monkeypatch, budget_mib = 7, seen = seen)
    s = h3i.make_list_streamer(
        torch.nn.Module(), _layers(), "cpu", 0, "text_encoder", later_host_bytes = 3 << 20
    )
    assert isinstance(s, h3i.SplitListStreamer)
    assert len(s.parts[0].members) == 4 and s.streamed == 4


def test_later_host_bytes_cover_the_parked_pieces_and_the_render():
    plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_STREAM, dit_gb = 21.9, te_gb = 27.2, vae_gb = 2.34, vram_gb = 7.6, host_gb = 54.0
    )
    assert h3i.te_later_host_bytes(plan) == 0
    plan.parked = True
    assert (
        h3i.te_later_host_bytes(plan)
        == int(h3i.DIT_REFINER_GB * 1e9) + h3i.TE_PARTIAL_PIN_RENDER_BYTES
    )
    plan.vae_parked = True
    assert (
        h3i.te_later_host_bytes(plan)
        == int((h3i.DIT_REFINER_GB + 2.34) * 1e9) + h3i.TE_PARTIAL_PIN_RENDER_BYTES
    )


def test_a_dit_streamed_again_after_a_replan_keeps_its_pin_over_the_encoder_split(
    tmp_path, monkeypatch
):
    """A re-plan restreaming the DiT after the encoder keeps its pinned streamer: the encoder split leaves its pins."""
    import copy

    import _minimax_h3_image_tiny as tiny
    from core.inference import diffusion_memory, diffusion_offload_prefetch

    root = tmp_path / "rel"
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    scale = 64  # tiny weights in MiB-sized units, the pin budget's granularity
    monkeypatch.setattr(
        diffusion_memory, "_storage_nbytes", lambda t: [t.numel() * t.element_size() * scale]
    )

    def nbytes(modules):
        return sum(
            t.numel() * t.element_size() * scale
            for m in modules
            for t in list(m.parameters()) + list(m.buffers())
        )

    te_bytes = nbytes(te.model.language_model.layers)
    state = {"budget": None, "held": 0}

    def fake_stream(owner, blocks, device, resident, **kw):
        blocks = list(blocks)
        need = nbytes(blocks[resident:])
        if state["budget"] is None:  # early DiT streamer
            state["budget"] = nbytes(blocks) + need + te_bytes // 2
        if need + kw.get("reserve_bytes", 0) > state["budget"] - state["held"]:
            return None
        state["held"] += (
            need  # a removed streamer's pins stay held until its blocks are packed again
        )
        pinned = types.SimpleNamespace(
            members = blocks[resident:], pinned_bytes = need, replaced_bytes = need, _host = {}
        )
        pinned.stats = lambda: {"groups": len(pinned.members)}
        pinned.remove = lambda: None
        return pinned

    monkeypatch.setattr(diffusion_offload_prefetch, "stream_module_list", fake_stream)
    monkeypatch.setattr(
        diffusion_memory,
        "_pin_budget_mib",
        lambda: None if state["budget"] is None else (state["budget"] - state["held"]) >> 20,
    )
    monkeypatch.setattr(h3i, "TE_PARTIAL_PIN_SLACK_BYTES", 0)
    monkeypatch.setattr(h3i, "pin_room_bytes", lambda plan, pinned = (): None)
    real_plan = h3i.plan_placement

    def stream_plan(**kw):
        p = real_plan(**kw)
        p.placement, p.dit_resident_blocks, p.te_resident_layers = h3i.PLACE_STREAM, 1, 0
        p.parked = p.vae_parked = False
        return p

    def replan(
        plan,
        te_gb,
        plan_args,
        slack_gb = 1.0,
    ):
        new = copy.copy(plan)
        new.dit_resident_blocks = 0  # the encoder's real size moved the DiT's resident blocks
        return new

    monkeypatch.setattr(h3i, "plan_placement", stream_plan)
    monkeypatch.setattr(h3i, "replan_for_encoder", replan)
    pipe = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    try:
        roles = {s._unsloth_role: s for s in pipe._unsloth_h3_streamers}
        assert isinstance(roles["transformer"], types.SimpleNamespace)
        assert roles["transformer"]._unsloth_resident == 0
        assert isinstance(roles["text_encoder"], h3i.SplitListStreamer)
        assert 0 < len(roles["text_encoder"].parts[0].members) < len(te.model.language_model.layers)
    finally:
        pipe.release()


def test_only_a_dit_streamed_after_the_encoder_is_reserved_for():
    blocks = _layers(4)
    transformer = types.SimpleNamespace(transformer_blocks = blocks)
    plan = types.SimpleNamespace(placement = h3i.PLACE_STREAM, dit_resident_blocks = 1)
    early = types.SimpleNamespace(_unsloth_resident = 1)
    assert h3i.dit_restream_pin_bytes(transformer, plan, early) == 0  # the early streamer stays
    assert h3i.dit_restream_pin_bytes(transformer, plan, None) == 3 * (1 << 20)
    early._unsloth_resident = 2
    assert h3i.dit_restream_pin_bytes(transformer, plan, early) == 3 * (1 << 20)
    for placement in (h3i.PLACE_RESIDENT, h3i.PLACE_DIT_RESIDENT, h3i.PLACE_SEQUENTIAL):
        plan.placement = placement
        assert h3i.dit_restream_pin_bytes(transformer, plan, None) == 0
