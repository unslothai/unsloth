# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image text encoder partial pinning: where the whole encoder does not fit the host pin budget, the
leading layers that do are pinned and the rest stream pageable (CPU, the pinned streamer faked)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

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
    # each layer holds `mib` MiB of fp32 weights
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
