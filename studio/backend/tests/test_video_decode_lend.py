# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The VAE decode's residency room is lent to the denoise (begin_denoise) and taken back at the decode (prepare_decode)."""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path

import pytest

import core.inference.diffusion_memory as dm
import core.inference.video_stream_residency as vr

torch = pytest.importorskip("torch")

KEY = vr.calibration_key(
    "ltx-2.5", "Lightricks/LTX-2.5-Diffusers", "group", "cuda:0", revision = "abc"
)
W, H, F = 1280, 704, 121
WORK = W * H * F
FREE, CACHE = 4694, 9477
WHOLE_PEAK, DENOISE_PEAK = 7930, 3200


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in (
        vr.VIDEO_DIT_RESIDENT_ENV,
        vr.VIDEO_DIT_RESIDENT_BLOCKS_ENV,
        vr.VIDEO_DIT_RESIDENT_MEASURED_ENV,
        vr.VIDEO_DIT_RESIDENT_WARM_ONLY_ENV,
        vr.VIDEO_DIT_DECODE_LEND_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(vr, "_PEAKS", {KEY: {WORK: WHOLE_PEAK}})
    monkeypatch.setattr(vr, "_WARM", {KEY})
    monkeypatch.setattr(vr, "_PRE_DECODE", {})


class _Card:
    def __init__(self, monkeypatch):
        self.held = 0
        self.allocated = 20000 << 20
        self.peak = self.allocated
        self.resets = 0
        self.kept: list = []
        self.released: list = []
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda device = None: ((FREE - self.held) << 20, 23034 << 20)
        )
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: self.allocated)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device = None: self.peak)
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", self._reset)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: CACHE << 20)
        monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda total: 0)
        monkeypatch.setattr(dm, "_keep_groups_resident", self._keep)
        monkeypatch.setattr(dm, "release_resident_groups", self._release)
        monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
        monkeypatch.setattr(dm, "_offload_groups", lambda module: [])
        monkeypatch.setattr(vr, "resident_mib", lambda module: self.held)
        monkeypatch.setattr(vr, "resident_bytes", lambda module: self.held << 20)
        monkeypatch.setattr(vr, "room_mib", lambda **kw: 0)
        monkeypatch.setattr(vr, "_open_window", lambda module, dev: dev)
        monkeypatch.setattr(vr, "_failed_captures", lambda: 0)
        monkeypatch.setattr(vr, "_store_key", lambda key: None)
        self.module = types.SimpleNamespace(_unsloth_video_calibration_key = KEY)
        self.pipe = types.SimpleNamespace(
            transformer = self.module, components = {"transformer": self.module}
        )

    def _reset(self, device = None):
        self.resets += 1
        self.peak = self.allocated

    def _keep(
        self,
        module,
        room,
        dev,
        logger = None,
    ):
        self.kept.append(room)
        added = max(0, int(room) - self.held)
        self.held += added
        self.allocated += added << 20
        return added

    def _release(
        self,
        pipe,
        need,
        logger = None,
        denoisers_only = False,
        reason = "",
        device = None,
    ):
        assert denoisers_only
        freed = min(self.held, int(need))
        self.held -= freed
        self.allocated -= freed << 20
        self.released.append((freed, reason))
        return lambda: None

    def fit(self):
        return vr.fit_for_request(
            self.pipe,
            device = "cuda:0",
            floor_mib = 9564,
            width = W,
            height = H,
            frames = F,
            logger = logging.getLogger("t"),
        )

    def render(
        self,
        denoise_peak = DENOISE_PEAK,
        decode_peak = WHOLE_PEAK,
    ):
        """One request after the fit: the denoise hook, denoise peak, the decode hook, decode peak, the record."""
        vr.begin_denoise(self.pipe, logger = logging.getLogger("t"))
        self.lent_seen = self.held
        self.peak = self.allocated + (denoise_peak << 20)
        vr.prepare_decode(self.pipe, logger = logging.getLogger("t"))
        self.peak = max(self.peak, self.allocated + (decode_peak << 20))
        return vr.record_request_peak(self.pipe, logger = logging.getLogger("t"))


def _base_room():
    return FREE + CACHE - int(WHOLE_PEAK * vr.MEASURED_PEAK_MARGIN) - vr.MEASURED_SLACK_MIB


def _lend_room():
    return FREE + CACHE - int(DENOISE_PEAK * vr.MEASURED_PEAK_MARGIN) - vr.MEASURED_SLACK_MIB


def test_first_render_only_measures_the_denoise_then_the_next_lends_the_decode_room(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    assert card.kept == [_base_room()]
    assert card.module._unsloth_video_decode_room is None
    assert card.render() == WHOLE_PEAK
    assert card.released == []
    assert vr.pre_decode_extra_mib(card.module, WORK) == DENOISE_PEAK
    card.fit()
    assert card.held == _base_room()
    assert card.module._unsloth_video_decode_room == _base_room()
    assert card.module._unsloth_video_lend == (_lend_room(), DENOISE_PEAK)
    vr.begin_denoise(card.pipe)
    assert card.kept[-1] == _lend_room()
    assert card.held == _lend_room()
    assert card.module._unsloth_resident_room == _lend_room()
    assert vr.begin_denoise(card.pipe) == 0


def test_the_decode_starts_from_the_whole_request_residency_and_records_on_it(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    resets = card.resets
    assert card.render() == WHOLE_PEAK
    lent = card.lent_seen - _base_room()
    assert lent > 0
    # streamed again down to exactly the whole-request room, before the decode allocated anything
    assert card.released == [(lent, "the VAE decode (lent to the denoise)")]
    assert card.held == _base_room()
    assert card.module._unsloth_resident_room == _base_room()
    assert card.resets == resets + 1
    # the stored whole-request peak is the same as without the lend, so the next fit keeps the same decode room
    assert vr._PEAKS[KEY][WORK] == WHOLE_PEAK
    card.fit()
    assert card.module._unsloth_video_decode_room == _base_room()


def test_a_denoise_peak_above_the_decode_peak_still_records_the_larger(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    assert card.render(denoise_peak = DENOISE_PEAK, decode_peak = 1000) == DENOISE_PEAK


def test_no_lend_while_another_render_shares_the_card(monkeypatch):
    """An image render opened its own window on the card after the fit: the denoise keeps the fit's residency."""
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    assert card.module._unsloth_video_lend is not None
    card.module._unsloth_video_window = ("cuda:0", types.SimpleNamespace(overlapped = True))
    assert vr.begin_denoise(card.pipe) == 0
    assert card.held == _base_room()
    assert card.module._unsloth_video_decode_room is None
    assert vr.prepare_decode(card.pipe) == 0
    assert card.released == []


def test_kill_switch_keeps_the_whole_request_room(monkeypatch):
    monkeypatch.setenv(vr.VIDEO_DIT_DECODE_LEND_ENV, "0")
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    assert card.kept == [_base_room()]
    assert card.held == _base_room()
    assert card.module._unsloth_video_decode_room is None
    card.render()
    assert card.released == []


def test_no_lend_when_the_denoise_needs_as_much_as_the_decode(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render(denoise_peak = WHOLE_PEAK)
    card.fit()
    assert card.module._unsloth_video_decode_room is None
    assert card.held == _base_room()


def test_no_lend_beside_live_cuda_graphs(monkeypatch):
    card = _Card(monkeypatch)
    card.pipe._unsloth_cuda_graphs = (object(),)
    card.fit()
    card.render()
    card.fit()
    assert card.module._unsloth_video_decode_room is None
    assert card.held == _base_room()


def test_top_group_only_lends_at_most_the_top_group(monkeypatch):
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_BLOCKS_ENV, "0")
    card = _Card(monkeypatch)
    monkeypatch.setattr(vr, "_top_group_mib", lambda module: 1095)
    card.fit()
    card.render()
    card.fit()
    assert card.held == 1095
    assert card.module._unsloth_video_decode_room is None


def test_prepare_decode_is_a_no_op_without_a_request_and_runs_once_per_request(monkeypatch):
    card = _Card(monkeypatch)
    assert vr.prepare_decode(card.pipe) == 0
    card.fit()
    card.render()
    card.fit()
    vr.begin_denoise(card.pipe)
    card.peak = card.allocated + (DENOISE_PEAK << 20)
    first = vr.prepare_decode(card.pipe)
    assert first > 0
    assert vr.prepare_decode(card.pipe) == 0  # a second decoder (audio) in the same request
    assert len(card.released) == 1


def test_a_discarded_request_drops_its_lend(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    vr.discard_request(card.pipe)
    assert card.module._unsloth_video_decode_room is None
    assert vr.prepare_decode(card.pipe) == 0


def test_the_video_decode_hook_returns_lent_groups():
    """``generate``'s decode-boundary callback calls prepare_decode on the loaded pipe."""
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    hook = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_on_decode"
    )
    calls = [
        n
        for n in ast.walk(hook)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "prepare_decode"
    ]
    assert calls


def test_the_denoise_borrows_no_more_than_the_card_holds_when_the_loop_opens(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device = None: ((FREE - card.held - 2048) << 20, 23034 << 20),
    )
    vr.begin_denoise(card.pipe)
    assert card.held == _lend_room() - 2048


def test_the_denoise_phase_is_measured_above_the_lent_groups(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    card.render(denoise_peak = 2500, decode_peak = 100)
    assert (
        vr.pre_decode_extra_mib(card.module, WORK) == DENOISE_PEAK
    )  # max of 3200 and 2500, not inflated


def test_the_video_denoise_hook_promotes_lent_groups():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    hook = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_on_denoise_entry"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "begin_denoise"
        for n in ast.walk(hook)
    )
    assert any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "_denoise_phase"
        and any(getattr(a, "id", None) == "_on_denoise_entry" for a in n.args)
        for n in ast.walk(tree)
    )


def test_lent_groups_are_plain_tensors_inside_the_renders_inference_mode(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    card.fit()
    seen = []
    keep = card._keep
    monkeypatch.setattr(
        dm,
        "_keep_groups_resident",
        lambda *a, **k: seen.append(torch.is_inference_mode_enabled()) or keep(*a, **k),
    )
    with torch.inference_mode():
        vr.begin_denoise(card.pipe)
    assert seen == [False]


def test_an_overlapped_request_publishes_no_denoise_peak(monkeypatch):
    import core.inference.diffusion_measured_peaks as mp

    card = _Card(monkeypatch)
    card.fit()
    card.module._unsloth_video_window = ("cuda:0", object())
    monkeypatch.setattr(mp, "close_window", lambda device, window: False)
    assert card.render(denoise_peak = 1200) is None
    assert vr.pre_decode_extra_mib(card.module, WORK) is None
    card.fit()
    assert card.module._unsloth_video_lend is None


@pytest.mark.parametrize("pad, stored", [(500, DENOISE_PEAK + 500), (None, None)])
def test_the_denoise_peak_is_stored_on_the_streamed_baseline(monkeypatch, pad, stored):
    card = _Card(monkeypatch)
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: pad)
    card.fit()
    card.render()
    assert vr.pre_decode_extra_mib(card.module, WORK) == stored


def test_measured_off_never_lends(monkeypatch):
    card = _Card(monkeypatch)
    card.fit()
    card.render()
    monkeypatch.setenv(vr.VIDEO_DIT_RESIDENT_MEASURED_ENV, "0")
    assert not vr.decode_lend_enabled()
    card.fit()
    assert card.module._unsloth_video_lend is None
    assert vr.begin_denoise(card.pipe) == 0
