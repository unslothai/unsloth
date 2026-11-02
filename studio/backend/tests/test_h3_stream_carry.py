# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 (no step callback) drives the cross-forward carry from its scheduler.step wrapper."""

from __future__ import annotations

import os
import sys
import types

import pytest

import core.inference.diffusion_offload_prefetch as op
import core.inference.video as video
from core.inference.video_minimax_h3_residency import H3_STREAM_CARRY_ENV, h3_stream_carry_enabled

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_offload_cross_forward import (  # noqa: E402
    _clean_env,  # noqa: F401 - autouse fixture
    _copies,
    _cpu_prefetcher,
    _forward,
)


@pytest.fixture(autouse = True)
def _no_carry_env(monkeypatch):
    monkeypatch.delenv(H3_STREAM_CARRY_ENV, raising = False)


def _h3_pipe(pf):
    pipe = types.SimpleNamespace(
        components = {"transformer": types.SimpleNamespace(**{op.PREFETCHER_ATTR: pf})},
        _interrupt = False,
    )
    pipe.scheduler = types.SimpleNamespace(step = lambda *a, **k: "stepped")
    return pipe


def _h3_render(
    pf,
    groups,
    steps,
    calls = None,
    cancelled = lambda: False,
):
    """H3's loop: ``steps`` forwards, scheduler.step after all but the last (H3 makes 29 calls for 30 steps)."""
    pipe = _h3_pipe(pf)
    calls = steps - 1 if calls is None else calls
    ticks = []
    done = video._scheduler_step_carry(pipe, steps, ticks.append, cancelled)
    with op.cross_forward_render(pipe):
        with video._scheduler_step_progress(pipe, lambda n: None, done):
            for i in range(steps):
                _forward(pf, groups)
                if i < calls:
                    pipe.scheduler.step()
        decode_view = (dict(pf.ready), set(pf.carried), pf.inflight_bytes)
        hits = op.end_cross_forward(pipe)
    return ticks, decode_view, hits


def test_forwards_after_the_first_read_the_carry_and_nothing_reaches_the_decode():
    pf, groups, log = _cpu_prefetcher("tabcde", depth = 2)
    _forward(pf, groups)
    log.clear()
    ticks, (ready, carried, inflight), hits = _h3_render(pf, groups, steps = 6)
    assert ticks == [1, 2, 3, 4, 5]
    assert pf.stats["carried_hits"] == 10 and hits == 10
    assert not ready and not carried and inflight == 0
    assert pf.stats.get("carried_dropped", 0) == 0
    assert _copies(log) == list("tabcde") * 6


def test_one_call_per_step_also_ends_clean():
    pf, groups, _ = _cpu_prefetcher("tabcde", depth = 2)
    _forward(pf, groups)
    _, (ready, carried, inflight), _ = _h3_render(pf, groups, steps = 5, calls = 5)
    assert not ready and not carried and inflight == 0


def test_too_few_step_calls_are_dropped_at_the_decode():
    pf, groups, _ = _cpu_prefetcher("tabcde", depth = 2)
    _forward(pf, groups)
    pipe = _h3_pipe(pf)
    done = video._scheduler_step_carry(pipe, 10, lambda n: None, lambda: False)
    with op.cross_forward_render(pipe):
        with video._scheduler_step_progress(pipe, lambda n: None, done):
            for i in range(3):
                _forward(pf, groups)
                pipe.scheduler.step()
        assert pf.carried  # a render shorter than its step count would leave these
        op.end_cross_forward(pipe)
        assert not pf.ready and not pf.carried and pf.inflight_bytes == 0 and not pf.armed


def test_cancel_skips_the_carry_step():
    pf, groups, _ = _cpu_prefetcher("tabcde", depth = 2)
    _forward(pf, groups)
    pipe = _h3_pipe(pf)
    seen = []
    done = video._scheduler_step_carry(pipe, 10, seen.append, lambda: True)
    with op.cross_forward_render(pipe):
        _forward(pf, groups)
        done(1)
        assert (
            seen == [1] and pf.armed
        )  # the cancel raises before the next step; render exit disarms
    assert not pf.armed and not pf.carried


@pytest.mark.parametrize(
    "fam_name, streamed, has_cb, env, expected",
    [
        ("minimax-h3", True, False, None, True),
        ("minimax-h3", True, False, "0", False),
        ("minimax-h3", True, True, None, False),
        ("minimax-h3", False, False, None, False),
        ("hunyuanvideo-1.5", True, False, None, False),
    ],
)
def test_gate(monkeypatch, fam_name, streamed, has_cb, env, expected):
    if env is not None:
        monkeypatch.setenv(H3_STREAM_CARRY_ENV, env)
    fam = types.SimpleNamespace(name = fam_name)
    state = types.SimpleNamespace(denoiser_streamed = "stream" if streamed else False)
    assert video._h3_carry_by_scheduler(fam, state, has_cb) is expected


@pytest.mark.parametrize(
    "value, on", [(None, True), ("1", True), ("0", False), ("off", False), ("no", False)]
)
def test_kill_switch(monkeypatch, value, on):
    if value is not None:
        monkeypatch.setenv(H3_STREAM_CARRY_ENV, value)
    assert h3_stream_carry_enabled() is on


def test_kill_switch_render_carries_nothing(monkeypatch):
    monkeypatch.setenv(H3_STREAM_CARRY_ENV, "0")
    pf, groups, log = _cpu_prefetcher("tabcde", depth = 2)
    _forward(pf, groups)
    log.clear()
    pipe = _h3_pipe(pf)
    gate = video._h3_carry_by_scheduler(
        types.SimpleNamespace(name = "minimax-h3"),
        types.SimpleNamespace(denoiser_streamed = "stream"),
        False,
    )
    with op.cross_forward_render(pipe, enabled = gate):
        for _ in range(4):
            _forward(pf, groups)
    assert not gate and not pf.stats.get("carried") and _copies(log) == list("tabcde") * 4


def _top_forward(pf, top, blocks):
    pf.begin()
    pf.onload(top)
    for g in blocks:
        pf.onload(g)
        pf.offload(g)
    pf.offload(top)
    pf.end()


def _top_render(
    keep,
    steps = 6,
    nbytes = 10,
):
    pf, groups, log = _cpu_prefetcher("tabcde", nbytes = nbytes, depth = 2)
    pf.nbytes[id(groups[0])] = 2 * nbytes
    pf.window = 2 * nbytes + 2 * nbytes
    pf.keep_first = keep
    top, blocks = groups[0], groups[1:]
    _top_forward(pf, top, blocks)
    log.clear()
    peak_before = pf.peak_inflight_bytes
    pipe = _h3_pipe(pf)
    done = video._scheduler_step_carry(pipe, steps, lambda n: None, lambda: False)
    with op.cross_forward_render(pipe):
        with video._scheduler_step_progress(pipe, lambda n: None, done):
            for i in range(steps):
                _top_forward(pf, top, blocks)
                if i < steps - 1:
                    pipe.scheduler.step()
        decode_view = (dict(pf.ready), set(pf.carried), pf.inflight_bytes)
        op.end_cross_forward(pipe)
    return pf, log, decode_view, peak_before


def test_streamed_top_group_stays_for_the_next_forward():
    pf, log, (ready, carried, inflight), peak_before = _top_render(keep = True)
    copies = _copies(log)
    assert copies.count("t") == 1 and all(copies.count(b) == 6 for b in "abcde")
    assert pf.stats["kept"] == 5
    assert not ready and not carried and inflight == 0
    assert pf.peak_inflight_bytes <= max(peak_before, pf.window)


def test_without_keep_the_top_group_reuploads_every_forward():
    pf, log, (ready, carried, inflight), _ = _top_render(keep = False)
    assert _copies(log).count("t") == 6 and not pf.stats.get("kept")
    assert not ready and not carried and inflight == 0


def test_kept_top_group_is_released_when_the_render_is_cut_short():
    pf, groups, log = _cpu_prefetcher("tabcde", depth = 2)
    pf.keep_first = True
    top, blocks = groups[0], groups[1:]
    _top_forward(pf, top, blocks)
    pipe = _h3_pipe(pf)
    with pytest.raises(RuntimeError):
        with op.cross_forward_render(pipe):
            pf.armed = True
            _top_forward(pf, top, blocks)
            assert id(top) in pf.ready and id(top) in pf.carried
            raise RuntimeError("cancelled")
    assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
