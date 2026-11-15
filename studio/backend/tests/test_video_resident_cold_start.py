# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stored warm peak must not widen residency on a new process's first render; an OOM with groups resident retries streamed."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

import core.inference.diffusion_memory as dm
import core.inference.video_stream_residency as vr

torch = pytest.importorskip("torch")

KEY = vr.calibration_key(
    "ltx-2.5", "Lightricks/LTX-2.5-Diffusers", "streaming", "cuda:0", revision = "abc"
)
WORK = 768 * 512 * 49


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in (
        vr.VIDEO_DIT_RESIDENT_ENV,
        vr.VIDEO_DIT_RESIDENT_BLOCKS_ENV,
        vr.VIDEO_DIT_RESIDENT_MEASURED_ENV,
        "UNSLOTH_VIDEO_DIT_RESIDENT_WARM_ONLY",
        "UNSLOTH_VIDEO_DIT_RESIDENT_OOM_RETRY",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(vr, "_PEAKS", {KEY: {WORK: 2031}})
    if hasattr(vr, "_WARM"):
        monkeypatch.setattr(vr, "_WARM", set())


def _card(monkeypatch):
    kept = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (7863 << 20, 23034 << 20))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 7000 << 20)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device = None: 9031 << 20)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device = None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(dm, "cuda_reusable_cache_bytes", lambda device = None: 25 << 20)
    monkeypatch.setattr(dm, "_user_budget_reserve_mib", lambda total: 0)
    monkeypatch.setattr(
        dm, "_keep_groups_resident", lambda module, room, dev, logger = None: kept.append(room)
    )
    monkeypatch.setattr(dm, "stream_window_pad_mib", lambda pipe: 0)
    monkeypatch.setattr(dm, "_offload_groups", lambda module: [])
    monkeypatch.setattr(vr, "resident_mib", lambda module: 0)
    monkeypatch.setattr(vr, "room_mib", lambda **kw: 0)
    monkeypatch.setattr(vr, "_open_window", lambda module, dev: dev)
    monkeypatch.setattr(vr, "_failed_captures", lambda: 0)
    monkeypatch.setattr(vr, "_store_key", lambda key: None)
    module = types.SimpleNamespace(_unsloth_video_calibration_key = KEY)
    pipe = types.SimpleNamespace(transformer = module, components = {"transformer": module})
    return pipe, kept


def _fit(pipe):
    import logging
    return vr.fit_for_request(
        pipe,
        device = "cuda:0",
        floor_mib = 8938,
        width = 768,
        height = 512,
        frames = 49,
        logger = logging.getLogger("t"),
    )


def test_a_stored_peak_does_not_widen_the_first_render_of_a_new_process(monkeypatch):
    pipe, kept = _card(monkeypatch)
    _fit(pipe)
    assert kept == []
    assert vr.record_request_peak(pipe) == 2031
    _fit(pipe)
    assert (
        kept and kept[-1] == 7863 + 25 - int(2031 * vr.MEASURED_PEAK_MARGIN) - vr.MEASURED_SLACK_MIB
    )


def test_warm_only_kill_switch_trusts_the_stored_peak_at_once(monkeypatch):
    monkeypatch.setenv("UNSLOTH_VIDEO_DIT_RESIDENT_WARM_ONLY", "0")
    pipe, kept = _card(monkeypatch)
    _fit(pipe)
    assert kept == [7863 + 25 - int(2031 * vr.MEASURED_PEAK_MARGIN) - vr.MEASURED_SLACK_MIB]


def _resident_pipe(monkeypatch, held_mib):
    state = {"held": held_mib}
    monkeypatch.setattr(vr, "resident_mib", lambda module: state["held"])

    def release(
        pipe,
        need,
        logger = None,
        denoisers_only = False,
        reason = "",
        device = None,
    ):
        assert denoisers_only
        state["held"] = 0
        return lambda: None

    monkeypatch.setattr(dm, "release_resident_groups", release)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device = None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device = None: 7000 << 20)
    module = types.SimpleNamespace(
        _unsloth_resident_room = 4529, _unsloth_video_pending = (WORK, 11484 << 20, "cuda:0")
    )
    return types.SimpleNamespace(transformer = module), state


def test_an_oom_with_groups_resident_streams_them_and_rearms_the_measurement(monkeypatch):
    pipe, state = _resident_pipe(monkeypatch, 4484)
    assert vr.may_stream_after_oom(pipe)
    assert vr.stream_after_oom(pipe) is True
    assert state["held"] == 0
    assert pipe.transformer._unsloth_resident_room is None
    assert pipe.transformer._unsloth_video_pending == (WORK, 7000 << 20, "cuda:0")
    assert not vr.may_stream_after_oom(pipe)  # once per request: nothing left to stream


def test_oom_retry_kill_switch_and_a_fully_streamed_render(monkeypatch):
    pipe, _ = _resident_pipe(monkeypatch, 0)
    assert not vr.may_stream_after_oom(pipe)
    assert vr.stream_after_oom(pipe) is False
    pipe, state = _resident_pipe(monkeypatch, 4484)
    monkeypatch.setenv("UNSLOTH_VIDEO_DIT_RESIDENT_OOM_RETRY", "0")
    assert not vr.may_stream_after_oom(pipe)
    assert vr.stream_after_oom(pipe) is False and state["held"] == 4484


def test_video_render_runs_through_the_oom_retry():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    runs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "render_thread"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "video"
    ]
    assert len(runs) == 1
    assert (
        isinstance(runs[0].args[1], ast.Name) and runs[0].args[1].id == "_render_streamed_after_oom"
    )
    retry = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_render_streamed_after_oom"
    )
    body = ast.unparse(retry)
    assert body.index("may_stream_after_oom") < body.index(
        "video_stream_residency.stream_after_oom("
    )
    assert body.index("video_stream_residency.stream_after_oom(") < body.index(
        "manual_seed(int(seed))"
    )
    assert (
        body.index("video_stream_residency.stream_after_oom(")
        < body.index("restart_protect_generation(pipe)")
        < body.rindex("render_call()")
    )


def test_a_retried_render_restarts_the_nvfp4_protect_schedule():
    from core.inference.diffusion_nvfp4_protect import NVFP4StepController

    ctl = NVFP4StepController("0,-1")
    assert ctl.begin(5) == (0, 4)
    ctl.advance()
    ctl.advance()
    ctl.restart()
    seen = [ctl.protected]
    for _ in range(4):
        ctl.advance()
        seen.append(ctl.protected)
    assert seen == [True, False, False, False, True]
    assert ctl.protected_steps_seen == 2
    idle = NVFP4StepController("0,-1")
    idle.restart()  # no generation begun: stays idle
    assert idle.index == 0 and not idle.protected
