# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Semantic image and video progress replaces successful access-poll lines."""

import pytest

from loggers import media_progress


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **fields):
        self.events.append((event, fields))


@pytest.fixture
def capture(monkeypatch):
    out = _Capture()
    monkeypatch.setattr(media_progress, "logger", out)
    media_progress.reset_media_generation_progress("image")
    media_progress.reset_media_generation_progress("video")
    media_progress.reset_media_load_progress("image")
    media_progress.reset_media_load_progress("video")
    return out


def _progress(
    *,
    active = True,
    phase = None,
    step = 0,
    total = 10,
    fraction = None,
    eta = None,
    error = None,
):
    return {
        "active": active,
        "phase": phase,
        "step": step,
        "total_steps": total,
        "fraction": step / total if fraction is None and total else (fraction or 0.0),
        "eta_seconds": eta,
        "error": error,
    }


def test_image_logs_start_and_ten_percent_milestones(capture):
    for reading in (
        _progress(step = 0, eta = 12.34),
        _progress(step = 0, eta = 12.34),
        _progress(step = 1, eta = 10.02),
        _progress(step = 1, fraction = 0.19),
        _progress(step = 2, eta = 8.0),
    ):
        media_progress.log_media_generation_progress("image", reading)

    assert [event for event, _fields in capture.events] == [
        "image_generation_progress",
        "image_generation_progress",
        "image_generation_progress",
    ]
    assert [fields["percent"] for _event, fields in capture.events] == [0, 10, 20]
    assert capture.events[0][1] == {
        "phase": "denoise",
        "percent": 0,
        "step": 0,
        "total_steps": 10,
        "eta_seconds": 12.3,
    }


def test_generation_phase_changes_log_but_terminal_poll_does_not(capture):
    readings = (
        _progress(phase = "queued", step = 0),
        _progress(phase = "denoise", step = 0),
        _progress(phase = "denoise", step = 5),
        _progress(phase = "export", step = 10),
        _progress(active = False, phase = "completed", step = 10),
        _progress(active = False, phase = "completed", step = 10),
    )
    for reading in readings:
        media_progress.log_media_generation_progress("video", reading)

    assert [fields["phase"] for _event, fields in capture.events] == [
        "queued",
        "denoise",
        "denoise",
        "export",
    ]
    assert [fields["percent"] for _event, fields in capture.events] == [0, 0, 50, 100]


def test_failed_generation_poll_is_silent(capture):
    media_progress.log_media_generation_progress(
        "video",
        _progress(
            active = False,
            phase = "failed",
            total = 0,
            error = "Video generation was cancelled.",
        ),
    )
    media_progress.log_media_generation_progress(
        "video",
        _progress(
            active = False,
            phase = "failed",
            total = 0,
            error = "Video generation was cancelled.",
        ),
    )

    assert capture.events == []


def test_idle_image_poll_is_silent_and_next_run_is_detected(capture):
    idle = _progress(active = False, total = 0)
    media_progress.log_media_generation_progress("image", idle)
    assert capture.events == []

    media_progress.log_media_generation_progress("image", _progress(step = 8))
    media_progress.log_media_generation_progress("image", _progress(step = 1))
    assert [fields["percent"] for _event, fields in capture.events] == [80, 10]


def test_a_repeat_run_is_silent_until_the_stream_is_reset(capture):
    # The UI drops its 300 ms poll as soon as the POST settles, so a run's last recorded
    # reading is often its final active one. A next run of the same length whose first poll
    # lands on that step is neither a restart nor a new bucket, so it logs nothing.
    media_progress.log_media_generation_progress("image", _progress(step = 10))
    assert [fields["percent"] for _event, fields in capture.events] == [100]

    capture.events.clear()
    media_progress.log_media_generation_progress("image", _progress(step = 10))
    assert capture.events == []

    media_progress.reset_media_generation_progress("image")
    media_progress.log_media_generation_progress("image", _progress(step = 10))
    assert [fields["percent"] for _event, fields in capture.events] == [100]


def test_load_logs_phase_changes_and_ten_percent_milestones(capture):
    for phase, fraction in (
        ("downloading", 0.0),
        ("downloading", 0.09),
        ("downloading", 0.10),
        ("downloading", 0.19),
        ("downloading", 0.20),
        ("finalizing", 0.20),
        ("finalizing", 1.0),
        ("ready", 1.0),
    ):
        media_progress.log_media_load_progress("image", phase, fraction)

    assert [event for event, _fields in capture.events] == ["diffusion_load_progress"] * 5
    assert [fields for _event, fields in capture.events] == [
        {"phase": "downloading", "percent": 0},
        {"phase": "downloading", "percent": 10},
        {"phase": "downloading", "percent": 20},
        {"phase": "finalizing", "percent": 20},
        {"phase": "finalizing", "percent": 100},
    ]


def test_load_terminal_rearms_the_next_load(capture):
    media_progress.log_media_load_progress("video", "downloading", 0.4)
    media_progress.log_media_load_progress("video", "ready", 1.0)
    media_progress.log_media_load_progress("video", "downloading", 0.0)

    assert [fields["percent"] for _event, fields in capture.events] == [40, 0]


def test_load_progress_sanitizes_ratios_and_never_logs_a_regression(capture):
    media_progress.log_media_load_progress("video", "downloading", float("nan"))
    media_progress.log_media_load_progress("video", "downloading", 0.8)
    media_progress.log_media_load_progress("video", "downloading", 0.2)
    media_progress.log_media_load_progress("video", "downloading", 0.3)

    assert [fields["percent"] for _event, fields in capture.events] == [0, 80]
    assert media_progress.byte_fraction(5, 20) == 0.25
    assert media_progress.byte_fraction(5, 0) == 0.0
    assert media_progress.byte_fraction("bad", 20) == 0.0
