# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /api/inference/load-progress throttle: one line per 10% step, reset per load."""

import pytest

import routes.inference as ri


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, kw))


@pytest.fixture
def cap(monkeypatch):
    capture = _Capture()
    monkeypatch.setattr(ri, "logger", capture)
    ri._reset_load_progress_step()
    return capture


def _percents(cap):
    return [kw["percent"] for _event, kw in cap.events]


def test_new_load_first_step_logs_after_reset(cap):
    # Load A reaches 100%.
    ri._log_load_progress_step(1.0, "ready")
    assert _percents(cap) == [100]
    # Same value keeps deduping (steady poll on a finished load stays quiet).
    ri._log_load_progress_step(1.0, "ready")
    assert _percents(cap) == [100]
    # A new load arms the throttle, so a cached load B that reports 100% on its
    # first poll still emits its progress line instead of hitting step == prev.
    ri._reset_load_progress_step()
    ri._log_load_progress_step(1.0, "ready")
    assert _percents(cap) == [100, 100]


def test_steady_poll_dedups_within_a_load(cap):
    for _ in range(3):
        ri._log_load_progress_step(0.3, "mmap")
    assert _percents(cap) == [30]  # one line per 10% step, not one per poll
