# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training progress-log throttle key must be evicted on terminal events.

A long run logs a throttled "Training progress" heartbeat keyed by
("training", job_id). That key has to be released on BOTH complete and error/stop
(not just complete), so a re-run reusing the job_id logs its first heartbeat
immediately and the entry doesn't linger for the process lifetime.
"""

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
if _backend not in sys.path:
    sys.path.insert(0, _backend)

from core.training.training import TrainingBackend
from loggers.progress import progress_throttle


@pytest.fixture(autouse = True)
def _not_verbose(monkeypatch):
    # Verbose makes should_log() always True, which would mask the throttle.
    monkeypatch.delenv("UNSLOTH_STUDIO_VERBOSE", raising = False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def _seed_throttled(job_id: str):
    # First call logs, the immediate repeat is throttled within the 10s window.
    key = ("training", job_id)
    progress_throttle.reset(key)
    assert progress_throttle.should_log(key) is True
    assert progress_throttle.should_log(key) is False
    return key


def test_error_event_resets_progress_throttle():
    b = TrainingBackend()
    b.current_job_id = "job-err"
    key = _seed_throttled("job-err")
    b._handle_event({"type": "error", "error": "boom"})
    # Evicted -> the next run's first heartbeat logs at once.
    assert progress_throttle.should_log(key) is True


def test_complete_event_resets_progress_throttle():
    b = TrainingBackend()
    b.current_job_id = "job-done"
    key = _seed_throttled("job-done")
    b._handle_event({"type": "complete"})
    assert progress_throttle.should_log(key) is True
