# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Heartbeat throttle for repeated progress logs."""

import pytest

from loggers.progress import ProgressThrottle


@pytest.fixture(autouse = True)
def _not_verbose(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_VERBOSE", raising = False)
    monkeypatch.setenv("LOG_LEVEL", "INFO")


def test_first_message_logs_then_identical_repeats_throttle():
    t = ProgressThrottle(interval_s = 100)  # large window: only the first logs
    k = "load"
    assert t.should_log(k, "Downloading...") is True   # first
    assert t.should_log(k, "Downloading...") is False  # identical repeat within window
    assert t.should_log(k, "Downloading...") is False


def test_message_change_always_logs():
    t = ProgressThrottle(interval_s = 100)
    k = "load"
    assert t.should_log(k, "Loading model...") is True
    assert t.should_log(k, "Loading model...") is False
    assert t.should_log(k, "Importing Unsloth...") is True   # phase change logs


def test_heartbeat_emits_after_interval():
    import time
    t = ProgressThrottle(interval_s = 0.05)
    k = "load"
    assert t.should_log(k, "same") is True
    assert t.should_log(k, "same") is False
    time.sleep(0.06)
    assert t.should_log(k, "same") is True  # interval elapsed -> heartbeat


def test_empty_message_is_pure_time_heartbeat():
    # A step counter changes every tick, so callers pass an empty message and rely
    # only on the interval. Identical empty messages must not all log.
    t = ProgressThrottle(interval_s = 100)
    k = ("training", "job1")
    assert t.should_log(k) is True
    assert t.should_log(k) is False
    assert t.should_log(k) is False


def test_reset_logs_next_immediately():
    t = ProgressThrottle(interval_s = 100)
    k = "load"
    assert t.should_log(k, "x") is True
    assert t.should_log(k, "x") is False
    t.reset(k)
    assert t.should_log(k, "x") is True  # fresh after reset


def test_zero_interval_logs_everything():
    t = ProgressThrottle(interval_s = 0)
    assert all(t.should_log("k", "same") for _ in range(5))


def test_verbose_disables_throttle(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_VERBOSE", "1")
    t = ProgressThrottle(interval_s = 100)
    assert all(t.should_log("k", "same") for _ in range(5))


def test_distinct_keys_are_independent():
    t = ProgressThrottle(interval_s = 100)
    assert t.should_log("a", "m") is True
    assert t.should_log("b", "m") is True   # different key, independent
    assert t.should_log("a", "m") is False
