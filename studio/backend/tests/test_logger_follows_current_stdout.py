# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A module logger that first wrote while stdout pointed somewhere else keeps
following stdout. cache_logger_on_first_use freezes the logger, so the stream
must be resolved per record, not when the logger is built."""

import io
import sys

import structlog

from loggers.config import LogConfig, _CurrentStdoutLogger


def test_a_cached_logger_writes_to_the_stdout_of_the_moment(monkeypatch):
    LogConfig.setup_logging(env = "production", quiet_progress_bars = False)
    logger = structlog.get_logger("tests.follows_stdout")

    first, second = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", first)
    logger.info("first-line")  # first use: the bound logger is cached from here on
    monkeypatch.setattr(sys, "stdout", second)
    logger.info("second-line")

    assert "first-line" in first.getvalue() and "second-line" not in first.getvalue()
    assert "second-line" in second.getvalue()


def test_the_factory_hands_out_the_current_stdout_logger():
    LogConfig.setup_logging(env = "production", quiet_progress_bars = False)
    factory = structlog.get_config()["logger_factory"]
    assert isinstance(factory("any", "args"), _CurrentStdoutLogger)


def test_a_missing_stdout_drops_the_record_instead_of_raising(monkeypatch):
    monkeypatch.setattr(sys, "stdout", None)
    _CurrentStdoutLogger().info("nowhere")


def test_a_closed_stdout_still_raises_like_a_print_logger(monkeypatch):
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(sys, "stdout", stream)
    try:
        _CurrentStdoutLogger().info("closed")
    except ValueError:
        return
    raise AssertionError("a closed stream must surface, as PrintLogger does")
