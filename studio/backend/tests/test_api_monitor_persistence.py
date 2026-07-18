# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from core.inference.api_monitor import ApiMonitor


def test_api_monitor_on_finish():
    monitor = ApiMonitor()

    calls = []

    def callback(entry):
        calls.append(entry)

    monitor.on_finish = callback

    eid = monitor.start(endpoint = "/test", method = "POST", model = "llama", prompt = "hello")
    assert len(calls) == 0

    monitor.finish(eid)
    assert len(calls) == 1
    assert calls[0].id == eid
    assert calls[0].status == "completed"

    # Idempotent finish shouldn't double-call
    monitor.finish(eid)
    assert len(calls) == 1


def test_api_monitor_on_fail():
    monitor = ApiMonitor()
    calls = []
    monitor.on_finish = lambda e: calls.append(e)

    eid = monitor.start(endpoint = "/test", method = "POST", model = "llama", prompt = "hello")

    monitor.fail(eid, "error msg")
    assert len(calls) == 1
    assert calls[0].status == "error"
    assert calls[0].error == "error msg"


def test_api_monitor_safe_execution():
    monitor = ApiMonitor()

    def broken_callback(entry):
        raise ValueError("boom")

    monitor.on_finish = broken_callback

    eid = monitor.start(endpoint = "/test", method = "POST", model = "llama", prompt = "hello")

    # Should not raise exception
    monitor.finish(eid)

    entry = monitor.snapshot()[0]
    assert entry["status"] == "completed"
