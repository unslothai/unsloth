# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mixed FIFO regression controls; the worker finishes before the first read."""

import threading

import pytest

from core.inference import tool_stream_exec as stream


@pytest.mark.parametrize("size", [0, 3, 8, 9, 100])
@pytest.mark.parametrize("metadata_first", [False, True])
def test_completion_survives_output_backlog(monkeypatch, size, metadata_first):
    monkeypatch.setattr(stream, "TOOL_OUTPUT_STREAM_MAX_CHARS", 8)
    original_start = threading.Thread.start

    def start_and_finish(worker):
        original_start(worker)
        worker.join(timeout = 2)
        assert not worker.is_alive()

    monkeypatch.setattr(threading.Thread, "start", start_and_finish)
    record = {"type": "tool_execution", "verified": True}
    cancel = threading.Event()

    def invoke(output, completion):
        if metadata_first:
            completion(record)
        output("x" * size)
        if not metadata_first:
            completion(record)
        return "final-result"

    gen = stream.stream_tool_execution(
        invoke,
        tool_name = "python",
        cancel_event = cancel,
        launch_event_factory = lambda value: value,
    )
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            assert stop.value == "final-result"
            break
    assert not cancel.is_set()
    assert [event for event in events if event["type"] == "tool_execution"] == [record]
    text = "".join(event["text"] for event in events if event["type"] == "tool_output")
    assert text == "x" * min(size, 8) + (stream._STREAM_CAPPED_NOTICE if size > 8 else "")
    if size:
        assert (events[0] == record) is metadata_first


def test_failed_execution_does_not_fabricate_completion():
    def invoke(output, completion):
        output("before failure")
        raise ValueError("execution failed")

    gen = stream.stream_tool_execution(invoke, tool_name = "python", launch_event_factory = lambda r: r)
    events = []
    with pytest.raises(ValueError, match = "execution failed"):
        while True:
            events.append(next(gen))
    assert all(event["type"] != "tool_execution" for event in events)


@pytest.mark.parametrize("budget", [None, 0, 3])
def test_batch_stops_at_control_before_next_text_even_when_dropping(budget):
    import queue

    pending = []
    fifo = queue.Queue()
    sentinel = object()
    record = {"type": "tool_execution"}
    for item in ["long output", record, "later output", sentinel]:
        fifo.put(item)
    text, done = stream._drain_queue(fifo, sentinel, budget, pending)
    assert not done
    assert pending == [record]
    assert text == ("long output" if budget is None else "long output"[: budget + 1])
    assert fifo.get_nowait() == "later output"
    assert fifo.get_nowait() is sentinel


def test_close_cancels_worker_without_success_record():
    cancel = threading.Event()
    stopped = threading.Event()

    def invoke(output, completion):
        output("started")
        cancel.wait(2)
        stopped.set()
        return "cancelled"

    gen = stream.stream_tool_execution(
        invoke, tool_name = "python", cancel_event = cancel, launch_event_factory = lambda r: r
    )
    assert next(gen)["type"] == "tool_output"
    gen.close()
    assert cancel.is_set()
    assert stopped.wait(2)
