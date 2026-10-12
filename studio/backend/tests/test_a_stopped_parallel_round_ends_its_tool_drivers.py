# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A stopped overlapped tool round ends the threads driving its tools.

Each call of a parallel GGUF round is read on a driver thread into a queue. Once the request is
cancelled the tool wrapper heartbeats on every poll, and nothing reads that queue after the
round is gone, so a tool that ignored the cancel kept a driver alive and its queue growing
until it returned. The provider pump stops on the cancel; this driver now does too.
"""

from __future__ import annotations

import threading
import time

from core.inference import llama_cpp as llama_mod
from core.inference.tool_stream_exec import stream_tool_execution


def _blocked_tool(release: threading.Event):
    def invoke(_on_output):
        # Ignores the cancel, as a slow tool can.
        release.wait(10)
        return "done"

    return invoke


def _driver(name: str):
    return next(t for t in threading.enumerate() if t.name == f"tool-drive-{name}")


def _decision(name: str):
    return type("Decision", (), {"tool_name": name})()


def test_a_cancel_ends_the_driver_and_closes_the_stream():
    cancel = threading.Event()
    release = threading.Event()
    name = "slow_search_cancelled"
    stream = stream_tool_execution(
        _blocked_tool(release),
        tool_name = name,
        cancel_event = cancel,
        poll_interval_s = 0.01,
    )
    entry = llama_mod._start_tool_call(_decision(name), stream, [0, 0], [False], cancel_event = cancel)
    driver = _driver(name)
    time.sleep(0.05)
    cancel.set()
    driver.join(1.0)
    try:
        assert not driver.is_alive(), "the driver kept draining a stopped round's tool"
        assert stream.gi_frame is None, "the wrapper was never closed, so the tool was never told"
        out_queue = entry[2]
        kinds = []
        while not out_queue.empty():
            kinds.append(out_queue.get_nowait()[0])
        assert kinds and kinds[-1] == "error", kinds
    finally:
        release.set()


def test_an_uncancelled_call_still_returns_its_result():
    cancel = threading.Event()
    release = threading.Event()
    name = "slow_search_finished"
    stream = stream_tool_execution(
        _blocked_tool(release),
        tool_name = name,
        cancel_event = cancel,
        poll_interval_s = 0.01,
    )
    entry = llama_mod._start_tool_call(_decision(name), stream, [0, 0], [False], cancel_event = cancel)
    release.set()
    _driver_threads = [t for t in threading.enumerate() if t.name == f"tool-drive-{name}"]
    for thread in _driver_threads:
        thread.join(2.0)
    out_queue = entry[2]
    items = []
    while not out_queue.empty():
        items.append(out_queue.get_nowait())
    assert items[-1] == ("result", "done"), items


def test_every_round_launch_hands_the_driver_the_cancel():
    import inspect

    source = inspect.getsource(llama_mod.LlamaCppBackend.generate_chat_completion_with_tools)
    launch = source.index("_pending_calls[_entry_index] = _start_tool_call(")
    assert "cancel_event = cancel_event" in source[launch : launch + 600]
