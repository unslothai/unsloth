# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streamed tool output and heartbeats must leave the final tool result byte-identical."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest


def _shared_setup_1(events, gen, release):
    while True:
        event = next(gen)
        events.append(event)
        if len([e for e in events if e["type"] == "heartbeat"]) >= 2:
            release.set()


def _shared_setup_2(baseline, code, target):
    _os.remove(target)
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
    assert streamed == baseline
    assert _os.path.isfile(target)


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference.tool_call_parser import BUDGET_EXHAUSTED_NUDGE
from core.inference.tool_stream_exec import (
    TOOL_OUTPUT_STREAM_MAX_CHARS,
    stream_tool_execution,
)
from core.inference.tools import _bash_exec, _python_exec

from test_llama_cpp_tool_loop import _done, _make_backend, _sse


# Grandchild-survival probes: the grandchild spins on a test-owned gate file, so it
# outlives any timeout; a survivor of the group kill writes its sentinel within one
# poll interval of the gate opening, a killed one never can.

_GATE_POLL_S = 0.02
# How long a surviving grandchild is given to notice the gate and write. Two
# orders of magnitude over its own poll interval; a leak shows up in ~20ms.
_LEAK_WINDOW_S = 1.0


def _gated_grandchild_sh(gate: Path, sentinel: Path) -> str:
    """Shell for a grandchild that holds stdout and writes *sentinel* once *gate* exists."""
    return f"while [ ! -f '{gate}' ]; do sleep {_GATE_POLL_S}; done; touch '{sentinel}'"


def _os_isolated_tools() -> bool:
    """Whether PID-namespace teardown reaps descendants when the leader exits."""
    from core.inference import os_sandbox

    capability = os_sandbox.capability_snapshot()
    return capability.available and (
        "detached_processes_die_with_the_call" in capability.limitations
    )


def _assert_grandchild_was_killed(gate: Path, sentinel: Path) -> None:
    """Open the gate, then require the sentinel to stay absent for the whole window."""
    gate.write_text("go")
    deadline = time.monotonic() + _LEAK_WINDOW_S
    while time.monotonic() < deadline:
        assert (
            not sentinel.exists()
        ), "a grandchild survived the process-group kill and wrote its sentinel"
        time.sleep(0.02)


def _run_stream(invoke, **kwargs):
    """Drive the wrapper generator; return (events, result)."""
    gen = stream_tool_execution(invoke, **kwargs)
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


def test_result_returned_verbatim_without_output():
    events, result = _run_stream(
        lambda _cb: "final result",
        tool_name = "web_search",
    )
    assert result == "final result"
    assert [e for e in events if e["type"] == "tool_output"] == []


def test_incremental_output_streams_as_tool_output_events():
    def tool(callback):
        callback("line 1\n")
        callback("line 2\n")
        return "line 1\nline 2\n"

    events, result = _run_stream(tool, tool_name = "python", tool_call_id = "call_1")
    assert result == "line 1\nline 2\n"
    outputs = [e for e in events if e["type"] == "tool_output"]
    assert outputs, "expected tool_output events"
    assert "".join(e["text"] for e in outputs) == "line 1\nline 2\n"
    assert all(e["tool_name"] == "python" for e in outputs)
    assert all(e["tool_call_id"] == "call_1" for e in outputs)


def test_heartbeats_emitted_while_tool_blocks():
    release = threading.Event()

    def tool(_cb):
        release.wait(timeout = 5)
        return "done"

    gen = stream_tool_execution(
        tool,
        tool_name = "web_search",
        heartbeat_interval_s = 0.04,
        poll_interval_s = 0.02,
    )
    events = []
    result = None
    try:
        _shared_setup_1(events, gen, release)
    except StopIteration as stop:
        result = stop.value
    assert result == "done"
    assert len([e for e in events if e["type"] == "heartbeat"]) >= 2


def test_output_resets_heartbeat_pacing():
    def tool(callback):
        for i in range(5):
            callback(f"tick {i}\n")
            time.sleep(0.01)
        return "ok"

    events, result = _run_stream(
        tool,
        tool_name = "python",
        heartbeat_interval_s = 10.0,
        poll_interval_s = 0.02,
    )
    assert result == "ok"
    assert [e for e in events if e["type"] == "heartbeat"] == []


def test_tool_exception_propagates_after_stream():
    def tool(_cb):
        raise RuntimeError("boom")

    gen = stream_tool_execution(tool, tool_name = "python")
    try:
        while True:
            next(gen)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")


def test_output_before_worker_raises_is_preserved():
    def tool(callback):
        callback("partial before crash\n")
        time.sleep(0.02)
        raise RuntimeError("late boom")

    gen = stream_tool_execution(tool, tool_name = "python", poll_interval_s = 0.01)
    events = []
    with pytest.raises(RuntimeError, match = "late boom"):
        while True:
            events.append(next(gen))
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert "partial before crash" in streamed


def test_generator_close_cancels_observing_tool():
    # gen.close() (SSE disconnect) sets the shared cancel_event.
    cancel_event = threading.Event()
    started = threading.Event()
    returned = threading.Event()

    def tool(_cb):
        started.set()
        cancel_event.wait(timeout = 5)
        returned.set()
        return "cancelled cleanly"

    gen = stream_tool_execution(
        tool,
        tool_name = "web_search",
        cancel_event = cancel_event,
        heartbeat_interval_s = 0.02,
        poll_interval_s = 0.01,
    )
    next(gen)
    assert started.wait(timeout = 2)
    gen.close()
    assert cancel_event.is_set()
    assert returned.wait(timeout = 2)


def test_generator_close_is_bounded_for_cancel_ignoring_tool(monkeypatch):
    # A tool that ignores cancel_event must not stall teardown: gen.close() waits
    # at most the bounded join, not the tool's full runtime.
    monkeypatch.setattr("core.inference.tool_stream_exec._WORKER_JOIN_TIMEOUT_S", 0.2)
    release = threading.Event()

    def tool(_cb):
        release.wait(timeout = 30)
        return "slow"

    gen = stream_tool_execution(
        tool,
        tool_name = "web_search",
        cancel_event = threading.Event(),
        heartbeat_interval_s = 0.02,
        poll_interval_s = 0.01,
    )
    next(gen)
    started = time.monotonic()
    gen.close()
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 2.0


def test_cancel_event_not_set_on_clean_finish():
    # cancel_event is shared across a turn; a clean finish must leave it unset so
    # the next tool in the same turn is not aborted.
    cancel_event = threading.Event()

    def tool(_cb):
        return "ok"

    events, result = _run_stream(
        tool,
        tool_name = "python",
        cancel_event = cancel_event,
    )
    assert result == "ok"
    assert not cancel_event.is_set()


def test_no_worker_thread_leak_under_repeated_close(monkeypatch):
    monkeypatch.setattr("core.inference.tool_stream_exec._WORKER_JOIN_TIMEOUT_S", 0.2)

    def _live_tool_workers():
        return [t for t in threading.enumerate() if t.name.startswith("tool-exec-")]

    for _ in range(50):
        if not _live_tool_workers():
            break
        time.sleep(0.02)
    baseline = len(_live_tool_workers())

    for _ in range(60):
        cancel_event = threading.Event()

        def tool(_cb, _ev = cancel_event):
            _ev.wait(timeout = 5)
            return "done"

        gen = stream_tool_execution(
            tool,
            tool_name = "soak",
            cancel_event = cancel_event,
            heartbeat_interval_s = 0.02,
            poll_interval_s = 0.01,
        )
        next(gen)
        gen.close()

    for _ in range(100):
        if len(_live_tool_workers()) <= baseline:
            break
        time.sleep(0.02)
    assert len(_live_tool_workers()) <= baseline


def test_streamed_output_is_capped_but_result_is_not():
    big = "x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 5000)

    def tool(callback):
        callback(big)
        return big

    events, result = _run_stream(tool, tool_name = "python")
    assert result == big
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert len(streamed) < len(big)
    assert "further live output not streamed" in streamed


def test_heartbeats_continue_while_capped_output_flows():
    # After the cap, discarded chunks must not starve the keepalive heartbeat.
    release = threading.Event()

    def tool(callback):
        callback("x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 10))
        while not release.is_set():
            callback("post-cap spam")
            time.sleep(0.005)
        return "done"

    # Watchdog turns a regressed hang into a clean assertion failure.
    watchdog = threading.Timer(8.0, release.set)
    watchdog.start()
    gen = stream_tool_execution(
        tool,
        tool_name = "python",
        heartbeat_interval_s = 0.04,
        poll_interval_s = 0.02,
    )
    events = []
    result = None
    try:
        _shared_setup_1(events, gen, release)
    except StopIteration as stop:
        result = stop.value
    finally:
        release.set()
        watchdog.cancel()
    assert result == "done"
    assert len([e for e in events if e["type"] == "heartbeat"]) >= 2
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert "further live output not streamed" in streamed
    assert "post-cap spam" not in streamed


def test_drain_queue_bounds_the_over_cap_batch():
    # _drain_queue stops concatenating once the cap is first exceeded and discards
    # the rest in place, so a chatty tool's huge backlog never defeats the memory ceiling.
    import queue as _queue

    from core.inference.tool_stream_exec import _drain_queue

    q: _queue.Queue = _queue.Queue()
    sentinel = object()
    chunk = "z" * 1000
    for _ in range(5000):
        q.put(chunk)
    q.put(sentinel)
    text, hit_sentinel = _drain_queue(q, sentinel, max_chars = 100)
    assert hit_sentinel is True
    assert len(text) <= 100 + len(chunk)
    assert q.empty()


def test_drain_queue_does_not_materialize_surplus_crossing_chunk():
    # The chunk that first crosses the cap is kept to one char past the budget, not in full.
    import queue as _queue

    from core.inference.tool_stream_exec import _drain_queue

    sentinel = object()
    huge = "z" * 1_000_000

    for cap in (0, -500):
        q: _queue.Queue = _queue.Queue()
        q.put(huge)
        q.put("more")
        q.put(sentinel)
        text, hit_sentinel = _drain_queue(q, sentinel, max_chars = cap)
        assert hit_sentinel is True
        assert len(text) == 1
        assert huge.startswith(text)
        assert q.empty()

    q = _queue.Queue()
    q.put(huge)
    q.put(sentinel)
    text, hit_sentinel = _drain_queue(q, sentinel, max_chars = 100)
    assert len(text) == 101
    assert text == huge[:101]


def test_drain_queue_unbounded_joins_everything():
    import queue as _queue

    from core.inference.tool_stream_exec import _drain_queue

    q: _queue.Queue = _queue.Queue()
    sentinel = object()
    for i in range(3):
        q.put(f"c{i}")
    q.put(sentinel)
    text, hit_sentinel = _drain_queue(q, sentinel, max_chars = None)
    assert hit_sentinel is True
    assert text == "c0c1c2"


def test_over_cap_crossing_batch_streams_capped_output():
    chunk = "z" * 1000

    def tool(callback):
        for _ in range(3000):
            callback(chunk)
        return "final"

    events, result = _run_stream(tool, tool_name = "python")
    assert result == "final"
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert len(streamed) <= TOOL_OUTPUT_STREAM_MAX_CHARS + len(
        "\n... (further live output not streamed)\n"
    )
    assert "further live output not streamed" in streamed


_PY_CODE = "for i in range(5):\n    print('row', i)\n"


def test_python_exec_result_identical_with_streaming():
    baseline = _python_exec(_PY_CODE, timeout = 60)
    chunks: list[str] = []
    streamed = _python_exec(_PY_CODE, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert "".join(chunks) == "".join(f"row {i}\n" for i in range(5))


def test_python_exec_streams_lines_incrementally():
    code = (
        "import time\n"
        "print('first', flush=True)\n"
        "time.sleep(1.0)\n"
        "print('second', flush=True)\n"
    )
    first_seen_at: list[float] = []

    def on_chunk(_text: str) -> None:
        if not first_seen_at:
            first_seen_at.append(time.monotonic())

    started = time.monotonic()
    result = _python_exec(code, timeout = 60, output_callback = on_chunk)
    finished = time.monotonic()
    assert "first" in result and "second" in result
    assert first_seen_at, "callback never invoked"
    assert first_seen_at[0] - started < finished - started - 0.5


def test_python_exec_unflushed_print_streams_live_and_result_identical():
    # -u forces unbuffered stdout so an unflushed line streams live; timing only.
    code = "import time\nprint('progress')\ntime.sleep(1.0)\nprint('done')\n"
    first_seen_at: list[float] = []

    def on_chunk(_text: str) -> None:
        if not first_seen_at:
            first_seen_at.append(time.monotonic())

    baseline = _python_exec(code, timeout = 60)
    started = time.monotonic()
    streamed = _python_exec(code, timeout = 60, output_callback = on_chunk)
    finished = time.monotonic()
    assert streamed == baseline
    assert "progress" in streamed and "done" in streamed
    assert first_seen_at, "callback never invoked for unflushed print"
    assert first_seen_at[0] - started < finished - started - 0.5


def test_python_exec_error_exit_identical_with_streaming():
    code = "print('before')\nraise SystemExit(3)\n"
    baseline = _python_exec(code, timeout = 60)
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
    assert streamed == baseline
    assert streamed.startswith("Exit code 3:")


def test_python_exec_timeout_message_identical_with_streaming():
    code = "import time\ntime.sleep(30)\n"
    baseline = _python_exec(code, timeout = 1)
    streamed = _python_exec(code, timeout = 1, output_callback = lambda _t: None)
    assert streamed == baseline == "Execution timed out after 1 seconds."


def test_python_exec_timeout_keeps_output_already_printed():
    # The run printed a progress line before it overran the timeout. That text was
    # captured; it must reach the model instead of only the status line.
    code = "import sys, time\nprint('progress')\nsys.stdout.flush()\ntime.sleep(30)\n"
    baseline = _python_exec(code, timeout = 1)
    streamed = _python_exec(code, timeout = 1, output_callback = lambda _t: None)
    assert streamed == baseline
    assert "progress" in baseline
    assert baseline.endswith("Execution timed out after 1 seconds.")


def test_bash_exec_timeout_keeps_output_already_printed():
    command = "echo progress; sleep 30"
    baseline = _bash_exec(command, timeout = 1)
    streamed = _bash_exec(command, timeout = 1, output_callback = lambda _t: None)
    assert streamed == baseline
    assert "progress" in baseline
    assert baseline.endswith("Execution timed out after 1 seconds.")


def test_python_exec_timeout_still_says_so_when_the_output_printed_a_marker():
    from core.inference.tool_loop_controller import strip_result_for_model

    code = "print('__RAG_SOURCES__:[]')\nimport time\ntime.sleep(30)\n"
    result = _python_exec(code, timeout = 1)

    assert result.endswith("Execution timed out after 1 seconds.")
    assert "__RAG_SOURCES__:[]" in result
    assert strip_result_for_model(result, "python") == result


def test_bash_exec_timeout_still_says_so_when_the_output_printed_a_marker():
    from core.inference.tool_loop_controller import strip_result_for_model

    result = _bash_exec("echo '__RAG_SOURCES__:[]'; sleep 30", timeout = 1)

    assert result.endswith("Execution timed out after 1 seconds.")
    assert "__RAG_SOURCES__:[]" in result
    assert strip_result_for_model(result, "terminal") == result


def test_a_truncated_timeout_card_does_not_repeat_the_output():
    # The card keeps the live stream when the result is a prefix of it (`preferFullToolOutput`
    # in tool-output-result.ts), so captured output must LEAD the result.
    code = "print('x' * 200000)\nimport sys, time\nsys.stdout.flush()\ntime.sleep(30)\n"
    result = _python_exec(code, timeout = 1)

    assert "\n\n... (truncated" in result, "not truncated, so nothing was measured"
    body = result.split("\n\n... (truncated")[0]
    assert body.startswith("x"), result[:120]


def test_python_exec_callback_errors_do_not_break_execution():
    def bad_callback(_text: str) -> None:
        raise ValueError("observer bug")

    result = _python_exec("print('ok')", timeout = 60, output_callback = bad_callback)
    assert result.strip() == "ok"


def test_bash_exec_result_identical_with_streaming():
    command = "echo one; echo two"
    baseline = _bash_exec(command, timeout = 60)
    chunks: list[str] = []
    streamed = _bash_exec(command, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert "".join(chunks) == "one\ntwo\n"


def test_bash_exec_invalid_utf8_identical_with_streaming():
    # The pipe decodes with errors="replace" so the streaming reader thread cannot die.
    command = "printf 'ok\\377bad\\n'"  # \377 = 0xFF, invalid UTF-8
    baseline = _bash_exec(command, timeout = 60)
    chunks: list[str] = []
    streamed = _bash_exec(command, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert not baseline.startswith("Execution error")
    assert "ok" in baseline and "bad" in baseline
    assert "�" in baseline
    assert "".join(chunks) == "ok�bad\n"


@pytest.mark.skipif(
    _os_isolated_tools(),
    reason = "a PID namespace kills the background job with the leader; see below",
)
def test_bash_exec_unlimited_timeout_waits_for_grandchild_output():
    # With timeout=None the drain waits for EOF like communicate(), keeping late output.
    command = "( sleep 7; echo late-grandchild-output ) & echo parent-done"
    chunks: list[str] = []
    result = _bash_exec(command, timeout = None, output_callback = chunks.append)
    assert "parent-done" in result
    assert "late-grandchild-output" in result
    assert "late-grandchild-output" in "".join(chunks)


@pytest.mark.skipif(
    not _os_isolated_tools(), reason = "no PID namespace here, so nothing reaps the job"
)
def test_bash_exec_unlimited_timeout_does_not_wait_for_a_job_the_namespace_reaps():
    from core.inference import tools as tools_module

    command = "( sleep 7; echo late-grandchild-output ) & echo parent-done"
    started = time.monotonic()
    result = _bash_exec(command, timeout = None, output_callback = lambda _t: None)
    elapsed = time.monotonic() - started
    assert "parent-done" in result
    if tools_module._last_tool_execution_record.os_isolation:
        assert elapsed < 5
    else:
        assert "late-grandchild-output" in result


def test_bash_exec_finite_timeout_kills_grandchild_holding_stdout(tmp_path):
    # The parent already exited, so the drain must kill the pgid captured before the wait.
    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    command = f"( {_gated_grandchild_sh(gate, sentinel)} ) & echo parent-done"
    result = _bash_exec(command, timeout = 1, output_callback = lambda _t: None)
    if not _os_isolated_tools():
        assert "timed out" in result
    _assert_grandchild_was_killed(gate, sentinel)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_bash_exec_nonstreaming_timeout_kills_grandchild(tmp_path):
    # The non-streaming path must also kill the group captured right after spawn.
    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    command = f"( {_gated_grandchild_sh(gate, sentinel)} ) & echo parent-done"
    result = _bash_exec(command, timeout = 1)
    if not _os_isolated_tools():
        assert "timed out" in result
    _assert_grandchild_was_killed(gate, sentinel)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_python_exec_nonstreaming_timeout_kills_grandchild(tmp_path):
    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['bash', '-c', \"{_gated_grandchild_sh(gate, sentinel)}\"])\n"
        "print('parent-done')\n"
        "import time; time.sleep(30)\n"
    )
    result = _python_exec(code, timeout = 1)
    assert "timed out" in result
    _assert_grandchild_was_killed(gate, sentinel)


def test_drain_process_output_without_posix_process_group_apis(monkeypatch):
    # Windows lacks os.getpgid/os.killpg; the drain must still read output.
    import subprocess as _sp

    from core.inference.tools import _drain_process_output

    monkeypatch.delattr(os, "getpgid", raising = False)
    monkeypatch.delattr(os, "killpg", raising = False)
    monkeypatch.setattr(os, "name", "nt")

    proc = _sp.Popen(
        [sys.executable, "-c", "print('ok-no-pgid')"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
    )
    output, timed_out, _ = _drain_process_output(proc, 10, lambda _t: None)
    assert not timed_out
    assert "ok-no-pgid" in output


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_captured_group_survives_fast_leader_reap(tmp_path):
    # The pre-captured pgid must reap the grandchild even after os.getpgid(pid) would fail.
    import subprocess as _sp

    from core.inference.tools import _capture_process_group, _drain_process_output

    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    proc = _sp.Popen(
        ["bash", "-c", f"( {_gated_grandchild_sh(gate, sentinel)} ) & echo parent-done"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None
    proc.wait()

    output, timed_out, _ = _drain_process_output(proc, 0.5, None, pgid = pgid)
    assert timed_out
    assert "parent-done" in output
    _assert_grandchild_was_killed(gate, sentinel)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_finite_drain_honors_cancel_after_leader_exit(tmp_path):
    # Once the leader exits the cancel watcher is gone, so the drain itself must honor cancel.
    import subprocess as _sp
    import threading as _th

    from core.inference.tools import _capture_process_group, _drain_process_output

    sentinel = tmp_path / "grandchild_late"
    gate = tmp_path / "gate"
    # Gate, not a sleep fuse; the leader exits at once so the drain enters the finite branch.
    proc = _sp.Popen(
        [
            "bash",
            "-c",
            f"( while [ ! -f '{gate}' ]; do echo tick; sleep 0.2; done; "
            f"touch '{sentinel}' ) & echo parent-done",
        ],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None
    proc.wait()

    cancel_event = _th.Event()
    _th.Timer(0.6, cancel_event.set).start()

    started = time.monotonic()
    output, timed_out, _ = _drain_process_output(proc, 30, lambda _t: None, cancel_event, pgid = pgid)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"finite drain ignored cancel_event (took {elapsed:.1f}s)"
    assert not timed_out
    assert "parent-done" in output
    _assert_grandchild_was_killed(gate, sentinel)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_streamed_wait_timeout_kills_grandchild_when_leader_reaped(tmp_path, monkeypatch):
    # _kill_process_tree is a no-op to model it sampling a reaped leader; the
    # captured-pgid kill must still reap the grandchild.
    import subprocess as _sp

    from core.inference import tools as _tools_mod
    from core.inference.tools import _capture_process_group, _drain_process_output

    monkeypatch.setattr(_tools_mod, "_kill_process_tree", lambda proc: None)

    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    proc = _sp.Popen(
        ["bash", "-c", f"( {_gated_grandchild_sh(gate, sentinel)} ) & sleep 30"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None

    output, timed_out, _ = _drain_process_output(proc, 0.5, None, pgid = pgid)
    assert timed_out
    _assert_grandchild_was_killed(gate, sentinel)


def _run_gguf_tool_turn(monkeypatch, fake_execute_tool):
    tool_stream = [
        _sse(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "index": 0,
                        "function": {
                            "name": "python",
                            "arguments": json.dumps({"code": "print('hi')"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": "All done."}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [tool_stream, final_stream], payloads)
    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "run it"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            max_tool_iterations = 1,
        )
    )
    return events, payloads


def test_gguf_loop_final_tool_message_unchanged_by_streaming(monkeypatch):
    result_text = "hi\nline 2\n"

    def plain_tool(name, arguments, **_kwargs):
        return result_text

    def streaming_tool(
        name,
        arguments,
        output_callback = None,
        **_kwargs,
    ):
        if output_callback is not None:
            output_callback("hi\n")
            output_callback("line 2\n")
        return result_text

    events_plain, payloads_plain = _run_gguf_tool_turn(monkeypatch, plain_tool)
    events_streaming, payloads_streaming = _run_gguf_tool_turn(monkeypatch, streaming_tool)

    def _tool_messages(payloads):
        return [
            msg for payload in payloads for msg in payload["messages"] if msg.get("role") == "tool"
        ]

    # Streaming is observational: the model-visible tool message must be byte-identical.
    assert _tool_messages(payloads_streaming) == _tool_messages(payloads_plain)
    # The budget-exhausted instruction rides the last tool result: templates gate replayed
    # reasoning on the newest user turn.
    assert _tool_messages(payloads_streaming) == [
        {
            "role": "tool",
            "name": "python",
            "content": f"{result_text}\n\n{BUDGET_EXHAUSTED_NUDGE}",
            "tool_call_id": "call_1",
        }
    ]

    ends_plain = [e for e in events_plain if e["type"] == "tool_end"]
    ends_streaming = [e for e in events_streaming if e["type"] == "tool_end"]
    assert [e["result"] for e in ends_streaming] == [e["result"] for e in ends_plain]


def test_gguf_loop_emits_tool_output_between_start_and_end(monkeypatch):
    def streaming_tool(
        name,
        arguments,
        output_callback = None,
        **_kwargs,
    ):
        if output_callback is not None:
            output_callback("progress 1\n")
            output_callback("progress 2\n")
        return "progress 1\nprogress 2\n"

    events, _payloads = _run_gguf_tool_turn(monkeypatch, streaming_tool)
    types = [e["type"] for e in events]
    assert "tool_output" in types
    start_idx = types.index("tool_start")
    end_idx = types.index("tool_end")
    output_indices = [i for i, t in enumerate(types) if t == "tool_output"]
    assert all(start_idx < i < end_idx for i in output_indices)
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert streamed == "progress 1\nprogress 2\n"
    for e in events:
        if e["type"] == "tool_output":
            assert e["tool_name"] == "python"
            assert e["tool_call_id"] == "call_1"


def test_gguf_loop_plain_tool_yields_no_tool_output(monkeypatch):
    def plain_tool(name, arguments, **_kwargs):
        return "quiet"

    events, _payloads = _run_gguf_tool_turn(monkeypatch, plain_tool)
    assert [e for e in events if e["type"] == "tool_output"] == []


import os as _os
import uuid as _uuid

from core.inference.tools import (
    PYTHON_TOOL,
    TERMINAL_TOOL,
    _MAX_OUTPUT_CHARS,
    _env_int,
    _missing_path_hint,
    _truncate,
    get_sandbox_workdir,
)


def test_truncate_notice_is_neutral_and_mentions_workdir():
    out = _truncate("y" * 50, limit = 10)
    assert out.startswith("y" * 10)
    assert "truncated" in out and "50 chars total" in out
    assert "persist in the working directory" in out
    # Must not claim the user saw the output: non-streaming callers use this too.
    assert "the user was shown the full output" not in out
    assert "shown" not in out
    assert _truncate("short", limit = 10) == "short"


def test_truncated_result_identical_and_notice_neutral_with_streaming():
    # The notice must be byte-identical with and without an output_callback.
    code = f"print('x' * {_MAX_OUTPUT_CHARS + 5000})"
    baseline = _python_exec(code, timeout = 60)
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
    assert streamed == baseline
    assert "truncated" in baseline
    assert "the user was shown the full output" not in baseline
    assert "persist in the working directory" in baseline


def test_drain_retains_bounded_head_and_tail_of_runaway_output(monkeypatch):
    import subprocess as _sp

    from core.inference import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "_SPILL_MAX_BYTES", 1000)
    monkeypatch.setattr(_tools_mod, "_DRAIN_TAIL_CHARS", 500)
    proc = _sp.Popen(
        [sys.executable, "-c", "for i in range(5000): print(f'{i:09d}')"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
    )
    streamed = []
    output, timed_out, (chars, lines) = _tools_mod._drain_process_output(proc, 30, streamed.append)
    full = "".join(streamed)
    assert not timed_out
    assert len(full) == 50000
    assert len(output) <= 1010 + 510
    assert output.startswith(full[:1000]) and full.endswith(output[-500:])
    assert len(output) + chars == len(full)
    assert output.count("\n") + lines == 5000


def test_drain_bounds_runaway_output_without_newlines(monkeypatch):
    import subprocess as _sp

    from core.inference import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "_SPILL_MAX_BYTES", 1000)
    monkeypatch.setattr(_tools_mod, "_DRAIN_TAIL_CHARS", 500)
    proc = _sp.Popen(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 50000)"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
    )
    streamed = []
    output, timed_out, (chars, lines) = _tools_mod._drain_process_output(proc, 30, streamed.append)
    assert not timed_out
    assert "".join(streamed) == "x" * 50000
    assert len(output) <= 1500 + 500
    assert output == "x" * len(output)
    assert len(output) + chars == 50000
    assert lines == 0


def test_drain_head_memory_tracks_chars_not_line_count(monkeypatch):
    import subprocess as _sp
    import tracemalloc

    from core.inference import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "_SPILL_MAX_BYTES", 1_000_000)
    monkeypatch.setattr(_tools_mod, "_DRAIN_TAIL_CHARS", 500)
    proc = _sp.Popen(
        [sys.executable, "-c", "import sys\nfor _ in range(600000): sys.stdout.write('x\\n')"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
    )
    tracemalloc.start()
    try:
        output, timed_out, (chars, lines) = _tools_mod._drain_process_output(proc, 60, None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not timed_out
    assert len(output) + chars == 1_200_000
    assert output.count("\n") + lines == 600_000
    # ~500k head lines: one str each is ~30 MB, coalesced it is a few copies of the 1 MB head.
    assert peak < 8 * 1024 * 1024, f"drain peaked at {peak / 2**20:.1f} MiB for a 1 MB head"


def test_runaway_output_reports_true_size_and_keeps_trailing_hint(monkeypatch):
    from core.inference import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "_SPILL_MAX_BYTES", 2000)
    monkeypatch.setattr(_tools_mod, "_DRAIN_TAIL_CHARS", 2000)
    code = "for i in range(3000): print('x' * 99)\nopen('/mnt/data/x.html')\n"
    streamed = []
    baseline = _python_exec(code, timeout = 60)
    out = _python_exec(code, timeout = 60, output_callback = streamed.append)
    assert out == baseline
    total = len("Exit code 1:\n") + len("".join(streamed))
    assert f"{total} chars total" in out
    assert "'x.html', not '/mnt/data/x.html'" in out


@pytest.mark.parametrize(
    "run, code",
    [
        (_python_exec, "while True: print('x' * 99)"),
        (_bash_exec, "while :; do printf '%099d\\n' 0; done"),
    ],
    ids = ["python", "bash"],
)
def test_timed_out_runaway_output_reports_true_size_and_line_count(
    monkeypatch, tmp_path, run, code
):
    from core.inference import tools as _tools_mod

    monkeypatch.setattr(_tools_mod, "_SPILL_MAX_BYTES", 2000)
    monkeypatch.setattr(_tools_mod, "_DRAIN_TAIL_CHARS", 500)
    monkeypatch.setattr(_tools_mod, "_get_workdir", lambda _sid = None: str(tmp_path))
    streamed = []
    out = run(
        code,
        timeout = 2,
        session_id = "runaway",
        output_callback = streamed.append,
    )
    full = "".join(streamed)
    assert len(full) > 100_000
    assert f"of {full.count(chr(10)) + 1}, {len(full)} chars total" in out
    assert out.rstrip().endswith("Execution timed out after 2 seconds.")


def test_result_cap_env_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", raising = False)
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "50000")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 50000
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "lots")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "-5")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000


def test_missing_path_hint_detection():
    err = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    hint = _missing_path_hint(err)
    assert "working directory is writable" in hint
    assert "relative path" in hint
    assert "'x.html', not '/mnt/data/x.html'" in hint
    assert _missing_path_hint("FileNotFoundError: 'local.txt'") == ""
    assert _missing_path_hint("saved to /mnt/data, all good") == ""
    assert _missing_path_hint("") == ""


def test_missing_path_hint_generalizes_beyond_convention_prefixes():
    err = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/home/ubuntu/Sandbox/flappy_bird.html'"
    )
    hint = _missing_path_hint(err)
    assert "working directory is writable" in hint
    assert "'flappy_bird.html', not '/home/ubuntu/Sandbox/flappy_bird.html'" in hint
    bash_err = "cat: /var/data/report.csv: No such file or directory"
    assert "'report.csv', not '/var/data/report.csv'" in _missing_path_hint(bash_err)


def test_missing_path_hint_respects_project_workdir():
    # Project sessions run outside ~/studio_sandbox; a miss inside the project is local.
    workdir = "/srv/projroot/session_area"
    missing = "/srv/projroot/session_area/data/missing.csv"
    output = f"FileNotFoundError: [Errno 2] No such file or directory: '{missing}'"
    assert "working directory is writable" in _missing_path_hint(output)
    assert _missing_path_hint(output, workdir) == ""
    outside_err = "FileNotFoundError: [Errno 2] No such file or directory: '/srv/other/x.html'"
    assert "working directory is writable" in _missing_path_hint(outside_err, workdir)


def test_missing_path_hint_project_workdir_under_convention_prefix():
    # A project under a convention prefix like /workspace: an in-project miss must not
    # be flattened to its basename.
    workdir = "/workspace/proj"
    nested = "/workspace/proj/sub/data.csv"
    output = f"FileNotFoundError: [Errno 2] No such file or directory: '{nested}'"
    assert _missing_path_hint(output, workdir) == ""
    at_root = "/workspace/proj/data.csv"
    root_output = f"FileNotFoundError: [Errno 2] No such file or directory: '{at_root}'"
    assert _missing_path_hint(root_output, workdir) == ""
    outside = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    assert "'x.html', not '/mnt/data/x.html'" in _missing_path_hint(outside, workdir)
    assert "working directory is writable" in _missing_path_hint(root_output)


def test_missing_path_hint_convention_scoped_to_failing_line():
    # A convention prefix outside the failing-path line must not trigger the hint.
    frame_err = (
        "Traceback (most recent call last):\n"
        '  File "/workspace/proj/script.py", line 5, in <module>\n'
        "    open('data.csv')\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'"
    )
    assert _missing_path_hint(frame_err) == ""
    printed_err = (
        "outputs go to /mnt/data normally\n"
        "FileNotFoundError: [Errno 2] No such file or directory: 'notes.txt'"
    )
    assert _missing_path_hint(printed_err) == ""
    on_line = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    assert "'x.html', not '/mnt/data/x.html'" in _missing_path_hint(on_line)


def test_code_tool_descriptions_mention_relative_paths():
    for tool in (PYTHON_TOOL, TERMINAL_TOOL):
        description = tool["function"]["description"]
        assert "relative paths" in description
        if sys.platform == "win32":
            # POSIX-only paths read as "you are on Linux" and models refuse Windows programs.
            assert "Windows" in description
        else:
            assert "/mnt/data" in description


def test_python_exec_mnt_data_open_is_remapped_into_workdir():
    # The shim remaps open()/os.makedirs() on /mnt/data into the sandbox CWD and
    # prints a one-line stderr notice, identically with and without streaming.
    fname = f"remap_{_uuid.uuid4().hex}.txt"
    code = (
        "import os\n"
        "os.makedirs('/mnt/data', exist_ok=True)\n"
        f"with open('/mnt/data/{fname}', 'w') as f:\n"
        "    f.write('hello remap')\n"
        f"print(open('/mnt/data/{fname}').read())\n"
    )
    target = _os.path.join(get_sandbox_workdir(), fname)
    try:
        baseline = _python_exec(code, timeout = 60)
        assert _os.path.isfile(target), baseline
        with open(target) as f:
            assert f.read() == "hello remap"
        assert "hello remap" in baseline
        assert "/mnt/data does not exist in this sandbox" in baseline
        _shared_setup_2(baseline, code, target)
    finally:
        if _os.path.exists(target):
            _os.remove(target)


def test_python_exec_pathlib_write_text_is_remapped_into_workdir():
    # pathlib.Path.open / write_text / read_text call io.open directly,
    # bypassing the builtins.open patch, so the shim must remap io.open too.
    fname = f"remap_{_uuid.uuid4().hex}.txt"
    code = (
        "from pathlib import Path\n"
        f"p = Path('/mnt/data/{fname}')\n"
        "p.write_text('pathlib remap')\n"
        "print(p.read_text())\n"
    )
    target = _os.path.join(get_sandbox_workdir(), fname)
    try:
        baseline = _python_exec(code, timeout = 60)
        assert _os.path.isfile(target), baseline
        with open(target) as f:
            assert f.read() == "pathlib remap"
        assert "pathlib remap" in baseline
        assert "/mnt/data does not exist in this sandbox" in baseline
        _shared_setup_2(baseline, code, target)
    finally:
        if _os.path.exists(target):
            _os.remove(target)


def test_python_exec_hallucinated_absolute_write_is_remapped_into_workdir():
    fname = f"remap_{_uuid.uuid4().hex}.html"
    hallucinated = f"/nonexistent_root_xyz/Sandbox/{fname}"
    code = (
        f"with open('{hallucinated}', 'w') as f:\n"
        "    f.write('hello fallback')\n"
        f"print(open('{fname}').read())\n"
    )
    target = _os.path.join(get_sandbox_workdir(), fname)
    try:
        baseline = _python_exec(code, timeout = 60)
        assert _os.path.isfile(target), baseline
        with open(target) as f:
            assert f.read() == "hello fallback"
        assert "hello fallback" in baseline
        assert "does not exist in this sandbox" in baseline
        _shared_setup_2(baseline, code, target)
    finally:
        if _os.path.exists(target):
            _os.remove(target)


def test_python_exec_unremapped_mnt_data_failure_gets_hint():
    # os.listdir is deliberately not remapped: the failure carries the retry hint
    # instead, identically with and without streaming.
    import re as _re

    code = "import os\nos.listdir('/mnt/data/nonexistent_dir_xyz')\n"
    baseline = _python_exec(code, timeout = 60)
    assert "FileNotFoundError" in baseline
    assert "working directory is writable" in baseline
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)

    def normalize(text: str) -> str:
        return _re.sub(r"studio_exec_\w+\.py", "studio_exec.py", text)

    assert normalize(streamed) == normalize(baseline)


def test_bash_exec_missing_path_hint():
    baseline = _bash_exec("cat /mnt/data/definitely_missing.txt", timeout = 60)
    assert "No such file or directory" in baseline
    assert "working directory is writable" in baseline
    streamed = _bash_exec(
        "cat /mnt/data/definitely_missing.txt", timeout = 60, output_callback = lambda _t: None
    )
    assert streamed == baseline


def test_bash_exec_local_failure_gets_no_hint():
    result = _bash_exec("cat definitely_missing_local_file.txt", timeout = 60)
    assert "No such file or directory" in result
    assert "working directory is writable" not in result


def test_producer_queue_is_bounded_under_tight_print_loop(monkeypatch):
    # The producer boundary discards callbacks past the cap so the queue stays bounded.
    import queue as _queue

    from core.inference import tool_stream_exec

    observed = []

    class _TrackingQueue(_queue.Queue):
        def put(self, *args, **kwargs):
            result = super().put(*args, **kwargs)
            observed.append(self.qsize())
            return result

    monkeypatch.setattr(tool_stream_exec.queue, "Queue", _TrackingQueue)

    def tool(callback):
        for _ in range(200_000):
            callback("x")
        return "done"

    events, result = _run_stream(tool, tool_name = "python")
    assert result == "done"
    assert observed
    assert max(observed) <= TOOL_OUTPUT_STREAM_MAX_CHARS + 2


def test_continuous_over_cap_output_does_not_starve_heartbeats():
    # Past the cap, callbacks never enter the queue so the idle heartbeat resumes.
    release = threading.Event()

    def tool(callback):
        callback("x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 10))
        while not release.is_set():
            callback("spam")
        return "done"

    watchdog = threading.Timer(8.0, release.set)
    watchdog.start()
    gen = stream_tool_execution(
        tool,
        tool_name = "python",
        heartbeat_interval_s = 0.04,
        poll_interval_s = 0.02,
    )
    events = []
    result = None
    try:
        _shared_setup_1(events, gen, release)
    except StopIteration as stop:
        result = stop.value
    finally:
        release.set()
        watchdog.cancel()
    assert result == "done"
    assert len([e for e in events if e["type"] == "heartbeat"]) >= 2


def test_accepts_output_callback_signature_detection():
    from core.inference.tool_stream_exec import accepts_output_callback

    def legacy(
        name,
        arguments,
        cancel_event = None,
        timeout = None,
    ):
        return "ok"

    def modern(
        name,
        arguments,
        output_callback = None,
    ):
        return "ok"

    def kwargs_only(name, arguments, **kw):
        return "ok"

    assert accepts_output_callback(legacy) is False
    assert accepts_output_callback(modern) is True
    assert accepts_output_callback(kwargs_only) is True
    assert accepts_output_callback(len) is False


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_bash_exec_nonstreaming_cancel_kills_grandchild_after_leader_exit(tmp_path):
    # Non-streaming cancel: the leader exits at once while a grandchild holds stdout;
    # the drain must kill the captured group.
    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    command = f"( {_gated_grandchild_sh(gate, sentinel)} ) & echo parent-done"
    cancel_event = threading.Event()
    timer = threading.Timer(0.5, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = _bash_exec(command, cancel_event = cancel_event, timeout = 30)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.5
    if not _os_isolated_tools():
        assert result == "Execution cancelled."
    _assert_grandchild_was_killed(gate, sentinel)


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_python_exec_nonstreaming_cancel_kills_grandchild_after_leader_exit(tmp_path):
    sentinel = tmp_path / "grandchild_ran"
    gate = tmp_path / "gate"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['bash', '-c', \"{_gated_grandchild_sh(gate, sentinel)}\"])\n"
        "print('parent-done')\n"
    )
    cancel_event = threading.Event()
    timer = threading.Timer(0.5, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = _python_exec(code, cancel_event = cancel_event, timeout = 30)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.5
    if not _os_isolated_tools():
        assert result == "Execution cancelled."
    _assert_grandchild_was_killed(gate, sentinel)
