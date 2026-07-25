# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live tool-output streaming and heartbeats for server-side tool execution.

Covers three invariants:

* ``stream_tool_execution`` yields incremental ``tool_output`` events and
  ``heartbeat`` events while a tool blocks, and returns the tool's result
  byte-identical to a direct call;
* ``_python_exec`` / ``_bash_exec`` produce the same result string with and
  without an ``output_callback`` (the final tool message the model sees is
  untouched by streaming);
* the GGUF agentic loop emits ``tool_output`` between ``tool_start`` and
  ``tool_end`` and feeds the model the same ``role=tool`` message as before.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference.tool_stream_exec import (
    TOOL_OUTPUT_STREAM_MAX_CHARS,
    stream_tool_execution,
)
from core.inference.tools import _bash_exec, _python_exec

from test_llama_cpp_tool_loop import _done, _make_backend, _sse


def _run_stream(invoke, **kwargs):
    """Drive the wrapper generator; return (events, result)."""
    gen = stream_tool_execution(invoke, **kwargs)
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


# ── stream_tool_execution ────────────────────────────────────────


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
        while True:
            event = next(gen)
            events.append(event)
            if len([e for e in events if e["type"] == "heartbeat"]) >= 2:
                release.set()
    except StopIteration as stop:
        result = stop.value
    assert result == "done"
    assert len([e for e in events if e["type"] == "heartbeat"]) >= 2


def test_output_resets_heartbeat_pacing():
    # A steady output stream means no heartbeats are needed.
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
    # Output streamed before the worker raises survives; the exception still propagates.
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
    # gen.close() (SSE client disconnect) sets the shared cancel_event, so a
    # cancel-observing tool returns at once.
    cancel_event = threading.Event()
    started = threading.Event()
    returned = threading.Event()

    def tool(_cb):
        started.set()
        cancel_event.wait(timeout = 5)  # cancel-observing: unblocks on cancel
        returned.set()
        return "cancelled cleanly"

    gen = stream_tool_execution(
        tool,
        tool_name = "web_search",
        cancel_event = cancel_event,
        heartbeat_interval_s = 0.02,
        poll_interval_s = 0.01,
    )
    next(gen)  # prime the worker; returns a heartbeat while the tool blocks
    assert started.wait(timeout = 2)
    gen.close()  # GeneratorExit -> sets cancel_event, then bounded join
    assert cancel_event.is_set()
    assert returned.wait(timeout = 2)  # the tool actually observed cancellation


def test_generator_close_is_bounded_for_cancel_ignoring_tool(monkeypatch):
    # A tool that ignores cancel_event must not stall teardown: gen.close() waits
    # at most the bounded join, not the tool's full runtime.
    monkeypatch.setattr("core.inference.tool_stream_exec._WORKER_JOIN_TIMEOUT_S", 0.2)
    release = threading.Event()

    def tool(_cb):
        # Ignores cancel_event; stands in for a web_search/MCP call that never polls it.
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
    release.set()  # let the daemon worker finish so no sleeper lingers
    assert elapsed < 2.0  # bounded by _WORKER_JOIN_TIMEOUT_S, not the 30s tool


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
    # Repeated start-then-close must not leak worker threads: each cancel-observing
    # worker exits once close() signals it.
    monkeypatch.setattr("core.inference.tool_stream_exec._WORKER_JOIN_TIMEOUT_S", 0.2)

    def _live_tool_workers():
        return [t for t in threading.enumerate() if t.name.startswith("tool-exec-")]

    for _ in range(50):  # let workers from earlier tests drain
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
        gen.close()  # sets cancel_event -> tool returns -> worker exits

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
    assert result == big  # final result untouched by the stream cap
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert len(streamed) < len(big)
    assert "further live output not streamed" in streamed


def test_heartbeats_continue_while_capped_output_flows():
    # After the cap, discarded chunks must not starve the keepalive: a chatty tool
    # keeps the queue non-empty, so without the fix no heartbeat fires and the SSE
    # stream stays silent past proxy idle timeouts.
    release = threading.Event()

    def tool(callback):
        callback("x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 10))  # trip the cap
        while not release.is_set():
            callback("post-cap spam")
            time.sleep(0.005)
        return "done"

    # Watchdog: on regressed code next(gen) blocks forever while spam flows; the
    # timer ends the tool, turning that hang into a clean assertion failure.
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
        while True:
            event = next(gen)
            events.append(event)
            if len([e for e in events if e["type"] == "heartbeat"]) >= 2:
                release.set()
    except StopIteration as stop:
        result = stop.value
    finally:
        release.set()
        watchdog.cancel()
    assert result == "done"
    assert len([e for e in events if e["type"] == "heartbeat"]) >= 2
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert "further live output not streamed" in streamed
    assert "post-cap spam" not in streamed  # cap still enforced


def test_drain_queue_bounds_the_over_cap_batch():
    # _drain_queue stops concatenating once the cap is first exceeded and discards
    # the rest in place, so a chatty tool's huge backlog never defeats the memory ceiling.
    import queue as _queue

    from core.inference.tool_stream_exec import _drain_queue

    q: _queue.Queue = _queue.Queue()
    sentinel = object()
    chunk = "z" * 1000
    for _ in range(5000):  # 5 MB queued ahead of the drain
        q.put(chunk)
    q.put(sentinel)
    text, hit_sentinel = _drain_queue(q, sentinel, max_chars = 100)
    assert hit_sentinel is True
    # At most cap + one chunk is joined, not the full 5 MB backlog.
    assert len(text) <= 100 + len(chunk)
    assert q.empty()  # surplus still drained so completion is detected


def test_drain_queue_does_not_materialize_surplus_crossing_chunk():
    # The single chunk that first crosses the cap must not be materialized in full
    # (a tool can emit one multi-megabyte line). Keep just one char past the budget
    # to preserve the overflow signal and byte-identical truncation, even when the
    # budget is already met (max_chars <= 0).
    import queue as _queue

    from core.inference.tool_stream_exec import _drain_queue

    sentinel = object()
    huge = "z" * 1_000_000

    # Budget already met (non-positive): keep one char, a true prefix.
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

    # Positive cap crossed by one huge chunk: bounded to cap + 1, prefix kept.
    q = _queue.Queue()
    q.put(huge)
    q.put(sentinel)
    text, hit_sentinel = _drain_queue(q, sentinel, max_chars = 100)
    assert len(text) == 101
    assert text == huge[:101]


def test_drain_queue_unbounded_joins_everything():
    # Without a cap the join is complete and ordered (the sub-cap path streams
    # every chunk verbatim on this).
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
    # End-to-end: a burst crossing the cap in one drain still yields a capped live
    # stream and an untouched final result.
    chunk = "z" * 1000

    def tool(callback):
        for _ in range(3000):  # ~3 MB, well past the cap, in one burst
            callback(chunk)
        return "final"

    events, result = _run_stream(tool, tool_name = "python")
    assert result == "final"
    streamed = "".join(e["text"] for e in events if e["type"] == "tool_output")
    assert len(streamed) <= TOOL_OUTPUT_STREAM_MAX_CHARS + len(
        "\n... (further live output not streamed)\n"
    )
    assert "further live output not streamed" in streamed


# ── python / terminal executors ──────────────────────────────────

_PY_CODE = "for i in range(5):\n    print('row', i)\n"


def test_python_exec_result_identical_with_streaming():
    baseline = _python_exec(_PY_CODE, timeout = 60)
    chunks: list[str] = []
    streamed = _python_exec(_PY_CODE, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert "".join(chunks) == "".join(f"row {i}\n" for i in range(5))


def test_python_exec_streams_lines_incrementally():
    # The first of two sleep-separated prints must reach the callback well before exit.
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
    # First line arrived before the sleep completed (margin for slow interpreter start).
    assert first_seen_at[0] - started < finished - started - 0.5


def test_python_exec_unflushed_print_streams_live_and_result_identical():
    # A bare print() WITHOUT flush=True then a sleep. -u forces the child's stdout
    # unbuffered so the line reaches the callback before exit (else CPython
    # block-buffers the pipe and the live pane stays empty). -u changes timing only,
    # so the joined result stays byte-identical to the non-streaming run.
    code = (
        "import time\n"
        "print('progress')\n"  # no flush=True
        "time.sleep(1.0)\n"
        "print('done')\n"
    )
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
    # Unflushed line arrived before the sleep finished: streamed live, not at exit.
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
    # Invalid UTF-8 must not kill either path: the pipe decodes with
    # errors="replace", so the streaming reader thread cannot die on the
    # UnicodeDecodeError readline raises, and both paths return the same replaced text.
    command = "printf 'ok\\377bad\\n'"  # \377 = 0xFF, invalid UTF-8
    baseline = _bash_exec(command, timeout = 60)
    chunks: list[str] = []
    streamed = _bash_exec(command, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert not baseline.startswith("Execution error")
    assert "ok" in baseline and "bad" in baseline
    assert "�" in baseline  # replacement character, not a crash
    assert "".join(chunks) == "ok�bad\n"


def test_bash_exec_unlimited_timeout_waits_for_grandchild_output():
    # A background grandchild holds the pipe open past the shell's exit and writes
    # ~7s later. With timeout=None the drain must wait for EOF like
    # communicate(timeout=None), so the late output is included.
    command = "( sleep 7; echo late-grandchild-output ) & echo parent-done"
    chunks: list[str] = []
    result = _bash_exec(command, timeout = None, output_callback = chunks.append)
    assert "parent-done" in result
    assert "late-grandchild-output" in result
    assert "late-grandchild-output" in "".join(chunks)


def test_bash_exec_finite_timeout_kills_grandchild_holding_stdout(tmp_path):
    # A backgrounded grandchild holds the pipe open past the finite timeout, then
    # would write a sentinel. The parent shell has already exited, so killing only
    # the reaped parent leaves the grandchild running; the drain must kill the
    # process group captured before the wait so the grandchild never writes.
    sentinel = tmp_path / "grandchild_ran"
    command = f"( sleep 3; touch '{sentinel}' ) & echo parent-done"
    result = _bash_exec(command, timeout = 1, output_callback = lambda _t: None)
    assert "timed out" in result
    time.sleep(4.0)  # past the grandchild's 3s sleep
    assert not sentinel.exists(), "grandchild survived the timeout process-group kill"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_bash_exec_nonstreaming_timeout_kills_grandchild(tmp_path):
    # The NON-streaming path (communicate() + _kill_process_tree) short-circuits
    # once the reaped leader has exited, so a stdout-holding grandchild survives
    # unless the group captured right after spawn is killed too. Must match the
    # streaming path's exited-leader handling.
    sentinel = tmp_path / "grandchild_ran"
    command = f"( sleep 3; touch '{sentinel}' ) & echo parent-done"
    result = _bash_exec(command, timeout = 1)  # no output_callback -> communicate path
    assert "timed out" in result
    time.sleep(4.0)
    assert not sentinel.exists(), "non-streaming timeout leaked a stdout-holding grandchild"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_python_exec_nonstreaming_timeout_kills_grandchild(tmp_path):
    sentinel = tmp_path / "grandchild_ran"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['bash', '-c', \"sleep 3; touch '{sentinel}'\"])\n"
        "print('parent-done')\n"
        "import time; time.sleep(30)\n"
    )
    result = _python_exec(code, timeout = 1)  # no output_callback -> communicate path
    assert "timed out" in result
    time.sleep(4.0)
    assert not sentinel.exists(), "non-streaming timeout leaked a stdout-holding grandchild"


def test_drain_process_output_without_posix_process_group_apis(monkeypatch):
    # On Windows os.getpgid / os.killpg are absent; _drain_process_output must not
    # raise AttributeError before reading the child's output. Removing the APIs and
    # flipping os.name: the child still runs and is captured, only the group kill is skipped.
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
    output, timed_out = _drain_process_output(proc, 10, lambda _t: None)
    assert not timed_out
    assert "ok-no-pgid" in output


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_captured_group_survives_fast_leader_reap(tmp_path):
    # Capture the group after spawn, reap the leader first (as a polling cancel
    # watcher would), then drain: the pre-captured pgid must still reap the
    # stdout-holding grandchild even though os.getpgid(pid) would now fail.
    import subprocess as _sp

    from core.inference.tools import _capture_process_group, _drain_process_output

    sentinel = tmp_path / "grandchild_ran"
    proc = _sp.Popen(
        ["bash", "-c", f"( sleep 3; touch '{sentinel}' ) & echo parent-done"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None
    proc.wait()  # reap the leader before draining

    output, timed_out = _drain_process_output(proc, 0.5, None, pgid = pgid)
    assert timed_out
    assert "parent-done" in output
    time.sleep(4.0)
    assert not sentinel.exists(), "pre-captured group failed to reap the grandchild"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_finite_drain_honors_cancel_after_leader_exit(tmp_path):
    # Once the leader exits the cancel watcher (which loops on proc.poll()) is gone,
    # so the finite-timeout drain itself must honor cancellation: a mid-drain
    # cancel_event must break the drain promptly and kill the process group instead
    # of draining a chatty grandchild for the whole large budget.
    import subprocess as _sp
    import threading as _th

    from core.inference.tools import _capture_process_group, _drain_process_output

    sentinel = tmp_path / "grandchild_late"
    # Grandchild holds the pipe open, streams every 0.2s, and touches the sentinel
    # only after 10s -- well past the cancel. The leader exits immediately, so the
    # drain enters the finite branch with a live, chatty reader.
    proc = _sp.Popen(
        [
            "bash",
            "-c",
            "( for i in $(seq 1 100); do echo tick-$i; sleep 0.2; done; "
            f"touch '{sentinel}' ) & echo parent-done",
        ],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None
    proc.wait()  # leader exits at once; the cancel watcher would now be gone

    cancel_event = _th.Event()
    _th.Timer(0.6, cancel_event.set).start()  # cancel shortly into the drain

    started = time.monotonic()
    # Large finite timeout (30s); without the cancel poll the drain keeps reading
    # the grandchild until the pipe closes ~20s later.
    output, timed_out = _drain_process_output(proc, 30, lambda _t: None, cancel_event, pgid = pgid)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"finite drain ignored cancel_event (took {elapsed:.1f}s)"
    # Cancellation is not a timeout: the budget never elapsed.
    assert not timed_out
    assert "parent-done" in output
    time.sleep(11.0)  # past the grandchild's 10s sentinel write
    assert not sentinel.exists(), "cancel did not kill the stdout-holding grandchild group"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_streamed_wait_timeout_kills_grandchild_when_leader_reaped(tmp_path, monkeypatch):
    # The proc.wait() timeout branch normally kills the group via _kill_process_tree.
    # But the leader can exit before _kill_process_tree samples its pgid, which then
    # short-circuits on the reaped leader and leaves a stdout-holding grandchild.
    # Model that race with _kill_process_tree as a no-op; the captured-pgid kill in
    # the timeout branch must still reap the grandchild, matching non-streaming.
    import subprocess as _sp

    from core.inference import tools as _tools_mod
    from core.inference.tools import _capture_process_group, _drain_process_output

    monkeypatch.setattr(_tools_mod, "_kill_process_tree", lambda proc: None)

    sentinel = tmp_path / "grandchild_ran"
    # Leader sleeps past the timeout so proc.wait() genuinely times out; a same-group
    # grandchild holds stdout and would touch the sentinel unless the group is killed.
    proc = _sp.Popen(
        ["bash", "-c", f"( sleep 3; touch '{sentinel}' ) & sleep 30"],
        stdout = _sp.PIPE,
        stderr = _sp.STDOUT,
        text = True,
        preexec_fn = os.setsid,
    )
    pgid = _capture_process_group(proc)
    assert pgid is not None

    output, timed_out = _drain_process_output(proc, 0.5, None, pgid = pgid)
    assert timed_out
    time.sleep(4.0)  # past the grandchild's 3s sleep
    assert not sentinel.exists(), (
        "streamed wait timeout leaked a stdout-holding grandchild when the "
        "process-tree kill short-circuited on the reaped leader"
    )


# ── GGUF loop regression: model-visible messages unchanged ───────


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

    # The role=tool message fed to the model is byte-identical: streaming is purely
    # observational and must not perturb parsing/nudging/healing.
    assert _tool_messages(payloads_streaming) == _tool_messages(payloads_plain)
    assert _tool_messages(payloads_streaming) == [
        {
            "role": "tool",
            "name": "python",
            "content": result_text,
            "tool_call_id": "call_1",
        }
    ]

    # tool_end results match too.
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


# ── result truncation notice, env cap, missing-path healing ──────

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
    # The notice must NOT claim the user saw the output: this wrapper also serves
    # non-streaming callers where no output_callback delivers anything.
    assert "the user was shown the full output" not in out
    assert "shown" not in out
    # Under the limit: untouched.
    assert _truncate("short", limit = 10) == "short"


def test_truncated_result_identical_and_notice_neutral_with_streaming():
    # The truncation notice must be byte-identical with and without an
    # output_callback (the streaming vs non-streaming invariant a mode-dependent
    # notice would break) and must not claim the user was shown the full output.
    code = f"print('x' * {_MAX_OUTPUT_CHARS + 5000})"
    baseline = _python_exec(code, timeout = 60)
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
    assert streamed == baseline
    assert "truncated" in baseline
    assert "the user was shown the full output" not in baseline
    assert "persist in the working directory" in baseline


def test_result_cap_env_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", raising = False)
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "50000")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 50000
    # Garbage and non-positive values fall back to the default.
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "lots")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000
    monkeypatch.setenv("UNSLOTH_TOOL_RESULT_MAX_CHARS", "-5")
    assert _env_int("UNSLOTH_TOOL_RESULT_MAX_CHARS", 16000) == 16000


def test_missing_path_hint_detection():
    err = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    hint = _missing_path_hint(err)
    assert "working directory is writable" in hint
    assert "relative path" in hint
    # The hint echoes the actual failing path, not a canned example.
    assert "'x.html', not '/mnt/data/x.html'" in hint
    # A failure on a local path gets no hint.
    assert _missing_path_hint("FileNotFoundError: 'local.txt'") == ""
    # Mentioning /mnt/data without a file error gets no hint.
    assert _missing_path_hint("saved to /mnt/data, all good") == ""
    assert _missing_path_hint("") == ""


def test_missing_path_hint_generalizes_beyond_convention_prefixes():
    # A hallucinated absolute path outside the enumerated prefixes still earns the
    # hint, echoing that path.
    err = (
        "FileNotFoundError: [Errno 2] No such file or directory: "
        "'/home/ubuntu/Sandbox/flappy_bird.html'"
    )
    hint = _missing_path_hint(err)
    assert "working directory is writable" in hint
    assert "'flappy_bird.html', not '/home/ubuntu/Sandbox/flappy_bird.html'" in hint
    # A bash-style error on an absolute path outside the workdir is echoed too.
    bash_err = "cat: /var/data/report.csv: No such file or directory"
    assert "'report.csv', not '/var/data/report.csv'" in _missing_path_hint(bash_err)


def test_missing_path_hint_respects_project_workdir():
    # Project-backed sessions run under a root OUTSIDE ~/studio_sandbox. A legitimate
    # miss INSIDE that project workspace must not be misclassified as an external
    # habit path and flattened to its basename; judged against the real workdir it
    # gets no hint. The fabricated paths carry no convention prefix, so only the
    # workdir judgement decides.
    workdir = "/srv/projroot/session_area"
    missing = "/srv/projroot/session_area/data/missing.csv"
    output = f"FileNotFoundError: [Errno 2] No such file or directory: '{missing}'"
    # Against the static sandbox root (no workdir) it looks external and wrongly earns the hint.
    assert "working directory is writable" in _missing_path_hint(output)
    # Against the real project workdir it is local -> no hint.
    assert _missing_path_hint(output, workdir) == ""
    # A path genuinely outside the project workdir still earns the hint.
    outside_err = "FileNotFoundError: [Errno 2] No such file or directory: '/srv/other/x.html'"
    assert "working directory is writable" in _missing_path_hint(outside_err, workdir)


def test_missing_path_hint_project_workdir_under_convention_prefix():
    # A project workdir can live under a convention prefix like /workspace (common in
    # containers). A genuine miss INSIDE it carries the "/workspace" substring but is
    # a real local path, not a habit path: the convention fast path must not fire and
    # flatten it to a bare basename (which would drop the project subdirectory).
    workdir = "/workspace/proj"
    nested = "/workspace/proj/sub/data.csv"
    output = f"FileNotFoundError: [Errno 2] No such file or directory: '{nested}'"
    # Against the real project workdir the miss is local -> no hint, so
    # /workspace/proj/sub is not flattened away.
    assert _missing_path_hint(output, workdir) == ""
    # A miss at the project root itself is likewise local.
    at_root = "/workspace/proj/data.csv"
    root_output = f"FileNotFoundError: [Errno 2] No such file or directory: '{at_root}'"
    assert _missing_path_hint(root_output, workdir) == ""
    # A convention path genuinely outside the project workdir still earns the
    # hint (e.g. a /mnt/data habit path with a /workspace-rooted project).
    outside = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    assert "'x.html', not '/mnt/data/x.html'" in _missing_path_hint(outside, workdir)
    # Without an explicit workdir the default sandbox root applies, so a
    # /workspace path is out of sandbox and keeps the habit-path hint.
    assert "working directory is writable" in _missing_path_hint(root_output)


def test_missing_path_hint_convention_scoped_to_failing_line():
    # A convention prefix appearing only OUTSIDE the failing-path line (a traceback
    # frame under /workspace, or the user's code printing /mnt/data) must not trigger
    # the hint when the actual miss was a relative / in-workdir path.
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
    # But a convention path ON the error line still earns the hint.
    on_line = "FileNotFoundError: [Errno 2] No such file or directory: '/mnt/data/x.html'"
    assert "'x.html', not '/mnt/data/x.html'" in _missing_path_hint(on_line)


def test_code_tool_descriptions_mention_relative_paths():
    for tool in (PYTHON_TOOL, TERMINAL_TOOL):
        description = tool["function"]["description"]
        assert "relative paths" in description
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
        _os.remove(target)
        streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
        assert streamed == baseline
        assert _os.path.isfile(target)
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
        _os.remove(target)
        streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
        assert streamed == baseline
        assert _os.path.isfile(target)
    finally:
        if _os.path.exists(target):
            _os.remove(target)


def test_python_exec_hallucinated_absolute_write_is_remapped_into_workdir():
    # The model invents an absolute path outside the enumerated prefixes and opens
    # it for writing; the write-mode fallback redirects it to the basename in the
    # sandbox workdir instead of dying with FileNotFoundError.
    fname = f"remap_{_uuid.uuid4().hex}.html"
    hallucinated = f"/nonexistent_root_xyz/Sandbox/{fname}"
    # Read-back goes through the mapped basename: reads are never redirected, only
    # the write is healed.
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
        _os.remove(target)
        streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)
        assert streamed == baseline
        assert _os.path.isfile(target)
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

    # Normalize each run's random temp filename (byte-identity is per-execution).
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
    # The consumer-side cap only bounds the concatenated stream; a fast worker can
    # still enqueue unboundedly while the SSE consumer is backpressured. The producer
    # boundary now discards callbacks past the cap so the queue cannot grow without
    # limit (finding 12).
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
    # At most cap + 1 chars enter the queue, so 1-char items cannot exceed that
    # regardless of consumer lag.
    assert observed
    assert max(observed) <= TOOL_OUTPUT_STREAM_MAX_CHARS + 2


def test_continuous_over_cap_output_does_not_starve_heartbeats():
    # Once the cap is tripped, a continuously producing tool must not spin the drain
    # forever with no heartbeat: callbacks past the budget never enter the queue, so
    # the idle heartbeat path resumes (finding 13).
    release = threading.Event()

    def tool(callback):
        callback("x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 10))  # trip the cap
        while not release.is_set():
            callback("spam")  # discarded at the producer boundary
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
        while True:
            event = next(gen)
            events.append(event)
            if len([e for e in events if e["type"] == "heartbeat"]) >= 2:
                release.set()
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
    # Uninspectable callables (e.g. some builtins) fall back to not-supported.
    assert accepts_output_callback(len) is False


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_bash_exec_nonstreaming_cancel_kills_grandchild_after_leader_exit(tmp_path):
    # NON-streaming cancellation: the leader exits at once while a grandchild holds
    # stdout. The cancel watcher loops on the leader's poll() and is gone, so before
    # the fix communicate() blocked until the grandchild finished. The unified drain
    # kills the captured group on cancel instead.
    sentinel = tmp_path / "grandchild_ran"
    command = f"( sleep 3; touch '{sentinel}' ) & echo parent-done"
    cancel_event = threading.Event()
    timer = threading.Timer(0.5, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = _bash_exec(command, cancel_event = cancel_event, timeout = 30)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.5
    assert result == "Execution cancelled."
    time.sleep(3.5)
    assert not sentinel.exists(), "non-streaming cancel leaked a stdout-holding grandchild"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
def test_python_exec_nonstreaming_cancel_kills_grandchild_after_leader_exit(tmp_path):
    sentinel = tmp_path / "grandchild_ran"
    code = (
        "import subprocess\n"
        f"subprocess.Popen(['bash', '-c', \"sleep 3; touch '{sentinel}'\"])\n"
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
    assert result == "Execution cancelled."
    time.sleep(3.5)
    assert not sentinel.exists(), "non-streaming cancel leaked a stdout-holding grandchild"
