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
import sys
import threading
import time
from pathlib import Path

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
    # After the stream cap, discarded chunks must not starve the keepalive:
    # a tool that keeps printing keeps the queue non-empty, so no idle poll
    # (and before the fix no heartbeat) would ever fire, leaving the SSE
    # stream silent past proxy idle timeouts for the rest of the run.
    release = threading.Event()

    def tool(callback):
        callback("x" * (TOOL_OUTPUT_STREAM_MAX_CHARS + 10))  # trip the cap
        while not release.is_set():
            callback("post-cap spam")
            time.sleep(0.005)
        return "done"

    # Watchdog: on regressed code the generator never yields while spam
    # flows, so next(gen) would block forever; ending the tool from a timer
    # turns that hang into a clean assertion failure (zero heartbeats).
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


# ── python / terminal executors ──────────────────────────────────

_PY_CODE = "for i in range(5):\n    print('row', i)\n"


def test_python_exec_result_identical_with_streaming():
    baseline = _python_exec(_PY_CODE, timeout = 60)
    chunks: list[str] = []
    streamed = _python_exec(_PY_CODE, timeout = 60, output_callback = chunks.append)
    assert streamed == baseline
    assert "".join(chunks) == "".join(f"row {i}\n" for i in range(5))


def test_python_exec_streams_lines_incrementally():
    # Two prints separated by a sleep: the first line must arrive via the
    # callback well before the process exits.
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
    # The first line arrived before the sleep completed (with margin for a
    # slow interpreter start, assert it beat process completion clearly).
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
    # Invalid UTF-8 in tool output must not kill either path: the pipe
    # decodes with errors="replace" (like _python_exec), so the streaming
    # reader thread cannot die on UnicodeDecodeError (which readline raises
    # as a ValueError subclass and the reader used to swallow, silently
    # truncating output) and both paths return the same replaced text.
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
    # A background grandchild inherits stdout, keeps the pipe open past the
    # main shell's exit, and writes ~7s later, beyond the bounded 5s drain a
    # short-circuiting join would allow. With timeout=None the drain must wait
    # for EOF like communicate(timeout=None), so the late output is included.
    command = "( sleep 7; echo late-grandchild-output ) & echo parent-done"
    chunks: list[str] = []
    result = _bash_exec(command, timeout = None, output_callback = chunks.append)
    assert "parent-done" in result
    assert "late-grandchild-output" in result
    assert "late-grandchild-output" in "".join(chunks)


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

    # The role=tool message fed to the model is byte-identical: streaming is
    # purely observational and must not perturb parsing/nudging/healing.
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
    # The notice must NOT claim the user saw the output: this same wrapper
    # serves non-streaming chat/API and direct execute_tool() callers where no
    # output_callback delivers anything to anyone.
    assert "the user was shown the full output" not in out
    assert "shown" not in out
    # Under the limit: untouched.
    assert _truncate("short", limit = 10) == "short"


def test_truncated_result_identical_and_notice_neutral_with_streaming():
    # A long output crosses the model-visible cap. The truncation notice must be
    # byte-identical with and without an output_callback (the streaming vs
    # non-streaming hard invariant, which a mode-dependent notice would break),
    # and must not claim the user was shown the full output.
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
    # A failure on a local path gets no hint.
    assert _missing_path_hint("FileNotFoundError: 'local.txt'") == ""
    # Mentioning /mnt/data without a file error gets no hint.
    assert _missing_path_hint("saved to /mnt/data, all good") == ""
    assert _missing_path_hint("") == ""


def test_code_tool_descriptions_mention_relative_paths():
    for tool in (PYTHON_TOOL, TERMINAL_TOOL):
        description = tool["function"]["description"]
        assert "relative paths" in description
        assert "/mnt/data" in description


def test_python_exec_mnt_data_open_is_remapped_into_workdir():
    # The sitecustomize shim remaps open()/os.makedirs() on /mnt/data into the
    # sandbox CWD and prints a one-line stderr notice, identically with and
    # without streaming.
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


def test_python_exec_unremapped_mnt_data_failure_gets_hint():
    # os.listdir is deliberately not remapped: the failure must carry the
    # model-visible retry hint instead, identically with and without streaming.
    import re as _re

    code = "import os\nos.listdir('/mnt/data/nonexistent_dir_xyz')\n"
    baseline = _python_exec(code, timeout = 60)
    assert "FileNotFoundError" in baseline
    assert "working directory is writable" in baseline
    streamed = _python_exec(code, timeout = 60, output_callback = lambda _t: None)

    # The traceback embeds each run's random temp filename; normalize it (the
    # byte-identity invariant is per-execution, and these are two executions).
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
