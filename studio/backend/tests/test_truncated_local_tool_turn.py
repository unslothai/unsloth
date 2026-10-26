# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A generation limit must not turn an unfinished rewrite into an executable tool call."""

import pytest

from core.inference import tools
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from .test_llama_cpp_tool_loop import (
    _backend_and_payloads,
    _done,
    _finish,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
)


_COMMAND = "cat > app.py <<'EOF'\n" + "".join(f"x_{i} = {i}\n" for i in range(44))
_PARTIAL = "<tool_call><function=terminal><parameter=command>" + _COMMAND
_CLOSED = _PARTIAL + "EOF\n</parameter></function></tool_call>"
_ORIGINAL = "".join(f"x_{i} = {i}\n" for i in range(175))


@pytest.mark.parametrize("reason", ["length", "content_filter"])
@pytest.mark.parametrize("shape", ["xml", "prefaced_xml", "structured", "siblings"])
def test_gguf_never_executes_calls_from_a_truncated_turn(monkeypatch, reason, shape):
    if shape == "structured":
        stream = [_tool_call_sse("terminal", {"command": _COMMAND}, "call_write")]
    elif shape == "siblings":
        stream = [_sse({"content": _CLOSED + _PARTIAL})]
    elif shape == "prefaced_xml":
        stream = [_sse({"content": "Updating your file. "}), _sse({"content": _PARTIAL})]
    else:
        stream = [_sse({"content": _PARTIAL})]
    stream += [_finish(reason), _done()]
    backend, payloads = _backend_and_payloads(
        monkeypatch, [stream, [_sse({"content": "Done."}), _finish("stop"), _done()]]
    )
    calls = _record_tool_calls(monkeypatch, "must not run")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert calls == []
    assert len(payloads) == 1
    assert events[-1]["finish_reason"] == reason
    text = [event["text"] for event in events if event["type"] == "content"][-1]
    assert "These tool calls were not run" in text
    if shape == "prefaced_xml":
        assert text.startswith("Updating your file.")
    starts = {event["tool_call_id"] for event in events if event["type"] == "tool_start"}
    ends = {event["tool_call_id"] for event in events if event["type"] == "tool_end"}
    assert starts <= ends


@pytest.mark.parametrize("heal", [True, False])
@pytest.mark.parametrize("reason", ["stop", "tool_calls"])
def test_gguf_complete_calls_still_run(monkeypatch, heal, reason):
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": _CLOSED}), _finish(reason), _done()],
            [_sse({"content": "Done."}), _finish("stop"), _done()],
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 100)
    calls = _record_tool_calls(monkeypatch, "written")
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.TERMINAL_TOOL],
        auto_heal_tool_calls = heal,
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert len(calls) == 1


def test_truncated_gguf_edit_preserves_the_real_file(monkeypatch, tmp_path):
    target = tmp_path / "app.py"
    target.write_text(_ORIGINAL)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    arguments = {
        "path": "app.py",
        "edits": [
            {"old_string": _ORIGINAL, "new_string": "".join(_ORIGINAL.splitlines(True)[:44])}
        ],
    }
    backend, _ = _backend_and_payloads(
        monkeypatch,
        [
            [_tool_call_sse("edit_file", arguments, "call_write"), _finish("length"), _done()],
            [_sse({"content": "Done."}), _finish("stop"), _done()],
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 100)
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Update app.py"}],
        [tools.EDIT_FILE_TOOL],
        permission_mode = "off",
        session_id = "truncated-edit",
    )
    assert target.read_text() == _ORIGINAL


@pytest.mark.parametrize(
    "stats",
    [{"truncated": True}, {"finish_reason": "length"}, {"finish_reason": "content_filter"}],
)
@pytest.mark.parametrize("name", ["terminal", "render_html"])
def test_safetensors_and_mlx_truncated_turns_do_not_execute(stats, name):
    holder = {}
    calls = []

    def single_turn(messages, **kwargs):
        key = "command" if name == "terminal" else "code"
        prefix = f"<tool_call><function={name}><parameter={key}>"
        yield prefix
        yield prefix + "print('partial')"
        holder["stats"] = stats

    def execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return "must not run"

    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "Update the file"}],
            tools = [{"type": "function", "function": {"name": name}}],
            execute_tool = execute,
            generation_stats_holder = holder,
            permission_mode = "off",
            sandbox_level = "low",
        )
    )
    assert calls == []
    assert "These tool calls were not run" in events[-1]["text"]
    starts = {event["tool_call_id"] for event in events if event["type"] == "tool_start"}
    ends = {event["tool_call_id"] for event in events if event["type"] == "tool_end"}
    assert starts <= ends


@pytest.mark.parametrize("stats", [{"truncated": False}, {"finish_reason": "stop"}, {}])
def test_safetensors_complete_turn_still_executes(stats):
    holder = {}
    calls = []

    def single_turn(messages, **kwargs):
        yield _CLOSED if not calls else "Done."
        holder["stats"] = stats

    def execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return "written"

    list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "Update the file"}],
            tools = [tools.TERMINAL_TOOL],
            execute_tool = execute,
            generation_stats_holder = holder,
            permission_mode = "off",
            sandbox_level = "low",
        )
    )
    assert len(calls) == 1
