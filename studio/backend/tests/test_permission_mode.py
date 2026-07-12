# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for permission_mode ("Ask for approval" / "Approve for me" /
"Off" / "Full access") permission levels.

Covers the auto-mode safety classifier in tools.py and the loop-level
behavior of run_safetensors_tool_loop: in "auto" mode only calls detected
as potentially unsafe pause for confirmation, in "full" mode nothing
pauses and the sandbox is dropped, and unset/unknown modes behave as
"ask" (every call pauses when confirm_tool_calls is on).
"""

import pytest

from core.inference.mcp_client import MCP_TOOL_PREFIX
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tools import is_potentially_unsafe_tool_call
from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
from state import tool_approvals
from state.tool_approvals import resolve_tool_decision

_SESSION = "perm-mode-session"


@pytest.fixture(autouse = True)
def _clear_pending():
    with tool_approvals._lock:
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()


# ── classifier ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("command", "unsafe"),
    [
        ("ls -la", False),
        ("cat foo.txt | grep hello", False),
        ("find . -name '*.py' | head -5", False),
        ("env FOO=1 grep -r pattern .", False),
        ("echo hi > out.txt", True),  # write redirection
        ("rm -rf /", True),
        ("ls; rm x", True),  # unsafe after separator
        ("xargs rm", True),  # wrapper forwards to unsafe target
        ("sudo ls", True),
        ("git push origin main", True),
        ("pip install requests", True),
        ("echo `whoami`", True),  # substitution fails closed
        ("python -c 'print(1)'", True),  # arbitrary code
        ("find . -exec rm {} ;", True),  # find can execute
        ("find . -delete", True),  # find can delete
        ("fd -x rm", True),  # fd runs a command per result
        ("fd --exec-batch rm", True),
        ("fd -e py pattern", False),  # plain fd search stays read only
        ("sort -o out.txt in.txt", True),  # -o writes a file
        ("sort --output=out in", True),
        ("sort --compress-program=sh big.txt", True),  # runs an external program
        ("sort in.txt", False),  # plain sort stays read only
        ("rg --pre sh needle f.sh", True),  # rg preprocessor runs a command
        ("rg --pre=/tmp/x needle .", True),
        ("rg --hostname-bin /tmp/x foo .", True),
        ("rg --pre-glob '*.txt' needle .", False),  # glob filter stays read only
        ("rg needle .", False),  # plain rg stays read only
        ("/tmp/cat secrets", True),  # path-qualified command is an arbitrary binary
        ("./ls -la", True),
        ("env /tmp/cat x", True),  # path-qualified target after a wrapper
        ("tree -o out.txt", True),  # -o writes a file
        ("xxd -r dump.hex out.bin", True),  # -r can write
        ("awk '{print}' file", True),  # awk can system()/write
        ("grep -o x file", False),  # grep -o is stdout only
        ("ls\nrm -rf x", True),  # newline separates commands
        ("ls\r\nrm x", True),  # CRLF separates commands
        ("ls\n\n\nrm x", True),  # blank lines collapse to one separator
        ("ls\npwd", False),  # multi-line stays safe when every line is
        ("ls\n", False),
        ("sort -o/tmp/out /tmp/in", True),  # attached short output flag
        ("sort -uo out.txt in.txt", True),  # -o bundled in a short cluster
        ("sort -bo out in", True),
        ("sort -u in.txt", False),  # cluster without a write flag stays safe
        ("find . \\( -name x -delete \\)", True),  # -delete inside a group
        ("cat ../../.ssh/id_rsa", True),  # parent traversal read
        ("cat ~/.aws/credentials", True),  # credential path
        ("cat /etc/passwd", True),  # sensitive system file
        ("cat /proc/self/environ", True),  # procfs env dump
        ("cat /proc/1/cmdline", True),
        ("head /proc/self/maps", True),
        ("LD_PRELOAD=/tmp/hook.so ls", True),  # code-loading env prefix
        ("PATH=. ls", True),  # command-lookup env prefix
        ("IFS=x ls", True),
        ("FOO=1 grep -r x .", False),  # benign env prefix stays safe
        ("ps auxe", True),  # ps can dump process env; not on the safe list
        ("ps aux", True),
        ("cat logs/app.log", False),  # ordinary relative read
    ],
)
def test_terminal_classifier(command, unsafe):
    assert is_potentially_unsafe_tool_call("terminal", {"command": command}) is unsafe


@pytest.mark.parametrize(
    ("code", "unsafe"),
    [
        ("print(1+1)", False),
        ("import math\nprint(math.pi)", False),
        ("print(open('x.txt').read())", False),  # read-mode open
        ("open('x.txt', 'w').write('hi')", True),
        ("import shutil; shutil.rmtree('x')", True),
        ("import os; os.remove('x')", True),
        ("import requests", True),  # network module
        ("exec('print(1)')", True),
        ("from os import remove\nremove('x')", True),  # from-import binding
        ("from os import remove as rm\nrm('x')", True),
        ("from os import *", True),  # star import hides anything
        ("import os\nprint(os.getcwd())", False),  # read-only os use
        ("f = os.remove\nf('x')", True),  # indirect reference
        ("import os\nrm = os.remove\nrm('x')", True),  # alias assignment
        ("from pathlib import Path\nPath('x').open('w')", True),  # Path.open mode
        ("from pathlib import Path\nprint(Path('x').open().read())", False),
        ("import zipfile\nprint(zipfile.ZipFile('a').open('n.txt'))", False),
        ("print(open('../../.ssh/id_rsa').read())", True),  # traversal read
        ("print(open('creds.env').read())", True),  # credential file
        ("import os\nos.open('data.txt', os.O_CREAT)", True),  # os.open writes fd
        ("import tempfile\ntempfile.mkstemp()", True),  # tempfile side effects
        ("getattr(os, 'remove')('x')", True),  # dynamic call target
        ("import os as o\no.open('out.txt', o.O_CREAT)", True),  # os.open via alias
        ("from os import open as o, O_CREAT\no('out', O_CREAT)", True),  # os.open bare name
        ("from pathlib import Path\nPath('l').symlink_to('t')", True),  # pathlib link
        ("import importlib\nimportlib.import_module('subprocess')", True),  # dynamic import
        ("import os\nos.mkfifo('p')", True),  # node creation
        ("import os\nos.utime('x', None)", True),  # metadata mutation
        ("f = open\nf('x', 'w')", True),  # builtin open aliased to a name
        ("from builtins import open as w\nw('x', 'w')", True),
        ("globals()['open']('x', 'w')", True),  # dynamic open lookup
        ("import pickle\npickle.loads(b'')", True),  # code exec on load
        ("import io\nio.FileIO('out', 'w')", True),  # raw write handle
        (
            "import zipfile\nprint(zipfile.ZipFile('a').open('n.txt', 'r'))",
            False,
        ),  # explicit read mode
    ],
)
def test_python_classifier(code, unsafe):
    assert is_potentially_unsafe_tool_call("python", {"code": code}) is unsafe


def test_builtin_readonly_tools_are_safe():
    assert is_potentially_unsafe_tool_call("web_search", {"query": "hi"}) is False
    assert is_potentially_unsafe_tool_call("search_knowledge_base", {}) is False
    assert is_potentially_unsafe_tool_call("render_html", {}) is False


def test_unknown_tools_fail_closed():
    assert is_potentially_unsafe_tool_call("mystery_tool", {}) is True


@pytest.mark.parametrize(
    ("tool", "unsafe"),
    [
        ("get_weather", False),
        ("list_files", False),
        ("search", False),
        ("send_email", True),
        ("create_issue", True),
        ("delete_row", True),
        ("get_or_create_issue", True),  # mutating verb overrides read prefix
        ("read_and_delete_file", True),
        ("find_and_update_row", True),
        ("get_and_commit_changes", True),  # commit/save/archive are mutating
        ("read_and_save_file", True),
        ("list_and_archive", True),
    ],
)
def test_mcp_classifier(tool, unsafe):
    name = f"{MCP_TOOL_PREFIX}srv1__{tool}"
    assert is_potentially_unsafe_tool_call(name, {}) is unsafe


# ── loop behavior ───────────────────────────────────────────────────

_DEFAULT_TOOLS = [
    {"type": "function", "function": {"name": "python"}},
    {"type": "function", "function": {"name": "web_search"}},
]


class _FakeExecuteTool:
    def __init__(self):
        self.calls = []
        self.disable_sandbox_seen = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        self.disable_sandbox_seen.append(disable_sandbox)
        return f"RESULT[{name}]"


def _tool_call(name, args_json):
    return f'<tool_call>{{"name": "{name}", "arguments": {args_json}}}</tool_call>'


def _multi_turn(turns):
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            yield next(turn_iter)
        except StopIteration:
            return

    return _gen


def _drive(turns, decisions, **loop_kwargs):
    """Run the loop, resolving each gated tool_start with the next decision."""
    decision_iter = iter(decisions)
    exec_fn = _FakeExecuteTool()
    gen = run_safetensors_tool_loop(
        single_turn = _multi_turn(turns),
        messages = [{"role": "user", "content": "hi"}],
        tools = _DEFAULT_TOOLS,
        execute_tool = exec_fn,
        session_id = _SESSION,
        **loop_kwargs,
    )
    events = []
    for ev in gen:
        events.append(ev)
        if ev["type"] == "tool_start" and ev.get("awaiting_confirmation"):
            resolve_tool_decision(ev["approval_id"], next(decision_iter), session_id = _SESSION)
    return events, exec_fn


def _tool_starts(events):
    return [e for e in events if e["type"] == "tool_start"]


def test_auto_mode_does_not_gate_safe_calls():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        [],
        confirm_tool_calls = True,
        permission_mode = "auto",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""
    assert exec_fn.calls == [("python", {"code": "print(1)"})]
    assert exec_fn.disable_sandbox_seen == [False]  # sandbox stays on in auto


def test_auto_mode_gates_unsafe_calls():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
        permission_mode = "auto",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True
    assert starts[0]["approval_id"]
    assert len(exec_fn.calls) == 1
    assert exec_fn.disable_sandbox_seen == [False]


def test_ask_mode_gates_even_safe_calls():
    events, _ = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
        permission_mode = "ask",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True


def test_unset_mode_behaves_as_ask():
    events, _ = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        ["allow"],
        confirm_tool_calls = True,
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is True


def test_off_mode_never_gates_and_keeps_sandbox():
    # "Off": no prompts even for unsafe calls, but the sandbox stays on.
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        [],
        confirm_tool_calls = True,  # off must win over a stray confirm flag
        permission_mode = "off",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert starts[0]["approval_id"] == ""
    assert exec_fn.disable_sandbox_seen == [False]


def test_full_mode_never_gates_and_drops_sandbox():
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "import os; os.remove(\\"x\\")"}'), "final"],
        [],
        confirm_tool_calls = True,  # full must win over the confirm gate
        permission_mode = "full",
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert exec_fn.disable_sandbox_seen == [True]


def test_bypass_flag_implies_full_mode():
    # Legacy callers that only set bypass_permissions keep the same behavior.
    events, exec_fn = _drive(
        [_tool_call("python", '{"code": "print(1)"}'), "final"],
        [],
        confirm_tool_calls = True,
        bypass_permissions = True,
    )
    starts = _tool_starts(events)
    assert starts and starts[0]["awaiting_confirmation"] is False
    assert exec_fn.disable_sandbox_seen == [True]


def test_bypass_permissions_folds_to_full_on_request_models():
    # A legacy bypass caller that also sends a stale ask/auto mode normalizes to
    # full, so the route guards (which reject ask/auto) don't 400 the request.
    for cls in (ChatCompletionRequest, AnthropicMessagesRequest):
        req = cls(
            messages = [{"role": "user", "content": "hi"}],
            bypass_permissions = True,
            permission_mode = "auto",
        )
        assert req.permission_mode == "full"
        assert req.bypass_permissions is True


def test_ask_auto_self_enable_confirm_on_chat_request():
    # A direct /chat/completions caller that requests ask/auto but omits the
    # legacy confirm flag must still hit the confirmation gate.
    for mode in ("ask", "auto"):
        req = ChatCompletionRequest(
            messages = [{"role": "user", "content": "hi"}],
            permission_mode = mode,
        )
        assert req.confirm_tool_calls is True
    # A contradictory confirm=False is overridden by the explicit mode.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        permission_mode = "ask",
        confirm_tool_calls = False,
    )
    assert req.confirm_tool_calls is True
    # Legacy callers with no permission_mode keep their confirm flag untouched.
    req = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}],
        confirm_tool_calls = False,
    )
    assert req.confirm_tool_calls is False
