# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox Low/High: Low runs Python and Terminal on software safeguards only.

High is the behaviour before the level existed. Low never probes or uses the OS sandbox, so its
calls count as not isolated: "off" asks before their risky calls. Full access overrides both.
"""

import ast
import asyncio
import inspect
import itertools
import json
import sys
import textwrap
import threading
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import mxc_policy, mxc_probe, os_sandbox, tools
from models.inference import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ChatCountTokensRequest,
)
from state import tool_policy

MODES = ("ask", "auto", "off", "full")
LEVELS = ("high", "low")
ISOLATION = (True, False, None)
TOOLS = ("python", "terminal", "web_search")
_SESSION = "__LOCALID_sandbox_level"


def _risk(name, arguments):
    return bool(arguments.get("risky"))


def _never(name):
    return False


@pytest.fixture
def isolation(monkeypatch):
    state = {}
    monkeypatch.setattr(os_sandbox, "cached_tool_isolation", lambda tool: state.get(tool))
    return state


def _confirm_expected(mode, level, tool, isolated, risky):
    if mode == "full":
        return False
    if mode == "ask":
        return True
    if mode == "auto":
        return risky
    os_tool = tool in ("python", "terminal")
    return os_tool and (level == "low" or isolated is not True) and risky


@pytest.mark.parametrize(
    "mode,level,tool,isolated,risky",
    list(itertools.product(MODES, LEVELS, TOOLS, ISOLATION, (True, False))),
)
def test_decision_matrix(isolation, mode, level, tool, isolated, risky):
    isolation["python"] = isolation["terminal"] = isolated
    decision = dict(
        confirm_tool_calls = True,
        bypass_permissions = mode == "full",
        permission_mode = mode,
        name = tool,
        arguments = {"risky": risky},
        sandbox_level = level,
    )
    prompted = tool_policy.needs_tool_confirmation(
        **decision, is_high_risk = _risk, never_needs = _never
    )
    assert prompted is _confirm_expected(mode, level, tool, isolated, risky)
    strict = tool_policy.requires_os_isolation(**decision, prompted = prompted, is_high_risk = _risk)
    # Low never demands the OS sandbox; High keeps today's strict launch for an unasked risky call.
    assert strict is (
        level == "high"
        and mode == "off"
        and tool in ("python", "terminal")
        and risky
        and not prompted
    )


@pytest.mark.parametrize("level,isolated", list(itertools.product(LEVELS, ISOLATION)))
def test_off_may_prompt_for_low_even_when_the_os_sandbox_works(isolation, level, isolated):
    isolation["python"] = isolated
    may_prompt = tool_policy.tool_call_may_prompt(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "off",
        name = "python",
        sandbox_level = level,
    )
    assert may_prompt is (level == "low" or isolated is not True)
    assert tool_policy.off_mode_still_gates("python", level) is may_prompt
    assert tool_policy.off_mode_still_gates("web_search", level) is False


def test_the_policy_level_defaults_to_high_and_rejects_typos():
    assert tool_policy.normalize_sandbox_level(None) == "high"
    assert tool_policy.normalize_sandbox_level(" LOW ") == "low"
    with pytest.raises(ValueError):
        tool_policy.normalize_sandbox_level("lowest")


def _chat(**extra):
    return ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}], **extra)


def _count(**extra):
    return ChatCountTokensRequest(messages = [{"role": "user", "content": "hi"}], **extra)


def _anthropic(**extra):
    return AnthropicMessagesRequest(
        model = "m", max_tokens = 8, messages = [{"role": "user", "content": "hi"}], **extra
    )


@pytest.mark.parametrize("build", [_chat, _count, _anthropic], ids = ["chat", "count", "anthropic"])
@pytest.mark.parametrize(
    "sent,expected",
    [
        ({}, "high"),
        ({"sandbox_level": None}, "high"),
        ({"sandbox_level": " LOW "}, "low"),
        ({"sandbox_level": "High"}, "high"),
    ],
)
def test_the_request_field_is_normalized(build, sent, expected):
    assert build(**sent).sandbox_level == expected


def test_an_unknown_level_is_a_422():
    app = FastAPI()

    @app.post("/chat")
    def chat(payload: ChatCompletionRequest):
        return {"level": payload.sandbox_level}

    @app.post("/messages")
    def messages(payload: AnthropicMessagesRequest):
        return {"level": payload.sandbox_level}

    client = TestClient(app)
    chat_body = {"messages": [{"role": "user", "content": "hi"}]}
    anthropic_body = {**chat_body, "model": "m", "max_tokens": 8}
    for path, body in (("/chat", chat_body), ("/messages", anthropic_body)):
        assert client.post(path, json = {**body, "sandbox_level": "Low"}).json() == {"level": "low"}
        for bad in ("medium", "", 1, True):
            response = client.post(path, json = {**body, "sandbox_level": bad})
            assert response.status_code == 422, (path, bad)


@pytest.fixture
def no_probe(monkeypatch):
    def probe(**_kw):
        raise AssertionError("Sandbox Low must not probe the OS sandbox")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", probe)


def test_the_software_mode_is_internal_only():
    assert "software" in os_sandbox.TOOL_EXECUTION_MODES
    assert os_sandbox.PUBLIC_TOOL_EXECUTION_MODES == ("auto", "required")


def test_the_planner_runs_low_on_software_safeguards_without_a_probe(no_probe, tmp_path):
    os_sandbox.forget_tool_isolation()
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "pass"),
        workdir = str(tmp_path),
        env = {},
        execution_kind = "python",
        requested_mode = "software",
        timeout_seconds = 5,
    )
    launch = os_sandbox.prepare_tool_launch(plan)
    record = launch.execution_record
    assert launch.argv == plan.argv and launch.backend == "software-safeguards"
    assert record.requested_mode == "software"
    assert record.effective_mode == "software_safeguards"
    assert record.os_isolation is False
    assert "no_os_isolation" in record.limitations
    assert "timeout" in record.retained_safeguards
    assert not os_sandbox.has_tool_isolation_answer("python")


@pytest.mark.parametrize(
    "run,expected",
    [
        (
            lambda: tools._python_exec(
                "print(6 * 7)", None, 60, _SESSION, tool_execution_mode = "software"
            ),
            "42",
        ),
        (
            lambda: tools._bash_exec("echo 42", None, 60, _SESSION, tool_execution_mode = "software"),
            "42",
        ),
    ],
    ids = ["python", "terminal"],
)
def test_low_runs_both_tools_on_software_safeguards(no_probe, run, expected):
    tools._last_tool_execution_record = None
    assert expected in run()
    record = tools._last_tool_execution_record
    assert record.requested_mode == "software"
    assert record.effective_mode == "software_safeguards"
    assert record.os_isolation is False


def test_full_access_overrides_low(no_probe):
    tools._last_tool_execution_record = None
    out = tools._python_exec(
        "print(5)", None, 60, _SESSION, disable_sandbox = True, tool_execution_mode = "software"
    )
    assert "5" in out
    assert tools._last_tool_execution_record.effective_mode == "full"


class _RecordingConfinement:
    preexec = None
    confines = True

    def __init__(self):
        self.wrapped = []

    def wrap(self, argv):
        self.wrapped.append(list(argv))
        return argv


@pytest.mark.parametrize(
    "function,payload",
    [(tools._python_exec, "print('MANAGED_OK')"), (tools._bash_exec, "echo MANAGED_OK")],
    ids = ["python", "terminal"],
)
def test_low_keeps_the_managed_account_boundary(monkeypatch, no_probe, function, payload):
    confinement = _RecordingConfinement()
    monkeypatch.setattr(tools, "_account_confinement", lambda: confinement)
    tools._last_tool_execution_record = None
    assert "MANAGED_OK" in function(payload, None, 60, _SESSION, tool_execution_mode = "software")
    assert confinement.wrapped, "the account boundary was skipped"
    assert tools._last_tool_execution_record is None


_BASH = r"C:\Program Files\Git\bin\bash.exe"


@pytest.fixture
def windows_cmd_host(monkeypatch):
    """Git Bash fails inside MXC and cmd qualifies, so High runs the Terminal isolated on cmd."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.delenv("UNSLOTH_MXC_TERMINAL_CMD", raising = False)
    monkeypatch.setattr(tools, "_request_profile", [None, 0.0])
    monkeypatch.setattr(tools, "_windows_bash", lambda: _BASH)
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: True)

    def snapshot(*, selected_executable = None, **_kw):
        available = selected_executable != _BASH
        reason = "" if available else mxc_probe.MSYS_NAMESPACE_REASON
        return os_sandbox.SandboxCapability(backend = "mxc", available = available, reason = reason)

    monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)


def _terminal_tool():
    return [{"type": "function", "function": {"name": "terminal", "description": "Run a command."}}]


def test_windows_low_advertises_the_host_shell(windows_cmd_host):
    assert tools._terminal_profile() == "cmd_isolated"
    high = tools.apply_terminal_profile_for_request(_terminal_tool(), "high")
    low = tools.apply_terminal_profile_for_request(_terminal_tool(), "low")
    assert low == tools.apply_terminal_profile_description(_terminal_tool(), "bash")
    assert high == tools.apply_terminal_profile_description(_terminal_tool(), "cmd_isolated")
    assert low != high


def test_windows_low_launches_on_the_host_shell(windows_cmd_host, monkeypatch):
    seen = []

    class _Stop(Exception):
        pass

    def profile(disable_sandbox = False):
        seen.append(disable_sandbox)
        raise _Stop

    monkeypatch.setattr(tools, "_terminal_profile", profile)
    for mode in ("auto", "software"):
        with pytest.raises(_Stop):
            tools._bash_exec("echo hi", None, 60, _SESSION, tool_execution_mode = mode)
    assert seen == [False, True]


@pytest.mark.parametrize("level,expected", [("high", [False]), ("low", [True])])
def test_the_risk_check_reads_the_terminal_shell_of_the_requested_level(
    windows_cmd_host, monkeypatch, level, expected
):
    from state import tool_policy

    seen = []
    real = tools._terminal_profile

    def profile(disable_sandbox = False):
        seen.append(disable_sandbox)
        return real(disable_sandbox)

    monkeypatch.setattr(tools, "_terminal_profile", profile)
    # cmd reads ' and ^ unlike bash, so only the isolated cmd profile looks twice.
    tool_policy.needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "auto",
        name = "terminal",
        arguments = {"command": "echo 'a ^& b'"},
        sandbox_level = level,
    )
    assert seen[:1] == expected
    assert tools._classifying_sandbox_level.get() is None


class _ModeRecordingExecuteTool:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
        tool_execution_mode = "auto",
        **_kwargs,
    ):
        self.calls.append((name, tool_execution_mode, disable_sandbox))
        return f"RESULT[{name}]"


_LOOP_CASES = [
    # level, tool, bypass, expected mode
    ("low", "python", False, "software"),
    ("low", "terminal", False, "software"),
    ("low", "web_search", False, "auto"),
    ("high", "python", False, "auto"),
    ("low", "python", True, "full"),
]


def _arguments(tool):
    return {"query": "x"} if tool == "web_search" else {"code": "print(1)", "command": "echo 1"}


@pytest.mark.parametrize("level,tool,bypass,expected", _LOOP_CASES)
def test_the_safetensors_loop_launches_low_in_software_mode(level, tool, bypass, expected):
    from core.inference.safetensors_agentic import run_safetensors_tool_loop

    call = json.dumps({"name": tool, "arguments": _arguments(tool)})
    turns = iter([f"<tool_call>{call}</tool_call>", "final"])

    def single_turn(_messages):
        try:
            yield next(turns)
        except StopIteration:
            return

    exec_fn = _ModeRecordingExecuteTool()
    list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": tool}}],
            execute_tool = exec_fn,
            session_id = f"level-{uuid.uuid4().hex}",
            confirm_tool_calls = False,
            bypass_permissions = bypass,
            permission_mode = "full" if bypass else "off",
            sandbox_level = level,
        )
    )
    assert len(exec_fn.calls) == 1
    _name, mode, disable_sandbox = exec_fn.calls[0]
    assert disable_sandbox is bypass
    # Full access wins inside execute_tool, whatever mode rides along.
    assert tools._requested_execution_mode(mode, disable_sandbox) == expected


def _studio_call_line(tool):
    function = {"name": tool, "arguments": json.dumps(_arguments(tool))}
    delta = {"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": function}]}
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]})


class _FakeTransport:
    heals_text_tool_calls = False

    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        lines = self.turns.pop(0) if self.turns else ["data: [DONE]"]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


@pytest.mark.parametrize("level,tool,bypass,expected", _LOOP_CASES)
def test_the_shared_loop_launches_low_in_software_mode(monkeypatch, level, tool, bypass, expected):
    from core.inference import studio_tool_loop as loop_mod

    exec_fn = _ModeRecordingExecuteTool()
    monkeypatch.setattr(loop_mod, "execute_tool", exec_fn)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    finish = "data: " + json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
    )
    transport = _FakeTransport([[_studio_call_line(tool), finish], ["data: [DONE]"]])

    async def _collect():
        stream = loop_mod.stream_with_studio_tools(
            transport,
            run = loop_mod.ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}], session_id = "s1", thread_id = "t1"
            ),
            policy = loop_mod.ToolLoopPolicy(
                tools = [{"type": "function", "function": {"name": tool, "parameters": {}}}],
                max_calls = 5,
                timeout = 60,
                permission_mode = "full" if bypass else "off",
                confirm_calls = False,
                bypass_permissions = bypass,
                rag_scope = None,
                sandbox_level = level.upper(),
            ),
            cancel_event = threading.Event(),
        )
        return [line async for line in stream]

    asyncio.run(asyncio.wait_for(_collect(), timeout = 30))
    assert len(exec_fn.calls) == 1
    _name, mode, disable_sandbox = exec_fn.calls[0]
    assert tools._requested_execution_mode(mode, disable_sandbox) == expected


def _calls_in(source, names):
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in names:
                yield name, node


def test_every_decision_in_the_gguf_loop_reads_the_level():
    # The GGUF loop needs a live llama-server, so (as the other llama_cpp tests do) read its source.
    from core.inference import llama_cpp

    source = inspect.getsource(llama_cpp.LlamaCppBackend.generate_chat_completion_with_tools)
    decisions = {"tool_call_may_prompt", "needs_tool_confirmation", "requires_os_isolation"}
    found = list(_calls_in(source, decisions))
    assert {name for name, _ in found} == decisions
    for name, node in found:
        assert "sandbox_level" in {kw.arg for kw in node.keywords}, name
    assert 'kwargs["tool_execution_mode"] = "software"' in source
    assert "sandbox_level = normalize_sandbox_level(sandbox_level)" in source


def test_every_loop_launches_low_through_the_mode_parameter():
    from core.inference import llama_cpp, safetensors_agentic, studio_tool_loop
    for module in (studio_tool_loop, llama_cpp, safetensors_agentic):
        source = inspect.getsource(module)
        assert 'kwargs["tool_execution_mode"] = "software"' in source, module.__name__
        assert "runs_without_os_sandbox(" in source, module.__name__


@pytest.mark.parametrize(
    "path", ["routes/inference.py", "routes/managed_engine_chat.py"], ids = ["inference", "managed"]
)
def test_every_route_that_forwards_the_mode_forwards_the_level(path):
    source = (_BACKEND_ROOT / path).read_text(encoding = "utf-8")
    forwarded = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        mode = keywords.get("permission_mode")
        if mode is None or "payload" not in ast.unparse(mode):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if callee == "load_mentioned_skills":
            continue
        assert "sandbox_level" in keywords, f"{path}:{node.lineno} {callee}"
        forwarded += 1
    assert forwarded


def test_the_safetensors_orchestrator_forwards_the_level():
    from core.inference.orchestrator import InferenceOrchestrator

    function = InferenceOrchestrator.generate_chat_completion_with_tools
    assert "sandbox_level" in inspect.signature(function).parameters
    loops = list(_calls_in(inspect.getsource(function), {"run_safetensors_tool_loop"}))
    assert loops
    for _name, node in loops:
        level = {kw.arg: kw.value for kw in node.keywords}.get("sandbox_level")
        assert isinstance(level, ast.Name) and level.id == "sandbox_level"


def test_the_codex_loop_forwards_the_level(monkeypatch):
    from core.inference import openai_codex_tool_loop as loop_mod

    entered = {}

    def _loop(*a, **k):
        entered.update(k)

        async def gen():
            yield "data: [DONE]\n\n"

        return gen()

    monkeypatch.setattr(loop_mod, "stream_with_studio_tools", _loop)
    run = loop_mod.CodexRunContext(
        provider_id = "p",
        thread_id = None,
        session_id = None,
        messages = [],
        model = "m",
        reasoning_effort = None,
    )
    policy = loop_mod.CodexToolPolicy(
        tools = [],
        max_calls = 1,
        timeout = 1,
        permission_mode = "off",
        confirm_calls = False,
        bypass_permissions = False,
        rag_scope = None,
        sandbox_level = "low",
    )
    loop_mod.stream_codex_with_studio_tools(
        object(), run = run, policy = policy, cancel_event = threading.Event()
    )
    assert entered["policy"].sandbox_level == "low"


@pytest.mark.parametrize("sent,expected", [("low", "low"), ("LOW", "low"), (None, "high")])
def test_a_durable_run_keeps_the_level_for_resume(sent, expected):
    from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request

    request = {"messages": [{"role": "user", "content": "hi"}], "permission_mode": "off"}
    if sent is not None:
        request["sandbox_level"] = sent
    sanitized = _sanitize_request(
        CreateChatGenerationRun(
            runId = "run1",
            threadId = "thread1",
            userMessageId = "u1",
            assistantMessageId = "a1",
            requestPayload = request,
        )
    )
    assert sanitized["sandbox_level"] == expected
    # The stored payload is replayed through the same model on resume and retry.
    replayed = ChatCompletionRequest.model_validate(json.loads(json.dumps(sanitized)))
    assert replayed.sandbox_level == expected
