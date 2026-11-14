# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Approving a call that names host paths runs it without the OS sandbox: the Unsloth UI's choice, not an API key's."""

import asyncio
import json
import shutil
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import llama_cpp, os_sandbox, safetensors_agentic, tools
from core.inference import studio_tool_loop as loop_mod
from routes import inference
from state import tool_approvals, tool_policy
from utils.account_context import (
    OWNER,
    as_api_caller,
    bind_account,
    reset_account,
    via_api_key_context,
)

# The loop harnesses sit alongside and import as top-level modules.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_OPT_IN = "UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS"
_API = as_api_caller(OWNER)
_HOST_CODE = "print(open('/home/someone/notes.txt').read())"
_SANDBOXED_CODE = "print(1)"
_LOOPS = ("safetensors", "llama_cpp", "studio")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising = False)
    with tool_approvals._lock:
        saved = dict(tool_approvals._pending)
        tool_approvals._pending.clear()
    yield
    with tool_approvals._lock:
        tool_approvals._pending.clear()
        tool_approvals._pending.update(saved)


def _client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


class _Launcher:
    """Stands in for the python tool: the real launch decision, with the OS sandbox stubbed out."""

    def __init__(self, monkeypatch, workdir):
        self.workdir = str(workdir)
        self.launches: list[tuple[str, str]] = []
        monkeypatch.setattr(os_sandbox, "prepare_tool_launch", self._os_sandbox)

    @staticmethod
    def _os_sandbox(plan):
        return types.SimpleNamespace(
            backend = "os-sandbox",
            requested_mode = plan.requested_mode,
            preexec_fn = None,
            execution_record = None,
            env = plan.env,
            cleanup_paths = [],
        )

    def __call__(self, name, arguments, **kwargs):
        code = arguments["code"]
        plan = os_sandbox.ToolLaunchPlan(
            argv = ("python", "-c", code),
            workdir = self.workdir,
            env = {},
            requested_mode = tools._requested_execution_mode(
                kwargs.get("tool_execution_mode", "auto"), kwargs.get("disable_sandbox", False)
            ),
            execution_kind = "python",
        )
        approved = kwargs.get("host_access_approved", False)
        prepared = tools._prepare_tool_launch(
            plan,
            host_access_approved = approved and tools._reaches_host_paths("python", code),
            host_script = code,
        )
        for path in getattr(prepared, "cleanup_paths", None) or []:
            shutil.rmtree(path, ignore_errors = True)
        mode = getattr(prepared, "requested_mode", None) or plan.requested_mode
        self.launches.append((prepared.backend, mode))
        return "OK"


class _Approver:
    """Answers each parked call through POST /api/inference/tool-confirm, as a client would."""

    def __init__(self, client, decision):
        self.client = client
        self.decision = decision
        self.statuses: list[int] = []

    def _post(self, approval_id, session, decision):
        response = self.client.post(
            "/api/inference/tool-confirm",
            json = {"approval_id": approval_id, "session_id": session, "decision": decision},
        )
        self.statuses.append(response.status_code)
        return response

    def __call__(self, slot, approval_id, *_args, **_kwargs):
        session = slot["session"]
        response = self._post(approval_id, session, self.decision)
        if response.status_code == 403:
            assert _OPT_IN in response.json()["detail"]
            # A refused approval leaves the call parked; the client can still deny it.
            assert self._post(approval_id, session, "deny").status_code == 200
        return slot["decision"] or "deny"


def _tool_call_markup(code):
    return f'<tool_call>{{"name": "python", "arguments": {json.dumps({"code": code})}}}</tool_call>'


def _drive(loop, monkeypatch, tmp_path, *, origin, approver, decision, code, mode):
    launcher = _Launcher(monkeypatch, tmp_path)
    with _client_for(approver) as client:
        answer = _Approver(client, decision)
        token = bind_account(origin)
        try:
            _run_loop(loop, monkeypatch, launcher, answer, code, mode)
        finally:
            reset_account(token)
    return answer.statuses, launcher.launches


def _run_loop(loop, monkeypatch, launcher, answer, code, mode):
    session = f"api-confirm-{uuid.uuid4().hex}"
    if loop == "safetensors":
        monkeypatch.setattr(safetensors_agentic, "wait_tool_decision", answer)
        turns = iter([_tool_call_markup(code), "done"])

        def single_turn(_messages):
            try:
                yield next(turns)
            except StopIteration:
                return

        list(
            safetensors_agentic.run_safetensors_tool_loop(
                single_turn = single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "python"}}],
                execute_tool = launcher,
                session_id = session,
                confirm_tool_calls = True,
                permission_mode = mode,
                max_tool_iterations = 1,
            )
        )
        return
    if loop == "llama_cpp":
        from test_llama_cpp_tool_loop import (
            _backend_and_payloads,
            _done,
            _run_tool_loop,
            _sse,
            _streamed_structured_tool_call,
        )

        monkeypatch.setattr(llama_cpp, "wait_tool_decision", answer)
        monkeypatch.setattr("core.inference.tools.execute_tool", launcher)
        backend, _payloads = _backend_and_payloads(
            monkeypatch,
            [
                _streamed_structured_tool_call("python", {"code": code}, "call_1"),
                [_sse({"content": "Done."}), _done()],
            ],
        )
        _run_tool_loop(
            backend,
            [{"role": "user", "content": "run it"}],
            [{"type": "function", "function": {"name": "python"}}],
            confirm_tool_calls = True,
            permission_mode = mode,
            session_id = session,
        )
        return
    from test_studio_tool_loop import PY, FakeTransport, _DONE, _sse as _studio_sse
    from core.inference.studio_tool_loop import (
        ToolLoopPolicy,
        ToolLoopRun,
        stream_with_studio_tools,
    )

    monkeypatch.setattr(loop_mod, "wait_tool_decision", answer)
    monkeypatch.setattr(loop_mod, "execute_tool", launcher)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    call = {
        "index": 0,
        "id": "c1",
        "function": {"name": "python", "arguments": json.dumps({"code": code})},
    }
    transport = FakeTransport(
        [
            [_studio_sse({"tool_calls": [call]}), _studio_sse(finish = "tool_calls"), _DONE],
            [_studio_sse({"content": "ok"}), _studio_sse(finish = "stop"), _DONE],
        ]
    )

    async def _collect():
        agen = stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}],
                session_id = session,
                thread_id = "t1",
            ),
            policy = ToolLoopPolicy(
                tools = [PY],
                max_calls = 1,
                timeout = 300,
                permission_mode = mode,
                confirm_calls = True,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for _line in agen:
            pass

    asyncio.run(_collect())


@pytest.mark.parametrize("loop", _LOOPS)
def test_api_key_cannot_approve_a_host_path_call(monkeypatch, tmp_path, loop):
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = _API,
        approver = _API,
        decision = "allow",
        code = _HOST_CODE,
        mode = "auto",
    )
    assert statuses == [403, 200]
    assert launches == []


@pytest.mark.parametrize("loop", _LOOPS)
def test_operator_opt_in_restores_api_key_host_approval(monkeypatch, tmp_path, loop):
    monkeypatch.setenv(_OPT_IN, "1")
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = _API,
        approver = _API,
        decision = "allow",
        code = _HOST_CODE,
        mode = "auto",
    )
    assert statuses == [200]
    assert launches == [("software-safeguards", "auto")]


@pytest.mark.parametrize("loop", _LOOPS)
def test_ui_session_host_approval_is_unchanged(monkeypatch, tmp_path, loop):
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = OWNER,
        approver = OWNER,
        decision = "allow",
        code = _HOST_CODE,
        mode = "auto",
    )
    assert statuses == [200]
    assert launches == [("software-safeguards", "auto")]


@pytest.mark.parametrize("loop", _LOOPS)
def test_api_key_can_still_deny(monkeypatch, tmp_path, loop):
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = _API,
        approver = _API,
        decision = "deny",
        code = _HOST_CODE,
        mode = "auto",
    )
    assert statuses == [200]
    assert launches == []


@pytest.mark.parametrize("loop", _LOOPS)
def test_api_key_can_still_approve_a_sandboxed_call(monkeypatch, tmp_path, loop):
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = _API,
        approver = _API,
        decision = "allow",
        code = _SANDBOXED_CODE,
        mode = "ask",
    )
    assert statuses == [200]
    assert launches == [("os-sandbox", "auto")]


@pytest.mark.parametrize("loop", _LOOPS)
def test_a_ui_approval_does_not_unsandbox_an_api_key_request(monkeypatch, tmp_path, loop):
    """The launch side holds on its own: an API-key request keeps the OS sandbox whoever approves."""
    statuses, launches = _drive(
        loop,
        monkeypatch,
        tmp_path,
        origin = _API,
        approver = OWNER,
        decision = "allow",
        code = _HOST_CODE,
        mode = "auto",
    )
    assert statuses == [200]
    # Required, so a sandbox that went away refuses instead of falling back to software safeguards.
    assert launches == [("os-sandbox", "required")]


def test_owner_api_key_tool_thread_keeps_the_api_key_flag():
    from core.inference.tool_stream_exec import stream_tool_execution

    seen = []
    stream = tool_policy.account_tool_stream
    token = bind_account(_API)
    try:
        wrapped = stream(stream_tool_execution)
    finally:
        reset_account(token)
    list(wrapped(lambda _output: seen.append(via_api_key_context()) or "done", tool_name = "t"))
    assert seen == [True]
    assert stream(stream_tool_execution) is stream_tool_execution


@pytest.mark.parametrize(
    ("name", "arguments", "grants"),
    [
        ("python", {"code": _HOST_CODE}, True),
        ("python", json.dumps({"code": _HOST_CODE}), True),
        ("python", {"code": _SANDBOXED_CODE}, False),
        ("terminal", {"command": "cat /home/someone/notes.txt"}, True),
        ("terminal", {"command": "ls"}, False),
        ("web_search", {"query": "/home/someone"}, False),
        ("python", "not json", False),
    ],
)
def test_approval_grants_host_access_matches_the_executor(name, arguments, grants):
    assert tools.approval_grants_host_access(name, arguments) is grants


@pytest.mark.skipif(sys.platform == "win32", reason = "terminal runs bash here")
@pytest.mark.parametrize("api_key", [True, False])
def test_an_approved_api_key_terminal_call_refuses_when_the_sandbox_is_gone(
    monkeypatch, tmp_path, api_key
):
    """The terminal holds like python: an approved host-path command from an API key never falls back to the host."""
    plans, fallbacks = [], []
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid: str(tmp_path))
    monkeypatch.setattr(tools, "_account_confinement", lambda: None)

    def gone(plan):
        plans.append(plan)
        raise os_sandbox.SandboxUnavailableError("sandbox went away")

    def safeguards(plan, reason, **_kw):
        fallbacks.append((plan.requested_mode, reason))
        raise os_sandbox.SandboxUnavailableError("stop before spawning")

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", gone)
    monkeypatch.setattr(tools, "_software_safeguards_launch", safeguards)
    token = bind_account(_API if api_key else OWNER)
    try:
        out = tools._bash_exec(
            "cat /home/someone/notes.txt",
            None,
            60,
            "__LOCALID_api_terminal",
            host_access_approved = True,
            tool_execution_mode = "auto",
        )
    finally:
        reset_account(token)
    assert "sandbox" in out.lower() or "stop before spawning" in out
    if api_key:
        assert [p.requested_mode for p in plans] == ["required"]
        assert fallbacks == []
    else:
        # A UI request keeps its approval: software safeguards off the workdir, as before.
        assert plans == []
        assert fallbacks == [("auto", "user_approved_host_access")]
