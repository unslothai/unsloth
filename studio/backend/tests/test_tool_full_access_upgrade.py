# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException

from state import tool_approvals as approvals
from utils.account_context import AccountContext, OWNER, as_api_caller, run_as

# Reuse the neighboring scripted tool transports and authenticated client harness.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)


@pytest.fixture(autouse = True)
def clean(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS", raising = False)
    with approvals._lock:
        approvals._pending.clear()
    yield
    with approvals._lock:
        approvals._pending.clear()


def _slot(
    *,
    session = "chat-a",
    eligible = True,
    bind = True,
    cancel = None,
):
    permissions = approvals.LiveToolPermissions(session, cancel)
    approval_id = approvals.new_approval_id()
    slot = approvals.begin_tool_decision(session, approval_id)
    if bind:
        approvals.bind_permission_upgrade(slot, permissions, eligible = eligible)
    return approval_id, slot, permissions


def test_upgrade_is_atomic_and_local_to_one_bound_loop():
    approval_id, slot, permissions = _slot()
    other = approvals.LiveToolPermissions("chat-a")
    assert not approvals.resolve_tool_decision(
        approval_id, "allow", session_id = "wrong", full_access = True
    )
    assert not slot["event"].is_set()
    assert approvals.resolve_tool_decision(
        approval_id, "allow", session_id = "chat-a", full_access = True
    )
    assert slot["event"].is_set() and slot["decision"] == "allow"
    assert permissions.apply("ask", False) == ("full", True)
    assert other.apply("ask", False) == ("ask", False)
    assert not approvals.resolve_tool_decision(approval_id, "deny", session_id = "chat-a")


@pytest.mark.parametrize(
    "case", ["unbound", "image", "expired", "cancelled", "answered", "no-session"]
)
def test_unavailable_approval_never_grants_full_access(case):
    cancel = threading.Event()
    approval_id, slot, permissions = _slot(
        bind = case != "unbound", eligible = case != "image", cancel = cancel
    )
    if case == "expired":
        slot["reason"] = approvals.DECISION_EXPIRED
    if case == "cancelled":
        cancel.set()
    if case == "answered":
        assert approvals.resolve_tool_decision(approval_id, "deny", session_id = "chat-a")
    assert not approvals.resolve_tool_decision(
        approval_id,
        "allow",
        session_id = None if case == "no-session" else "chat-a",
        full_access = True,
    )
    assert permissions.apply("ask", False) == ("ask", False)
    if case != "answered":
        assert not slot["event"].is_set()


def test_another_account_cannot_answer_or_upgrade_the_slot():
    approval_id, slot, permissions = _slot()
    user = AccountContext("other", "other")
    assert not run_as(
        user,
        approvals.resolve_tool_decision,
        approval_id,
        "allow",
        session_id = "chat-a",
        full_access = True,
    )
    assert not slot["event"].is_set()
    assert permissions.apply("ask", False) == ("ask", False)


def test_ui_cannot_upgrade_an_api_origin_without_the_operator_opt_in(monkeypatch):
    approval_id, slot, permissions = run_as(as_api_caller(OWNER), _slot)
    with pytest.raises(HTTPException, match = "API keys"):
        approvals.resolve_tool_decision(approval_id, "allow", session_id = "chat-a", full_access = True)
    assert not slot["event"].is_set()
    assert permissions.apply("ask", False) == ("ask", False)
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS", "1")
    assert approvals.resolve_tool_decision(
        approval_id, "allow", session_id = "chat-a", full_access = True
    )


def test_route_acknowledges_only_a_real_upgrade_and_rejects_stale_requests():
    from test_api_key_tool_confirm import _client_for
    approval_id, _pending, permissions = _slot()
    with _client_for(OWNER) as client:
        response = client.post(
            "/api/inference/tool-confirm",
            json = {
                "approval_id": approval_id,
                "session_id": "chat-a",
                "decision": "allow",
                "permission_mode": "full",
            },
        )
        assert response.status_code == 200
        assert response.json() == {"resolved": True, "full_access_granted": True}
        assert permissions.apply("ask", False) == ("full", True)
        assert (
            client.post(
                "/api/inference/tool-confirm",
                json = {
                    "approval_id": approval_id,
                    "session_id": "chat-a",
                    "decision": "allow",
                    "permission_mode": "full",
                },
            ).status_code
            == 404
        )
        aid, _slot_state, ordinary = _slot()
        response = client.post(
            "/api/inference/tool-confirm",
            json = {
                "approval_id": aid,
                "session_id": "chat-a",
                "decision": "allow",
            },
        )
        assert response.json() == {"resolved": True}
        assert ordinary.apply("ask", False) == ("ask", False)


@pytest.mark.parametrize("account", [AccountContext("managed", "managed"), as_api_caller(OWNER)])
def test_route_rejects_disallowed_confirmers_without_resolving(account):
    from test_api_key_tool_confirm import _client_for

    approval_id, slot, permissions = _slot()
    with _client_for(account) as client:
        response = client.post(
            "/api/inference/tool-confirm",
            json = {
                "approval_id": approval_id,
                "session_id": "chat-a",
                "decision": "allow",
                "permission_mode": "full",
            },
        )
    assert response.status_code == 403
    assert not slot["event"].is_set()
    assert permissions.apply("ask", False) == ("ask", False)


def test_full_access_request_requires_allow_and_the_exact_session():
    from pydantic import ValidationError
    from models.inference import ToolConfirmRequest
    for patch in ({"decision": "deny", "session_id": "chat-a"}, {"decision": "allow"}):
        with pytest.raises(ValidationError):
            ToolConfirmRequest(permission_mode = "full", **patch)


def _run_two_calls(loop, monkeypatch, execute, answer):
    from core.inference import llama_cpp, safetensors_agentic, studio_tool_loop
    from test_api_key_tool_confirm import _tool_call_markup

    session = "chat-a"
    tool = {"type": "function", "function": {"name": "python"}}
    if loop == "safetensors":
        monkeypatch.setattr(safetensors_agentic, "wait_tool_decision", answer)
        turns = iter([_tool_call_markup("print(1)"), _tool_call_markup("print(2)"), "done"])

        def turn(_messages):
            yield next(turns)

        list(
            safetensors_agentic.run_safetensors_tool_loop(
                single_turn = turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [tool],
                execute_tool = execute,
                session_id = session,
                confirm_tool_calls = True,
                permission_mode = "ask",
                max_tool_iterations = 3,
            )
        )
        return
    if loop == "llama_cpp":
        from test_llama_cpp_tool_loop import (
            _backend_and_payloads,
            _run_tool_loop,
            _streamed_structured_tool_call,
            _sse,
            _done,
        )

        monkeypatch.setattr(llama_cpp, "wait_tool_decision", answer)
        monkeypatch.setattr("core.inference.tools.execute_tool", execute)
        backend, _ = _backend_and_payloads(
            monkeypatch,
            [
                _streamed_structured_tool_call("python", {"code": "print(1)"}, "call_1"),
                _streamed_structured_tool_call("python", {"code": "print(2)"}, "call_2"),
                [_sse({"content": "Done."}), _done()],
            ],
        )
        monkeypatch.setattr(backend, "count_chat_tokens", lambda *args, **kwargs: 20)
        _run_tool_loop(
            backend,
            [{"role": "user", "content": "hi"}],
            [tool],
            max_tool_iterations = 3,
            confirm_tool_calls = True,
            permission_mode = "ask",
            session_id = session,
        )
        return
    from test_studio_tool_loop import FakeTransport, _sse, _DONE

    monkeypatch.setattr(studio_tool_loop, "wait_tool_decision", answer)
    monkeypatch.setattr(studio_tool_loop, "execute_tool", execute)
    monkeypatch.setattr(studio_tool_loop, "build_rag_autoinject", lambda *a, **k: None)
    turns = []
    for index in (1, 2):
        call = {
            "index": 0,
            "id": f"c{index}",
            "function": {"name": "python", "arguments": json.dumps({"code": f"print({index})"})},
        }
        turns.append([_sse({"tool_calls": [call]}), _sse(finish = "tool_calls"), _DONE])
    turns.append([_sse({"content": "ok"}), _sse(finish = "stop"), _DONE])

    async def collect():
        async for _ in studio_tool_loop.stream_with_studio_tools(
            FakeTransport(turns),
            run = studio_tool_loop.ToolLoopRun(
                messages = [{"role": "user", "content": "hi"}], session_id = session
            ),
            policy = studio_tool_loop.ToolLoopPolicy(
                tools = [tool],
                max_calls = 3,
                timeout = 300,
                permission_mode = "ask",
                confirm_calls = True,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        ):
            pass

    asyncio.run(collect())


@pytest.mark.parametrize("loop", ["safetensors", "llama_cpp", "studio"])
@pytest.mark.parametrize("upgrade", [False, True])
def test_current_and_later_calls_use_confirmed_live_permissions(monkeypatch, loop, upgrade):
    from test_api_key_tool_confirm import _client_for

    asks, executions = [], []

    def execute(name, arguments, **kwargs):
        executions.append(kwargs)
        return "OK"

    with _client_for(OWNER) as client:

        def answer(slot, approval_id, *args, **kwargs):
            asks.append(approval_id)
            response = client.post(
                "/api/inference/tool-confirm",
                json = {
                    "approval_id": approval_id,
                    "session_id": "chat-a",
                    "decision": "allow",
                    **({"permission_mode": "full"} if upgrade else {}),
                },
            )
            assert response.status_code == 200, response.text
            if upgrade:
                assert response.json()["full_access_granted"] is True
            return approvals.wait_tool_decision(slot, approval_id, timeout = 0.1)

        _run_two_calls(loop, monkeypatch, execute, answer)
    assert len(asks) == (1 if upgrade else 2)
    assert len(executions) == 2
    assert [call["disable_sandbox"] for call in executions] == [upgrade, upgrade]
    if upgrade:
        assert all(call.get("tool_execution_mode") != "required" for call in executions)


def test_skill_preload_upgrade_uses_the_same_live_policy_for_later_mentions(monkeypatch):
    from core.inference import skill_mentions

    permissions = approvals.LiveToolPermissions("chat-a")
    asked = []
    monkeypatch.setattr(
        skill_mentions,
        "list_skills",
        lambda: [
            {"name": name, "valid": True, "shadowed": False, "enabled": True}
            for name in ("first", "second")
        ],
    )
    monkeypatch.setattr(
        skill_mentions, "read_skill_instructions", lambda name: f"Complete instructions for {name}"
    )

    def answer(slot, approval_id, **kwargs):
        asked.append(approval_id)
        assert approvals.resolve_tool_decision(
            approval_id, "allow", session_id = "chat-a", full_access = True
        )
        return approvals.wait_tool_decision(slot, approval_id, timeout = 0.1)

    monkeypatch.setattr(skill_mentions, "wait_tool_decision", answer)
    events = list(
        skill_mentions.load_mentioned_skills(
            [{"role": "user", "content": "@first @second"}],
            [{"function": {"name": "read_skill"}}],
            permission_mode = "ask",
            confirm_tool_calls = True,
            session_id = "chat-a",
            permission_upgrade = permissions,
        )
    )
    assert len(asked) == 1
    assert [event["name"] for event in events if event["status"] == "loaded"] == ["first", "second"]
    assert permissions.apply("ask", False) == ("full", True)


class _AnswerDuringPoll(threading.Event):
    """The first poll times out just as a confirmation lands: the race a Stop or expiry meets."""

    def __init__(self, answer):
        super().__init__()
        self._answer = answer

    def wait(self, timeout = None):
        answer, self._answer = self._answer, None
        if answer is not None:
            answer()
            return False
        return super().wait(timeout)


@pytest.mark.parametrize("full_access", [True, False])
def test_a_committed_allow_wins_over_a_concurrent_stop(full_access):
    cancel = threading.Event()
    approval_id, slot, permissions = _slot(cancel = cancel)

    def answer():
        assert approvals.resolve_tool_decision(
            approval_id, "allow", session_id = "chat-a", full_access = full_access
        )
        cancel.set()

    slot["event"] = _AnswerDuringPoll(answer)
    slot["event"].clear()
    verdict = approvals.wait_tool_decision(slot, approval_id, cancel_event = cancel)
    assert verdict == "allow"
    assert slot["reason"] == approvals.DECISION_ANSWERED
    assert permissions.apply("ask", False) == (("full", True) if full_access else ("ask", False))


@pytest.mark.parametrize("full_access", [True, False])
def test_a_settled_expiry_refuses_a_late_confirmation(full_access):
    approval_id, slot, permissions = _slot()
    slot["reason"] = approvals.DECISION_EXPIRED
    assert not approvals.resolve_tool_decision(
        approval_id, "allow", session_id = "chat-a", full_access = full_access
    )
    assert not slot["event"].is_set() and slot["decision"] is None
    assert permissions.apply("ask", False) == ("ask", False)


def test_gguf_skill_preload_and_tool_loop_share_one_live_permission():
    import ast

    source = (Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    fn = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "gguf_generate_with_tools"
    )
    passed = {
        call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id: {
            kw.arg: ast.unparse(kw.value) for kw in call.keywords
        }
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and isinstance(call.func, (ast.Name, ast.Attribute))
        and (getattr(call.func, "id", None) or getattr(call.func, "attr", None))
        in ("load_mentioned_skills", "generate_chat_completion_with_tools")
    }
    assert passed["load_mentioned_skills"]["permission_upgrade"] == "gguf_live_permissions"
    assert (
        passed["generate_chat_completion_with_tools"]["live_permissions"] == "gguf_live_permissions"
    )
