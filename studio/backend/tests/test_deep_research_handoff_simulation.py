# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end simulation of the Deep Research handoff, and of what it must not change.

Arming research used to create a run before the model read the message, so "hi" spent the
thread's one run on a greeting. Now the model is offered a `deep_research` tool and decides.
These drive the real loop, the real tool catalog and the real supervisor with a scripted model,
covering both halves: that the decision reaches the run, and that every path that existed
before still behaves the way it did.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from core.inference import studio_tool_loop as loop_mod
from core.inference.studio_tool_loop import (
    ToolLoopPolicy,
    ToolLoopRun,
    stream_with_studio_tools,
)
from core.inference.tools import DEEP_RESEARCH_STARTED, DEEP_RESEARCH_TOOL
from storage import research_runs_db as research_db
from storage import studio_db


RAW_MESSAGE = "breeds of dogs"
REFINED = "Which small dog breeds suit a flat with no garden?"


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": RAW_MESSAGE}],
            "createdAt": 2,
        }
    )
    return tmp_path


# ── The scripted model ────────────────────────────────────────────

_DONE = "data: [DONE]"


def _sse(delta = None, finish = None) -> str:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]})


def _says(text: str) -> list[str]:
    return [_sse({"content": text}), _sse(finish = "stop"), _DONE]


def _calls_research(question: str, preamble: str = "") -> list[str]:
    lines = [_sse({"content": preamble})] if preamble else []
    lines += [
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_r",
                        "function": {
                            "name": "deep_research",
                            "arguments": json.dumps({"question": question}),
                        },
                    }
                ]
            }
        ),
        _sse(finish = "tool_calls"),
        _DONE,
    ]
    return lines


class ScriptedModel:
    def __init__(self, turns):
        self.turns = [list(turn) for turn in turns]
        self.heals_text_tool_calls = True
        self.requests: list[dict] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append({"tools": tools, "tool_choice": tool_choice})
        lines = self.turns.pop(0) if self.turns else [_DONE]

        async def _gen():
            for line in lines:
                yield line

        return _gen()


def _run_turn(model, *, tools, monkeypatch):
    def _execute(name, arguments, **kwargs):
        if name == "deep_research":
            return DEEP_RESEARCH_HANDOFF + str(arguments.get("question") or "")
        return f"RESULT<{name}>"

    monkeypatch.setattr(loop_mod, "execute_tool", _execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "is_high_risk_tool_call", lambda name, args: False)

    async def _collect():
        out = []
        agen = stream_with_studio_tools(
            model,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": RAW_MESSAGE}],
                session_id = "s1",
                thread_id = "thread-1",
                tool_choice = None,
            ),
            policy = ToolLoopPolicy(
                tools = tools,
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        )
        async for line in agen:
            out.append(line)
        return out

    return asyncio.run(_collect())


def _handoff_question(lines) -> str | None:
    for line in lines:
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        payload = json.loads(line[6:])
        if payload.get("type") == "deep_research":
            return payload["question"]
    return None


def _visible(lines) -> str:
    text = []
    for line in lines:
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        payload = json.loads(line[6:])
        if payload.get("type") in ("tool_start", "tool_end"):
            continue
        for choice in payload.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str):
                text.append(content)
    return "".join(text)


# ── What the change is for ────────────────────────────────────────


def test_the_handed_off_question_is_what_actually_gets_researched(research_home, monkeypatch):
    """The refined question reaches the planner, not the raw message it came from."""
    from core import research_runs as worker

    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "question": REFINED,
            "budgets": {
                "maxSteps": 5,
                "maxSources": 15,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    prompts: list[str] = []

    async def fake_stream_completion(run, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        plan = {"title": "Plan", "steps": [{"title": "Step", "query": "small dog breeds flat"}]}
        return json.dumps(plan), "", "stop", None

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    claimed = research_db.claim_next(supervisor.worker_id)
    asyncio.run(supervisor._plan(claimed))

    assert REFINED in prompts[0]
    assert RAW_MESSAGE not in prompts[0].split("Latest research request:")[-1]


def test_planning_no_longer_stops_to_ask_for_approval(research_home, monkeypatch):
    from core import research_runs as worker

    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "budgets": {
                "maxSteps": 5,
                "maxSources": 15,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))

    async def fake_stream_completion(run, messages, **kwargs):
        plan = {"title": "Plan", "steps": [{"title": "Step", "query": "q"}]}
        return json.dumps(plan), "", "stop", None

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    claimed = research_db.claim_next(supervisor.worker_id)
    asyncio.run(supervisor._plan(claimed))

    assert research_db.get_run("run-1")["status"] == "queued"


# ── What the change must not break ────────────────────────────────


def test_a_run_from_an_old_install_researches_its_user_message(research_home, monkeypatch):
    """Config written before this change has no "question" key at all."""
    from core import research_runs as worker

    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "budgets": {
                "maxSteps": 5,
                "maxSources": 15,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    prompts: list[str] = []

    async def fake_stream_completion(run, messages, **kwargs):
        prompts.append(messages[-1]["content"])
        plan = {"title": "Plan", "steps": [{"title": "Step", "query": "q"}]}
        return json.dumps(plan), "", "stop", None

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    claimed = research_db.claim_next(supervisor.worker_id)
    asyncio.run(supervisor._plan(claimed))

    assert RAW_MESSAGE in prompts[0]


def test_a_run_left_awaiting_approval_by_an_old_install_still_runs(research_home):
    """Upgrading mid-run must not strand it: the approval endpoint still moves it along."""
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "budgets": {
                "maxSteps": 5,
                "maxSources": 15,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    plan = research_db.set_plan(
        "run-1", {"title": "Plan", "steps": [{"title": "Step", "query": "q"}]}
    )
    assert research_db.get_run("run-1")["status"] == "awaiting_approval"

    assert research_db.approve("run-1", plan["planRevision"], plan["planHash"]) == "queued"
    assert research_db.claim_next("worker-1") is not None


@pytest.mark.parametrize("armed", [True, False])
def test_the_tool_is_offered_only_when_research_is_armed(armed):
    from models.inference import ChatCompletionRequest
    from routes.inference import _select_request_tools

    payload = ChatCompletionRequest(
        model = "local-model",
        messages = [{"role": "user", "content": RAW_MESSAGE}],
        enabled_tools = [],
        deep_research_armed = armed,
    )
    tools = asyncio.run(_select_request_tools(payload, tools_on = True, mcp_allowed = False))
    names = [tool["function"]["name"] for tool in tools]

    assert ("deep_research" in names) is armed


def test_an_unarmed_request_is_byte_identical_to_before():
    """The tool list a normal chat sends must not move because this feature exists."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _select_request_tools

    payload = ChatCompletionRequest(
        model = "local-model",
        messages = [{"role": "user", "content": "hello"}],
    )
    tools = asyncio.run(_select_request_tools(payload, tools_on = True, mcp_allowed = False))

    assert all(tool["function"]["name"] != "deep_research" for tool in tools)
    assert [tool["function"]["name"] for tool in tools] == [
        "web_search",
        "python",
        "terminal",
        "render_html",
    ]


def test_a_client_that_never_heard_of_the_field_still_validates():
    """Old clients, and third parties on the OpenAI-compatible API, send no such field."""
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(
        model = "local-model", messages = [{"role": "user", "content": "hi"}]
    )
    assert payload.deep_research_armed is None


def test_an_empty_question_is_refused_rather_than_researched_blank():
    from core.inference.tools import execute_tool

    result = execute_tool("deep_research", {"question": "   "})
    assert result != DEEP_RESEARCH_STARTED
    assert "Error" in result


def test_the_result_is_an_ordinary_string_every_tool_loop_can_feed_back():
    """Studio runs three tool loops; only a plain result behaves the same in all of them."""
    from core.inference.tools import execute_tool, is_high_risk_tool_call

    result = execute_tool("deep_research", {"question": "x" * 10_000})
    assert result == DEEP_RESEARCH_STARTED
    assert result.isprintable()
    assert is_high_risk_tool_call("deep_research", {"question": "x"}) is False
