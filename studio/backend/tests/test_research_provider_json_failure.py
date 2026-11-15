# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Provider-side JSON validation must reach research's existing output fallbacks."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core import research_runs as worker
from storage import research_runs_db as research_db
from storage import studio_db
from .test_research_no_evidence import _claimed_run, research_home


# SSE envelope from Strata's prompt-and-validate server (serve/server.py).
JSON_FAILURE = {
    "error": {
        "type": "structured_output_failed",
        "code": "structured_output_failed",
        "message": "model did not return valid JSON: Expecting ',' delimiter: line 1 column 1177 (char 1176)",
    }
}
REPORT = "## Findings\n\nThe source confirms the result. [Result](https://example.test/result)"


def _content(text):
    return {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}


@pytest.fixture
def completion_transport(monkeypatch):
    sent = []
    revoked = []
    real_client = httpx.AsyncClient
    monkeypatch.setattr(worker, "_loaded_context_length", lambda *args: 8192)
    monkeypatch.setattr(worker.auth_storage, "create_api_key", lambda **kwargs: ("test", {"id": 1}))
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", revoked.append)

    def install(replies):
        replies = iter(replies)

        def serve(request):
            sent.append(json.loads(request.content))
            chunks = next(replies)
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(200, text = body + "data: [DONE]\n\n")

        transport = httpx.MockTransport(serve)
        monkeypatch.setattr(
            worker.httpx, "AsyncClient", lambda **kwargs: real_client(transport = transport, **kwargs)
        )

    return SimpleNamespace(install = install, sent = sent, revoked = revoked)


@pytest.mark.parametrize("unused_seed", [True, False], ids = ["continue-plan", "synthesize-evidence"])
@pytest.mark.parametrize("partial_output", [False, True], ids = ["buffered", "partial-output"])
def test_provider_json_failure_preserves_research_and_finishes(
    research_home, monkeypatch, completion_transport, unused_seed, partial_output
):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    plan = [{"title": "First", "query": "first query"}]
    if unused_seed:
        plan.append({"title": "Second", "query": "second query"})
    claimed = _claimed_run(supervisor, plan, 2, None)
    searches = []

    def search(name, arguments, **kwargs):
        assert name == "web_search"
        searches.append(arguments["query"])
        return "Title: Result\nURL: https://example.test/result\nSnippet: The result is confirmed."

    monkeypatch.setattr(worker, "execute_tool", search)
    rejected = []
    if partial_output:
        rejected.append(_content('{"action":"search","query":"rejected query"}'))
    rejected.append(JSON_FAILURE)
    completion_transport.install(
        [
            [_content(json.dumps({"action": "search", **plan[0]}))],
            rejected,
            rejected,
            [_content(REPORT)],
        ]
    )
    asyncio.run(supervisor._process(claimed))

    finished = research_db.get_run("run-1")
    assert finished["status"] == "completed", finished["error"]
    assert finished["error"] is None
    assert REPORT in finished["report"]
    assert [source["url"] for source in finished["sources"]] == ["https://example.test/result"]
    assert searches == [step["query"] for step in plan]
    assert all(step["status"] == "completed" for step in finished["steps"])
    messages = studio_db.list_chat_messages("thread-1")
    assistant = next(message for message in reversed(messages) if message["role"] == "assistant")
    assert assistant["metadata"]["researchStatus"] == "completed"
    assert len(completion_transport.sent) == 4, "Rejected generations must not be replayed"
    assert completion_transport.revoked == [1] * 4
    for request in completion_transport.sent[:3]:
        assert request["response_format"] == {"type": "json_object"}
        assert request["tool_choice"] == "none"
        assert request["enabled_tools"] == []
    events = research_db.list_events("run-1", 0)
    assert sum(event["type"] == "phase.started" for event in events) == 4
    assert sum(event["type"] == "phase.ended" for event in events) == 4


@pytest.mark.parametrize(
    "phase,json_mode,error",
    [
        ("planning", True, JSON_FAILURE),
        ("synthesis", False, JSON_FAILURE),
        ("decision", False, JSON_FAILURE),
        ("decision", True, {"error": {"code": "server_error", "message": "backend died"}}),
        ("decision", True, {"error": {"code": "401", "message": "Invalid API key"}}),
        ("decision", True, {"error": {"message": JSON_FAILURE["error"]["message"]}}),
    ],
    ids = ["planning", "report", "no-json-mode", "server-error", "auth-error", "message-only"],
)
def test_other_errors_remain_failures(monkeypatch, completion_transport, phase, json_mode, error):
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(supervisor, "_check_active", noop)
    monkeypatch.setattr(supervisor, "_note_phase", noop)
    completion_transport.install([[error]])
    run = {
        "id": "run-1",
        "ownerSubject": "alice",
        "config": {"budgets": {"modelTimeoutSeconds": 10}},
    }
    with pytest.raises(RuntimeError, match = error["error"]["message"].split(":")[0]):
        asyncio.run(
            supervisor._stream_completion(
                run, [], phase = phase, json_mode = json_mode, report_progress = False
            )
        )
    assert len(completion_transport.sent) == 1
    assert completion_transport.revoked == [1]


def test_saved_connection_relays_the_provider_validation_code(monkeypatch):
    from core.inference import external_provider
    from .test_external_hosted_tool_selection import _install, _payload, _request

    route = _install(monkeypatch, "custom")
    monkeypatch.setattr(route, "ExternalProviderClient", external_provider.ExternalProviderClient)
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, text = f"data: {json.dumps(JSON_FAILURE)}\n\ndata: [DONE]\n\n")

    async def relay():
        async with httpx.AsyncClient(transport = httpx.MockTransport(serve)) as client:
            monkeypatch.setattr(external_provider, "_client", lambda: client)
            response = await route._proxy_to_external_provider(
                _payload(
                    enable_tools = False,
                    tool_choice = "none",
                    enabled_tools = [],
                    response_format = {"type": "json_object"},
                ),
                _request(),
                current_subject = "alice",
            )
            return "".join([part async for part in response.body_iterator])

    body = asyncio.run(relay())
    errors = [
        json.loads(line[5:].strip())
        for line in body.splitlines()
        if line.startswith("data:") and '"error"' in line
    ]
    assert errors == [JSON_FAILURE]
    assert len(sent) == 1
    assert sent[0]["response_format"] == {"type": "json_object"}
