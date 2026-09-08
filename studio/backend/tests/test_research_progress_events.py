# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every Deep Research model call must bracket itself with timeline events.

Planning, per-step decisions, and the synthesis audit run with thinking disabled and report
progress off. Without these brackets they emit nothing at all, and the UI showed a static
"0 sources, 0 actions" card for the whole call.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "R", "modelType": "base", "modelId": "m", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "What changed?"}],
            "createdAt": 2,
        }
    )
    return tmp_path


def _create():
    return research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "m",
            "inferenceRequest": {"model": "m"},
            "budgets": {
                "maxSteps": 2,
                "maxSources": 5,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
                "firstOutputTimeoutSeconds": 30,
            },
        },
    )


def _stub_transport(monkeypatch, worker, body: str):
    """Serve one non-streaming chunk plus [DONE] to every completion call."""

    class FakeResponse:
        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

        async def aiter_lines(self):
            chunk = json.dumps({"choices": [{"delta": {"content": body}, "finish_reason": "stop"}]})
            yield f"data: {chunk}"
            yield "data: [DONE]"

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, *, stream):
            return FakeResponse()

    monkeypatch.setattr(worker.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        worker.auth_storage, "create_api_key", lambda **kwargs: ("token", {"id": 1})
    )
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", lambda key_id: None)


def _events(run_id: str) -> list[dict]:
    return research_db.list_events(run_id, 0)


def test_planning_emits_a_phase_bracket(research_home, monkeypatch):
    from core import research_runs as worker

    _create()
    plan = {"title": "Plan", "steps": [{"title": "One", "query": "first query"}]}
    _stub_transport(monkeypatch, worker, json.dumps(plan))
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = research_db.claim_next(supervisor.worker_id)

    asyncio.run(supervisor._plan(run))

    types = [event["type"] for event in _events("run-1")]
    assert "phase.started" in types
    assert types.index("phase.started") < types.index("plan.ready")
    started = next(e for e in _events("run-1") if e["type"] == "phase.started")
    ended = next(e for e in _events("run-1") if e["type"] == "phase.ended")
    assert started["data"]["phase"] == "planning"
    assert started["data"]["callId"] == ended["data"]["callId"]


def test_phase_bracket_closes_when_the_call_fails(research_home, monkeypatch):
    from core import research_runs as worker

    _create()

    class ExplodingClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, *, stream):
            raise RuntimeError("backend gone")

    monkeypatch.setattr(worker.httpx, "AsyncClient", ExplodingClient)
    monkeypatch.setattr(
        worker.auth_storage, "create_api_key", lambda **kwargs: ("token", {"id": 1})
    )
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", lambda key_id: None)
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = research_db.claim_next(supervisor.worker_id)

    with pytest.raises(RuntimeError):
        asyncio.run(
            supervisor._stream_completion(run, [{"role": "user", "content": "q"}], phase = "decision")
        )

    types = [event["type"] for event in _events("run-1")]
    # A stuck row is worse than none: the bracket must close even when the call raises.
    assert types.count("phase.started") == 1
    assert types.count("phase.ended") == 1


def test_plan_titles_stream_before_the_plan_is_complete(research_home, monkeypatch):
    from core import research_runs as worker

    _create()
    plan = {
        "title": "Overall plan",
        "steps": [
            {"title": "Find the spec", "query": "spec"},
            {"title": "Check adoption", "query": "adoption"},
        ],
    }
    _stub_transport(monkeypatch, worker, json.dumps(plan))
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = research_db.claim_next(supervisor.worker_id)

    asyncio.run(supervisor._plan(run))

    labels = [
        event["data"]["label"] for event in _events("run-1") if event["type"] == "phase.progress"
    ]
    assert labels == ["Overall plan", "Find the spec", "Check adoption"]


def test_titles_split_across_tokens_still_publish(research_home, monkeypatch):
    from core import research_runs as worker

    _create()
    plan = {
        "title": "Overall plan",
        "steps": [{"title": "Find the spec", "query": "spec"}],
    }
    body = json.dumps(plan)

    class ChunkedResponse:
        def raise_for_status(self):
            return None

        async def aclose(self):
            return None

        async def aiter_lines(self):
            # Three chars per token, so a title's closing quote rarely lands on a boundary.
            for index in range(0, len(body), 3):
                chunk = json.dumps({"choices": [{"delta": {"content": body[index : index + 3]}}]})
                yield f"data: {chunk}"
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

    class ChunkedClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, *, stream):
            return ChunkedResponse()

    monkeypatch.setattr(worker.httpx, "AsyncClient", ChunkedClient)
    monkeypatch.setattr(
        worker.auth_storage, "create_api_key", lambda **kwargs: ("token", {"id": 1})
    )
    monkeypatch.setattr(worker.auth_storage, "revoke_internal_api_key", lambda key_id: None)
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = research_db.claim_next(supervisor.worker_id)

    asyncio.run(supervisor._plan(run))

    labels = [
        event["data"]["label"] for event in _events("run-1") if event["type"] == "phase.progress"
    ]
    assert labels == ["Overall plan", "Find the spec"]


def test_partial_titles_are_not_published(research_home, monkeypatch):
    from core import research_runs as worker

    # Only closed JSON strings count, so a title still being written never reaches the UI.
    assert worker._streamed_titles('{"title":"Complete","steps":[{"title":"Half') == ["Complete"]
    assert worker._streamed_titles('{"title":"Escaped \\"quoted\\" title"}') == [
        'Escaped "quoted" title'
    ]
    assert worker._streamed_titles("") == []


def test_decision_phase_bracket_carries_its_step_position(research_home, monkeypatch):
    from core import research_runs as worker

    _create()
    _stub_transport(monkeypatch, worker, "{}")
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    run = research_db.claim_next(supervisor.worker_id)

    asyncio.run(
        supervisor._stream_completion(
            run,
            [{"role": "user", "content": "q"}],
            phase = "decision",
            step_position = 3,
            report_progress = False,
        )
    )

    started = next(e for e in _events("run-1") if e["type"] == "phase.started")
    assert started["data"]["stepPosition"] == 3
    assert started["data"]["phase"] == "decision"


def test_event_stream_is_reachable_over_post_as_well_as_get():
    # Proxies that stream POST /v1/chat/completions still hold a streamed GET until it closes.
    from routes.research_runs import router
    events = [route for route in router.routes if route.path == "/{run_id}/events"]
    assert {method for route in events for method in route.methods} >= {"GET", "POST"}


def test_event_stream_verbs_do_not_share_one_operation_id():
    # A single api_route for both verbs gave them one operationId, which FastAPI warns about and
    # OpenAPI generators resolve by dropping one of the two operations.
    import warnings

    from fastapi import FastAPI
    from fastapi.openapi.utils import get_openapi

    from routes.research_runs import router

    app = FastAPI()
    app.include_router(router, prefix = "/api/chat/research-runs")
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        spec = get_openapi(title = "t", version = "1", routes = app.routes)
    assert not [w for w in caught if "Duplicate Operation ID" in str(w.message)]
    operations = spec["paths"]["/api/chat/research-runs/{run_id}/events"]
    assert list(operations) == ["post"]
