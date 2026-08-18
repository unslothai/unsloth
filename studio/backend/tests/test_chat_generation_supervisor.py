# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from core.inference import llama_keepwarm
from core.inference.chat_generation_runs import ChatGenerationSupervisor
from models.inference import ChatCompletionRequest
from routes import chat_generation_runs as run_routes
from routes import inference
from state import active_generations
from storage import chat_generation_runs_db as runs_db
from storage import studio_db


@pytest.fixture
def durable_run(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )
    run, _created = runs_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "cancel_id": "run-1",
            "thread_id": "thread-1",
            "generation_run_id": "run-1",
        },
    )
    active_generations.reset_for_tests()
    yield run
    active_generations.reset_for_tests()


@pytest.mark.asyncio
async def test_public_chat_wrapper_keeps_cancel_on_disconnect(monkeypatch):
    observed = []

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        observed.append(cancel_on_disconnect)
        return "response"

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    payload = ChatCompletionRequest(
        model = "local",
        messages = [{"role": "user", "content": "Hello"}],
    )
    assert await inference.openai_chat_completions(payload, object(), "alice") == "response"
    assert observed == [True]


@pytest.mark.asyncio
async def test_create_route_schedules_producer_on_request_loop(monkeypatch):
    started = []
    supervisor = SimpleNamespace(
        start = lambda run_id, **identity: started.append((run_id, identity))
    )
    request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor))
    )
    payload = run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {
            "model": "local",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    def create_run(**_kwargs):
        assert started == []
        return (
            {
                "id": "run-1",
                "status": "queued",
                "threadId": "thread-1",
                "requestPayload": {"model": "local"},
            },
            True,
        )

    monkeypatch.setattr(run_routes.db, "create_run", create_run)
    response = await run_routes.create_chat_generation_run(payload, request, "alice")
    assert response["created"] is True
    assert started == [("run-1", {"thread_id": "thread-1", "model": "local"})]


@pytest.mark.asyncio
async def test_terminal_idempotent_create_does_not_reserve_generation(monkeypatch):
    terminal = {
        "id": "run-1",
        "status": "completed",
        "threadId": "thread-1",
        "requestPayload": {"model": "local"},
    }
    supervisor = SimpleNamespace(
        start = lambda *_args, **_kwargs: pytest.fail("terminal run must not start"),
    )
    request = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor))
    )
    payload = run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {"model": "local", "messages": [{"role": "user", "content": "Hi"}]},
    )
    monkeypatch.setattr(run_routes.db, "create_run", lambda **_kwargs: (terminal, False))
    response = await run_routes.create_chat_generation_run(payload, request, "alice")
    assert response["created"] is False


@pytest.mark.asyncio
async def test_background_producer_persists_chunks_and_completes(durable_run, monkeypatch):
    observed = []
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]

    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        observed.append(cancel_on_disconnect)
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    await supervisor._produce("run-1")

    run = runs_db.get_run("run-1", "alice")
    assert observed == [False]
    assert (run["status"], run["finishReason"]) == ("completed", "stop")
    assert [event["payload"] for event in runs_db.list_events("run-1") if event["type"] == "chunk"] == chunks


@pytest.mark.asyncio
async def test_event_batch_flushes_while_upstream_is_idle(durable_run, monkeypatch):
    release = asyncio.Event()
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
    ]

    async def body():
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        await release.wait()
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    task = asyncio.create_task(supervisor._produce("run-1"))
    await asyncio.sleep(0.2)
    stored = [e["payload"] for e in runs_db.list_events("run-1") if e["type"] == "chunk"]
    assert stored == chunks
    release.set()
    await task


@pytest.mark.asyncio
async def test_model_lifecycle_cancel_reaches_same_registered_event(durable_run, monkeypatch):
    registered = asyncio.Event()

    async def body(cancel_event):
        with active_generations.ActiveGeneration(
            cancel_event,
            thread_id = "thread-1",
            run_id = "run-1",
        ):
            registered.set()
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
        if False:
            yield ""

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(
            status_code = 200,
            body_iterator = body(request.state.generation_cancel_event),
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    task = asyncio.create_task(supervisor._produce("run-1"))
    await asyncio.wait_for(registered.wait(), timeout = 2)
    assert active_generations.cancel_all() == 1
    await asyncio.wait_for(task, timeout = 2)
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"
    metadata = studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]
    assert metadata["incomplete"] == {"reason": "cancelled"}
    assert active_generations.count() == 0


@pytest.mark.asyncio
async def test_cancel_before_registration_signals_load_event(durable_run, monkeypatch):
    entered = asyncio.Event()
    cancel_ids = []
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))

    async def fake(_payload, request, *_args, **_kwargs):
        event = request.state.generation_cancel_event
        entered.set()
        while not event.is_set():
            await asyncio.sleep(0.01)

        async def body():
            if False:
                yield ""

        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    monkeypatch.setattr(
        inference,
        "_cancel_by_cancel_id_or_stash",
        lambda run_id: cancel_ids.append(run_id) or 0,
    )
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    supervisor.cancel("run-1")
    await asyncio.wait_for(supervisor._tasks["run-1"], timeout = 2)
    assert cancel_ids == ["run-1"]
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_start_reserves_slot_and_lifecycle_before_worker_runs(durable_run, monkeypatch):
    monkeypatch.setattr(llama_keepwarm, "_pending", 0)
    monkeypatch.setattr(llama_keepwarm, "_inflight", 0)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1", thread_id = "thread-1", model = "local")
    assert (llama_keepwarm._pending, active_generations.count()) == (1, 1)
    assert active_generations.snapshot()[0]["thread_id"] == "thread-1"
    assert active_generations.cancel_all() == 1
    await asyncio.wait_for(supervisor._tasks["run-1"], timeout = 2)
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"
    assert (llama_keepwarm._pending, llama_keepwarm._inflight) == (0, 0)


@pytest.mark.asyncio
async def test_cancelled_producer_error_is_cancelled(durable_run, monkeypatch):
    entered = asyncio.Event()

    async def fake(_payload, request, *_args, **_kwargs):
        entered.set()
        while not request.state.generation_cancel_event.is_set():
            await asyncio.sleep(0.01)
        raise RuntimeError("Generation cancelled")

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    assert active_generations.cancel_all() == 1
    await asyncio.wait_for(supervisor._tasks["run-1"], timeout = 2)
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelled"


@pytest.mark.asyncio
async def test_uncancelled_partial_eof_is_interrupted(durable_run, monkeypatch):
    async def body():
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")


@pytest.mark.asyncio
async def test_graceful_supervisor_shutdown_is_interrupted(durable_run, monkeypatch):
    entered = asyncio.Event()

    async def body(request):
        entered.set()
        while not request.state.generation_cancel_event.is_set():
            await asyncio.sleep(0.01)
        if False:
            yield ""

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body(request))

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1")
    await asyncio.wait_for(entered.wait(), timeout = 2)
    await supervisor.stop()
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
    assert run["error"] == "Studio shut down during generation"


def test_thread_delete_captures_durable_run_before_cascade(durable_run):
    research_ids, chat_ids = studio_db.delete_chat_threads_with_active_runs(["thread-1"])
    assert research_ids == []
    assert chat_ids == ["run-1"]
    assert runs_db.get_run("run-1", "alice") is None


def test_project_delete_captures_durable_run_before_cascade(durable_run):
    studio_db.upsert_chat_project(
        {"id": "project-1", "name": "Project", "createdAt": 1, "updatedAt": 1}
    )
    studio_db.update_chat_thread("thread-1", {"projectId": "project-1"})
    deleted = studio_db.delete_chat_project("project-1")
    assert deleted["activeChatGenerationRunIds"] == ["run-1"]


def test_clear_captures_durable_run_before_cascade(durable_run):
    removed, research_ids, chat_ids = studio_db.clear_chat_history(
        include_chat_generation_runs = True
    )
    assert (removed, research_ids, chat_ids) == (["thread-1"], [], ["run-1"])


def test_startup_reconcile_marks_stored_assistant_interrupted(durable_run):
    worker_token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", worker_token)
    assert runs_db.reconcile_orphaned_runs() == 1
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"]["generationStatus"] == "failed"
    assert message["metadata"]["incomplete"] == {"reason": "interrupted"}
