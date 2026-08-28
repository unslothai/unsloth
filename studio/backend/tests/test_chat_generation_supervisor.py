# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sqlite3
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
def durable_run(request):
    engine = getattr(request, "param", "gguf")
    model = "local.gguf" if engine == "gguf" else "local.safetensors"
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": model, "createdAt": 1}
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
            "model": model,
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


def _create_payload(content = "Hello"):
    return run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {"model": "local", "messages": [{"role": "user", "content": content}]},
    )


def _route_request(supervisor):
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor))
    )


@pytest.mark.asyncio
async def test_stop_closes_the_lifespan_wal_keeper():
    keeper = sqlite3.connect(":memory:")
    supervisor = ChatGenerationSupervisor(
        SimpleNamespace(state = SimpleNamespace()),
        wal_keeper = keeper,
    )

    await supervisor.stop()

    with pytest.raises(sqlite3.ProgrammingError, match = "closed database"):
        keeper.execute("SELECT 1")


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


@pytest.mark.parametrize(
    "is_mlx,durable_run,generation_run_id,completion_tokens,expected",
    [
        (True, True, "run-1", 8, "length"),
        (True, True, "run-1", 7, "stop"),
        (True, False, "run-1", 8, "stop"),
        (False, True, "run-1", 8, "stop"),
    ],
)
def test_only_durable_mlx_normalizes_stop_at_token_cap(
    is_mlx, durable_run, generation_run_id, completion_tokens, expected
):
    payload = SimpleNamespace(
        generation_run_id = generation_run_id,
        max_tokens = 8,
        max_completion_tokens = None,
    )
    stats = {"usage": {"completion_tokens": completion_tokens}}
    assert (
        inference._safetensors_finish_reason(stats, payload, is_mlx = is_mlx, durable_run = durable_run)
        == expected
    )


@pytest.mark.asyncio
async def test_create_route_schedules_producer_on_request_loop(monkeypatch):
    started = []
    supervisor = SimpleNamespace(
        start = lambda run_id, **identity: started.append((run_id, identity))
    )
    request = _route_request(supervisor)

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
    response = await run_routes.create_chat_generation_run(_create_payload(), request, "alice")
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
    request = _route_request(supervisor)
    monkeypatch.setattr(run_routes.db, "create_run", lambda **_kwargs: (terminal, False))
    response = await run_routes.create_chat_generation_run(_create_payload("Hi"), request, "alice")
    assert response["created"] is False


@pytest.mark.asyncio
async def test_background_producer_persists_chunks_and_completes(durable_run, monkeypatch):
    observed = []
    leaked = []
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
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: leaked.append(context))
    try:
        await supervisor._produce("run-1")
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)
    run = runs_db.get_run("run-1", "alice")
    assert leaked == []
    assert observed == [False]
    assert (run["status"], run["finishReason"]) == ("completed", "stop")
    assert [
        event["payload"] for event in runs_db.list_events("run-1") if event["type"] == "chunk"
    ] == chunks


async def _subscriber_sequences(after = 0):
    response = await run_routes.chat_generation_events(
        "run-1",
        SimpleNamespace(is_disconnected = AsyncMock(return_value = True)),
        after = after,
        last_event_id = None,
        current_subject = "alice",
    )
    raw = ""
    async for part in response.body_iterator:
        raw += part.decode() if isinstance(part, bytes) else part
    return [int(line[4:]) for line in raw.splitlines() if line.startswith("id: ")]


def _route_engine(monkeypatch, model, first, release):
    gguf = model.endswith(".gguf")

    def generate(*, stats_holder = None, **_kwargs):
        first.set()
        yield "A"
        assert release.wait(5)
        yield "AB"
        if stats_holder is None:
            yield {"type": "metadata", "finish_reason": "stop"}
        else:
            stats_holder["stats"] = {"usage": {"completion_tokens": 2}}

    llama = SimpleNamespace(
        is_loaded = gguf,
        model_identifier = model,
        base_url = "http://llama.test",
        effective_parallel_slots = 1,
        supports_tools = False,
        is_vision = False,
        _is_audio = False,
        context_length = None,
        generate_chat_completion = generate,
    )
    mlx = SimpleNamespace(
        active_model_name = model,
        models = {model: {"is_mlx": True, "chat_template_info": {"template": "chatml"}}},
        generate_chat_response = generate,
        reset_generation_state = lambda *_a, **_k: None,
    )
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: mlx)
    monkeypatch.setattr(inference, "_automatic_model_load_may_run", lambda: False)
    monkeypatch.setattr(
        inference, "_detect_safetensors_features", lambda *_a, **_k: {"supports_tools": False}
    )
    monkeypatch.setattr(inference, "_effective_enable_tools", lambda _payload: False)

    async def no_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", no_switch)


@pytest.mark.asyncio
@pytest.mark.parametrize("durable_run", ["gguf", "mlx"], indirect = True)
async def test_subscribers_detach_then_replay_the_same_engine_run(durable_run, monkeypatch):
    first_chunk, release = threading.Event(), threading.Event()
    _route_engine(monkeypatch, durable_run["requestPayload"]["model"], first_chunk, release)
    task = asyncio.create_task(
        ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    )
    if not await asyncio.to_thread(first_chunk.wait, 5):
        await task
        pytest.fail(str(runs_db.get_run("run-1", "alice")))
    while len(runs_db.list_events("run-1")) < 4:
        await asyncio.sleep(0.01)
    first, second = await asyncio.gather(_subscriber_sequences(), _subscriber_sequences())
    assert first == second == list(range(1, max(first) + 1))
    assert runs_db.get_run("run-1", "alice")["status"] == "running"
    release.set()
    await task
    tail = await _subscriber_sequences(after = max(first))
    run = runs_db.get_run("run-1", "alice")
    assert first + tail == list(range(1, run["lastEventSeq"] + 1))
    deltas = [
        event["payload"].get("choices", [{}])[0].get("delta", {}).get("content")
        for event in runs_db.list_events("run-1")
        if event["type"] == "chunk"
    ]
    assert [text for text in deltas if text] == ["A", "B"]


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
    supervisor.cancel("run-1")
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
async def test_a_caught_up_reconnect_to_a_settled_run_does_not_block(durable_run, monkeypatch):
    """Nothing left to replay must return at once, not after the 15s event wait.

    Otherwise a finished answer reads as still generating for the whole timeout and one of
    the event-wait workers is held for it.
    """

    async def body():
        yield 'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    async def fake(*_args, **_kwargs):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    settled = runs_db.get_run("run-1", "alice")
    assert settled["status"] == "completed"

    caught_up = await asyncio.wait_for(
        _subscriber_sequences(after = int(settled["lastEventSeq"])),
        timeout = 5,
    )
    assert caught_up == []
    # A client that is behind still gets the whole ledger.
    assert await _subscriber_sequences() == list(range(1, int(settled["lastEventSeq"]) + 1))


@pytest.mark.asyncio
async def test_streamed_error_outranks_cleanup_cancellation(durable_run, monkeypatch):
    """A backend failure must not be recorded as if the user pressed Stop.

    ``gguf_stream_chunks`` emits the error in band, follows it with ``[DONE]`` and then
    sets this same ``cancel_event`` from its ``finally`` because the stream did not
    complete. Nobody asked to cancel, so the run has to settle as ``failed`` carrying
    the diagnostic the user needs to act on.
    """

    async def body(cancel_event):
        try:
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield 'data: {"error": {"message": "Out of memory"}}\n\ndata: [DONE]\n\n'
        finally:
            cancel_event.set()

    async def fake(_payload, request, *_args, **_kwargs):
        return SimpleNamespace(
            status_code = 200,
            body_iterator = body(request.state.generation_cancel_event),
        )

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    await ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))._produce("run-1")
    run = runs_db.get_run("run-1", "alice")
    assert run["status"] == "failed"
    assert run["error"] == "Out of memory"
    assert run["finishReason"] != "cancelled"
    metadata = studio_db.get_chat_message("thread-1", "assistant-1")["metadata"]
    assert metadata["incomplete"] != {"reason": "cancelled"}


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


@pytest.mark.asyncio
async def test_shutdown_returns_even_when_a_producer_will_not_unwind(durable_run, monkeypatch):
    """A generator whose teardown blocks must not take uvicorn's shutdown with it.

    The grace period is bounded, but the gather after task.cancel() has to be too:
    an engine draining a subprocess inside aclose never completes its cancellation,
    and stop() would then wait on it forever.
    """
    import core.inference.chat_generation_runs as chat_generation_runs

    monkeypatch.setattr(chat_generation_runs, "_SHUTDOWN_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(chat_generation_runs, "_SHUTDOWN_CANCEL_SECONDS", 0.5)

    wedged = asyncio.Event()
    release = asyncio.Event()

    async def body():
        yield 'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        wedged.set()
        try:
            await release.wait()
        except (asyncio.CancelledError, GeneratorExit):
            await release.wait()
            raise
        yield "data: [DONE]\n\n"

    async def fake(_payload, _request, _subject, *, cancel_on_disconnect):
        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    supervisor.start("run-1", thread_id = "thread-1", model = "local.gguf")
    await asyncio.wait_for(wedged.wait(), 10)

    try:
        await asyncio.wait_for(supervisor.stop(), timeout = 10)
    finally:
        release.set()
        await asyncio.sleep(0)
