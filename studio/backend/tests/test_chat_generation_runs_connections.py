# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved connections run durably: the server holds their key, so the reply outlives the tab."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from core.inference import llama_keepwarm
from core.inference.chat_generation_runs import ChatGenerationSupervisor, _background_request
from routes import chat_generation_runs as run_routes
from state import active_generations
from storage import chat_generation_runs_db as runs_db
from storage import credential_secrets, providers_db, studio_db
from utils.keyless_api_access import KEYLESS_ADMISSION_STATE_KEY

MODEL = "gpt-test"


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        server.requests.append((self.headers.get("Authorization"), json.loads(body)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for index in range(server.chunks):
                delta = {"choices": [{"index": 0, "delta": {"content": f"w{index} "}}]}
                self.wfile.write(f"data: {json.dumps(delta)}\n\n".encode())
                self.wfile.flush()
                server.sent += 1
                time.sleep(server.delay)
            self.wfile.write(
                b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
            self.wfile.flush()
            server.finished = True
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server:
    def __init__(
        self,
        chunks = 2,
        delay = 0.0,
    ) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
        self._httpd.daemon_threads = True
        self._httpd.requests = []
        self._httpd.chunks = chunks
        self._httpd.delay = delay
        self._httpd.sent = 0
        self._httpd.finished = False

    def __enter__(self):
        threading.Thread(target = self._httpd.serve_forever, daemon = True).start()
        return self._httpd

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def thread(monkeypatch):
    from routes import inference

    # A Stop stashes a pending cancel under the run id; a later test reusing "run-1" would inherit it.
    monkeypatch.setattr(inference, "_PENDING_CANCELS", {})
    studio_db.upsert_chat_thread(
        {"id": "thread-1", "title": "Chat", "modelType": "base", "modelId": MODEL, "createdAt": 1}
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
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def _connection(base_url = "http://127.0.0.1:9/v1", key = "sk-saved"):
    providers_db.create_provider(
        id = "conn-1",
        provider_type = "custom",
        display_name = "Connection",
        base_url = base_url,
        models = [MODEL],
    )
    if key:
        credential_secrets.save_provider_api_key("conn-1", key)


def _payload(**overrides):
    request = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
        "provider_id": "conn-1",
        "provider_type": "custom",
        "external_model": MODEL,
        "provider_base_url": "http://127.0.0.1:9/v1",
        "provider_api_type": "chat_completions",
        "enable_tools": False,
    }
    request.update(overrides)
    return run_routes.CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = {key: value for key, value in request.items() if value is not None},
    )


def _ui_request(
    supervisor = None,
    *,
    keyless = False,
    authorization = None,
):
    return SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor)),
        headers = {"authorization": authorization} if authorization else {},
        state = SimpleNamespace(**{KEYLESS_ADMISSION_STATE_KEY: keyless}),
    )


def test_a_saved_connection_persists_its_routing_but_never_a_key(thread):
    _connection()
    sanitized = run_routes._sanitize_request(_payload(), _ui_request())
    assert sanitized["provider_id"] == "conn-1"
    assert sanitized["external_model"] == MODEL
    # The recovered run reads these to parse think tags and match provider compaction.
    assert sanitized["provider_type"] == "custom"
    assert sanitized["provider_base_url"] == "http://127.0.0.1:9/v1"
    assert "encrypted_api_key" not in sanitized
    assert "extra_headers" not in sanitized


@pytest.mark.parametrize(
    "overrides,enabled",
    [
        ({"encrypted_api_key": "browser-held"}, True),
        ({"extra_headers": {"X-Tenant": "browser-held"}}, True),
        ({"provider_id": None}, True),
        ({"external_model": None}, True),
        ({"provider_id": "missing"}, True),
        ({}, False),
    ],
    ids = ["browser-key", "browser-headers", "no-connection", "no-model", "unknown", "disabled"],
)
def test_a_turn_the_server_cannot_replay_falls_back_to_the_live_stream(thread, overrides, enabled):
    _connection()
    if not enabled:
        providers_db.update_provider("conn-1", is_enabled = False)
    with pytest.raises(HTTPException) as exc_info:
        run_routes._sanitize_request(_payload(**overrides), _ui_request())
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        400,
        run_routes.EXTERNAL_LEGACY_DETAIL,
    )


def test_a_keyless_caller_cannot_spend_a_saved_connection(thread):
    _connection()
    with pytest.raises(HTTPException) as exc_info:
        run_routes._sanitize_request(_payload(), _ui_request(keyless = True))
    assert exc_info.value.status_code == 403


def test_an_api_key_caller_keeps_the_saved_key_gate(thread):
    _connection()
    with pytest.raises(HTTPException) as exc_info:
        run_routes._sanitize_request(
            _payload(), _ui_request(authorization = "Bearer sk-unsloth-not-a-real-key")
        )
    assert exc_info.value.detail == run_routes.EXTERNAL_LEGACY_DETAIL


def test_only_a_connection_run_records_its_caller_as_credentialed():
    event = threading.Event()
    external = _background_request(None, "run-1", event, external = True)
    local = _background_request(None, "run-1", event)
    assert getattr(external.state, KEYLESS_ADMISSION_STATE_KEY) is False
    assert not hasattr(local.state, KEYLESS_ADMISSION_STATE_KEY)


async def _create(supervisor):
    return await run_routes.create_chat_generation_run(_payload(), _ui_request(supervisor), "alice")


@pytest.mark.asyncio
async def test_a_stream_that_sets_the_run_event_in_teardown_still_completes(thread, monkeypatch):
    from routes import inference

    async def fake(_payload, request, _subject, *, cancel_on_disconnect):
        event = request.state.generation_cancel_event

        async def body():
            # The provider proxy's shape: [DONE], then `finally: cancel_event.set()`.
            try:
                yield 'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                yield "data: [DONE]\n\n"
            finally:
                event.set()

        return SimpleNamespace(status_code = 200, body_iterator = body())

    monkeypatch.setattr(inference, "produce_openai_chat_completions", fake)
    _connection()
    supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
    await _create(supervisor)
    await asyncio.wait_for(supervisor._tasks["run-1"], 10)
    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("completed", "stop")


def _chunk_text(run_id = "run-1"):
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for event in runs_db.list_events(run_id)
        if event["type"] == "chunk"
        for choice in event["payload"].get("choices", [])
    )


@pytest.mark.asyncio
async def test_a_connection_run_finishes_with_no_tab_attached(thread):
    with _Server(chunks = 3) as upstream:
        _connection(f"http://127.0.0.1:{upstream.server_address[1]}/v1")
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
        # Another request in flight: the run must not drop its count from the keep-warm tally.
        llama_keepwarm._note_start()
        try:
            await _create(supervisor)
            await asyncio.wait_for(supervisor._tasks["run-1"], 30)
            assert llama_keepwarm._inflight == 1
        finally:
            llama_keepwarm._note_end()
    run = runs_db.get_run("run-1", "alice")
    # The proxy sets the shared cancel event in its teardown; that must not read as a Stop.
    assert (run["status"], run["finishReason"], run["error"]) == ("completed", "stop", None)
    assert _chunk_text() == "w0 w1 w2 "
    authorization, body = upstream.requests[-1]
    assert authorization == "Bearer sk-saved"
    assert body["model"] == MODEL


@pytest.mark.asyncio
async def test_stop_reaches_a_connection_run_and_closes_the_upstream(thread):
    with _Server(chunks = 400, delay = 0.02) as upstream:
        _connection(f"http://127.0.0.1:{upstream.server_address[1]}/v1")
        supervisor = ChatGenerationSupervisor(SimpleNamespace(state = SimpleNamespace()))
        await _create(supervisor)
        deadline = time.monotonic() + 20
        while not _chunk_text() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert _chunk_text()
        # Registered as external, so a local model swap neither waits for nor cancels it.
        assert active_generations.count(None) == 1
        assert active_generations.count(None, include_external = False) == 0
        run_routes.cancel_chat_generation_run("run-1", _ui_request(supervisor), "alice")
        await asyncio.wait_for(supervisor._tasks["run-1"], 20)
        run = runs_db.get_run("run-1", "alice")
        assert run["status"] == "cancelled"
        await asyncio.sleep(0.3)
        assert not upstream.finished
        assert upstream.sent < 400
