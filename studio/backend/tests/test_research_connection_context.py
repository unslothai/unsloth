# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import unquote

import httpx
import pytest

from core.inference import external_provider
from storage import providers_db
from storage import research_runs_db as research_db
from storage import studio_db

STEPS = 8
WINDOW = 4_096
MODEL = "qwen3:1.7b"


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    # Other test modules leave a mock transport on the shared client.
    monkeypatch.setattr(external_provider, "_http_client", httpx.AsyncClient(trust_env = False))
    external_provider._served_windows.clear()
    external_provider._ollama_last_windows.clear()
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": MODEL,
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "how does the pump work?"}],
            "createdAt": 2,
        }
    )
    return tmp_path


@contextmanager
def _server(routes: dict[str, dict], required_headers: dict[str, str] | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = routes.get(self.path.split("?", 1)[0])
            for name, value in (required_headers or {}).items():
                if unquote(self.headers.get(name, "")) != value:
                    body = None
            raw = json.dumps(body if body is not None else {"error": "not found"}).encode()
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _llama_server(n_ctx: int | None) -> dict[str, dict]:
    entry = {"id": MODEL, "owned_by": "llamacpp", **({"meta": {"n_ctx": n_ctx}} if n_ctx else {})}
    return {"/v1/models": {"object": "list", "data": [entry]}, "/props": {}}


def _ollama(context_length: int) -> dict[str, dict]:
    return {
        "/api/ps": {"models": [{"name": MODEL, "model": MODEL, "context_length": context_length}]}
    }


def _web_results(step: int) -> str:
    return "\n\n---\n\n".join(
        f"Title: Result {step}-{rank}\nURL: https://example.test/{step}/{rank}\n"
        f"Snippet: web-{step}-{rank} " + "w" * 1_500
        for rank in range(5)
    )


def _prompt_chars(
    monkeypatch,
    provider_type: str,
    base_url: str,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, int]:
    from core import research_runs as worker

    providers_db.create_provider(
        id = "p1",
        provider_type = provider_type,
        display_name = "Connection",
        base_url = base_url,
        models = [MODEL],
        extra_headers = extra_headers,
    )
    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": MODEL,
            "inferenceRequest": {
                "model": MODEL,
                "providerType": provider_type,
                "providerId": "p1",
                "externalModel": MODEL,
            },
            "ragScope": None,
            "instructions": "",
            "question": "how does the pump work?",
            "budgets": {
                "maxSteps": STEPS,
                "maxSources": 40,
                "modelTimeoutSeconds": 900,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan(
        "run-1",
        {
            "title": "Pump",
            "steps": [{"title": f"pump {step}", "query": f"pump {step}"} for step in range(STEPS)],
        },
    )
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    searches = iter(range(STEPS))
    longest: dict[str, int] = {}

    async def fake_stream_completion(run, messages, **kwargs):
        phase = kwargs.get("phase")
        chars = sum(len(message["content"]) for message in messages)
        longest[phase] = max(longest.get(phase, 0), chars)
        if phase == "synthesis":
            return "## Findings\n\nThe pump moves water.", "", "stop", None
        return "not json", "", "stop", None

    monkeypatch.setattr(
        worker, "execute_tool", lambda *args, **kwargs: _web_results(next(searches))
    )
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(research_db.claim_next(supervisor.worker_id)))
    assert research_db.get_run("run-1")["status"] == "completed"
    return longest


@pytest.mark.parametrize(
    ("provider_type", "routes", "path"),
    [
        ("llama_cpp", _llama_server(WINDOW), "/v1"),
        ("custom", _llama_server(WINDOW), "/v1"),
        ("ollama", _ollama(WINDOW), ""),
    ],
    ids = ["llama_cpp", "custom", "ollama"],
)
def test_connection_prompts_fit_the_window_its_server_reports(
    research_home, monkeypatch, provider_type, routes, path
):
    from core.research_runs import _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN

    with _server(routes) as url:
        longest = _prompt_chars(monkeypatch, provider_type, url + path)

    assert {"decision", "synthesis"} <= set(longest)
    assert max(longest.values()) / _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN <= WINDOW


def test_window_probe_sends_the_templated_headers_of_the_run(research_home, monkeypatch):
    from core.research_runs import _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN

    required = {"X-Model": MODEL, "X-Session": "research:run-1"}
    with _server(_llama_server(WINDOW), required) as url:
        longest = _prompt_chars(
            monkeypatch,
            "llama_cpp",
            url + "/v1",
            {"X-Model": "{{MODEL}}", "X-Session": "{{CHAT_SESSION_ID}}"},
        )

    assert max(longest.values()) / _SYNTHESIS_EVIDENCE_CHARS_PER_TOKEN <= WINDOW


def test_window_probe_reads_the_connection_under_its_config_guard(research_home, monkeypatch):
    from routes import provider_credentials
    from storage import credential_secrets

    guarded = []
    read_key = credential_secrets.get_provider_api_key

    def guarded_read(provider_id):
        locks = [
            lock
            for per_loop in list(provider_credentials._provider_config_locks.values())
            for lock in per_loop.values()
        ]
        guarded.append(any(lock.locked() for lock in locks))
        return read_key(provider_id)

    monkeypatch.setattr(credential_secrets, "get_provider_api_key", guarded_read)
    with _server(_llama_server(WINDOW)) as url:
        _prompt_chars(monkeypatch, "llama_cpp", url + "/v1")

    assert guarded and all(guarded)


def test_connection_keeps_the_full_budget_when_its_server_reports_no_window(
    research_home, monkeypatch
):
    with _server(_llama_server(None)) as url:
        longest = _prompt_chars(monkeypatch, "llama_cpp", url + "/v1")

    assert longest["synthesis"] > 20_000


def test_hosted_connection_keeps_the_full_budget(research_home, monkeypatch):
    with _server(_llama_server(WINDOW)) as url:
        longest = _prompt_chars(monkeypatch, "openrouter", url + "/v1")

    assert longest["synthesis"] > 20_000
