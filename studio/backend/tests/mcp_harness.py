# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve a Studio MCP server behind the real gate in front of a fake Studio, so tool tests drive the whole path (gate, caller, forwarder) without a GPU or the real routes. Built fresh per test: at the fastmcp floor an http_app's session manager runs once per instance."""

from __future__ import annotations

import asyncio
import base64
import builtins
import inspect
import json
import secrets
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional
from urllib.parse import parse_qsl

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import Mount

from studio_mcp import gate
from studio_mcp.caller import Caller
from studio_mcp.gate import StudioMcpGate

TEST_TOKEN = "sk-unsloth-test"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}
SENTINEL_ROOTS = ["/srv/mcp-sentinel-host", "C:\\mcp-sentinel"]
# Client kwargs for TestClient: a loopback agent may name host paths, a remote one may not.
LOCAL = {"base_url": "http://127.0.0.1:8888", "client": ("127.0.0.1", 50000)}
REMOTE = {"base_url": "http://192.168.1.20:8888", "client": ("192.0.2.7", 50000)}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 32
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64
WHISPER = "openai/whisper-small"
# /load and /unload failures that arrive late, inside a padded 200.
DEFERRED = [
    {"status_code": 409, "detail": "Another model is loading"},
    {"status_code": 500, "detail": "RuntimeError: CUDA out of memory"},
]
CONFIG = {
    "model_name": "unsloth/Qwen3-0.6B",
    "training_type": "LoRA/QLoRA",
    "format_type": "auto",
    "hf_dataset": "mlabonne/FineTome-100k",
    "max_steps": 30,
    "hf_token": "hf_cfg",
}
DIFFUSION = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "data_dir": "my-photos",
    "output_dir": "my-lora",
    "train_steps": 500,
}
OUT = "/srv/unsloth/outputs"
RUN_CHECKPOINTS = {
    "outputs_dir": OUT,
    "models": [
        {
            "name": "qwen-lora",
            "checkpoints": [
                {"display_name": "qwen-lora", "path": f"{OUT}/qwen-lora", "loss": 0.9},
                {
                    "display_name": "checkpoint-30",
                    "path": f"{OUT}/qwen-lora/checkpoint-30",
                    "loss": 1.1,
                },
            ],
            "base_model": "unsloth/Qwen3-0.6B",
            "peft_type": "LORA",
            "lora_rank": 16,
            "is_quantized": True,
        },
        {
            "name": "qwen-lora-2",
            "checkpoints": [
                {
                    "display_name": "checkpoint-30",
                    "path": f"{OUT}/qwen-lora-2/checkpoint-30",
                    "loss": 1.0,
                }
            ],
            "base_model": "unsloth/Qwen3-0.6B",
        },
    ],
}


def fake_studio(
    routes: Optional[dict[tuple[str, str], Any]] = None, *, fallback: Any = None
) -> FastAPI:
    """A Studio stand-in. A callable handler gets ``(request, body)`` and returns a Response or JSON; any other value is the answer itself. Every call is recorded in ``app.state.calls`` as ``(method, path, headers, body)``, and its query string in ``app.state.queries`` as ``(method, path, query)``. With ``fallback``, every other route answers it, so nothing the tool calls goes unrecorded."""
    app = FastAPI()
    app.state.calls = []
    app.state.queries = []

    def bind(method: str, path: str, handler: Any) -> None:
        async def endpoint(request: Request) -> Response:
            body = await request.body()
            app.state.calls.append((method, request.url.path, dict(request.headers), body))
            app.state.queries.append((method, request.url.path, request.url.query))
            result = handler
            if callable(handler) and not isinstance(handler, Response):
                result = handler(request, body)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, Response) else JSONResponse(result)

        app.add_api_route(path, endpoint, methods = [method])

    for (method, path), handler in (routes or {}).items():
        bind(method, path, handler)
    if fallback is not None:
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            bind(method, "/{rest:path}", fallback)
    return app


def sequence(*answers: Any) -> Callable[..., Any]:
    """A handler giving each answer in turn; the last one repeats."""
    left = list(answers)
    return lambda request, body: left.pop(0) if len(left) > 1 else left[0]


def slow(answer: Any, seconds: float) -> Callable[..., Any]:
    """A handler that gives ``answer`` after ``seconds``."""

    async def handler(request: Request, body: bytes) -> Any:
        await asyncio.sleep(seconds)
        return answer

    return handler


def _accept_test_key(token: str) -> tuple[Optional[dict], str]:
    if token == TEST_TOKEN:
        return {"account_id": "owner", "username": "unsloth", "role": "owner"}, ""
    return None, "Invalid or expired API key"


def served(
    mcp: Any,
    studio_app: Any,
    *,
    monkeypatch: Any,
    enabled: bool = True,
    validate: Callable[[str], tuple[Optional[dict], str]] = _accept_test_key,
) -> Starlette:
    monkeypatch.setattr(gate, "is_mcp_enabled", lambda: enabled)
    monkeypatch.setattr(gate, "_validate_key", validate)
    mcp_app = mcp.http_app(path = "/", stateless_http = True)
    app = Starlette(
        routes = [Mount("/mcp", StudioMcpGate(mcp_app)), Mount("/", studio_app)],
        lifespan = mcp_app.lifespan,
    )
    app.state.server_port = 8888
    app.state.cloudflare_url = None
    return app


def call_tool(
    client: Any,
    name: str,
    args: Optional[dict] = None,
    token: Optional[str] = TEST_TOKEN,
    headers: Optional[dict] = None,
) -> dict:
    """``tools/call`` over the streamable HTTP transport; returns the JSON-RPC ``result`` (or ``error``)."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args or {}},
    }
    sent = {**MCP_HEADERS, **(headers or {})}
    if token is not None:
        sent["Authorization"] = f"Bearer {token}"
    response = client.post("/mcp/", json = body, headers = sent)
    assert response.status_code == 200, response.text
    data = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
    message = json.loads(data[-1])
    return message.get("result", message.get("error"))


def poison(payload: Any) -> Any:
    """Suffix every string with both sentinel host paths, so a leak shows up wherever it comes from."""
    if isinstance(payload, str):
        return f"{payload} {SENTINEL_ROOTS[0]}/x {SENTINEL_ROOTS[1]}\\x"
    if isinstance(payload, dict):
        return {key: poison(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [poison(value) for value in payload]
    return payload


def run_tool(
    monkeypatch: Any,
    studio: Any,
    name: str,
    args: Optional[dict] = None,
    *,
    headers: Optional[dict] = None,
    fallback: Any = None,
    **client: Any,
) -> tuple[dict, FastAPI]:
    """Serve the Studio MCP server in front of ``studio`` (a fake_studio, or its routes and ``fallback``) and call one tool; returns ``(result, studio)``."""
    from mcp_server import create_studio_mcp

    studio = studio if isinstance(studio, FastAPI) else fake_studio(studio, fallback = fallback)
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch), **client) as http:
        return call_tool(http, name, args, headers = headers), studio


def call_with_progress(
    monkeypatch: Any,
    studio: FastAPI,
    name: str,
    args: dict,
    token: str = "p",
) -> tuple[list[dict], dict]:
    """Call a tool with a progress token; returns the progress notifications' params and the final result."""
    from mcp_server import create_studio_mcp

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": args, "_meta": {"progressToken": token}},
    }
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        response = http.post(
            "/mcp/", json = request, headers = {**MCP_HEADERS, "Authorization": f"Bearer {TEST_TOKEN}"}
        )
    messages = [
        json.loads(line[5:]) for line in response.text.splitlines() if line.startswith("data:")
    ]
    progress = [m["params"] for m in messages if m.get("method") == "notifications/progress"]
    return progress, messages[-1].get("result", messages[-1].get("error"))


def call_to(studio: FastAPI, path: str) -> tuple:
    return next(c for c in studio.state.calls if c[1] == path)


def bodies(studio: FastAPI, path: str) -> list[Any]:
    return [json.loads(c[3]) for c in studio.state.calls if c[1] == path]


def queries(studio: FastAPI, path: str) -> list[dict[str, str]]:
    return [
        dict(parse_qsl(q, keep_blank_values = True)) for _m, p, q in studio.state.queries if p == path
    ]


def form(studio: FastAPI, path: str) -> dict[str, tuple[Optional[str], bytes]]:
    """The multipart fields of the first call to ``path``, as ``{name: (filename, data)}``."""
    from email.parser import BytesParser
    from email.policy import default

    _m, _p, headers, body = call_to(studio, path)
    message = BytesParser(policy = default).parsebytes(
        b"Content-Type: " + headers["content-type"].encode() + b"\r\n\r\n" + body
    )
    return {
        part.get_param("name", header = "content-disposition"): (
            part.get_filename(),
            part.get_payload(decode = True),
        )
        for part in message.iter_parts()
    }


def openai_error(
    message: str,
    status: int,
    *,
    type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    headers: Optional[dict] = None,
) -> JSONResponse:
    error = {"message": message, "type": type, "param": param, "code": code}
    return JSONResponse({"error": error}, status_code = status, headers = headers)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{b64(data)}"


PNG_URL = {"data_url": data_url(PNG, "image/png")}
WAV_INPUT = {"data_base64": b64(WAV), "filename": "a.wav"}


def make_caller(*, cloudflare_url: Optional[str] = None, **fields: Any) -> Caller:
    # As the gate would admit a loopback agent; fields override any part.
    defaults = {
        "token": TEST_TOKEN,
        "account_id": "owner",
        "direct_local": True,
        "public_base": "http://127.0.0.1:8888",
        "studio_app": SimpleNamespace(state = SimpleNamespace(cloudflare_url = cloudflare_url)),
    }
    return Caller(**{**defaults, **fields})


def stt_state(
    *,
    downloaded: Any = (),
    loaded: Optional[str] = None,
    download: Optional[dict] = None,
    **fields: Any,
) -> dict:
    return {
        "loaded_model": loaded,
        "loading": False,
        "models": [WHISPER, "openai/whisper-large-v3"],
        "downloaded_models": list(downloaded),
        "download": download or {"downloading": False, "completed_download_ids": []},
        **fields,
    }


def register_export_job(
    account_id: str = "owner",
    job_id: str = "job-a",
    **fields: Any,
) -> Any:
    from studio_mcp import export_jobs

    job = export_jobs.ExportJob(job_id = job_id, account_id = account_id, format = "gguf", **fields)
    export_jobs._jobs[f"{account_id}:{job_id}"] = job
    return job


def run_export_job(monkeypatch: Any, studio: FastAPI, args: dict) -> tuple[dict, Optional[dict]]:
    """Serve ``studio``, start export_model, then poll get_job until the job settles; returns ``(started, job)``."""
    from mcp_server import create_studio_mcp
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        started = call_tool(http, "export_model", args)
        if started.get("isError"):
            return started, None
        job_id = started["structuredContent"]["job_id"]
        # By time, not a fixed count: back-to-back polls can beat the background job on a slow host.
        deadline = time.monotonic() + 30.0
        while True:
            job = call_tool(http, "get_job", {"kind": "export", "id": job_id})
            if job["structuredContent"]["status"] != "running" or time.monotonic() > deadline:
                return started, job
            time.sleep(0.02)


def route_studio(
    router: Any = None,
    prefix: str = "",
    *,
    subject: str = "unsloth",
    overrides: Optional[dict] = None,
) -> FastAPI:
    from auth.authentication import get_current_subject

    app = FastAPI()
    app.state.bind_host = "127.0.0.1"
    if router is not None:
        app.include_router(router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: subject
    app.dependency_overrides.update(overrides or {})
    return app


def _reset_auth_caches() -> None:
    from auth import policy, storage
    from utils import mcp_access
    from utils.keyless_api_access import _reset_scope_cache

    storage._reset_api_key_hash_cache()
    policy.invalidate_account_cache()
    _reset_scope_cache()
    mcp_access._reset_cache()


@pytest.fixture
def isolated_auth(tmp_path, monkeypatch):
    """An empty auth database of this test's own, with the MCP switch env unset and every cache reset."""
    from auth import storage
    from utils.mcp_access import ENV_FORCE

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    monkeypatch.delenv(ENV_FORCE, raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MCP_TOKEN", raising = False)
    _reset_auth_caches()
    yield
    _reset_auth_caches()


def seed_owner() -> None:
    from auth import storage
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )


@pytest.fixture
def fast_polls(monkeypatch):
    from studio_mcp import loading
    monkeypatch.setattr(loading, "POLL_INTERVAL_S", 0.01)


@pytest.fixture
def owner_auth(isolated_auth):
    # isolated_auth with the owner account created.
    seed_owner()


@pytest.fixture
def file_spy(monkeypatch):
    """Records every way the tool could open or inspect a file."""
    touched = []
    real_open, real_stat, real_read = builtins.open, Path.stat, Path.read_bytes

    def spy_open(file, *args, **kwargs):
        if "mcp-input" in str(file):
            touched.append(("open", str(file)))
        return real_open(file, *args, **kwargs)

    def spy_stat(self, *args, **kwargs):
        if "mcp-input" in str(self):
            touched.append(("stat", str(self)))
        return real_stat(self, *args, **kwargs)

    def spy_read(self):
        if "mcp-input" in str(self):
            touched.append(("read", str(self)))
        return real_read(self)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "stat", spy_stat)
    monkeypatch.setattr(Path, "read_bytes", spy_read)
    return touched
