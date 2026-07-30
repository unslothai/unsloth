# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# The path gate is all that keeps /api and the login page off the public
# tunnel; these tests pin its deny-by-default behavior (unit level and in a
# real uvicorn server), plus the share-link kill switch and teardown paths.

import asyncio
import json
from pathlib import Path
import sys
import types as _types

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror test_preview_routes.py: the real `loggers` package pulls in heavy handlers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import httpx
from fastapi import FastAPI

import preview_public_server as pps
import preview_share_link as psl
import routes.preview as preview
import utils.preview_token as preview_token


_TEST_SECRET = b"unit-test-preview-secret-0123456789"


def _make_run(outputs: Path, name: str = "demorun") -> None:
    run = outputs / name
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    # The real preview router plus a private /api route that must stay gated.
    outputs = tmp_path / "outputs"
    _make_run(outputs)
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET)
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)

    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    application = FastAPI()
    application.include_router(preview.router, prefix = "/p")

    @application.get("/api/health")
    def _health():  # must never be reachable through the gate
        return {"service": "Unsloth UI Backend"}

    return application


# ── Path gate ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/p/demorun",
        "/p/demorun/v1/models",
        "/p/_health",
        "/p/_assets/logo.png",
    ],
)
def test_gate_allows_public_preview_paths(path):
    assert pps.is_public_preview_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "/api/health",
        "/api/settings/preview-link",
        "/",
        "/index.html",
        "/v1/chat/completions",
        "/mcp",
        # Bare /p is the *authenticated* listing route, not a share surface.
        "/p",
        # Near-misses that must not satisfy the prefix check.
        "/pp/demorun",
        "/private/p/demorun",
        "",
    ],
)
def test_gate_denies_everything_else(path):
    assert pps.is_public_preview_path(path) is False


def test_gate_returns_404_without_calling_the_app():
    called = {"hit": False}

    async def _app(scope, receive, send):
        called["hit"] = True

    sent = []

    async def _send(message):
        sent.append(message)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    gate = pps.PreviewOnlyGate(_app)
    asyncio.run(gate({"type": "http", "path": "/api/health"}, _receive, _send))

    assert called["hit"] is False
    assert sent[0]["status"] == 404
    assert json.loads(sent[1]["body"]) == {"detail": "Not found"}


def test_gate_closes_websockets():
    async def _app(scope, receive, send):
        raise AssertionError("websocket must not reach the app")

    sent = []

    async def _send(message):
        sent.append(message)

    gate = pps.PreviewOnlyGate(_app)
    asyncio.run(gate({"type": "websocket", "path": "/p/demorun"}, None, _send))

    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_gate_acks_lifespan_without_forwarding():
    # Forwarding would double-run the app's startup/shutdown in one process.
    async def _app(scope, receive, send):
        raise AssertionError("lifespan must not reach the app")

    messages = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def _receive():
        return messages.pop(0)

    async def _send(message):
        sent.append(message)

    gate = pps.PreviewOnlyGate(_app)
    asyncio.run(gate({"type": "lifespan"}, _receive, _send))

    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


# ── Real listener ─────────────────────────────────────────────────────────


def _serve_and_get(app, paths):
    async def _run():
        listener = pps.PublicPreviewListener()
        port = await listener.start(app)
        statuses = {}
        try:
            async with httpx.AsyncClient(base_url = f"http://127.0.0.1:{port}") as client:
                for path in paths:
                    statuses[path] = (await client.get(path)).status_code
        finally:
            await listener.stop()
        return statuses

    return asyncio.run(_run())


def test_listener_serves_only_preview_paths(app):
    token = preview_token.sign_preview_ref("demorun")
    statuses = _serve_and_get(
        app,
        [
            "/p/_health",
            f"/p/demorun?k={token}",
            "/api/health",
            "/",
        ],
    )
    assert statuses["/p/_health"] == 200
    assert statuses[f"/p/demorun?k={token}"] == 200
    # The private surface is unreachable even though it exists on the app.
    assert statuses["/api/health"] == 404
    assert statuses["/"] == 404


def test_listener_still_requires_a_capability_token(app):
    # The gate opens the path; the HMAC token is what authorizes the run.
    statuses = _serve_and_get(app, ["/p/demorun", "/p/demorun?k=not-a-valid-token"])
    assert statuses["/p/demorun"] == 404
    assert statuses["/p/demorun?k=not-a-valid-token"] == 404


def test_health_marker_matches_the_tunnel_probe(app):
    # start_preview_tunnel only advertises a URL when this exact marker answers.
    import cloudflare_tunnel as ct
    async def _run():
        listener = pps.PublicPreviewListener()
        port = await listener.start(app)
        try:
            async with httpx.AsyncClient(base_url = f"http://127.0.0.1:{port}") as client:
                return (await client.get(ct._PREVIEW_PROBE_PATH)).json()
        finally:
            await listener.stop()

    assert asyncio.run(_run())["service"] == ct._PREVIEW_PROBE_MARKER


def test_health_does_not_shadow_a_run_on_the_authenticated_app(app, tmp_path):
    # The probe lives in the gate, not the shared router: a run literally named
    # "_health" keeps its token-checked page on the authenticated app.
    from fastapi.testclient import TestClient

    _make_run(tmp_path / "outputs", name = "_health")
    token = preview_token.sign_preview_ref("_health")
    response = TestClient(app).get(f"/p/_health?k={token}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # And without a token it 404s like any other run, not like an open probe.
    assert TestClient(app).get("/p/_health").status_code == 404


def test_listener_start_is_idempotent(app):
    async def _run():
        listener = pps.PublicPreviewListener()
        first = await listener.start(app)
        second = await listener.start(app)
        try:
            return first, second, listener.port
        finally:
            await listener.stop()

    first, second, port = asyncio.run(_run())
    assert first == second == port


def test_listener_leaves_process_signal_handlers_alone(app):
    # asyncio.run puts the loop on the main thread, exactly where stock
    # uvicorn would swap SIGINT/SIGTERM handlers.
    import signal

    async def _run():
        before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        listener = pps.PublicPreviewListener()
        await listener.start(app)
        during = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
        await listener.stop()
        return before, during

    before, during = asyncio.run(_run())
    assert before == during


def test_listener_stop_clears_port(app):
    async def _run():
        listener = pps.PublicPreviewListener()
        await listener.start(app)
        await listener.stop()
        await listener.stop()  # idempotent: a second stop is a no-op
        return listener.port

    assert asyncio.run(_run()) is None


# ── Share link ────────────────────────────────────────────────────────────


class _FakeListener:
    def __init__(self, port = 51234):
        self._port = port
        self.started = 0
        self.stopped = 0

    async def start(self, app):
        self.started += 1
        return self._port

    async def stop(self):
        self.stopped += 1


class _FakeApp:
    def __init__(self, cloudflare_url = None):
        self.state = _types.SimpleNamespace(cloudflare_url = cloudflare_url)


@pytest.fixture
def link(monkeypatch):
    monkeypatch.setattr(psl, "get_preview_sharing_enabled", lambda: True)
    return psl.PreviewShareLink()


def test_ensure_refuses_when_sharing_is_off(monkeypatch, link):
    monkeypatch.setattr(psl, "get_preview_sharing_enabled", lambda: False)
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)

    with pytest.raises(psl.PreviewSharingDisabled):
        asyncio.run(link.ensure(_FakeApp()))

    # Kill switch short-circuits before anything is bound or spawned.
    assert fake.started == 0


def test_ensure_prefers_an_already_public_studio(monkeypatch, link):
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(
        psl, "start_preview_tunnel", lambda port: pytest.fail("must not start a 2nd tunnel")
    )

    app = _FakeApp(cloudflare_url = "https://studio.trycloudflare.com")
    assert asyncio.run(link.ensure(app)) == "https://studio.trycloudflare.com"
    assert fake.started == 0


def test_ensure_starts_a_tunnel_to_the_listener_port(monkeypatch, link):
    fake = _FakeListener(port = 45678)
    monkeypatch.setattr(psl, "listener", fake)
    seen = {}

    def _start(port):
        seen["port"] = port
        return "https://preview.trycloudflare.com"

    monkeypatch.setattr(psl, "start_preview_tunnel", _start)

    app = _FakeApp()
    assert asyncio.run(link.ensure(app)) == "https://preview.trycloudflare.com"
    assert seen["port"] == 45678
    assert link.current(app) == "https://preview.trycloudflare.com"


def test_ensure_reuses_the_open_tunnel(monkeypatch, link):
    monkeypatch.setattr(psl, "listener", _FakeListener())
    calls = {"n": 0}

    def _start(port):
        calls["n"] += 1
        return "https://preview.trycloudflare.com"

    monkeypatch.setattr(psl, "start_preview_tunnel", _start)

    app = _FakeApp()
    asyncio.run(link.ensure(app))
    asyncio.run(link.ensure(app))
    assert calls["n"] == 1


def test_ensure_unbinds_the_listener_when_the_tunnel_fails(monkeypatch, link):
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(psl, "start_preview_tunnel", lambda port: None)

    with pytest.raises(psl.PreviewLinkUnavailable):
        asyncio.run(link.ensure(_FakeApp()))

    # No dangling listener when there is no tunnel in front of it.
    assert fake.stopped == 1
    assert link.current(_FakeApp()) is None


def test_stop_tears_down_tunnel_and_listener(monkeypatch, link):
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(psl, "start_preview_tunnel", lambda port: "https://x.trycloudflare.com")
    stopped = {"n": 0}
    monkeypatch.setattr(
        psl, "stop_studio_tunnel", lambda: stopped.__setitem__("n", stopped["n"] + 1)
    )

    app = _FakeApp()
    asyncio.run(link.ensure(app))
    asyncio.run(link.stop())

    assert stopped["n"] == 1
    assert fake.stopped == 1
    assert link.current(app) is None


def test_ensure_registers_an_atexit_backstop(monkeypatch, link):
    # Covers exits that bypass _graceful_shutdown (plain sys.exit): the quick
    # tunnel must not outlive the studio process.
    monkeypatch.setattr(psl, "listener", _FakeListener())
    monkeypatch.setattr(psl, "start_preview_tunnel", lambda port: "https://x.trycloudflare.com")
    registered = []
    monkeypatch.setattr(psl.atexit, "register", registered.append)

    asyncio.run(link.ensure(_FakeApp()))

    assert registered == [psl.stop_studio_tunnel]


def test_current_hides_the_base_while_sharing_is_off(monkeypatch, link):
    app = _FakeApp(cloudflare_url = "https://studio.trycloudflare.com")
    assert link.current(app) == "https://studio.trycloudflare.com"
    monkeypatch.setattr(psl, "get_preview_sharing_enabled", lambda: False)
    # Even a live studio-wide tunnel is not a share base: every /p request 404s.
    assert link.current(app) is None


def test_stop_without_a_link_leaves_the_shared_tunnel_slot_alone(monkeypatch, link):
    # The slot may hold a --cloudflare launch's studio-wide tunnel; teardown
    # must only touch a tunnel the share link itself started.
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(
        psl, "stop_studio_tunnel", lambda: pytest.fail("must not stop a tunnel it does not own")
    )

    asyncio.run(link.stop())

    assert fake.stopped == 0


def test_ensure_unbinds_the_listener_when_the_tunnel_thread_raises(monkeypatch, link):
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)

    def _boom(port):
        raise OSError("no exec permission for cloudflared")

    monkeypatch.setattr(psl, "start_preview_tunnel", _boom)

    with pytest.raises(OSError):
        asyncio.run(link.ensure(_FakeApp()))

    assert fake.stopped == 1
