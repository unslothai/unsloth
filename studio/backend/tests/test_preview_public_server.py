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


@pytest.mark.parametrize(
    "path",
    [
        # Paths only main.py's SPA catch-all would answer: the gate must 404
        # them, or the public port serves index.html (and, via dot segments,
        # the whole frontend build) to anyone.
        "/p/",
        "/p/a/b/c",
        "/p/demorun/ckpt/nope/deeper",
        "/p/../index.html",
        "/p/../assets/app.js",
        "/p/../api/health",
        "/p/./demorun",
        "/p/demorun/..",
    ],
)
def test_gate_denies_spa_catch_all_paths_before_the_app(path):
    async def _app(scope, receive, send):
        raise AssertionError("path must not reach the app")

    sent = []

    async def _send(message):
        sent.append(message)

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    gate = pps.PreviewOnlyGate(_app)
    scope = {"type": "http", "method": "GET", "path": path, "query_string": b"", "headers": []}
    asyncio.run(gate(scope, _receive, _send))

    assert sent[0]["status"] == 404
    assert json.loads(sent[1]["body"]) == {"detail": "Not found"}


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
            pps._PREVIEW_HEALTH_PATH,
            f"/p/demorun?k={token}",
            "/api/health",
            "/",
        ],
    )
    assert statuses[pps._PREVIEW_HEALTH_PATH] == 200
    assert statuses[f"/p/demorun?k={token}"] == 200
    # The private surface is unreachable even though it exists on the app.
    assert statuses["/api/health"] == 404
    assert statuses["/"] == 404


def test_listener_never_serves_the_spa_catch_all(app):
    # Mirror main.py's last-resort handler: any unmatched GET serves the SPA.
    from fastapi.responses import PlainTextResponse

    @app.get("/{full_path:path}")
    def _spa(full_path: str):
        return PlainTextResponse(f"SPA-INDEX {full_path}")

    token = preview_token.sign_preview_ref("demorun")
    statuses = _serve_and_get(
        app,
        [f"/p/demorun?k={token}", "/p/", "/p/a/b/c", "/p/x/y/z/w"],
    )
    assert statuses[f"/p/demorun?k={token}"] == 200
    assert statuses["/p/"] == 404
    assert statuses["/p/a/b/c"] == 404
    assert statuses["/p/x/y/z/w"] == 404


def test_listener_leaves_global_logging_alone(app):
    # uvicorn.Config would dictConfig() the process-wide loggers by default,
    # downgrading the primary server's logging when sharing is turned on.
    import logging

    uv_error = logging.getLogger("uvicorn.error")
    before = (uv_error.level, list(uv_error.handlers))

    async def _run():
        listener = pps.PublicPreviewListener()
        await listener.start(app)
        await listener.stop()

    asyncio.run(_run())
    assert (uv_error.level, list(uv_error.handlers)) == before


def test_listener_still_requires_a_capability_token(app):
    # The gate opens the path; the HMAC token is what authorizes the run.
    statuses = _serve_and_get(app, ["/p/demorun", "/p/demorun?k=not-a-valid-token"])
    assert statuses["/p/demorun"] == 404
    assert statuses["/p/demorun?k=not-a-valid-token"] == 404


def test_listener_rejects_unsupported_methods(app):
    # A 405 before the token check would reveal which routes exist; the gate
    # answers the same generic 404 instead.
    token = preview_token.sign_preview_ref("demorun")

    async def _run():
        listener = pps.PublicPreviewListener()
        port = await listener.start(app)
        try:
            async with httpx.AsyncClient(base_url = f"http://127.0.0.1:{port}") as client:
                put = await client.put(f"/p/demorun?k={token}")
                delete = await client.delete("/p/demorun/v1/models")
                options = await client.options(f"/p/demorun?k={token}")
        finally:
            await listener.stop()
        return put, delete, options

    put, delete, options = asyncio.run(_run())
    assert put.status_code == delete.status_code == options.status_code == 404


def test_gate_blocks_unauthenticated_chat_posts(app):
    # receive=None proves the body is never read: FastAPI would parse it
    # before the handler's token check, unmetered.
    called = {"hit": False}

    async def _app(scope, receive, send):
        called["hit"] = True

    sent = []

    async def _send(message):
        sent.append(message)

    gate = pps.PreviewOnlyGate(_app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/p/demorun/v1/chat/completions",
        "query_string": b"",
        "headers": [],
    }
    asyncio.run(gate(scope, None, _send))
    assert called["hit"] is False
    assert sent[0]["status"] == 404

    scope["query_string"] = b"k=not-a-valid-token"
    asyncio.run(gate(scope, None, _send))
    assert called["hit"] is False


def test_gate_forwards_tokened_chat_posts(app):
    token = preview_token.sign_preview_ref("demorun")
    calls = {"n": 0}

    async def _app(scope, receive, send):
        calls["n"] += 1

    gate = pps.PreviewOnlyGate(_app)

    async def _send(message):
        pass

    by_query = {
        "type": "http",
        "method": "POST",
        "path": "/p/demorun/v1/chat/completions",
        "query_string": f"k={token}".encode(),
        "headers": [],
    }
    asyncio.run(gate(by_query, None, _send))
    by_bearer = {
        "type": "http",
        "method": "POST",
        "path": "/p/demorun/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    asyncio.run(gate(by_bearer, None, _send))
    assert calls["n"] == 2


def test_listener_masks_method_mismatches_on_real_routes(app):
    # An allowed verb on the wrong route would 405 in FastAPI before the token
    # check; the gate rewrites that to the same generic 404.
    token = preview_token.sign_preview_ref("demorun")

    async def _run():
        listener = pps.PublicPreviewListener()
        port = await listener.start(app)
        try:
            async with httpx.AsyncClient(base_url = f"http://127.0.0.1:{port}") as client:
                get_on_chat = await client.get("/p/demorun/v1/chat/completions")
                post_on_page = await client.post(f"/p/demorun?k={token}")
        finally:
            await listener.stop()
        return get_on_chat, post_on_page

    get_on_chat, post_on_page = asyncio.run(_run())
    assert get_on_chat.status_code == 404
    assert post_on_page.status_code == 404


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
    # The probe lives in the gate, outside /p: a run literally named "_health"
    # keeps its token-checked page on the authenticated app.
    from fastapi.testclient import TestClient

    _make_run(tmp_path / "outputs", name = "_health")
    token = preview_token.sign_preview_ref("_health")
    response = TestClient(app).get(f"/p/_health?k={token}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # And without a token it 404s like any other run, not like an open probe.
    assert TestClient(app).get("/p/_health").status_code == 404


def test_signed_health_run_is_served_through_the_listener(app, tmp_path):
    # The reviewer scenario: a shared link for a run named "_health" must reach
    # the token-checked page on the public listener, not the probe.
    _make_run(tmp_path / "outputs", name = "_health")
    token = preview_token.sign_preview_ref("_health")
    statuses = _serve_and_get(app, [f"/p/_health?k={token}", "/p/_health"])
    assert statuses[f"/p/_health?k={token}"] == 200
    # Unsigned, it 404s like any run; the probe lives on its own path.
    assert statuses["/p/_health"] == 404


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


def test_ensure_waits_out_a_starting_studio_tunnel(monkeypatch, link):
    # Uvicorn serves before a --cloudflare launch finishes its startup tunnel;
    # ensure must wait for that outcome, not race it for the shared slot.
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(
        psl, "start_preview_tunnel", lambda port: pytest.fail("must not race the startup tunnel")
    )

    class _State:
        def __init__(self):
            self.cloudflare_url = None
            self._polls = 0

        @property
        def cloudflare_tunnel_pending(self):
            self._polls += 1
            if self._polls >= 2:
                self.cloudflare_url = "https://studio.trycloudflare.com"
                return False
            return True

    app = _types.SimpleNamespace(state = _State())
    assert asyncio.run(link.ensure(app)) == "https://studio.trycloudflare.com"
    assert fake.started == 0


def test_ensure_rechecks_the_kill_switch_after_waiting(monkeypatch, link):
    # A disable persists the setting before queuing behind ensure's lock; a
    # waiting create must fail, not return the freshly published studio URL.
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    state = {"on": True}
    monkeypatch.setattr(psl, "get_preview_sharing_enabled", lambda: state["on"])

    class _State:
        def __init__(self):
            self.cloudflare_url = "https://studio.trycloudflare.com"
            self._polls = 0

        @property
        def cloudflare_tunnel_pending(self):
            self._polls += 1
            if self._polls >= 2:
                state["on"] = False
                return False
            return True

    app = _types.SimpleNamespace(state = _State())
    with pytest.raises(psl.PreviewSharingDisabled):
        asyncio.run(link.ensure(app))
    assert fake.started == 0


def test_ensure_rechecks_the_startup_tunnel_after_binding_the_listener(monkeypatch, link):
    # A request can pass the first pending check before run.py arms the flag;
    # the recheck after listener.start must catch the started tunnel instead
    # of racing it for the shared slot.
    class _State:
        def __init__(self):
            self.cloudflare_url = None
            self.armed = False

        @property
        def cloudflare_tunnel_pending(self):
            if self.armed:
                # The startup tunnel settles by publishing its URL.
                self.cloudflare_url = "https://studio.trycloudflare.com"
            return False

    state = _State()
    app = _types.SimpleNamespace(state = state)

    class _RacingListener(_FakeListener):
        async def start(self, app):
            state.armed = True  # the startup tunnel began while we bound
            return await super().start(app)

    fake = _RacingListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(
        psl, "start_preview_tunnel", lambda port: pytest.fail("must not race the startup tunnel")
    )

    assert asyncio.run(link.ensure(app)) == "https://studio.trycloudflare.com"
    assert fake.stopped == 1


def test_ensure_gives_up_when_the_studio_tunnel_never_settles(monkeypatch, link):
    monkeypatch.setattr(psl, "_STUDIO_TUNNEL_WAIT_SECONDS", 0.0)
    app = _FakeApp()
    app.state.cloudflare_tunnel_pending = True
    with pytest.raises(psl.PreviewLinkUnavailable):
        asyncio.run(link.ensure(app))


def test_ensure_rechecks_the_kill_switch_after_tunnel_setup(monkeypatch, link):
    # An admin can disable sharing while the tunnel is still coming up; the
    # queued disable cannot see this tunnel, so ensure must undo it itself.
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    state = {"on": True}
    monkeypatch.setattr(psl, "get_preview_sharing_enabled", lambda: state["on"])
    stopped = []
    monkeypatch.setattr(psl, "stop_tunnel_if_url", lambda url: (stopped.append(url), True)[1])

    def _start(port):
        state["on"] = False
        return "https://preview.trycloudflare.com"

    monkeypatch.setattr(psl, "start_preview_tunnel", _start)

    with pytest.raises(psl.PreviewSharingDisabled):
        asyncio.run(link.ensure(_FakeApp()))

    assert stopped == ["https://preview.trycloudflare.com"]
    assert fake.stopped == 1


def test_stop_tears_down_tunnel_and_listener(monkeypatch, link):
    fake = _FakeListener()
    monkeypatch.setattr(psl, "listener", fake)
    monkeypatch.setattr(psl, "start_preview_tunnel", lambda port: "https://x.trycloudflare.com")
    stopped = []
    monkeypatch.setattr(psl, "stop_tunnel_if_url", lambda url: (stopped.append(url), True)[1])

    app = _FakeApp()
    asyncio.run(link.ensure(app))
    asyncio.run(link.stop())

    # The targeted stop gets the exact URL this link opened.
    assert stopped == ["https://x.trycloudflare.com"]
    assert fake.stopped == 1
    assert link.current(app) is None


def test_stop_leaves_a_replacement_tunnel_running(monkeypatch, link):
    # If the shared slot was replaced after the preview tunnel opened, the
    # conditional stop must no-op instead of killing the replacement.
    import cloudflare_tunnel as ct

    class _FakeTunnel:
        def __init__(self, url):
            self.url = url
            self.stopped = False

        def stop(self):
            self.stopped = True

    replacement = _FakeTunnel("https://studio.trycloudflare.com")
    monkeypatch.setattr(ct, "_active_tunnel", replacement)

    assert ct.stop_tunnel_if_url("https://preview.trycloudflare.com") is False
    assert replacement.stopped is False
    assert ct._active_tunnel is replacement

    assert ct.stop_tunnel_if_url("https://studio.trycloudflare.com") is True
    assert replacement.stopped is True
    assert ct._active_tunnel is None


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
    monkeypatch.setattr(
        psl, "stop_tunnel_if_url", lambda url: pytest.fail("must not stop a tunnel it does not own")
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
