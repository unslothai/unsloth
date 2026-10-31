# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /mcp and /mcp/decisions apps are built on first use, not at import: same routes, same
answers, one build under concurrent first requests, and the MCP lifespan still runs."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from auth import storage
from utils.lazy_asgi_app import LazyLifespanApp

from . import mcp_harness
from .mcp_harness import MCP_HEADERS, seed_owner

isolated_auth = mcp_harness.isolated_auth

LISTING = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
BACKEND = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.usefixtures("isolated_auth")


def test_import_main_builds_no_mcp_app():
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("UNSLOTH_STUDIO_ENABLE_MCP", "UNSLOTH_STUDIO_MCP_TOKEN")
    }
    probe = (
        "import json, sys, main\n"
        "from starlette.routing import Mount\n"
        "mods = sorted(m for m in sys.modules if m.split('.')[0] in ('fastmcp', 'mcp_server'))\n"
        "mounts = [r.path for r in main.app.routes if isinstance(r, Mount)]\n"
        "print('PROBE ' + json.dumps({'mods': mods, 'mounts': mounts}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd = BACKEND,
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [l for l in result.stdout.splitlines() if l.startswith("PROBE ")][-1]
    seen = json.loads(line[len("PROBE ") :])
    assert seen["mods"] == []
    assert "/mcp/decisions" in seen["mounts"] and "/mcp" in seen["mounts"]
    assert seen["mounts"].index("/mcp/decisions") < seen["mounts"].index("/mcp")


class Counting:
    """A built app whose builds, lifespan entries and exits are counted."""

    def __init__(self, fail = False):
        self.builds = 0
        self.entered = 0
        self.exited = 0
        self.fail = fail
        self._lock = threading.Lock()

    def build(self):
        with self._lock:
            self.builds += 1
        if self.fail:
            raise ValueError("cannot build")
        started = {}

        async def endpoint(request):
            return PlainTextResponse("up" if started.get("on") else "not started")

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(app):
            self.entered += 1
            started["on"] = True
            try:
                yield
            finally:
                started["on"] = False
                self.exited += 1

        return Starlette(routes = [Route("/", endpoint, methods = ["GET", "POST"])], lifespan = lifespan)


def parent(lazy):
    return Starlette(routes = [Mount("/m", lazy)], lifespan = lazy.lifespan)


def test_nothing_is_built_until_the_first_request_then_once_for_concurrent_ones():
    counting = Counting()
    lazy = LazyLifespanApp(counting.build)
    app = parent(lazy)
    with TestClient(app) as client:
        assert counting.builds == 0 and counting.entered == 0

        async def burst():
            transport = httpx.ASGITransport(app = app)
            async with httpx.AsyncClient(transport = transport, base_url = "http://t") as http:
                results = []

                async def one():
                    results.append((await http.get("/m/")).text)

                async with anyio.create_task_group() as tg:
                    for _ in range(16):
                        tg.start_soon(one)
                return results

        assert client.portal.call(burst) == ["up"] * 16
        assert counting.builds == 1 and counting.entered == 1 and counting.exited == 0
    assert counting.exited == 1


def test_wanted_at_startup_starts_before_the_first_request_and_restarts_each_cycle():
    counting = Counting()
    lazy = LazyLifespanApp(counting.build, wanted_at_startup = lambda: True)
    for cycle in (1, 2):
        with TestClient(parent(lazy)) as client:
            assert (counting.builds, counting.entered) == (1, cycle)
            assert client.get("/m/").text == "up"
        assert counting.exited == cycle


def test_a_built_app_is_started_with_later_lifespans():
    counting = Counting()
    lazy = LazyLifespanApp(counting.build)
    with TestClient(parent(lazy)) as client:
        assert client.get("/m/").text == "up"
    with TestClient(parent(lazy)) as client:
        assert counting.entered == 2
    assert (counting.builds, counting.exited) == (1, 2)


def test_without_a_parent_lifespan_the_app_is_served_unstarted():
    counting = Counting()
    client = TestClient(parent(LazyLifespanApp(counting.build)))
    assert client.get("/m/").text == "not started"
    assert counting.entered == 0


def test_build_failures_fail_the_request_or_the_startup_that_wanted_it():
    counting = Counting(fail = True)
    with TestClient(
        parent(LazyLifespanApp(counting.build)), raise_server_exceptions = False
    ) as client:
        assert client.get("/m/").status_code == 500
        assert client.get("/m/").status_code == 500
    assert counting.builds == 1
    with pytest.raises(ValueError, match = "cannot build"):
        with TestClient(parent(LazyLifespanApp(counting.build, wanted_at_startup = lambda: True))):
            pass


def test_threads_share_one_build():
    counting = Counting()
    lazy = LazyLifespanApp(counting.build)
    barrier = threading.Barrier(8)
    built = []

    def go():
        barrier.wait()
        built.append(lazy.build())

    threads = [threading.Thread(target = go) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert counting.builds == 1 and len({id(app) for app in built}) == 1


def _studio_mcp(on_at_startup):
    import main
    from studio_mcp.gate import StudioMcpGate
    from utils.mcp_access import set_mcp_enabled

    builds = []

    def build():
        builds.append(1)
        return main._build_studio_mcp_app()

    seed_owner()
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    set_mcp_enabled(on_at_startup)
    from utils.mcp_access import is_mcp_enabled

    lazy = LazyLifespanApp(build, wanted_at_startup = is_mcp_enabled)
    app = Starlette(routes = [Mount("/mcp", StudioMcpGate(lazy))], lifespan = lazy.lifespan)
    app.state.server_port = 8888
    app.state.cloudflare_url = None
    return app, raw_key, builds, set_mcp_enabled


@pytest.mark.parametrize("on_at_startup", [False, True])
def test_studio_mcp_answers_whether_switched_on_at_startup_or_later(on_at_startup):
    app, raw_key, builds, set_mcp_enabled = _studio_mcp(on_at_startup)
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {raw_key}"}
    with TestClient(app, base_url = "http://127.0.0.1:8888") as http:
        assert len(builds) == int(on_at_startup)
        if not on_at_startup:
            assert http.post("/mcp/", json = LISTING, headers = headers).status_code == 404
            assert builds == []
            set_mcp_enabled(True)
        assert http.post("/mcp/", json = LISTING, headers = MCP_HEADERS).status_code == 401
        response = http.post("/mcp/", json = LISTING, headers = headers)
        assert response.status_code == 200, response.text
        assert "studio_status" in response.text
        assert len(builds) == 1


def test_decisions_mcp_is_built_by_its_first_signed_in_request(monkeypatch):
    import main
    from routes import systemone

    async def signed_in(credentials):
        return "tester"

    lazy = LazyLifespanApp(main._build_decisions_mcp_app)
    app = Starlette(
        routes = [Mount(systemone.MCP_PATH, systemone.RequireStudioAuth(lazy))],
        lifespan = lazy.lifespan,
    )
    with TestClient(app) as http:
        assert (
            http.post(f"{systemone.MCP_PATH}/", json = LISTING, headers = MCP_HEADERS).status_code
            == 401
        )
        assert not lazy.built
        monkeypatch.setattr(systemone, "get_current_subject", signed_in)
        response = http.post(
            f"{systemone.MCP_PATH}/",
            json = LISTING,
            headers = {**MCP_HEADERS, "Authorization": "Bearer t"},
        )
        assert response.status_code == 200, response.text
        assert lazy.built


def test_main_mounts_the_lazy_apps():
    import main
    from starlette.routing import Mount as StarletteMount

    mounts = {r.path: r.app for r in main.app.routes if isinstance(r, StarletteMount)}
    assert mounts["/mcp"].app is main._studio_mcp_app
    assert mounts["/mcp/decisions"].app is main._decisions_mcp_app
    assert isinstance(main._studio_mcp_app, LazyLifespanApp)
