# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Building the OpenAPI schema must not stall the event loop (#9712). The measured build is a
sleep of _BUILD_S, so the result does not depend on machine speed."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import fastapi.applications
import httpx
import pytest
from fastapi.testclient import TestClient

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_BUILD_S = 3.0


@pytest.fixture(scope = "module")
def main_module():
    import main as _main
    return _main


@pytest.fixture(scope = "module")
def real_schema(main_module):
    """The real schema, built before any loop is measured. Generating it holds the GIL in
    stretches that grow with machine speed (1.6s on a loaded runner, against a 1.5s limit),
    so it must not run inside the window the probe watches."""
    return main_module.app.openapi()


@pytest.fixture
def slow_cold_build(main_module, real_schema, monkeypatch):
    """A cold schema cache whose build sleeps _BUILD_S, returns the real schema, and counts its calls."""
    calls = []

    def slow(*args, **kwargs):
        calls.append(threading.current_thread().name)
        time.sleep(_BUILD_S)
        return real_schema

    monkeypatch.setattr(fastapi.applications, "get_openapi", slow)
    saved = main_module.app.openapi_schema
    main_module.app.openapi_schema = None
    yield calls
    main_module.app.openapi_schema = saved


async def _loop_delay_during_cold_fetches(app, fetches: int):
    transport = httpx.ASGITransport(app = app)
    async with httpx.AsyncClient(transport = transport, base_url = "http://127.0.0.1") as client:
        worst = 0.0
        done = asyncio.Event()

        async def probe():
            nonlocal worst
            while not done.is_set():
                started = time.perf_counter()
                await asyncio.sleep(0.02)
                worst = max(worst, time.perf_counter() - started - 0.02)

        prober = asyncio.create_task(probe())
        await asyncio.sleep(0.1)
        responses = await asyncio.gather(*(client.get("/openapi.json") for _ in range(fetches)))
        done.set()
        await prober
        return worst, responses


def test_cold_schema_build_leaves_the_loop_answering(main_module, slow_cold_build):
    worst, (response,) = asyncio.run(_loop_delay_during_cold_fetches(main_module.app, 1))
    assert response.status_code == 200
    assert worst < _BUILD_S / 2, f"event loop blocked {worst:.2f}s while the schema was built"
    assert slow_cold_build and all(name != "MainThread" for name in slow_cold_build)


def test_concurrent_cold_fetches_build_once_and_agree(main_module, slow_cold_build):
    worst, responses = asyncio.run(_loop_delay_during_cold_fetches(main_module.app, 3))
    assert worst < _BUILD_S / 2
    assert len(slow_cold_build) == 1
    assert len({r.content for r in responses}) == 1
    # What the chat sidebar reads to decide whether title repair is safe.
    patch = responses[0].json()["components"]["schemas"]["ChatThreadPatch"]["properties"]
    assert {"expectedTitle", "expectedOpeningMessageId"} <= set(patch)


def test_schema_route_keeps_fastapi_behaviour(main_module):
    plain = TestClient(main_module.app).get("/openapi.json")
    assert plain.status_code == 200
    assert plain.json() == main_module.app.openapi()
    assert "/openapi.json" not in plain.json()["paths"]

    # As FastAPI's own route: a path-stripping proxy's prefix is named in servers.
    prefixed = TestClient(main_module.app, root_path = "/studio").get("/openapi.json").json()
    assert prefixed["servers"][0] == {"url": "/studio"}
    assert "servers" not in main_module.app.openapi()
