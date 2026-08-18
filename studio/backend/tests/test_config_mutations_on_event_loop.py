# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MCP and provider config mutations must keep running on the event loop thread.

Each of these handlers reads (a row, or the whole server list) and then writes based on what
it read, with nothing below them serializing the pair: `mcp_servers.url` has no uniqueness
constraint, and `credential_secrets` has no foreign key to `providers`. An await-free
``async def`` runs to completion without yielding, so the loop is what makes that
read-then-write atomic. As a plain ``def`` FastAPI dispatches it to its threadpool, where two
imports both build `seen_urls` before either inserts, and an update interleaves with a
concurrent delete and writes a credential for a provider that is already gone.

The read handlers beside them do belong in the threadpool, which is the point of the change
this guards, so both directions are pinned here.

Asserts which thread each handler ran on rather than racing two requests.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.mcp_servers as mcp_routes
import routes.providers as provider_routes
from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)


class _Case(NamedTuple):
    method: str
    path: str
    body: dict | None
    store: str  # the storage module the route module imported
    read: str  # the first store call the handler makes
    result: object  # what that call returns
    status: int  # the status the request ends with once the read is stubbed


# The stubbed call is the first store touch in each handler, so a 404 right after it is all
# this needs: the thread the handler runs on is already decided by then.
_MUTATIONS = {
    "mcp-import": _Case(
        "post",
        "/api/mcp-servers/import",
        {"config": {"mcpServers": {}}},
        "mcp_servers_db",
        "list_servers",
        [],
        200,
    ),
    "provider-update": _Case(
        "put",
        "/api/providers/p1",
        {"display_name": "renamed"},
        "providers_db",
        "get_provider",
        None,
        404,
    ),
    "provider-migrate": _Case(
        "put",
        "/api/providers/p1/api-key/migrate",
        {"encrypted_api_key": "k"},
        "providers_db",
        "get_provider",
        None,
        404,
    ),
}

_READS = {
    "mcp-list": _Case("get", "/api/mcp-servers/", None, "mcp_servers_db", "list_servers", [], 200),
    "provider-list": _Case(
        "get", "/api/providers/", None, "providers_db", "list_providers", [], 200
    ),
}


def _record(threads: list[int], result):
    def _call(*_args, **_kwargs):
        threads.append(threading.get_ident())
        return result

    return _call


def _drive(monkeypatch, case: _Case):
    """Run one route through real FastAPI dispatch; return (handler threads, loop thread)."""
    threads: list[int] = []
    route_module = mcp_routes if case.store == "mcp_servers_db" else provider_routes
    monkeypatch.setattr(getattr(route_module, case.store), case.read, _record(threads, case.result))

    app = FastAPI()
    app.include_router(mcp_routes.router, prefix = "/api/mcp-servers")
    app.include_router(provider_routes.router, prefix = "/api/providers")
    app.dependency_overrides[get_current_subject] = lambda: "u"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[get_current_credential] = lambda: ("u", None)

    loop_threads: list[int] = []

    @app.get("/loop-thread")
    async def _loop_thread():  # an async route runs on the loop, so this ident is the loop's
        loop_threads.append(threading.get_ident())
        return {}

    with TestClient(app) as client:
        assert client.get("/loop-thread").status_code == 200
        body = {} if case.body is None else {"json": case.body}
        assert client.request(case.method, case.path, **body).status_code == case.status

    assert threads, f"{case.path} never touched the store"
    assert loop_threads, "the reference route never ran"
    return threads, loop_threads[0]


@pytest.mark.parametrize("case", _MUTATIONS.values(), ids = list(_MUTATIONS))
def test_a_config_mutation_runs_on_the_event_loop_thread(monkeypatch, case):
    threads, loop_thread = _drive(monkeypatch, case)
    assert (
        threads[0] == loop_thread
    ), f"{case.path} ran in the threadpool, so its read-then-write is no longer serialized"


@pytest.mark.parametrize("case", _READS.values(), ids = list(_READS))
def test_a_config_read_stays_off_the_event_loop_thread(monkeypatch, case):
    threads, loop_thread = _drive(monkeypatch, case)
    assert threads[0] != loop_thread, f"{case.path} read the store on the event loop thread"
