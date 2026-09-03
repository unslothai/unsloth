# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""State-changing auth, MCP and provider handlers must keep running on the event loop thread.

Each of these reads (a row, the server list, a rate-limit bucket) and then writes based on what
it read, with nothing below them serializing the pair:

  * import_mcp_servers builds `seen_urls` and then inserts, and `mcp_servers.url` has no
    uniqueness constraint.
  * update_provider_config / migrate_provider_api_key read the row and then save a credential,
    and `credential_secrets` has no foreign key to `providers`.
  * login clears its admission check before verify_password reaches _record_login_failure, so
    the bucket lock guards each call but not the sequence.
  * refresh consumes a token and inserts its replacement, and logout deletes every token in
    between without rotating the credential generation.

They are await-free, so the loop is what makes those sequences atomic. In the threadpool, two
imports duplicate a server row, an update racing a delete writes a credential for a provider that
is gone, a burst of guesses passes admission together, and a logout leaves the refresh token that
landed after it. The read-only handlers beside them do belong in the threadpool, so both
directions are pinned here.

Asserts which thread each handler ran on rather than racing two requests.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.auth as auth_routes
import routes.mcp_servers as mcp_routes
import routes.providers as provider_routes
from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
    get_current_subject_allow_password_change,
)


class _Case(NamedTuple):
    method: str
    path: str
    body: dict | None
    module: object
    store: str
    read: str
    result: object
    status: int


# The first store call is enough to identify the handler's dispatch thread.
_MUTATIONS = {
    "mcp-import": _Case(
        "post",
        "/api/mcp-servers/import",
        {"config": {"mcpServers": {}}},
        mcp_routes,
        "mcp_servers_db",
        "list_servers",
        [],
        200,
    ),
    "provider-update": _Case(
        "put",
        "/api/providers/p1",
        {"display_name": "renamed"},
        provider_routes,
        "providers_db",
        "get_provider",
        None,
        404,
    ),
    "provider-migrate": _Case(
        "put",
        "/api/providers/p1/api-key/migrate",
        {"encrypted_api_key": "k"},
        provider_routes,
        "providers_db",
        "get_provider",
        None,
        404,
    ),
    "auth-login": _Case(
        "post",
        "/api/auth/login",
        {"username": "u", "password": "p"},
        auth_routes,
        "storage",
        "get_user_and_secret",
        None,
        401,
    ),
    "auth-refresh": _Case(
        "post",
        "/api/auth/refresh",
        {"refresh_token": "t"},
        auth_routes,
        "storage",
        "consume_refresh_token",
        None,
        401,
    ),
    "auth-logout": _Case(
        "post",
        "/api/auth/logout",
        None,
        auth_routes,
        "storage",
        "revoke_user_refresh_tokens",
        None,
        204,
    ),
}

_READS = {
    "mcp-list": _Case(
        "get", "/api/mcp-servers/", None, mcp_routes, "mcp_servers_db", "list_servers", [], 200
    ),
    "provider-list": _Case(
        "get", "/api/providers/", None, provider_routes, "providers_db", "list_providers", [], 200
    ),
    "auth-api-keys": _Case(
        "get", "/api/auth/api-keys", None, auth_routes, "storage", "list_api_keys", [], 200
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
    monkeypatch.setattr(getattr(case.module, case.store), case.read, _record(threads, case.result))
    # A failed login from an earlier test in this session would otherwise 429 this one.
    monkeypatch.setattr(auth_routes, "_LOGIN_BUCKETS", {})
    monkeypatch.setattr(auth_routes, "_LOGIN_IP_BUCKETS", {})

    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    app.include_router(mcp_routes.router, prefix = "/api/mcp-servers")
    app.include_router(provider_routes.router, prefix = "/api/providers")
    app.dependency_overrides[get_current_subject] = lambda: "u"
    app.dependency_overrides[get_current_subject_allow_password_change] = lambda: "u"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[auth_routes.authenticated_without_credential] = lambda: False
    app.dependency_overrides[mcp_routes.request_admitted_without_credential] = lambda: False
    app.dependency_overrides[get_current_credential] = lambda: ("u", None)

    loop_threads: list[int] = []

    @app.get("/loop-thread")
    async def _loop_thread():  # Reference event-loop thread.
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
def test_a_state_changing_handler_runs_on_the_event_loop_thread(monkeypatch, case):
    threads, loop_thread = _drive(monkeypatch, case)
    assert (
        threads[0] == loop_thread
    ), f"{case.path} ran in the threadpool, so its check-then-write is no longer serialized"


@pytest.mark.parametrize("case", _READS.values(), ids = list(_READS))
def test_a_read_only_handler_stays_off_the_event_loop_thread(monkeypatch, case):
    threads, loop_thread = _drive(monkeypatch, case)
    assert threads[0] != loop_thread, f"{case.path} read its store on the event loop thread"
