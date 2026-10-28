# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The router skips entries whose literal path prefix rules a request out, and answers exactly as before."""

from __future__ import annotations

import asyncio
import copy
import itertools
import re

import pytest
from fastapi import APIRouter, FastAPI
from starlette.responses import PlainTextResponse
from starlette.routing import BaseRoute, Host, Match, Mount, Route


def install_route_prefix_guards(routes):
    # Imported per call so the main-app tests run (and fail) on a tree without the module.
    from utils.route_prefix_guard import install_route_prefix_guards as install
    return install(routes)


def _leaf_paths(routes, prefix = ""):
    for route in routes:
        original = getattr(route, "original_router", None)
        if original is not None:
            yield from _leaf_paths(original.routes, prefix + route.include_context.prefix)
        else:
            path = getattr(route, "path", None)
            if isinstance(path, str):
                yield prefix + path


def _path_corpus(routes) -> list:
    """Each route path filled in, its slash flip, one level deeper, and every truncation at a separator."""
    paths = {
        "/",
        "",
        "/a",
        "/api",
        "/api/",
        "/v1",
        "/v1/",
        "/assets/x.js",
        "/mcp",
        "/mcp/",
        "/mcp/x",
        "//api/health",
    }
    for path in _leaf_paths(routes):
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        paths.update(
            {concrete, concrete + "/", concrete.rstrip("/"), concrete + "/x", concrete[:-1]}
        )
        for i, ch in enumerate(concrete):
            if ch == "/":
                paths.add(concrete[:i])
    return sorted(paths)


def _scope(
    method: str,
    path: str,
    *,
    kind: str = "http",
    root_path: str = "",
) -> dict:
    return {
        "type": kind,
        "method": method,
        "path": path,
        "raw_path": path.encode("latin-1", "replace"),
        "root_path": root_path,
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1")],
    }


def _scopes(routes, *, variants: bool = True):
    # GET finds FULL matches, PATCH (served by almost nothing) the PARTIAL ones behind a 405.
    for path in _path_corpus(routes):
        yield _scope("GET", path)
        yield _scope("PATCH", path)
        if variants:
            yield _scope("GET", path, kind = "websocket")
            yield _scope("GET", "/root" + path, root_path = "/root")


def _original(entry):
    guarded = vars(entry).get("matches")
    return guarded.__wrapped__ if guarded is not None else entry.matches


def _decision(routes, scope, *, original: bool):
    """The router loop's choice: first FULL, else first PARTIAL, with the child scope it would apply."""
    partial = None
    for index, entry in enumerate(routes):
        call = _original(entry) if original else entry.matches
        match, child = call(copy.copy(scope))
        if match == Match.FULL:
            return "FULL", index, child
        if match == Match.PARTIAL and partial is None:
            partial = ("PARTIAL", index, child)
    return partial or ("NONE", -1, {})


@pytest.fixture(scope = "module")
def studio_app():
    from main import app
    return app


def test_the_studio_app_guards_most_of_its_route_table(studio_app):
    routes = studio_app.router.routes
    guarded = [entry for entry in routes if "matches" in vars(entry)]
    assert len(guarded) >= len(routes) * 0.8, (len(guarded), len(routes))


def test_a_late_route_no_longer_runs_every_earlier_regex(studio_app, monkeypatch):
    """/api/liveness is polled by the desktop app and sits behind ~800 routes. Before the guard every request
    ran ~1600 route matches (an included router matches each candidate twice); with it, only entries whose
    prefix fits the path are tried."""
    import fastapi.routing as fastapi_routing
    import starlette.routing as starlette_routing

    calls = [0]
    for cls in (
        fastapi_routing.APIRoute,
        starlette_routing.Route,
        starlette_routing.Mount,
        getattr(fastapi_routing, "_EffectiveRouteContext", None),
    ):
        if cls is None or "matches" not in cls.__dict__:
            continue
        stock = cls.__dict__["matches"]

        def counted(
            self,
            scope,
            _stock = stock,
        ):
            calls[0] += 1
            return _stock(self, scope)

        monkeypatch.setattr(cls, "matches", counted)

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = _scope("GET", "/api/liveness")
    scope.update(client = ("127.0.0.1", 50000), server = ("127.0.0.1", 8888), scheme = "http")
    asyncio.run(studio_app(scope, receive, send))
    assert sent and sent[0]["status"] == 200
    assert calls[0] < 400, calls[0]


def test_every_guarded_entry_answers_like_the_original(studio_app):
    routes = studio_app.router.routes
    guarded = [entry for entry in routes if "matches" in vars(entry)]
    assert guarded
    skipped = 0
    for entry in guarded:
        for scope in _scopes(routes):
            fast = entry.matches(copy.copy(scope))
            if fast == (Match.NONE, {}):
                slow = _original(entry)(copy.copy(scope))
                assert slow[0] == Match.NONE, (
                    entry,
                    scope["method"],
                    scope["path"],
                    scope["root_path"],
                )
                skipped += 1
    assert skipped > 10_000


def test_the_router_picks_the_same_route_for_every_request(studio_app):
    routes = studio_app.router.routes
    for scope in _scopes(routes, variants = False):
        assert _decision(routes, scope, original = False) == _decision(
            routes, scope, original = True
        ), (scope["method"], scope["path"], scope["root_path"])


def _endpoint(request):
    return PlainTextResponse("ok")


def test_unprovable_entries_are_left_alone():
    class HeaderRoute(Route):
        def matches(self, scope):
            return Match.FULL, {"endpoint": self.endpoint}

    shared = Route("/shared/{x}", _endpoint)
    inner = APIRouter()
    inner.routes.append(shared)
    app = FastAPI()
    app.router.routes.append(Host("example.com", app = PlainTextResponse("host")))
    app.router.routes.append(HeaderRoute("/custom", _endpoint))
    app.router.routes.append(Route("/{anything:path}", _endpoint))
    app.router.routes.append(shared)
    app.include_router(inner, prefix = "/inner")
    with_host = APIRouter()
    with_host.routes.append(Host("example.com", app = PlainTextResponse("host")))
    app.include_router(with_host, prefix = "/hosted")

    install_route_prefix_guards(app.router.routes)

    by_kind = {type(entry).__name__: entry for entry in app.router.routes}
    assert "matches" not in vars(by_kind["Host"])
    assert "matches" not in vars(by_kind["HeaderRoute"])
    catch_all = next(e for e in app.router.routes if getattr(e, "path", None) == "/{anything:path}")
    assert "matches" not in vars(catch_all)
    included = [e for e in app.router.routes if getattr(e, "original_router", None) is not None]
    if included:
        assert "matches" not in vars(shared)
        hosted = [e for e in included if e.original_router is with_host]
        assert hosted and "matches" not in vars(hosted[0])


def test_root_path_requests_take_the_original_path():
    app = FastAPI()
    app.add_api_route("/api/thing", lambda: {"ok": True})
    install_route_prefix_guards(app.router.routes)
    entry = next(e for e in app.router.routes if getattr(e, "path", None) == "/api/thing")
    assert "matches" in vars(entry)
    match, _ = entry.matches(_scope("GET", "/prefix/api/thing", root_path = "/prefix"))
    assert match == Match.FULL


def test_install_is_idempotent_and_keeps_order():
    app = FastAPI()
    app.add_api_route("/a/b", lambda: 1)
    app.mount("/static", PlainTextResponse("s"))
    router = APIRouter()
    router.add_api_route("/c", lambda: 2)
    app.include_router(router, prefix = "/r")
    before = list(app.router.routes)
    first = install_route_prefix_guards(app.router.routes)
    wrapped = {id(e): vars(e).get("matches") for e in app.router.routes}
    second = install_route_prefix_guards(app.router.routes)
    assert first == second >= 3
    assert list(app.router.routes) == before
    assert {id(e): vars(e).get("matches") for e in app.router.routes} == wrapped


def test_mount_guard_matches_the_mount_regex():
    mount = Mount("/mcp", app = PlainTextResponse("m"))
    install_route_prefix_guards([mount])
    assert "matches" in vars(mount)
    for path, method in itertools.product(
        ("/mcp", "/mcp/", "/mcp/x", "/mcpx", "/mc", "/", "/other/mcp/"), ("GET", "POST")
    ):
        scope = _scope(method, path)
        assert mount.matches(copy.copy(scope)) == _original(mount)(copy.copy(scope)), path


def test_guard_is_a_plain_route_attribute():
    """Openapi, url_path_for and isinstance checks see the same route objects."""
    app = FastAPI()
    app.add_api_route("/x/{y}", lambda y: y, name = "xy")
    install_route_prefix_guards(app.router.routes)
    assert all(isinstance(e, BaseRoute) for e in app.router.routes)
    assert app.url_path_for("xy", y = "1") == "/x/1"
    assert "/x/{y}" in app.openapi()["paths"]
