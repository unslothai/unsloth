# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""routes/llama_compat.py: the discovery surface a third-party client probes.

Reproduces a real user report. A client pointed at Studio's port ran this exact
sequence and got these exact answers:

    GET  /api/v1/models  404      GET  /props    200 text/html
    GET  /api/tags       404      GET  /version  200 text/html
    GET  /v1/props       404      POST /api/show 405

The two 200s are the bug. /props and /version are not Studio routes, so they fell
through the SPA catch-all in main.py, which serves index.html for anything outside
/api/ and /v1/. A probe that checks the status before the body reads that as
"supported" and then fails on HTML it cannot parse -- strictly worse than the 404
the other four returned.

The module is loaded standalone and handed a double for routes.inference: the real
one pulls the whole inference stack, and every catalog helper it needs is injected
here anyway, so the wire shapes can be asserted without a loaded model.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


CATALOG = [
    {
        "id": "unsloth/Qwen3.8-27B-GGUF",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "unsloth",
        "loaded": True,
        "quant": "UD-Q4_K_XL",
    },
    {
        "id": "unsloth/Laguna-S-2.1-GGUF",
        "object": "model",
        "created": 1_700_000_000,
        "owned_by": "unsloth",
        "loaded": False,
    },
]


class _Backend:
    """Enough of LlamaCppBackend for the props handler."""

    def __init__(
        self,
        *,
        loaded = True,
        props = None,
    ):
        self.is_loaded = loaded
        self.model_identifier = "/media/models/qwen/model.gguf"
        self._openai_advertised_id = "unsloth/Qwen3.8-27B-GGUF"
        self.context_length = 262144
        self.chat_template = "{{ messages }}"
        self._props = props

    def _query_server_props(self):
        return self._props


def _inference_double(
    backend,
    catalog,
    orchestrator = None,
):
    """Stand-in for routes.inference, injected through llama_compat._inference().

    Deliberately not a sys.modules entry: an earlier revision stubbed
    routes.inference globally and every module imported while it was live captured
    the stub, which broke two unrelated route tests when the suite ran in one process.
    """
    inference = types.SimpleNamespace()
    inference.get_llama_cpp_backend = lambda: backend
    inference._llama_public_model_id = lambda b, fallback = None: b._openai_advertised_id
    inference._peek_inference_backend = lambda: orchestrator
    inference._orchestrator_public_model_id = lambda b: b.active_model_name

    async def _objects():
        return catalog

    inference._openai_catalog_objects = _objects
    return inference


_MOD = None


def _load(
    backend = None,
    catalog = None,
    orchestrator = None,
):
    """The real routes/llama_compat.py, with routes.inference replaced by a double."""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "llama_compat_under_test", str(_BACKEND / "routes" / "llama_compat.py")
        )
        _MOD = importlib.util.module_from_spec(spec)
        sys.modules["llama_compat_under_test"] = _MOD
        spec.loader.exec_module(_MOD)
    double = _inference_double(
        backend if backend is not None else _Backend(),
        CATALOG if catalog is None else catalog,
        orchestrator,
    )
    _MOD._inference = lambda: double
    return _MOD


def _client(mod):
    app = FastAPI()
    app.include_router(mod.router)

    # The SPA catch-all, reproduced in main.py's order: real routes first, then the
    # fallback. A test that omitted it could not tell "serves JSON" from "serves the
    # app shell with a 200", which is the whole defect.
    @app.get("/{full_path:path}")
    async def _spa(full_path: str):
        if full_path in {"api", "v1"} or full_path.startswith(("api/", "v1/")):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")
        if mod.is_engine_probe_path(full_path):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")
        return Response(content = b"<!doctype html>", media_type = "text/html")

    # Override the exact object the router closed over.
    app.dependency_overrides[mod.get_current_subject] = lambda: "test"
    return TestClient(app)


@pytest.fixture(scope = "module", autouse = True)
def _teardown():
    yield
    sys.modules.pop("llama_compat_under_test", None)


# ── the reported probe sequence ───────────────────────────────────────────────


def test_the_reported_probe_sequence_no_longer_returns_html():
    """THE REGRESSION THIS FILE EXISTS FOR.

    Every path the client probed must answer JSON or 404, never 200 text/html.
    """
    mod = _load()
    with _client(mod) as c:
        # The two that used to answer with the app shell now answer with JSON.
        for path in ("/props", "/v1/props", "/version"):
            r = c.get(path)
            assert r.status_code == 200, (path, r.status_code)
            assert "application/json" in r.headers["content-type"], path

        # The Ollama pair stays a 404 on purpose: see test_the_ollama_surface_is_not
        # _advertised. What matters is that no probe in the sequence gets HTML.
        # 405 for POST /api/show is main.py's standing answer for a POST to any unknown
        # /api/ path, unchanged by this PR; what matters is that none of them is HTML.
        for method, path in (
            ("GET", "/api/v1/models"),
            ("GET", "/api/tags"),
            ("POST", "/api/show"),
        ):
            r = c.request(method, path, json = {} if method == "POST" else None)
            assert r.status_code in (404, 405), (method, path, r.status_code)
            assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_the_html_fallback_is_what_used_to_answer_these():
    """The control: an ordinary unknown path still gets the app shell.

    Without this the test above would pass on a build that 404s everything, which
    would break deep links into the UI.
    """
    mod = _load()
    with _client(mod) as c:
        r = c.get("/some-ui-deep-link")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


def test_engine_endpoints_studio_does_not_serve_are_404_not_html():
    mod = _load()
    with _client(mod) as c:
        for path in ("/slots", "/completion", "/metrics", "/tokenize", "/health"):
            r = c.get(path)
            assert r.status_code == 404, path
            assert "text/html" not in r.headers.get("content-type", ""), path


def test_probe_matching_ignores_case_and_stray_slashes():
    mod = _load()
    assert mod.is_engine_probe_path("slots")
    assert mod.is_engine_probe_path("/slots/")
    assert mod.is_engine_probe_path("Metrics")
    assert not mod.is_engine_probe_path("chat")
    # Served paths are deliberately absent: they are real routes, matched first.
    assert not mod.is_engine_probe_path("props")
    assert not mod.is_engine_probe_path("version")


# ── /props ────────────────────────────────────────────────────────────────────


def test_props_never_echoes_the_on_disk_gguf_path():
    """llama-server reports model_path as the absolute .gguf path. This endpoint is
    reachable over LAN, so it publishes the public id instead, like every other
    Studio response."""
    upstream = {
        "model_path": "/media/llm4/HDD-Models1/models/hub/models--unsloth--Qwen/model.gguf",
        "total_slots": 4,
        "default_generation_settings": {"n_ctx": 65536},
    }
    mod = _load(backend = _Backend(props = upstream))
    with _client(mod) as c:
        raw = c.get("/props")
    body = raw.json()
    assert body["model_path"] == "unsloth/Qwen3.8-27B-GGUF"
    assert "HDD-Models1" not in raw.text
    assert body["total_slots"] == 4
    assert body["default_generation_settings"]["n_ctx"] == 65536


def test_props_degrades_to_the_local_view_when_the_engine_cannot_be_read():
    """_query_server_props returns None on a mid-restart engine. The probe must
    still answer, from what the backend knows locally."""
    mod = _load(backend = _Backend(props = None))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["model_path"] == "unsloth/Qwen3.8-27B-GGUF"
    assert body["default_generation_settings"]["n_ctx"] == 262144
    assert body["total_slots"] == 0
    assert body["build_info"].startswith("unsloth-studio/")


def test_props_with_nothing_loaded_reports_no_model():
    mod = _load(backend = _Backend(loaded = False))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert "model_path" not in body
    # No llama-server, so no llama-server props. Reporting total_slots 0 and an empty
    # template would describe an engine that is not running.
    assert "total_slots" not in body
    assert "chat_template" not in body
    assert body["build_info"].startswith("unsloth-studio/")


def test_v1_props_and_props_agree():
    mod = _load(backend = _Backend(props = {"total_slots": 2}))
    with _client(mod) as c:
        assert c.get("/props").json() == c.get("/v1/props").json()


# ── /version ──────────────────────────────────────────────────────────────────


def test_version_answers_on_the_bare_path_only():
    """Ollama spells it /api/version; answering there is part of claiming to be Ollama."""
    mod = _load()
    with _client(mod) as c:
        bare = c.get("/version")
        prefixed = c.get("/api/version")
    assert bare.status_code == 200
    assert isinstance(bare.json()["version"], str) and bare.json()["version"]
    assert prefixed.status_code == 404


# ── platform and robustness, from the sandbox run ─────────────────────────────


@pytest.mark.parametrize(
    "upstream",
    [
        None,
        [],
        "not a dict",
        42,
        {},
        {"default_generation_settings": None},
        {"default_generation_settings": []},
        {"total_slots": None},
        {"model_path": "/media/llm4/HDD/models/x.gguf"},
    ],
)
def test_props_never_500s_and_never_leaks_a_path_on_a_hostile_upstream(upstream):
    mod = _load(backend = _Backend(props = upstream))
    with _client(mod) as c:
        r = c.get("/props")
    assert r.status_code == 200, r.text
    assert "/media/" not in r.text
    assert r.json()["model_path"] == "unsloth/Qwen3.8-27B-GGUF"


def test_props_survives_an_engine_query_that_raises():
    """_query_server_props is documented to return None rather than raise. A probe is
    the wrong place to find out that slipped, and the local view is always answerable."""

    class _Raises(_Backend):
        def _query_server_props(self):
            raise RuntimeError("engine unreachable")

    mod = _load(backend = _Raises())
    with _client(mod) as c:
        r = c.get("/props")
    assert r.status_code == 200
    assert r.json()["model_path"] == "unsloth/Qwen3.8-27B-GGUF"


def test_the_version_lookup_survives_an_import_failure(monkeypatch):
    """The import sits inside the guard, not above it: an ImportError there would 500
    a /props probe that has nothing to do with versions."""
    mod = _load()
    mod._studio_version.cache_clear()
    monkeypatch.setitem(sys.modules, "utils.studio_version", None)
    try:
        assert mod._studio_version() == "dev"
    finally:
        mod._studio_version.cache_clear()


def test_the_version_lookup_is_resolved_once(monkeypatch):
    """get_studio_version() shells out to git twice on a source checkout and /version
    is an async handler, so an uncached call would put those spawns on the event loop."""
    mod = _load()
    mod._studio_version.cache_clear()
    calls = []
    fake = types.ModuleType("utils.studio_version")
    fake.get_studio_version = lambda repo_root = None: (calls.append(1), "v0.1.999")[1]
    monkeypatch.setitem(sys.modules, "utils.studio_version", fake)
    try:
        for _ in range(25):
            assert mod._studio_version() == "v0.1.999"
        assert len(calls) == 1
    finally:
        mod._studio_version.cache_clear()


def test_a_real_asset_wins_over_the_probe_deny_list(tmp_path):
    """The guard runs after the static lookup, so a build that ever ships a file named
    `metrics` still serves it rather than 404-ing with no obvious cause."""
    mod = _load()
    (tmp_path / "metrics").write_text("asset-body")
    app = FastAPI()
    app.include_router(mod.router)

    @app.get("/{full_path:path}")
    async def _spa(full_path: str):
        candidate = tmp_path / full_path
        if candidate.is_file():
            return Response(content = candidate.read_bytes(), media_type = "text/plain")
        if mod.is_engine_probe_path(full_path):
            raise HTTPException(status_code = 404, detail = "API endpoint not found")
        return Response(content = b"<!doctype html>", media_type = "text/html")

    app.dependency_overrides[mod.get_current_subject] = lambda: "test"
    with TestClient(app) as c:
        assert c.get("/metrics").text == "asset-body"
        assert c.get("/slots").status_code == 404


def test_no_client_side_ui_route_is_shadowed_by_the_deny_list():
    """Reads the routes the frontend actually declares, so adding a UI page named
    /metrics or /health fails here instead of 404-ing in the browser."""
    routes_dir = _BACKEND.parent / "frontend" / "src" / "app" / "routes"
    if not routes_dir.is_dir():
        pytest.skip("frontend sources not present in this checkout")
    mod = _load()
    declared = set()
    for path in routes_dir.glob("*.tsx"):
        for line in path.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if line.startswith('path: "') and line.endswith('",'):
                declared.add(line[len('path: "') : -2])
    assert declared, "no client routes parsed; the parser needs updating"
    shadowed = {p for p in declared if mod.is_engine_probe_path(p)}
    assert not shadowed, f"UI route(s) shadowed by the probe deny-list: {shadowed}"


# ── the four findings from review ─────────────────────────────────────────────


def test_the_ollama_surface_is_not_advertised():
    """Answering /api/tags identifies this server as Ollama, and a client that
    completes Ollama discovery then posts to /api/chat or /api/generate, which Studio
    does not serve. The reporting user's client got 404 here, fell back to the OpenAI
    surface, and worked; advertising a protocol we do not implement would have broken
    exactly that client. Same defect as the HTML 200, one layer up."""
    mod = _load()
    routes = {r.path for r in mod.router.routes}
    assert "/api/tags" not in routes
    assert "/api/show" not in routes
    assert "/api/version" not in routes
    # And the inference endpoints an Ollama client would go on to call are absent, which
    # is what makes advertising the discovery pair wrong rather than merely incomplete.
    assert "/api/chat" not in routes and "/api/generate" not in routes


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize(
    "path",
    [
        "/completion",
        "/tokenize",
        "/detokenize",
        "/rerank",
        "/infill",
        "/embedding",
        "/apply-template",
        "/slots",
    ],
)
def test_unserved_engine_endpoints_404_on_their_real_method_not_405(method, path):
    """These are POST endpoints in llama-server. The deny-list guard lives in the SPA's
    GET catch-all, so before this a POST matched the GET-only route and returned 405,
    which a discovery client reads as "exists, wrong method" -- the same false positive
    the 404 exists to close."""
    mod = _load()
    with _client(mod) as c:
        r = c.request(method, path, json = {})
    assert r.status_code == 404, (method, path, r.status_code)
    assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_get_on_a_probe_path_still_reaches_the_asset_lookup(tmp_path):
    """The non-GET routes deliberately omit GET, or they would shadow the catch-all's
    static lookup and re-break the asset case."""
    mod = _load()
    assert not any(
        "GET" in (getattr(r, "methods", None) or set()) and r.path == "/completion"
        for r in mod.router.routes
    )


def test_props_does_not_pair_an_orchestrator_model_with_llama_server_fields():
    """With a Transformers or MLX model resident the llama backend is unloaded. Reading
    its fields anyway advertised the orchestrator's model alongside n_ctx 0, an empty
    template and total_slots 0: a self-contradictory description of a model that is in
    fact serving."""

    class _Orchestrator:
        active_model_name = "unsloth/Llama-3.2-3B"

    mod = _load(backend = _Backend(loaded = False), orchestrator = _Orchestrator())
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["model_path"] == "unsloth/Llama-3.2-3B"
    for llama_only in ("total_slots", "chat_template", "default_generation_settings"):
        assert llama_only not in body, llama_only


def test_the_probe_paths_are_admitted_under_keyless_inference_scope():
    """keyless_api_access_scope="inference" lets a keyless client list models and chat.
    A 401 on /props in front of that surface reads as an auth wall over the whole thing
    and stops discovery, which is the opposite of what the scope grants."""
    from utils.keyless_api_access import _INFERENCE_ROUTES

    assert ("GET", "/v1/models") in _INFERENCE_ROUTES, "baseline moved; re-derive this"
    for path in ("/props", "/v1/props", "/version"):
        assert ("GET", path) in _INFERENCE_ROUTES, path
    # Not the Ollama paths: they are not served at all.
    for path in ("/api/tags", "/api/show", "/api/version"):
        assert ("GET", path) not in _INFERENCE_ROUTES, path


# ── round two ─────────────────────────────────────────────────────────────────


def test_head_on_an_unserved_probe_path_is_404_not_405():
    """Starlette does NOT admit HEAD on a GET route. Measured on the pinned
    fastapi 0.141.1 / starlette 1.6.0: HEAD against a bare GET catch-all returns 405,
    so a HEAD probe would read the endpoint as existing. A comment here previously
    claimed the opposite."""
    mod = _load()
    with _client(mod) as c:
        for path in ("/completion", "/tokenize", "/metrics", "/slots"):
            assert c.request("HEAD", path).status_code == 404, path


def test_head_is_405_on_a_plain_get_route_which_is_why_it_is_listed():
    """Pins the framework behaviour the fix depends on, so a future FastAPI that starts
    admitting HEAD does not leave the reason for this list unexplained."""
    app = FastAPI()

    @app.get("/{p:path}")
    async def _catchall(p: str):
        return {"p": p}

    with TestClient(app) as c:
        assert c.get("/anything").status_code == 200
        assert c.request("HEAD", "/anything").status_code == 405


@pytest.mark.parametrize("path", ["/slots/0", "/slots/3", "/slots/abc", "/slots/0/"])
def test_the_nested_slot_endpoint_is_denied_too(path):
    """/slots/:id_slot is a real llama-server route, and Studio calls it itself in
    restore_slots_for_resume, so an exact-membership check on "slots" missed it."""
    mod = _load()
    assert mod.is_engine_probe_path(path.strip("/"))
    with _client(mod) as c:
        for method in ("GET", "HEAD", "POST", "DELETE"):
            r = c.request(method, path, json = {} if method == "POST" else None)
            assert r.status_code == 404, (method, path, r.status_code)


def test_the_deny_list_matches_llama_servers_own_route_table():
    """Transcribed from tools/server/server.cpp rather than grown one client complaint
    at a time. /props is the single bare route Studio serves, so it is the only
    omission; anything else missing here means a probe still reaches the app shell."""
    mod = _load()
    bare_llama_routes = {
        "apply-template",
        "audio/transcriptions",
        "chat/completions",
        "chat/completions/input_tokens",
        "completion",
        "completions",
        "cors-proxy",
        "detokenize",
        "embedding",
        "embeddings",
        "health",
        "infill",
        "lora-adapters",
        "metrics",
        "models",
        "models/load",
        "models/sse",
        "models/unload",
        "props",
        "rerank",
        "reranking",
        "responses",
        "responses/input_tokens",
        "slots",
        "tokenize",
        "tools",
    }
    served = {"props"}
    for route in bare_llama_routes - served:
        assert mod.is_engine_probe_path(route), route
    assert not mod.is_engine_probe_path("props"), "Studio serves /props"


def test_a_bare_llama_route_404s_on_every_method_a_client_would_use():
    mod = _load()
    with _client(mod) as c:
        for path in (
            "/chat/completions",
            "/responses",
            "/models",
            "/tools",
            "/audio/transcriptions",
        ):
            for method in ("GET", "HEAD", "POST"):
                r = c.request(method, path, json = {} if method == "POST" else None)
                assert r.status_code == 404, (method, path, r.status_code)
                assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_props_does_not_advertise_child_endpoints_studio_denies():
    """llama-server puts endpoint_slots / endpoint_props / endpoint_metrics in its props
    (tools/server/server-context.cpp), and Studio launches it with --metrics, so
    endpoint_metrics arrives true while public /metrics is one of the paths this module
    deliberately 404s. Copying the child's route table onto the public surface is the
    same over-claim as answering a probe with HTML."""
    upstream = {
        "endpoint_slots": True,
        "endpoint_props": True,
        "endpoint_metrics": True,
        "ui": True,
        "ui_settings": {"theme": "dark"},
        "cors_proxy_enabled": True,
        "total_slots": 4,
    }
    mod = _load(backend = _Backend(props = upstream))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["endpoint_slots"] is False
    assert body["endpoint_props"] is False
    assert body["endpoint_metrics"] is False
    for child_only in ("ui", "ui_settings", "cors_proxy_enabled"):
        assert child_only not in body, child_only
    # The fields that do describe the model still come through untouched.
    assert body["total_slots"] == 4


def test_the_denied_endpoints_props_advertises_really_are_denied():
    """Ties the two halves together: every endpoint_* flag set to False above names a
    path that actually 404s, so the props answer and the route table agree."""
    mod = _load()
    with _client(mod) as c:
        for path in ("/slots", "/metrics"):
            assert c.get(path).status_code == 404, path
        # POST /props is llama-server's props-change endpoint; Studio serves GET only.
        assert c.post("/props", json = {}).status_code in (404, 405)
