# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""routes/llama_compat.py: the discovery surface a third-party client probes.

Reproduces a real user report. A client pointed at Studio's port ran this sequence:

    GET  /api/v1/models  404      GET  /props    200 text/html
    GET  /api/tags       404      GET  /version  200 text/html
    GET  /v1/props       404      POST /api/show 405

The two 200s are the bug: neither was a Studio route, so both fell through the SPA
catch-all and returned index.html, which a probe reads as "supported" before failing
on the body. The module is loaded standalone with a double for routes.inference.
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
        slots = 4,
    ):
        self.is_loaded = loaded
        self.effective_parallel_slots = slots
        self.model_identifier = "/media/models/qwen/model.gguf"
        self._openai_advertised_id = "unsloth/Qwen3.8-27B-GGUF"
        self.context_length = 262144
        self.chat_template = "{{ messages }}"
        self._props = props
        self._port = 52345
        self._auth_headers = {"Authorization": "Bearer child-key"}

    def _query_server_props(self):
        return self._props


def _inference_double(
    backend,
    catalog,
    orchestrator = None,
):
    """Stand-in for routes.inference. Deliberately not a sys.modules entry: stubbing it
    globally broke two unrelated route tests when the suite ran in one process."""
    inference = types.SimpleNamespace()
    inference.get_llama_cpp_backend = lambda: backend
    inference._llama_public_model_id = lambda b, fallback = None: b._openai_advertised_id
    inference._peek_inference_backend = lambda: orchestrator
    inference._orchestrator_public_model_id = lambda b: b.active_model_name

    async def _objects(quant_pins = False):
        return catalog

    inference._openai_catalog_objects = _objects
    inference.model_slots = types.SimpleNamespace(visible_loading = lambda: None)
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

    # The SPA catch-all in main.py's order. Without it a test cannot tell "serves JSON"
    # from "serves the app shell with a 200", which is the whole defect.
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
    """THE REGRESSION THIS FILE EXISTS FOR: every probed path answers JSON or 404,
    never 200 text/html."""
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
    """The control: without it the test above passes on a build that 404s everything,
    which would break every deep link into the UI."""
    mod = _load()
    with _client(mod) as c:
        r = c.get("/some-ui-deep-link")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


def test_engine_endpoints_studio_does_not_serve_are_404_not_html():
    mod = _load()
    with _client(mod) as c:
        for path in ("/completion", "/tokenize", "/detokenize", "/health"):
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
    """Upstream reports the absolute .gguf path and this is LAN reachable."""
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
    """A mid-restart engine returns None; the probe still answers from the local view."""
    mod = _load(backend = _Backend(props = None))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["model_path"] == "unsloth/Qwen3.8-27B-GGUF"
    assert body["default_generation_settings"]["n_ctx"] == 262144
    # Not 0: this same response advertises the model as loaded, and a client reading
    # props for capacity would conclude it cannot serve. The backend knows what it
    # launched with.
    assert body["total_slots"] == 4
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
    """A probe is the wrong place to find out that contract slipped."""

    class _Raises(_Backend):
        def _query_server_props(self):
            raise RuntimeError("engine unreachable")

    mod = _load(backend = _Raises())
    with _client(mod) as c:
        r = c.get("/props")
    assert r.status_code == 200
    assert r.json()["model_path"] == "unsloth/Qwen3.8-27B-GGUF"


def test_the_version_lookup_survives_an_import_failure(monkeypatch):
    """An ImportError above the guard would 500 a probe about something else."""
    mod = _load()
    mod._studio_version.cache_clear()
    monkeypatch.setitem(sys.modules, "utils.studio_version", None)
    try:
        assert mod._studio_version() == "dev"
    finally:
        mod._studio_version.cache_clear()


def test_the_version_lookup_is_resolved_once(monkeypatch):
    """Two git subprocesses per call, on the event loop, if uncached."""
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
    """The guard runs after the static lookup, so a shipped file named `health` wins."""
    mod = _load()
    (tmp_path / "health").write_text("asset-body")
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
        assert c.get("/health").text == "asset-body"
        assert c.get("/completion").status_code == 404


def test_no_ui_route_or_public_asset_takes_a_forwarded_get_path():
    """These GET routes sit ahead of the SPA catch-all, so a page or asset by that name would be shadowed."""
    frontend = _BACKEND.parent / "frontend"
    if not frontend.is_dir():
        pytest.skip("frontend sources not present in this checkout")
    mod = _load()
    forwarded = {path for method, path in mod._SERVED_ROUTES if method == "GET"}
    public = (
        {p.name.lower() for p in (frontend / "public").iterdir()}
        if (frontend / "public").is_dir()
        else set()
    )
    declared = set()
    for path in (frontend / "src" / "app" / "routes").glob("*.tsx"):
        for line in path.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if line.startswith('path: "') and line.endswith('",'):
                declared.add(line[len('path: "') : -2].strip("/").lower())
    assert not forwarded & public, forwarded & public
    assert not forwarded & declared, forwarded & declared


def test_no_client_side_ui_route_is_shadowed_by_the_deny_list():
    """Reads the routes the frontend declares, so a UI page named /metrics fails here
    rather than 404-ing in the browser."""
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
    """Answering /api/tags makes a client select Ollama and then fail on /api/chat.
    The reporting user's client got 404 here, fell back to OpenAI, and worked, so
    advertising the pair would have broken the very client this started from."""
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
        "/rerank",
        "/infill",
        "/embedding",
        "/apply-template",
        "/slots",
    ],
)
def test_unserved_engine_endpoints_404_on_their_real_method_not_405(method, path):
    """POST endpoints in llama-server, so before this they returned 405 from the
    GET-only catch-all, which reads as "exists, wrong method"."""
    mod = _load()
    with _client(mod) as c:
        r = c.request(method, path, json = {})
    assert r.status_code == 404, (method, path, r.status_code)
    assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_get_on_a_probe_path_still_reaches_the_asset_lookup(tmp_path):
    """Omitting GET is what keeps the catch-all's static lookup reachable."""
    mod = _load()
    assert not any(
        "GET" in (getattr(r, "methods", None) or set()) and r.path == "/completion"
        for r in mod.router.routes
    )


def test_props_does_not_pair_an_orchestrator_model_with_llama_server_fields():
    """With MLX resident the llama backend is unloaded, so reading its fields anyway
    described a serving model as having no context and no slots."""

    class _Orchestrator:
        active_model_name = "unsloth/Llama-3.2-3B"

    mod = _load(backend = _Backend(loaded = False), orchestrator = _Orchestrator())
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["model_path"] == "unsloth/Llama-3.2-3B"
    for llama_only in ("total_slots", "chat_template", "default_generation_settings"):
        assert llama_only not in body, llama_only


def test_the_probe_paths_are_admitted_under_keyless_inference_scope():
    """A 401 on /props in front of a surface the scope grants reads as an auth wall
    over the whole thing and stops discovery."""
    from utils.keyless_api_access import _INFERENCE_ROUTES

    assert ("GET", "/v1/models") in _INFERENCE_ROUTES, "baseline moved; re-derive this"
    for path in ("/props", "/v1/props", "/version", "/models"):
        assert ("GET", path) in _INFERENCE_ROUTES, path
    for path in ("/models/load", "/models/unload"):
        assert ("POST", path) not in _INFERENCE_ROUTES, path
    # Not the Ollama paths: they are not served at all.
    for path in ("/api/tags", "/api/show", "/api/version"):
        assert ("GET", path) not in _INFERENCE_ROUTES, path
    for path in ("/tokenize", "/detokenize"):
        assert ("POST", path) in _INFERENCE_ROUTES, path
    for path in ("/slots", "/metrics"):
        assert ("GET", path) not in _INFERENCE_ROUTES, path


# ── round two ─────────────────────────────────────────────────────────────────


def test_head_on_an_unserved_probe_path_is_404_not_405():
    """Starlette does NOT admit HEAD on a GET route (measured on the pinned fastapi
    0.141.1 / starlette 1.6.0). A comment here previously claimed the opposite."""
    mod = _load()
    with _client(mod) as c:
        for path in ("/completion", "/tokenize", "/metrics", "/slots"):
            assert c.request("HEAD", path).status_code == 404, path


def test_head_is_405_on_a_plain_get_route_which_is_why_it_is_listed():
    """Pins the framework behaviour the fix depends on."""
    app = FastAPI()

    @app.get("/{p:path}")
    async def _catchall(p: str):
        return {"p": p}

    with TestClient(app) as c:
        assert c.get("/anything").status_code == 200
        assert c.request("HEAD", "/anything").status_code == 405


@pytest.mark.parametrize("path", ["/slots/0", "/slots/3", "/slots/abc", "/slots/0/"])
def test_the_nested_slot_endpoint_is_denied_too(path):
    """A real llama-server route Studio calls itself, missed by exact membership."""
    mod = _load()
    assert mod.is_engine_probe_path(path.strip("/"))
    with _client(mod) as c:
        for method in ("GET", "HEAD", "POST", "DELETE"):
            r = c.request(method, path, json = {} if method == "POST" else None)
            assert r.status_code == 404, (method, path, r.status_code)


def test_the_deny_list_matches_llama_servers_own_route_table():
    """Transcribed from tools/server/server.cpp. /props is the only bare route Studio
    serves, so anything else missing means a probe still reaches the app shell."""
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
            "/models/sse",
            "/tools",
            "/audio/transcriptions",
        ):
            for method in ("GET", "HEAD", "POST"):
                r = c.request(method, path, json = {} if method == "POST" else None)
                assert r.status_code == 404, (method, path, r.status_code)
                assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_props_advertises_only_the_child_endpoints_studio_forwards():
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
    assert body["endpoint_slots"] is True
    assert body["endpoint_props"] is False
    assert body["endpoint_metrics"] is True
    for child_only in ("ui", "ui_settings", "cors_proxy_enabled"):
        assert child_only not in body, child_only
    # The fields that do describe the model still come through untouched.
    assert body["total_slots"] == 4


def test_props_does_not_advertise_endpoints_the_child_has_off():
    upstream = {"endpoint_slots": False, "endpoint_metrics": False}
    mod = _load(backend = _Backend(props = upstream))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["endpoint_slots"] is False
    assert body["endpoint_metrics"] is False


def test_props_does_not_advertise_forwarded_endpoints_with_nothing_loaded():
    mod = _load(backend = _Backend(loaded = False, props = {"endpoint_metrics": True}))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["endpoint_slots"] is False
    assert body["endpoint_metrics"] is False


def test_props_does_not_advertise_forwarded_endpoints_to_a_managed_account(monkeypatch):
    mod = _load(backend = _Backend(props = {"endpoint_slots": True, "endpoint_metrics": True}))
    monkeypatch.setattr(mod, "_caller_is_managed_account", lambda: True)
    monkeypatch.setattr(mod, "_resident_hidden_from_caller", lambda: False)
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["endpoint_slots"] is False
    assert body["endpoint_metrics"] is False


def test_post_props_is_still_not_served():
    mod = _load()
    with _client(mod) as c:
        assert c.post("/props", json = {}).status_code in (404, 405)


class _Upstream:
    def __init__(self, *, fail = False):
        self.calls = []
        self.fail = fail

    def handler(self, request):
        import httpx

        self.calls.append(request)
        if self.fail:
            raise httpx.ConnectError("connection refused", request = request)
        path = request.url.path
        if path == "/metrics":
            return httpx.Response(
                200,
                content = b"llamacpp:prompt_tokens_total 12\n",
                headers = {"content-type": "text/plain; version=0.0.4"},
            )
        if path == "/slots":
            return httpx.Response(200, json = [{"id": 0, "is_processing": False}])
        if path == "/tokenize":
            return httpx.Response(200, json = {"tokens": [9906, 1917]})
        if path == "/detokenize":
            return httpx.Response(200, json = {"content": "Hello world"})
        return httpx.Response(404, json = {"error": "not found"})


def _with_upstream(monkeypatch, mod, upstream):
    import httpx
    monkeypatch.setattr(
        mod,
        "_upstream_client",
        lambda: httpx.AsyncClient(transport = httpx.MockTransport(upstream.handler)),
    )


@pytest.mark.parametrize(
    "method, path, expect",
    [
        ("GET", "/slots", [{"id": 0, "is_processing": False}]),
        ("POST", "/tokenize", {"tokens": [9906, 1917]}),
        ("POST", "/detokenize", {"content": "Hello world"}),
    ],
)
def test_the_forwarded_endpoints_reach_the_child_with_its_key(monkeypatch, method, path, expect):
    import json

    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        r = c.request(method, path, json = {"content": "Hello world"} if method == "POST" else None)
    assert r.status_code == 200, r.text
    assert r.json() == expect
    (sent,) = upstream.calls
    assert sent.method == method
    assert str(sent.url) == "http://127.0.0.1:52345" + path
    assert sent.headers["authorization"] == "Bearer child-key"
    if method == "POST":
        assert json.loads(sent.content) == {"content": "Hello world"}
        assert sent.headers["content-type"].startswith("application/json")


def test_metrics_reaches_the_child_and_keeps_the_prometheus_content_type(monkeypatch):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        r = c.get("/metrics")
    assert r.status_code == 200
    assert r.content == b"llamacpp:prompt_tokens_total 12\n"
    assert r.headers["content-type"].startswith("text/plain"), r.headers["content-type"]
    assert upstream.calls[0].headers["authorization"] == "Bearer child-key"


def test_the_slash_form_is_forwarded_too(monkeypatch):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        assert c.post("/tokenize/", json = {"content": "x"}).status_code == 200
        assert c.get("/metrics/").status_code == 200
    assert [r.url.path for r in upstream.calls] == ["/tokenize", "/metrics"]


def test_the_query_string_is_forwarded(monkeypatch):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        c.get("/slots?fail_on_no_slot=1")
    assert upstream.calls[0].url.params["fail_on_no_slot"] == "1"


def test_nothing_loaded_is_503_in_llama_servers_error_shape(monkeypatch):
    mod = _load(backend = _Backend(loaded = False))
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        r = c.post("/tokenize", json = {"content": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == 503
    assert upstream.calls == []


def test_an_unload_racing_the_request_is_503_not_500(monkeypatch):
    """unload_model flips is_loaded before it clears _port."""
    backend = _Backend()
    backend._port = None
    mod = _load(backend = backend)
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        r = c.get("/slots")
    assert r.status_code == 503
    assert upstream.calls == []


def test_an_unreachable_child_is_503_not_500(monkeypatch):
    mod = _load()
    _with_upstream(monkeypatch, mod, _Upstream(fail = True))
    with _client(mod) as c:
        r = c.get("/metrics")
    assert r.status_code == 503
    assert "text/html" not in r.headers.get("content-type", "")


def test_a_managed_account_gets_the_404_and_the_child_is_never_asked(monkeypatch):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    monkeypatch.setattr(mod, "_caller_is_managed_account", lambda: True)
    with _client(mod) as c:
        for method, path in (("GET", "/metrics"), ("GET", "/slots"), ("POST", "/tokenize")):
            r = c.request(method, path, json = {} if method == "POST" else None)
            assert r.status_code == 404, (method, path, r.status_code)
    assert upstream.calls == []


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/slots"),
        ("POST", "/slots/0"),
        ("DELETE", "/slots/0"),
        ("POST", "/metrics"),
        ("GET", "/tokenize"),
        ("PUT", "/detokenize"),
    ],
)
def test_only_the_read_only_methods_are_forwarded(monkeypatch, method, path):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _client(mod) as c:
        r = c.request(method, path, json = {} if method in ("POST", "PUT") else None)
    assert r.status_code == 404, (method, path, r.status_code)
    assert upstream.calls == []


def test_the_forwarded_gets_survive_api_only_mode(monkeypatch):
    mod = _load()
    upstream = _Upstream()
    _with_upstream(monkeypatch, mod, upstream)
    with _api_only_client(mod) as c:
        assert c.get("/metrics").status_code == 200
        assert c.get("/slots").status_code == 200
        assert c.get("/health").status_code == 404


# ── round three ───────────────────────────────────────────────────────────────


def test_a_transient_props_failure_does_not_report_zero_slots():
    """total_slots 0 beside a model this response advertises as loaded reads as
    "cannot serve"."""
    mod = _load(backend = _Backend(props = None, slots = 8))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert body["model_path"] == "unsloth/Qwen3.8-27B-GGUF"
    assert body["total_slots"] == 8


@pytest.mark.parametrize("slots", [0, None, "x", -1])
def test_an_unreadable_slot_count_is_omitted_rather_than_reported_as_zero(slots):
    mod = _load(backend = _Backend(props = None, slots = slots))
    with _client(mod) as c:
        body = c.get("/props").json()
    assert "total_slots" not in body, body.get("total_slots")
    assert body["model_path"] == "unsloth/Qwen3.8-27B-GGUF"


def test_a_backend_that_raises_on_the_slot_count_still_answers():
    class _Raises(_Backend):
        # A property, not the plain attribute the base class sets, so the read itself
        # raises the way a backend mid-restart would.
        @property
        def effective_parallel_slots(self):
            raise RuntimeError("backend mid-restart")

        @effective_parallel_slots.setter
        def effective_parallel_slots(self, _value):
            pass

    mod = _load(backend = _Raises(props = None))
    with _client(mod) as c:
        r = c.get("/props")
    assert r.status_code == 200
    assert "total_slots" not in r.json()


def test_the_upstream_slot_count_still_wins_when_it_can_be_read():
    mod = _load(backend = _Backend(props = {"total_slots": 2}, slots = 8))
    with _client(mod) as c:
        assert c.get("/props").json()["total_slots"] == 2


@pytest.mark.parametrize(
    "path",
    [
        "/v1/rerank",
        "/v1/reranking",
        "/v1/chat/completions/control",
        "/v1/chat/completions/input_tokens",
        "/v1/responses/input_tokens",
        "/v1/health",
        "/v1/stream",
        "/v1/streams/lookup",
    ],
)
@pytest.mark.parametrize("method", ["HEAD", "POST", "DELETE"])
def test_unserved_v1_aliases_404_on_their_real_method(path, method):
    """GET already 404s through main.py's /v1/ rule; the rest came back 405."""
    mod = _load()
    with _client(mod) as c:
        r = c.request(method, path, json = {} if method == "POST" else None)
    assert r.status_code == 404, (method, path, r.status_code)
    assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_the_v1_deny_list_never_shadows_a_path_studio_serves():
    """The drift guard: implementing one of these must fail here rather than let a 404
    shadow a real route."""
    import os

    os.environ.setdefault("TMPDIR", os.environ.get("UNSLOTH_WORKSPACE", "/tmp") + "/temp")
    try:
        import main
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"main.py not importable in this environment: {exc}")

    from routes.llama_compat import _probe_not_found

    def _record(into, path, endpoint):
        # This module registers its own 404 handlers on exactly these paths, so a naive
        # walk reports every one of them as served and the guard can never fire.
        if path and endpoint is not _probe_not_found:
            into.add(path)

    served = set()
    for route in main.app.routes:
        _record(served, getattr(route, "path", None), getattr(route, "endpoint", None))
        context = getattr(route, "include_context", None)
        inner = getattr(context, "included_router", None) if context is not None else None
        if inner is not None:
            prefix = getattr(context, "prefix", "")
            for sub in inner.routes:
                _record(
                    served,
                    (prefix + getattr(sub, "path", "")) if getattr(sub, "path", None) else None,
                    getattr(sub, "endpoint", None),
                )

    assert "/v1/models" in served, "route walk found nothing; re-derive this"
    mod = _load()
    for probe in mod._UNSERVED_V1_PROBE_PATHS:
        assert (
            f"/{probe}" not in served
        ), f"Studio now serves /{probe}; drop it from _UNSERVED_V1_PROBE_PATHS"


@pytest.mark.parametrize("path", ["/props/", "/v1/props/", "/version/"])
def test_the_trailing_slash_forms_answer_json_not_the_app_shell(path):
    """No slash redirect fires here: the catch-all fully matches "/props/", so before
    this these returned the app shell with a 200, the defect this module exists to fix."""
    mod = _load()
    with _client(mod) as c:
        r = c.get(path)
    assert r.status_code == 200, (path, r.status_code)
    assert "application/json" in r.headers["content-type"], path


def test_the_slash_and_bare_forms_agree():
    mod = _load()
    with _client(mod) as c:
        assert c.get("/props").json() == c.get("/props/").json()
        assert c.get("/version").json() == c.get("/version/").json()


# ── 405 leaks the same signal a 200 did ───────────────────────────────────────


def _lifespan_app(mod, *, frontend_mounted = False):
    """An app wired the way main.py wires one: installed at startup, because that is
    the point both launch paths reach after the frontend decision."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        if not getattr(app.state, "frontend_mounted", False):
            mod.add_get_denials(app)
        yield

    app = FastAPI(lifespan = lifespan)
    app.include_router(mod.router)
    if frontend_mounted:
        app.state.frontend_mounted = True
    app.dependency_overrides[mod.get_current_subject] = lambda: "test"
    return app


def _api_only_client(mod):
    return TestClient(_lifespan_app(mod))


def _directly_added_get_paths(app):
    """Paths the app itself registered for GET. include_router() is lazy in this FastAPI
    (routes stay behind an _IncludedRouter), so this sees only add_api_route() calls."""
    return {
        r.path
        for r in app.routes
        if getattr(r, "methods", None) and "GET" in r.methods and hasattr(r, "path")
    }


def test_the_denials_are_installed_only_when_no_frontend_is_mounted():
    """So an app already serving a catch-all does not carry a second, dead copy of
    every engine path. Ordering, not this flag, is what keeps assets reachable."""
    mod = _load()
    with TestClient(_lifespan_app(mod, frontend_mounted = False)):
        pass
    unmounted = _directly_added_get_paths(_started(_lifespan_app(mod, frontend_mounted = False)))
    mounted = _directly_added_get_paths(_started(_lifespan_app(mod, frontend_mounted = True)))
    assert "/completion" in unmounted, "API-only mode must gain the GET denial"
    assert "/completion" not in mounted, "a mounted frontend must not gain a second copy"


def _started(app):
    with TestClient(app):
        pass
    return app


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/completion/"),
        ("HEAD", "/metrics/"),
        ("PUT", "/tokenize/"),
        ("DELETE", "/slots/"),
        ("POST", "/v1/rerank/"),
        ("POST", "/v1/health/"),
        ("POST", "/slots/3/"),
    ],
)
def test_the_slash_form_of_a_denied_path_404s_rather_than_405(method, path):
    """405 says "endpoint exists, wrong method", which is the signal this module removes.
    No redirect rescues these: the catch-all is GET-only, so the slash form matched it on
    path and missed on method."""
    mod = _load()
    with _client(mod) as c:
        r = c.request(method, path)
    assert r.status_code == 404, (method, path, r.status_code)


@pytest.mark.parametrize(
    "path",
    ["/completion", "/completion/", "/health", "/tokenize", "/v1/health", "/slots/3"],
)
def test_a_get_probe_404s_in_api_only_mode(path):
    """--api-only mounts no frontend, so the catch-all that supplies the GET 404 is not
    registered and these answered 405 on method alone."""
    mod = _load()
    with _api_only_client(mod) as c:
        r = c.get(path)
    assert r.status_code == 404, (path, r.status_code)


def test_api_only_mode_still_serves_the_three_real_routes():
    mod = _load()
    with _api_only_client(mod) as c:
        for path in ("/props", "/v1/props", "/version", "/models"):
            r = c.get(path)
            assert r.status_code == 200, (path, r.status_code)
            assert "application/json" in r.headers["content-type"], path


def test_the_get_denials_are_not_added_when_a_frontend_is_mounted():
    """A shipped asset named like an engine path must still be served by the catch-all,
    so the GET denial is added only for an app that has no catch-all at all."""
    mod = _load()
    app = FastAPI()
    app.include_router(mod.router)

    @app.get("/{full_path:path}")
    async def _spa(full_path: str):
        return Response(content = b"asset", media_type = "text/plain")

    app.dependency_overrides[mod.get_current_subject] = lambda: "test"
    with TestClient(app) as c:
        assert c.post("/completion").status_code == 404, "sanity: the deny route is live"
        r = c.get("/completion")
    assert (
        r.status_code == 200 and r.content == b"asset"
    ), "the router claimed GET, so a shipped asset by that name became unreachable"


def test_the_version_lookup_leaves_the_event_loop():
    """get_studio_version() shells out to git twice on a source checkout; on the event
    loop that stalls every other request, including the launcher's health probe."""
    import inspect

    src = inspect.getsource(_load().studio_version)
    assert "to_thread" in src, "the version handler must resolve off the event loop"


def test_props_hides_a_resident_that_belongs_to_another_account(monkeypatch):
    from hub.services.models import account_access

    upstream = {"model_path": "/media/models/alice-private.gguf", "chat_template": "ALICE_TEMPLATE"}
    mod = _load(backend = _Backend(props = upstream))
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "resident_hidden", lambda modality, reference = None: True)
    with _client(mod) as c:
        for path in ("/props", "/props/", "/v1/props", "/v1/props/"):
            raw = c.get(path)
            assert raw.status_code == 200
            assert raw.json() == {"loaded": True, "yours": False}
            assert "ALICE_TEMPLATE" not in raw.text and "Qwen" not in raw.text

    monkeypatch.setattr(account_access, "resident_hidden", lambda modality, reference = None: False)
    with _client(mod) as c:
        assert c.get("/props").json()["chat_template"] == "ALICE_TEMPLATE"
    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    monkeypatch.setattr(
        account_access,
        "resident_hidden",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("owner path consulted policy")),
    )
    with _client(mod) as c:
        assert c.get("/props").json()["model_path"] == "unsloth/Qwen3.8-27B-GGUF"


# ── Open WebUI llama.cpp model management (#10610) ──────────────────────────

QWEN = "unsloth/Qwen3.8-27B-GGUF"


def _management(
    load_status = 200,
    catalog = None,
    backend = None,
):
    """The routes double plus Studio's load and unload lifecycle, recording what they were asked."""
    mod = _load(backend = backend, catalog = catalog)
    double = mod._inference()
    calls = []

    async def _load_gated(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        calls.append(("load", request.model_path, request.gguf_variant, request._advertised_id))
        if load_status != 200:
            raise HTTPException(status_code = load_status, detail = "busy")
        return {"status": "loaded", "model": request.model_path}

    async def _unload_impl(request, current_subject):
        calls.append(("unload", request.model_path))
        return {"status": "unloaded", "model": request.model_path}

    async def _tunnel_safe_json(coro, *, label):
        return await coro

    async def _gpu_ids_resolve(gpu_ids, kind):
        return True

    double.load_model_gated = _load_gated
    double._unload_model_impl = _unload_impl
    double._tunnel_safe_json = _tunnel_safe_json
    double._override_gpu_ids_still_resolve = _gpu_ids_resolve
    return mod, calls


@pytest.fixture
def _no_local_gguf(monkeypatch):
    import core.inference.local_model_resolver as resolver
    import utils.openai_auto_switch_settings as overrides

    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(overrides, "resolve_override_for_load", lambda *a, **k: (None, {}))
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *a, **k: True)
    return resolver


def test_management_list_is_the_v1_chat_catalog():
    media = {"id": "unsloth/FLUX.2-klein-GGUF", "task": "text-to-image", "loaded": True}
    mod, _ = _management(catalog = CATALOG + [media])
    with _client(mod) as c:
        for path in ("/models", "/models/"):
            r = c.get(path)
            assert r.status_code == 200, path
            assert r.json() == {"object": "list", "data": CATALOG}


def test_management_load_resolves_a_listed_id_through_studios_load(_no_local_gguf):
    mod, calls = _management()
    with _client(mod) as c:
        assert c.post("/models/load", json = {"model": QWEN}).status_code == 200
        # A quant this machine does not have is refused, never fetched.
        assert c.post("/models/load/", json = {"model": QWEN.lower() + ":Q8_0"}).status_code == 404
    assert calls == [("load", QWEN, None, None)]


def test_management_load_keeps_a_colon_in_an_exact_row_id(_no_local_gguf):
    """An NPU row such as "lemonade:qwen" is the id itself, not a repo with a quant pin."""
    row = {"id": "lemonade:qwen", "object": "model", "loaded": True}
    mod, calls = _management(catalog = [row])
    with _client(mod) as c:
        assert c.post("/models/load", json = {"model": "lemonade:qwen"}).status_code == 200
    assert calls == [("load", "lemonade:qwen", None, None)]


def test_management_load_keeps_the_listed_alias_advertised(monkeypatch, _no_local_gguf):
    """A path load would otherwise reappear in /v1/models under the path's basename."""
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/models/qwen", "UD-Q4_K_XL", "lmstudio-community/qwen"),
    )
    mod, calls = _management()
    with _client(mod) as c:
        r = c.post("/models/load", json = {"model": QWEN})
    assert r.status_code == 200
    # The answer names the listed row, never the on-disk path.
    assert r.json() == {"success": True, "model": QWEN}
    assert calls == [("load", "/models/qwen", "UD-Q4_K_XL", "lmstudio-community/qwen")]


def test_management_load_applies_saved_launch_settings(monkeypatch, _no_local_gguf):
    """The settings auto-switch would launch this row with, looked up by path then alias."""
    import utils.openai_auto_switch_settings as overrides

    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        # A single-file GGUF resolves with no quant, and is still llama.cpp's.
        lambda requested, **k: ("/models/qwen.gguf", None, QWEN),
    )
    asked = []

    def _override(
        load_id,
        alias_id = None,
        variant = None,
    ):
        asked.append((load_id, alias_id, variant))
        return load_id, {"n_parallel": 2, "kv_cache_dtype": "q8_0", "gpu_layers": 20}

    monkeypatch.setattr(overrides, "resolve_override_for_load", _override)
    mod, _ = _management()
    seen = []
    double = mod._inference()
    original = double.load_model_gated

    async def _capture(request, *args, **kwargs):
        seen.append((request.n_parallel, request.cache_type_kv, request.gpu_layers))
        return await original(request, *args, **kwargs)

    double.load_model_gated = _capture
    with _client(mod) as c:
        assert c.post("/models/load", json = {"model": QWEN}).status_code == 200
    assert asked == [("/models/qwen.gguf", QWEN, None)]
    assert seen == [(2, "q8_0", 20)]


def test_management_unload_maps_the_advertised_alias_to_the_resident(_no_local_gguf):
    """/unload matches the resident's identifier, not an alias such as an LM Studio name."""
    mod, calls = _management()
    with _client(mod) as c:
        r = c.post("/models/unload", json = {"model": QWEN.lower()})
    assert r.status_code == 200
    # The answer names the listed id, never the resident's path.
    assert r.json() == {"success": True, "model": QWEN}
    assert calls == [("unload", "/media/models/qwen/model.gguf")]


def test_management_unload_of_an_idle_row_never_reaches_the_lifecycle(_no_local_gguf):
    """/unload releases a managed account's shared resident before it compares names."""
    mod, calls = _management()
    with _client(mod) as c:
        r = c.post("/models/unload", json = {"model": CATALOG[1]["id"]})
    assert r.json() == {"success": True, "model": CATALOG[1]["id"]}
    assert calls == []


def test_management_unload_cancels_a_load_under_the_path_it_runs_as(monkeypatch, _no_local_gguf):
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/models/laguna", "Q4_K_M", requested),
    )
    backend = _Backend(loaded = False)
    backend.is_active = True
    backend.model_identifier = "/models/laguna"
    backend._openai_advertised_id = None
    mod, calls = _management(backend = backend)
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": CATALOG[1]["id"]}).status_code == 200
    assert calls == [("unload", "/models/laguna")]


@pytest.mark.parametrize("path", ["/models/load", "/models/unload"])
def test_management_rejects_unlisted_task_or_missing_models(path, _no_local_gguf):
    media = {"id": "unsloth/FLUX.2-klein-GGUF", "task": "text-to-image", "loaded": True}
    mod, calls = _management(catalog = CATALOG + [media])
    with _client(mod) as c:
        assert c.post(path, json = {"model": "someone/else"}).status_code == 404
        # Image, video and speech rows load through their own engines, not /load and /unload.
        assert c.post(path, json = {"model": media["id"]}).status_code == 404
        assert c.post(path, json = {"model": ""}).status_code == 422
        assert c.post(path, json = {}).status_code == 422
    assert calls == []


def test_management_load_keeps_studios_error_status(_no_local_gguf):
    mod, _ = _management(load_status = 409)
    with _client(mod) as c:
        assert c.post("/models/load", json = {"model": QWEN}).status_code == 409


def test_management_requires_a_credential():
    mod, calls = _management()
    app = FastAPI()
    app.include_router(mod.router)
    with TestClient(app) as c:
        assert c.get("/models").status_code == 401
        for path in ("/models/load", "/models/unload"):
            assert c.post(path, json = {"model": QWEN}).status_code == 401
    assert calls == []


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/models"),
        ("DELETE", "/models"),
        ("HEAD", "/models/load"),
        ("GET", "/models/load"),
        ("GET", "/models/unload"),
        ("GET", "/models/sse"),
    ],
)
def test_unserved_management_operations_still_404(method, path):
    mod, _ = _management()
    with _client(mod) as c:
        r = c.request(method, path, json = {} if method == "POST" else None)
    assert r.status_code == 404, (method, path, r.status_code)
    assert "text/html" not in r.headers.get("content-type", "")


def test_management_unload_matches_an_ollama_tag_before_its_suffix(_no_local_gguf):
    """ ":latest" is part of the advertised Ollama id, not a quant pin to strip first."""
    tag = "ollama/llama3:latest"
    backend = _Backend()
    backend.model_identifier = "ollama-manifest:/models/llama3/latest"
    backend._openai_advertised_id = tag
    mod = _load(backend = backend, catalog = [{"id": tag, "object": "model", "loaded": True}])
    calls = []

    async def _unload_impl(request, current_subject):
        calls.append(request.model_path)

    async def _tunnel_safe_json(coro, *, label):
        return await coro

    mod._inference()._unload_model_impl = _unload_impl
    mod._inference()._tunnel_safe_json = _tunnel_safe_json
    with _client(mod) as c:
        r = c.post("/models/unload", json = {"model": tag})
    assert r.json() == {"success": True, "model": tag}
    assert calls == ["ollama-manifest:/models/llama3/latest"]


def test_management_unload_finds_a_path_loaded_row_with_no_alias_stamp(monkeypatch, _no_local_gguf):
    """A Studio load by path advertises nothing, yet the catalog lists the row under its own id."""
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/models/lms/qwen", None, requested),
    )
    backend = _Backend()
    backend.model_identifier = "/models/lms/qwen"
    backend._openai_advertised_id = None
    mod, calls = _management(backend = backend)
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": QWEN}).status_code == 200
    assert calls == [("unload", "/models/lms/qwen")]


def test_management_unload_leaves_an_unrelated_load_alone(monkeypatch, _no_local_gguf):
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/models/laguna", None, requested),
    )
    backend = _Backend(loaded = False)
    backend.is_active = True
    backend.model_identifier = "/models/other"
    backend._openai_advertised_id = None
    mod, calls = _management(backend = backend)
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": CATALOG[1]["id"]}).status_code == 200
    assert calls == []


def test_management_unload_keeps_a_resident_of_another_quant(_no_local_gguf):
    backend = _Backend()
    backend.hf_variant = "UD-Q4_K_XL"
    mod, calls = _management(backend = backend)
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": QWEN + ":Q8_0"}).status_code == 200
        assert c.post("/models/unload", json = {"model": QWEN + ":UD-Q4_K_XL"}).status_code == 200
    assert calls == [("unload", "/media/models/qwen/model.gguf")]


def test_management_unload_of_a_bare_id_ignores_the_preferred_quant(monkeypatch, _no_local_gguf):
    """A bare id resolves to the preferred quant on disk, which need not be the resident one."""
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/media/models/qwen/model.gguf", "Q4_K_M", requested),
    )
    backend = _Backend()
    backend.hf_variant = "Q8_0"
    mod, calls = _management(backend = backend)
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": QWEN}).status_code == 200
    assert calls == [("unload", "/media/models/qwen/model.gguf")]


def test_management_unload_cancels_a_switch_while_the_old_model_stays_active(
    monkeypatch, _no_local_gguf
):
    monkeypatch.setattr(
        _no_local_gguf,
        "resolve_local_gguf",
        lambda requested, **k: ("/models/laguna", None, requested),
    )
    orchestrator = types.SimpleNamespace(
        active_model_name = "/models/old",
        get_loading_model = lambda: "/models/laguna",
        _openai_advertised_id = None,
    )
    mod = _load(backend = _Backend(loaded = False), orchestrator = orchestrator)
    double = mod._inference()
    calls = []

    async def _unload_impl(request, current_subject):
        calls.append(request.model_path)

    async def _tunnel_safe_json(coro, *, label):
        return await coro

    double._unload_model_impl = _unload_impl
    double._tunnel_safe_json = _tunnel_safe_json
    with _client(mod) as c:
        assert c.post("/models/unload", json = {"model": CATALOG[1]["id"]}).status_code == 200
    assert calls == ["/models/laguna"]
