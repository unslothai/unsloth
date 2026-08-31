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
    """The guard runs after the static lookup, so a shipped file named `metrics` wins."""
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
    for path in ("/props", "/v1/props", "/version"):
        assert ("GET", path) in _INFERENCE_ROUTES, path
    # Not the Ollama paths: they are not served at all.
    for path in ("/api/tags", "/api/show", "/api/version"):
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
            "/models",
            "/tools",
            "/audio/transcriptions",
        ):
            for method in ("GET", "HEAD", "POST"):
                r = c.request(method, path, json = {} if method == "POST" else None)
                assert r.status_code == 404, (method, path, r.status_code)
                assert "text/html" not in r.headers.get("content-type", ""), (method, path)


def test_props_does_not_advertise_child_endpoints_studio_denies():
    """tools/server/server-context.cpp puts all three flags in the payload, and Studio
    launches with --metrics, so endpoint_metrics arrived true while /metrics 404s here."""
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
    """Every flag set False above names a path that really 404s."""
    mod = _load()
    with _client(mod) as c:
        for path in ("/slots", "/metrics"):
            assert c.get(path).status_code == 404, path
        # POST /props is llama-server's props-change endpoint; Studio serves GET only.
        assert c.post("/props", json = {}).status_code in (404, 405)


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
    ["/completion", "/completion/", "/health", "/metrics", "/slots", "/v1/health", "/slots/3"],
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
        for path in ("/props", "/v1/props", "/version"):
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
