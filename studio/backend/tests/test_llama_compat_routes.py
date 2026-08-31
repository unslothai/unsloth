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


def _inference_double(backend, catalog):
    """Stand-in for routes.inference, injected through llama_compat._inference().

    Deliberately not a sys.modules entry: an earlier revision stubbed
    routes.inference globally and every module imported while it was live captured
    the stub, which broke two unrelated route tests when the suite ran in one process.
    """
    inference = types.SimpleNamespace()
    inference.get_llama_cpp_backend = lambda: backend
    inference._llama_public_model_id = lambda b, fallback = None: b._openai_advertised_id
    inference._orchestrator_public_model_id = lambda b: None
    inference._peek_inference_backend = lambda: None

    async def _objects():
        return catalog

    inference._openai_catalog_objects = _objects
    return inference


_MOD = None


def _load(backend = None, catalog = None):
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
        for method, path in (
            ("GET", "/props"),
            ("GET", "/v1/props"),
            ("GET", "/version"),
            ("GET", "/api/tags"),
            ("POST", "/api/show"),
        ):
            r = c.request(method, path, json = {} if method == "POST" else None)
            assert r.status_code == 200, (method, path, r.status_code)
            assert "application/json" in r.headers["content-type"], (method, path)
            assert "text/html" not in r.headers["content-type"], (method, path)


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
    assert body["total_slots"] == 0


def test_v1_props_and_props_agree():
    mod = _load(backend = _Backend(props = {"total_slots": 2}))
    with _client(mod) as c:
        assert c.get("/props").json() == c.get("/v1/props").json()


# ── /version ──────────────────────────────────────────────────────────────────


def test_version_answers_on_both_spellings():
    mod = _load()
    with _client(mod) as c:
        bare = c.get("/version").json()
        prefixed = c.get("/api/version").json()
    assert bare == prefixed
    assert isinstance(bare["version"], str) and bare["version"]


# ── /api/tags ─────────────────────────────────────────────────────────────────


def test_tags_publishes_the_same_ids_as_v1_models():
    mod = _load()
    with _client(mod) as c:
        models = c.get("/api/tags").json()["models"]
    assert [m["name"] for m in models] == [m["id"] for m in CATALOG]
    assert all(m["name"] == m["model"] for m in models)


def test_tags_modified_at_is_rfc3339_not_an_epoch():
    """Ollama clients parse this as a timestamp; a bare integer makes them throw."""
    mod = _load()
    with _client(mod) as c:
        first = c.get("/api/tags").json()["models"][0]
    assert first["modified_at"] == "2023-11-14T22:13:20Z"


def test_tags_reports_the_quant_it_knows_and_no_invented_digest():
    mod = _load()
    with _client(mod) as c:
        models = c.get("/api/tags").json()["models"]
    assert models[0]["details"]["quantization_level"] == "UD-Q4_K_XL"
    # The unquantised catalog row must not borrow the previous row's value.
    assert models[1]["details"]["quantization_level"] == ""
    # Studio has no blob digest for these. Empty, never fabricated: a client that
    # compared a made-up digest would re-pull on every check.
    assert all(m["digest"] == "" and m["size"] == 0 for m in models)


def test_tags_on_an_empty_catalog_is_an_empty_list_not_an_error():
    mod = _load(catalog = [])
    with _client(mod) as c:
        r = c.get("/api/tags")
    assert r.status_code == 200
    assert r.json() == {"models": []}


# ── /api/show ─────────────────────────────────────────────────────────────────


def test_show_accepts_both_ollama_spellings():
    mod = _load()
    with _client(mod) as c:
        by_model = c.post("/api/show", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
        by_name = c.post("/api/show", json = {"name": "unsloth/Laguna-S-2.1-GGUF"})
    assert by_model.status_code == 200
    assert by_model.json() == by_name.json()


def test_show_matches_case_insensitively_like_v1_models():
    mod = _load()
    with _client(mod) as c:
        r = c.post("/api/show", json = {"model": "UNSLOTH/laguna-s-2.1-gguf"})
    assert r.status_code == 200
    assert r.json()["details"]["quantization_level"] == ""


def test_show_with_an_empty_body_falls_back_to_the_loaded_model():
    """A probe often POSTs {} just to see whether the endpoint exists. 422 would
    make it conclude the server is not Ollama-compatible."""
    mod = _load()
    with _client(mod) as c:
        r = c.post("/api/show", json = {})
    assert r.status_code == 200
    assert r.json()["details"]["quantization_level"] == "UD-Q4_K_XL"


def test_show_404s_an_unknown_model():
    mod = _load()
    with _client(mod) as c:
        r = c.post("/api/show", json = {"model": "not/a-real-model"})
    assert r.status_code == 404
    assert "text/html" not in r.headers["content-type"]


def test_show_404s_rather_than_500s_when_nothing_is_loaded_and_no_model_is_named():
    mod = _load(backend = _Backend(loaded = False))
    with _client(mod) as c:
        assert c.post("/api/show", json = {}).status_code == 404


# ── platform and robustness, from the sandbox run ─────────────────────────────


@pytest.mark.parametrize("osname", ["Linux", "Darwin", "Windows"])
@pytest.mark.parametrize(
    "created",
    [0, 1, -1, 1_700_000_000, 2**31, 2**40, 10**18, 1.5, None, "nope", float("nan")],
)
def test_modified_at_is_a_four_digit_year_stamp_on_every_platform(osname, created, monkeypatch):
    """gmtime() disagrees across the three platforms at both ends of its range: Linux
    accepts a huge epoch and returns a five digit year no client can parse, macOS
    rejects anything before 1970, Windows rejects both ends. Clamping first is what
    makes one catalog row render identically everywhere."""
    mod = _load()
    real = time.gmtime

    def platform_gmtime(value):
        if osname == "Windows" and not (0 <= value < 32536850000):
            raise OSError(22, "Invalid argument")
        if osname == "Darwin" and value < 0:
            raise OSError(22, "Invalid argument")
        return real(value)

    monkeypatch.setattr(mod.time, "gmtime", platform_gmtime)
    out = mod._rfc3339(created)
    time.strptime(out, "%Y-%m-%dT%H:%M:%SZ")
    assert len(out) == 20 and out.endswith("Z"), out
    assert 1970 <= int(out[:4]) <= 9999, out


def test_the_timestamp_fallback_does_not_itself_call_gmtime(monkeypatch):
    """An earlier revision fell back to gmtime(0), which raises too on a platform
    strict enough to have rejected the first call, taking /api/tags with it."""
    mod = _load()
    monkeypatch.setattr(
        mod.time, "gmtime", lambda _v: (_ for _ in ()).throw(OSError(22, "Invalid argument"))
    )
    assert mod._rfc3339(1_700_000_000) == "1970-01-01T00:00:00Z"
    with _client(mod) as c:
        assert len(c.get("/api/tags").json()["models"]) == len(CATALOG)


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
