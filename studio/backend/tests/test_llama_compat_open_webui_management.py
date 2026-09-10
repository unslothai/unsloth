# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Open WebUI llama.cpp management compatibility (#10610).

Covers:
- residency metadata (``loaded`` + ``status.value``) that survives OWUI's merge
- ``GET /models`` management catalog (outside ``/v1``)
- authenticated ``POST /models/load`` / ``POST /models/unload``
- route separation from ``GET /v1/models``
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ── Open WebUI provider merge (extracted contract from openai.py @ 0a7c158) ──

LLAMACPP_LOADED_STATES = {"loaded", "sleeping"}
LLAMACPP_UNLOADED_STATES = {"loading", "unloaded"}


def get_provider_model_loaded_state(
    model: dict,
    provider: str,
    manual_model_ids: bool = False,
):
    """Pinned Open WebUI helper: llama.cpp without ``status`` is assumed loaded."""
    if provider != "llama.cpp":
        return None
    status = model.get("status")
    if isinstance(status, dict):
        value = status.get("value")
        if value in LLAMACPP_LOADED_STATES:
            return True
        if value in LLAMACPP_UNLOADED_STATES:
            return False
    if not manual_model_ids and "status" not in model:
        return True
    return None


def get_merged_models(model_lists, *, provider: str = "llama.cpp"):
    models = {}
    for model_list in model_lists:
        if model_list is None or "error" in (model_list or {}):
            continue
        for model in model_list:
            model_id = model.get("id") or model.get("name")
            if not model_id or model_id in models:
                continue
            merged = {**model, "provider": provider}
            loaded = get_provider_model_loaded_state(model, provider)
            if loaded is not None:
                merged["loaded"] = loaded
            models[model_id] = merged
    return models


# ── llama_compat harness ─────────────────────────────────────────────────────

_MOD = None


class _Backend:
    def __init__(self, *, loaded = False):
        self.is_loaded = loaded
        self.model_identifier = "/media/models/qwen/model.gguf"
        self._openai_advertised_id = "unsloth/Qwen3.8-27B-GGUF"
        self.context_length = 4096
        self.hf_variant = "Q4_K_M"
        self.effective_parallel_slots = 4
        self.chat_template = ""

    def _query_server_props(self):
        return None


def _catalog(*, resident = "unsloth/Qwen3.8-27B-GGUF", extra_unloaded = True):
    rows = []
    if resident:
        rows.append(
            {
                "id": resident,
                "object": "model",
                "created": 1,
                "owned_by": "unsloth-studio",
                "loaded": True,
                "status": {"value": "loaded"},
            }
        )
    if extra_unloaded:
        rows.append(
            {
                "id": "unsloth/Laguna-S-2.1-GGUF",
                "object": "model",
                "created": 1,
                "owned_by": "unsloth-studio",
                "loaded": False,
                "status": {"value": "unloaded"},
            }
        )
    return rows


def _inference_double(
    catalog,
    *,
    load_impl = None,
    unload_impl = None,
    backend = None,
):
    state = {"resident": {m["id"] for m in catalog if m.get("loaded")}}

    async def _objects():
        out = []
        for row in catalog:
            loaded = row["id"] in state["resident"]
            out.append(
                {
                    **row,
                    "loaded": loaded,
                    "status": {"value": "loaded" if loaded else "unloaded"},
                }
            )
        return out

    async def _load(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        if load_impl is not None:
            return await load_impl(
                request, fastapi_request, current_subject, user_initiated = user_initiated
            )
        mid = getattr(request, "model_path", None)
        state["resident"].add(mid)
        for row in catalog:
            if row["id"] == mid or mid.endswith(row["id"].split("/")[-1]):
                state["resident"].add(row["id"])
        return types.SimpleNamespace(status = "loaded", model = mid)

    async def _unload(request, current_subject):
        if unload_impl is not None:
            return await unload_impl(request, current_subject)
        mid = getattr(request, "model_path", None)
        state["resident"].discard(mid)
        return types.SimpleNamespace(status = "unloaded", model = mid)

    inference = types.SimpleNamespace()
    inference.get_llama_cpp_backend = lambda: backend or _Backend(loaded = bool(state["resident"]))
    inference._llama_public_model_id = lambda b, fallback = None: getattr(
        b, "_openai_advertised_id", fallback
    )
    inference._peek_inference_backend = lambda: None
    inference._orchestrator_public_model_id = lambda b: None
    inference._openai_catalog_objects = _objects
    inference.load_model_gated = _load
    inference._unload_model_impl = _unload
    inference._state = state
    return inference


def _load_mod(catalog = None, **kw):
    global _MOD
    if _MOD is None or not hasattr(_MOD, "router"):
        sys.modules.pop("llama_compat_owui_under_test", None)
        spec = importlib.util.spec_from_file_location(
            "llama_compat_owui_under_test",
            str(_BACKEND / "routes" / "llama_compat.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llama_compat_owui_under_test"] = mod
        spec.loader.exec_module(mod)
        _MOD = mod
    double = _inference_double(_catalog() if catalog is None else catalog, **kw)
    _MOD._inference = lambda d = double: d
    return _MOD, double


def _client(mod, *, auth = True):
    app = FastAPI()
    app.include_router(mod.router)
    if auth:
        app.dependency_overrides[mod.get_current_subject] = lambda: "test"
    else:
        # Force auth failure without pulling the full credential stack.
        def _deny():
            raise HTTPException(status_code = 401, detail = "Not authenticated")

        app.dependency_overrides[mod.get_current_subject] = _deny
    return TestClient(app, raise_server_exceptions = False)


@pytest.fixture(scope = "module", autouse = True)
def _teardown():
    yield
    sys.modules.pop("llama_compat_owui_under_test", None)


# ── A. Residency / Open WebUI merge regression ───────────────────────────────


def test_nonresident_catalog_row_keeps_loaded_false_after_owui_llama_cpp_merge():
    """THE #10610 REGRESSION: OWUI overwrites loaded=false → true without status."""
    row = {
        "id": "unsloth/Laguna-S-2.1-GGUF",
        "object": "model",
        "loaded": False,
        "status": {"value": "unloaded"},
    }
    # Control: bare loaded without status is wrongly assumed resident.
    broken = {"id": "x", "loaded": False}
    assert get_provider_model_loaded_state(broken, "llama.cpp") is True
    assert get_merged_models([[broken]])["x"]["loaded"] is True

    # Plain OpenAI provider leaves the boolean alone.
    openai_merged = get_merged_models([[row]], provider = "openai")
    assert openai_merged[row["id"]]["loaded"] is False

    merged = get_merged_models([[row]])
    assert merged[row["id"]]["loaded"] is False
    assert merged[row["id"]]["status"]["value"] == "unloaded"


def test_management_catalog_status_survives_owui_merge():
    mod, _ = _load_mod(catalog = _catalog(resident = None))
    with _client(mod) as c:
        data = c.get("/models").json()["data"]
    merged = get_merged_models([data])
    assert merged["unsloth/Laguna-S-2.1-GGUF"]["loaded"] is False
    assert merged["unsloth/Laguna-S-2.1-GGUF"]["status"]["value"] == "unloaded"


# ── B/C/D/E/F. Management routes ─────────────────────────────────────────────


def test_get_models_lists_nonresident_downloaded_models():
    mod, _ = _load_mod()
    with _client(mod) as c:
        r = c.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"]: m for m in body["data"]}
    assert ids["unsloth/Laguna-S-2.1-GGUF"]["loaded"] is False
    assert ids["unsloth/Laguna-S-2.1-GGUF"]["status"]["value"] == "unloaded"
    assert ids["unsloth/Qwen3.8-27B-GGUF"]["loaded"] is True


def test_empty_management_catalog():
    mod, _ = _load_mod(catalog = [])
    with _client(mod) as c:
        r = c.get("/models")
    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": []}


def test_management_requires_auth():
    mod, _ = _load_mod()
    with _client(mod, auth = False) as c:
        assert c.get("/models").status_code == 401
        assert (
            c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"}).status_code == 401
        )
        assert (
            c.post("/models/unload", json = {"model": "unsloth/Qwen3.8-27B-GGUF"}).status_code == 401
        )


def test_load_unload_open_webui_contract_and_residency_transition(monkeypatch):
    catalog = _catalog(resident = None)
    loads = []
    unloads = []

    async def _load(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        loads.append(
            {
                "model_path": request.model_path,
                "gguf_variant": request.gguf_variant,
                "user_initiated": user_initiated,
            }
        )
        return types.SimpleNamespace(status = "loaded", model = request.model_path)

    async def _unload(request, current_subject):
        unloads.append(request.model_path)
        return types.SimpleNamespace(status = "unloaded", model = request.model_path)

    mod, double = _load_mod(catalog = catalog, load_impl = _load, unload_impl = _unload)

    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda requested, **_k: (
            "/data/Laguna.gguf",
            "Q4_K_M",
            "unsloth/Laguna-S-2.1-GGUF",
        ),
    )

    with _client(mod) as c:
        before = {m["id"]: m for m in c.get("/models").json()["data"]}
        assert before["unsloth/Laguna-S-2.1-GGUF"]["loaded"] is False

        load = c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
        assert load.status_code == 200, load.text
        assert load.json()["success"] is True
        assert load.json()["loaded"] is True
        assert loads and loads[0]["user_initiated"] is True
        assert loads[0]["model_path"] == "/data/Laguna.gguf"
        assert loads[0]["gguf_variant"] == "Q4_K_M"

        double._state["resident"].add("unsloth/Laguna-S-2.1-GGUF")
        after_load = {m["id"]: m for m in c.get("/models").json()["data"]}
        assert after_load["unsloth/Laguna-S-2.1-GGUF"]["loaded"] is True

        unload = c.post("/models/unload", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
        assert unload.status_code == 200, unload.text
        assert unload.json()["success"] is True
        assert unload.json()["loaded"] is False
        assert unloads == ["unsloth/Laguna-S-2.1-GGUF"]

        double._state["resident"].discard("unsloth/Laguna-S-2.1-GGUF")
        after_unload = {m["id"]: m for m in c.get("/models").json()["data"]}
        assert after_unload["unsloth/Laguna-S-2.1-GGUF"]["loaded"] is False


def test_load_unknown_model_is_404(monkeypatch):
    mod, _ = _load_mod(catalog = _catalog(resident = None, extra_unloaded = False))
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda *_a, **_k: None,
    )
    with _client(mod) as c:
        r = c.post("/models/load", json = {"model": "nobody/missing-model"})
    assert r.status_code == 404


def test_load_invalid_body_is_4xx():
    mod, _ = _load_mod()
    with _client(mod) as c:
        assert c.post("/models/load", json = {}).status_code == 422
        assert c.post("/models/load", json = {"model": ""}).status_code == 422
        assert c.post("/models/unload", json = {"nope": 1}).status_code == 422


def test_load_failure_surfaces_error_without_claiming_success(monkeypatch):
    async def _boom(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        raise HTTPException(status_code = 500, detail = "Failed to load model")

    mod, double = _load_mod(catalog = _catalog(resident = None), load_impl = _boom)
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda requested, **_k: ("/data/x.gguf", "Q4_K_M", "unsloth/Laguna-S-2.1-GGUF"),
    )
    with _client(mod) as c:
        r = c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
    assert r.status_code == 500
    assert "unsloth/Laguna-S-2.1-GGUF" not in double._state["resident"]


def test_repeated_load_uses_lifecycle_already_loaded(monkeypatch):
    calls = {"n": 0}

    async def _load(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        calls["n"] += 1
        status = "already_loaded" if calls["n"] > 1 else "loaded"
        return types.SimpleNamespace(status = status, model = request.model_path)

    mod, _ = _load_mod(catalog = _catalog(resident = None), load_impl = _load)
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda requested, **_k: ("/data/x.gguf", None, "unsloth/Laguna-S-2.1-GGUF"),
    )
    with _client(mod) as c:
        a = c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
        b = c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"})
    assert a.status_code == 200 and b.status_code == 200
    assert a.json()["status"] == "loaded"
    assert b.json()["status"] == "already_loaded"


def test_models_routes_are_not_under_v1(monkeypatch):
    """Management paths must not be registered only as /v1/... aliases."""
    mod, _ = _load_mod()
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda requested, **_k: ("/data/x.gguf", None, "unsloth/Laguna-S-2.1-GGUF"),
    )
    paths = {getattr(r, "path", None) for r in mod.router.routes}
    assert "/models" in paths
    assert "/models/load" in paths
    assert "/models/unload" in paths
    assert "/v1/models/load" not in paths
    assert "/v1/models/unload" not in paths

    with _client(mod) as c:
        assert c.get("/models").status_code == 200
        assert (
            c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF"}).status_code == 200
        )
        # /v1/models is registered on the inference router, not llama_compat.
        assert c.get("/v1/models").status_code == 404


def test_quant_suffix_resolves_through_existing_resolver(monkeypatch):
    seen = {}

    async def _load(
        request,
        fastapi_request,
        current_subject,
        *,
        user_initiated = False,
    ):
        seen["path"] = request.model_path
        seen["variant"] = request.gguf_variant
        return types.SimpleNamespace(status = "loaded", model = request.model_path)

    catalog = [
        {
            "id": "unsloth/Laguna-S-2.1-GGUF",
            "object": "model",
            "created": 1,
            "owned_by": "unsloth-studio",
            "loaded": False,
            "status": {"value": "unloaded"},
            "quant": "Q4_K_M",
        }
    ]
    mod, _ = _load_mod(catalog = catalog, load_impl = _load)
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_trusted_cached_local_gguf",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "core.inference.local_model_resolver.resolve_local_gguf",
        lambda requested, **_k: (
            "/data/Laguna-Q4_K_M.gguf",
            "Q4_K_M",
            "unsloth/Laguna-S-2.1-GGUF",
        )
        if "Laguna" in requested
        else None,
    )
    with _client(mod) as c:
        r = c.post("/models/load", json = {"model": "unsloth/Laguna-S-2.1-GGUF:Q4_K_M"})
    assert r.status_code == 200
    assert seen["path"] == "/data/Laguna-Q4_K_M.gguf"
    assert seen["variant"] == "Q4_K_M"
