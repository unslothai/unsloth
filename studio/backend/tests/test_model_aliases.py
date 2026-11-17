# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-defined model aliases (#11649): validation, the /v1 body rewrite, and /v1/models rows."""

import asyncio
import json

import pytest

import utils.model_aliases as aliases_mod
from utils.model_aliases import (
    ModelAliasMiddleware,
    alias_model_objects,
    normalize_model_aliases,
    set_model_aliases,
)


@pytest.fixture
def store(monkeypatch):
    import storage.studio_db as db
    import utils.openai_auto_switch_settings as settings

    data: dict = {}
    monkeypatch.setattr(db, "get_app_setting", lambda k, d = None: data.get(k, d))
    monkeypatch.setattr(db, "upsert_app_settings", lambda mapping: data.update(mapping))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: data.get(k, d))
    return data


def test_normalize_accepts_valid_and_strips():
    assert normalize_model_aliases(
        [{"name": " fast-chat ", "model": " unsloth/Qwen3-4B-GGUF:Q4_K_M "}]
    ) == [{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M"}]
    assert normalize_model_aliases(None) == []


@pytest.mark.parametrize(
    "entries",
    [
        [{"name": "org/model", "model": "x"}],
        [{"name": "chat:Q4", "model": "x"}],
        [{"name": "", "model": "x"}],
        [{"name": "Default", "model": "x"}],
        [{"name": "chat", "model": ""}],
        [{"name": "chat", "model": "a"}, {"name": "CHAT", "model": "b"}],
        [{"name": "a", "model": "b"}, {"name": "b", "model": "c"}],
    ],
)
def test_normalize_rejects_bad_entries(entries):
    with pytest.raises(ValueError):
        normalize_model_aliases(entries)


def test_set_is_idempotent_and_a_bad_stored_row_disables_aliases(store):
    entries = [{"name": "code", "model": "unsloth/Qwen3-Coder-GGUF"}]
    assert set_model_aliases(entries) == set_model_aliases(entries) == entries
    assert aliases_mod.resolve_model_alias("CODE") == "unsloth/Qwen3-Coder-GGUF"
    store[aliases_mod.MODEL_ALIASES_SETTING_KEY] = [{"name": "bad/name", "model": "x"}]
    assert aliases_mod.get_model_aliases() == []


def _run_middleware(
    body: bytes,
    headers = None,
    path = "/v1/chat/completions",
    method = "POST",
):
    seen = {}

    async def app(scope, receive, send):
        message = await receive()
        seen["body"] = message["body"]
        seen["headers"] = dict(scope["headers"])
        seen["scope"] = scope

    sent = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return sent.pop(0)

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers
        if headers is not None
        else [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))],
    }
    asyncio.run(ModelAliasMiddleware(app)(scope, receive, None))
    return seen


def test_middleware_rewrites_alias_and_content_length(store):
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M"}])
    body = json.dumps({"model": "Fast-Chat", "messages": [{"role": "user", "content": "hé"}]})
    seen = _run_middleware(body.encode())
    payload = json.loads(seen["body"])
    assert payload["model"] == "unsloth/Qwen3-4B-GGUF:Q4_K_M"
    assert payload["messages"][0]["content"] == "hé"
    assert seen["headers"][b"content-length"] == str(len(seen["body"])).encode()


@pytest.mark.parametrize(
    "body, kwargs",
    [
        (b'{"model": "unsloth/Qwen3-4B-GGUF", "messages": []}', {}),
        (b'{"model": "fast-chat"}', {"path": "/api/settings/x"}),
        (b'{"model": "fast-chat"}', {"path": "/v1/systemone"}),
        (b'{"model": "fast-chat"}', {"path": "/api/inference/audio/stt/load"}),
        (b'{"model": "fast-chat"}', {"path": "/v1/modelsx/chat/completions"}),
        (b'{"model": "fast-chat"}', {"method": "PUT"}),
        (b"not json", {}),
        (b'["model"]', {}),
        (b'{"model": "fast-chat"}', {"headers": [(b"content-type", b"multipart/form-data")]}),
        (b'{"model": "fast-chat"}', {"headers": [(b"content-type", b"audio/wav")]}),
        (b'{"model": "fast-chat"}', {"headers": [(b"content-type", b"application/octet-stream")]}),
        (b'{"model": "fast-chat"}', {"headers": []}),
        (
            b'{"model": "fast-chat"}',
            {"path": "/v1/audio/inputs/"},
        ),
    ],
)
def test_middleware_leaves_other_requests_byte_identical(store, body, kwargs):
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF"}])
    assert _run_middleware(body, **kwargs)["body"] == body


def test_middleware_is_passthrough_without_aliases(store):
    body = b'{"model": "fast-chat"}'
    assert _run_middleware(body)["body"] == body


def test_alias_rows_follow_the_visible_catalog(store):
    set_model_aliases(
        [
            {"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M"},
            {"name": "hidden", "model": "someone/not-on-disk"},
            {"name": "local-model", "model": "unsloth/Qwen3-4B-GGUF"},
            {"name": "dictate", "model": "unsloth/whisper-tiny"},
            {"name": "clip", "model": "Wan-AI/Wan2.2-TI2V-5B"},
        ]
    )
    catalog = [
        {
            "id": "unsloth/Qwen3-4B-GGUF",
            "object": "model",
            "loaded": True,
            "quant": "Q8_0",
            "display_name": "Qwen3 4B",
        },
        {"id": "local-model", "object": "model", "loaded": False},
        {"id": "unsloth/whisper-tiny", "object": "model", "task": "automatic-speech-recognition"},
        {"id": "Wan-AI/Wan2.2-TI2V-5B", "object": "model", "task": "text-to-video"},
    ]

    def pinned(target, objects):
        # Stand-in for routes.inference._pinned_quant_object: only Q4_K_M is on disk.
        if target.lower().endswith(":q4_k_m"):
            return {**objects[0], "id": target, "quant": "Q4_K_M", "loaded": False}
        return None

    local = {
        "id": "local-model",
        "object": "model",
        "loaded": True,
        "alias_for": "unsloth/Qwen3-4B-GGUF",
    }
    # An alias the rewrite applies is listed even over a same-named model row.
    assert alias_model_objects(catalog) == [local]
    rows = alias_model_objects(catalog, pinned)
    assert rows == [
        {
            "id": "fast-chat",
            "object": "model",
            "loaded": False,
            "alias_for": "unsloth/Qwen3-4B-GGUF:Q4_K_M",
        },
        local,
    ]
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q8_0"}])
    assert alias_model_objects(catalog, pinned) == []


def test_catalog_listing_drops_the_model_row_an_alias_shadows(store, monkeypatch):
    import routes.inference as inference

    async def catalog(quant_pins = False):
        return [
            {"id": "unsloth/Qwen3-4B-GGUF", "object": "model", "loaded": False},
            {"id": "local-model", "object": "model", "loaded": False},
        ]

    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)
    set_model_aliases([{"name": "local-model", "model": "unsloth/Qwen3-4B-GGUF"}])
    listed = asyncio.run(inference._openai_catalog_with_aliases())
    assert [(m["id"], m.get("alias_for")) for m in listed] == [
        ("unsloth/Qwen3-4B-GGUF", None),
        ("local-model", "unsloth/Qwen3-4B-GGUF"),
    ]


def test_set_rejects_an_alias_that_reuses_a_real_model_id(store):
    with pytest.raises(ValueError, match = "already the name of a model"):
        set_model_aliases([{"name": "My-Model", "model": "unsloth/Qwen3-4B-GGUF"}], ["my-model"])
    assert store == {}


def test_pydantic_route_sees_the_target_behind_the_body_cap(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from pydantic import BaseModel

    import main

    class Body(BaseModel):
        model: str = "default"
        temperature: float = 1.0

    app = FastAPI()

    @app.post("/v1/chat/completions")
    def echo(body: Body):
        return body.model_dump()

    app.add_middleware(ModelAliasMiddleware)
    app.add_middleware(
        main.MaxBodyMiddleware, max_bytes_getter = lambda: 64, protected_prefixes = ("/v1",)
    )
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF"}])
    client = TestClient(app)
    ok = client.post("/v1/chat/completions", json = {"model": "fast-chat", "temperature": 0.2})
    assert ok.json() == {"model": "unsloth/Qwen3-4B-GGUF", "temperature": 0.2}
    # The cap still runs first: an oversized aliased body is refused, never buffered here.
    assert (
        client.post("/v1/chat/completions", json = {"model": "fast-chat", "x": "y" * 100}).status_code
        == 413
    )


def test_main_app_registers_the_alias_middleware_inside_the_body_cap():
    import main
    classes = [m.cls for m in main.app.user_middleware]
    assert classes.index(main.MaxBodyMiddleware) < classes.index(ModelAliasMiddleware)


def test_retrieve_prefers_an_alias_over_a_same_named_loaded_model(store, monkeypatch):
    import routes.inference as inference

    loaded = [{"id": "local-model", "object": "model"}]

    async def catalog(quant_pins = False):
        return [{"id": "unsloth/Qwen3-4B-GGUF", "object": "model", "loaded": False}, *loaded]

    monkeypatch.setattr(inference, "_openai_model_objects", lambda: loaded)
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)
    set_model_aliases([{"name": "local-model", "model": "unsloth/Qwen3-4B-GGUF"}])
    got = asyncio.run(inference.openai_retrieve_model("local-model", current_subject = "owner"))
    assert got["alias_for"] == "unsloth/Qwen3-4B-GGUF"


def test_list_keeps_quant_pin_rows_next_to_aliases(store, monkeypatch):
    import routes.inference as inference

    async def catalog(quant_pins = False):
        rows = [{"id": "unsloth/Qwen3-4B-GGUF", "object": "model", "loaded": False}]
        if quant_pins:
            rows.append({"id": "unsloth/Qwen3-4B-GGUF:Q4_K_M", "object": "model", "loaded": False})
        return rows

    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M"}])
    listed = asyncio.run(inference.openai_list_models(current_subject = "owner"))["data"]
    assert [(m["id"], m.get("alias_for")) for m in listed] == [
        ("unsloth/Qwen3-4B-GGUF", None),
        ("unsloth/Qwen3-4B-GGUF:Q4_K_M", None),
        ("fast-chat", "unsloth/Qwen3-4B-GGUF:Q4_K_M"),
    ]


def test_keepwarm_sees_the_markers_a_route_sets_on_an_aliased_request(store):
    from core.inference import llama_keepwarm
    from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware, note_admitted_inference

    # Relative to the process-wide counter: tests sharing an xdist worker can leave it raised.
    baseline = llama_keepwarm.other_admitted_inference_count()

    async def route(scope, receive, send):
        await receive()
        note_admitted_inference(scope)
        assert llama_keepwarm.other_admitted_inference_count() == baseline + 1
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    async def send(message):
        pass

    body = b'{"model": "fast-chat", "messages": []}'
    sent = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        return sent.pop(0)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"content-type", b"application/json"), (b"content-length", b"%d" % len(body))],
    }
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF"}])
    asyncio.run(LlamaKeepWarmMiddleware(ModelAliasMiddleware(route))(scope, receive, send))
    assert llama_keepwarm.other_admitted_inference_count() == baseline


def test_retrieve_alias_to_an_unloaded_pin_reports_status(store, monkeypatch):
    import core.inference.local_model_resolver as resolver
    import routes.inference as inference

    async def catalog(quant_pins = False):
        return [
            {
                "id": "unsloth/Qwen3-4B-GGUF",
                "object": "model",
                "loaded": False,
                "status": {"value": "unloaded"},
            }
        ]

    monkeypatch.setattr(inference, "_openai_model_objects", lambda: [])
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)
    monkeypatch.setattr(resolver, "local_gguf_pinned_variant", lambda model_id, **_: "Q4_K_M")
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF:Q4_K_M"}])
    got = asyncio.run(inference.openai_retrieve_model("fast-chat", current_subject = "owner"))
    assert got["alias_for"] == "unsloth/Qwen3-4B-GGUF:Q4_K_M"
    assert got["status"] == {"value": "unloaded"}


@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions/", "/api/inference/chat/completions", "/v1/messages", "/v1/responses"],
)
def test_middleware_rewrites_on_every_chat_route(store, path):
    set_model_aliases([{"name": "fast-chat", "model": "unsloth/Qwen3-4B-GGUF"}])
    seen = _run_middleware(b'{"model": "fast-chat"}', path = path)
    assert json.loads(seen["body"])["model"] == "unsloth/Qwen3-4B-GGUF"


def test_saving_refuses_a_decision_model_id(store, monkeypatch):
    from fastapi import HTTPException

    import routes.inference as inference
    import routes.settings as settings_routes

    async def catalog():
        return [{"id": "unsloth/Qwen3-4B-GGUF", "object": "model"}]

    # With the Decision API off nothing lists clef, and it must still be refused.
    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)
    payload = settings_routes.ModelAliasesPayload(
        aliases = [{"name": "clef", "model": "unsloth/Qwen3-4B-GGUF"}]
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(settings_routes.update_model_aliases_setting(payload, current_subject = "owner"))
    assert err.value.status_code == 400


_CONFIG = {"max_seq_length": 98304, "kv_cache_dtype": "q5_1", "speculative_type": "ngram"}


@pytest.mark.parametrize(
    "path",
    [
        "/chat/completions",
        "/completions",
        "/chat/count_tokens",
        "/embeddings",
        "/messages",
        "/messages/count_tokens",
        "/responses",
    ],
)
def test_configured_connection_is_rejected_before_any_endpoint_dispatch(store, monkeypatch, path):
    # Every alias-capable endpoint must reject unsupported configurations before its own shortcuts.
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from routes import inference
    from utils.api_errors import install_api_error_handlers

    set_model_aliases([{"name": "remote", "model": "external::c1::model", "load_config": _CONFIG}])
    monkeypatch.setattr(
        inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    app = FastAPI()
    app.include_router(inference.router, prefix = "/v1")
    app.add_middleware(ModelAliasMiddleware)
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    install_api_error_handlers(app)
    with TestClient(app) as client:
        response = client.post(
            "/v1" + path,
            json = {
                "model": "remote",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 8,
                "prompt": "hello",
                "input": "hello",
            },
        )
    assert response.status_code == 400, response.text
    assert "error" in response.json()


@pytest.fixture
def settings_api(store, monkeypatch):
    """GET and PUT /api/settings/model-aliases as a client calls them."""
    import routes.inference as inference
    import routes.settings as settings_routes

    async def catalog():
        return []

    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)

    def get():
        return settings_routes.get_model_aliases_setting().model_dump()

    def put(aliases, etag = None):
        payload = settings_routes.ModelAliasesPayload(aliases = aliases, etag = etag)
        return asyncio.run(
            settings_routes.update_model_aliases_setting(payload, current_subject = "owner")
        ).model_dump()

    return SimpleNamespace(get = get, put = put)


from types import SimpleNamespace  # noqa: E402

_LONG = {"name": "long", "model": "org/model", "load_config": _CONFIG}


def _refused(call, status):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        call()
    assert err.value.status_code == status
    return err.value.detail


def test_a_client_without_load_config_keeps_configurations_it_cannot_see(settings_api, store):
    # An older Web UI tab renames a plain alias and sends the configured one back without the field.
    settings_api.put([_LONG, {"name": "fast", "model": "org/model"}])
    saved = settings_api.put(
        [{"name": "long", "model": "org/model"}, {"name": "quick", "model": "org/model"}]
    )
    assert saved["aliases"] == [_LONG, {"name": "quick", "model": "org/model"}]


@pytest.mark.parametrize(
    "aliases",
    [
        # Renamed (a case-only rename changes the public id too), retargeted, or removed.
        [{"name": "longer", "model": "org/model"}],
        [{"name": "Long", "model": "org/model"}],
        [{"name": "long", "model": "org/other"}],
        [],
    ],
)
def test_a_client_without_load_config_cannot_drop_a_configuration(settings_api, store, aliases):
    settings_api.put([_LONG])
    before = json.loads(json.dumps(store))
    detail = _refused(lambda: settings_api.put(aliases), 409)
    assert "Reload the page" in detail
    assert store == before


def test_a_client_that_sends_load_config_may_change_or_clear_it(settings_api):
    settings_api.put([_LONG])
    fast = {"max_seq_length": 4096, "kv_cache_dtype": "q4_0"}
    assert settings_api.put([{"name": "long", "model": "org/other", "load_config": fast}])[
        "aliases"
    ] == [{"name": "long", "model": "org/other", "load_config": fast}]
    assert settings_api.put([{"name": "long", "model": "org/model", "load_config": None}])[
        "aliases"
    ] == [{"name": "long", "model": "org/model"}]


def test_a_current_etag_replaces_the_list_and_a_stale_one_is_refused(settings_api, store):
    first = settings_api.put([_LONG])
    assert first["etag"] == settings_api.get()["etag"]
    # Full replacement: a client that read the current list may rename or remove a configured alias.
    second = settings_api.put([{"name": "longer", "model": "org/model"}], etag = first["etag"])
    assert second["aliases"] == [{"name": "longer", "model": "org/model"}]
    assert second["etag"] != first["etag"]
    before = json.loads(json.dumps(store))
    # A second tab that read the first list is refused even for a plain-only edit.
    _refused(
        lambda: settings_api.put([{"name": "x", "model": "org/model"}], etag = first["etag"]), 409
    )
    assert store == before


@pytest.mark.parametrize("versioned", [False, True], ids = ["legacy", "etag"])
def test_alias_save_keeps_newer_presets_after_an_overlapping_read(monkeypatch, versioned):
    # A late read may repopulate the cache, but a stale save must not erase newer presets.
    from concurrent.futures import ThreadPoolExecutor
    import threading

    import storage.studio_db as db
    import utils.openai_auto_switch_settings as settings

    key = aliases_mod.MODEL_ALIASES_SETTING_KEY
    original = [{"name": "plain", "model": "org/model"}]
    db.upsert_app_settings({key: original})
    settings._invalidate(key)
    old_etag = aliases_mod.model_aliases_etag(original)
    read_started = threading.Event()
    finish_read = threading.Event()
    read_setting = db.get_app_setting

    def delayed_read(key, fallback = None):
        value = read_setting(key, fallback)
        if threading.current_thread().name.startswith("alias-cache-reader"):
            read_started.set()
            assert finish_read.wait(5)
        return value

    monkeypatch.setattr(db, "get_app_setting", delayed_read)
    with ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "alias-cache-reader") as pool:
        pending_read = pool.submit(aliases_mod.get_model_aliases)
        try:
            assert read_started.wait(5)
            set_model_aliases([_LONG], etag = old_etag)
        finally:
            finish_read.set()
        assert pending_read.result(timeout = 5) == original

    with pytest.raises(aliases_mod.ModelAliasConflict):
        set_model_aliases(original, etag = old_etag if versioned else None)
    assert db.get_app_setting(key) == [_LONG]


def test_settings_round_trip_returns_what_was_stored(settings_api):
    saved = settings_api.put([_LONG, {"name": "plain", "model": "org/model"}])
    assert saved["aliases"] == [_LONG, {"name": "plain", "model": "org/model"}]
    assert settings_api.get() == saved


@pytest.mark.parametrize(
    "config",
    [
        {**_CONFIG, "max_seq_length": True},
        {**_CONFIG, "max_seq_length": "4096"},
        {**_CONFIG, "max_seq_length": -1},
        {**_CONFIG, "max_seq_length": 1048577},
        {**_CONFIG, "kv_cache_dtype": "bogus"},
        {**_CONFIG, "speculative_type": "eagle3"},
        {**_CONFIG, "speculative_type": "bogus"},
        {"cache_type_kv": "q4_0"},
        {**_CONFIG, "llama_extra_args": ["--port", "9"]},
        {"n_parallel": 0},
        {"n_parallel": 65},
        {"gpu_layers": -1},
        {"gpu_layers": 2**31},
        {"gpu_layers": "all"},
        {"n_cpu_moe": 2},
        {"n_cpu_moe": 2, "gpu_layers": "auto"},
        {"disable_vision": "yes"},
        "q4_0",
    ],
)
def test_configured_alias_rejects_invalid_config(config):
    # Invalid or unsupported launch settings are refused, never stored as if usable.
    with pytest.raises(ValueError):
        normalize_model_aliases([{"name": "long", "model": "org/model", "load_config": config}])


@pytest.mark.parametrize(
    "config, stored",
    [
        (
            {"max_seq_length": 0, "kv_cache_dtype": " Q5_1 ", "speculative_type": "draft-mtp"},
            {"max_seq_length": 0, "kv_cache_dtype": "q5_1", "speculative_type": "mtp"},
        ),
        ({"speculative_type": "none"}, {"speculative_type": "off"}),
        (
            {"gpu_layers": 20, "n_cpu_moe": 0, "n_parallel": 2, "disable_vision": False},
            {"gpu_layers": 20, "n_cpu_moe": 0, "n_parallel": 2, "disable_vision": False},
        ),
        ({"gpu_layers": "auto", "max_seq_length": None}, {"gpu_layers": "auto"}),
        ({}, None),
    ],
)
def test_configured_alias_stores_canonical_settings(config, stored):
    (alias,) = normalize_model_aliases([{"name": "a", "model": "org/model", "load_config": config}])
    assert alias.get("load_config") == stored


_FUTURE = {"max_seq_length": 4096, "rope_scale": 2}


def test_settings_from_a_newer_release_disable_only_their_alias(store, settings_api):
    # A downgrade must not turn every alias off because one entry holds a field it cannot read.
    store[aliases_mod.MODEL_ALIASES_SETTING_KEY] = [
        {"name": "future", "model": "org/model", "load_config": _FUTURE},
        _LONG,
        {"name": "plain", "model": "org/model"},
    ]
    assert aliases_mod.get_model_aliases() == [
        {"name": "future", "model": "org/model", "load_config_unsupported": True},
        _LONG,
        {"name": "plain", "model": "org/model"},
    ]
    assert [row["id"] for row in alias_model_objects([{"id": "org/model"}])] == ["long", "plain"]
    scope = _run_middleware(b'{"model":"future"}')["scope"]
    assert aliases_mod.request_alias_entry(scope, "org/model")["unsupported"] is True
    # The settings API shows the entry as stored, and the same entry may be saved back.
    listed = settings_api.get()
    assert listed["aliases"][0] == {"name": "future", "model": "org/model", "load_config": _FUTURE}
    saved = settings_api.put(listed["aliases"], etag = listed["etag"])
    assert saved["aliases"] == listed["aliases"]
    # Changing it means writing settings this release understands.
    changed = {"name": "future", "model": "org/model", "load_config": {**_FUTURE, "rope_scale": 3}}
    _refused(lambda: settings_api.put([changed, *listed["aliases"][1:]]), 400)


def test_configured_alias_middleware_binds_trusted_config_to_target(store):
    # Only a configured top-level alias carries server-side metadata for its rewritten target.
    set_model_aliases([{"name": "long", "model": "org/model", "load_config": _CONFIG}])
    seen = _run_middleware(b'{"model":"long","load_config":{"max_seq_length":1}}')
    entry = aliases_mod.request_alias_entry(seen["scope"], "org/model")
    assert entry == {"model": "org/model", "load_config": _CONFIG, "unsupported": False}
    assert aliases_mod.request_alias_entry(seen["scope"], "another/model") is None
    for body in (
        b'{"model":"org/model","load_config":{"max_seq_length":1}}',
        b'{"model":"org/model","nested":{"model":"long"}}',
    ):
        scope = _run_middleware(body)["scope"]
        assert aliases_mod.request_alias_entry(scope, "org/model") is None


def test_configured_alias_uses_one_settings_snapshot(monkeypatch):
    # An alias edit during body receipt cannot detach configuration from the rewritten target.
    before = [{"name": "long", "model": "org/before", "load_config": _CONFIG}]
    snapshots = iter([before, [{"name": "long", "model": "org/after"}]])
    monkeypatch.setattr(aliases_mod, "get_model_aliases", lambda: next(snapshots))
    seen = _run_middleware(b'{"model":"long"}')
    assert json.loads(seen["body"])["model"] == "org/before"
    assert aliases_mod.request_alias_entry(seen["scope"], "org/before")["load_config"] == _CONFIG


def test_configured_alias_catalog_follows_visible_target(store):
    # Different configurations remain separate choices without advertising hidden model targets.
    set_model_aliases(
        [
            {"name": "long", "model": "org/model", "load_config": _CONFIG},
            {
                "name": "fast",
                "model": "org/model",
                "load_config": {**_CONFIG, "max_seq_length": 53248},
            },
            {"name": "hidden", "model": "private/model", "load_config": _CONFIG},
        ]
    )
    rows = alias_model_objects([{"id": "org/model", "object": "model"}])
    assert [row["id"] for row in rows] == ["long", "fast"]
    assert all(row["alias_for"] == "org/model" for row in rows)


def test_configured_alias_catalog_reports_only_a_satisfied_runtime_loaded(store, monkeypatch):
    # An unloaded configuration cannot advertise another configuration's live context or residency.
    import routes.inference as inf

    set_model_aliases(
        [
            {"name": "long", "model": "org/model", "load_config": _CONFIG},
            {
                "name": "short",
                "model": "org/model",
                "load_config": {**_CONFIG, "max_seq_length": 2048},
            },
        ]
    )

    async def catalog(**kwargs):
        return [
            {
                "id": "org/model",
                "object": "model",
                "loaded": True,
                "context_length": 98304,
                "max_model_len": 98304,
                "residency": {"loaded": True},
            }
        ]

    seen_requests = []

    async def satisfies(model, config, fastapi_request):
        seen_requests.append(fastapi_request)
        return config["max_seq_length"] == 98304

    monkeypatch.setattr(inf, "_openai_catalog_objects", catalog)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = True))
    monkeypatch.setattr(inf, "_configured_alias_resident_satisfies", satisfies)
    marker = object()
    rows = asyncio.run(inf._openai_catalog_with_aliases(fastapi_request = marker))
    long = next(row for row in rows if row["id"] == "long")
    short = next(row for row in rows if row["id"] == "short")
    assert long["loaded"] is True and long["context_length"] == 98304
    assert short == {"id": "short", "object": "model", "loaded": False, "alias_for": "org/model"}
    # The server's slot default comes from the request, as it does for a chat request.
    assert seen_requests == [marker, marker]


# The reuse check below runs the loader's own comparison against a real backend.


def _resident(**state):
    """A llama-server resident as the loader records it after launching ``state``."""
    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    class _Process:
        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

        def kill(self):
            pass

        def poll(self):
            return 0

    backend = LlamaCppBackend()
    backend._process = _Process()
    backend._healthy = True
    backend._model_identifier = "org/model"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 98304
    backend._requested_n_parallel = 1
    backend._requested_cache_types = ("q5_1", "q5_1")
    backend._requested_spec_mode = "ngram"
    backend._requested_extra_args = []
    backend._extra_args = []
    backend._reasoning_budget = -1
    backend._reasoning_budget_message = ""
    backend._chat_template_override = None
    for key, value in state.items():
        setattr(backend, key, value)
    backend._last_load_intent = GgufLoadIntent(
        model_identifier = "org/model",
        hf_variant = "Q4_K_M",
        n_ctx = backend._requested_n_ctx,
        speculative_type = backend._requested_spec_mode,
        extra_args = tuple(backend._requested_extra_args),
    )
    return backend


def _satisfied(
    monkeypatch,
    backend,
    config,
    *,
    saved = None,
    fastapi_request = None,
):
    import routes.inference as inf
    import utils.openai_auto_switch_settings as settings

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        settings, "get_model_overrides", lambda: {"org/model:Q4_K_M": saved} if saved else {}
    )
    override, kwargs = inf._alias_load_kwargs("org/model", "Q4_K_M", "org/model", config)
    return asyncio.run(inf._alias_launch_satisfied(kwargs, override, fastapi_request))


@pytest.mark.parametrize(
    "config, satisfied",
    [
        (_CONFIG, True),
        ({**_CONFIG, "max_seq_length": 4096}, False),
        ({**_CONFIG, "kv_cache_dtype": "q4_0"}, False),
        ({**_CONFIG, "speculative_type": "off"}, False),
        ({**_CONFIG, "n_parallel": 2}, False),
        ({**_CONFIG, "disable_vision": True}, False),
        ({**_CONFIG, "gpu_layers": 10}, False),
    ],
)
def test_the_reuse_check_compares_every_launch_setting(monkeypatch, config, satisfied):
    assert _satisfied(monkeypatch, _resident(), config) is satisfied


@pytest.mark.parametrize("config", [{"n_parallel": 8}, {"disable_vision": True}])
def test_configured_alias_cannot_reuse_a_diffusion_runner(monkeypatch, config):
    backend = _resident(
        _is_diffusion = True, _requested_cache_types = ("f16", "f16"), _requested_spec_mode = "off"
    )
    backend._last_load_intent = None
    assert not _satisfied(
        monkeypatch,
        backend,
        {"max_seq_length": 98304, "kv_cache_dtype": "f16", "speculative_type": "off", **config},
    )


def test_diffusion_catalog_does_not_advertise_a_configured_alias_loaded(store, monkeypatch):
    import routes.inference as inf
    from core.inference import local_model_resolver as resolver
    from utils import openai_auto_switch_settings as settings

    set_model_aliases(
        [
            {
                "name": "diffusion",
                "model": "org/model",
                "load_config": {"max_seq_length": 98304, "n_parallel": 8},
            }
        ]
    )
    backend = _resident(
        _is_diffusion = True, _requested_cache_types = ("f16", "f16"), _requested_spec_mode = "auto"
    )
    backend._last_load_intent = None

    async def catalog(**kwargs):
        return [{"id": "org/model", "object": "model", "loaded": True, "context_length": 98304}]

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_model_overrides", lambda: {})
    monkeypatch.setattr(inf, "_openai_catalog_objects", catalog)
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(
        resolver,
        "resolve_local_gguf_for_switch",
        lambda *a, **k: ("org/model", "Q4_K_M", "org/model"),
    )
    rows = asyncio.run(inf._openai_catalog_with_aliases())
    assert next(row for row in rows if row["id"] == "diffusion") == {
        "id": "diffusion",
        "object": "model",
        "loaded": False,
        "alias_for": "org/model",
    }


@pytest.mark.parametrize(
    "requested, resolved_variant, satisfied",
    [
        ("org/model", "Q4_K_M", True),
        ("org/model:latest", "Q4_K_M", True),
        ("org/model:Q8_0", "Q8_0", True),
        ("org/model:Q4_K_M", "Q4_K_M", False),
    ],
)
def test_alias_reuse_respects_the_requested_quant_pin(
    monkeypatch, requested, resolved_variant, satisfied
):
    # A bare alias keeps the resident quant; an explicit pin still requires its own weights.
    from dataclasses import replace
    import routes.inference as inf
    from core.inference import local_model_resolver as resolver
    from utils import openai_auto_switch_settings as settings

    backend = _resident(_hf_variant = "Q8_0", _requested_n_parallel = 2)
    backend._last_load_intent = replace(backend.last_load_intent, hf_variant = "Q8_0", n_parallel = 2)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        settings, "get_model_overrides", lambda: {"org/model:Q8_0": {"n_parallel": 2}}
    )
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(
        resolver,
        "resolve_local_gguf_for_switch",
        lambda *a, **k: ("org/model", resolved_variant, "org/model"),
    )
    assert (
        asyncio.run(inf._configured_alias_resident_satisfies(requested, _CONFIG, None)) is satisfied
    )


@pytest.mark.parametrize(
    "requested, satisfied",
    [
        ("/models/Group", True),
        ("/models/Group:Q4_K_M", True),
        ("/models/Group:Q8_0", False),
        ("/models/Other", False),
    ],
)
def test_alias_reuse_matches_a_resident_loaded_by_file_path(monkeypatch, requested, satisfied):
    # A file loaded by path shares no spelling with the indexed directory an alias names.
    import routes.inference as inf
    from core.inference import local_model_resolver as resolver
    from core.inference.llama_cpp import GgufLoadIntent
    from utils import openai_auto_switch_settings as settings

    gguf = "/models/Group/model-Q4_K_M.gguf"
    backend = _resident(_model_identifier = gguf, _gguf_path = gguf)
    backend._last_load_intent = GgufLoadIntent(
        model_identifier = gguf,
        gguf_path = gguf,
        hf_variant = "Q4_K_M",
        n_ctx = backend._requested_n_ctx,
        speculative_type = backend._requested_spec_mode,
    )
    base = requested.split(":")[0]
    quant = requested.split(":")[1] if ":" in requested else "Q4_K_M"
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_model_overrides", lambda: {})
    monkeypatch.setattr(inf, "_innermost_indexed_owner", lambda path: "/models/Group")
    monkeypatch.setattr(resolver, "resolve_trusted_cached_local_gguf", lambda *a, **k: None)
    monkeypatch.setattr(
        resolver, "resolve_local_gguf_for_switch", lambda *a, **k: (base, quant, base)
    )
    assert inf._loaded_satisfies(requested) is False
    assert (
        asyncio.run(inf._configured_alias_resident_satisfies(requested, _CONFIG, None)) is satisfied
    )


def test_auto_speculation_matches_whatever_algorithm_auto_picked(monkeypatch):
    # The loader records the requested mode; Auto running MTP is still Auto.
    backend = _resident(_requested_spec_mode = "auto", _speculative_type = "draft-mtp")
    assert _satisfied(monkeypatch, backend, {**_CONFIG, "speculative_type": "auto"})
    assert not _satisfied(monkeypatch, backend, {**_CONFIG, "speculative_type": "mtp"})


def test_a_default_cache_runtime_satisfies_an_f16_alias(monkeypatch):
    backend = _resident(_requested_cache_types = ("f16", "f16"))
    assert _satisfied(monkeypatch, backend, {**_CONFIG, "kv_cache_dtype": "f16"})


@pytest.mark.parametrize(
    "recorded, selected, satisfied",
    [
        (("f16", "f16"), "f16", True),
        (("q4_0", "q4_0"), "q4_0", True),
        (("q4_0", "f16"), "f16", False),
        (("q4_0", "q4_0"), "f16", False),
    ],
)
def test_a_later_cache_environment_does_not_change_the_resident_pair(
    monkeypatch, recorded, selected, satisfied
):
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "f32")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "f32")
    backend = _resident(_requested_cache_types = recorded)
    config = {**_CONFIG, "kv_cache_dtype": selected}
    assert _satisfied(monkeypatch, backend, config) is satisfied


def test_omitted_alias_fields_compare_against_the_saved_settings(monkeypatch):
    # The resident runs the model's saved 4 slots; an alias that leaves slots alone inherits them.
    saved = {"n_parallel": 4}
    backend = _resident(_requested_n_parallel = 4)
    assert _satisfied(monkeypatch, backend, _CONFIG, saved = saved)
    assert not _satisfied(monkeypatch, backend, _CONFIG)


def test_the_server_slot_default_comes_from_the_request(monkeypatch):
    backend = _resident(_requested_n_parallel = 4)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 4)))
    assert _satisfied(monkeypatch, backend, _CONFIG, fastapi_request = request)


def test_hand_launched_flags_are_not_the_alias_launch(monkeypatch):
    # Even flags that agree on the cache pair: the alias launches with its own (none), as /load would.
    flags = ["--cache-type-k", "q5_1", "--cache-type-v", "q5_1"]
    backend = _resident(_requested_extra_args = list(flags), _extra_args = list(flags))
    assert not _satisfied(monkeypatch, backend, _CONFIG)
    saved = {"llama_extra_args": ["--threads", "2"]}
    backend = _resident(_requested_extra_args = ["--threads", "2"], _extra_args = ["--threads", "2"])
    assert _satisfied(monkeypatch, backend, _CONFIG, saved = saved)


def test_a_bundled_template_compares_as_the_loader_resolves_it(monkeypatch):
    import core.inference.chat_templates as templates

    monkeypatch.setattr(
        templates,
        "resolve_effective_chat_template_override",
        lambda *, model_identifier, user_override: user_override or "{{ bundled }}",
    )
    backend = _resident(_chat_template_override = "{{ bundled }}")
    assert _satisfied(monkeypatch, backend, _CONFIG)


def test_a_stale_saved_gpu_pin_is_dropped_before_comparing(monkeypatch):
    import routes.inference as inf

    async def unresolved(*args):
        return False

    monkeypatch.setattr(inf, "_override_gpu_ids_still_resolve", unresolved)
    assert _satisfied(monkeypatch, _resident(), _CONFIG, saved = {"gpu_ids": [7]})


def test_alias_settings_replace_the_saved_flags_they_shadow(monkeypatch):
    # Even values normalization drops (Auto context, vision on) must remove the saved flag for them.
    import routes.inference as inf
    import utils.openai_auto_switch_settings as settings

    saved = {
        "max_seq_length": 4096,
        "llama_extra_args": ["-ngl", "30", "-c", "8192", "--no-mmproj", "--threads", "2"],
    }
    monkeypatch.setattr(settings, "get_model_overrides", lambda: {"org/model:Q4_K_M": saved})
    config = {"gpu_layers": 20, "max_seq_length": 0, "disable_vision": False}
    _override, kwargs = inf._alias_load_kwargs("org/model", "Q4_K_M", "org/model", config)
    assert kwargs["llama_extra_args"] == ["--threads", "2"]
    assert (kwargs["gpu_memory_mode"], kwargs["gpu_layers"]) == ("manual", 20)
    assert kwargs.get("max_seq_length", 0) == 0
    # A -c equal to the alias context is the user's opt-in to exceed the fit estimate: it stays.
    saved["llama_extra_args"] = ["-c", "98304"]
    _override, kwargs = inf._alias_load_kwargs("org/model", "Q4_K_M", "org/model", _CONFIG)
    assert kwargs["llama_extra_args"] == ["-c", "98304"]


def test_the_reuse_check_reads_manual_offload_flags_as_the_loader_does(monkeypatch):
    # Under Manual GPU memory the loader turns a saved -ngl into gpu_layers before comparing.
    backend = _resident(_gpu_memory_mode = "manual", _gpu_layers = 30, _n_cpu_moe = 0)
    saved = {"gpu_memory_mode": "manual", "llama_extra_args": ["-ngl", "30"]}
    assert _satisfied(monkeypatch, backend, _CONFIG, saved = saved)
    saved = {"gpu_memory_mode": "manual", "llama_extra_args": ["-ngl", "12"]}
    assert not _satisfied(monkeypatch, backend, _CONFIG, saved = saved)


def test_a_stored_config_of_any_shape_reads_back(store, settings_api):
    store[aliases_mod.MODEL_ALIASES_SETTING_KEY] = [
        {"name": "future", "model": "org/model", "load_config": ["preset"]}
    ]
    listed = settings_api.get()
    assert listed["aliases"] == [
        {"name": "future", "model": "org/model", "load_config": ["preset"]}
    ]
    assert aliases_mod.get_model_aliases() == [
        {"name": "future", "model": "org/model", "load_config_unsupported": True}
    ]


def test_the_reuse_check_records_nothing(monkeypatch):
    backend = _resident()
    intent = backend.last_load_intent
    assert _satisfied(monkeypatch, backend, _CONFIG)
    assert backend.last_load_intent is intent
