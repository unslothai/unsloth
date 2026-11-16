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


_CONFIG = {"max_seq_length": 98304, "cache_type_kv": "q5_1", "speculative_type": "ngram"}


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


def test_save_without_a_stored_load_config_is_refused(store, monkeypatch):
    # A Web UI tab from before load_config renames a plain alias and PUTs the configured one bare.
    from fastapi import HTTPException

    import routes.inference as inference
    import routes.settings as settings_routes

    async def catalog():
        return []

    monkeypatch.setattr(inference, "_openai_catalog_objects", catalog)

    def save(aliases):
        payload = settings_routes.ModelAliasesPayload(aliases = aliases)
        return asyncio.run(
            settings_routes.update_model_aliases_setting(payload, current_subject = "owner")
        ).model_dump()["aliases"]

    configured = {"name": "long", "model": "org/model", "load_config": _CONFIG}
    stored = save([configured, {"name": "fast", "model": "org/model"}])
    with pytest.raises(HTTPException) as err:
        save([{"name": "long", "model": "org/model"}, {"name": "quick", "model": "org/model"}])
    assert err.value.status_code == 400
    assert "Refresh the page" in err.value.detail
    assert aliases_mod.get_model_aliases() == stored

    # A current client sends the configuration back, or null to remove it on purpose.
    assert save([configured, {"name": "quick", "model": "org/model"}])[0] == configured
    assert save([{"name": "long", "model": "org/model", "load_config": None}]) == [
        {"name": "long", "model": "org/model"}
    ]
    assert save([{"name": "long", "model": "org/other"}]) == [
        {"name": "long", "model": "org/other"}
    ]


def test_configured_alias_roundtrips_settings_schema(store):
    # Saving and returning an alias retains its independently selectable launch configuration.
    from routes.settings import ModelAliasesPayload, ModelAliasesResponse, get_model_aliases_setting

    entries = [{"name": "long", "model": "org/model:Q4_K_M", "load_config": _CONFIG}]
    payload = ModelAliasesPayload(aliases = entries)
    assert set_model_aliases([e.model_dump(exclude_none = True) for e in payload.aliases]) == entries
    assert get_model_aliases_setting().model_dump() == {"aliases": entries}
    assert ModelAliasesResponse(aliases = [{"name": "plain", "model": "org/model"}]).model_dump() == {
        "aliases": [{"name": "plain", "model": "org/model"}]
    }


@pytest.mark.parametrize(
    "config",
    [
        {},
        {**_CONFIG, "max_seq_length": True},
        {**_CONFIG, "max_seq_length": "4096"},
        {**_CONFIG, "max_seq_length": 0},
        {**_CONFIG, "max_seq_length": 1048577},
        {**_CONFIG, "max_seq_length": None},
        {**_CONFIG, "cache_type_kv": "bogus"},
        {**_CONFIG, "speculative_type": "eagle3"},
        {**_CONFIG, "llama_extra_args": ["--port", "9"]},
    ],
)
def test_configured_alias_rejects_invalid_config(config):
    # Invalid or unsupported launch settings cannot be persisted as apparently usable configurations.
    with pytest.raises(ValueError):
        normalize_model_aliases([{"name": "long", "model": "org/model", "load_config": config}])


def test_configured_alias_middleware_binds_trusted_config_to_target(store):
    # Only a configured top-level alias carries server-side metadata for its rewritten target.
    set_model_aliases([{"name": "long", "model": "org/model", "load_config": _CONFIG}])
    seen = _run_middleware(b'{"model":"long","load_config":{"max_seq_length":1}}')
    assert aliases_mod.request_alias_load_config(seen["scope"], "org/model") == _CONFIG
    assert aliases_mod.request_alias_load_config(seen["scope"], "another/model") == {}
    for body in (
        b'{"model":"org/model","load_config":{"max_seq_length":1}}',
        b'{"model":"org/model","nested":{"model":"long"}}',
    ):
        scope = _run_middleware(body)["scope"]
        assert aliases_mod.request_alias_load_config(scope, "org/model") == {}


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


def test_alias_config_compares_requested_launch_intent():
    # Auto speculation is matched against the requested intent rather than its chosen runtime algorithm.
    from types import SimpleNamespace
    from dataclasses import replace
    from core.inference.llama_cpp import GgufLoadIntent

    intent = GgufLoadIntent(
        model_identifier = "org/model", n_ctx = 98304, cache_type_kv = "q5_1", speculative_type = "auto"
    )
    backend = SimpleNamespace(last_load_intent = intent, speculative_type = "mtp")
    config = {**_CONFIG, "speculative_type": "auto"}
    assert aliases_mod.alias_config_matches(backend, config)
    for changed in (
        replace(intent, n_ctx = 4096),
        replace(intent, cache_type_kv = "q4_0"),
        replace(intent, speculative_type = "off"),
        None,
    ):
        backend.last_load_intent = changed
        assert not aliases_mod.alias_config_matches(backend, config)


def test_configured_alias_uses_one_settings_snapshot(monkeypatch):
    # An alias edit during body receipt cannot detach configuration from the rewritten target.
    before = [{"name": "long", "model": "org/before", "load_config": _CONFIG}]
    snapshots = iter([before, [{"name": "long", "model": "org/after"}]])
    monkeypatch.setattr(aliases_mod, "get_model_aliases", lambda: next(snapshots))
    seen = _run_middleware(b'{"model":"long"}')
    assert json.loads(seen["body"])["model"] == "org/before"
    assert aliases_mod.request_alias_load_config(seen["scope"], "org/before") == _CONFIG


@pytest.mark.parametrize(
    "extras",
    [
        ["--ctx-size", "2048"],
        ["--cache-type-k", "q4_0"],
        ["--cache-type-v=q4_0"],
        ["--spec-type", "none"],
    ],
)
def test_alias_config_does_not_trust_scalars_shadowed_by_extra_flags(extras):
    # A manually loaded server's flags can disagree with the scalar intent it retained.
    from types import SimpleNamespace
    from core.inference.llama_cpp import GgufLoadIntent

    intent = GgufLoadIntent(
        model_identifier = "org/model",
        n_ctx = 98304,
        cache_type_kv = "q5_1",
        speculative_type = "ngram",
        extra_args = extras,
    )
    backend = SimpleNamespace(last_load_intent = intent, requested_extra_args = extras)
    assert not aliases_mod.alias_config_matches(backend, _CONFIG)


def test_configured_alias_catalog_reports_only_matching_runtime_loaded(store, monkeypatch):
    # An unloaded configuration cannot advertise another configuration's live context or residency.
    import routes.inference as inf
    from core.inference.llama_cpp import GgufLoadIntent
    from types import SimpleNamespace

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

    backend = SimpleNamespace(
        is_loaded = True,
        last_load_intent = GgufLoadIntent(
            model_identifier = "org/model",
            n_ctx = 98304,
            cache_type_kv = "q5_1",
            speculative_type = "ngram",
        ),
    )
    monkeypatch.setattr(inf, "_openai_catalog_objects", catalog)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inf, "_loaded_satisfies", lambda model: model == "org/model")
    rows = asyncio.run(inf._openai_catalog_with_aliases())
    long = next(row for row in rows if row["id"] == "long")
    short = next(row for row in rows if row["id"] == "short")
    assert long["loaded"] is True and long["context_length"] == 98304
    assert short == {"id": "short", "object": "model", "loaded": False, "alias_for": "org/model"}
