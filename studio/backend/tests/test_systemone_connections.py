# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import external_provider as ep
from core.systemone import catalog, laya_runtime
from routes import systemone
from storage import credential_secrets, providers_db
from utils import systemone_settings
from utils.account_context import AccountContext, run_as

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": {"ask": "Reply within the hour?"}},
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"outage": {"signals": ["down"]}, "billing": "charges"},
    },
}
UPSTREAM = {
    "model": "jev-1.13",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.93},
        "team": {"type": "choice", "choice": "outage", "confidence": 0.8},
    },
    "usage": {"input_tokens": 50, "output_tokens": 0, "cost_usd": 0.0001},
}
OPENROUTER_MODELS = {
    "data": [
        {
            "id": "upstage/solar-decide",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]},
        },
        {
            "id": "typesafe/jev-1.13",
            "architecture": {"input_modalities": ["text"], "output_modalities": ["decisions"]},
        },
        {
            "id": "openai/gpt-4o",
            "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
        },
    ]
}


@pytest.fixture(autouse = True)
def studio(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_MODEL", raising = False)
    monkeypatch.delenv("UNSLOTH_SYSTEMONE_DISABLE", raising = False)
    for module in (credential_secrets, providers_db):
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    import storage.studio_db as studio_db

    settings = {systemone_settings.ENABLED_KEY: True}
    monkeypatch.setattr(systemone_settings, "_owner_setting", settings.get)
    monkeypatch.setattr(studio_db, "upsert_app_settings", settings.update)
    monkeypatch.setattr(
        systemone_settings, "runtime_unavailable_reason", lambda: "The Decision API needs PyTorch."
    )

    def no_laya(*args):
        raise AssertionError("a connection must not load Laya")

    monkeypatch.setattr(laya_runtime, "decide", no_laya)
    monkeypatch.setattr(catalog, "LISTED_DECISION_MODELS", {})
    return settings


@pytest.fixture
def upstream(monkeypatch):
    calls = []
    replies = [httpx.Response(200, json = UPSTREAM)]

    def handle(request):
        calls.append(request)
        return replies[0]

    monkeypatch.setattr(
        ep, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handle))
    )
    return SimpleNamespace(calls = calls, replies = replies)


def _connection(
    provider_type = "typesafe",
    api_type = "chat_completions",
    base_url = "https://api.typesafe.ai/v1",
):
    providers_db.create_provider(
        id = "deciders",
        provider_type = provider_type,
        display_name = "TypeSafe",
        base_url = base_url,
        models = ["jev-latest"],
        api_type = api_type,
    )
    credential_secrets.save_provider_api_key("deciders", "ts-key")
    return "connection:deciders:jev-latest"


def _client():
    from routes.settings import router as settings_router

    app = FastAPI()
    app.include_router(systemone.router, prefix = "/v1")
    app.include_router(settings_router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app)


def _post(client, model = "jev-latest"):
    return client.post(
        "/v1/systemone", json = {"model": model, "state": "The site is down.", "questions": QUESTIONS}
    )


def test_decision_api_answers_through_a_saved_connection(upstream):
    client = _client()
    name = _connection()
    assert client.get("/api/settings/systemone/connections").json() == [
        {"name": name, "provider_id": "deciders", "provider": "TypeSafe", "model": "jev-latest"}
    ]
    saved = client.put("/api/settings/systemone", json = {"model": name})
    assert saved.status_code == 200, saved.text
    assert saved.json()["model"] == name

    for model in ("default", name):
        response = _post(client, model)
        assert response.status_code == 200, response.text
        assert response.json() == UPSTREAM

    request = upstream.calls[-1]
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer ts-key"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == "The site is down."
    assert body["questions"]["urgent"] == {
        "type": "noul",
        "instructions": json.dumps({"ask": "Reply within the hour?"}),
    }
    assert body["questions"]["team"]["criteria"] == {
        "outage": json.dumps({"signals": ["down"]}),
        "billing": "charges",
    }


def test_a_connection_can_be_enabled_without_local_torch(upstream, studio):
    client = _client()
    studio[systemone_settings.ENABLED_KEY] = False
    refused = client.put("/api/settings/systemone", json = {"enabled": True})
    assert refused.status_code == 400
    name = _connection()
    enabled = client.put("/api/settings/systemone", json = {"enabled": True, "model": name})
    assert enabled.status_code == 200, enabled.text
    assert client.get("/api/settings/systemone/resolve").json()["cached"] is True
    other = "connection:deciders:jev-1.13"
    plan = client.get("/api/settings/systemone/resolve", params = {"model": other})
    assert plan.status_code == 200, plan.text
    assert plan.json()["cached"] is True


def test_a_remote_decision_api_cannot_switch_to_a_local_model_without_torch(studio):
    client = _client()
    studio[systemone_settings.ENABLED_KEY] = False
    name = _connection()
    assert (
        client.put("/api/settings/systemone", json = {"enabled": True, "model": name}).status_code
        == 200
    )
    for route in ("validate", ""):
        refused = client.request(
            "POST" if route else "PUT",
            f"/api/settings/systemone/{route}".rstrip("/"),
            json = {"model": "laya-english"},
        )
        assert refused.status_code == 400
        assert refused.json()["detail"] == "The Decision API needs PyTorch."
    assert client.get("/api/settings/systemone").json()["model"] == name
    off = client.put("/api/settings/systemone", json = {"enabled": False, "model": "laya-english"})
    assert off.status_code == 200, off.text


def test_a_custom_system_one_connection_is_called_at_its_base_url(upstream):
    client = _client()
    name = _connection("custom", "systemone", "http://decider.example/v1/systemone")
    assert client.put("/api/settings/systemone", json = {"model": name}).status_code == 200
    assert _post(client).status_code == 200
    assert str(upstream.calls[-1].url) == "http://decider.example/v1/systemone"


def test_openrouter_offers_its_decision_models(upstream):
    upstream.replies[0] = httpx.Response(200, json = OPENROUTER_MODELS)
    providers_db.create_provider(
        id = "router",
        provider_type = "openrouter",
        display_name = "OpenRouter",
        base_url = "https://openrouter.ai/api/v1",
        models = ["openai/gpt-4o"],
    )
    credential_secrets.save_provider_api_key("router", "or-key")
    client = _client()
    listed = client.get("/api/settings/systemone/connections").json()
    assert [option["model"] for option in listed] == ["upstage/solar-decide", "typesafe/jev-1.13"]
    request = upstream.calls[0]
    assert request.url.params["output_modalities"] == "decisions"
    assert request.headers["authorization"] == "Bearer or-key"

    client.get("/api/settings/systemone/connections")
    assert len(upstream.calls) == 1
    saved = client.put("/api/settings/systemone", json = {"model": listed[1]["name"]})
    assert saved.status_code == 200, saved.text
    upstream.replies[0] = httpx.Response(200, json = UPSTREAM)
    assert _post(client).status_code == 200
    assert json.loads(upstream.calls[-1].content)["model"] == "typesafe/jev-1.13"
    assert str(upstream.calls[-1].url) == "https://openrouter.ai/api/v1/systemone"


def test_an_openrouter_decision_model_saves_without_a_listed_copy(upstream, monkeypatch):
    upstream.replies[0] = httpx.Response(200, json = OPENROUTER_MODELS)
    providers_db.create_provider(
        id = "router",
        provider_type = "openrouter",
        display_name = "OpenRouter",
        base_url = "https://openrouter.ai/api/v1",
        models = ["openai/gpt-4o"],
    )
    credential_secrets.save_provider_api_key("router", "or-key")
    client = _client()
    name = "connection:router:typesafe/jev-1.13"
    assert client.post("/api/settings/systemone/validate", json = {"model": name}).status_code == 204
    assert client.put("/api/settings/systemone", json = {"model": name}).status_code == 200

    monkeypatch.setattr(catalog, "LISTED_DECISION_MODELS", {})
    upstream.replies[0] = httpx.Response(503, json = {"error": {"message": "down"}})
    client.get("/api/settings/systemone/connections")
    upstream.replies[0] = httpx.Response(200, json = OPENROUTER_MODELS)
    assert client.put("/api/settings/systemone", json = {"model": name}).status_code == 200
    chat = client.put("/api/settings/systemone", json = {"model": "connection:router:openai/gpt-4o"})
    assert chat.status_code == 400


def test_an_unreachable_openrouter_lists_no_decision_models(upstream):
    upstream.replies[0] = httpx.Response(401, json = {"error": {"message": "No auth"}})
    providers_db.create_provider(
        id = "router",
        provider_type = "openrouter",
        display_name = "OpenRouter",
        base_url = "https://openrouter.ai/api/v1",
        models = ["openai/gpt-4o"],
    )
    credential_secrets.save_provider_api_key("router", "or-key")
    response = _client().get("/api/settings/systemone/connections")
    assert response.status_code == 200
    assert response.json() == []


def test_only_saved_decision_connections_are_accepted_as_the_model():
    client = _client()
    _connection()
    providers_db.create_provider(
        id = "chat", provider_type = "openai", display_name = "OpenAI", base_url = "", models = ["gpt-5.5"]
    )
    providers_db.create_provider(
        id = "vllm", provider_type = "custom", display_name = "vLLM", base_url = "", models = ["qwen"]
    )
    for model in (
        "connection:deciders:other",
        "connection:chat:gpt-5.5",
        "connection:vllm:qwen",
        "connection:gone:x",
    ):
        response = client.put("/api/settings/systemone", json = {"model": model})
        assert response.status_code == 400, model
    saved = client.put("/api/settings/systemone", json = {"model": "connection:deciders:jev-latest"})
    assert saved.status_code == 200, saved.text


@pytest.mark.parametrize("status", [429, 503, 529])
def test_upstream_backpressure_keeps_the_decision_api_error_shape(upstream, status):
    client = _client()
    client.put("/api/settings/systemone", json = {"model": _connection()})
    upstream.replies[0] = httpx.Response(
        status,
        headers = {"Retry-After": "7"},
        json = {"detail": {"error_type": "rate_limited", "message": "Slow down"}},
    )
    response = _post(client)
    assert response.status_code == status
    assert response.headers["retry-after"] == "7"
    assert response.json() == {"detail": {"error_type": "rate_limited", "message": "Slow down"}}


def test_an_upstream_text_detail_reaches_the_caller(upstream):
    client = _client()
    client.put("/api/settings/systemone", json = {"model": _connection()})
    upstream.replies[0] = httpx.Response(400, json = {"detail": "Unknown model jev-9"})
    response = _post(client)
    assert response.status_code == 502
    assert response.json()["detail"]["message"] == "Unknown model jev-9"


def test_durable_research_refuses_a_decision_connection():
    from routes.research_runs import CreateResearchRun, _sanitize_config

    _connection("custom", "systemone", "http://decider.example/v1")
    payload = CreateResearchRun(
        threadId = "t",
        userMessageId = "u",
        inferenceRequest = {
            "providerId": "deciders",
            "providerType": "custom",
            "externalModel": "jev-latest",
        },
    )
    with pytest.raises(HTTPException) as refused:
        _sanitize_config(payload, {"modelId": "local-model"})
    assert refused.value.status_code == 400


def test_a_model_unticked_on_the_connection_is_not_called(upstream):
    client = _client()
    client.put("/api/settings/systemone", json = {"model": _connection()})
    providers_db.update_provider("deciders", models = ["jev-1.13"])
    response = _post(client)
    assert response.status_code == 503
    assert "no longer enabled" in response.json()["detail"]["message"]
    assert upstream.calls == []


def test_a_disabled_connection_is_not_called(upstream):
    client = _client()
    client.put("/api/settings/systemone", json = {"model": _connection()})
    providers_db.update_provider("deciders", is_enabled = False)
    response = _post(client)
    assert response.status_code == 503
    assert "disabled" in response.json()["detail"]["message"]
    assert upstream.calls == []


def test_decisions_mcp_uses_the_connection(upstream):
    from fastmcp import Client

    _client().put("/api/settings/systemone", json = {"model": _connection()})

    async def call():
        async with Client(systemone.decisions_mcp) as mcp:
            return await mcp.call_tool("decide", {"state": "x", "questions": QUESTIONS})

    assert asyncio.run(call()).structured_content == UPSTREAM
    assert len(upstream.calls) == 1


def test_decision_settings_are_read_off_the_event_loop(upstream, studio, monkeypatch):
    import threading

    from fastmcp import Client

    client = _client()
    loop_threads = []
    read_on_loop = []

    @client.app.middleware("http")
    async def note_loop_thread(request, call_next):
        loop_threads.append(threading.current_thread())
        return await call_next(request)

    assert client.put("/api/settings/systemone", json = {"model": _connection()}).status_code == 200

    def read(key, fallback = None):
        if threading.current_thread() in loop_threads:
            read_on_loop.append(key)
        return studio.get(key, fallback)

    monkeypatch.setattr(systemone_settings, "_owner_setting", read)
    assert _post(client, "default").status_code == 200

    async def call():
        loop_threads.append(threading.current_thread())
        async with Client(systemone.decisions_mcp) as mcp:
            return await mcp.call_tool("decide", {"state": "x", "questions": QUESTIONS})

    assert asyncio.run(call()).structured_content == UPSTREAM
    assert read_on_loop == []


def test_a_managed_caller_keeps_its_egress_policy_on_an_owner_connection(upstream):
    from utils.account_context import arun_as

    _client().put("/api/settings/systemone", json = {"model": _connection()})
    providers_db.update_provider("deciders", base_url = "http://127.0.0.1:9/v1")
    alice = AccountContext("a" * 32, "alice")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            arun_as(
                alice,
                systemone._decide(
                    catalog.Connection("deciders", "jev-latest"),
                    "x",
                    {name: systemone.QuestionIn(**q) for name, q in QUESTIONS.items()},
                ),
            )
        )
    assert refused.value.status_code == 503
    assert "removed" not in refused.value.detail["message"]
    assert upstream.calls == []


def test_managed_accounts_lose_the_decisions_mcp_bypass_for_a_connection():
    from core.inference.mcp_client import validate_mcp_address

    alice = AccountContext("a" * 32, "alice")
    url = f"http://127.0.0.1:8888{systemone.MCP_PATH}/"
    run_as(alice, validate_mcp_address, url)
    _client().put("/api/settings/systemone", json = {"model": _connection()})
    with pytest.raises(HTTPException):
        run_as(alice, validate_mcp_address, url)


def _providers_post(path, payload):
    from routes import providers

    app = FastAPI()
    app.include_router(providers.router, prefix = "/providers")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app).post(path, json = payload).json()


def _provider_test(payload):
    return _providers_post("/providers/test", payload)


def test_a_system_one_connection_lists_only_decision_models(upstream):
    upstream.replies[0] = httpx.Response(200, json = OPENROUTER_MODELS)
    listed = _providers_post(
        "/providers/models",
        {
            "provider_type": "custom",
            "api_type": "systemone",
            "base_url": "http://localhost:8888/v1",
        },
    )
    assert [model["id"] for model in listed] == ["upstage/solar-decide", "typesafe/jev-1.13"]
    assert upstream.calls[0].url.params["output_modalities"] == "decisions"

    _providers_post(
        "/providers/models",
        {
            "provider_type": "custom",
            "api_type": "systemone",
            "base_url": "http://localhost:8888/v1/systemone",
        },
    )
    assert upstream.calls[-1].url.path == "/v1/models"

    upstream.replies[0] = httpx.Response(200, json = {"data": [{"id": "HuggingFaceTB/SmolLM2-135M"}]})
    chat = _providers_post(
        "/providers/models", {"provider_type": "custom", "base_url": "http://localhost:8888/v1"}
    )
    assert [model["id"] for model in chat] == ["HuggingFaceTB/SmolLM2-135M"]
    assert "output_modalities" not in upstream.calls[-1].url.params


@pytest.mark.parametrize(
    "reply,detail",
    [
        (httpx.Response(401, json = {"detail": "no key"}), "refused the API key (HTTP 401)"),
        (httpx.Response(500, text = "boom"), "answered HTTP 500"),
        (httpx.Response(404, text = "not found"), None),
        (httpx.Response(200, text = "<html>welcome</html>"), None),
        (httpx.Response(200, json = [{"id": "gpt2"}]), None),
        (httpx.Response(200, json = {"data": ["jev", {"id": "x", "architecture": None}]}), None),
    ],
)
def test_a_system_one_model_list_explains_what_went_wrong(upstream, reply, detail):
    upstream.replies[0] = reply
    listed = _providers_post(
        "/providers/models",
        {
            "provider_type": "custom",
            "api_type": "systemone",
            "base_url": "http://localhost:8888/v1",
        },
    )
    if detail is None:
        assert listed == []
    else:
        assert detail in listed["detail"]


def test_studio_lists_its_decision_models_only_when_asked(monkeypatch, studio):
    from routes import inference

    async def chat_catalog():
        return [{"id": "unsloth/Qwen3-0.6B", "object": "model"}]

    monkeypatch.setattr(inference, "_openai_catalog_objects", chat_catalog)

    def ids(output_modalities = None):
        listed = asyncio.run(
            inference.openai_list_models(output_modalities = output_modalities, current_subject = "t")
        )
        return [model["id"] for model in listed["data"]]

    # Without torch only the GGUF-only entries, which llama.cpp serves, stay listed.
    gguf_only = [n for n in catalog.CHECKPOINTS if systemone_settings.llama_cpp_only(n)]
    assert gguf_only and ids("decisions") == ["default", *gguf_only]
    monkeypatch.setattr(systemone_settings, "runtime_unavailable_reason", lambda: None)
    laya = ["default", *catalog.CHECKPOINTS]
    assert ids() == ["unsloth/Qwen3-0.6B"]
    assert ids("text") == ids("image") == ["unsloth/Qwen3-0.6B"]
    assert ids("decisions") == laya
    assert ids("text,decisions") == ids("all") == ["unsloth/Qwen3-0.6B", *laya]
    studio[systemone_settings.ENABLED_KEY] = False
    assert ids("decisions") == []
    assert ids() == ["unsloth/Qwen3-0.6B"]


@pytest.mark.parametrize(
    "reply,success",
    [
        (httpx.Response(200, json = {"answers": {"outage": {"type": "noul", "noul": 0.9}}}), True),
        (httpx.Response(200, json = {"choices": [{"message": {"content": "hi"}}]}), False),
        (httpx.Response(200, text = "<html>welcome</html>"), False),
        (httpx.Response(404, json = {"detail": "Not Found"}), False),
    ],
)
def test_connection_test_checks_the_system_one_shape(upstream, reply, success):
    upstream.replies[0] = reply
    result = _provider_test(
        {
            "provider_type": "custom",
            "api_type": "systemone",
            "base_url": "http://localhost:8080/v1",
            "model_id": "jev-latest",
        }
    )
    assert result["success"] is success, result
    if not success:
        assert "does not speak the System One API" in result["message"]
    assert json.loads(upstream.calls[0].content)["questions"]["outage"]["type"] == "noul"
    assert upstream.calls[0].url.path == "/v1/systemone"


def test_typesafe_check_asks_its_decision_endpoint(upstream):
    upstream.replies[0] = httpx.Response(
        200, json = {"answers": {"outage": {"type": "noul", "noul": 0.9}}}
    )
    result = _provider_test({"provider_type": "typesafe", "model_id": "jev-latest"})
    assert result["success"] is True, result
    assert str(upstream.calls[0].url) == "https://api.typesafe.ai/v1/systemone"


@pytest.mark.parametrize(
    "provider_type,api_type", [("typesafe", "chat_completions"), ("custom", "systemone")]
)
def test_chat_refuses_a_decision_connection(upstream, provider_type, api_type):
    from models.inference import ChatCompletionRequest
    from routes import inference

    _connection(provider_type, api_type)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "Hi"}],
        stream = False,
        provider_id = "deciders",
        external_model = "jev-latest",
    )
    request = SimpleNamespace(headers = {}, state = SimpleNamespace(), url = SimpleNamespace(path = "/"))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(inference._proxy_to_external_provider(payload, request))
    assert refused.value.status_code == 400
    assert "Decision API" in refused.value.detail
    assert upstream.calls == []
