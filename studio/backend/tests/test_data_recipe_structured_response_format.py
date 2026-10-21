# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import storage
from models.inference import ChatCompletionRequest, ChatMessage
from routes.data_recipe import jobs as route

from .test_sf_client_tools_passthrough import _call, _fixed, _ScriptedBackend

MODEL = "unsloth/Qwen3-0.6B"
SCHEMA = {
    "type": "object",
    "required": ["name", "score"],
    "properties": {"name": {"type": "string"}, "score": {"type": "integer"}},
}


def structured_recipe():
    return {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [{"alias": "local_model", "model": MODEL, "provider": "local"}],
        "columns": [
            {"column_type": "llm-text", "name": "blurb", "model_alias": "local_model"},
            {
                "column_type": "llm-structured",
                "name": "rating",
                "model_alias": "local_model",
                "output_format": copy.deepcopy(SCHEMA),
            },
        ],
    }


def inject_with_loaded_model(monkeypatch, *, gguf):
    llama = SimpleNamespace(is_loaded = gguf, model_identifier = MODEL, hf_variant = "")
    backend = SimpleNamespace(active_model_name = "" if gguf else MODEL)
    monkeypatch.setitem(
        sys.modules, "routes.inference", SimpleNamespace(get_llama_cpp_backend = lambda: llama)
    )
    monkeypatch.setitem(
        sys.modules, "core.inference", SimpleNamespace(get_inference_backend = lambda: backend)
    )
    monkeypatch.setattr(
        route, "_resolve_local_v1_endpoint", lambda request: "http://127.0.0.1:8888/v1"
    )
    monkeypatch.setattr(storage, "create_api_key", lambda **kwargs: ("sk-unsloth-test", {"id": 7}))
    recipe = structured_recipe()
    assert route._inject_local_providers(recipe, SimpleNamespace()) == 7
    return recipe


def response_formats(recipe):
    return {
        mc["alias"]: mc["inference_parameters"]["extra_body"]["response_format"]
        for mc in recipe["model_configs"]
        if "response_format" in mc["inference_parameters"]["extra_body"]
    }


def column_aliases(recipe):
    return {column["name"]: column["model_alias"] for column in recipe["columns"]}


def test_gguf_model_gets_grammar_response_format(monkeypatch):
    recipe = inject_with_loaded_model(monkeypatch, gguf = True)

    assert column_aliases(recipe) == {
        "blurb": "local_model",
        "rating": "local_model__rating_structured",
    }
    assert response_formats(recipe) == {
        "local_model__rating_structured": {"type": "json_schema", "schema": SCHEMA}
    }


def test_non_gguf_model_keeps_prompt_level_json(monkeypatch):
    recipe = inject_with_loaded_model(monkeypatch, gguf = False)

    # Whole columns, not just aliases: output_format feeds the fallback's prompt-level schema.
    assert recipe["columns"] == structured_recipe()["columns"]
    assert response_formats(recipe) == {}
    assert [mc["alias"] for mc in recipe["model_configs"]] == ["local_model"]
    extra_body = recipe["model_configs"][0]["inference_parameters"]["extra_body"]
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_injected_response_format_is_what_a_non_gguf_backend_refuses(monkeypatch):
    """The gate's premise: without llama.cpp, /v1 refuses the payload the GGUF path injects.
    Teaching the safetensors/MLX backend guided decoding must break this, not go unnoticed."""
    # Its own context so the fake backend modules unwind before the route is called for real.
    with pytest.MonkeyPatch.context() as injection:
        recipe = inject_with_loaded_model(injection, gguf = True)
    extra_body = recipe["model_configs"][-1]["inference_parameters"]["extra_body"]
    payload = ChatCompletionRequest(
        model = "default",
        messages = [ChatMessage(role = "user", content = "rate it")],
        **extra_body,
    )

    with pytest.raises(HTTPException) as caught:
        _call(payload, monkeypatch, _ScriptedBackend(_fixed('{"name": "Riverside", "score": 8}')))

    assert caught.value.status_code == 400
    assert caught.value.detail["error"]["param"] == "response_format"
