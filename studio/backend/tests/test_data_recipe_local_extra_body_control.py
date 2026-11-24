# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A recipe's extra_body for the loaded local model reaches Studio's own /v1 with a minted workflow key:
only generation settings pass, never tools, approval, sandbox or provider rerouting."""

import sys
from types import SimpleNamespace

from auth import storage
from routes.data_recipe import jobs as route

MODEL = "unsloth/Qwen3-0.6B"


def _inject(monkeypatch, extra_body):
    llama = SimpleNamespace(is_loaded = True, model_identifier = MODEL, hf_variant = "")
    backend = SimpleNamespace(active_model_name = "")
    monkeypatch.setitem(
        sys.modules, "routes.inference", SimpleNamespace(get_llama_cpp_backend = lambda: llama)
    )
    monkeypatch.setitem(
        sys.modules,
        "core.inference",
        SimpleNamespace(
            get_inference_backend = lambda: backend,
            model_slots = SimpleNamespace(visible = lambda: [], in_slot = lambda slot, fn: fn()),
        ),
    )
    monkeypatch.setattr(
        route, "_resolve_local_v1_endpoint", lambda request: "http://127.0.0.1:8888/v1"
    )
    monkeypatch.setattr(storage, "create_api_key", lambda **kwargs: ("sk-unsloth-test", {"id": 7}))
    recipe = {
        "model_providers": [{"name": "local", "is_local": True}],
        "model_configs": [
            {
                "alias": "chat",
                "model": MODEL,
                "provider": "local",
                "inference_parameters": {"temperature": 0.7, "extra_body": extra_body},
            }
        ],
        "columns": [{"column_type": "llm-text", "name": "answer", "model_alias": "chat"}],
    }
    route._inject_local_providers(recipe, SimpleNamespace())
    return recipe["model_configs"][0]["inference_parameters"]["extra_body"]


def test_tool_and_provider_fields_are_dropped_for_the_local_model(monkeypatch):
    sent = _inject(
        monkeypatch,
        {
            "provider_type": "custom",
            "provider_base_url": "https://author.example/v1",
            "external_model": "m",
            "stream": True,
            "enable_tools": True,
            "enabled_tools": ["terminal"],
            "mcp_enabled": True,
            "permission_mode": "off",
            "sandbox_level": "low",
            "run_tools_locally": True,
        },
    )

    assert sent == {"chat_template_kwargs": {"enable_thinking": False}}


def test_generation_settings_still_reach_the_local_model(monkeypatch):
    sent = _inject(
        monkeypatch,
        {"top_k": 20, "min_p": 0.05, "repetition_penalty": 1.1, "seed": 3, "mirostat": 2},
    )

    assert sent == {
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "seed": 3,
        "mirostat": 2,
        "chat_template_kwargs": {"enable_thinking": False},
    }
