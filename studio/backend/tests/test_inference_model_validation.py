# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import DiffusionGenerateRequest, LoadRequest, ValidateModelRequest


def _base_load_request(**overrides):
    data = {
        "model_path": "unsloth/test-model-GGUF",
        "hf_token": None,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "is_lora": False,
        "gguf_variant": "Q4_K_M",
    }
    data.update(overrides)
    return LoadRequest.model_validate(data)


def test_validate_request_body_preserves_llama_extra_args_tri_state():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/validate-contract")
    def validate_contract(request: ValidateModelRequest):
        return {
            "fields_set": sorted(request.model_fields_set),
            "llama_extra_args": request.llama_extra_args,
        }

    client = TestClient(app)
    absent = client.post("/validate-contract", json = {"model_path": "model"})
    cleared = client.post(
        "/validate-contract", json = {"model_path": "model", "llama_extra_args": []}
    )
    replaced = client.post(
        "/validate-contract",
        json = {"model_path": "model", "llama_extra_args": ["--top-k", "20"]},
    )

    assert absent.status_code == cleared.status_code == replaced.status_code == 200
    assert "llama_extra_args" not in absent.json()["fields_set"]
    assert absent.json()["llama_extra_args"] is None
    assert cleared.json()["llama_extra_args"] == []
    assert "llama_extra_args" in cleared.json()["fields_set"]
    assert replaced.json()["llama_extra_args"] == ["--top-k", "20"]


def test_newly_blocked_inherited_llama_args_quarantine_the_whole_list(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException

    from routes import inference as route

    backend = SimpleNamespace(
        extra_args = ["--port", "9999", "--top-k", "20"],
        extra_args_source = ("owner/repo", "Q4_K_M"),
    )
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)
    request = ValidateModelRequest(
        model_path = "owner/repo",
        gguf_variant = "Q4_K_M",
    )
    config = SimpleNamespace(is_gguf = True, gguf_variant = "Q4_K_M")

    with pytest.raises(HTTPException) as excinfo:
        route._resolve_inherited_extra_args(
            request,
            config,
            "owner/repo",
            None,
        )

    assert excinfo.value.status_code == 409
    assert "Run Settings" in str(excinfo.value.detail)
    assert backend.extra_args == ["--port", "9999", "--top-k", "20"]


def test_api_key_field_presence_rejects_null_empty_and_nonempty():
    from fastapi import HTTPException
    from routes import inference as route

    omitted = LoadRequest(model_path = "owner/repo")
    route._reject_api_key_custom_arguments(omitted, True)
    normalized_omission = omitted.model_copy(update = {"llama_extra_args": None})
    assert not route._llama_args_value_supplied(normalized_omission)

    for value in (None, [], ["--top-k", "20"]):
        request = LoadRequest.model_validate(
            {"model_path": "owner/repo", "llama_extra_args": value}
        )
        with pytest.raises(HTTPException) as excinfo:
            route._reject_api_key_custom_arguments(request, True)
        assert excinfo.value.status_code == 403

    assert route._llama_args_value_supplied(
        LoadRequest(model_path = "owner/repo", llama_extra_args = [])
    )


def test_ui_explicit_empty_clears_and_omitted_same_runtime_inherits(monkeypatch):
    from types import SimpleNamespace
    from routes import inference as route

    backend = SimpleNamespace(
        extra_args = ["--top-k", "20"],
        extra_args_source = ("owner/repo", "Q4_K_M"),
    )
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: backend)
    config = SimpleNamespace(is_gguf = True, gguf_variant = "Q4_K_M")

    omitted = ValidateModelRequest(model_path = "owner/repo", gguf_variant = "Q4_K_M")
    cleared = ValidateModelRequest(
        model_path = "owner/repo", gguf_variant = "Q4_K_M", llama_extra_args = []
    )

    assert route._resolve_inherited_extra_args(omitted, config, "owner/repo", None) == [
        "--top-k",
        "20",
    ]
    assert route._resolve_inherited_extra_args(cleared, config, "owner/repo", []) == []


def test_api_omission_uses_saved_exact_args_not_unsaved_resident(monkeypatch):
    from types import SimpleNamespace
    from routes import inference as route
    from utils import openai_auto_switch_settings as settings

    resident = SimpleNamespace(
        extra_args = ["--temperature", "0.9"],
        extra_args_source = ("owner/repo", "Q4_K_M"),
    )
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: resident)
    monkeypatch.setattr(
        settings,
        "resolve_model_override_candidates",
        lambda *a, **k: (
            "owner/repo:Q4_K_M",
            {"llama_extra_args": ["--top-k", "20"]},
        ),
    )
    request = ValidateModelRequest(model_path = "owner/repo", gguf_variant = "Q4_K_M")
    config = SimpleNamespace(
        is_gguf = True,
        gguf_variant = "Q4_K_M",
        identifier = "owner/repo",
    )

    assert route._resolve_inherited_extra_args(
        request,
        config,
        "owner/repo",
        None,
        args_origin = route.LlamaArgsOrigin.API_REQUEST,
    ) == ["--top-k", "20"]


def test_api_omission_quarantines_a_present_saved_null(monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    from routes import inference as route
    from utils import openai_auto_switch_settings as settings

    monkeypatch.setattr(
        settings,
        "resolve_model_override_candidates",
        lambda *a, **k: ("owner/repo:Q4_K_M", {"llama_extra_args": None}),
    )
    request = ValidateModelRequest(model_path = "owner/repo", gguf_variant = "Q4_K_M")
    config = SimpleNamespace(
        is_gguf = True,
        gguf_variant = "Q4_K_M",
        identifier = "owner/repo",
    )

    with pytest.raises(HTTPException) as excinfo:
        route._resolve_inherited_extra_args(
            request,
            config,
            "owner/repo",
            None,
            args_origin = route.LlamaArgsOrigin.API_REQUEST,
        )
    assert excinfo.value.status_code == 409


def test_blocked_direct_load_rejects_before_monitor_state(monkeypatch):
    import asyncio
    from fastapi import HTTPException
    from routes import inference as route

    calls = []
    monkeypatch.setattr(
        route.api_monitor,
        "record_lifecycle",
        lambda **_kwargs: calls.append("monitor"),
    )
    request = _base_load_request(llama_extra_args = ["--rpc", "127.0.0.1:5000"])
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            route._load_model_impl(
                request,
                None,
                "tester",
                args_origin = route.LlamaArgsOrigin.UI_REQUEST,
            )
        )
    assert excinfo.value.status_code == 400
    assert calls == []


def test_blank_chat_template_override_normalizes_to_none():
    req = _base_load_request(chat_template_override = "   \n\t")

    assert req.chat_template_override is None


def test_nonblank_chat_template_override_is_preserved_verbatim():
    template = "  {{ messages }}  "
    req = _base_load_request(chat_template_override = template)

    assert req.chat_template_override == template


# ---------- ChatCompletionRequest tool_call_id walkback ----------

from models.inference import ChatCompletionRequest


def _req(messages, **overrides):
    payload = {"model": "x", "messages": messages, **overrides}
    return ChatCompletionRequest.model_validate(payload)


def test_tool_message_inherits_id_from_prior_assistant_tool_call():
    req = _req(
        [
            {"role": "user", "content": "what is 2+2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_real123",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},  # no tool_call_id
        ]
    )
    assert req.messages[-1].tool_call_id == "call_real123"


def test_tool_message_with_explicit_id_unchanged():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_user_supplied", "content": "ok"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_user_supplied"


def test_walkback_prefers_function_name_match():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "call_y",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_y"


def test_walkback_takes_first_unconsumed_when_no_name():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "content": "first result"},
            {"role": "tool", "content": "second result"},
        ]
    )
    assert req.messages[-2].tool_call_id == "call_a"
    assert req.messages[-1].tool_call_id == "call_b"


def test_walkback_falls_back_to_synth_when_no_assistant_turn():
    req = _req(
        [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "orphan"},
        ]
    )
    tcid = req.messages[-1].tool_call_id
    assert tcid is not None and tcid.startswith("call_") and len(tcid) > 5


def test_walkback_does_not_cross_user_turn():
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "old_call",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "old_call", "content": "4"},
            {"role": "user", "content": "next turn"},
            {"role": "tool", "content": "no parent in this turn"},
        ]
    )
    last = req.messages[-1].tool_call_id
    # Walkback must NOT pick old_call across a user turn; falls back to synth.
    assert last is not None
    assert last != "old_call"
    assert last.startswith("call_")


def test_walkback_skips_explicitly_consumed_tool_call_id():
    """An explicit-id tool result reserves its assistant slot so a
    follow-up missing-id result picks the OTHER tool call."""
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "4"},
            {"role": "tool", "content": "second result"},
        ]
    )
    assert [m.tool_call_id for m in req.messages if m.role == "tool"] == ["call_a", "call_b"]


def test_walkback_handles_malformed_function_string():
    """A tool_call with ``function`` as a string (provider quirk) must not
    raise; resolution falls back to id selection."""
    req = _req(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": "calc"},
                ],
            },
            {"role": "tool", "name": "calc", "content": "4"},
        ]
    )
    assert req.messages[-1].tool_call_id == "call_a"


# ── DiffusionLoadRequest.attention_backend casing (Literal validated before normalizer) ──
from pydantic import ValidationError

from models.inference import DiffusionLoadRequest


def _diff_load(**kw):
    return DiffusionLoadRequest(model_path = "repo", gguf_filename = "m.gguf", **kw)


def test_attention_backend_casing_and_whitespace_normalized():
    # The dispatcher accepts case/whitespace variants, so the before-validator must fold them or the lowercase Literal 422s a valid request.
    assert _diff_load(attention_backend = "CuDNN").attention_backend == "cudnn"
    assert _diff_load(attention_backend = "  sage ").attention_backend == "sage"


def test_attention_backend_none_preserved():
    assert _diff_load(attention_backend = None).attention_backend is None
    assert _diff_load().attention_backend is None


def test_attention_backend_unknown_still_rejected():
    with pytest.raises(ValidationError):
        _diff_load(attention_backend = "bogus")


def test_load_rejects_a_duplicate_lora_id_like_generate_does():
    """The load path bakes adapters into the quantized build, so it needs generate's guard too.

    _resolve_lora_set suffixes colliding adapter names, so a repeated id resolves the SAME adapter
    twice and set_adapters stacks both copies past the per-adapter weight bound. On the generation
    path that is one bad image; baked into a quantized build it rides every image until a reload.
    """
    dup = [{"id": "me/adapter", "weight": 0.8}, {"id": "me/adapter", "weight": 0.8}]
    with pytest.raises(ValidationError, match = "duplicate LoRA id"):
        _diff_load(loras = dup)
    with pytest.raises(ValidationError, match = "duplicate LoRA id"):
        DiffusionGenerateRequest(prompt = "a cat", loras = dup)
    # Distinct ids are untouched.
    assert (
        len(_diff_load(loras = [{"id": "me/a", "weight": 0.8}, {"id": "me/b", "weight": 0.5}]).loras)
        == 2
    )
