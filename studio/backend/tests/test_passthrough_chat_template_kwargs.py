# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests that ``_build_openai_passthrough_body`` consults the family
registry for ``chat_template_kwargs`` and merges them with per-request
overrides in the correct priority order.

Without the registry consult, external OpenAI SDK clients that hit
``/v1/chat/completions`` with ``tools=[...]`` on a gpt-oss / Nemotron
model never see the recommended template kwargs reach llama-server,
even though Studio Chat (which uses the non-passthrough path) gets
them via a different route.
"""

from models.inference import ChatCompletionRequest
from routes.inference import _build_openai_passthrough_body


def _make_payload(**fields):
    base = {
        "model": "gpt-oss",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    base.update(fields)
    return ChatCompletionRequest(**base)


class TestFamilyDefaults:
    def test_gpt_oss_family_default_reaches_outbound(self):
        payload = _make_payload()
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/gpt-oss-120b-GGUF"
        )
        assert body.get("chat_template_kwargs") == {"reasoning_effort": "medium"}

    def test_nemotron_family_default_reaches_outbound(self):
        payload = _make_payload()
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF"
        )
        assert body.get("chat_template_kwargs") == {"enable_thinking": True}

    def test_no_model_identifier_skips_registry(self):
        # Without a model_identifier we cannot look up the family, so
        # behaviour collapses to the prior (request-only) path.
        payload = _make_payload()
        body = _build_openai_passthrough_body(payload)
        assert "chat_template_kwargs" not in body

    def test_unknown_family_yields_no_template_kwargs(self):
        payload = _make_payload()
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/CompletelyMadeUp-99B"
        )
        assert "chat_template_kwargs" not in body


class TestPerRequestOverrides:
    def test_explicit_reasoning_effort_wins_over_family(self):
        payload = _make_payload(reasoning_effort = "high")
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/gpt-oss-120b-GGUF"
        )
        assert body["chat_template_kwargs"]["reasoning_effort"] == "high"

    def test_extra_body_chat_template_kwargs_wins_outright(self):
        payload = _make_payload(chat_template_kwargs = {"reasoning_effort": "low"})
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/gpt-oss-120b-GGUF"
        )
        assert body["chat_template_kwargs"]["reasoning_effort"] == "low"

    def test_enable_thinking_propagates_alongside_family_keys(self):
        # Family carries reasoning_effort, the caller sets enable_thinking.
        # Both should end up in the outbound dict.
        payload = _make_payload(enable_thinking = False)
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/gpt-oss-120b-GGUF"
        )
        kw = body["chat_template_kwargs"]
        assert kw["reasoning_effort"] == "medium"
        assert kw["enable_thinking"] is False


class TestNoSpuriousKwargs:
    def test_qwen3_no_family_kwargs_emits_no_field(self):
        payload = _make_payload()
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/Qwen3-8B-GGUF"
        )
        assert "chat_template_kwargs" not in body

    def test_empty_overrides_collapse_to_no_field(self):
        # No family kwargs and no per-request keys means no outbound
        # field (not an empty dict).
        payload = _make_payload()
        body = _build_openai_passthrough_body(
            payload, model_identifier = "unsloth/Qwen3-8B-GGUF"
        )
        assert "chat_template_kwargs" not in body
