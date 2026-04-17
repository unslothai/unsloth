# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the OpenAI Responses API schemas and input normalisation.
These tests do NOT require a running server or GPU -- they validate
the Pydantic models and the _normalise_responses_input helper.
"""

import sys
import os
import json
import re

# Ensure backend is on path
_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import (
    ResponsesRequest,
    ResponsesInputMessage,
    ResponsesInputTextPart,
    ResponsesInputImagePart,
    ResponsesOutputTextContent,
    ResponsesOutputMessage,
    ResponsesUsage,
    ResponsesResponse,
    ChatMessage,
    TextContentPart,
    ImageContentPart,
    ImageUrl,
    ChatCompletionRequest,
)


# ── _normalise_responses_input: copied from routes/inference.py ──
# We cannot import routes.inference directly because routes/__init__.py
# pulls in heavy dependencies (structlog/twisted/torch). This is a
# direct copy of the function for testing purposes.


def _normalise_responses_input(payload: ResponsesRequest) -> list:
    """Convert a ResponsesRequest into a list of ChatMessage for the completions backend."""
    messages = []

    # System / developer instructions
    if payload.instructions:
        messages.append(ChatMessage(role = "system", content = payload.instructions))

    # Simple string input
    if isinstance(payload.input, str):
        if payload.input:
            messages.append(ChatMessage(role = "user", content = payload.input))
        return messages

    # List of ResponsesInputMessage
    for msg in payload.input:
        role = "system" if msg.role == "developer" else msg.role

        if isinstance(msg.content, str):
            messages.append(ChatMessage(role = role, content = msg.content))
        else:
            # Convert Responses content parts -> Chat content parts
            parts = []
            for part in msg.content:
                if isinstance(part, ResponsesInputTextPart):
                    parts.append(TextContentPart(type = "text", text = part.text))
                elif isinstance(part, ResponsesInputImagePart):
                    parts.append(
                        ImageContentPart(
                            type = "image_url",
                            image_url = ImageUrl(url = part.image_url, detail = part.detail),
                        )
                    )
            messages.append(ChatMessage(role = role, content = parts if parts else ""))

    return messages


# =====================================================================
# Schema validation tests
# =====================================================================


class TestResponsesRequest:
    """Validate ResponsesRequest accepts the shapes the OpenAI SDK sends."""

    def test_minimal_string_input(self):
        req = ResponsesRequest(input = "Hello")
        assert req.input == "Hello"
        assert req.stream is False
        assert req.model == "default"

    def test_message_list_input(self):
        req = ResponsesRequest(
            input = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert len(req.input) == 2
        assert req.input[0].role == "user"
        assert req.input[0].content == "Hi"

    def test_multimodal_input(self):
        req = ResponsesRequest(
            input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/img.png",
                        },
                    ],
                },
            ],
        )
        parts = req.input[0].content
        assert len(parts) == 2
        assert isinstance(parts[0], ResponsesInputTextPart)
        assert isinstance(parts[1], ResponsesInputImagePart)

    def test_instructions_field(self):
        req = ResponsesRequest(
            input = "test",
            instructions = "You are a helpful assistant.",
        )
        assert req.instructions == "You are a helpful assistant."

    def test_extra_fields_accepted(self):
        """OpenAI SDK may send fields we don't model -- extra='allow' should pass."""
        req = ResponsesRequest(
            input = "test",
            tools = [{"type": "web_search_preview"}],
            store = True,
            metadata = {"key": "value"},
            previous_response_id = "resp_abc123",
        )
        assert req.tools == [{"type": "web_search_preview"}]
        assert req.store is True

    def test_stream_flag(self):
        req = ResponsesRequest(input = "test", stream = True)
        assert req.stream is True

    def test_temperature_and_top_p(self):
        req = ResponsesRequest(input = "test", temperature = 0.8, top_p = 0.9)
        assert req.temperature == 0.8
        assert req.top_p == 0.9

    def test_max_output_tokens(self):
        req = ResponsesRequest(input = "test", max_output_tokens = 512)
        assert req.max_output_tokens == 512

    def test_developer_role(self):
        req = ResponsesRequest(
            input = [{"role": "developer", "content": "System instructions"}],
        )
        assert req.input[0].role == "developer"


# =====================================================================
# Response model tests
# =====================================================================


class TestResponsesResponse:
    """Validate response models serialise correctly."""

    def test_basic_response(self):
        resp = ResponsesResponse(
            model = "test-model",
            output = [
                ResponsesOutputMessage(
                    content = [ResponsesOutputTextContent(text = "Hello!")]
                ),
            ],
            usage = ResponsesUsage(input_tokens = 10, output_tokens = 5, total_tokens = 15),
        )
        d = resp.model_dump()
        assert d["object"] == "response"
        assert d["status"] == "completed"
        assert d["output"][0]["type"] == "message"
        assert d["output"][0]["content"][0]["type"] == "output_text"
        assert d["output"][0]["content"][0]["text"] == "Hello!"
        assert d["usage"]["input_tokens"] == 10
        assert d["usage"]["output_tokens"] == 5
        assert d["usage"]["total_tokens"] == 15
        # Must NOT have prompt_tokens / completion_tokens
        assert "prompt_tokens" not in d["usage"]
        assert "completion_tokens" not in d["usage"]

    def test_id_format(self):
        resp = ResponsesResponse()
        assert resp.id.startswith("resp_")

    def test_output_message_id_format(self):
        msg = ResponsesOutputMessage()
        assert msg.id.startswith("msg_")

    def test_annotations_default_empty(self):
        part = ResponsesOutputTextContent(text = "hi")
        assert part.annotations == []

    def test_response_json_roundtrip(self):
        resp = ResponsesResponse(
            model = "gpt-4",
            output = [
                ResponsesOutputMessage(
                    content = [ResponsesOutputTextContent(text = "ok")],
                ),
            ],
            usage = ResponsesUsage(input_tokens = 1, output_tokens = 1, total_tokens = 2),
        )
        j = json.loads(resp.model_dump_json())
        assert j["object"] == "response"
        assert j["output"][0]["role"] == "assistant"
        assert j["output"][0]["status"] == "completed"


# =====================================================================
# Input normalisation tests
# =====================================================================


class TestNormaliseResponsesInput:
    """Test _normalise_responses_input converts Responses input to ChatMessages."""

    def test_string_input(self):
        payload = ResponsesRequest(input = "Hello world")
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 1
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hello world"

    def test_instructions_become_system_message(self):
        payload = ResponsesRequest(
            input = "Hi",
            instructions = "Be concise.",
        )
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be concise."
        assert msgs[1].role == "user"
        assert msgs[1].content == "Hi"

    def test_message_list(self):
        payload = ResponsesRequest(
            input = [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "user"

    def test_developer_role_maps_to_system(self):
        payload = ResponsesRequest(
            input = [{"role": "developer", "content": "Instructions"}],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "system"
        assert msgs[0].content == "Instructions"

    def test_multimodal_parts(self):
        payload = ResponsesRequest(
            input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this:"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,abc",
                        },
                    ],
                },
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 1
        content = msgs[0].content
        assert isinstance(content, list)
        assert len(content) == 2
        assert isinstance(content[0], TextContentPart)
        assert content[0].text == "Describe this:"
        assert isinstance(content[1], ImageContentPart)
        assert content[1].image_url.url == "data:image/png;base64,abc"

    def test_empty_string_input(self):
        payload = ResponsesRequest(input = "")
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 0

    def test_empty_list_input(self):
        payload = ResponsesRequest(input = [])
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 0

    def test_instructions_only(self):
        payload = ResponsesRequest(input = "", instructions = "System msg")
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 1
        assert msgs[0].role == "system"

    def test_instructions_plus_message_list(self):
        payload = ResponsesRequest(
            input = [{"role": "user", "content": "Hello"}],
            instructions = "Be brief.",
        )
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be brief."
        assert msgs[1].role == "user"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
