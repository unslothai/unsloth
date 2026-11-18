# SPDX-License-Identifier: AGPL-3.0-only
import json

import pytest

from unsloth_cli._chat_image_swap import image_tool_turn


@pytest.mark.parametrize(
    "text_load",
    [
        {"model_path": "gguf", "gguf_variant": "Q4_K_M", "max_seq_length": 4096},
        {"model_path": "transformers", "load_in_4bit": True},
    ],
)
def test_image_swap(text_load):
    text_id = text_load["model_path"]
    call = {
        "id": "call-42",
        "type": "function",
        "function": {
            "name": "generate_image",
            "arguments": json.dumps({"prompt": "A lighthouse"}),
        },
    }

    def api(path, body):
        if path == "/v1/chat/completions":
            assert body["tools"][0]["function"]["name"] == "generate_image"
            assert body["parallel_tool_calls"] is False
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": None, "tool_calls": [call]}}
                ]
            }
        assert path == "/api/inference/images/chat-swap"
        assert body == {
            "text_id": text_id,
            "image_load": {"model_path": "image"},
            "prompt": "A lighthouse",
        }
        return {"images": [{"id": "image-1", "url": "/gallery/image-1"}]}

    _, history = image_tool_turn(
        api, {"messages": []}, text_id, {"model_path": "image"}, lambda image: None
    )
    assert history[0]["tool_calls"] == [call]
    assert history[1]["tool_call_id"] == "call-42"
    assert json.loads(history[1]["content"])["gallery_image_id"] == "image-1"


def test_plain_reply_keeps_chat_loaded():
    def api(path, body):
        assert path == "/v1/chat/completions"
        return {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}

    choice, history = image_tool_turn(api, {}, "chat", {}, None)
    assert choice["message"]["content"] == "Hi!"
    assert history == []
