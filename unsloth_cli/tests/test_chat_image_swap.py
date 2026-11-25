# SPDX-License-Identifier: AGPL-3.0-only
import io
import json

from unsloth_cli._inference import HttpChatBackend


def test_image_generator_uses_the_regular_stream(monkeypatch):
    client = HttpChatBackend("http://localhost", "test")
    client.image_generator = "local/sdxl"
    requests = []

    def request(method, path, body):
        requests.append((method, path, body))
        chunk = {"choices": [{"delta": {"content": "Image saved."}, "finish_reason": "stop"}]}
        return io.BytesIO(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())

    monkeypatch.setattr(client, "_request", request)
    messages = [{"role": "user", "content": "Draw a kite"}]
    reply = list(
        client.stream(
            messages,
            system_prompt = "",
            temperature = None,
            top_p = None,
            top_k = None,
            max_new_tokens = None,
            repetition_penalty = None,
            enable_thinking = False,
        )
    )
    assert reply == ["Image saved."]
    assert len(requests) == 1
    method, path, body = requests[0]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert body["enabled_tools"] == ["generate_image"] and body["enable_tools"]
    assert body["image_generator"] == "local/sdxl"
    assert body["parallel_tool_calls"] is False
    assert body["max_tool_calls_per_message"] == 1
    assert body["stream"] and "tools" not in body
