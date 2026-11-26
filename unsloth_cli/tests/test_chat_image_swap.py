# SPDX-License-Identifier: AGPL-3.0-only
import io
import json
from urllib.error import URLError

import pytest

from unsloth_cli import _inference


@pytest.mark.parametrize("signing_available", [True, False])
def test_image_generator_uses_the_regular_stream(monkeypatch, capsys, signing_available):
    client = _inference.HttpChatBackend("http://localhost", "test")
    client.image_generator = "local/sdxl"
    requests = []
    gallery_path = "/api/inference/images/gallery/kite"
    signed_path = gallery_path + "/file-signed?token=example"

    def request(req, timeout):
        requests.append(req)
        if req.get_method() == "GET":
            if not signing_available:
                raise URLError("Signing temporarily unavailable")
            return io.BytesIO(json.dumps({"url": signed_path}).encode())
        events = [
            {
                "type": "tool_end",
                "tool_name": "generate_image",
                "result": 'Saved a kite.\n__FILES__:[{"name":"kite.png","gallery_image_id":"kite"}]',
            },
            {"choices": [{"delta": {"content": "Image saved."}, "finish_reason": "stop"}]},
        ]
        return io.BytesIO(
            (
                "".join("data: " + json.dumps(event) + "\n\n" for event in events)
                + "data: [DONE]\n\n"
            ).encode()
        )

    monkeypatch.setattr(_inference, "urlopen_no_redirect", request)
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
    output = capsys.readouterr().err
    if signing_available:
        assert "http://localhost" + signed_path in output
    else:
        assert "Open the Studio Images gallery" in output
        assert "http://" not in output
    assert len(requests) == 2
    chat, sign = requests
    assert chat.get_method() == "POST" and chat.selector == "/v1/chat/completions"
    assert {key.lower(): value for key, value in chat.header_items()}["x-unsloth-events"] == "1"
    assert sign.get_method() == "GET" and sign.selector == gallery_path + "/signed-url"
    assert sign.get_header("Authorization") == "Bearer test"
    body = json.loads(chat.data)
    assert body["enabled_tools"] == ["generate_image"] and body["enable_tools"]
    assert body["image_generator"] == "local/sdxl"
    assert body["parallel_tool_calls"] is False
    assert body["max_tool_calls_per_message"] == 1
    assert body["stream"] and "tools" not in body
