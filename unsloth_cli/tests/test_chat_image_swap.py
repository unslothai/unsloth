# SPDX-License-Identifier: AGPL-3.0-only
import io
import json
from urllib.error import HTTPError, URLError

import pytest
import typer

from unsloth_cli import _inference


@pytest.mark.parametrize(
    "settings,loaded,message",
    [
        (404, None, "Update Studio"),
        ({"enabled": False, "managed_by_env": True}, None, "UNSLOTH_CHAT_IMAGE_SWAP"),
        ({"enabled": False}, {"is_mlx": True}, "GGUF or Transformers"),
        ({"enabled": False}, {"is_gguf": True}, None),
    ],
)
def test_image_generator_requires_compatible_studio(monkeypatch, settings, loaded, message):
    monkeypatch.setattr(_inference, "find_studio_server", lambda: "http://localhost")
    monkeypatch.setattr(_inference, "verify_studio_identity", lambda _: True)
    monkeypatch.setattr(_inference, "_studio_token", lambda: "test")
    requests = []

    def request(req, timeout):
        requests.append(req.selector)
        if req.selector == "/api/settings/chat-image-swap":
            if settings == 404:
                raise HTTPError(req.full_url, 404, "Not Found", None, None)
            return io.BytesIO(json.dumps(settings).encode())
        return io.BytesIO(json.dumps(loaded if req.get_method() == "POST" else {}).encode())

    monkeypatch.setattr(_inference, "urlopen_no_redirect", request)
    options = {
        "image_generator": "local/sdxl",
        "hf_token": None,
        "max_seq_length": None,
        "load_in_4bit": None,
    }
    if message:
        with pytest.raises(typer.BadParameter, match = message):
            _inference.connect_studio_server("org/chat", **options)
    else:
        client = _inference.connect_studio_server("org/chat", **options)
        assert client.image_generator == "local/sdxl"
    assert requests[0] == "/api/settings/chat-image-swap"
    assert ("/api/inference/load" in requests) is (loaded is not None)


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


def test_image_generator_tool_failure_is_visible(monkeypatch, capsys):
    client = _inference.HttpChatBackend("http://localhost", "test")
    client.image_generator = "local/sdxl"
    failure = "Error: a different image model is loaded."
    event = {"type": "tool_end", "tool_name": "generate_image", "result": failure}
    monkeypatch.setattr(
        _inference,
        "urlopen_no_redirect",
        lambda req, timeout: io.BytesIO(f"data: {json.dumps(event)}\n\ndata: [DONE]\n".encode()),
    )
    assert (
        list(
            client.stream(
                [],
                system_prompt = "",
                temperature = None,
                top_p = None,
                top_k = None,
                repetition_penalty = None,
                max_new_tokens = None,
                enable_thinking = False,
            )
        )
        == []
    )
    assert failure in capsys.readouterr().err
