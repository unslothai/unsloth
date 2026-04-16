import asyncio
import os, sys
from unittest.mock import patch, MagicMock

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

from models.inference import ChatCompletionRequest


class _Llama:
    base_url = "http://127.0.0.1:0"
    _api_key = None


def _payload():
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    )


def _build_mock_client(status, payload_json):
    class _Client:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, *a, **kw):
            m = MagicMock()
            m.status_code = status
            m.text = ""
            m.json = lambda: payload_json
            return m
    return _Client


def test_verbatim_json_body_returned():
    from routes import inference as inf_mod

    native = {
        "id": "chatcmpl-foo",
        "object": "chat.completion",
        "model": "qwen-native",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }

    with patch.object(inf_mod.httpx, "AsyncClient", _build_mock_client(200, native)):
        resp = asyncio.run(inf_mod._openai_passthrough_non_streaming(_Llama(), _payload()))

    import json
    body = json.loads(resp.body.decode("utf-8"))
    assert body == native  # verbatim
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert body["usage"]["prompt_tokens"] == 42


def test_preserves_native_id_and_model_fields():
    from routes import inference as inf_mod

    native = {
        "id": "chatcmpl-native-xyz",
        "model": "llama-native",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with patch.object(inf_mod.httpx, "AsyncClient", _build_mock_client(200, native)):
        resp = asyncio.run(inf_mod._openai_passthrough_non_streaming(_Llama(), _payload()))

    import json
    body = json.loads(resp.body.decode("utf-8"))
    assert body["id"] == "chatcmpl-native-xyz"
    assert body["model"] == "llama-native"
