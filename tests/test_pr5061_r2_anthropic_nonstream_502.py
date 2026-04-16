import asyncio
import os, sys
from unittest.mock import patch

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

import httpx
import pytest
from fastapi import HTTPException


class _Llama:
    base_url = "http://127.0.0.1:0"
    _api_key = None


def test_httpx_connect_error_mapped_to_502():
    from routes import inference as inf_mod

    class _BadClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, *a, **kw):
            raise httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))

    with patch.object(inf_mod.httpx, "AsyncClient", _BadClient):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(
                inf_mod._anthropic_passthrough_non_streaming(
                    _Llama(),
                    openai_messages=[{"role": "user", "content": "q"}],
                    openai_tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    max_tokens=None,
                    message_id="msg_1",
                    model_name="stub",
                    tool_choice="auto",
                )
            )
    assert ei.value.status_code == 502
    assert "Lost connection" in ei.value.detail


def test_httpx_read_error_mapped_to_502():
    from routes import inference as inf_mod

    class _BadClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def post(self, *a, **kw):
            raise httpx.ReadError("eof", request=httpx.Request("POST", "http://x"))

    with patch.object(inf_mod.httpx, "AsyncClient", _BadClient):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(
                inf_mod._anthropic_passthrough_non_streaming(
                    _Llama(),
                    openai_messages=[{"role": "user", "content": "q"}],
                    openai_tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                    max_tokens=None,
                    message_id="msg_1",
                    model_name="stub",
                )
            )
    assert ei.value.status_code == 502
