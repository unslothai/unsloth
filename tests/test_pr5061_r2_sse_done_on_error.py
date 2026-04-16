import asyncio
import os, sys, threading
from unittest.mock import patch, MagicMock, AsyncMock

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

import httpx

from models.inference import ChatCompletionRequest


class _Llama:
    base_url = "http://127.0.0.1:0"
    _api_key = None


class _Req:
    async def is_disconnected(self):
        return False


def _collect(stream_response):
    async def _run():
        out = []
        async for chunk in stream_response.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
        return out
    return asyncio.run(_run())


def _make_payload():
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        stream=True,
    )


def test_done_emitted_after_non_200_error():
    from routes import inference as inf_mod

    class _Resp:
        status_code = 500
        async def aread(self):
            return b"server oops"
        async def aclose(self):
            pass

    async def _send(req, stream):
        return _Resp()

    with patch.object(inf_mod.httpx, "AsyncClient") as mock_cls:
        inst = MagicMock()
        inst.build_request = MagicMock(return_value=MagicMock())
        inst.send = _send
        inst.aclose = AsyncMock()
        mock_cls.return_value = inst

        resp = asyncio.run(
            inf_mod._openai_passthrough_stream(
                _Req(), threading.Event(), _Llama(), _make_payload()
            )
        )
        chunks = _collect(resp)

    assert any('"error"' in c for c in chunks)
    assert any(c.strip() == "data: [DONE]" for c in chunks)


def test_done_emitted_after_exception():
    from routes import inference as inf_mod

    async def _send(req, stream):
        raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://x"))

    with patch.object(inf_mod.httpx, "AsyncClient") as mock_cls:
        inst = MagicMock()
        inst.build_request = MagicMock(return_value=MagicMock())
        inst.send = _send
        inst.aclose = AsyncMock()
        mock_cls.return_value = inst

        resp = asyncio.run(
            inf_mod._openai_passthrough_stream(
                _Req(), threading.Event(), _Llama(), _make_payload()
            )
        )
        chunks = _collect(resp)

    assert any('"error"' in c for c in chunks)
    assert any(c.strip() == "data: [DONE]" for c in chunks)


def test_done_comes_after_error_chunk():
    from routes import inference as inf_mod

    async def _send(req, stream):
        raise httpx.ReadError("reset", request=httpx.Request("POST", "http://x"))

    with patch.object(inf_mod.httpx, "AsyncClient") as mock_cls:
        inst = MagicMock()
        inst.build_request = MagicMock(return_value=MagicMock())
        inst.send = _send
        inst.aclose = AsyncMock()
        mock_cls.return_value = inst

        resp = asyncio.run(
            inf_mod._openai_passthrough_stream(
                _Req(), threading.Event(), _Llama(), _make_payload()
            )
        )
        chunks = _collect(resp)

    err_idx = next(i for i, c in enumerate(chunks) if '"error"' in c)
    done_idx = next(i for i, c in enumerate(chunks) if c.strip() == "data: [DONE]")
    assert done_idx > err_idx
