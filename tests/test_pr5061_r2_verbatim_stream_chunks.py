import asyncio
import os, sys, threading
from unittest.mock import patch, MagicMock

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

from models.inference import ChatCompletionRequest


class _Llama:
    base_url = "http://127.0.0.1:0"
    _api_key = None


class _Req:
    async def is_disconnected(self):
        return False


def _payload():
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        stream=True,
    )


class _FakeResp:
    def __init__(self, lines):
        self.status_code = 200
        self._lines = list(lines)

    def aiter_lines(self):
        parent = self

        class _It:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(parent._lines):
                    raise StopAsyncIteration
                v = parent._lines[self._idx]
                self._idx += 1
                return v

            async def aclose(self):
                pass

        return _It()

    async def aread(self):
        return b""

    async def aclose(self):
        pass


def _run_stream(fake_lines):
    from routes import inference as inf_mod

    async def _send(req, stream):
        return _FakeResp(fake_lines)

    async def _aclose_noop():
        return None

    with patch.object(inf_mod.httpx, "AsyncClient") as mock_cls:
        inst = MagicMock()
        inst.build_request = MagicMock(return_value=MagicMock())
        inst.send = _send
        inst.aclose = _aclose_noop
        mock_cls.return_value = inst

        resp = asyncio.run(
            inf_mod._openai_passthrough_stream(
                _Req(), threading.Event(), _Llama(), _payload()
            )
        )

        async def _collect():
            return [
                c if isinstance(c, str) else c.decode("utf-8")
                async for c in resp.body_iterator
            ]

        return asyncio.run(_collect())


def test_passthrough_relays_data_lines_verbatim():
    chunks = _run_stream([
        'data: {"id":"abc","choices":[{"delta":{"content":"hi"}}]}',
        'data: [DONE]',
    ])
    assert any('"id":"abc"' in c for c in chunks)
    assert any("data: [DONE]" in c for c in chunks)


def test_passthrough_ignores_blank_and_non_data_lines():
    chunks = _run_stream([
        "",
        ": heartbeat",
        'data: {"x":1}',
        'data: [DONE]',
    ])
    # Only data: lines propagate.
    for c in chunks:
        assert c.startswith("data: ") or c == ""
    assert any('"x":1' in c for c in chunks)


def test_passthrough_breaks_on_done():
    chunks = _run_stream([
        'data: {"a":1}',
        'data: [DONE]',
        'data: {"should_not_appear":true}',
    ])
    assert not any("should_not_appear" in c for c in chunks)
