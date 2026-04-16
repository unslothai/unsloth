import asyncio
import os, sys
from unittest.mock import patch, MagicMock

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

import pytest
from fastapi import HTTPException

from models.inference import ChatCompletionRequest


class _Llama:
    base_url = "http://127.0.0.1:0"
    _api_key = None


def _payload():
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    )


def _mk_client(status, text):
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
            m.text = text
            m.json = lambda: {}
            return m
    return _Client


def test_nonstream_non_200_calls_logger_error_with_status_and_body():
    from routes import inference as inf_mod

    with patch.object(inf_mod.httpx, "AsyncClient", _mk_client(503, "backend overloaded detail")), \
         patch.object(inf_mod.logger, "error") as mock_error:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(inf_mod._openai_passthrough_non_streaming(_Llama(), _payload()))

    assert ei.value.status_code == 503
    mock_error.assert_called()
    # Check any error call mentions upstream status and body
    found = False
    for call in mock_error.call_args_list:
        msg = call.args[0] if call.args else ""
        combined = f"{msg} {call.args} {call.kwargs}"
        if "upstream error" in combined and "503" in combined and "backend overloaded" in combined:
            found = True
            break
    assert found, f"expected upstream error log with status and body, got {mock_error.call_args_list}"


def test_nonstream_200_does_not_call_logger_error():
    from routes import inference as inf_mod

    with patch.object(inf_mod.httpx, "AsyncClient", _mk_client(200, "{}")), \
         patch.object(inf_mod.logger, "error") as mock_error:
        asyncio.run(inf_mod._openai_passthrough_non_streaming(_Llama(), _payload()))

    # no upstream-error log on 200
    for call in mock_error.call_args_list:
        msg = call.args[0] if call.args else ""
        assert "upstream error" not in msg
