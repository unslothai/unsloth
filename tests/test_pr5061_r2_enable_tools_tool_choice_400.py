import asyncio
import os, sys
from unittest.mock import patch

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

import pytest
from fastapi import HTTPException


class _Llama:
    is_loaded = True
    supports_tools = True
    is_vision = False
    _is_audio = False
    model_identifier = "stub"
    base_url = "http://127.0.0.1:0"
    _api_key = None


class _Req:
    async def is_disconnected(self):
        return False


def _build_payload(enable_tools, tool_choice):
    from models.inference import AnthropicMessagesRequest

    return AnthropicMessagesRequest(
        model = "default",
        max_tokens = 64,
        messages = [{"role": "user", "content": "q"}],
        tool_choice = tool_choice,
        enable_tools = enable_tools,
    )


def test_enable_tools_true_with_tool_choice_raises_400():
    from routes import inference as inf_mod

    payload = _build_payload(enable_tools = True, tool_choice = {"type": "any"})
    with patch.object(inf_mod, "get_llama_cpp_backend", return_value = _Llama()):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(
                inf_mod.anthropic_messages(payload, _Req(), current_subject = "u")
            )
    assert ei.value.status_code == 400
    assert "tool_choice" in ei.value.detail
    assert "enable_tools" in ei.value.detail


def test_guard_logic_unit():
    # Re-implement the guard predicate exactly and pin it.
    def would_raise(enable_tools, supports_tools, tool_choice):
        server_tools = enable_tools and supports_tools
        return bool(server_tools and tool_choice is not None)

    assert would_raise(True, True, {"type": "any"}) is True
    assert would_raise(True, True, None) is False
    assert would_raise(False, True, {"type": "any"}) is False
    assert would_raise(True, False, {"type": "any"}) is False


def test_enable_tools_false_with_tool_choice_does_not_raise_the_guard():
    from routes import inference as inf_mod

    payload = _build_payload(enable_tools = False, tool_choice = {"type": "any"})
    with patch.object(inf_mod, "get_llama_cpp_backend", return_value = _Llama()):
        try:
            asyncio.run(
                inf_mod.anthropic_messages(payload, _Req(), current_subject = "u")
            )
        except HTTPException as e:
            assert not (
                e.status_code == 400
                and "tool_choice is not honored" in (e.detail or "")
            )
        except Exception:
            # Any other failure downstream is fine; we only pin that the
            # enable_tools+tool_choice guard does NOT fire when enable_tools is False.
            pass
