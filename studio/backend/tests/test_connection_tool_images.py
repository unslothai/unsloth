# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from starlette.requests import Request

from core.inference import external_provider as ep_mod
from core.inference.mcp_images import SENTINEL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_auto_compaction import _proxy  # noqa: E402
from test_external_provider_sampling_over_the_wire import _Handler  # noqa: E402

_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class _CapabilityHandler(_Handler):
    show: dict | None = None
    props: dict | None = None
    probes: list[str] = []

    def _json(self, body: dict | None) -> None:
        raw = json.dumps(body if body is not None else {"error": "not found"}).encode()
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        type(self).probes.append(self.path)
        self._json(self.props if self.path.startswith("/props") else None)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/show":
            return super().do_POST()
        type(self).probes.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._json(self.show)


def _server(**attrs):
    return type("Capability", (_CapabilityHandler,), {"probes": [], **attrs})


@pytest.fixture(autouse = True)
def _nothing_remembered(monkeypatch):
    monkeypatch.setattr(ep_mod, "_served_vision", {}, raising = False)


_TEXT_ONLY = {
    "ollama": {"show": {"capabilities": ["completion", "tools", "thinking"]}},
    "llama_cpp": {"props": {"modalities": {"vision": False, "audio": False}}},
}
_VISION = {
    "ollama": {"show": {"capabilities": ["completion", "vision", "tools"]}},
    "llama_cpp": {"props": {"modalities": {"vision": True, "audio": False}}},
}
_SILENT = {
    "ollama": {"show": {"details": {"family": "qwen3"}}},
    "llama_cpp": {"props": {"default_generation_settings": {"n_ctx": 8192}}},
}


def _chat_with_a_viewed_image() -> list[dict]:
    call = {"id": "call_0", "type": "function"}
    call["function"] = {"name": "view_image", "arguments": '{"path": "chart.png"}'}
    envelope = json.dumps([{"data": _PNG_B64, "mimeType": "image/png"}])
    return [
        {"role": "user", "content": "make a bar chart and check it"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "view_image",
            "content": f"Opened image.\n{SENTINEL}{envelope}",
        },
        {"role": "assistant", "content": "The chart is saved."},
        {"role": "user", "content": "thanks! what is 2+2?"},
    ]


def _image_parts(sent: dict) -> list[dict]:
    return [
        part
        for message in sent["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]


def _tool_names(sent: dict) -> set[str]:
    return {tool["function"]["name"] for tool in sent.get("tools") or []}


@pytest.fixture
def connected(monkeypatch):
    async def is_disconnected(self) -> bool:
        return False

    monkeypatch.setattr(Request, "is_disconnected", is_disconnected)


@pytest.mark.parametrize("provider_type", ["ollama", "llama_cpp"])
def test_a_text_only_connection_model_gets_the_tool_result_without_its_image(provider_type):
    _, sent = _proxy(
        provider_type, _chat_with_a_viewed_image(), handler = _server(**_TEXT_ONLY[provider_type])
    )
    assert not _image_parts(sent)
    assert SENTINEL not in json.dumps(sent)
    [result] = [m for m in sent["messages"] if m["role"] == "tool"]
    assert result["content"] == "Opened image."
    assert sent["messages"][-1]["content"] == "thanks! what is 2+2?"


@pytest.mark.parametrize("provider_type", ["ollama", "llama_cpp"])
def test_a_vision_connection_model_still_gets_the_tool_image(provider_type):
    _, sent = _proxy(
        provider_type, _chat_with_a_viewed_image(), handler = _server(**_VISION[provider_type])
    )
    [image] = _image_parts(sent)
    assert image["image_url"]["url"].startswith("data:image/")
    assert SENTINEL not in json.dumps(sent)


@pytest.mark.parametrize("provider_type", ["ollama", "llama_cpp"])
@pytest.mark.parametrize("answer", ["silent", "missing"])
def test_a_server_that_does_not_report_vision_keeps_the_tool_image(provider_type, answer):
    handler = _server(**_SILENT[provider_type]) if answer == "silent" else _server()
    _, sent = _proxy(provider_type, _chat_with_a_viewed_image(), handler = handler)
    assert len(_image_parts(sent)) == 1


@pytest.mark.parametrize("provider_type", ["ollama", "llama_cpp"])
def test_view_image_is_offered_only_to_a_connection_model_that_reads_images(
    provider_type, connected
):
    tools = {"enable_tools": True, "enabled_tools": ["python", "view_image"]}
    tools["permission_mode"] = "off"
    question = [{"role": "user", "content": "plot the populations"}]
    _, text_only = _proxy(
        provider_type, question, handler = _server(**_TEXT_ONLY[provider_type]), **tools
    )
    _, vision = _proxy(provider_type, question, handler = _server(**_VISION[provider_type]), **tools)
    assert _tool_names(text_only) == {"python"}
    assert _tool_names(vision) == {"python", "view_image"}


def test_a_chat_without_tools_or_tool_images_does_not_ask_the_server():
    handler = _server(**_TEXT_ONLY["ollama"])
    _proxy("ollama", [{"role": "user", "content": "hello"}], handler = handler)
    assert handler.probes == []


def test_the_model_asked_about_is_the_one_chatting():
    class Show(_CapabilityHandler):
        asked: list[dict] = []

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/show":
                return super().do_POST()
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            type(self).asked.append(json.loads(raw))
            self._json({"capabilities": ["completion"]})

    _proxy("ollama", _chat_with_a_viewed_image(), handler = Show, model = "qwen3:8b")
    assert Show.asked == [{"model": "qwen3:8b"}]
