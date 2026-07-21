# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client
from core.inference.mcp_client import (
    MAX_IMAGE_PAYLOAD_CHARS,
    MCP_IMAGES_SENTINEL,
    _flatten_result,
    call_tool_sync,
)
from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

PNG_B64 = "iVBORw0KGgoAAAANSUhEUg=="


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type = "text", text = value)


def _image(data: str = PNG_B64, mime: str = "image/png") -> SimpleNamespace:
    return SimpleNamespace(type = "image", data = data, mimeType = mime)


def _result(
    *blocks,
    is_error = False,
    structured = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content = list(blocks),
        is_error = is_error,
        structured_content = structured,
    )


def test_text_only_result_unchanged():
    assert _flatten_result(_result(_text("hello"))) == "hello"


def test_image_only_result_keeps_image_and_notes_model():
    flat = _flatten_result(_result(_image()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image attached; displayed to the user]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_text_plus_image_keeps_both():
    flat = _flatten_result(_result(_text("Took a screenshot"), _image()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "Took a screenshot\n[1 image attached; displayed to the user]"
    assert json.loads(payload)[0]["mimeType"] == "image/png"


def test_multiple_images_pluralized():
    flat = _flatten_result(_result(_image(), _image(mime = "image/jpeg")))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert "[2 images attached; displayed to the user]" in body
    assert [img["mimeType"] for img in json.loads(payload)] == [
        "image/png",
        "image/jpeg",
    ]


def test_strip_result_for_model_drops_image_payload():
    flat = _flatten_result(_result(_text("Took a screenshot"), _image()))
    stripped = strip_result_for_model(flat)
    assert stripped == "Took a screenshot\n[1 image attached; displayed to the user]"
    assert PNG_B64 not in stripped


def test_strip_preserves_literal_mcp_sentinel_in_text():
    # A tool that legitimately returns text containing the marker (e.g. reading
    # source/docs that quote it) must not be truncated: the suffix is not a
    # valid JSON image array.
    text = "before\n__MCP_IMAGES__: literal from source\nafter"
    assert strip_result_for_model(text) == text


def test_strip_preserves_non_image_json_after_marker():
    text = 'log line\n__MCP_IMAGES__:["not", "image", "dicts"]'
    assert strip_result_for_model(text) == text


def test_strip_removes_only_valid_terminal_envelope():
    text = (
        "Earlier mention: __MCP_IMAGES__: is documented here"
        "\n[1 image attached; displayed to the user]"
        '\n__MCP_IMAGES__:[{"data": "AAAA", "mimeType": "image/png"}]'
    )
    assert strip_result_for_model(text) == (
        "Earlier mention: __MCP_IMAGES__: is documented here"
        "\n[1 image attached; displayed to the user]"
    )


def test_strip_still_handles_images_and_rag_sentinels():
    assert strip_result_for_model("output\n__IMAGES__:['a.png']") == "output"
    assert strip_result_for_model("answer\n__RAG_SOURCES__:[{}]") == "answer"


def test_error_result_keeps_error_prefix_and_images():
    flat = _flatten_result(_result(_text("boom"), _image(), is_error = True))
    assert flat.startswith("Error: boom")
    assert is_tool_error(flat)
    assert MCP_IMAGES_SENTINEL in flat


def test_image_only_error_no_longer_reports_no_content():
    flat = _flatten_result(_result(_image(), is_error = True))
    assert flat.startswith("Error: [1 image attached")
    assert "tool returned no content" not in flat


def test_oversized_image_omitted_with_note():
    huge = "A" * (MAX_IMAGE_PAYLOAD_CHARS + 1)
    flat = _flatten_result(_result(_image(data = huge)))
    assert flat == "[1 image omitted (too large)]"
    assert MCP_IMAGES_SENTINEL not in flat


def test_oversized_budget_shared_across_images():
    big = "A" * (MAX_IMAGE_PAYLOAD_CHARS - 10)
    flat = _flatten_result(_result(_image(data = big), _image()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert "1 image attached" in body
    assert "1 image omitted (too large)" in body
    images = json.loads(payload)
    assert len(images) == 1 and images[0]["data"] == big


def test_non_image_binary_block_still_ignored():
    flat = _flatten_result(
        _result(SimpleNamespace(type = "audio", data = PNG_B64, mimeType = "audio/wav"))
    )
    assert flat == ""


def test_structured_content_fallback_still_used():
    flat = _flatten_result(_result(structured = {"ok": True}))
    assert flat == "{'ok': True}"


def test_call_tool_sync_passes_raise_on_error_false_and_keeps_error_images(monkeypatch):
    # Guards that call_tool_sync passes raise_on_error=False, so an is_error result
    # with image content reaches _flatten_result instead of FastMCP raising ToolError.
    seen = {}

    class _FakeClient:
        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            seen["raise_on_error"] = raise_on_error
            return _result(_text("boom"), _image(), is_error = True)

    @contextlib.asynccontextmanager
    async def _fake_client(url, headers, use_oauth):
        yield _FakeClient()

    monkeypatch.setattr(mcp_client, "_client", _fake_client)
    out = call_tool_sync("http://x", None, "take_screenshot", {})

    assert seen["raise_on_error"] is False
    assert out.startswith("Error: boom")
    assert MCP_IMAGES_SENTINEL in out
    assert is_tool_error(out)
