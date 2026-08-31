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


def _blob_resource(
    data: str = PNG_B64,
    mime: str | None = "image/png",
    uri: str = "file:///out/gen.png",
) -> SimpleNamespace:
    return SimpleNamespace(
        type = "resource",
        resource = SimpleNamespace(uri = uri, mimeType = mime, blob = data),
    )


def _text_resource(text: str, mime: str = "text/plain") -> SimpleNamespace:
    return SimpleNamespace(
        type = "resource",
        resource = SimpleNamespace(uri = "file:///out/log.txt", mimeType = mime, text = text),
    )


def _resource_link(uri: str = "file:///out/gen.png", name = None) -> SimpleNamespace:
    return SimpleNamespace(type = "resource_link", uri = uri, name = name, mimeType = "image/png")


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
    assert [img["mimeType"] for img in json.loads(payload)] == ["image/png", "image/jpeg"]


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


def test_stdio_session_call_also_passes_raise_on_error_false(monkeypatch):
    seen = {}

    class _FakeStdioClient:
        def __init__(self):
            self.connected = False
            self.transport = SimpleNamespace(_is_session_dead = lambda: False)

        async def __aenter__(self):
            self.connected = True
            return self

        async def __aexit__(self, *exc):
            self.connected = False

        def is_connected(self):
            return self.connected

        async def call_tool(
            self,
            name,
            args,
            raise_on_error = True,
        ):
            seen["raise_on_error"] = raise_on_error
            return _result(_text("boom"), _image(), is_error = True)

    monkeypatch.setattr(
        mcp_client, "_client", lambda url, headers, use_oauth = False: _FakeStdioClient()
    )
    try:
        out = call_tool_sync(
            "npx fake-stdio-server", None, "take_screenshot", {}, scope = "s=p:t=thread1"
        )
    finally:
        mcp_client.close_stdio_sessions()

    assert seen["raise_on_error"] is False
    assert out.startswith("Error: boom")
    assert MCP_IMAGES_SENTINEL in out
    assert is_tool_error(out)


def test_embedded_resource_image_is_rendered():
    flat = _flatten_result(_result(_blob_resource()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image attached; displayed to the user]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_embedded_resource_image_shares_budget_with_image_content():
    flat = _flatten_result(_result(_text("rendered"), _image(), _blob_resource(mime = "image/webp")))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "rendered\n[2 images attached; displayed to the user]"
    assert [img["mimeType"] for img in json.loads(payload)] == ["image/png", "image/webp"]


def test_oversized_embedded_resource_image_omitted():
    huge = "A" * (MAX_IMAGE_PAYLOAD_CHARS + 1)
    assert _flatten_result(_result(_blob_resource(data = huge))) == "[1 image omitted (too large)]"


def test_embedded_text_resource_contributes_its_text():
    assert (
        _flatten_result(_result(_text_resource("saved to /out/gen.png"))) == "saved to /out/gen.png"
    )


def test_embedded_non_image_blob_still_ignored():
    assert _flatten_result(_result(_blob_resource(mime = "application/pdf"))) == ""


def test_resource_link_keeps_its_uri():
    assert _flatten_result(_result(_resource_link())) == "[resource: <file:///out/gen.png>]"
    assert _flatten_result(_result(_resource_link(name = "gen.png"))) == (
        "[resource: gen.png <file:///out/gen.png>]"
    )


def test_resource_link_does_not_displace_structured_content():
    flat = _flatten_result(_result(_resource_link(), structured = {"path": "/out/gen.png"}))
    assert flat == "{'path': '/out/gen.png'}\n[resource: <file:///out/gen.png>]"


def test_server_text_still_wins_over_structured_content():
    flat = _flatten_result(_result(_resource_link(), _text("done"), structured = {"path": "/x"}))
    assert flat == "[resource: <file:///out/gen.png>]\ndone"


def test_fastmcp_file_format_png_is_rendered():
    # fastmcp File(data=..., format="png") labels the blob application/png
    flat = _flatten_result(_result(_blob_resource(mime = "application/png")))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image attached; displayed to the user]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_application_image_subtypes_are_normalised():
    for mime, expected in (
        ("application/apng", "image/apng"),
        ("application/jpeg", "image/jpeg"),
        ("application/jpg", "image/jpeg"),
        ("application/webp", "image/webp"),
        ("application/GIF", "image/gif"),
        ("application/bmp", "image/bmp"),
        ("application/avif", "image/avif"),
        ("application/tif", "image/tiff"),
        ("application/tiff", "image/tiff"),
        ("application/ico", "image/vnd.microsoft.icon"),
        ("application/heic", "image/heic"),
        ("application/svg", "image/svg+xml"),
        ("application/svg+xml", "image/svg+xml"),
    ):
        flat = _flatten_result(_result(_blob_resource(mime = mime)))
        payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
        assert json.loads(payload) == [{"data": PNG_B64, "mimeType": expected}], mime


def test_blob_resource_without_mime_uses_uri_extension():
    flat = _flatten_result(_result(_blob_resource(mime = None)))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]
    assert _flatten_result(_result(_blob_resource(mime = None, uri = "file:///out/report.pdf"))) == ""


def test_non_image_application_types_stay_ignored():
    for mime in ("application/pdf", "application/octet-stream", "application/json"):
        assert _flatten_result(_result(_blob_resource(mime = mime))) == "", mime


def test_image_content_mime_is_passed_through_unchanged():
    flat = _flatten_result(_result(_image(mime = "image/png")))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_mixed_case_image_mime_is_matched():
    # media type names are case-insensitive
    for mime in ("IMAGE/PNG", "Image/Png", "image/PNG"):
        flat = _flatten_result(_result(_blob_resource(mime = mime)))
        payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
        assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}], mime


def test_mime_parameters_are_dropped_from_the_data_url_type():
    flat = _flatten_result(_result(_image(mime = "image/png; charset=binary")))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_uri_query_and_fragment_are_not_part_of_the_name():
    # mimetypes only stopped reading the query and fragment in 3.11.9/3.12.3/3.13
    # (CPython gh-117217); on older supported interpreters this dropped the image.
    for uri in (
        "file:///out/gen.png?download=1",
        "file:///out/gen.png#preview",
        "file:///out/gen.png?download=1#preview",
        "https://host/out/gen.png?sig=abc123",
    ):
        flat = _flatten_result(_result(_blob_resource(mime = None, uri = uri)))
        payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
        assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}], uri


def test_extension_only_in_the_query_is_not_an_image():
    # the same defect the other way: a query naming a .png made a non-image render
    for uri in ("file:///out/download?name=gen.png", "file:///out/download#gen.png"):
        assert _flatten_result(_result(_blob_resource(mime = None, uri = uri))) == "", uri


def test_data_uri_still_resolves_its_own_type():
    flat = _flatten_result(
        _result(_blob_resource(mime = None, uri = "data:image/png;base64,iVBORw0KGgo="))
    )
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]
    assert (
        _flatten_result(
            _result(_blob_resource(mime = None, uri = "data:application/pdf;base64,JVBERi0="))
        )
        == ""
    )


def test_a_bare_host_is_not_a_file_name():
    # urlsplit puts gen.png in netloc, not path; 3.10 guessed image/png from it
    assert _flatten_result(_result(_blob_resource(mime = None, uri = "resource://gen.png"))) == ""
    flat = _flatten_result(_result(_blob_resource(mime = None, uri = "resource://images/gen.png")))
    assert MCP_IMAGES_SENTINEL in flat


def test_malformed_image_types_never_reach_the_data_url():
    # anything that survives is interpolated into data:<type>;base64, by the frontend
    for mime in (
        "image/",
        "image//png",
        "image/*",
        "image/<script>",
        'image/png"',
        "image/png\nX-Injected: 1",
    ):
        assert _flatten_result(_result(_blob_resource(mime = mime))) == "", mime


def test_unusual_but_valid_image_types_are_kept():
    for mime in ("image/svg+xml", "image/vnd.microsoft.icon", "image/x-icon", "image/jp2"):
        flat = _flatten_result(_result(_blob_resource(mime = mime)))
        payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
        assert json.loads(payload) == [{"data": PNG_B64, "mimeType": mime}], mime


def test_snake_case_mime_attribute_is_read_too():
    # mcp 2.x renames mimeType to mime_type and keeps camelCase only as an alias
    block = SimpleNamespace(
        type = "resource",
        resource = SimpleNamespace(uri = "file:///out/gen.bin", mime_type = "image/png", blob = PNG_B64),
    )
    flat = _flatten_result(_result(block))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]

    direct = SimpleNamespace(type = "image", data = PNG_B64, mime_type = "image/jpeg")
    flat = _flatten_result(_result(direct))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/jpeg"}]


def test_registry_case_does_not_decide_whether_an_image_survives(monkeypatch):
    # windows answers .jxl with image/JXL, linux and macos image/jxl; same type per RFC 9110
    monkeypatch.setattr(
        mcp_client.mimetypes, "guess_type", lambda name, strict = True: ("image/JXL", None)
    )
    flat = _flatten_result(_result(_blob_resource(mime = "application/jxl")))
    payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)[1]
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/jxl"}]
