# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import contextlib
import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import mcp_client
from core.inference import mcp_images
from core.inference.mcp_client import (
    MAX_IMAGE_PAYLOAD_CHARS,
    MCP_IMAGES_SENTINEL,
    _flatten_result,
    call_tool_sync,
)
from core.inference.tool_loop_controller import is_tool_error, strip_result_for_model

PNG_B64 = "iVBORw0KGgoAAAANSUhEUg=="
WAV_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="


def _png_pixel(rgba: bytes) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"\x89PNG\r\n\x1a\n"
    raw += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress(b"\x00" + rgba))
    raw += chunk(b"IEND", b"")
    return base64.b64encode(raw).decode("ascii")


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type = "text", text = value)


def _image(data: str = PNG_B64, mime: str = "image/png") -> SimpleNamespace:
    return SimpleNamespace(type = "image", data = data, mimeType = mime)


def _audio(data: str = WAV_B64, mime: str = "audio/wav") -> SimpleNamespace:
    return SimpleNamespace(type = "audio", data = data, mimeType = mime)


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
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_text_plus_image_keeps_both():
    flat = _flatten_result(_result(_text("Took a screenshot"), _image()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "Took a screenshot\n[1 image returned]"
    assert json.loads(payload)[0]["mimeType"] == "image/png"


def test_multiple_images_pluralized():
    flat = _flatten_result(_result(_image(), _image(mime = "image/jpeg")))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert "[2 images returned]" in body
    assert [img["mimeType"] for img in json.loads(payload)] == ["image/png", "image/jpeg"]


def test_strip_result_for_model_drops_image_payload():
    flat = _flatten_result(_result(_text("Took a screenshot"), _image()))
    stripped = strip_result_for_model(flat)
    assert stripped == "Took a screenshot\n[1 image returned]"
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
        "\n[1 image returned]"
        '\n__MCP_IMAGES__:[{"data": "AAAA", "mimeType": "image/png"}]'
    )
    assert strip_result_for_model(text) == (
        "Earlier mention: __MCP_IMAGES__: is documented here\n[1 image returned]"
    )


def test_strip_still_handles_images_and_rag_sentinels():
    assert strip_result_for_model('output\n__IMAGES__:["a.png"]') == "output"
    assert strip_result_for_model("answer\n__RAG_SOURCES__:[{}]") == "answer"


def test_error_result_keeps_error_prefix_and_images():
    flat = _flatten_result(_result(_text("boom"), _image(), is_error = True))
    assert flat.startswith("Error: boom")
    assert is_tool_error(flat)
    assert MCP_IMAGES_SENTINEL in flat


def test_image_only_error_no_longer_reports_no_content():
    flat = _flatten_result(_result(_image(), is_error = True))
    assert flat.startswith("Error: [1 image returned")
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
    assert "1 image returned" in body
    assert "1 image omitted (too large)" in body
    images = json.loads(payload)
    assert len(images) == 1 and images[0]["data"] == big


def test_audio_only_result_notes_the_attachment():
    assert (
        _flatten_result(_result(_audio()))
        == "[audio attachment (audio/wav) not shown to the model]"
    )


def test_text_plus_audio_keeps_text_and_appends_note():
    flat = _flatten_result(_result(_text("Recorded 1s"), _audio()))
    assert flat == "Recorded 1s\n[audio attachment (audio/wav) not shown to the model]"


def test_audio_mirrored_in_structured_content_is_not_dumped():
    structured = {"content": [{"type": "audio", "data": WAV_B64, "mimeType": "audio/wav"}]}
    flat = _flatten_result(_result(_audio(), structured = structured))
    assert flat == "[audio attachment (audio/wav) not shown to the model]"
    assert WAV_B64 not in flat


def test_image_mirrored_in_structured_content_is_not_dumped():
    structured = {"content": [{"type": "image", "data": PNG_B64, "mimeType": "image/png"}]}
    flat = _flatten_result(_result(_image(), structured = structured))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]
    assert PNG_B64 not in strip_result_for_model(flat)
    huge = "A" * (MAX_IMAGE_PAYLOAD_CHARS + 1)
    structured = {"content": [{"type": "image", "data": huge, "mimeType": "image/png"}]}
    assert _flatten_result(_result(_image(data = huge), structured = structured)) == (
        "[1 image omitted (too large)]"
    )


def test_image_and_audio_share_one_note():
    flat = _flatten_result(_result(_image(), _audio()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == ("[1 image returned; audio attachment (audio/wav) not shown to the model]")
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_independent_structured_content_is_kept_beside_attachments():
    flat = _flatten_result(_result(_image(), structured = {"rows": 3, "max": 41.5}))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "{'rows': 3, 'max': 41.5}\n[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]
    assert _flatten_result(_result(_audio(), structured = {"duration_s": 1.0})) == (
        "{'duration_s': 1.0}\n[audio attachment (audio/wav) not shown to the model]"
    )


def test_zero_byte_attachments_are_noted():
    assert _flatten_result(_result(_audio(data = ""))) == (
        "[audio attachment (audio/wav) not shown to the model]"
    )
    assert (
        _flatten_result(
            _result(_blob_resource(data = "", mime = "text/csv", uri = "file:///out/empty.csv"))
        )
        == "[file attachment (text/csv) <file:///out/empty.csv> not shown to the model]"
    )


def test_partially_mirrored_structured_content_keeps_its_own_fields():
    structured = {
        "content": [{"type": "audio", "data": WAV_B64, "mimeType": "audio/wav"}],
        "duration_s": 1.0,
        "transcript": "hello",
    }
    flat = _flatten_result(_result(_audio(), structured = structured))
    assert flat == (
        "{'duration_s': 1.0, 'transcript': 'hello'}\n"
        "[audio attachment (audio/wav) not shown to the model]"
    )
    blob = _blob_resource(mime = "application/pdf", uri = "file:///out/report.pdf")
    mirrored = {
        "content": [
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///out/report.pdf",
                    "mimeType": "application/pdf",
                    "blob": PNG_B64,
                },
            }
        ]
    }
    assert _flatten_result(_result(blob, structured = mirrored)) == (
        "[file attachment (application/pdf) <file:///out/report.pdf> not shown to the model]"
    )
    wrapped = {
        "type": "success",
        "content": [{"type": "audio", "data": WAV_B64, "mimeType": "audio/wav"}],
    }
    assert _flatten_result(_result(_audio(), structured = wrapped)) == (
        "{'type': 'success'}\n[audio attachment (audio/wav) not shown to the model]"
    )
    nested = {"type": "success", "attachment": {"type": "audio", "data": WAV_B64}}
    assert _flatten_result(_result(_audio(), structured = nested)) == (
        "{'type': 'success'}\n[audio attachment (audio/wav) not shown to the model]"
    )
    colocated = {"type": "audio", "data": WAV_B64, "mimeType": "audio/wav", "transcript": "hello"}
    assert _flatten_result(_result(_audio(), structured = colocated)) == (
        "{'transcript': 'hello'}\n[audio attachment (audio/wav) not shown to the model]"
    )
    inner = {
        "content": [
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///out/report.pdf",
                    "mimeType": "application/pdf",
                    "blob": PNG_B64,
                    "pages": 12,
                },
            }
        ]
    }
    assert _flatten_result(_result(blob, structured = inner)) == (
        "{'content': [{'resource': {'pages': 12}}]}\n"
        "[file attachment (application/pdf) <file:///out/report.pdf> not shown to the model]"
    )


def test_audio_only_error_keeps_error_prefix():
    flat = _flatten_result(_result(_audio(), is_error = True))
    assert flat == "Error: [audio attachment (audio/wav) not shown to the model]"
    assert is_tool_error(flat)


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
        mcp_client.close_mcp_sessions()

    assert seen["raise_on_error"] is False
    assert out.startswith("Error: boom")
    assert MCP_IMAGES_SENTINEL in out
    assert is_tool_error(out)


def test_embedded_resource_image_is_rendered():
    flat = _flatten_result(_result(_blob_resource()))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_embedded_resource_image_shares_budget_with_image_content():
    flat = _flatten_result(_result(_text("rendered"), _image(), _blob_resource(mime = "image/webp")))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "rendered\n[2 images returned]"
    assert [img["mimeType"] for img in json.loads(payload)] == ["image/png", "image/webp"]


def test_oversized_embedded_resource_image_omitted():
    huge = "A" * (MAX_IMAGE_PAYLOAD_CHARS + 1)
    assert _flatten_result(_result(_blob_resource(data = huge))) == "[1 image omitted (too large)]"


def test_embedded_text_resource_contributes_its_text():
    assert (
        _flatten_result(_result(_text_resource("saved to /out/gen.png"))) == "saved to /out/gen.png"
    )


def test_embedded_non_image_blob_notes_type_and_uri():
    flat = _flatten_result(
        _result(_blob_resource(mime = "application/pdf", uri = "file:///out/report.pdf"))
    )
    assert (
        flat
        == "[file attachment (application/pdf) <file:///out/report.pdf> not shown to the model]"
    )
    assert PNG_B64 not in flat


def test_embedded_blobs_are_noted_one_by_one():
    flat = _flatten_result(
        _result(
            _blob_resource(mime = "text/csv", uri = "file:///out/table.csv"),
            _blob_resource(mime = "application/zip", uri = "file:///out/archive.zip"),
        )
    )
    assert flat == (
        "[file attachment (text/csv) <file:///out/table.csv> not shown to the model; "
        "file attachment (application/zip) <file:///out/archive.zip> not shown to the model]"
    )


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
    assert body == "[1 image returned]"
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
    assert _flatten_result(_result(_blob_resource(mime = None, uri = "file:///out/report.pdf"))) == (
        "[file attachment <file:///out/report.pdf> not shown to the model]"
    )


def test_non_image_application_types_are_noted_not_rendered():
    for mime in ("application/pdf", "application/octet-stream", "application/json"):
        assert _flatten_result(_result(_blob_resource(mime = mime))) == (
            f"[file attachment ({mime}) <file:///out/gen.png> not shown to the model]"
        ), mime


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
        assert _flatten_result(_result(_blob_resource(mime = None, uri = uri))) == (
            f"[file attachment <{uri}> not shown to the model]"
        ), uri


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
        == "[file attachment not shown to the model]"
    )


def test_a_bare_host_is_not_a_file_name():
    # urlsplit puts gen.png in netloc, not path; 3.10 guessed image/png from it
    assert _flatten_result(_result(_blob_resource(mime = None, uri = "resource://gen.png"))) == (
        "[file attachment <resource://gen.png> not shown to the model]"
    )
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
        flat = _flatten_result(_result(_blob_resource(mime = mime)))
        assert MCP_IMAGES_SENTINEL not in flat and "image attached" not in flat, mime


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


def test_invalid_mcp_envelope_fails_closed_for_an_mcp_tool():
    # A result that claims images it cannot parse used to leak whole as tool text.
    huge = "A" * 40_000
    text = "log\n" + MCP_IMAGES_SENTINEL + "{oops: " + huge
    stripped = strip_result_for_model(text, "mcp__fs__read_media_file")
    assert stripped == mcp_images.MCP_IMAGE_PARSE_ERROR_TEXT
    assert huge not in stripped


def test_invalid_mcp_envelope_is_not_touched_without_mcp_provenance():
    # The gate is the mcp__ prefix the envelope is trusted on, never the marker alone.
    literal = "before\n__MCP_IMAGES__: literal from source\nafter"
    assert strip_result_for_model(literal) == literal
    assert strip_result_for_model(literal, "read_file") == literal
    bad_json = 'log\n__MCP_IMAGES__:["not", "image", "dicts"]'
    assert strip_result_for_model(bad_json, "web_search") == bad_json


def test_valid_mcp_envelope_is_still_stripped_for_an_mcp_tool():
    text = (
        "Took a screenshot\n[1 image returned]"
        '\n__MCP_IMAGES__:[{"data": "AAAA", "mimeType": "image/png"}]'
    )
    assert (
        strip_result_for_model(text, "mcp__fs__screenshot")
        == "Took a screenshot\n[1 image returned]"
    )


def test_cap_tool_text_bounds_any_tool_text_marker_or_not():
    short = "x" * (mcp_images.MAX_TOOL_TEXT_CHARS - 1)
    assert mcp_images.cap_tool_text(short) == short
    huge = "a" * (mcp_images.MAX_TOOL_TEXT_CHARS + 100)
    capped = mcp_images.cap_tool_text(huge)
    assert len(capped) <= mcp_images.MAX_TOOL_TEXT_CHARS
    assert capped.startswith("a" * 100)
    assert "truncated" in capped


def test_sanitize_stays_suffix_only_for_the_envelope_recovery_split():
    # tools._split_frontend_suffix subtracts the strip from the original to recover
    # the envelope, so sanitize must edit only a suffix.
    text = (
        "short\n[1 image returned]" '\n__MCP_IMAGES__:[{"data": "AAAA", "mimeType": "image/png"}]'
    )
    assert (
        mcp_images.sanitize_tool_text(text, "mcp__fs__read_media_file")
        == "short\n[1 image returned]"
    )
    assert text.startswith(mcp_images.sanitize_tool_text(text, "mcp__fs__read_media_file"))


def test_promote_history_fails_closed_on_an_invalid_mcp_envelope():
    huge = "A" * 30_000
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "mcp__fs__read_media_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "mcp__fs__read_media_file",
            "content": "head\n" + MCP_IMAGES_SENTINEL + "{oops: " + huge,
        },
    ]
    out = mcp_images.promote_history(messages, vision = False)
    tool = out[-1]
    assert tool["role"] == "tool"
    assert tool["content"] == mcp_images.MCP_IMAGE_PARSE_ERROR_TEXT
    assert huge not in tool["content"]


def test_promote_history_caps_oversized_replay_text():
    long_text = "L" * (mcp_images.MAX_TOOL_TEXT_CHARS + 50)
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "mcp__fs__read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "name": "mcp__fs__read", "content": long_text},
    ]
    out = mcp_images.promote_history(messages, vision = False)
    assert len(out[-1]["content"]) < len(long_text)
    assert len(out[-1]["content"]) <= mcp_images.MAX_TOOL_TEXT_CHARS


def test_promote_history_keeps_in_budget_text_byte_identical():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "mcp__fs__read", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "mcp__fs__read",
            "content": "plain text",
        },
    ]
    assert mcp_images.promote_history(messages, vision = False)[-1] is messages[-1]


def test_multi_turn_replay_re_attaches_both_envelopes():
    first, second = _png_pixel(b"\xde\x00\x00\xff"), _png_pixel(b"\x00\x00\xde\xff")
    envelope = lambda data: (
        "[1 image returned]\n"
        + MCP_IMAGES_SENTINEL
        + json.dumps([{"data": data, "mimeType": "image/png"}])
    )
    call = lambda call_id: {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "mcp__fs__read_media_file", "arguments": "{}"},
            }
        ],
    }
    messages = [
        {"role": "user", "content": "look at this"},
        call("call_0"),
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "mcp__fs__read_media_file",
            "content": envelope(first),
        },
        {"role": "user", "content": "and this"},
        call("call_1"),
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "mcp__fs__read_media_file",
            "content": envelope(second),
        },
    ]
    text_only = mcp_images.promote_history(messages, vision = False)
    assert [tool["content"] for tool in text_only if tool["role"] == "tool"] == [
        "[1 image returned]",
        "[1 image returned]",
    ]
    assert not any(first in str(message) or second in str(message) for message in text_only)
    promoted: list = []
    out = mcp_images.promote_history(messages, vision = True, promoted_out = promoted)
    urls = [part["image_url"]["url"] for part in promoted]
    assert len(urls) == 2
    assert all(url.startswith("data:image/png;base64,") for url in urls)
    assert urls[0] != urls[1]
    for message in out:
        text = message.get("content")
        if isinstance(text, str):
            assert first not in text and second not in text
        else:
            assert not any(first in str(part) or second in str(part) for part in (text or []))


def test_dict_content_blocks_still_become_an_envelope():
    flat = _flatten_result(_result({"type": "image", "data": PNG_B64, "mimeType": "image/png"}))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_structured_content_only_image_is_recovered():
    structured = {"content": [{"type": "image", "data": PNG_B64, "mimeType": "image/png"}]}
    flat = _flatten_result(_result(structured = structured))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]
    assert PNG_B64 not in strip_result_for_model(flat)


def test_recovered_image_repeats_as_body_text_only_once():
    structured = {"content": [{"type": "image", "data": PNG_B64, "mimeType": "image/png"}]}
    flat = _flatten_result(_result(_text("Took a screenshot"), structured = structured))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "Took a screenshot\n[1 image returned]"
    assert PNG_B64 not in body
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_image_wrapper_obj_in_structured_content_is_recovered():
    structured = {"tool": "camera", "image": {"data": PNG_B64, "mimeType": "image/png"}}
    flat = _flatten_result(_result(structured = structured))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "{'tool': 'camera', 'image': {'mimeType': 'image/png'}}\n[1 image returned]"
    assert PNG_B64 not in body
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_data_uri_resource_image_is_recovered():
    block = SimpleNamespace(
        type = "resource",
        resource = SimpleNamespace(uri = f"data:image/png;base64,{PNG_B64}", blob = None),
    )
    flat = _flatten_result(_result(block))
    body, payload = flat.split("\n" + MCP_IMAGES_SENTINEL, 1)
    assert body == "[1 image returned]"
    assert json.loads(payload) == [{"data": PNG_B64, "mimeType": "image/png"}]


def test_non_image_structured_objects_are_not_recovered():
    svg = "<svg width='5' height='5' xmlns='http://www.w3.org/2000/svg'><rect/></svg>"
    structured = {"content": [{"type": "image", "data": svg, "mimeType": "image/png"}]}
    flat = _flatten_result(_result(structured = structured))
    assert MCP_IMAGES_SENTINEL not in flat
    assert svg in flat


def test_recovered_envelope_round_trips_through_split_images():
    structured = {"content": [{"type": "image", "data": PNG_B64, "mimeType": "image/png"}]}
    data_uri = SimpleNamespace(
        type = "resource",
        resource = SimpleNamespace(uri = f"data:image/png;base64,{PNG_B64}", blob = None),
    )
    result = _result(
        _text("Took a screenshot"),
        {"type": "image", "data": PNG_B64, "mimeType": "image/png"},
        data_uri,
        structured = structured,
    )
    flat = _flatten_result(result)
    head, images = mcp_images.split_images(flat)
    assert head == "Took a screenshot\n[2 images returned]"
    assert [img["data"] for img in images] == [PNG_B64, PNG_B64]
    assert strip_result_for_model(flat) == head


def test_frontend_backend_marker_and_cap_parity():
    assert "\n" + MCP_IMAGES_SENTINEL == "\n__MCP_IMAGES__:"
    assert mcp_images.SENTINEL == "__MCP_IMAGES__:"
    assert mcp_images.MAX_TOOL_TEXT_CHARS == 256_000
