# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared fetcher undoes gzip and deflate Content-Encoding. Servers send one unasked (python.org
does even for ``Accept-Encoding: identity``), and the browser panel, web_fetch and image search all
read through this fetcher."""

from __future__ import annotations

import gzip
import sys
import urllib.error
import zlib
from email.message import Message
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools

_PAGE = (
    b"<!doctype html><html><head><title>About</title></head><body><h1>About Python</h1>"
    + b"x" * 200_000
    + b"</body></html>"
)


class _FakeResp:
    def __init__(
        self,
        body: bytes,
        headers: dict,
        step: int = 4096,
    ):
        self._body = body
        self._pos = 0
        self._step = step
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value

    def read1(self, n: int = -1) -> bytes:
        chunk = self._body[self._pos : self._pos + min(n, self._step)]
        self._pos += len(chunk)
        return chunk

    read = read1


def _serve(monkeypatch, responses):
    """`responses`: a list, one per request, of a _FakeResp or an exception to raise. Returns the sent headers."""
    sent = []
    queue = list(responses)

    class _Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            sent.append(dict(req.header_items()))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["93.184.216.34"])
    )
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *a, **k: _Opener())
    return sent


def _raw_deflate(data: bytes) -> bytes:
    packer = zlib.compressobj(wbits = -15)
    return packer.compress(data) + packer.flush()


@pytest.mark.parametrize(
    "coding, body",
    [
        ("gzip", gzip.compress(_PAGE)),
        ("x-gzip", gzip.compress(_PAGE)),
        ("deflate", zlib.compress(_PAGE)),
        # Some servers send raw deflate for "deflate".
        ("deflate", _raw_deflate(_PAGE)),
        # Concatenated members are one body (RFC 1952 2.2).
        ("gzip", gzip.compress(_PAGE[:1000]) + gzip.compress(_PAGE[1000:])),
        # Padding after the last member is ignored, as browsers do.
        ("gzip", gzip.compress(_PAGE) + b"\0\0\0\0"),
        ("GZIP ", gzip.compress(_PAGE)),
    ],
    ids = [
        "gzip",
        "x-gzip",
        "zlib-deflate",
        "raw-deflate",
        "two-members",
        "padding",
        "case-and-space",
    ],
)
def test_a_coded_page_reaches_the_browser_panel_decoded(monkeypatch, coding, body):
    _serve(
        monkeypatch,
        [_FakeResp(body, {"Content-Type": "text/html", "Content-Encoding": coding}, step = 1000)],
    )
    error, raw, content_type = tools._fetch_url_raw(
        "https://www.python.org/about/", raw_bytes_max = 10_000_000
    )
    assert error is None
    assert raw == _PAGE
    assert content_type == "text/html"


def test_raw_deflate_output_zlib_held_back_at_the_end_is_not_truncation(monkeypatch):
    # The last wire bytes fill the 64 KiB output buffer; zlib keeps the rest with no input left.
    body = b"a" * 65_537
    _serve(
        monkeypatch,
        [
            _FakeResp(
                _raw_deflate(body), {"Content-Type": "text/plain", "Content-Encoding": "deflate"}
            )
        ],
    )
    error, raw, _ = tools._fetch_url_raw("https://example.com/a.txt", raw_bytes_max = 10_000_000)
    assert error is None
    assert raw == body


def test_web_fetch_reads_a_gzip_page_as_text(monkeypatch):
    _serve(
        monkeypatch,
        [
            _FakeResp(
                gzip.compress(_PAGE),
                {"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "gzip"},
            )
        ],
    )
    text = tools._fetch_page_text("https://www.python.org/about/", timeout = 5)
    assert "About Python" in text
    assert "binary content" not in text


def test_the_fetcher_asks_for_the_codings_it_can_undo(monkeypatch):
    sent = _serve(
        monkeypatch,
        [
            _FakeResp(_PAGE, {"Content-Type": "text/html"}),
            _FakeResp(_PAGE, {"Content-Type": "text/html"}),
        ],
    )
    assert tools._fetch_url_raw("https://example.com/", raw_bytes_max = 10_000_000)[0] is None
    assert sent[0]["Accept-encoding"] == "gzip, deflate"
    # A caller that asks for something else still gets its way.
    tools._fetch_url_raw(
        "https://example.com/",
        raw_bytes_max = 10_000_000,
        extra_headers = {"Accept-Encoding": "identity"},
    )
    assert sent[1]["Accept-encoding"] == "identity"


def test_an_empty_coded_body_is_an_empty_page(monkeypatch):
    _serve(monkeypatch, [_FakeResp(b"", {"Content-Type": "text/html", "Content-Encoding": "gzip"})])
    error, raw, _ = tools._fetch_url_raw("https://example.com/", raw_bytes_max = 1000)
    assert error is None and raw == b""


@pytest.mark.parametrize(
    "coding, body, message",
    [
        ("br", b"\x0b\x02\x80hello\x03", "unsupported Content-Encoding 'br'"),
        ("gzip", gzip.compress(_PAGE)[:-40], "truncated compressed response"),
        ("gzip", b"<html>not compressed</html>", "Failed to decode the compressed response"),
    ],
    ids = ["brotli", "truncated", "not-gzip"],
)
def test_a_body_that_cannot_be_decoded_is_an_error_not_garbage(monkeypatch, coding, body, message):
    _serve(
        monkeypatch, [_FakeResp(body, {"Content-Type": "text/html", "Content-Encoding": coding})]
    )
    error, raw, _ = tools._fetch_url_raw("https://example.com/", raw_bytes_max = 10_000_000)
    assert message in error
    assert raw == ""


def test_a_compressed_bomb_stops_at_the_decoded_limit(monkeypatch):
    bomb = gzip.compress(b"\0" * 64_000_000)
    _serve(
        monkeypatch,
        [
            _FakeResp(
                bomb,
                {"Content-Type": "application/octet-stream", "Content-Encoding": "gzip"},
                step = 65536,
            )
        ],
    )
    error, raw, _ = tools._fetch_url_raw("https://example.com/", raw_bytes_max = 1_000_000)
    assert error == "(content exceeds the 1000000 byte limit)"


def test_a_compressed_body_that_never_decodes_stops_at_the_wire_limit(monkeypatch):
    # A gzip header with a huge comment field: input keeps coming, output never does.
    padded = b"\x1f\x8b\x08\x10\x00\x00\x00\x00\x00\xff" + b"a" * 3_000_000
    _serve(
        monkeypatch,
        [_FakeResp(padded, {"Content-Type": "text/html", "Content-Encoding": "gzip"}, step = 65536)],
    )
    error, raw, _ = tools._fetch_url_raw("https://example.com/", raw_bytes_max = 1_000_000)
    assert error.startswith("(compressed content exceeds the ")
    assert raw == ""


def test_a_coded_error_page_is_decoded_too(monkeypatch):
    page = gzip.compress(b"<html><body>Not Found</body></html>")
    error_resp = _FakeResp(page, {"Content-Type": "text/html", "Content-Encoding": "gzip"})
    http_error = urllib.error.HTTPError(
        "https://example.com/x", 404, "Not Found", error_resp.headers, None
    )
    http_error.read1 = error_resp.read1
    http_error.read = error_resp.read1
    _serve(monkeypatch, [http_error])
    error, raw, _ = tools._fetch_url_raw(
        "https://example.com/x", raw_bytes_max = 1000, error_page = True
    )
    assert error.startswith("Failed to fetch URL: HTTP 404")
    assert raw == b"<html><body>Not Found</body></html>"
