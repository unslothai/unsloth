# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Web fetch follows a <meta http-equiv="refresh"> redirect stub through the same hop gates as an HTTP redirect."""

from __future__ import annotations

import sys
from email.message import Message
from pathlib import Path
from urllib.parse import urlsplit

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools

REAL = b"<html><head><title>Real</title></head><body><main><h1>Linear</h1><p>REAL_PAGE_MARKER</p></main></body></html>"


def _stub(content: str) -> bytes:
    return (
        b'<!DOCTYPE html><meta charset="utf-8"><title>Redirecting&hellip;</title>'
        b'<meta http-equiv="refresh" content="' + content.encode() + b'">'
        b"<p>STUB_MARKER</p>"
    )


class _Resp:
    def __init__(self, body: bytes):
        self._body = body
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"

    def read(self, n: int | None = None) -> bytes:
        chunk, self._body = (self._body, b"") if n is None else (self._body[:n], self._body[n:])
        return chunk


def _serve(monkeypatch, pages: dict[str, bytes]) -> list[str]:
    requested = []

    def validate(host, port):
        if host == "example.com":
            return True, "", ["93.184.216.34"]
        return original(host, port)

    class Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            parts = urlsplit(req.full_url)
            url = f"{req.get_header('Host')}{parts.path}" + (
                f"?{parts.query}" if parts.query else ""
            )
            requested.append(url)
            return _Resp(pages[url])

    original = tools._validate_and_resolve_host
    monkeypatch.setattr(tools, "_validate_and_resolve_host", validate)
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *a, **k: Opener())
    return requested


def test_relative_meta_refresh_is_followed(monkeypatch):
    requested = _serve(
        monkeypatch,
        {
            "example.com/docs/stable/a/stub.html": _stub("0; url=../../2.9/a/real.html"),
            "example.com/docs/2.9/a/real.html": REAL,
        },
    )
    out = tools._fetch_page_text("https://example.com/docs/stable/a/stub.html", timeout = 5)
    assert "REAL_PAGE_MARKER" in out
    assert "STUB_MARKER" not in out
    assert requested == ["example.com/docs/stable/a/stub.html", "example.com/docs/2.9/a/real.html"]


@pytest.mark.parametrize(
    "location, query",
    [
        ("/p?a=1&section=2&region=eu", "a=1&section=2&region=eu"),
        ("/p?a=1&amp;copy=2", "a=1&copy=2"),
        ("/p?a=1&#38;b=2", "a=1&b=2"),
    ],
)
def test_refresh_url_character_references_decode_like_a_browser(monkeypatch, location, query):
    requested = _serve(
        monkeypatch,
        {"example.com/page": _stub(f"0; url={location}"), f"example.com/p?{query}": REAL},
    )
    assert "REAL_PAGE_MARKER" in tools._fetch_page_text("https://example.com/page", timeout = 5)
    assert requested == ["example.com/page", f"example.com/p?{query}"]


@pytest.mark.parametrize(
    "content",
    [
        "30",
        "0",
        "0; url=https://example.com/page#top",
        "600; url=/elsewhere",
        "0; url=javascript:alert(1)",
    ],
)
def test_reload_or_non_redirect_refresh_returns_the_page(monkeypatch, content):
    requested = _serve(monkeypatch, {"example.com/page": _stub(content)})
    out = tools._fetch_page_text("https://example.com/page", timeout = 5)
    assert "STUB_MARKER" in out
    assert requested == ["example.com/page"]


@pytest.mark.parametrize(
    "wrapper",
    [
        "<script>if (!ok) document.write('{meta}');</script>",
        "<template>{meta}</template>",
        "<textarea>{meta}</textarea>",
        "<style>/* {meta} */</style>",
    ],
)
def test_meta_refresh_inside_inert_markup_is_ignored(monkeypatch, wrapper):
    meta = '<meta http-equiv="refresh" content="0; url=/elsewhere">'
    body = f"<html><head>{wrapper.format(meta = meta)}</head><body><p>STUB_MARKER</p></body></html>"
    requested = _serve(monkeypatch, {"example.com/page": body.encode()})
    assert "STUB_MARKER" in tools._fetch_page_text("https://example.com/page", timeout = 5)
    assert requested == ["example.com/page"]


def test_meta_refresh_resolves_against_a_preceding_base_href(monkeypatch):
    body = (
        b'<head><base href="/docs/"><meta http-equiv="refresh" content="0; url=next.html"></head>'
    )
    requested = _serve(
        monkeypatch, {"example.com/a/b/stub.html": body, "example.com/docs/next.html": REAL}
    )
    assert "REAL_PAGE_MARKER" in tools._fetch_page_text(
        "https://example.com/a/b/stub.html", timeout = 5
    )
    assert requested == ["example.com/a/b/stub.html", "example.com/docs/next.html"]


def test_noscript_meta_refresh_is_ignored(monkeypatch):
    body = b'<html><head><noscript><meta http-equiv="refresh" content="0; url=/nojs"></noscript></head><body><p>STUB_MARKER</p></body></html>'
    requested = _serve(monkeypatch, {"example.com/page": body})
    assert "STUB_MARKER" in tools._fetch_page_text("https://example.com/page", timeout = 5)
    assert requested == ["example.com/page"]


def test_meta_refresh_loop_is_bounded_by_the_hop_limit(monkeypatch):
    requested = _serve(
        monkeypatch,
        {"example.com/a": _stub("0; url=/b"), "example.com/b": _stub("0; url=/a")},
    )
    out = tools._fetch_page_text("https://example.com/a", timeout = 5)
    assert out == "Failed to fetch URL: too many redirects."
    assert len(requested) == 5


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://localhost/",
    ],
)
def test_meta_refresh_to_private_host_is_blocked(monkeypatch, target):
    requested = _serve(monkeypatch, {"example.com/page": _stub(f"0; url={target}")})
    out = tools._fetch_page_text("https://example.com/page", timeout = 5)
    assert out.startswith("Blocked")
    assert requested == ["example.com/page"]


def test_meta_refresh_rechecks_website_policy(monkeypatch):
    requested = _serve(
        monkeypatch, {"example.com/page": _stub("0; url=https://other.example.org/x")}
    )
    out = tools._fetch_page_text(
        "https://example.com/page",
        timeout = 5,
        website_policy = {"allowedDomains": ["example.com"], "blockedDomains": []},
    )
    assert "Blocked: website access policy disallows other.example.org" in out
    assert requested == ["example.com/page"]
