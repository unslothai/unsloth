# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat's browser panel: pages come back through the SSRF-guarded fetcher and render in an
opaque-origin shell. What matters is that the shell stays isolated, that a page cannot navigate
the frame out from under the proxy, and that non-HTML bodies pass through untouched."""

import asyncio
import json

import pytest
from fastapi import HTTPException

import routes.browser as browser_mod


def _shell_response():
    return asyncio.run(browser_mod.browser_frame())


def test_the_shell_is_sandboxed_without_same_origin():
    csp = _shell_response().headers["content-security-policy"]
    assert "sandbox allow-scripts allow-forms;" in csp
    assert "allow-same-origin" not in csp
    assert "object-src 'none';" in csp
    # Submits are routed by the injected script; a real one would leave the proxy.
    assert "form-action 'none';" in csp
    assert "frame-ancestors 'self'" in csp


def test_the_shell_does_not_inherit_studios_loopback_address():
    # Pages must not inherit the shell's loopback address.
    csp = _shell_response().headers["content-security-policy"]
    assert csp.rstrip().endswith("treat-as-public-address")
    # No plain-http requests of any kind, so pages can't hit local services.
    for part in csp.split(";"):
        name, *sources = part.split()
        if name.endswith("-src"):
            assert "http:" not in sources and "ws:" not in sources, name
    assert "upgrade-insecure-requests;" in csp


def test_the_shell_is_exempt_from_frame_denial():
    import main
    assert browser_mod.BROWSER_FRAME_PATH in main._FRAMEABLE_PATHS
    assert "X-Unsloth-Browser-Kind" in browser_mod.EXPOSED_HEADERS


def test_the_shell_only_obeys_its_parent():
    shell = browser_mod._FRAME_HTML
    assert "if (event.source !== parent || page) return;" in shell
    # The navigation script and <base> go ahead of the page's own markup.
    assert "inject(data.html, base + script)" in shell


def test_the_page_cannot_navigate_its_own_frame():
    shell = browser_mod._FRAME_HTML
    # The page runs in a srcdoc child; afterwards the shell refuses every navigation of it.
    assert "page.srcdoc = inject(" in shell
    assert shell.index("document.body.appendChild(page)") < shell.index("lock.content = \"frame-src 'none'\"")
    assert "document.write(" not in shell
    # Only the child's messages are relayed.
    assert "event.source === page.contentWindow" in shell


def test_prepare_page_strips_what_would_escape_the_sandbox():
    page = (
        '<html><head><base href="/docs/">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'">'
        '<meta http-equiv="refresh" content="2; url=next.html">'
        '<meta charset="utf-8"><title>t</title></head><body></body></html>'
    )
    html, base, refresh = browser_mod._prepare_page(page, "https://example.com/a/b")
    assert "<base" not in html
    assert "Content-Security-Policy" not in html
    assert "refresh" not in html
    assert '<meta charset="utf-8">' in html
    assert base == "https://example.com/docs/"
    assert refresh == {"delay": 2.0, "url": "https://example.com/docs/next.html"}


def test_a_slow_refresh_is_dropped_not_followed():
    _, _, refresh = browser_mod._prepare_page(
        '<meta http-equiv="refresh" content="600; url=/later">', "https://example.com/"
    )
    assert refresh is None


def test_decode_prefers_the_header_then_the_meta_charset():
    body = "café".encode("latin-1")
    assert browser_mod._decode_html(body, "latin-1") == "café"
    meta = b'<meta charset="latin-1">' + body
    assert browser_mod._decode_html(meta, None).endswith("café")


def _fetch(
    monkeypatch,
    result,
    meta = None,
):
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append((url, kwargs))
        if meta is not None:
            kwargs["meta_out"].update(meta)
        return result

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    return calls


class _Client:
    def __init__(self, disconnected = False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


def _call(request, client = None):
    return asyncio.run(
        browser_mod.browser_fetch(request, client or _Client(), current_subject = "user")
    )


def test_html_comes_back_as_json_with_its_final_url(monkeypatch):
    _fetch(
        monkeypatch,
        (None, b"<html><head><title>x</title></head></html>", "text/html"),
        {"url": "https://example.com/final", "charset": "utf-8"},
    )
    response = _call(browser_mod.BrowserFetchRequest(url = "example.com"))
    assert response.headers["x-unsloth-browser-kind"] == "html"
    payload = json.loads(response.body)
    assert payload["url"] == "https://example.com/final"
    assert payload["base"] == "https://example.com/final"
    assert "<title>x</title>" in payload["html"]


def test_other_bodies_pass_through_untouched(monkeypatch):
    pdf = b"%PDF-1.7\n..."
    _fetch(monkeypatch, (None, pdf, "application/pdf"), {"url": "https://example.com/p.pdf"})
    response = _call(browser_mod.BrowserFetchRequest(url = "https://example.com/p.pdf"))
    assert response.headers["x-unsloth-browser-kind"] == "raw"
    assert response.headers["x-unsloth-browser-url"] == "https://example.com/p.pdf"
    assert response.body == pdf
    assert response.media_type == "application/pdf"
    assert response.headers["content-security-policy"] == "sandbox"


def test_a_form_post_sends_its_body(monkeypatch):
    calls = _fetch(monkeypatch, (None, b"<html></html>", "text/html"))
    _call(browser_mod.BrowserFetchRequest(url = "https://example.com/s", method = "POST", body = "q=a+b"))
    assert calls[0][1]["post_data"] == b"q=a+b"


def test_a_refused_fetch_is_a_bad_gateway(monkeypatch):
    _fetch(monkeypatch, ("Blocked: refusing to fetch non-public address 127.0.0.1.", "", ""))
    with pytest.raises(HTTPException) as caught:
        _call(browser_mod.BrowserFetchRequest(url = "http://127.0.0.1:8888/"))
    assert caught.value.status_code == 502
    assert "non-public" in caught.value.detail


def test_fetches_run_in_their_own_pool_and_pass_a_cancel_event(monkeypatch):
    import threading

    threads = []

    def fake_fetch(url, **kwargs):
        threads.append(threading.current_thread().name)
        assert isinstance(kwargs["cancel_event"], threading.Event)
        return None, b"<html></html>", "text/html"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    _call(browser_mod.BrowserFetchRequest(url = "https://example.com/"))
    # Not the default executor, which chat inference shares.
    assert threads[0].startswith("browser-fetch")


def test_a_closed_request_cancels_the_fetch(monkeypatch):
    seen = {}

    def slow_fetch(url, **kwargs):
        seen["cancelled"] = kwargs["cancel_event"].wait(5)
        return "cancelled", "", ""

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", slow_fetch)
    with pytest.raises(HTTPException) as caught:
        _call(
            browser_mod.BrowserFetchRequest(url = "https://example.com/"), _Client(disconnected = True)
        )
    assert caught.value.status_code == 499
    # Wait for the pool thread to see the cancel.
    for _ in range(browser_mod._FETCH_POOL._max_workers):
        browser_mod._FETCH_POOL.submit(lambda: None).result(5)
    assert seen["cancelled"] is True


def test_the_shell_keeps_widget_links_and_popups_in_check():
    shell = browser_mod._FRAME_HTML
    # href="#" belongs to the page's own click handler; it must not reload the page.
    assert 'if (raw.startsWith("#"))' in shell
    # window.open without a click or key press is a popup, and is dropped.
    assert "navigator.userActivation?.isActive !== false" in shell
    # Title reports are deduplicated, and only <head> is observed, not every DOM change.
    assert "if (title === lastTitle) return;" in shell
    assert "titleObserver.observe(document.head" in shell
    assert "observe(document.documentElement" not in shell
