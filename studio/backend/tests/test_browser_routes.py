# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat's browser panel: pages come back through the SSRF-guarded fetcher and render in an
isolated, opaque-origin shell that a page cannot navigate out from under the proxy."""

import asyncio
import json
import threading
import time
from email.message import Message

import pytest
from fastapi import HTTPException

import routes.browser as browser_mod


def test_the_shell_is_isolated():
    import main

    csp = asyncio.run(browser_mod.browser_frame()).headers["content-security-policy"]
    assert "sandbox allow-scripts allow-forms;" in csp
    assert "allow-same-origin" not in csp
    for directive in ("object-src 'none';", "form-action 'none';", "upgrade-insecure-requests;"):
        assert directive in csp
    assert "frame-ancestors 'self'" in csp
    assert csp.rstrip().endswith("treat-as-public-address")
    # No plain-http requests of any kind, so pages can't hit local services.
    for part in csp.split(";"):
        name, *sources = part.split()
        if name.endswith("-src"):
            assert "http:" not in sources and "ws:" not in sources, name
    assert browser_mod.BROWSER_FRAME_PATH in main._FRAME_SHELL_PATHS
    assert "X-Unsloth-Browser-Kind" in browser_mod.EXPOSED_HEADERS


def test_the_print_shell_runs_only_its_own_script():
    import base64
    import hashlib

    import main

    response = asyncio.run(browser_mod.browser_print())
    csp = response.headers["content-security-policy"]
    digest = base64.b64encode(hashlib.sha256(browser_mod._PRINT_SCRIPT.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}';" in csp
    assert "unsafe-inline' https" not in csp.split("script-src")[1].split(";")[0]
    assert "sandbox allow-scripts allow-modals;" in csp
    assert "allow-same-origin" not in csp and "connect-src" not in csp
    assert csp.startswith("default-src 'none';")
    assert browser_mod._PRINT_SCRIPT in response.body.decode()
    for guard in (
        "event.source !== parent",
        'copy.querySelectorAll("script, iframe, frame, frameset, object, embed, meta[http-equiv]")',
        "if (/^on/i.test(attribute.name)) node.removeAttribute(attribute.name);",
    ):
        assert guard in browser_mod._PRINT_SCRIPT, guard
    assert browser_mod.BROWSER_PRINT_PATH in main._FRAME_SHELL_PATHS


def test_the_shell_guards():
    shell = browser_mod._FRAME_HTML
    for guard in (
        "if (event.source !== parent) return;",
        "event.source === page.contentWindow",
        'if (data && data.type === "unsloth:browser-command") page.contentWindow.postMessage(data, "*");',
        "page.srcdoc = inject(data.html, base + script)",
        'if (raw.startsWith("#"))',
        "navigator.userActivation?.isActive !== false",
        "if (title === lastTitle) return;",
        "titleObserver.observe(document.head",
        'post({ type: "upload" }); return;',
        'event.effectiveDirective !== "frame-src"',
        "if (raw.length > 1) followHash(raw);",
        "if (result === false) cancelled = true;",
    ):
        assert guard in shell, guard
    # The child is locked against navigation only after it is in place.
    assert shell.index("document.body.appendChild(page)") < shell.index(
        "lock.content = \"frame-src 'none'\""
    )
    assert "document.write(" not in shell
    assert "observe(document.documentElement" not in shell
    # Links and forms go through the proxy only when the page's own handlers leave them alone.
    assert shell.count("unlessCancelled(event, ") == 2
    assert "requestSubmit = " not in shell


def test_annotate_code_is_sent_only_when_annotating():
    shell, code = browser_mod._FRAME_HTML, browser_mod._ANNOTATE_JS
    # Not in the shell (so not in every page), installed once and only from the shell's own message.
    assert "const BLOCK =" in code and "const BLOCK =" not in shell
    assert 'const value = node.type === "password" ? "" : node.value;' in code
    assert code.rstrip().endswith("return { start, stop, forget, number };")
    assert 'if (data.command === "annotateInstall") install(data.code);' in shell
    assert 'if (annotation || typeof code !== "string"' in shell
    assert shell.index("const compile = Function;") < shell.index("const install = ")
    response = asyncio.run(browser_mod.browser_annotate_script())
    assert response.body.decode() == code
    assert response.media_type.startswith("text/javascript")
    assert response.headers["x-content-type-options"] == "nosniff"


def test_prepare_page_strips_what_would_escape_the_sandbox():
    page = (
        '<html><head><base href="/docs/">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'">'
        '<meta http-equiv="refresh" content="2; url=next.html">'
        '<meta charset="utf-8"><title>t</title></head><body></body></html>'
    )
    html, base, refresh = browser_mod._prepare_page(page, "https://example.com/a/b")
    assert "<base" not in html and "Content-Security-Policy" not in html and "refresh" not in html
    assert '<meta charset="utf-8">' in html
    assert base == "https://example.com/docs/"
    assert refresh == {"delay": 2.0, "url": "https://example.com/docs/next.html"}
    slow = '<meta http-equiv="refresh" content="600; url=/later">'
    assert browser_mod._prepare_page(slow, "https://example.com/")[2] is None
    # Addresses urljoin cannot parse are ignored, not a failed fetch.
    bad = '<base href="http://[bad"><meta http-equiv="refresh" content="1; url=http://[bad">'
    assert browser_mod._prepare_page(bad, "https://example.com/a")[1:] == (
        "https://example.com/a",
        None,
    )


def test_a_failed_fetch_of_an_unparseable_address_still_answers_502():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        browser_mod._build_response("http://[bad", "Invalid host", b"", "", {})
    assert caught.value.status_code == 502
    quoted = '<meta http-equiv="Content-Security-Policy" content="a<b>c"><p>x</p>'
    assert browser_mod._prepare_page(quoted, "https://example.com/")[0] == "<p>x</p>"


def test_prepare_page_is_linear_on_unclosed_tags():
    # Unclosed "<meta" runs were quadratic, holding the GIL (and so the whole backend) for minutes.
    start = time.monotonic()
    # 23 s quadratic, a few ms linear: room for a loaded CI box without losing the regression.
    browser_mod._prepare_page("<meta<base" * 80_000, "https://example.com/")
    assert time.monotonic() - start < 5


def test_decode_prefers_the_header_then_the_meta_charset():
    body = "café".encode("latin-1")
    assert browser_mod._decode_html(body, "latin-1") == "café"
    assert browser_mod._decode_html(b'<meta charset="latin-1">' + body, None).endswith("café")


def test_legacy_labels_and_unlabelled_pages_decode_as_windows_1252():
    quoted = "\u201cquoted\u201d \u2014 caf\u00e9".encode("cp1252")
    assert browser_mod._decode_html(quoted, "ISO-8859-1") == "\u201cquoted\u201d \u2014 caf\u00e9"
    assert browser_mod._decode_html(quoted, None) == "\u201cquoted\u201d \u2014 caf\u00e9"


def test_an_unquoted_refresh_is_followed():
    _, _, refresh = browser_mod._prepare_page(
        "<meta http-equiv=refresh content=0;url=/next><p>x</p>", "https://example.com/a"
    )
    assert refresh == {"delay": 0.0, "url": "https://example.com/next"}


def _fetch(
    monkeypatch,
    result,
    meta = None,
):
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(kwargs)
        kwargs["meta_out"].update(meta or {})
        return result

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    return calls


class _Client:
    def __init__(self, disconnected = False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


def _call(client = None, **request):
    request.setdefault("url", "https://example.com/")
    return asyncio.run(
        browser_mod.browser_fetch(
            browser_mod.BrowserFetchRequest(**request), client or _Client(), current_subject = "user"
        )
    )


def test_html_comes_back_as_json_with_its_final_url(monkeypatch):
    _fetch(
        monkeypatch,
        (None, b"<html><head><title>x</title></head></html>", "text/html"),
        {"url": "https://example.com/final", "charset": "utf-8"},
    )
    response = _call(url = "example.com")
    assert response.headers["x-unsloth-browser-kind"] == "html"
    payload = json.loads(response.body)
    assert payload["url"] == payload["base"] == "https://example.com/final"
    assert "<title>x</title>" in payload["html"]


def test_raw_text_is_transcoded_from_its_charset(monkeypatch):
    body = "a,café\n".encode("latin-1")
    _fetch(
        monkeypatch,
        (None, body, "text/csv"),
        {"url": "https://example.com/c.csv", "charset": "iso-8859-1"},
    )
    response = _call(url = "https://example.com/c.csv")
    assert response.body.decode("utf-8") == "a,café\n"
    assert response.headers["content-type"] == "text/csv; charset=utf-8"


def test_other_bodies_pass_through_untouched(monkeypatch):
    pdf = b"%PDF-1.7\n..."
    _fetch(monkeypatch, (None, pdf, "application/pdf"), {"url": "https://example.com/p.pdf"})
    response = _call(url = "https://example.com/p.pdf")
    assert response.headers["x-unsloth-browser-kind"] == "raw"
    assert response.headers["x-unsloth-browser-url"] == "https://example.com/p.pdf"
    assert response.body == pdf and response.media_type == "application/pdf"
    assert response.headers["content-security-policy"] == "sandbox"
    assert "x-unsloth-browser-filename" not in response.headers


def test_a_download_keeps_the_servers_file_name(monkeypatch):
    meta = {"url": "https://example.com/download?id=1", "filename": "../x/re\x07port 1.pdf"}
    _fetch(monkeypatch, (None, b"%PDF", "application/pdf"), meta)
    assert _call().headers["x-unsloth-browser-filename"] == "report%201.pdf"
    assert browser_mod.NAME_HEADER in browser_mod.EXPOSED_HEADERS


def test_fetch_arguments(monkeypatch):
    calls = _fetch(monkeypatch, (None, b"<html></html>", "text/html"))
    _call(method = "POST", body = "q=a+b", max_bytes = 4096)
    assert calls[0]["post_data"] == b"q=a+b"
    assert calls[0]["raw_bytes_max"] == 4096
    assert isinstance(calls[0]["cancel_event"], threading.Event)
    with pytest.raises(Exception):
        browser_mod.BrowserFetchRequest(
            url = "https://example.com/", max_bytes = browser_mod._MAX_BROWSER_FETCH_BYTES + 1
        )


@pytest.mark.parametrize(
    "result, detail",
    [
        (("Blocked: refusing to fetch non-public address 127.0.0.1.", "", ""), "non-public"),
        ((None, b"x" * (browser_mod._MAX_BROWSER_HTML_BYTES + 1), "text/html"), "byte limit"),
    ],
)
def test_refused_and_oversized_fetches_are_a_bad_gateway(monkeypatch, result, detail):
    _fetch(monkeypatch, result)
    with pytest.raises(HTTPException) as caught:
        _call()
    assert caught.value.status_code == 502 and detail in caught.value.detail


def test_fetches_run_in_their_own_pool(monkeypatch):
    threads = []

    def fake_fetch(url, **kwargs):
        threads.append(threading.current_thread().name)
        return None, b"<html></html>", "text/html"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    _call()
    # Not the default executor, which chat inference shares.
    assert threads[0].startswith("browser-fetch")


def test_a_closed_request_cancels_the_fetch(monkeypatch):
    seen = {}

    def slow_fetch(url, **kwargs):
        seen["cancelled"] = kwargs["cancel_event"].wait(5)
        return "cancelled", "", ""

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", slow_fetch)
    with pytest.raises(HTTPException) as caught:
        _call(_Client(disconnected = True))
    assert caught.value.status_code == 499
    for _ in range(browser_mod._FETCH_POOL._max_workers):
        browser_mod._FETCH_POOL.submit(lambda: None).result(5)
    assert seen["cancelled"] is True


def test_a_bot_check_is_reported_as_one(monkeypatch):
    _fetch(monkeypatch, ("Failed to fetch URL: HTTP 403 Forbidden", "", ""), {"bot_check": True})
    with pytest.raises(HTTPException) as caught:
        _call()
    assert caught.value.status_code == 502
    assert caught.value.detail == {
        "message": "Failed to fetch URL: HTTP 403 Forbidden",
        "botCheck": True,
    }


def test_bot_checks_are_told_apart_from_plain_refusals():
    from core.inference.tools import _is_bot_check

    def headers(**values):
        message = Message()
        for name, value in values.items():
            message[name.replace("_", "-")] = value
        return message

    assert _is_bot_check(403, headers(cf_mitigated = "challenge"))
    assert _is_bot_check(503, headers(cf_mitigated = "challenge"))
    assert _is_bot_check(403, headers(x_datadome = "protected"))
    assert _is_bot_check(403, headers(Server = "cloudflare"))
    # A rate limit or an outage behind Cloudflare is not a bot check.
    assert not _is_bot_check(429, headers(Server = "cloudflare"))
    assert not _is_bot_check(503, headers(Server = "cloudflare"))
    assert not _is_bot_check(404, headers(Server = "cloudflare"))
    assert not _is_bot_check(403, headers(Server = "nginx"))


def test_a_byte_order_mark_decides_the_encoding():
    page = "<p>caf\u00e9</p>"
    assert browser_mod._decode_html(page.encode("utf-16"), None) == page
    assert browser_mod._decode_html(b"\xef\xbb\xbf" + page.encode("utf-8"), "iso-8859-1") == page
