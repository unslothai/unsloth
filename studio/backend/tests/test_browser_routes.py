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


def test_only_unsloth_hosts_get_the_studio_header(monkeypatch):
    calls = _fetch(monkeypatch, (None, b"<html></html>", "text/html"))
    _call()
    headers = calls[0]["host_headers"]
    for host in ("unsloth.ai", "www.unsloth.ai", "docs.UNSLOTH.ai", "unsloth.ai."):
        assert headers(host) == {"X-Unsloth-Studio": "1"}, host
    for host in ("example.com", "notunsloth.ai", "unsloth.ai.example.com", "unsloth.aix"):
        assert headers(host) == {}, host


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
    # The route returns 499 without waiting for its pool task, so wait on the fetch itself. Draining
    # the pool with no-ops was not enough: idle workers finish those while the fetch's own thread
    # has not yet recorded what it saw, and a loaded CI worker read an empty dict (KeyError).
    finished = threading.Event()

    def slow_fetch(url, **kwargs):
        try:
            seen["cancelled"] = kwargs["cancel_event"].wait(5)
        finally:
            finished.set()
        return "cancelled", "", ""

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", slow_fetch)
    with pytest.raises(HTTPException) as caught:
        _call(_Client(disconnected = True))
    assert caught.value.status_code == 499
    assert finished.wait(10), "the pooled fetch never ran"
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


@pytest.fixture(autouse = True)
def _empty_module_cache(monkeypatch):
    monkeypatch.setattr(browser_mod, "_MODULE_CACHE", browser_mod.OrderedDict())
    monkeypatch.setattr(browser_mod, "_module_cache_chars", 0)


def _modules(
    monkeypatch,
    scripts,
    allow_origin = None,
    cache_control = None,
    age = None,
):
    """Serve each URL in `scripts` as (error, body, content_type); return the URLs fetched."""
    fetched = []

    def fake_fetch(url, **kwargs):
        fetched.append(url)
        if allow_origin is not None:
            kwargs["meta_out"]["allow_origin"] = allow_origin
        kwargs["meta_out"].update(cache_control = cache_control, age = age)
        return scripts.get(url, ("Failed to fetch URL: HTTP 404", "", ""))

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    return fetched


def test_self_contained_module_scripts_are_inlined(monkeypatch):
    _modules(
        monkeypatch,
        {
            "https://example.com/js/app.js": (
                None,
                b'customElements.define("x-a", A);s="</script>"',
                "text/javascript",
            )
        },
    )
    page = '<p>hi</p><script type="module" defer src="js/app.js" crossorigin integrity="sha-x" data-k="a>b"></script><p>end</p>'
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert out == (
        '<p>hi</p><script type="module" defer data-k="a>b">'
        'customElements.define("x-a", A);s="<\\/script>"</script><p>end</p>'
    )


@pytest.mark.parametrize(
    "result",
    [
        (None, b'import {a} from "./b.js";a()', "text/javascript"),
        (None, b'const m = await import("./lazy.js")', "application/javascript"),
        (None, b'export * from "./c.js"', "text/javascript"),
        # Resolves against the module's own URL, which inlining would change to the page's.
        (None, b'new Worker(new URL("./w.js", import.meta.url))', "text/javascript"),
        (None, b'import /* webpackChunkName: "lazy" */ ("./chunk.js")', "text/javascript"),
        (None, b'import//x\n("./chunk.js")', "text/javascript"),
        (None, b'export { a } from /* re-export */ "./a.js"', "text/javascript"),
        (None, b'x="<!--";y="<script>"', "text/javascript"),
        (None, b"a()", "text/plain"),
        ("Failed to fetch URL: HTTP 404", "", ""),
    ],
)
def test_modules_that_cannot_be_inlined_keep_their_tag(monkeypatch, result):
    _modules(monkeypatch, {"https://example.com/m.js": result})
    page = '<script type="module" src="/m.js"></script>'
    assert browser_mod._inline_module_scripts(page, "https://example.com/page") == page


def test_only_live_https_module_tags_are_fetched(monkeypatch):
    fetched = _modules(monkeypatch, {})
    page = (
        '<script src="/classic.js"></script>'
        '<script type="module">inline()</script>'
        '<script type="module" src="http://example.com/plain.js"></script>'
        '<!-- <script type="module" src="/old.js"></script> -->'
        '<script type="module" src="/live.js"></script>'
    )
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    assert fetched == ["https://example.com/live.js"]


def test_a_fetched_page_gets_its_modules_inlined(monkeypatch):
    def fake_fetch(url, **kwargs):
        if url.endswith(".js"):
            assert kwargs["raw_bytes_max"] == browser_mod._MAX_MODULE_BYTES
            return None, b"ready()", "text/javascript"
        return None, b'<html><script type="module" src="/a.js"></script></html>', "text/html"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    html = json.loads(_call().body)["html"]
    assert '<script type="module">ready()</script>' in html


def _sri(algorithm, body):
    import base64
    import hashlib
    return f"{algorithm}-" + base64.b64encode(hashlib.new(algorithm, body).digest()).decode()


def test_an_inlined_module_still_has_its_integrity_checked(monkeypatch):
    _modules(monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")})
    tag = '<script type="module" src="/m.js" integrity="{}"></script>'
    good = tag.format(f"{_sri('sha256', b'other')} {_sri('sha384', b'ready()')}")
    bad = tag.format(f"{_sri('sha256', b'ready()')} {_sri('sha512', b'other')}")
    assert "ready()" in browser_mod._inline_module_scripts(good, "https://example.com/")
    assert browser_mod._inline_module_scripts(bad, "https://example.com/") == bad


def test_modules_any_origin_may_load_keep_their_tag(monkeypatch):
    _modules(
        monkeypatch,
        {"https://cdn.example/m.js": (None, b"ready()", "application/javascript")},
        allow_origin = "*",
    )
    page = '<script type="module" src="https://cdn.example/m.js"></script>'
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    # A wildcard doesn't cover a credentialed request.
    credentialed = page.replace("<script ", '<script crossorigin="use-credentials" ')
    out = browser_mod._inline_module_scripts(credentialed, "https://example.com/")
    assert out == '<script type="module">ready()</script>'


def test_a_module_is_fetched_once_for_many_pages(monkeypatch):
    fetched = _modules(
        monkeypatch,
        {"https://example.com/m.js": (None, b"ready()", "text/javascript")},
        cache_control = "public, max-age=31536000, immutable",
    )
    page = '<script type="module" src="/m.js"></script>'
    for _ in range(3):
        assert "ready()" in browser_mod._inline_module_scripts(page, "https://example.com/")
    assert fetched == ["https://example.com/m.js"]


@pytest.mark.parametrize(
    "cache_control, age",
    [
        (None, None),
        ("private, max-age=0", None),
        ("max-age=600, no-cache", None),
        ("no-store", None),
        ("public, max-age=300", "300"),
        ("private, max-age=600", None),
    ],
)
def test_a_module_the_host_says_to_recheck_is_fetched_again(monkeypatch, cache_control, age):
    fetched = _modules(
        monkeypatch,
        {"https://example.com/m.js": (None, b"ready()", "text/javascript")},
        cache_control = cache_control,
        age = age,
    )
    page = '<script type="module" src="/m.js"></script>'
    for _ in range(2):
        assert "ready()" in browser_mod._inline_module_scripts(page, "https://example.com/")
    assert fetched == ["https://example.com/m.js"] * 2


def test_a_module_is_kept_only_while_fresh():
    assert browser_mod._fresh_for("public, max-age=60", "20") == 40
    assert browser_mod._fresh_for("max-age=60, s-maxage=5", None) == 5
    assert browser_mod._fresh_for("max-age=99999999", None) == browser_mod._MODULE_CACHE_TTL_S


def test_inlined_modules_stay_within_the_page_limit(monkeypatch):
    _modules(
        monkeypatch,
        {
            "https://example.com/a.js": (None, b"a" * 60, "text/javascript"),
            "https://example.com/b.js": (None, b"b" * 30, "text/javascript"),
        },
    )
    page = '<script type="module" src="/a.js"></script><script type="module" src="/b.js"></script>'
    monkeypatch.setattr(browser_mod, "_MAX_BROWSER_HTML_BYTES", len(page) + 50)
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert (
        out
        == '<script type="module" src="/a.js"></script><script type="module">'
        + "b" * 30
        + "</script>"
    )


def test_a_module_fetch_that_raises_leaves_the_page_alone(monkeypatch):
    def boom(url, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", boom)
    page = '<p>x</p><script type="module" src="/m.js"></script>'
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page


def test_script_attributes_are_read_from_the_tag_not_from_values(monkeypatch):
    fetched = _modules(
        monkeypatch,
        {
            "https://example.com/main.js": (None, b"main()", "text/javascript"),
            "https://example.com/fallback.js": (None, b"fallback()", "text/javascript"),
            "https://example.com/classic.js": (None, b"classic()", "text/javascript"),
        },
    )
    page = (
        """<script type="module" onerror="this.src='/fallback.js'" src="/main.js"></script>"""
        '<script data-note="type=module" src="/classic.js"></script>'
        '<script type="text/javascript" type="module" src="/classic.js"></script>'
    )
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert out == (
        """<script type="module" onerror="this.src='/fallback.js'">main()</script>"""
        '<script data-note="type=module" src="/classic.js"></script>'
        '<script type="text/javascript" type="module" src="/classic.js"></script>'
    )
    assert fetched == ["https://example.com/main.js"]


@pytest.mark.parametrize(
    "wrap",
    [
        "<textarea>{}</textarea>",
        "<title>{}</title>",
        "<style>{}</style>",
        "<noscript>{}</noscript>",
        "<xmp>{}</xmp>",
        "<textarea>{}",
        "<plaintext>{}",
    ],
)
def test_a_module_tag_written_as_text_is_left_alone(monkeypatch, wrap):
    fetched = _modules(
        monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")}
    )
    page = wrap.format('<script type="module" src="/m.js"></script>')
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    assert fetched == []


def test_text_elements_in_script_code_dont_hide_later_modules(monkeypatch):
    _modules(monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")})
    page = (
        "<script>var css = '<style' + '>'; var t = '<textarea>';</script>"
        "<!-- <title> -->"
        '<textarea><script type="module" src="/m.js"></script></textarea>'
        '<script type="module" src="/m.js"></script>'
    )
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert out.endswith('</textarea><script type="module">ready()</script>')
    assert out.count("ready()") == 1


def test_the_page_limit_is_counted_in_bytes(monkeypatch):
    # 60 two-byte characters: counted as characters, the page would seem to have room.
    _modules(monkeypatch, {"https://example.com/m.js": (None, b"m" * 50, "text/javascript")})
    page = "é" * 60 + '<script type="module" src="/m.js"></script>'
    monkeypatch.setattr(browser_mod, "_MAX_BROWSER_HTML_BYTES", 200)
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    monkeypatch.setattr(browser_mod, "_MAX_BROWSER_HTML_BYTES", 220)
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert "m" * 50 in out and len(out.encode()) <= 220


@pytest.mark.parametrize(
    "page",
    [
        """<div data-example='<script type="module" src="/m.js"></script>'>x</div>""",
        """<a title="a > b" data-x='<script type=module src=/m.js></script>'>x</a>""",
        """<img alt="<script type='module' src='/m.js'></script>">""",
    ],
)
def test_a_module_tag_inside_an_attribute_value_is_left_alone(monkeypatch, page):
    fetched = _modules(
        monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")}
    )
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    assert fetched == []


def test_a_module_redirected_to_plain_http_keeps_its_tag(monkeypatch):
    def fake_fetch(url, **kwargs):
        kwargs["meta_out"]["url"] = "http://example.com/m.js"
        return None, b"ready()", "text/javascript"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    page = '<p class="a">x</p><script type="module" src="/m.js"></script>'
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page


@pytest.mark.parametrize(
    "page",
    [
        '<template><script type="module" src="/m.js"></script></template>',
        '<template><template></template><script type="module" src="/m.js"></script></template>',
        '<template id="t"><div><script type="module" src="/m.js"></script>',
    ],
)
def test_a_module_tag_in_a_template_is_left_alone(monkeypatch, page):
    fetched = _modules(
        monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")}
    )
    assert browser_mod._inline_module_scripts(page, "https://example.com/") == page
    assert fetched == []


def test_a_module_after_a_template_is_still_inlined(monkeypatch):
    _modules(monkeypatch, {"https://example.com/m.js": (None, b"ready()", "text/javascript")})
    page = '<template><p>x</p></template><script type="module" src="/m.js"></script>'
    out = browser_mod._inline_module_scripts(page, "https://example.com/")
    assert out == '<template><p>x</p></template><script type="module">ready()</script>'


def test_a_redirected_module_is_not_cached(monkeypatch):
    fetched = []

    def fake_fetch(url, **kwargs):
        fetched.append(url)
        kwargs["meta_out"].update(
            url = "https://example.com/v1.js", cache_control = "public, max-age=600", age = None
        )
        return None, b"ready()", "text/javascript"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    page = '<script type="module" src="/latest.js"></script>'
    for _ in range(2):
        assert "ready()" in browser_mod._inline_module_scripts(page, "https://example.com/")
    assert fetched == ["https://example.com/latest.js"] * 2


def _serve(
    monkeypatch,
    *parts,
    gap = 0,
):
    import socket

    from core.inference import tools

    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(10)

    def respond():
        with server:
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(65536)
                    try:
                        for part in parts:
                            conn.sendall(part)
                            time.sleep(gap)
                    except OSError:
                        pass

    threading.Thread(target = respond, daemon = True).start()
    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["127.0.0.1"])
    )
    return f"http://example.com:{server.getsockname()[1]}/missing"


def _error_response(
    code,
    body,
    headers = None,
):
    headers = {
        "Content-Type": "text/html; charset=utf-8",
        "Content-Length": str(len(body)),
        **(headers or {}),
    }
    head = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    return f"HTTP/1.1 {code} Not Found\r\n{head}\r\n".encode() + body


_SITE_404 = b"<html><head><title>Page not found</title></head><body>SITE_404_PAGE</body></html>"


def test_a_sites_own_error_page_is_shown_in_a_tab(monkeypatch):
    from core.inference import tools

    url = _serve(monkeypatch, _error_response(404, _SITE_404))
    response = _call(url = url, error_page = True)
    assert response.headers["x-unsloth-browser-kind"] == "html"
    payload = json.loads(response.body)
    assert payload["url"] == url
    assert "SITE_404_PAGE" in payload["html"]
    assert tools._fetch_page_text(url, timeout = 5) == "Failed to fetch URL: HTTP 404 Not Found"


@pytest.mark.parametrize("request_fields", [{}, {"max_bytes": 256 * 1024}])
def test_a_download_or_icon_of_an_error_page_fails(monkeypatch, request_fields):
    url = _serve(monkeypatch, _error_response(404, _SITE_404))
    with pytest.raises(HTTPException) as caught:
        _call(url = url, **request_fields)
    assert caught.value.status_code == 502
    assert caught.value.detail == "Failed to fetch URL: HTTP 404 Not Found"


def test_a_slow_error_page_stops_at_the_fetch_deadline(monkeypatch):
    monkeypatch.setattr(browser_mod, "_FETCH_TIMEOUT_S", 1.5)
    head = _error_response(404, b"<html>" + b"x" * 9)[:-9]
    url = _serve(monkeypatch, head, *[b"x"] * 9, gap = 1)
    started = time.monotonic()
    with pytest.raises(HTTPException) as caught:
        _call(url = url, error_page = True)
    assert caught.value.status_code == 502
    assert time.monotonic() - started < 1.8


@pytest.mark.parametrize(
    "code, body, headers, detail",
    [
        (404, b"", {}, "HTTP 404 Not Found"),
        (404, b"no such page", {"Content-Type": "text/plain"}, "HTTP 404 Not Found"),
        (
            403,
            b"<html>challenge</html>",
            {"Server": "cloudflare"},
            {"message": "Failed to fetch URL: HTTP 403 Not Found", "botCheck": True},
        ),
    ],
)
def test_error_pages_that_cannot_be_shown_stay_errors(monkeypatch, code, body, headers, detail):
    url = _serve(monkeypatch, _error_response(code, body, headers))
    with pytest.raises(HTTPException) as caught:
        _call(url = url, error_page = True)
    assert caught.value.status_code == 502
    if isinstance(detail, dict):
        assert caught.value.detail == detail
    else:
        assert caught.value.detail == f"Failed to fetch URL: {detail}"
