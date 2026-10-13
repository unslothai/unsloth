# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A page's requests to its own site (POST /api/browser/request): made as a browser makes a
same-origin request, through the SSRF-guarded fetcher, and only to the page's own origin."""

from __future__ import annotations

import base64
import io
import sys
import threading
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.browser as browser_mod
from core.inference import tools

PAGE = "https://site.test/models#top"


def _message(headers: dict) -> Message:
    message = Message()
    for name, value in headers.items():
        message[name] = value
    return message


class _Resp:
    reason = "OK"

    def __init__(self, status: int, body: bytes, headers: dict):
        self._status = status
        self._body = body
        self.headers = _message(headers)

    def getcode(self) -> int:
        return self._status

    def read(self, n: int | None = None) -> bytes:
        chunk, self._body = (self._body, b"") if n is None else (self._body[:n], self._body[n:])
        return chunk

    def close(self) -> None:
        pass


def _serve(monkeypatch, routes: dict) -> list[dict]:
    """``routes``: "host/path" -> (status, body, headers); returns what each hop sent."""
    sent = []

    class Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            key = f"{req.get_header('Host')}{urlsplit(req.full_url).path}"
            sent.append(
                {
                    "url": key,
                    "method": req.get_method(),
                    "data": req.data,
                    "headers": {name.lower(): value for name, value in req.header_items()},
                }
            )
            status, body, headers = routes[key]
            if status >= 300:
                raise HTTPError(req.full_url, status, "Moved", _message(headers), io.BytesIO(body))
            return _Resp(status, body, headers)

    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["93.184.216.34"])
    )
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *a, **k: Opener())
    return sent


def _request(**fields) -> dict:
    return browser_mod._page_request(
        browser_mod.BrowserPageRequest(page = PAGE, **fields), threading.Event()
    )


def test_a_same_origin_request_comes_back_as_the_site_sent_it(monkeypatch):
    sent = _serve(
        monkeypatch,
        {
            "site.test/models-json": (
                200,
                b'{"models": []}',
                {"Content-Type": "application/json", "X-Total": "3", "Set-Cookie": "s=1"},
            )
        },
    )
    answer = _request(
        url = "https://site.test/models-json?search=q", headers = [["X-Requested-With", "page"]]
    )
    assert answer["status"] == 200
    assert answer["redirected"] is False and answer["cors"] is False
    assert answer["url"] == "https://site.test/models-json?search=q"
    assert base64.b64decode(answer["body"]) == b'{"models": []}'
    headers = dict(answer["headers"])
    # Same-origin: every header but Set-Cookie is the page's to read.
    assert headers["X-Total"] == "3"
    assert "Set-Cookie" not in headers
    hop = sent[0]
    assert hop["method"] == "GET"
    assert hop["headers"]["x-requested-with"] == "page"
    # What a browser adds, the fragment left out of Referer; a GET has no Origin.
    assert hop["headers"]["referer"] == "https://site.test/models"
    assert hop["headers"]["sec-fetch-site"] == "same-origin"
    assert "origin" not in hop["headers"]


def test_an_error_status_is_the_pages_to_handle(monkeypatch):
    _serve(
        monkeypatch,
        {"site.test/api": (404, b'{"error": "none"}', {"Content-Type": "application/json"})},
    )
    answer = _request(url = "https://site.test/api")
    assert answer["status"] == 404
    assert base64.b64decode(answer["body"]) == b'{"error": "none"}'


def test_a_put_carries_its_body_type_and_origin(monkeypatch):
    sent = _serve(monkeypatch, {"site.test/api": (200, b"ok", {})})
    _request(
        url = "https://site.test/api",
        method = "put",
        headers = [["Content-Type", "application/json"]],
        body = base64.b64encode(b'{"a":1}').decode(),
    )
    hop = sent[0]
    assert hop["method"] == "PUT"
    assert hop["data"] == b'{"a":1}'
    assert hop["headers"]["origin"] == "https://site.test"
    assert hop["headers"]["content-type"] == "application/json"


def test_an_ipv6_pages_origin_is_bracketed(monkeypatch):
    sent = _serve(monkeypatch, {"[2606:4700:4700::1111]:8443/api": (200, b"ok", {})})
    browser_mod._page_request(
        browser_mod.BrowserPageRequest(
            page = "https://[2606:4700:4700::1111]:8443/",
            url = "https://[2606:4700:4700::1111]:8443/api",
            method = "POST",
        ),
        threading.Event(),
    )
    assert sent[0]["headers"]["origin"] == "https://[2606:4700:4700::1111]:8443"


@pytest.mark.parametrize(
    "url",
    [
        "https://other.test/api",
        "http://site.test/api",
        "https://site.test:8443/api",
        "ftp://site.test/x",
    ],
)
def test_only_the_pages_own_origin(monkeypatch, url):
    sent = _serve(monkeypatch, {})
    with pytest.raises(HTTPException) as raised:
        _request(url = url)
    assert raised.value.status_code == 403
    assert sent == []


def test_headers_no_page_could_send_are_dropped(monkeypatch):
    sent = _serve(monkeypatch, {"site.test/api": (200, b"", {})})
    _request(
        url = "https://site.test/api",
        headers = [
            ["Cookie", "a=1"],
            ["Host", "evil.test"],
            ["Origin", "https://evil.test"],
            ["Sec-Fetch-Site", "none"],
            ["Proxy-Authorization", "x"],
            ["Bad Name", "x"],
            ["X-Split", "a\r\nInjected: 1"],
            ["X-Ok", "1"],
        ],
    )
    headers = sent[0]["headers"]
    assert "cookie" not in headers
    assert "proxy-authorization" not in headers
    assert "x-split" not in headers and "injected" not in headers
    assert headers["host"] == "site.test"
    assert headers["sec-fetch-site"] == "same-origin"
    assert headers["x-ok"] == "1"


def test_a_repeated_header_is_combined_as_setrequestheader_does(monkeypatch):
    sent = _serve(monkeypatch, {"site.test/api": (200, b"", {})})
    _request(
        url = "https://site.test/api",
        headers = [["Accept", "text/html"], ["accept", "application/json"], ["X-Tag", "a"]],
    )
    headers = sent[0]["headers"]
    assert headers["accept"] == "text/html, application/json"
    assert headers["x-tag"] == "a"


def test_forbidden_methods_are_refused(monkeypatch):
    sent = _serve(monkeypatch, {})
    with pytest.raises(HTTPException) as raised:
        _request(url = "https://site.test/api", method = "TRACE")
    assert raised.value.status_code == 400
    assert sent == []


def test_a_redirect_to_another_site_needs_its_cors_consent(monkeypatch):
    sent = _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": "https://other.test/data"}),
            "other.test/data": (200, b"secret", {"Content-Type": "text/plain"}),
        },
    )
    with pytest.raises(HTTPException) as raised:
        _request(url = "https://site.test/go", headers = [["Authorization", "Bearer page"]])
    assert raised.value.status_code == 502
    # The page's own headers stayed with its own site.
    assert sent[0]["headers"]["authorization"] == "Bearer page"
    assert "authorization" not in sent[1]["headers"]


def test_a_redirect_to_a_site_that_allows_any_origin_is_read(monkeypatch):
    _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": "https://other.test/data"}),
            "other.test/data": (200, b"public", {"Access-Control-Allow-Origin": "*"}),
        },
    )
    answer = _request(url = "https://site.test/go")
    assert answer["url"] == "https://other.test/data"
    assert base64.b64decode(answer["body"]) == b"public"
    assert answer["redirected"] is True and answer["cors"] is True


@pytest.mark.parametrize("allow", [None, "null"])
def test_a_chain_back_to_the_pages_origin_stays_cors(monkeypatch, allow):
    # Back on the page's origin after another site, the request is tainted: Origin is "null".
    _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": "https://sso.test/bounce"}),
            "sso.test/bounce": (302, b"", {"Location": "https://site.test/home"}),
            "site.test/home": (
                200,
                b"home",
                {"Access-Control-Allow-Origin": allow} if allow else {},
            ),
        },
    )
    if allow is None:
        with pytest.raises(HTTPException):
            _request(url = "https://site.test/go")
        return
    answer = _request(url = "https://site.test/go")
    assert answer["url"] == "https://site.test/home"
    assert answer["redirected"] is True and answer["cors"] is True


@pytest.mark.parametrize(
    "expose, readable",
    [
        (None, {"Content-Type"}),
        ("X-Request-Id", {"Content-Type", "X-Request-Id"}),
        ("*", {"Content-Type", "X-Request-Id", "X-Internal-Token"}),
    ],
)
def test_a_response_from_another_site_shows_only_the_headers_it_exposes(
    monkeypatch, expose, readable
):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "text/plain",
        "X-Request-Id": "r1",
        "X-Internal-Token": "secret",
    }
    if expose:
        headers["Access-Control-Expose-Headers"] = expose
    _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": "https://cdn.test/data"}),
            "cdn.test/data": (200, b"public", headers),
        },
    )
    shown = {name for name, _value in _request(url = "https://site.test/go")["headers"]}
    assert shown - {"Access-Control-Allow-Origin", "Access-Control-Expose-Headers"} == readable


@pytest.mark.parametrize("allow, readable", [("https://site.test", True), ("null", False)])
def test_a_redirect_to_another_site_carries_the_pages_origin(monkeypatch, allow, readable):
    # A first hop off the page's origin is not tainted: Origin is the page's, which the site must allow.
    sent = _serve(
        monkeypatch,
        {
            "site.test/file": (302, b"", {"Location": "https://bucket.test/file"}),
            "bucket.test/file": (200, b"data", {"Access-Control-Allow-Origin": allow}),
        },
    )
    if readable:
        assert base64.b64decode(_request(url = "https://site.test/file")["body"]) == b"data"
    else:
        with pytest.raises(HTTPException):
            _request(url = "https://site.test/file")
    assert "origin" not in sent[0]["headers"]
    assert sent[1]["headers"]["origin"] == "https://site.test"


def test_a_second_hop_between_other_sites_sends_a_null_origin(monkeypatch):
    sent = _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": "https://a.test/go"}),
            "a.test/go": (302, b"", {"Location": "https://b.test/data"}),
            "b.test/data": (200, b"public", {"Access-Control-Allow-Origin": "null"}),
        },
    )
    assert base64.b64decode(_request(url = "https://site.test/go")["body"]) == b"public"
    assert [hop["headers"].get("origin") for hop in sent] == [None, "https://site.test", "null"]


def test_a_same_origin_redirect_still_shows_every_header(monkeypatch):
    _serve(
        monkeypatch,
        {
            "site.test/old": (301, b"", {"Location": "/new"}),
            "site.test/new": (200, b"", {"X-Internal-Token": "mine"}),
        },
    )
    answer = _request(url = "https://site.test/old")
    assert ["X-Internal-Token", "mine"] in answer["headers"]
    assert answer["redirected"] is True and answer["cors"] is False


def test_redirect_error_fails_at_the_redirect(monkeypatch):
    sent = _serve(
        monkeypatch,
        {"site.test/a": (302, b"", {"Location": "/b"}), "site.test/b": (200, b"", {})},
    )
    with pytest.raises(HTTPException) as raised:
        _request(url = "https://site.test/a", redirect = "error")
    assert raised.value.status_code == 502
    assert [hop["url"] for hop in sent] == ["site.test/a"]


def test_redirect_manual_stops_at_an_opaque_redirect(monkeypatch):
    sent = _serve(
        monkeypatch,
        {
            "site.test/a": (302, b"", {"Location": "/b", "X-Secret": "1"}),
            "site.test/b": (200, b"", {}),
        },
    )
    answer = _request(url = "https://site.test/a", redirect = "manual")
    assert answer["opaqueRedirect"] is True
    assert answer["status"] == 0 and answer["headers"] == [] and answer["body"] == ""
    assert [hop["url"] for hop in sent] == ["site.test/a"]


@pytest.mark.parametrize("target", ["https://site.test:8443/data", "http://site.test/data"])
def test_the_pages_headers_stay_with_its_origin_not_just_its_host(monkeypatch, target):
    # Same host, another port or scheme: another origin, so Authorization is not forwarded.
    host_path = target.split("://", 1)[1]
    sent = _serve(
        monkeypatch,
        {
            "site.test/go": (302, b"", {"Location": target}),
            host_path: (200, b"public", {"Access-Control-Allow-Origin": "*"}),
        },
    )
    _request(url = "https://site.test/go", headers = [["Authorization", "Bearer page"]])
    assert sent[0]["headers"]["authorization"] == "Bearer page"
    assert "authorization" not in sent[1]["headers"]
    assert "referer" not in sent[1]["headers"]


@pytest.mark.parametrize("status, method", [(307, "DELETE"), (308, "PUT"), (307, "POST")])
def test_a_write_redirected_to_another_site_is_never_sent_there(monkeypatch, status, method):
    sent = _serve(
        monkeypatch,
        {
            "site.test/api": (status, b"", {"Location": "https://other.test/api"}),
            "other.test/api": (200, b"", {"Access-Control-Allow-Origin": "*"}),
        },
    )
    with pytest.raises(HTTPException) as raised:
        _request(url = "https://site.test/api", method = method, body = base64.b64encode(b"x").decode())
    assert raised.value.status_code == 502
    assert [hop["url"] for hop in sent] == ["site.test/api"]


def test_a_post_redirected_as_a_get_may_go_to_another_site(monkeypatch):
    # 303 turns it into a bodiless GET, which a browser sends to another site without a preflight.
    sent = _serve(
        monkeypatch,
        {
            "site.test/upload": (303, b"", {"Location": "https://cdn.test/result"}),
            "cdn.test/result": (200, b"done", {"Access-Control-Allow-Origin": "*"}),
        },
    )
    answer = _request(
        url = "https://site.test/upload", method = "POST", body = base64.b64encode(b"x").decode()
    )
    assert base64.b64decode(answer["body"]) == b"done"
    assert sent[1]["method"] == "GET" and sent[1]["data"] is None


def test_a_write_redirected_as_a_get_drops_its_body_headers(monkeypatch):
    sent = _serve(
        monkeypatch,
        {
            "site.test/form": (303, b"", {"Location": "/done"}),
            "site.test/done": (200, b"ok", {}),
        },
    )
    _request(
        url = "https://site.test/form",
        method = "POST",
        headers = [["Content-Type", "application/json"], ["X-Csrf", "t"]],
        body = base64.b64encode(b"{}").decode(),
    )
    first, second = (hop["headers"] for hop in sent)
    assert first["content-type"] == "application/json" and first["origin"] == "https://site.test"
    assert sent[1]["method"] == "GET"
    assert "content-type" not in second and "origin" not in second
    assert second["x-csrf"] == "t"


@pytest.mark.parametrize(
    "status, method, kept",
    [
        (301, "POST", False),
        (302, "POST", False),
        (303, "PUT", False),
        (303, "HEAD", True),
        (301, "PUT", True),
        (307, "POST", True),
        (308, "DELETE", True),
    ],
)
def test_redirects_change_the_method_as_browsers_do(monkeypatch, status, method, kept):
    sent = _serve(
        monkeypatch,
        {"site.test/a": (status, b"", {"Location": "/b"}), "site.test/b": (200, b"", {})},
    )
    _request(url = "https://site.test/a", method = method, body = base64.b64encode(b"x").decode())
    second = sent[1]
    if kept:
        assert second["method"] == method
        assert second["data"] == (None if method == "HEAD" else b"x")
    else:
        assert second["method"] == "GET"
        assert second["data"] is None


def test_a_form_post_without_a_method_keeps_its_old_redirect(monkeypatch):
    # The panel's own form posts (no method given) still become a GET on 302.
    sent = _serve(
        monkeypatch,
        {"site.test/form": (302, b"", {"Location": "/done"}), "site.test/done": (200, b"", {})},
    )
    tools._fetch_url_raw("https://site.test/form", timeout = 5, post_data = b"q=1", raw_bytes_max = 100)
    assert [hop["method"] for hop in sent] == ["POST", "GET"]
    assert sent[0]["headers"]["content-type"] == "application/x-www-form-urlencoded"


def test_a_body_past_the_cap_is_refused():
    with pytest.raises(ValidationError):
        browser_mod.BrowserPageRequest(
            page = PAGE,
            url = "https://site.test/api",
            method = "POST",
            body = "A" * (4 * (browser_mod._MAX_PAGE_REQUEST_BODY // 3 + 2)),
        )
