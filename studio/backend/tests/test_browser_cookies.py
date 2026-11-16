# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sites' cookies for the panel's pages: a sign-in has to last past the page that made it."""

import asyncio
import base64
import json
import threading
import time
from email.message import Message

import pytest

import routes.browser as browser_mod
import routes.browser_cookies as cookies


@pytest.fixture(autouse = True)
def _empty():
    cookies._jars.clear()
    yield
    cookies._jars.clear()


def _headers(*set_cookies):
    message = Message()
    for value in set_cookies:
        message["Set-Cookie"] = value
    return message


class _Client:
    async def is_disconnected(self):
        return False


def _site(
    monkeypatch,
    hops,
    final_headers = (),
    body = b"<html></html>",
):
    """A fake fetcher: ``hops`` are (next url, redirect headers); returns what each hop was sent."""
    sent = []

    def fake_fetch(url, **kwargs):
        current = url
        for next_url, headers in hops:
            sent.append((current, kwargs["host_headers"](current.split("/")[2]).get("Cookie")))
            refused = kwargs["on_redirect"](next_url, "GET", False, headers)
            if refused:
                return refused, "", ""
            current = next_url
        sent.append((current, kwargs["host_headers"](current.split("/")[2]).get("Cookie")))
        kwargs["meta_out"].update(
            url = current,
            status = 200,
            headers = list(final_headers),
            allow_origin = dict(final_headers).get("Access-Control-Allow-Origin"),
        )
        return None, body, "text/html"

    monkeypatch.setattr(browser_mod, "_fetch_url_raw", fake_fetch)
    return sent


def _load(
    url,
    subject = "user",
    **request,
):
    response = asyncio.run(
        browser_mod.browser_fetch(
            browser_mod.BrowserFetchRequest(url = url, **request), _Client(), current_subject = subject
        )
    )
    return json.loads(response.body)


def test_a_sign_in_that_redirects_stays_signed_in_on_the_next_page(monkeypatch):
    sent = _site(
        monkeypatch,
        [("https://app.example.com/home", _headers("sid=abc; Path=/; HttpOnly; Secure"))],
    )
    page = _load("https://app.example.com/login", method = "POST", body = "user=a")
    # The redirect's cookie goes with the very next hop, and a script never sees an HttpOnly one.
    assert sent == [
        ("https://app.example.com/login", None),
        ("https://app.example.com/home", "sid=abc"),
    ]
    assert page["cookies"] == []

    sent = _site(monkeypatch, [])
    _load("https://app.example.com/account")
    assert sent == [("https://app.example.com/account", "sid=abc")]


def test_cookies_stay_with_their_site_scheme_and_account(monkeypatch):
    _site(
        monkeypatch,
        [],
        [
            ("Set-Cookie", "sid=abc; Secure"),
            ("Set-Cookie", "wide=1; Domain=example.com"),
            ("Set-Cookie", "tld=1; Domain=com"),
        ],
    )
    _load("https://app.example.com/")
    assert cookies.cookie_header("user", "https://app.example.com/x") == "sid=abc; wide=1"
    assert cookies.cookie_header("user", "http://app.example.com/x") == "wide=1"
    assert cookies.cookie_header("user", "https://www.example.com/") == "wide=1"
    assert cookies.cookie_header("user", "https://example.org/") is None
    assert cookies.cookie_header("other", "https://app.example.com/x") is None


def test_a_site_signing_out_and_clear_browsing_data_both_forget(monkeypatch):
    cookies.store_cookies(
        "user", "https://a.example.com/", [("Set-Cookie", "sid=abc"), ("Set-Cookie", "keep=1")]
    )
    cookies.store_cookies(
        "user", "https://a.example.com/logout", [("Set-Cookie", "sid=; Max-Age=0")]
    )
    assert cookies.cookie_header("user", "https://a.example.com/") == "keep=1"
    asyncio.run(browser_mod.browser_clear_cookies(current_subject = "user"))
    assert cookies.cookie_header("user", "https://a.example.com/") is None


def test_the_page_sees_script_readable_cookies_and_its_own_writes_reach_the_site(monkeypatch):
    _site(
        monkeypatch,
        [],
        [("Set-Cookie", "sid=abc; Path=/; HttpOnly"), ("Set-Cookie", "XSRF-TOKEN=tok; Path=/")],
    )
    page = _load("https://app.example.com/dash")
    assert page["cookies"] == [["XSRF-TOKEN=/", "|tok"]]

    def write(key, value):
        asyncio.run(
            browser_mod.browser_page_cookie(
                browser_mod.BrowserPageCookie(
                    page = "https://app.example.com/", key = key, value = value
                ),
                current_subject = "user",
            )
        )

    write("theme=/", "|dark")
    write("soon=/", f"{int(time.time() * 1000) + 60_000}|1")
    # No script can replace an HttpOnly cookie, or smuggle a second one in.
    write("sid=/", "|stolen")
    write("x=/", "|1; Domain=example.com")
    assert (
        cookies.cookie_header("user", "https://app.example.com/")
        == "sid=abc; XSRF-TOKEN=tok; theme=dark; soon=1"
    )
    write("theme=/", None)
    assert "theme" not in cookies.cookie_header("user", "https://app.example.com/")
    assert cookies.cookie_header("user", "https://www.example.com/") is None


def test_a_pages_request_carries_cookies_to_its_own_site_only(monkeypatch):
    cookies.store_cookies("user", "https://app.example.com/", [("Set-Cookie", "sid=abc")])
    cookies.store_cookies("user", "https://cdn.example.net/", [("Set-Cookie", "other=1")])
    sent = _site(
        monkeypatch,
        [("https://cdn.example.net/file", _headers("hop=1; Path=/"))],
        [("Set-Cookie", "tracker=1"), ("Access-Control-Allow-Origin", "*")],
        body = b"{}",
    )

    def request(url):
        return browser_mod._page_request(
            browser_mod.BrowserPageRequest(page = "https://app.example.com/dash", url = url),
            threading.Event(),
            "user",
        )

    answer = request("https://app.example.com/api/me")
    # Fetch's default credentials mode: nothing goes to, or is kept from, the other site.
    assert sent == [
        ("https://app.example.com/api/me", "sid=abc"),
        ("https://cdn.example.net/file", None),
    ]
    assert cookies.cookie_header("user", "https://cdn.example.net/") == "other=1"
    assert cookies.cookie_header("user", "https://app.example.com/") == "sid=abc; hop=1"
    assert base64.b64decode(answer["body"]) == b"{}"
    assert all(name.lower() != "set-cookie" for name, _ in answer["headers"])

    _site(monkeypatch, [], [("Set-Cookie", "csrf=t; Path=/")], body = b"{}")
    assert request("https://app.example.com/api/login")["cookies"] == [
        ("sid=/", "|abc"),
        ("hop=/", "|1"),
        ("csrf=/", "|t"),
    ]
    assert cookies.cookie_header("user", "https://app.example.com/") == "sid=abc; hop=1; csrf=t"


def test_the_store_is_bounded():
    for index in range(cookies._MAX_PER_SITE + 40):
        cookies.store_cookies("user", "https://many.example.com/", [("Set-Cookie", f"k{index}=1")])
    held = cookies.cookie_header("user", "https://many.example.com/").split("; ")
    assert len(held) == cookies._MAX_PER_SITE and held[-1] == f"k{cookies._MAX_PER_SITE + 39}=1"
    cookies.store_cookies("user", "https://big.example.com/", [("Set-Cookie", "big=" + "x" * 9000)])
    assert cookies.cookie_header("user", "https://big.example.com/") is None
    for index in range(cookies._MAX_ACCOUNTS + 5):
        cookies.store_cookies(f"account{index}", "https://a.example.com/", [("Set-Cookie", "s=1")])
    assert len(cookies._jars) == cookies._MAX_ACCOUNTS
