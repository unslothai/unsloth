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
        for next_url, headers, *method in hops:
            sent.append((current, kwargs["host_headers"](current.split("/")[2]).get("Cookie")))
            refused = kwargs["on_redirect"](next_url, (method or ["GET"])[0], False, headers)
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
            ("Set-Cookie", "wide=1; Domain=app.example.com"),
            ("Set-Cookie", "parent=1; Domain=example.com"),
            ("Set-Cookie", "tld=1; Domain=com"),
        ],
    )
    _load("https://app.example.com/")
    assert (
        cookies.cookie_header("user", "https://app.example.com/x")
        == "sid=abc; parent=1; tld=1; wide=1"
    )
    assert cookies.cookie_header("user", "http://app.example.com/x") == "parent=1; tld=1; wide=1"
    # A Domain wider than the host (no public suffix list here) stays on the host.
    assert cookies.cookie_header("user", "https://api.app.example.com/") == "wide=1"
    assert cookies.cookie_header("user", "https://www.example.com/") is None
    assert cookies.cookie_header("user", "https://example.org/") is None
    assert cookies.cookie_header("other", "https://app.example.com/x") is None


def test_a_navigation_from_another_site_follows_samesite(monkeypatch):
    cookies.store_cookies(
        "user",
        "https://bank.example/",
        [
            ("Set-Cookie", "strict=1; SameSite=Strict"),
            ("Set-Cookie", "lax=1; samesite=lax"),
            ("Set-Cookie", "plain=1"),
            ("Set-Cookie", "none=1; SameSite=None; Secure"),
        ],
    )
    every = "strict=1; lax=1; plain=1; none=1"
    sent = _site(monkeypatch, [])
    _load("https://bank.example/pay", method = "POST", body = "to=evil", initiator = "https://evil.test/")
    assert sent == [("https://bank.example/pay", "none=1")]
    _load("https://bank.example/", initiator = "https://evil.test/")
    assert sent[-1] == ("https://bank.example/", "lax=1; plain=1; none=1")
    _load("https://bank.example/pay", method = "POST", body = "to=me")
    _load(
        "https://bank.example/pay", method = "POST", body = "to=me", initiator = "https://BANK.example./a"
    )
    assert sent[-2:] == [("https://bank.example/pay", every)] * 2
    # Stricter than a browser: a subdomain is another site.
    _load(
        "https://bank.example/pay", method = "POST", body = "x", initiator = "https://www.bank.example/"
    )
    assert sent[-1] == ("https://bank.example/pay", "none=1")
    # So is the same host over http.
    _load("https://bank.example/pay", method = "POST", body = "x", initiator = "http://bank.example/")
    assert sent[-1] == ("https://bank.example/pay", "none=1")


def test_a_redirect_chain_that_left_the_site_stays_cross_site(monkeypatch):
    cookies.store_cookies(
        "user", "https://bank.example/", [("Set-Cookie", "strict=1; SameSite=Strict")]
    )
    sent = _site(
        monkeypatch,
        [("https://evil.test/bounce", None, "POST"), ("https://bank.example/pay", None, "POST")],
    )
    _load("https://bank.example/start", method = "POST", body = "x", initiator = "https://bank.example/")
    assert sent == [
        ("https://bank.example/start", "strict=1"),
        ("https://evil.test/bounce", None),
        ("https://bank.example/pay", None),
    ]
    sent = _site(
        monkeypatch, [("https://www.bank.example/", None), ("https://bank.example/", None)]
    )
    _load("https://bank.example/")
    assert sent[-1] == ("https://bank.example/", "strict=1")


def test_host_only_and_public_suffix_cookies_never_reach_another_site():
    cookies.store_cookies("user", "https://example.com/", [("Set-Cookie", "sid=1; Path=/")])
    assert cookies.cookie_header("user", "https://evil.example.com/") is None
    assert cookies.page_cookies("user", "https://evil.example.com/") == []
    assert cookies.cookie_header("user", "https://example.com/") == "sid=1"
    cookies.store_cookies(
        "user", "https://attacker.co.uk/", [("Set-Cookie", "sid=evil; Domain=co.uk; Path=/")]
    )
    assert cookies.cookie_header("user", "https://bank.co.uk/") is None


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


def test_a_response_deletes_only_cookies_its_own_host_could_set():
    bank = "https://bank.example/"
    cookies.store_cookies(
        "user", bank, _headers("sid=s; Domain=bank.example; Path=/; Secure; HttpOnly")
    )
    cookies.store_cookies("user", "https://www.example.com/", _headers("sid=w; Path=/"))
    cookies.store_cookies("user", "https://example.com/", _headers("sid=e; Path=/"))
    for url, deletion in (
        ("https://evil.example/", "sid=; Domain=bank.example; Path=/; Max-Age=0"),
        ("https://evil.example/", "sid=; Domain=example.com; Path=/; Max-Age=0"),
        ("https://app.example.com/", "sid=; Domain=example.com; Path=/; Max-Age=0"),
        ("https://app.example.com/", "sid=; Domain=www.example.com; Path=/; Max-Age=0"),
    ):
        cookies.store_cookies("user", url, _headers(deletion))
    assert cookies.cookie_header("user", bank) == "sid=s"
    assert cookies.cookie_header("user", "https://www.example.com/") == "sid=w"
    assert cookies.cookie_header("user", "https://example.com/") == "sid=e"


@pytest.mark.parametrize("set_domain", ["", "; Domain=app.example.com", "; Domain=example.com"])
@pytest.mark.parametrize(
    "del_domain", ["", "; Domain=app.example.com", "; Domain=.app.example.com"]
)
def test_a_site_signing_out_deletes_its_cookie_however_it_names_the_domain(set_domain, del_domain):
    site = "https://app.example.com/"
    cookies.store_cookies("user", "https://example.com/", _headers("sid=e; Path=/"))
    cookies.store_cookies("user", site, _headers(f"sid=1; Path=/; HttpOnly{set_domain}"))
    assert cookies.cookie_header("user", site) == "sid=1"
    cookies.store_cookies("user", site, _headers(f"sid=; Path=/; Max-Age=0{del_domain}"))
    assert cookies.cookie_header("user", site) is None
    assert cookies.cookie_header("user", "https://example.com/") == "sid=e"


def test_a_parent_domain_deletion_removes_the_host_only_cookie_it_would_have_set():
    site = "https://app.example.com/"
    cookies.store_cookies("user", site, _headers("sid=1; Path=/; Domain=example.com"))
    cookies.store_cookies("user", "https://example.com/", _headers("sid=e; Path=/"))
    cookies.store_cookies("user", site, _headers("sid=; Path=/; Max-Age=0; Domain=example.com"))
    assert cookies.cookie_header("user", site) is None
    assert cookies.cookie_header("user", "https://example.com/") == "sid=e"


def test_a_page_script_deletes_its_own_cookie_but_not_an_http_only_one():
    site = "https://app.example.com/"
    cookies.store_cookies("user", site, _headers("sid=1; Path=/; HttpOnly", "theme=dark; Path=/"))
    cookies.set_page_cookie("user", site, "sid=/", None)
    cookies.set_page_cookie("user", site, "theme=/", None)
    assert cookies.cookie_header("user", site) == "sid=1"


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
    write("sid=/", "|stolen")
    write("x=/", "|1; Domain=example.com")
    assert (
        cookies.cookie_header("user", "https://app.example.com/")
        == "sid=abc; XSRF-TOKEN=tok; theme=dark; soon=1"
    )
    write("theme=/", None)
    assert "theme" not in cookies.cookie_header("user", "https://app.example.com/")
    assert cookies.cookie_header("user", "https://www.example.com/") is None


def test_a_script_cookie_set_over_https_stays_off_http():
    cookies.set_page_cookie("user", "https://app.example.com/", "token=/", "|secret")
    cookies.set_page_cookie("user", "http://app.example.com/", "theme=/", "|dark")
    assert cookies.cookie_header("user", "https://app.example.com/") == "token=secret; theme=dark"
    assert cookies.cookie_header("user", "http://app.example.com/") == "theme=dark"
    assert cookies.page_cookies("user", "http://app.example.com/") == [("theme=/", "|dark")]
    cookies.set_page_cookie("user", "http://app.example.com/", "token=/", "|forged")
    cookies.set_page_cookie("user", "http://app.example.com/", "token=/", None)
    assert ("token=/", "|secret") in cookies.page_cookies("user", "https://app.example.com/")
    # A deeper path is refused too: it would be sent ahead of the Secure one.
    cookies.set_page_cookie("user", "http://app.example.com/", "token=/app", "|forged")
    assert (
        cookies.cookie_header("user", "https://app.example.com/app/x") == "token=secret; theme=dark"
    )


def test_an_http_response_leaves_a_secure_cookie_alone():
    site = "https://app.example.com/"
    cookies.store_cookies("user", site, _headers("sid=secret; HttpOnly; Secure; Path=/"))
    for forged in ("sid=attacker; Path=/", "sid=attacker; Path=/app", "sid=; Path=/; Max-Age=0"):
        cookies.store_cookies("user", "http://app.example.com/", _headers(forged))
        assert cookies.cookie_header("user", site + "app/x") == "sid=secret"
    cookies.store_cookies("user", "http://app.example.com/a", _headers("theme=dark; Path=/"))
    assert cookies.cookie_header("user", "http://app.example.com/") == "theme=dark"
    cookies.store_cookies("user", "https://app.example.com/", _headers("k=s; Secure; Path=/app"))
    cookies.store_cookies("user", "http://app.example.com/", _headers("k=h; Path=/"))
    assert cookies.cookie_header("user", "http://app.example.com/app") == "theme=dark; k=h"
    cookies.store_cookies("user", site, _headers("sid=; Path=/; Max-Age=0"))
    assert "sid=" not in cookies.cookie_header("user", site)


def test_secure_and_prefixed_cookies_need_https_and_their_prefix_rules():
    http, https = "http://app.example.com/", "https://app.example.com/a/"
    for value in ("s=1; Secure", "__Secure-a=1; Secure", "__Host-a=1; Secure; Path=/"):
        cookies.store_cookies("user", http, _headers(value))
    for value in (
        "__Secure-b=1",
        "__Host-b=1; Secure",
        "__Host-c=1; Secure; Path=/a",
        "__Host-d=1; Secure; Path=/; Domain=app.example.com",
        "__host-e=1; Path=/",
    ):
        cookies.store_cookies("user", https, _headers(value))
    assert cookies.cookie_header("user", https) is None
    cookies.store_cookies(
        "user", https, _headers("__Secure-f=1; Secure", "__Host-g=1; Secure; Path=/")
    )
    assert cookies.cookie_header("user", https) == "__Secure-f=1; __Host-g=1"


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

    sent = _site(monkeypatch, [], [("Set-Cookie", "anon=1; Path=/")], body = b"{}")
    browser_mod._page_request(
        browser_mod.BrowserPageRequest(
            page = "https://app.example.com/dash",
            url = "https://app.example.com/api/public",
            credentials = "omit",
        ),
        threading.Event(),
        "user",
    )
    assert sent == [("https://app.example.com/api/public", None)]
    assert "anon" not in cookies.cookie_header("user", "https://app.example.com/")


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


def test_responses_arriving_together_at_the_limit_all_store():
    for index in range(cookies._MAX_PER_SITE):
        cookies.store_cookies("user", "https://busy.example.com/", [("Set-Cookie", f"k{index}=1")])
    failed = []

    def respond(worker):
        try:
            for index in range(60):
                cookies.store_cookies(
                    "user", "https://busy.example.com/", [("Set-Cookie", f"w{worker}n{index}=1")]
                )
        except Exception as error:
            failed.append(error)

    workers = [threading.Thread(target = respond, args = (worker,)) for worker in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert failed == []
    held = cookies.cookie_header("user", "https://busy.example.com/").split("; ")
    assert len(held) == cookies._MAX_PER_SITE


def test_a_favicon_fetch_sends_and_keeps_no_cookies(monkeypatch):
    cookies.store_cookies(
        "user", "https://bank.example/", [("Set-Cookie", "strict=1; SameSite=Strict")]
    )
    sent = _site(monkeypatch, [], [("Set-Cookie", "icon=1")])
    _load("https://bank.example/favicon.ico", credentials = "omit")
    assert sent == [("https://bank.example/favicon.ico", None)]
    assert cookies.cookie_header("user", "https://bank.example/") == "strict=1"


def test_cookie_snapshots_are_ordered_by_store_changes(monkeypatch):
    _site(monkeypatch, [], [("Set-Cookie", "csrf=old")])
    older = _load("https://app.example.com/")["cookiesVersion"]
    _site(monkeypatch, [], [("Set-Cookie", "csrf=new")])
    newer = _load("https://app.example.com/")["cookiesVersion"]
    assert older[0] == newer[0] and older[1] < newer[1]
    cookies.clear_cookies("user")
    assert cookies.versioned_page_cookies("user", "https://app.example.com/")[0][1] > newer[1]
