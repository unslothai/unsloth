# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""site_cookie_state reads the jar once and answers exactly as page_cookies + has_site_cookies."""

import http.cookiejar
import random
import time
from email.message import Message

import pytest

import routes.browser_cookies as cookies


@pytest.fixture(autouse = True)
def _empty():
    cookies._jars.clear()
    yield
    cookies._jars.clear()


def _headers(values):
    message = Message()
    for value in values:
        message["Set-Cookie"] = value
    return message


_PATHS = ["/", "/a", "/a/", "/a/b", "/a/b/", "/a/b/c", "/ab", "/x/y/z/"]
_SAME_SITE = ["", "; SameSite=Lax", "; SameSite=Strict", "; SameSite=None", "; SameSite=bogus"]


def _fill(rng: random.Random, subject: str, hosts: list[str], total: int, big: bool) -> None:
    per_host = max(1, total // len(hosts))
    for host in hosts:
        values = []
        for i in range(min(per_host, cookies._MAX_PER_SITE)):
            size = rng.randint(5000, 8000) if big and rng.random() < 0.7 else rng.randint(1, 40)
            value = f"c{i}={'v' * size}; Path={rng.choice(_PATHS)}"
            labels = host.split(".")
            if not host[0].isdigit() and len(labels) > 2 and rng.random() < 0.4:
                # Domain= the host itself (kept with a leading dot) or a parent (made host-only).
                value += "; Domain=" + rng.choice([host, "." + host, ".".join(labels[1:])])
            if rng.random() < 0.3:
                value += "; HttpOnly"
            if rng.random() < 0.3:
                value += "; Secure"
            if rng.random() < 0.3:
                value += f"; Max-Age={rng.randint(60, 10**6)}"
            value += rng.choice(_SAME_SITE)
            values.append(value)
        scheme = rng.choice(["http", "https"])
        cookies.store_cookies(subject, f"{scheme}://{host}/a/b/c", _headers(values))
    jar = cookies._jar(subject)
    now = int(time.time())
    for host in rng.sample(hosts, k = min(5, len(hosts))):
        for i in range(3):
            # Expired but not yet swept: return_ok must still refuse them.
            jar.set_cookie(
                http.cookiejar.Cookie(
                    0,
                    f"old{i}",
                    "x",
                    None,
                    False,
                    host,
                    False,
                    False,
                    rng.choice(_PATHS),
                    True,
                    False,
                    now - 100,
                    False,
                    None,
                    None,
                    {},
                )
            )


def _old_state(subject: str, url: str):
    """The pre-change implementation: two full passes, a Request per cookie."""

    def site():
        page = cookies._request(url)
        if page is None:
            return
        jar = cookies._jar(subject)
        for cookie in list(jar):
            at_path = cookies._request(f"{page.type}://{page.host}{cookie.path or '/'}")
            if at_path is not None and jar._policy.return_ok(cookie, at_path):
                yield cookie

    found, room = [], cookies._MAX_PAGE_COOKIE_CHARS
    for cookie in site():
        if cookies._http_only(cookie):
            continue
        deadline = "" if cookie.expires is None else str(int(cookie.expires) * 1000)
        entry = (f"{cookie.name}={cookie.path or '/'}", f"{deadline}|{cookie.value or ''}")
        room -= len(entry[0]) + len(entry[1])
        if room < 0:
            break
        found.append(entry)
    return [cookies._EPOCH, cookies._version], found, next(iter(site()), None) is not None


def _pages(hosts: list[str]) -> list[str]:
    pages = ["ftp://a0.example/", "not a url", "http:///nohost", "https://unknown.example/"]
    for host in hosts:
        for scheme in ("http", "https"):
            for path in ("/", "/a/b/c", "/ab", "/x/y/z/q"):
                pages.append(f"{scheme}://{host}{path}")
            pages.append(f"{scheme}://sub.{host}/a/")
    return pages


@pytest.mark.parametrize("seed", range(6))
def test_site_cookie_state_matches_the_two_pass_answer(seed):
    rng = random.Random(seed)
    hosts = [f"a{i}.example" for i in range(4)] + [f"www.b{i}.example" for i in range(3)]
    hosts += ["10.0.0.1", "10.0.0.2:8080", "localhost"]
    big = seed % 2 == 1
    _fill(rng, "acct", hosts, total = rng.choice([200, 1000, 3000]), big = big)
    assert len(list(cookies._jar("acct"))) <= cookies._MAX_COOKIES + 15
    capped = 0
    for url in _pages(hosts):
        new = cookies.site_cookie_state("acct", url)
        assert new == _old_state("acct", url), url
        assert new[1] == cookies.page_cookies("acct", url)
        assert new[2] == cookies.has_site_cookies("acct", url)
        capped += sum(len(a) + len(b) for a, b in new[1]) > cookies._MAX_PAGE_COOKIE_CHARS - 9000
    if big:
        assert capped, "the 64 KiB room cap never cut a page's list"


def test_one_request_per_distinct_cookie_path(monkeypatch):
    rng = random.Random(0)
    _fill(rng, "acct", ["a.example", "b.example"], total = 300, big = False)
    jar = cookies._jar("acct")
    paths = {cookie.path or "/" for cookie in jar}
    made = []
    real = cookies._request
    monkeypatch.setattr(cookies, "_request", lambda url: made.append(url) or real(url))
    cookies.site_cookie_state("acct", "https://a.example/a/b")
    assert len(made) == 1 + len(paths)
    assert len(list(jar)) > len(paths)
