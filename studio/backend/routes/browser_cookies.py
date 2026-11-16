# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cookies sites set on pages the panel fetches, kept per Studio account so a sign-in lasts.

The panel's pages run in an opaque-origin frame with no cookie store, and every page load goes
through this server, so the site's session lives here. In memory only: a restart signs every
site out, as closing a private window would.
"""

from __future__ import annotations

import http.cookiejar
import threading
import time
import urllib.request
from collections import OrderedDict
from email.message import Message
from typing import Iterable, Optional

_MAX_ACCOUNTS = 64
_MAX_COOKIES = 3000
_MAX_PER_SITE = 180
_MAX_SET_COOKIE_CHARS = 8192
_MAX_PAGE_COOKIE_CHARS = 64 * 1024

_lock = threading.Lock()
_jars: "OrderedDict[str, http.cookiejar.CookieJar]" = OrderedDict()


def _jar(subject: str) -> http.cookiejar.CookieJar:
    with _lock:
        jar = _jars.get(subject)
        if jar is None:
            jar = _jars[subject] = http.cookiejar.CookieJar()
            while len(_jars) > _MAX_ACCOUNTS:
                _jars.popitem(last = False)
        _jars.move_to_end(subject)
        return jar


def _request(url: str) -> Optional[urllib.request.Request]:
    try:
        request = urllib.request.Request(url)
    except ValueError:
        return None
    return request if request.type in ("http", "https") and request.host else None


def cookie_header(subject: str, url: str) -> Optional[str]:
    """The Cookie header a browser would send ``url``, or None."""
    request = _request(url)
    if request is None:
        return None
    _jar(subject).add_cookie_header(request)
    return request.get_header("Cookie")


class _Response:
    def __init__(self, values: list[str]):
        self._message = Message()
        for value in values:
            self._message["Set-Cookie"] = value

    def info(self) -> Message:
        return self._message


def _set_cookie_values(headers) -> list[str]:
    if headers is None:
        return []
    if hasattr(headers, "get_all"):
        values: Iterable = headers.get_all("Set-Cookie") or []
    else:
        values = (value for name, value in headers if str(name).lower() == "set-cookie")
    return [
        value
        for value in values
        if isinstance(value, str) and len(value) <= _MAX_SET_COOKIE_CHARS and "\n" not in value
    ]


def _trim(jar: http.cookiejar.CookieJar) -> None:
    """Oldest first: the jar iterates in insertion order within a domain, so this drops the earliest set."""
    cookies = list(jar)
    by_site: dict[str, int] = {}
    for cookie in cookies:
        by_site[cookie.domain] = by_site.get(cookie.domain, 0) + 1
    over = len(cookies) - _MAX_COOKIES
    for cookie in cookies:
        if by_site[cookie.domain] > _MAX_PER_SITE or over > 0:
            jar.clear(cookie.domain, cookie.path, cookie.name)
            by_site[cookie.domain] -= 1
            over -= 1


def store_cookies(subject: str, url: str, headers) -> None:
    """Keep what ``url``'s response set; ``headers`` is a message or a list of pairs."""
    values = _set_cookie_values(headers)
    request = _request(url) if values else None
    if request is None:
        return
    jar = _jar(subject)
    jar.extract_cookies(_Response(values), request)
    _trim(jar)


def _http_only(cookie: http.cookiejar.Cookie) -> bool:
    return any(name.lower() == "httponly" for name in cookie._rest)  # type: ignore[attr-defined]


def page_cookies(subject: str, url: str) -> list[tuple[str, str]]:
    """What ``document.cookie`` may see on ``url``'s site, at any path, in the frame's own form
    (browser.py): ``"<name>=<path>"`` to ``"<deadline ms>|<value>"``. HttpOnly ones stay here."""
    page = _request(url)
    if page is None:
        return []
    jar = _jar(subject)
    found: list[tuple[str, str]] = []
    room = _MAX_PAGE_COOKIE_CHARS
    for cookie in list(jar):
        if _http_only(cookie):
            continue
        at_path = _request(f"{page.type}://{page.host}{cookie.path or '/'}")
        if at_path is None or not jar._policy.return_ok(cookie, at_path):  # type: ignore[attr-defined]
            continue
        deadline = "" if cookie.expires is None else str(int(cookie.expires) * 1000)
        entry = (f"{cookie.name}={cookie.path or '/'}", f"{deadline}|{cookie.value or ''}")
        room -= len(entry[0]) + len(entry[1])
        if room < 0:
            break
        found.append(entry)
    return found


def set_page_cookie(subject: str, url: str, key: str, stored: Optional[str]) -> None:
    """A page's own ``document.cookie`` write, in the frame's form; ``stored`` None removes it.
    Host-only, and never over an HttpOnly cookie, which no script can touch."""
    name, _, path = key.partition("=")
    if not path.startswith("/") or any(char in name + path for char in ";,\r\n \t"):
        return
    request = _request(url)
    if request is None:
        return
    jar = _jar(subject)
    for cookie in list(jar):
        if cookie.name == name and cookie.path == path and _http_only(cookie):
            probe = _request(f"{request.type}://{request.host}{path}")
            if probe is not None and jar._policy.return_ok(cookie, probe):  # type: ignore[attr-defined]
                return
    if stored is None:
        header = f"{name}=; Path={path}; Max-Age=0"
    else:
        deadline, _, value = stored.partition("|")
        if any(char in value for char in ";\r\n"):
            return
        header = f"{name}={value}; Path={path}"
        if deadline.isdigit():
            header += f"; Max-Age={max(0, int(deadline) // 1000 - int(time.time()))}"
    store_cookies(subject, url, [("Set-Cookie", header)])


def clear_cookies(subject: str) -> None:
    with _lock:
        _jars.pop(subject, None)
