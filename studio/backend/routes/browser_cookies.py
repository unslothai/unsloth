# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-account, in-memory cookies for the panel's pages (their opaque-origin frames have none)."""

from __future__ import annotations

import http.cookiejar
import secrets
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
_jars: "OrderedDict[str, _Jar]" = OrderedDict()
# Orders page_cookies snapshots; the epoch tells a restarted process's count from the old one.
_EPOCH = secrets.token_hex(8)
_version = 0


def _changed() -> None:
    global _version
    with _lock:
        _version += 1


class _Policy(http.cookiejar.DefaultCookiePolicy):
    def set_ok(self, cookie, request):
        # Secure only from https, and the __Secure- / __Host- prefixes hold (rfc6265bis 5.7 steps 13, 20, 21).
        name = (cookie.name or "").lower()
        if cookie.secure and request.type != "https":
            return False
        if name.startswith(("__secure-", "__host-")) and not cookie.secure:
            return False
        if name.startswith("__host-") and (
            cookie.domain_specified or cookie.path != "/" or not cookie.path_specified
        ):
            return False
        # No public suffix list, so a parent Domain stays host-only; others are refused (RFC 6265 5.3).
        req_host, erhn = http.cookiejar.eff_request_host(request)
        if cookie.domain_specified and req_host.endswith("." + cookie.domain.lstrip(".")):
            cookie.domain, cookie.domain_specified, cookie.domain_initial_dot = erhn, False, False
        return super().set_ok(cookie, request)

    def return_ok(self, cookie, request):
        # Cross-site: Strict never, Lax (the default) only with a safe method (rfc6265bis 5.8.3).
        cross = getattr(request, "cross_site", None)
        mode = _same_site(cookie) if cross else ""
        if cross and (mode == "strict" or (mode != "none" and not cross["safe"])):
            return False
        return super().return_ok(cookie, request)


def _domain_match(host: str, domain: str) -> bool:
    host, domain = host.lstrip("."), domain.lstrip(".")
    return host == domain or host.endswith("." + domain)


def _path_match(path: str, cookie_path: str) -> bool:
    return path == cookie_path or (
        path.startswith(cookie_path)
        and (cookie_path.endswith("/") or path[len(cookie_path)] == "/")
    )


class _Jar(http.cookiejar.CookieJar):
    """While ``insecure``, http sets and deletes leave a shadowed Secure cookie alone (rfc6265bis 5.7
    step 16). Deletions run in ``make_cookies`` before any policy check: ``clear`` limits them to
    ``request``'s own host, where ``_Policy.set_ok`` would have put the cookie."""

    insecure = False
    request: Optional[urllib.request.Request] = None

    def _shadows_secure(self, domain, path, name) -> bool:
        return any(
            cookie.secure
            and cookie.name == name
            and (_domain_match(cookie.domain, domain) or _domain_match(domain, cookie.domain))
            and _path_match(path or "/", cookie.path or "/")
            for cookie in self
        )

    def set_cookie(self, cookie):
        if self.insecure and self._shadows_secure(cookie.domain, cookie.path, cookie.name):
            return
        super().set_cookie(cookie)

    def clear(
        self,
        domain = None,
        path = None,
        name = None,
    ):
        if self.request is None or name is None:
            if self.insecure and name is not None and self._shadows_secure(domain, path, name):
                return
            super().clear(domain, path, name)
            return
        # Host-only is stored under erhn, Domain=host under ".host", a parent Domain under erhn.
        req_host, erhn = http.cookiejar.eff_request_host(self.request)
        if domain in (erhn, "." + req_host):
            targets = (erhn, "." + req_host)
        elif domain and domain.startswith(".") and req_host.endswith(domain):
            targets = (erhn,)
        else:
            return
        for target in targets:
            if not (self.insecure and self._shadows_secure(target, path, name)):
                try:
                    super().clear(target, path, name)
                except KeyError:
                    pass


def _same_site(cookie: http.cookiejar.Cookie) -> str:
    for name, value in cookie._rest.items():  # type: ignore[attr-defined]
        if name.lower() == "samesite" and isinstance(value, str):
            return value.strip().lower()
    return "lax"


def _jar(subject: str) -> _Jar:
    with _lock:
        jar = _jars.get(subject)
        if jar is None:
            # Host-only cookies go back to the exact host that set them, never its subdomains.
            policy = _Policy(
                strict_ns_domain = http.cookiejar.DefaultCookiePolicy.DomainStrictNonDomain
            )
            jar = _jars[subject] = _Jar(policy)
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


def cookie_header(
    subject: str,
    url: str,
    cross_site: bool = False,
    method: str = "GET",
) -> Optional[str]:
    """The Cookie header a browser would send ``url``, or None; ``cross_site`` applies SameSite."""
    request = _request(url)
    if request is None:
        return None
    if cross_site:
        request.cross_site = {"safe": method.upper() in ("GET", "HEAD")}  # type: ignore[attr-defined]
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
    with jar._cookies_lock:  # type: ignore[attr-defined]
        jar.insecure, jar.request = request.type != "https", request
        try:
            jar.extract_cookies(_Response(values), request)
        finally:
            jar.insecure, jar.request = False, None
        # Under the lock: two responses trimming at once would pick the same cookie to drop.
        _trim(jar)
        _changed()


def _http_only(cookie: http.cookiejar.Cookie) -> bool:
    return any(name.lower() == "httponly" for name in cookie._rest)  # type: ignore[attr-defined]


def site_cookie_state(subject: str, url: str) -> tuple[list, list[tuple[str, str]], bool]:
    """``([epoch, version], page_cookies(...), has_site_cookies(...))``, read together so the version dates both."""
    jar = _jar(subject)
    with jar._cookies_lock:  # type: ignore[attr-defined]
        return [_EPOCH, _version], page_cookies(subject, url), has_site_cookies(subject, url)


def _site_cookies(subject: str, url: str) -> Iterable[http.cookiejar.Cookie]:
    """The cookies a same-origin request from ``url``'s page sends at some path, HttpOnly ones too."""
    page = _request(url)
    if page is None:
        return
    jar = _jar(subject)
    for cookie in list(jar):
        at_path = _request(f"{page.type}://{page.host}{cookie.path or '/'}")
        if at_path is not None and jar._policy.return_ok(cookie, at_path):  # type: ignore[attr-defined]
            yield cookie


def has_site_cookies(subject: str, url: str) -> bool:
    return next(iter(_site_cookies(subject, url)), None) is not None


def page_cookies(subject: str, url: str) -> list[tuple[str, str]]:
    """What ``document.cookie`` may see on ``url``'s site, at any path, in the frame's own form
    (browser.py): ``"<name>=<path>"`` to ``"<deadline ms>|<value>"``. HttpOnly ones stay here."""
    found: list[tuple[str, str]] = []
    room = _MAX_PAGE_COOKIE_CHARS
    for cookie in _site_cookies(subject, url):
        if _http_only(cookie):
            continue
        deadline = "" if cookie.expires is None else str(int(cookie.expires) * 1000)
        entry = (f"{cookie.name}={cookie.path or '/'}", f"{deadline}|{cookie.value or ''}")
        room -= len(entry[0]) + len(entry[1])
        if room < 0:
            break
        found.append(entry)
    return found


def set_page_cookie(
    subject: str,
    url: str,
    key: str,
    stored: Optional[str],
    same_site: Optional[str] = None,
) -> None:
    """A page's ``document.cookie`` write in the frame's form (``stored`` None removes): host-only,
    Secure over https, never over an HttpOnly cookie or a Secure one from http (rfc6265bis 5.7).
    ``same_site`` is the write's SameSite; an unknown value leaves the default (Lax)."""
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
        if request.type == "https":
            header += "; Secure"
        mode = (same_site or "").strip().lower()
        if mode == "none" and request.type != "https":
            return
        if mode in ("strict", "lax", "none"):
            header += f"; SameSite={mode.capitalize()}"
        if deadline.isdigit():
            header += f"; Max-Age={max(0, int(deadline) // 1000 - int(time.time()))}"
    store_cookies(subject, url, [("Set-Cookie", header)])


def clear_cookies(subject: str) -> None:
    with _lock:
        _jars.pop(subject, None)
    _changed()
