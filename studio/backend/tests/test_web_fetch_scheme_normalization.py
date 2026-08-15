# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bare hosts ("google.com") must be fetched as https, not refused."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import tools  # noqa: E402


@pytest.fixture
def resolved(monkeypatch):
    seen: dict = {}

    def fake_resolve(hostname, port, deadline, cancel_event):
        seen["hostname"] = hostname
        seen["port"] = port
        return False, "stopped", None

    monkeypatch.setattr(tools, "_resolve_with_budget", fake_resolve)
    return seen


@pytest.mark.parametrize(
    "url, hostname, port",
    [
        ("google.com", "google.com", 443),
        ("www.google.com/x", "www.google.com", 443),
        ("//google.com", "google.com", 443),
        ("https://google.com", "google.com", 443),
        ("http://google.com", "google.com", 80),
        ("example.com:8443/path", "example.com", 8443),
        ("example.com:8443", "example.com", 8443),
        ("sub.example.co.uk:8080", "sub.example.co.uk", 8080),
    ],
)
def test_schemeless_urls_are_fetched_as_https(resolved, url, hostname, port):
    err, _, _ = tools._fetch_url_raw(url)
    assert resolved["hostname"] == hostname
    assert resolved["port"] == port
    assert "only http/https" not in (err or "")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://x.com",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "mailto:a@b.c",
        # scheme:digits must not masquerade as host:port
        "file:80",
        "javascript:443/path",
        "mailto:25",
        # out-of-range ports are not host:port either
        "example.com:99999",
        "example.com:0",
        # ports must match ASCII [0-9]: str.isdigit() is True for digits int() refuses
        "example.com:²",
        "example.com:²/x",
        "example.com:①",
        "example.com:1²",
        "//example.com:²",
        # non-ASCII decimal digits int() accepts are ports urlparse then refuses
        "example.com:٤٤٣",
        # root-relative paths have no host to fetch
        "/login",
        "/github.com/owner/repo",
    ],
)
def test_non_http_schemes_still_blocked(url):
    err, _, _ = tools._fetch_url_raw(url)
    assert err and "only http/https" in err


def test_absurdly_long_port_does_not_raise():
    err, _, _ = tools._fetch_url_raw("example.com:" + "9" * 4400)
    assert err and "only http/https" in err


def test_out_of_range_port_returns_error_instead_of_raising():
    # check_url_access owns the wording; what matters is a string, not a raise.
    err, _, _ = tools._fetch_url_raw("https://example.com:99999")
    assert err and err.startswith("Blocked:")


def test_redirect_to_out_of_range_port_is_blocked(monkeypatch):
    # A redirect target reads .port too, so it needs the same guard.
    import urllib.request
    from urllib.error import HTTPError

    monkeypatch.setattr(
        tools,
        "_resolve_with_budget",
        lambda host, port, deadline, cancel: (True, "", "93.184.216.34"),
    )

    class _Redirecting:
        def open(self, req, **kw):
            hdrs = {"Location": "https://example.org:99999/next"}
            raise HTTPError(req.full_url, 302, "Found", hdrs, None)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: _Redirecting())
    err, _, _ = tools._fetch_url_raw("https://example.com")
    assert err and err.startswith("Blocked:")


@pytest.mark.parametrize(
    "url",
    [
        # urlparse raises on these; a model-supplied URL must still return a string
        "//exam／ple.com",  # NFKC-decomposes into "/"
        "//example.com＠",  # NFKC-decomposes into "@"
        "//example.com：",  # NFKC-decomposes into ":"
        "https://[::1",  # unmatched IPv6 bracket
        "https://::1]",
    ],
)
def test_malformed_url_is_blocked_instead_of_raising(url):
    err, _, _ = tools._fetch_url_raw(url)
    assert err and err.startswith("Blocked:")


def test_idna_failure_is_reported_instead_of_raising(monkeypatch):
    # getaddrinfo raises UnicodeError, not OSError, when IDNA encoding fails.
    import socket

    def boom(*a, **k):
        raise UnicodeError("encoding with 'idna' codec failed")

    monkeypatch.setattr(socket, "getaddrinfo", boom)
    err, _, _ = tools._fetch_url_raw("https://münich.example")
    assert err and err.startswith("Failed to resolve host:")


@pytest.mark.parametrize(
    "url, hostname",
    [
        (" google.com", "google.com"),
        ("google.com\n", "google.com"),
        ("\t example.com:8443 ", "example.com"),
    ],
)
def test_surrounding_whitespace_is_stripped(resolved, url, hostname):
    # _web_search strips, but direct callers of the fetch layer do not.
    tools._fetch_url_raw(url)
    assert resolved["hostname"] == hostname


@pytest.mark.parametrize("url", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "192.168.1.1"])
def test_normalization_does_not_bypass_ssrf_guard(url):
    err, _, _ = tools._fetch_url_raw(url, timeout = 3)
    assert err and "non-public address" in err


def test_schemeless_github_repo_still_routes_to_readme_api():
    # Must run before _github_repo_readme_api_url, else a bare repo URL scrapes HTML.
    normalized = tools._normalize_url_scheme("github.com/unslothai/unsloth")
    assert tools._github_repo_readme_api_url(normalized) == (
        "https://api.github.com/repos/unslothai/unsloth/readme"
    )
