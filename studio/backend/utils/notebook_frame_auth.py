# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared notebook iframe authorization helpers."""

from __future__ import annotations

from http.cookies import SimpleCookie
import os
from collections.abc import Mapping
from urllib.parse import parse_qs


NOTEBOOK_FRAME_TOKEN_ENV = "UNSLOTH_STUDIO_NOTEBOOK_FRAME_TOKEN"
NOTEBOOK_FRAME_TOKEN_PARAM = "__unsloth_frame"
NOTEBOOK_FRAME_COOKIE = "__unsloth_frame"
NOTEBOOK_FRAME_COOKIE_MAX_AGE_SECONDS = 2 * 60 * 60
TRUST_CF_CONNECTING_IP_ENV = "UNSLOTH_STUDIO_TRUST_CF_CONNECTING_IP"
CF_CLIENT_IP_REQUIRES_FRAME_COOKIE_STATE = "cloudflare_client_ip_requires_frame_cookie"


def expected_notebook_frame_token(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(NOTEBOOK_FRAME_TOKEN_ENV, "").strip()


def notebook_frame_query_matches(scope, expected: str | None = None) -> bool:
    expected = expected_notebook_frame_token() if expected is None else expected
    if not expected:
        return False
    try:
        raw_query = scope.get("query_string", b"")
        if isinstance(raw_query, bytes):
            raw_query = raw_query.decode("latin-1")
        values = parse_qs(raw_query, keep_blank_values = True).get(
            NOTEBOOK_FRAME_TOKEN_PARAM,
            [],
        )
    except Exception:
        return False
    return expected in values


def _cookie_header(headers) -> str:
    if not headers:
        return ""
    if hasattr(headers, "get"):
        return headers.get("cookie", "") or headers.get("Cookie", "")

    values: list[str] = []
    for name, value in headers:
        if isinstance(name, bytes):
            name_text = name.decode("latin-1")
        else:
            name_text = str(name)
        if name_text.lower() != "cookie":
            continue
        if isinstance(value, bytes):
            values.append(value.decode("latin-1"))
        else:
            values.append(str(value))
    return "; ".join(values)


def notebook_frame_cookie_matches(headers, expected: str | None = None) -> bool:
    expected = expected_notebook_frame_token() if expected is None else expected
    if not expected:
        return False
    raw_cookie = _cookie_header(headers)
    if not raw_cookie:
        return False
    try:
        cookie = SimpleCookie()
        cookie.load(raw_cookie)
        morsel = cookie.get(NOTEBOOK_FRAME_COOKIE)
    except Exception:
        return False
    return morsel is not None and morsel.value == expected


def notebook_frame_cookie_header(token: str) -> str:
    cookie = SimpleCookie()
    cookie[NOTEBOOK_FRAME_COOKIE] = token
    morsel = cookie[NOTEBOOK_FRAME_COOKIE]
    morsel["path"] = "/"
    morsel["max-age"] = str(NOTEBOOK_FRAME_COOKIE_MAX_AGE_SECONDS)
    morsel["secure"] = True
    morsel["httponly"] = True
    morsel["samesite"] = "None"
    return cookie.output(header = "").strip() + "; Partitioned"


def _truthy_env(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in {"1", "true", "yes"}


def cloudflare_client_ip_trusted(headers, app_state = None, env: Mapping[str, str] | None = None) -> bool:
    """Whether CF-Connecting-IP should be honored for a loopback tunnel request."""
    env = os.environ if env is None else env
    if _truthy_env(env, TRUST_CF_CONNECTING_IP_ENV):
        return True
    if app_state is None or not bool(getattr(app_state, "trust_cloudflare_client_ip", False)):
        return False
    if not bool(getattr(app_state, CF_CLIENT_IP_REQUIRES_FRAME_COOKIE_STATE, True)):
        return True
    return notebook_frame_cookie_matches(headers)
