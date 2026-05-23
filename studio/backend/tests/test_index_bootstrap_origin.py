# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the bootstrap-pw cross-origin leak (PR 5739).

``_is_same_origin_request`` gates ``_inject_bootstrap`` so the seeded
admin password only ships to same-origin callers. See the ``CORS: GET /``
audit in tests/studio/studio_api_smoke.py.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _build_request(host: str, origin: str | None) -> MagicMock:
    request = MagicMock()
    request.url.scheme = "http"
    request.url.netloc = host
    request.headers = {"origin": origin} if origin is not None else {}
    return request


def test_is_same_origin_request_missing_origin_is_same_origin(monkeypatch):
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = None)
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_matching_origin_is_same_origin():
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "http://127.0.0.1:8888")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_evil_origin_is_cross_origin():
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "https://evil.example")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_scheme_mismatch_is_cross_origin():
    # https origin against an http listener is not same-origin.
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "https://127.0.0.1:8888")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_port_mismatch_is_cross_origin():
    # Same host different port is not same-origin per the web platform.
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "http://127.0.0.1:5173")
    assert _is_same_origin_request(req) is False
