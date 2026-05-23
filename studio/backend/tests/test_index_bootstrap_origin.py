# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression coverage for the bootstrap-password cross-origin leak surfaced
by tests/studio/studio_api_smoke.py.

Default web mode runs FastAPI's CORSMiddleware with
``allow_origins=["*"]`` and ``allow_credentials=True``, which reflects an
attacker-controlled ``Origin`` back on every request. The ``/`` route
serves index.html with an inline ``window.__UNSLOTH_BOOTSTRAP__`` script
containing the seeded admin password while a password change is pending.
That combination let a cross-origin page ``fetch('/')`` with credentials
and read the bootstrap password out of the HTML body.

``_is_same_origin_request`` is the gate that keeps the injection
same-origin-only; ``_build_index_response`` calls it and adds
``Vary: Origin`` so caches do not mix the two response shapes.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _build_request(host: str, origin: str | None) -> MagicMock:
    """Mock just enough of starlette.Request for the helper under test."""
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
    """https origin against an http listener must NOT be treated as same."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "https://127.0.0.1:8888")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_port_mismatch_is_cross_origin():
    """Same host different port is NOT same-origin per the web platform."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8888", origin = "http://127.0.0.1:5173")
    assert _is_same_origin_request(req) is False
