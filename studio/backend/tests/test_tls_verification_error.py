# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recognising a failed certificate verification, and what we say about it.

A provider behind a private CA fails deep in the transport: httpx re-raises
httpcore's error, which re-raises ssl's, so the type that says "certificate" is two
``__cause__`` hops below what the route catches. These pin the walk, and pin the
reply as constant text -- the hint is returned instead of the exception string, so
anything interpolated into it would be new disclosure on an error response.
"""

from __future__ import annotations

import ast
import pathlib
import ssl
import sys

import pytest

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.utils import is_tls_verification_error


def _verify_error() -> ssl.SSLCertVerificationError:
    return ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
        "local issuer certificate (_ssl.c:1032)"
    )


def _chain(*errors: BaseException) -> BaseException:
    """Link errors outermost-first, the way `raise X from Y` nests them."""
    for outer, inner in zip(errors, errors[1:]):
        outer.__cause__ = inner
    return errors[0]


def test_the_bare_ssl_error_is_recognised():
    assert is_tls_verification_error(_verify_error()) is True


def test_the_transport_chain_is_walked():
    """httpx.ConnectError -> httpcore.ConnectError -> ssl.SSLCertVerificationError."""
    outer = _chain(OSError("[Errno 1] connect failed"), OSError("connect failed"), _verify_error())
    assert is_tls_verification_error(outer) is True


def test_an_implicit_context_chain_is_walked():
    """A transport that re-raises inside `except` chains via __context__, not __cause__."""
    outer = OSError("connect failed")
    outer.__context__ = _verify_error()
    assert is_tls_verification_error(outer) is True


def test_a_cyclic_chain_terminates():
    first, second = OSError("a"), OSError("b")
    first.__cause__ = second
    second.__cause__ = first
    assert is_tls_verification_error(first) is False


def test_text_is_the_fallback_when_a_transport_stops_chaining():
    unchained = RuntimeError(
        "All connection attempts failed: [SSL: CERTIFICATE_VERIFY_FAILED] certificate "
        "verify failed"
    )
    assert is_tls_verification_error(unchained) is True


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        OSError("[Errno 111] Connection refused"),
        ValueError("Unknown provider type: nope"),
    ],
)
def test_ordinary_failures_are_not_claimed_as_tls(error):
    assert is_tls_verification_error(error) is False


# ── The reply ────────────────────────────────────────────────────


def _tls_branch_messages() -> list[ast.expr]:
    """Every `message = ...` returned under `if is_tls_verification_error(...)`.

    Source-level because the route needs an authenticated app, a provider config and a
    live endpoint to reach this line, and the property under test is what the string
    is made of.
    """
    tree = ast.parse((_BACKEND / "routes" / "providers.py").read_text(encoding = "utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Call)
            and getattr(test.func, "id", None) == "is_tls_verification_error"
        ):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.keyword) and inner.arg == "message":
                found.append(inner.value)
    return found


def test_the_hint_is_reachable():
    assert _tls_branch_messages(), "no TLS branch found in routes/providers.py"


def test_the_hint_interpolates_nothing():
    """Constant only: no exception text, no host, no path, nothing host-specific."""
    for message in _tls_branch_messages():
        assert isinstance(message, ast.Constant), (
            "the TLS hint must stay a literal; an f-string or a call would put "
            "host-specific text on an error response"
        )


def test_the_hint_does_not_send_users_to_ssl_cert_file():
    """SSL_CERT_FILE replaces the bundle rather than adding to it.

    Advising it trades a provider failure for a Hugging Face one, which is exactly the
    dead end the original report walked into.
    """
    for message in _tls_branch_messages():
        assert "SSL_CERT_FILE" not in message.value
        assert "REQUESTS_CA_BUNDLE" not in message.value
