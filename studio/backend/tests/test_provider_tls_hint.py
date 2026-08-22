# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the provider test route is allowed to say about a TLS failure.

``test_provider`` returns ``safe_curated_detail(exc)`` everywhere except the
certificate branch, which answers with a fixed sentence instead -- a verification
failure names neither the cause nor a next step, and anything host-specific
interpolated there would land on an error response. ``is_tls_verification_error``
has its own coverage in ``test_utils.py``; these guard the reply.

Source-level, because reaching that line needs an authenticated app, a saved
provider config and an endpoint that fails a handshake.
"""

from __future__ import annotations

import ast
import pathlib

_ROUTE = pathlib.Path(__file__).resolve().parents[1] / "routes" / "providers.py"


def _tls_branch_messages() -> list[ast.expr]:
    """Every ``message = ...`` returned under ``if is_tls_verification_error(...)``."""
    tree = ast.parse(_ROUTE.read_text(encoding = "utf-8"))
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and getattr(node.test.func, "id", None) == "is_tls_verification_error"
        ):
            continue
        found += [
            inner.value
            for inner in ast.walk(node)
            if isinstance(inner, ast.keyword) and inner.arg == "message"
        ]
    return found


def test_the_hint_is_reachable():
    assert _tls_branch_messages(), (
        "no is_tls_verification_error branch in routes/providers.py; a certificate "
        "failure is back to reporting the raw SSL error"
    )


def test_the_hint_interpolates_nothing():
    for message in _tls_branch_messages():
        assert isinstance(message, ast.Constant), (
            "the TLS hint must stay a literal: an f-string or a call would put "
            "host-specific text on an error response"
        )


def test_the_hint_does_not_send_users_to_ssl_cert_file():
    """On Linux those variables replace the bundle rather than adding to it.

    Advising them trades a provider failure for a Hugging Face one, which is the dead
    end #9218 reported. See the SSL_CERT_FILE paragraph in utils/native_tls.py.
    """
    for message in _tls_branch_messages():
        # unparse, not .value: an f-string has no .value, and this should report the
        # advice it found rather than erroring on the node type.
        text = ast.unparse(message)
        assert "SSL_CERT_FILE" not in text
        assert "REQUESTS_CA_BUNDLE" not in text
