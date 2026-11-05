# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bootstrap origin gate edge cases: IPv6 netlocs, opaque origins; localhost and 127.0.0.1 differ."""

from unittest.mock import MagicMock

import pytest


def _build_request(
    host: str,
    origin,
    scheme: str = "http",
) -> MagicMock:
    request = MagicMock()
    request.url.scheme = scheme
    request.url.netloc = host
    request.headers = {"origin": origin} if origin is not None else {}
    return request


@pytest.mark.parametrize(
    "host, origin, scheme, same",
    [
        # IPv6: ``-H ::1`` binds give a bracketed netloc that a bare ``partition(":")`` mis-parses.
        pytest.param(
            "[::1]:8902", "http://[::1]:8902", "http", True, id = "ipv6_loopback_same_origin"
        ),
        pytest.param(
            "[2001:db8::1]:8443",
            "https://[2001:db8::1]:8443",
            "https",
            True,
            id = "ipv6_full_address_same_origin",
        ),
        pytest.param("[::1]:80", "http://[::1]", "http", True, id = "ipv6_default_port_stripped"),
        # Hex digits in IPv6 are case-insensitive per RFC 5952.
        pytest.param(
            "[2001:DB8::1]:8443",
            "https://[2001:db8::1]:8443",
            "https",
            True,
            id = "ipv6_case_insensitive",
        ),
        pytest.param(
            "[::1]:8902",
            "http://[2001:db8::1]:8902",
            "http",
            False,
            id = "ipv6_different_host_cross_origin",
        ),
        pytest.param(
            "[::1]:8902", "http://[::1]:9999", "http", False, id = "ipv6_port_mismatch_cross_origin"
        ),
        pytest.param(
            "user:pass@[::1]:8902", "http://[::1]:8902", "http", True, id = "ipv6_userinfo_stripped"
        ),
        # Opaque origins have no host, so they are never same-origin; ``file://`` is what older engines sent.
        pytest.param(
            "127.0.0.1:8902",
            "data:text/html,<script>alert(1)</script>",
            "http",
            False,
            id = "data_url_origin_is_cross_origin",
        ),
        pytest.param(
            "127.0.0.1:8902",
            "blob:http://127.0.0.1:8902/uuid",
            "http",
            False,
            id = "blob_url_origin_is_cross_origin",
        ),
        pytest.param(
            "127.0.0.1:8902", "file://", "http", False, id = "file_url_origin_is_cross_origin"
        ),
        # Starlette joins repeated headers with ``, ``, which cannot be split safely.
        pytest.param(
            "127.0.0.1:8902",
            "http://127.0.0.1:8902, http://evil.example",
            "http",
            False,
            id = "comma_joined_origins_cross_origin",
        ),
        # Browsers treat ``localhost`` and ``127.0.0.1`` as distinct origins: no DNS collapse.
        pytest.param(
            "127.0.0.1:8902",
            "http://localhost:8902",
            "http",
            False,
            id = "localhost_vs_127_is_cross_origin",
        ),
        pytest.param(
            "localhost:8902",
            "http://127.0.0.1:8902",
            "http",
            False,
            id = "127_vs_localhost_is_cross_origin",
        ),
        # ``urlparse`` raises ValueError on these (CVE-2024-11168 hardening); the gate must
        # swallow it and refuse rather than 500 the SPA handler.
        pytest.param(
            "127.0.0.1:8902",
            "http://[malformed",
            "http",
            False,
            id = "malformed_ipv6_bracket_is_cross_origin",
        ),
        pytest.param(
            "127.0.0.1:8902",
            "http://[::g]:8902",
            "http",
            False,
            id = "invalid_ipv6_address_is_cross_origin",
        ),
        pytest.param(
            "127.0.0.1:8902",
            "http://[2001:db8::1]extra:8902",
            "http",
            False,
            id = "bracket_with_trailing_garbage_is_cross_origin",
        ),
        # An explicit empty Origin is not a missing header.
        pytest.param("127.0.0.1:8902", "", "http", False, id = "empty_origin_header_is_cross_origin"),
    ],
)
def test_is_same_origin_request(host, origin, scheme, same):
    from main import _is_same_origin_request
    assert _is_same_origin_request(_build_request(host, origin, scheme)) is same
