# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The preview shell picks its CSP from ``allow_network`` and nothing else, so
both directions of that switch are worth pinning: the permissive variant must
require the flag, and its absence must land on the strict one. Every canvas
reaches this route now, fenced HTML included, not just approved render_html
output."""

import asyncio

import routes.inference as inf_mod


def _csp(*args) -> str:
    response = asyncio.run(inf_mod.artifact_preview_frame(*args))
    return response.headers["content-security-policy"]


def test_omitting_the_flag_serves_the_strict_csp():
    # The fail-closed direction: a caller that says nothing gets no network.
    csp = _csp()
    assert "default-src 'none';" in csp
    assert "script-src 'unsafe-inline';" in csp
    assert "connect-src 'none';" in csp
    assert "http:" not in csp.split("frame-ancestors")[0]


def test_allow_network_serves_the_permissive_csp():
    csp = _csp(True)
    assert "script-src-elem 'unsafe-inline' http: https:" in csp
    assert "connect-src http: https: ws: wss:" in csp


def test_the_flag_is_what_changes_the_policy():
    assert _csp(True) != _csp(False)


def test_the_sandbox_holds_in_both_variants():
    # Network access widens what the canvas may fetch, never how it is isolated.
    for csp in (_csp(False), _csp(True)):
        assert "sandbox allow-scripts" in csp
        assert "object-src 'none';" in csp
        assert "base-uri 'none';" in csp
        assert "form-action 'none';" in csp
        assert "frame-ancestors 'self'" in csp
