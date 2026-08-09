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


def test_the_shell_reports_blocked_resources():
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert '"unsloth:artifact-blocked"' in shell
    # Bound after document.close(), which drops listeners registered before it.
    # Binding earlier silently reports nothing and the banner never appears.
    write, listen = (
        shell.index("document.close();"),
        shell.index('document.addEventListener("securitypolicyviolation"'),
    )
    assert write < listen


def test_blocked_reports_carry_the_load_they_came_from():
    # event.source still matches the iframe across the swap navigation, so a
    # report from the outgoing canvas would read as the incoming one's and
    # prompt a network grant for a canvas that never hit the CSP. The stamp is
    # read once at load, not per report, so a rewritten document cannot forge a
    # different one.
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert 'get("v")' in shell
    assert "v: loadVersion," in shell
    read, report = (
        shell.index("const loadVersion"),
        shell.index("const reportBlocked"),
    )
    assert read < report


def _directives(csp: str) -> dict:
    out = {}
    for part in csp.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(" ")
        out[name] = value.strip()
    return out


def test_the_shell_reports_which_directive_was_violated():
    # Without it the banner cannot tell a blocked CDN script from an object-src
    # violation, and offers a grant that provably cannot fix the latter.
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert "effectiveDirective: event.effectiveDirective" in shell


def test_the_grant_widens_everything_but_the_locked_directives():
    # Pins GRANT_CANNOT_FIX in html-frame.tsx. If a directive is ever locked
    # down in both policies, this fails and that set has to be updated, or the
    # banner starts prompting for something the grant cannot fix.
    strict = _directives(inf_mod._ARTIFACT_PREVIEW_FRAME_STRICT_CSP)
    network = _directives(inf_mod._ARTIFACT_PREVIEW_FRAME_NETWORK_CSP)
    unchanged = {
        name for name, value in strict.items() if name in network and network[name] == value
    }
    # frame-ancestors and sandbox are not resource loads, so they never produce
    # a blocked-resource report to filter.
    assert unchanged == {"object-src", "base-uri", "form-action", "frame-ancestors", "sandbox"}
    for locked in ("object-src", "base-uri", "form-action"):
        assert network[locked] == "'none'"
