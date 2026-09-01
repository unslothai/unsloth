# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The preview shell picks its CSP from ``allow_network`` and nothing else, so
both directions of that switch are worth pinning: the permissive variant must
require the flag, and its absence must land on the strict one. Every canvas
reaches this route now, fenced HTML included, not just approved render_html
output."""

import asyncio
import pathlib

import routes.inference as inf_mod


def _csp(*args) -> str:
    response = asyncio.run(inf_mod.artifact_preview_frame(*args))
    return response.headers["content-security-policy"]


def test_omitting_the_flag_serves_the_strict_csp():
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
    for csp in (_csp(False), _csp(True)):
        assert "sandbox allow-scripts" in csp
        assert "object-src 'none';" in csp
        assert "base-uri 'none';" in csp
        assert "form-action 'none';" in csp
        assert "frame-ancestors 'self'" in csp


def test_the_shell_reports_blocked_resources():
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert '"unsloth:artifact-blocked"' in shell
    # document.close() drops listeners bound before it, so binding earlier reports nothing and the banner never appears.
    write, listen = (
        shell.index("document.close();"),
        shell.index('document.addEventListener("securitypolicyviolation"'),
    )
    assert write < listen


def test_blocked_reports_carry_the_load_they_came_from():
    # event.source survives the swap navigation, so without the stamp a report from the outgoing
    # canvas prompts a grant for one that never hit the CSP.
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
    # Without it the banner cannot tell a blocked CDN script from an object-src violation, and
    # offers a grant that cannot fix the latter.
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert "effectiveDirective: event.effectiveDirective" in shell


def test_the_grant_widens_everything_but_the_locked_directives():
    # Pins GRANT_CANNOT_FIX in html-frame.tsx: locking a fourth directive down in both policies
    # fails here rather than silently prompting for something the grant cannot fix.
    strict = _directives(inf_mod._ARTIFACT_PREVIEW_FRAME_STRICT_CSP)
    network = _directives(inf_mod._ARTIFACT_PREVIEW_FRAME_NETWORK_CSP)
    unchanged = {
        name for name, value in strict.items() if name in network and network[name] == value
    }
    # frame-ancestors and sandbox are not resource loads, so they never report.
    assert unchanged == {"object-src", "base-uri", "form-action", "frame-ancestors", "sandbox"}
    for locked in ("object-src", "base-uri", "form-action"):
        assert network[locked] == "'none'"


def test_the_permissive_policy_widens_every_hostless_scheme_but_one():
    # Pins GRANT_CANNOT_FIX_SCHEME in html-frame.tsx: a non-HTTP(S) violation reports a bare
    # scheme, so the grant is offered only where the permissive policy allows that scheme for
    # that directive (Chromium: a data: Worker reports worker-src/data under both).
    network = _directives(inf_mod._ARTIFACT_PREVIEW_FRAME_NETWORK_CSP)
    # Locked or not a resource load, so they never reach the scheme check.
    skip = {"object-src", "base-uri", "form-action", "frame-ancestors", "sandbox"}
    gaps = {
        name: scheme
        for name, value in network.items()
        if name not in skip
        for scheme in ("data:", "blob:")
        if scheme not in value.split()
    }
    assert gaps == {"worker-src": "data:"}


def test_the_shell_restores_randomuuid_for_insecure_canvases():
    # This test cannot execute the shell, so pin the fallback's required pieces.
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    assert 'typeof crypto.randomUUID === "function"' in shell
    assert "crypto.randomUUID = () =>" in shell
    assert "installRandomUUIDFallback();" in shell


def test_the_shell_generator_matches_the_app_one():
    # The strict CSP forbids sharing crypto-boot.js, so keep both copies aligned.
    shell = inf_mod._ARTIFACT_PREVIEW_FRAME_HTML
    boot = (
        pathlib.Path(__file__).resolve().parents[2] / "frontend/public/crypto-boot.js"
    ).read_text(encoding = "utf-8")
    for expression in (
        '"10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>',
        "(+c ^ (randomByte() & (15 >> (+c / 4)))).toString(16)",
    ):
        assert expression in boot
        assert expression in shell
