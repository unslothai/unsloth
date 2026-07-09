# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pin the auth-form input-count contract on the change-password page.

PR #5490 added a third "Current password" input, regressing first-boot UX to
three inputs; PR #5545 restores two by rendering it only when BOOTSTRAP is absent.
These tests inspect the source directly (no Studio/browser/network); runtime is
covered by tests/studio/playwright_chat_ui.py."""

from __future__ import annotations

import re
from pathlib import Path

AUTH_FORM = (
    Path(__file__).resolve().parents[2]
    / "studio/frontend/src/features/auth/components/auth-form.tsx"
)

CONDITIONAL_OPENER = "{!hasBootstrapPassword && ("


def _conditional_extent(src: str) -> tuple[int, int]:
    """(start, end) char offsets of the `{!hasBootstrapPassword && (...)}` JSX block."""
    start = src.find(CONDITIONAL_OPENER)
    assert start != -1, (
        "the {!hasBootstrapPassword && (...)} JSX block that hides the "
        "Current password input on first boot is missing -- PR #5545 has "
        "been reverted or the conditional was inlined as a ternary"
    )
    depth = 1
    i = start + len(CONDITIONAL_OPENER)
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    raise AssertionError("unterminated !hasBootstrapPassword JSX block")


def test_hasbootstrappassword_constant_is_derived_from_bootstrap_window_value():
    """The guard must read from window.__UNSLOTH_BOOTSTRAP__, matching the backend's
    bootstrap-injection contract in studio/backend/main.py::_inject_bootstrap."""
    src = AUTH_FORM.read_text()
    assert "const hasBootstrapPassword = Boolean(window.__UNSLOTH_BOOTSTRAP__?.password);" in src, (
        "hasBootstrapPassword constant missing or its derivation drifted; "
        "this is the gate that hides the Current password input on first boot"
    )


def test_exactly_one_hasBootstrapPassword_conditional_exists():
    """Only one `!hasBootstrapPassword` JSX check is allowed; a second would split
    rendering into branches and likely hide or duplicate the New / Confirm inputs."""
    src = AUTH_FORM.read_text()
    count = src.count("!hasBootstrapPassword")
    assert count == 1, (
        f"expected exactly one !hasBootstrapPassword usage, found {count}; "
        "extra conditionals can hide or duplicate the always-on inputs"
    )


def test_current_password_input_is_inside_the_hasBootstrapPassword_conditional():
    """`id="current-password"` must sit inside `{!hasBootstrapPassword && (...)}`,
    else it renders on first boot too, regressing the pre-#5490 UX that PR #5545 restores."""
    src = AUTH_FORM.read_text()
    s, e = _conditional_extent(src)
    idx = src.find('id="current-password"')
    assert idx != -1, "the Current password input was removed entirely"
    assert s < idx < e, (
        "Current password input is rendered unconditionally; this is the "
        "PR #5490 regression -- on first boot the bootstrap-derived "
        "password is reused silently and only New + Confirm should render"
    )


def test_new_password_input_is_outside_the_hasBootstrapPassword_conditional():
    """`id="new-password"` must sit outside `{!hasBootstrapPassword && (...)}`,
    else it disappears on admin-forced resets, regressing PR #5490."""
    src = AUTH_FORM.read_text()
    s, e = _conditional_extent(src)
    idx = src.find('id="new-password"')
    assert idx != -1, "the New password input was removed entirely"
    assert not (s < idx < e), (
        "New password is wrapped in !hasBootstrapPassword; that would "
        "hide the field on admin-forced resets, regressing PR #5490. "
        "New password must always render in change-password mode."
    )


def test_confirm_password_input_is_outside_the_hasBootstrapPassword_conditional():
    """Same as New password, for `id="confirm-password"`."""
    src = AUTH_FORM.read_text()
    s, e = _conditional_extent(src)
    idx = src.find('id="confirm-password"')
    assert idx != -1, "the Confirm password input was removed entirely"
    assert not (s < idx < e), (
        "Confirm password is wrapped in !hasBootstrapPassword; same "
        "regression as New password -- it must always render in "
        "change-password mode."
    )


def test_change_password_jsx_declares_exactly_three_password_inputs():
    """The change-password JSX block (`{!isLoginMode && (...)}`) must declare exactly
    current/new/confirm; a fourth would break the 2-input first-boot contract (the
    conditional only hides Current)."""
    src = AUTH_FORM.read_text()
    start = src.find("{!isLoginMode && (")
    assert start != -1, (
        "the change-password JSX subtree marker {!isLoginMode && (...)} "
        "is missing; the file's structure has drifted"
    )
    # Match the corresponding `)}` for {!isLoginMode && (...)}.
    depth = 1
    i = start + len("{!isLoginMode && (")
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    subtree = src[start:i]
    ids = sorted(re.findall(r'id="([a-z-]+-password)"', subtree))
    assert ids == [
        "confirm-password",
        "current-password",
        "new-password",
    ], (
        "change-password JSX must declare exactly current-password, "
        f"new-password, confirm-password; found {ids!r}. A fourth "
        "password input would almost certainly break the 2-input "
        "first-boot contract."
    )


def test_login_jsx_declares_exactly_one_password_input():
    """The login JSX block (`isLoginMode && (...)`) must declare exactly one password
    input (the bootstrap password pasted from the CLI); a second breaks the per-mode matrix."""
    src = AUTH_FORM.read_text()
    start = src.find("{isLoginMode && (")
    assert start != -1, "the login JSX subtree marker is missing"
    depth = 1
    i = start + len("{isLoginMode && (")
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    subtree = src[start:i]
    ids = re.findall(r'id="([a-z-]+)"', subtree)
    # Lock the count, not the spelling, so a rename does not falsely fail.
    pw_ids = [x for x in ids if "password" in x]
    assert len(pw_ids) == 1, (
        f"login JSX must declare exactly one password-typed input; " f"found {pw_ids!r}"
    )
