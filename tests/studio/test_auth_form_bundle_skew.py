# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Backend and bundle can be a version apart; the setup form must cope.

No `dist` is committed, so a backend and its bundle normally ship together and
this only arises from a stale build directory or an explicit `--frontend <path>`
at a previously built one. It is cheap to guarantee anyway, and the failure mode
(first login impossible on an install that upgraded half way) is expensive.

Three payload shapes exist in the wild:
  {"username", "password"}    an older backend, which embedded the seed
  {"username", "link_token"}  this backend, which embeds a one-time setup token
  nothing at all             already set up, or a public launch that injects none

The contract is that the two Current-password rules stay in step with what is
actually available: hide the field exactly when the page carries something that
can stand in for the current password, and show it otherwise so the operator can
type the seed from .bootstrap_password by hand.

Real source is sliced verbatim and evaluated under node, in the style of
tests/studio/_node_harness.py; only `window` is stubbed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AUTH_FORM = REPO / "studio/frontend/src/features/auth/components/auth-form.tsx"
MERGE_BASE = "a7dec18b6"

DERIVATIONS = ("currentPassword", "setupToken", "hasBootstrapPassword")

PAYLOADS = {
    "older backend, seed in page": {"username": "unsloth", "password": "seeded-pass"},
    "this backend, setup token": {"username": "unsloth", "link_token": "T"},
    "nothing injected": None,
}


def _node_or_skip() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node


def _old_source() -> str:
    proc = subprocess.run(
        ["git", "show", f"{MERGE_BASE}:studio/frontend/src/features/auth/components/auth-form.tsx"],
        cwd = REPO,
        capture_output = True,
        text = True,
    )
    if proc.returncode != 0:
        pytest.skip(f"merge base {MERGE_BASE} is not available in this checkout")
    return proc.stdout


def _slice_derivations(src: str) -> str:
    """The `const <name> = ...;` declarations that decide what the form shows."""
    out = []
    for name in DERIVATIONS:
        match = re.search(rf"^  const {name} = .*?;$", src, re.MULTILINE | re.DOTALL)
        if match:
            out.append(match.group(0))
    assert out, "none of the form's derivations could be located"
    return "\n".join(out)


def _evaluate(src: str, payload) -> dict:
    node = _node_or_skip()
    script = textwrap.dedent(
        """
        globalThis.window = { __UNSLOTH_BOOTSTRAP__: __PAYLOAD__ };
        const password = "";
        __SETUP_TOKEN_SHIM__
        __DERIVATIONS__
        console.log(JSON.stringify({
          hasBootstrapPassword: Boolean(hasBootstrapPassword),
          currentPasswordFieldShown: !hasBootstrapPassword,
          currentPassword,
          endpoint: setupToken
            ? "/api/auth/link-initial-password"
            : "/api/auth/change-password",
        }));
        """
    )
    derivations = _slice_derivations(src)
    # A bundle that predates the setup token never declares it; give the shim a
    # binding so the same probe runs against both versions.
    shim = "" if "const setupToken" in derivations else "const setupToken = undefined;"
    script = (
        script.replace("__PAYLOAD__", json.dumps(payload))
        .replace("__SETUP_TOKEN_SHIM__", shim)
        .replace("__DERIVATIONS__", derivations)
    )
    proc = subprocess.run(
        [node, "--input-type=module", "-e", script], capture_output = True, text = True, timeout = 60
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ── this bundle ──────────────────────────────────────────────────────


def test_this_bundle_hides_the_field_for_a_setup_token():
    result = _evaluate(AUTH_FORM.read_text(encoding = "utf-8"), PAYLOADS["this backend, setup token"])
    assert result["currentPasswordFieldShown"] is False
    assert result["endpoint"] == "/api/auth/link-initial-password"
    assert result["currentPassword"] == "", "the token must never back a password field"


def test_this_bundle_still_works_against_an_older_backend():
    """Backwards compatibility: an old backend sends the seed, not a token.

    The bundle must fall back to the legacy behaviour rather than sitting there
    waiting for a token that will never arrive.
    """
    result = _evaluate(
        AUTH_FORM.read_text(encoding = "utf-8"), PAYLOADS["older backend, seed in page"]
    )
    assert result["currentPasswordFieldShown"] is False
    assert result["endpoint"] == "/api/auth/change-password"
    assert result["currentPassword"] == "seeded-pass"


def test_this_bundle_shows_the_field_when_nothing_is_injected():
    """A public launch injects nothing, and the operator types the seed.

    This is also the admin-forced change-password path, which never had an
    injected value.
    """
    result = _evaluate(AUTH_FORM.read_text(encoding = "utf-8"), PAYLOADS["nothing injected"])
    assert result["currentPasswordFieldShown"] is True
    assert result["endpoint"] == "/api/auth/change-password"


# ── the previous bundle, against this backend ────────────────────────


def test_the_previous_bundle_degrades_rather_than_breaking():
    """Forwards compatibility: an old bundle cannot use a token it never heard of.

    It falls back to showing the Current password field, which still works off
    .bootstrap_password. Degraded (the operator types a passphrase that used to
    be filled in for them) but not broken, and it needs a stale dist to happen
    at all.
    """
    result = _evaluate(_old_source(), PAYLOADS["this backend, setup token"])
    assert result["currentPasswordFieldShown"] is True
    assert result["endpoint"] == "/api/auth/change-password"
    assert result["currentPassword"] == ""


def test_the_previous_bundle_is_unchanged_against_a_previous_backend():
    """The control: same bundle, same backend, still two inputs."""
    result = _evaluate(_old_source(), PAYLOADS["older backend, seed in page"])
    assert result["currentPasswordFieldShown"] is False
    assert result["endpoint"] == "/api/auth/change-password"


def test_both_bundles_agree_when_nothing_is_injected():
    """No injection is the case this change must not touch at all."""
    new = _evaluate(AUTH_FORM.read_text(encoding = "utf-8"), PAYLOADS["nothing injected"])
    old = _evaluate(_old_source(), PAYLOADS["nothing injected"])
    assert new["currentPasswordFieldShown"] == old["currentPasswordFieldShown"] is True
    assert new["endpoint"] == old["endpoint"]
