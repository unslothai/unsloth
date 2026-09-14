# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Minting a Studio API key, and the two ways that check proves nothing.

Creating a key proves the endpoint returns a string. Whether that string
authenticates anything is a separate question, and it is the one that matters:
a key Studio issues and then rejects fails in a user's integration rather than
here.

**Vacuity 1: passing on the session token.** If the bearer is not swapped, the
request succeeds because the session is still logged in and the key is never
exercised.

**Vacuity 2: a server that ignores the header.** Then any key "works", including
one that was never issued. The corrupted-key call is what rules that out, and
without it the positive check is satisfied by a server with no auth at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _assert_api_key_body() -> str:
    tree = ast.parse(SRC)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "assert_api_key"
    )
    return ast.get_source_segment(SRC, func) or ""


def test_the_key_is_created_through_the_real_endpoint():
    body = _assert_api_key_body()
    assert '"/api/auth/api-keys"' in body


def test_the_session_token_is_set_aside_while_the_key_is_driven():
    """Vacuity 1. Leaving the session token in place makes the call succeed for
    a reason that has nothing to do with the key."""
    body = _assert_api_key_body()
    assert "saved = self.studio.token" in body
    assert "self.studio.token = saved" in body, "the session token must be restored"

    # Structural, not a substring. `self.studio.token = raw_key[:-4] + "0000"`
    # CONTAINS the string "self.studio.token = raw_key", so a substring check
    # passes even when the swap line is deleted and only the corrupted-key
    # assignment remains -- which is exactly the mutation this guard exists to
    # catch, and it survived the first version of this test.
    tree = ast.parse(body)
    swaps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Name)
        and node.value.id == "raw_key"
        and any(isinstance(t, ast.Attribute) and t.attr == "token" for t in node.targets)
    ]
    assert swaps, (
        "the bearer is never set to the raw key itself, so the positive check "
        "runs on whatever token was already there"
    )


def test_a_corrupted_key_must_be_rejected():
    """Vacuity 2. Without this, a server that ignores the Authorization header
    entirely passes the positive check."""
    body = _assert_api_key_body()
    assert "bad_key_status" in body
    assert "a corrupted API key was accepted" in body


def test_the_raw_key_is_registered_as_a_secret_before_it_is_used():
    """It reaches request headers and, on failure, error strings that travel
    home in the evidence bundle."""
    body = _assert_api_key_body()
    add_at = body.index("self.secrets.add(raw_key)")
    use_at = body.index("self.studio.token = raw_key")
    assert add_at < use_at, "the key is used before it is registered for scrubbing"


def test_the_session_token_is_restored_even_when_the_call_raises():
    """A finally, not a trailing statement: an exception mid-check would
    otherwise leave the session authenticated as a corrupted API key and every
    later assertion would fail for the wrong reason."""
    tree = ast.parse(_assert_api_key_body())
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
    restores = [
        t
        for t in tries
        if "self.studio.token = saved" in "".join(ast.dump(s) for s in t.finalbody)
        or any("saved" in ast.dump(s) for s in t.finalbody)
    ]
    assert restores, "the token restore is not in a finally"


def test_it_runs_before_the_long_gpu_work():
    """It needs only a logged-in session. After a 20-minute training run, a
    training failure would hide whether API keys work at all."""
    api_at = SRC.index("self.assert_api_key()")
    train_at = SRC.index("trained = self.assert_training()")
    assert api_at < train_at
