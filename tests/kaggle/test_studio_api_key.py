# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A minted key only counts if it authenticates alone, and a corrupted key must be refused."""

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
    """Set the session token aside: left in place, the call succeeds regardless of the key."""
    body = _assert_api_key_body()
    assert "saved = self.studio.token" in body
    assert "self.studio.token = saved" in body, "the session token must be restored"

    # Structural, not substring: the corrupted-key assignment contains the swap line's text.
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
    """A server that ignores the Authorization header would pass the positive check."""
    body = _assert_api_key_body()
    assert "bad_key_status" in body
    assert "a corrupted API key was accepted" in body


def test_the_raw_key_is_registered_as_a_secret_before_it_is_used():
    """The key reaches request headers and the evidence bundle, so register it before use."""
    body = _assert_api_key_body()
    add_at = body.index("self.secrets.add(raw_key)")
    use_at = body.index("self.studio.token = raw_key")
    assert add_at < use_at, "the key is used before it is registered for scrubbing"


def test_the_session_token_is_restored_even_when_the_call_raises():
    """Restore the token in a finally, or an exception leaves the session on a corrupted key."""
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
    """Must run before the long training, so a training failure cannot hide whether keys work."""
    api_at = SRC.index("self.assert_api_key()")
    train_at = SRC.index("trained = self.assert_training()")
    assert api_at < train_at
