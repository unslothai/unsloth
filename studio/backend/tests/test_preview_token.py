# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit + rotation coverage for `/p` preview capability tokens.

The token turns a guessable preview ref into an unguessable bearer capability:
it must round-trip for the ref it was signed for, reject tampering / wrong refs,
and stop verifying once the signing secret is rotated (link revocation).
"""

from pathlib import Path
import sys
import types as _types


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror the other preview tests: avoid the heavy real `loggers` handlers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import auth.storage as storage
import utils.preview_token as preview_token


_S1 = b"secret-one-aaaaaaaaaaaaaaaaaaaaaaaa"
_S2 = b"secret-two-bbbbbbbbbbbbbbbbbbbbbbbb"


def test_sign_verify_roundtrip(monkeypatch):
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _S1)
    token = preview_token.sign_preview_ref("run/checkpoint-1")
    assert preview_token.verify_preview_ref("run/checkpoint-1", token)
    # URL-safe, unpadded, and high-entropy (SHA-256 -> 43 base64url chars).
    assert "=" not in token and "/" not in token and "+" not in token
    assert len(token) >= 40


def test_missing_or_tampered_token_rejected(monkeypatch):
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _S1)
    token = preview_token.sign_preview_ref("demorun")
    assert not preview_token.verify_preview_ref("demorun", None)
    assert not preview_token.verify_preview_ref("demorun", "")
    flipped = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert not preview_token.verify_preview_ref("demorun", flipped)
    # A token minted for one ref does not unlock another.
    assert not preview_token.verify_preview_ref("otherrun", token)
    # A non-ASCII token is invalid, not a crash (the route would 500 otherwise).
    assert not preview_token.verify_preview_ref("demorun", "tøken-é")


def test_secret_change_invalidates_token(monkeypatch):
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _S1)
    token = preview_token.sign_preview_ref("demorun")
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _S2)
    assert not preview_token.verify_preview_ref("demorun", token)


def test_rotation_revokes_links(tmp_path, monkeypatch):
    # Exercise the real storage helpers against a throwaway auth.db.
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_preview_link_secret_cache", None)

    token = preview_token.sign_preview_ref("demorun")
    assert preview_token.verify_preview_ref("demorun", token)
    # Secret persists across calls (the link keeps working until rotated).
    assert preview_token.verify_preview_ref("demorun", token)

    storage.rotate_preview_link_secret()
    # Old shared link is revoked; a freshly minted one works.
    assert not preview_token.verify_preview_ref("demorun", token)
    assert preview_token.verify_preview_ref(
        "demorun", preview_token.sign_preview_ref("demorun")
    )
