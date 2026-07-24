# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Colab public-launch auto-password + opt-in same-tab link token.

start(cloudflare=True) with no admin password set auto-generates one, shows it in
the cell, and lets the shared link proceed; a supplied password is respected. The
opt-in link token is appended to the SAME-TAB URL only. Imports the backend
directly, so run under the Unsloth venv."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import colab  # noqa: E402
from auth import storage  # noqa: E402


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


def _seed_admin(*, must_change_password: bool) -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "bootstrap-secret-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return storage.DEFAULT_ADMIN_USERNAME


# ── auto-generate on the public path ─────────────────────────────────


def test_auto_generate_sets_password_and_clears_must_change():
    admin = _seed_admin(must_change_password = True)
    assert storage.requires_password_change(admin) is True

    generated = colab._auto_generate_colab_admin_password()

    assert isinstance(generated, str) and len(generated) >= storage.MIN_PASSWORD_LENGTH
    # Committed: must_change cleared and the new password verifies.
    assert storage.requires_password_change(admin) is False
    salt, pwd_hash, _jwt, _mc = storage.get_user_and_secret(admin)
    from auth import hashing

    assert hashing.verify_password(generated, salt, pwd_hash) is True


def test_auto_generate_noop_when_password_already_set():
    admin = _seed_admin(must_change_password = False)
    # A supplied/changed password must NOT be overwritten.
    assert colab._auto_generate_colab_admin_password() is None
    assert storage.requires_password_change(admin) is False


def test_start_cloudflare_tunnel_autogenerates_then_proceeds(monkeypatch):
    admin = _seed_admin(must_change_password = True)
    shown = {}
    monkeypatch.setattr(
        colab, "_display_admin_credentials", lambda u, p: shown.update(user = u, pw = p)
    )
    # start_cloudflare_tunnel does `from cloudflare_tunnel import start_studio_tunnel`;
    # patch the module attribute so no real cloudflared is spawned.
    import cloudflare_tunnel

    monkeypatch.setattr(
        cloudflare_tunnel,
        "start_studio_tunnel",
        lambda port: "https://example.trycloudflare.com",
    )

    url = colab.start_cloudflare_tunnel(8888)

    assert url == "https://example.trycloudflare.com"
    assert shown["user"] == admin
    assert storage.requires_password_change(admin) is False


def test_start_cloudflare_tunnel_refuses_if_autogen_fails(monkeypatch):
    _seed_admin(must_change_password = True)
    # Simulate auto-generation failing (e.g. DB error): the pending gate then still
    # refuses the shared link (fail safe), returning None.
    monkeypatch.setattr(colab, "_auto_generate_colab_admin_password", lambda: None)
    assert colab.start_cloudflare_tunnel(8888) is None


# ── opt-in same-tab link token ───────────────────────────────────────


def test_link_token_opt_in_env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_COLAB_LINK_TOKEN", raising = False)
    assert colab._link_token_opt_in(False) is False
    assert colab._link_token_opt_in(True) is True
    monkeypatch.setenv("UNSLOTH_STUDIO_COLAB_LINK_TOKEN", "1")
    assert colab._link_token_opt_in(False) is True


def test_append_link_token_same_tab_only():
    assert colab._append_link_token("https://x/", None) == "https://x/"
    assert colab._append_link_token("https://x/", "") == "https://x/"
    assert colab._append_link_token("https://x/", "tok") == "https://x/?link_token=tok"
    assert colab._append_link_token("https://x/?a=1", "tok") == "https://x/?a=1&link_token=tok"


def test_mint_same_tab_link_token_exchangeable_once():
    admin = _seed_admin(must_change_password = False)
    token = colab._mint_same_tab_link_token()
    assert isinstance(token, str) and token
    from auth.authentication import exchange_link_token

    # It is a real, single-use token bound to the admin.
    assert exchange_link_token(token) == admin
    assert exchange_link_token(token) is None
