# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The served page carries a one-time setup token, never the seeded password.

Replaces the three test_index_bootstrap_{loopback,origin,origin_extra} suites.
Those pinned a "is this request from a local browser" gate that cannot be
written: a same-host reverse proxy with a stock ``proxy_pass
http://127.0.0.1:PORT;`` sends exactly what a genuine local browser sends. The
gate is gone and the payload is now a single-use, short-TTL link token, so what
needs pinning is different -- the seed must never appear in the page under any
request shape, and what does appear must be exchangeable exactly once.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import main as studio_main  # noqa: E402
from auth import authentication, storage  # noqa: E402

_SEED = "seeded-bootstrap-123"


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    yield


class _State:
    def __init__(
        self,
        bootstrap_password = None,
        **extra,
    ):
        self.bootstrap_password = bootstrap_password
        for key, value in extra.items():
            setattr(self, key, value)


class _App:
    def __init__(
        self,
        bootstrap_password = None,
        **extra,
    ):
        self.state = _State(bootstrap_password, **extra)


def _seed_admin(*, must_change_password: bool = True) -> str:
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = _SEED,
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = must_change_password,
    )
    return storage.DEFAULT_ADMIN_USERNAME


_HTML = b"<html><head><title>Unsloth</title></head><body></body></html>"


def test_page_never_contains_the_seeded_password():
    _seed_admin()
    # app.state still carries the seed, as it does in a real process.
    out, nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    assert _SEED.encode() not in out
    assert b"__UNSLOTH_BOOTSTRAP__" in out
    assert nonce


def test_injected_payload_is_a_usable_single_use_token():
    admin = _seed_admin()
    out, _nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    text = out.decode()
    # Pull the token back out of the page exactly as the browser would.
    import json
    import re

    match = re.search(r"window\.__UNSLOTH_BOOTSTRAP__=(\{.*?\})</script>", text)
    assert match, text
    payload = json.loads(match.group(1))
    assert payload["username"] == admin
    assert "password" not in payload

    token = payload["link_token"]
    assert authentication.exchange_link_token(token) == admin
    # Single use: the second exchange is refused.
    assert authentication.exchange_link_token(token) is None


def test_two_page_loads_get_independent_tokens():
    # Minted per response, so two browsers opening setup do not race for one
    # token and burn each other's.
    _seed_admin()
    first, _ = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    second, _ = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    assert first != second


def test_setup_token_outlives_the_time_an_operator_takes_to_type(monkeypatch):
    """The page token is minted on LOAD and redeemed on SUBMIT.

    The default link-token TTL suits a URL handoff redeemed in seconds. Reusing
    it here would expire under anyone who opened Studio, was interrupted, and
    came back a few minutes later, turning a first login that works today into
    an error. It is bound to the bootstrap deadline instead: Studio shuts down
    at that point anyway, so the token cannot outlive the window in which the
    seed it replaces would have been usable.
    """
    from datetime import datetime, timezone

    _seed_admin()
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))

    row = storage.get_connection().execute("SELECT expires_at FROM link_tokens").fetchone()
    remaining = (datetime.fromisoformat(row[0]) - datetime.now(timezone.utc)).total_seconds()
    assert remaining > authentication.LINK_TOKEN_EXPIRE_SECONDS, (
        "the setup token fell back to the short URL-handoff TTL; an operator who "
        "leaves the setup page open would get an error instead of a first login"
    )


def test_nothing_is_injected_once_a_password_is_set():
    _seed_admin(must_change_password = False)
    out, nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    assert out == _HTML
    assert nonce is None


def test_injection_does_not_depend_on_the_request():
    """The gate is gone on purpose; assert it has not crept back.

    _inject_bootstrap takes no Request, so there is nothing for a forged Origin,
    Host or X-Forwarded-For to influence. This is the property the deleted
    loopback/origin suites were trying and failing to guarantee.
    """
    import inspect

    params = inspect.signature(studio_main._inject_bootstrap).parameters
    assert "request" not in params, (
        "a per-request gate reappeared; a same-host reverse proxy is "
        "indistinguishable from a local browser, so it cannot be correct"
    )
    for gone in (
        "_should_inject_bootstrap",
        "_is_local_bootstrap_request",
        "_is_same_origin_request",
        "_host_header_is_loopback",
    ):
        assert not hasattr(studio_main, gone), f"{gone} was reintroduced"


def test_a_headless_public_launch_injects_nothing():
    """A public URL must carry no credential at all, seeded or minted.

    run.py's pre-bind gate nulls app.state.bootstrap_password (and sets
    suppress_bootstrap_injection) when a public Cloudflare URL is about to
    serve. That defence used to work because _inject_bootstrap read the seed
    from app.state, so a None there meant an empty page. A token that can set
    the first password is still an admin credential, so it has to obey the same
    gate: handing one to whoever loads the public URL first would be strictly
    worse than what this change set out to fix.
    """
    _seed_admin()
    app = _App(bootstrap_password = None, suppress_bootstrap_injection = True)
    out, nonce = studio_main._inject_bootstrap(_HTML, app)
    assert out == _HTML, "a public launch served a usable setup credential"
    assert nonce is None
    assert storage.get_connection().execute("SELECT COUNT(*) FROM link_tokens").fetchone()[0] == 0


def test_a_stripped_bootstrap_file_still_protects_version_independently():
    """unsloth_cli deletes .bootstrap_password before a public re-exec.

    Its docstring calls the removal itself the protection, precisely so that a
    re-exec'd child of ANY version is covered rather than relying on the child
    running a particular gate. Gating the mint on the seed's availability is
    what preserves that: no seed on disk, no token in the page.
    """
    _seed_admin()
    storage.clear_bootstrap_password()
    # Whatever the child does at startup, it now reads no seed into app.state.
    out, nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = None))
    assert out == _HTML
    assert nonce is None


def test_token_in_page_cannot_change_an_existing_password():
    """The injected credential's only power is setting the FIRST password."""
    admin = _seed_admin()
    out, _ = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    import json
    import re

    payload = json.loads(
        re.search(r"window\.__UNSLOTH_BOOTSTRAP__=(\{.*?\})</script>", out.decode()).group(1)
    )
    # Someone completes setup first.
    assert storage.update_password(admin, "chosen-elsewhere-789") is not None
    # The rotation revoked outstanding link tokens in the same transaction.
    assert authentication.exchange_link_token(payload["link_token"]) is None
