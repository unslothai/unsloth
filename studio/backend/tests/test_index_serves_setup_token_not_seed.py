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
    # Minted per response, so two browsers opening setup do not burn each other's.
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


def _token_ttl_seconds() -> float:
    from datetime import datetime, timezone
    row = storage.get_connection().execute("SELECT expires_at FROM link_tokens").fetchone()
    return (datetime.fromisoformat(row[0]) - datetime.now(timezone.utc)).total_seconds()


def test_a_loopback_launch_token_is_not_bound_to_an_hour(monkeypatch):
    """No deadline arms on a loopback launch, so nothing is going to shut down.

    The seed this replaces stayed usable for as long as the process ran. Bounding
    the token to the default hour here would protect nothing and would turn "left
    the setup tab open over lunch" into an error the seed never produced.
    """
    _seed_admin()
    monkeypatch.delenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", raising = False)
    app = _App(bootstrap_password = _SEED, bind_host = "127.0.0.1", secure = False)
    studio_main._inject_bootstrap(_HTML, app)

    from auth.bootstrap_timeout import DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS

    assert (
        _token_ttl_seconds() > DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
    ), "the setup token expires in an hour on a launch that never shuts down"


def test_an_exposed_launch_token_is_bound_by_the_shutdown_deadline(monkeypatch):
    """A deadline arms, so the token must not outlive the window it occupies."""
    _seed_admin()
    monkeypatch.setenv("UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT", "900")
    app = _App(bootstrap_password = _SEED, bind_host = "0.0.0.0", secure = False)
    studio_main._inject_bootstrap(_HTML, app)

    ttl = _token_ttl_seconds()
    assert ttl <= 900, f"token outlives the deadline that will stop Studio ({ttl}s)"
    assert ttl > authentication.LINK_TOKEN_EXPIRE_SECONDS


def test_nothing_is_injected_once_a_password_is_set():
    _seed_admin(must_change_password = False)
    out, nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    assert out == _HTML
    assert nonce is None


def test_the_unwritable_half_of_the_gate_stays_deleted():
    """No peer-address or forwarding-header test may come back.

    "Is this request from a local browser" cannot be answered: a same-host
    reverse proxy with a stock `proxy_pass http://127.0.0.1:PORT;` sends
    byte-identical bytes to a genuine local browser, so a peer address or an
    X-Forwarded-For cannot separate them.

    Two checks that DO exist are deliberately not in this list, because they ask
    answerable questions. `_is_same_origin_request` asks which origin is calling,
    which the browser reports honestly and script cannot forge.
    `_host_is_safe_from_rebinding` asks whether this Host could have been chosen
    by someone other than the operator, which is Host ALLOWLISTING, not the
    loopback test: it never tries to identify a proxy, it only refuses names the
    operator never configured.
    """
    for gone in (
        "_should_inject_bootstrap",
        "_is_local_bootstrap_request",
        "_host_header_is_loopback",
        "_is_loopback_ip",
        "_PROXIED_CLIENT_HEADERS",
    ):
        assert not hasattr(studio_main, gone), f"{gone} was reintroduced"


def test_a_cross_origin_request_is_not_same_origin():
    """The check that keeps a hostile page from reading the setup token.

    Studio's default CORS is allow_origins=["*"] with credentials, so any page
    the operator visits can fetch this index and read the body. Measured in
    Chromium against a real install before this check was restored: cross-origin
    GET / -> link-exchange 200 -> link-initial-password 200, and the attacker's
    password then logged in.
    """

    class _Req:
        def __init__(
            self,
            origin,
            netloc = "127.0.0.1:8990",
            scheme = "http",
        ):
            self.headers = {} if origin is None else {"origin": origin}
            self.url = type("U", (), {"scheme": scheme, "netloc": netloc})()

    assert studio_main._is_same_origin_request(_Req("http://evil.example")) is False
    assert studio_main._is_same_origin_request(_Req("http://localhost:8990")) is False
    assert studio_main._is_same_origin_request(_Req("null")) is False
    assert studio_main._is_same_origin_request(_Req("")) is False
    # A top-level navigation sends no Origin, and is how the operator arrives.
    assert studio_main._is_same_origin_request(_Req(None)) is True
    assert studio_main._is_same_origin_request(_Req("http://127.0.0.1:8990")) is True
    # Default ports are stripped by browsers (RFC 6454) and case is insensitive.
    assert (
        studio_main._is_same_origin_request(_Req("HTTP://127.0.0.1", netloc = "127.0.0.1:80")) is True
    )


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


def test_a_rotation_racing_the_mint_leaves_no_usable_token(monkeypatch):
    """The window between "setup is pending" and recording the nonce.

    _inject_bootstrap checks requires_password_change, then create_link_token
    reads the JWT secret and records the nonce. A rotation committing in between
    would mint against the NEW secret after update_password had deleted the old
    nonces, leaving a token that still exchanges once setup is complete and hands
    back an ordinary session. Forced here by rotating inside the mint.
    """
    admin = _seed_admin()
    real_save = storage.save_link_token
    fired = {"n": 0}

    def _rotate_then_save(jti, username, expires_at, **kwargs):
        if fired["n"] == 0:
            fired["n"] = 1
            # Setup completes elsewhere, between the guard and this write.
            storage.update_password(admin, "chosen-by-the-operator-1")
        return real_save(jti, username, expires_at, **kwargs)

    monkeypatch.setattr(storage, "save_link_token", _rotate_then_save)
    monkeypatch.setattr(authentication, "save_link_token", _rotate_then_save)

    out, nonce = studio_main._inject_bootstrap(_HTML, _App(bootstrap_password = _SEED))
    assert fired["n"] == 1, "the race was never triggered, so this proves nothing"
    assert out == _HTML, "a token was injected after setup had already completed"
    assert nonce is None
    assert storage.requires_password_change(admin) is False
    conn = storage.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM link_tokens").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_rebound_dns_name_is_refused_the_setup_token():
    """The gap same-origin alone leaves open.

    In a rebinding attack the operator visits attacker.example, which re-resolves
    to this listener. The browser then sends Host: attacker.example and either no
    Origin (top-level GET) or an Origin equal to that Host, so the same-origin
    check PASSES. Host allowlisting is the standard remedy for this, as in
    webpack allowedHosts, Django ALLOWED_HOSTS and Rails HostAuthorization.
    """

    class _Req:
        def __init__(self, host):
            self.headers = {"host": host}
            self.url = type("U", (), {"scheme": "http", "netloc": host})()

    class _A:
        state = type("S", (), {"bind_host": "0.0.0.0"})()

    app = _A()
    # Hostile names, including one that merely contains a loopback label.
    for hostile in (
        "attacker.example",
        "evil.test:8000",
        "localhost.attacker.example",
        "127.0.0.1.attacker.example",
    ):
        assert studio_main._host_is_safe_from_rebinding(_Req(hostile), app) is False, hostile
    # Loopback, however spelled.
    for ok in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "LOCALHOST"):
        assert studio_main._host_is_safe_from_rebinding(_Req(ok), app) is True, ok
    # An IP literal is not rebindable: a browser sends one only when typed, which
    # is what keeps `-H 0.0.0.0` usable from another machine.
    for ok in ("192.168.1.50:8000", "10.0.0.5:8000", "[fe80::1]:8000"):
        assert studio_main._host_is_safe_from_rebinding(_Req(ok), app) is True, ok

    # A name is allowed only when it is the host this launch was configured with.
    class _Named:
        state = type("S", (), {"bind_host": "studio.internal"})()

    assert studio_main._host_is_safe_from_rebinding(_Req("studio.internal:8000"), _Named()) is True
    assert studio_main._host_is_safe_from_rebinding(_Req("attacker.example"), _Named()) is False


def test_a_missing_host_header_is_refused():
    class _Req:
        headers = {}
        url = type("U", (), {"scheme": "http", "netloc": "127.0.0.1:8000"})()

    class _A:
        state = type("S", (), {"bind_host": "127.0.0.1"})()

    assert studio_main._host_is_safe_from_rebinding(_Req(), _A()) is False


def test_colab_notebook_proxy_still_gets_the_setup_token(monkeypatch):
    """The regression the rebinding guard introduced, and the merge base's rule.

    Colab serves Studio through Google's single-user proxy: the server binds
    0.0.0.0 and the browser sends the proxy's hostname in Host. That is neither a
    loopback name, nor an IP literal, nor the configured bind, so the guard
    refused it and a fresh notebook lost automatic first-boot setup entirely --
    the operator would have to read .bootstrap_password out of the runtime by
    hand. The merge base allowed this case explicitly; restoring it also restores
    its single exception, the shareable Cloudflare link.
    """

    class _Req:
        def __init__(self, host, headers = None):
            self.headers = {"host": host, **(headers or {})}
            self.url = type("U", (), {"scheme": "https", "netloc": host})()

    class _A:
        state = type("S", (), {"bind_host": "0.0.0.0"})()

    app = _A()
    proxy = _Req("abc123-colab.prod.colab.dev")

    monkeypatch.setattr(studio_main, "_IS_COLAB", False)
    assert studio_main._host_is_safe_from_rebinding(proxy, app) is False

    monkeypatch.setattr(studio_main, "_IS_COLAB", True)
    assert studio_main._host_is_safe_from_rebinding(proxy, app) is True
    # A shareable Cloudflare link marks its visitors, who are not the notebook's
    # owner. Withheld even on loopback, as at the merge base.
    tunnel = _Req("localhost:8000", {"cf-connecting-ip": "203.0.113.7"})
    assert studio_main._host_is_safe_from_rebinding(tunnel, app) is False


def test_the_index_mints_its_token_off_the_event_loop():
    """Minting opens SQLite under BEGIN IMMEDIATE, so it must not run on the loop.

    While setup is pending EVERY index GET mints a link token. A concurrent auth
    writer holding the database lock makes that wait out the busy timeout, and on
    the event loop the wait is charged to every other request in flight, not just
    this one. /link-exchange is a plain `def` for the same reason.

    Asserted on the source rather than by racing a lock: the property is "this
    call is not awaited inline", which a timing test can only sample.
    """
    import inspect
    import re

    source = inspect.getsource(studio_main.setup_frontend)
    for handler in ("serve_root", "serve_frontend"):
        body = source.split(f"def {handler}(", 1)[1].split("\n    @app.get", 1)[0]
        assert "_build_index_response" in body, handler
        for call in re.findall(r"[^\n]*_build_index_response\([^\n]*", body):
            assert "run_in_threadpool" in call, (handler, call.strip())
