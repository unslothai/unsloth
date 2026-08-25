# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Adversarial probes for keyless API access, from the review of PR #9102.

Each targets a property the merged suite asserts only at the predicate layer,
only in one direction, or not at all. Separate file so that suite is untouched.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auth import storage
from auth.authentication import (
    KEYLESS_FALLBACK_SCHEME,
    KEYLESS_SCHEME,
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
    security,
)
from utils import host_policy
from utils.keyless_api_access import (
    KEYLESS_ADMISSION_STATE_KEY,
    KeylessToolPolicyMiddleware,
    _reset_scope_cache,
    asgi_request_is_keyless,
    keyless_request_allowed,
    scope_covers,
    set_keyless_api_access,
)


@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    # The merged suite leaves this latched on, masking the transport checks below.
    monkeypatch.setattr(host_policy, "_remote_connector_active", False, raising = False)
    monkeypatch.setattr(host_policy, "_lan_connector_active", False, raising = False)
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


def seed_user():
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
    )


def app_state(**overrides):
    state = SimpleNamespace(
        bind_host = "127.0.0.1",
        secure = False,
        remote_access_is_colab = False,
        lan_access_is_colab = False,
        lan_access_secure_launch = False,
        cloudflare_url = None,
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def asgi_scope(
    *,
    path = "/v1/chat/completions",
    method = None,
    root_path = "",
    headers = None,
    state = None,
    server = ("127.0.0.1", 8000),
    client = ("127.0.0.1", 50000),
):
    return {
        "type": "http",
        "method": method or ("GET" if path.startswith("/v1/models") else "POST"),
        "path": path,
        "root_path": root_path,
        "query_string": b"",
        "scheme": "http",
        "server": server,
        "client": client,
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
        ],
        "app": SimpleNamespace(state = state or app_state()),
    }


def request_for(**kwargs):
    return Request(asgi_scope(**kwargs))


def resolve(request):
    return asyncio.run(security(request))


# ── the headline invariant: off means off, through the dependency itself ──────


def test_scope_off_is_refused_by_the_security_dependency_not_only_the_predicate():
    """The merged suite asserts scope=off at `scope_covers` level. Assert it where it counts."""
    seed_user()
    set_keyless_api_access("off")
    with pytest.raises(HTTPException) as caught:
        resolve(request_for())
    assert caught.value.status_code in (401, 403)
    # ...and the dummy bearers must not resurrect it either.
    for dummy in ("not-needed", "lm-studio", "ollama"):
        with pytest.raises(HTTPException):
            asyncio.run(
                get_current_subject(
                    resolve(request_for(headers = {"Authorization": f"Bearer {dummy}"}))
                )
            )


# ── privilege escalation: can a keyless caller widen its own grant? ───────────


def test_a_keyless_caller_cannot_widen_its_own_scope():
    """`_require_ui_session_for_keyless` is the only thing stopping self-promotion.

    Untested elsewhere, and it rests entirely on `authenticated_via_api_key`
    reporting True for a keyless caller.
    """
    from routes.settings import _require_ui_session_for_keyless

    seed_user()
    set_keyless_api_access("full", tools = False)
    credentials = resolve(request_for(path = "/api/settings/keyless-api-access", method = "PUT"))
    assert credentials.scheme == KEYLESS_SCHEME
    assert asyncio.run(authenticated_via_api_key(credentials)) is True
    with pytest.raises(HTTPException) as caught:
        _require_ui_session_for_keyless(via_api_key = True)
    assert caught.value.status_code == 403

    # An sk-unsloth key is held back by the same guard.
    raw_key, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "probe",
        expires_at = None,
    )
    key_credentials = resolve(
        request_for(
            path = "/api/settings/keyless-api-access",
            method = "PUT",
            headers = {"Authorization": f"Bearer {raw_key}"},
        )
    )
    assert asyncio.run(authenticated_via_api_key(key_credentials)) is True


# ── transport: the inference limb, isolated from the tunnel flag ──────────────


def test_inference_is_refused_from_a_public_bind_and_a_public_peer():
    """Nothing exercises the inference limb with a genuinely public transport."""
    seed_user()
    set_keyless_api_access("inference")
    public = app_state(bind_host = "64.227.100.5")
    assert (
        keyless_request_allowed(
            request_for(server = ("64.227.100.5", 8000), client = ("8.8.8.8", 51000), state = public)
        )
        is False
    )
    # CGNAT is not private either.
    assert (
        keyless_request_allowed(
            request_for(
                server = ("100.64.0.10", 8000),
                client = ("100.64.0.11", 51000),
                state = app_state(bind_host = "100.64.0.10"),
            )
        )
        is False
    )
    # A private peer arriving on a loopback socket is still not LAN admission.
    assert (
        keyless_request_allowed(
            request_for(server = ("127.0.0.1", 8000), client = ("192.168.1.90", 51000))
        )
        is False
    )


def test_full_scope_denials_survive_without_the_tunnel_flag():
    """The merged wildcard/LAN denials pass even with `_full_scope_transport_allowed` gone.

    `_remote_connector_active` is left True there, so `_public_tunnel_active`
    short-circuits first. With it cleared, the loopback rule has to carry them.
    """
    seed_user()
    set_keyless_api_access("full")
    assert keyless_request_allowed(request_for()) is True  # control: loopback works

    for bind in ("0.0.0.0", "::"):
        assert (
            keyless_request_allowed(request_for(state = app_state(bind_host = bind))) is False
        ), f"wildcard bind {bind} admitted under full scope"

    assert (
        keyless_request_allowed(
            request_for(
                server = ("192.168.1.24", 8888),
                client = ("192.168.1.90", 51000),
                state = app_state(bind_host = "192.168.1.24"),
            )
        )
        is False
    )


def test_every_hosted_mode_flag_closes_full_and_inference():
    """`lan_access_is_colab` and `lan_access_secure_launch` are otherwise unexercised."""
    seed_user()
    for scope in ("inference", "full"):
        set_keyless_api_access(scope)
        for flag in (
            "remote_access_is_colab",
            "lan_access_is_colab",
            "secure",
            "lan_access_secure_launch",
        ):
            assert (
                keyless_request_allowed(request_for(state = app_state(**{flag: True}))) is False
            ), f"{flag} did not close scope={scope}"
        assert (
            keyless_request_allowed(
                request_for(state = app_state(cloudflare_url = "https://x.trycloudflare.com"))
            )
            is False
        ), f"active tunnel did not close scope={scope}"


# ── route topology ───────────────────────────────────────────────────────────


def test_management_routes_are_never_covered_by_inference_scope():
    """Only the positive `full` form is asserted for /api/* elsewhere."""
    for method, path in (
        ("POST", "/api/train/start"),
        ("PUT", "/api/settings/keyless-api-access"),
        ("POST", "/api/auth/api-keys"),
        ("GET", "/api/auth/api-keys"),
        ("POST", "/api/mcp-servers/"),
    ):
        assert scope_covers("inference", method, path) is False


def test_root_path_and_trailing_slash_reach_the_same_verdict_end_to_end():
    """`root_path` is read off the ASGI scope but never driven through the entry point."""
    seed_user()
    set_keyless_api_access("inference")
    assert (
        keyless_request_allowed(
            request_for(path = "/studio/v1/models/", root_path = "/studio", method = "GET")
        )
        is True
    )
    assert (
        keyless_request_allowed(
            request_for(path = "/studio/v1/load", root_path = "/studio", method = "POST")
        )
        is False
    )
    # prefix confusion: a sibling mount must not borrow the root's allowlist
    assert (
        keyless_request_allowed(
            request_for(path = "/studio-v2/v1/models", root_path = "/studio", method = "GET")
        )
        is False
    )


def test_traversal_shaped_paths_never_borrow_an_allowlisted_route():
    for method, path in (
        ("POST", "/v1/chat/completions/../../v1/load"),
        ("POST", "/v1/models/../load"),
        ("POST", "/v1//load"),
        ("POST", "/v1/chat/completions/%2e%2e/load"),
        ("POST", "/v1/load;/v1/chat/completions"),
    ):
        assert scope_covers("inference", method, path) is False, f"{method} {path} was covered"


def test_the_dynamic_model_prefix_can_only_ever_reach_model_retrieval():
    """`scope_covers` admits every non-empty `GET /v1/models/...` suffix, not an exact pair.

    That is broader than the "exact HTTP method + normalized path allowlist" the PR
    describes, so the safety argument rests entirely on route topology: nothing but
    `openai_retrieve_model` can match those paths. Pin that, because the day another
    `GET /models/...` route is registered the allowlist silently grows with it.
    """
    from starlette.routing import Match

    from routes.inference import router

    traversals = [
        "/v1/models/../../api/train/start",
        "/v1/models/../load",
        "/v1/models/..%2f..%2fload",
        "/v1/models/../../auth/api-keys",
    ]
    for path in traversals:
        assert scope_covers("inference", "GET", path) is True  # documents the breadth
        matched = [
            route.path
            for route in router.routes
            if route.matches(
                {
                    "type": "http",
                    "method": "GET",
                    "path": path.removeprefix("/v1"),
                    "root_path": "",
                    "headers": [],
                }
            )[0]
            is not Match.NONE
        ]
        assert matched == ["/models/{model_id:path}"], f"{path} reached {matched}"


# ── credential precedence ────────────────────────────────────────────────────


def test_a_session_jwt_naming_an_unknown_subject_is_refused():
    """Covered for expired sessions, not for a well-formed token naming nobody."""
    seed_user()
    set_keyless_api_access("full")
    _salt, _hash, jwt_secret, _must_change = storage.get_user_and_secret(
        storage.DEFAULT_ADMIN_USERNAME
    )
    forged = jwt.encode(
        {"sub": "ghost", "exp": datetime.now(timezone.utc) + timedelta(minutes = 30)},
        jwt_secret,
        algorithm = "HS256",
    )
    with pytest.raises(HTTPException):
        asyncio.run(
            get_current_subject(resolve(request_for(headers = {"Authorization": f"Bearer {forged}"})))
        )


def test_the_asgi_twin_agrees_with_the_dependency_on_header_shapes():
    """`asgi_request_is_keyless` is imported by the merged suite and never called.

    A second implementation of the duplicate-header and dummy-bearer rules; only
    the `_BearerOrKeyless` copy is otherwise tested.
    """
    seed_user()
    set_keyless_api_access("inference")
    assert asgi_request_is_keyless(asgi_scope()) is True
    for dummy in ("not-needed", "lm-studio", "ollama"):
        assert (
            asgi_request_is_keyless(asgi_scope(headers = {"Authorization": f"Bearer {dummy}"}))
            is True
        )
    for hostile in (
        "Bearer sk-unsloth-nope",
        "Bearer",
        "Basic bm90LW5lZWRlZA==",
        "bearer  not-needed",
        "Bearer not-needed-extra",
    ):
        assert (
            asgi_request_is_keyless(asgi_scope(headers = {"Authorization": hostile})) is False
        ), hostile

    duplicated = asgi_scope()
    duplicated["headers"] = [
        (b"authorization", b"Bearer not-needed"),
        (b"authorization", b"Bearer not-needed"),
    ]
    assert asgi_request_is_keyless(duplicated) is False


def test_a_cross_site_page_cannot_reach_keyless_without_sending_origin():
    """`Origin` alone does not identify a browser.

    No engine attaches it to a same-origin GET or to a cross-site GET made in
    `no-cors` mode, and only Chromium withholds such a fetch from
    `http://127.0.0.1:<port>` (Private Network Access). Firefox and Safari send it.
    `Sec-Fetch-Site` is what actually says who initiated the request, and a page
    cannot forge it -- the `Sec-` prefix makes it a forbidden header name.
    """
    seed_user()
    for scope_name in ("inference", "full"):
        set_keyless_api_access(scope_name)
        for site in ("cross-site", "same-site", "CROSS-SITE", " cross-site "):
            assert (
                keyless_request_allowed(
                    request_for(path = "/v1/models", method = "GET", headers = {"Sec-Fetch-Site": site})
                )
                is False
            ), f"{site!r} was admitted under {scope_name}"

        # a page on Studio's own origin, and the user typing the URL, are not attacks
        for site in ("same-origin", "none"):
            assert (
                keyless_request_allowed(
                    request_for(path = "/v1/models", method = "GET", headers = {"Sec-Fetch-Site": site})
                )
                is True
            ), f"{site!r} was refused under {scope_name}"

        # absence must stay admitted: curl, the OpenAI SDKs and Safari < 16.4 send
        # no Sec-Fetch-* at all, and serving them is the entire point of the setting
        assert keyless_request_allowed(request_for(path = "/v1/models", method = "GET")) is True


def test_a_real_credential_authenticates_under_every_scope_and_transport(monkeypatch):
    """The setting adds an admission path. It must never take one away.

    A working key or session has to keep authenticating exactly as before, on
    every scope and on every transport -- including the ones keyless itself is
    refused on, since a usable bearer is resolved before any scope or transport
    check runs. It also has to authenticate *as itself*: a keyless scheme would
    hand an existing API client the keyless tool restriction it never had.
    """
    import lan_access

    seed_user()
    raw_key, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "always-on",
        expires_at = None,
    )
    _s, _h, jwt_secret, _m = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    session = jwt.encode(
        {
            "sub": storage.DEFAULT_ADMIN_USERNAME,
            "exp": datetime.now(timezone.utc) + timedelta(minutes = 30),
        },
        jwt_secret,
        algorithm = "HS256",
    )

    transports = {
        "loopback": dict(
            server = ("127.0.0.1", 8000), client = ("127.0.0.1", 51000), state = app_state()
        ),
        "private_lan": dict(
            server = ("192.168.1.24", 8888),
            client = ("192.168.1.90", 51000),
            state = app_state(bind_host = "0.0.0.0"),
        ),
        "public": dict(
            server = ("64.227.100.5", 8000),
            client = ("8.8.8.8", 51000),
            state = app_state(bind_host = "64.227.100.5"),
        ),
        "tunnel": dict(
            server = ("127.0.0.1", 8000),
            client = ("127.0.0.1", 51000),
            state = app_state(cloudflare_url = "https://x.trycloudflare.com"),
        ),
        "colab": dict(
            server = ("127.0.0.1", 8000),
            client = ("127.0.0.1", 51000),
            state = app_state(remote_access_is_colab = True),
        ),
        "secure": dict(
            server = ("127.0.0.1", 8000), client = ("127.0.0.1", 51000), state = app_state(secure = True)
        ),
        "browser_origin": dict(
            server = ("127.0.0.1", 8000),
            client = ("127.0.0.1", 51000),
            state = app_state(),
            headers = {"Origin": "https://evil.example"},
        ),
        "browser_cross_site": dict(
            server = ("127.0.0.1", 8000),
            client = ("127.0.0.1", 51000),
            state = app_state(),
            headers = {"Sec-Fetch-Site": "cross-site"},
        ),
    }

    for scope_name in ("off", "inference", "full"):
        set_keyless_api_access(scope_name)
        for label, transport in transports.items():
            # via monkeypatch, so the stub cannot outlive this test: it is a module
            # global, and test_lan_access_settings.py reads the real one
            monkeypatch.setattr(
                lan_access,
                "lan_listener_status",
                lambda t = transport: {
                    "running": True,
                    "port": t["server"][1],
                    "addresses": [t["server"][0]],
                    "error": None,
                },
            )
            for credential_name, token in (("api_key", raw_key), ("session", session)):
                headers = dict(transport.get("headers") or {})
                headers["Authorization"] = f"Bearer {token}"
                request = request_for(
                    server = transport["server"],
                    client = transport["client"],
                    state = transport["state"],
                    headers = headers,
                )
                credentials = resolve(request)
                assert credentials.scheme not in (
                    KEYLESS_SCHEME,
                    KEYLESS_FALLBACK_SCHEME,
                ), f"{credential_name} was downgraded to keyless on {label}/{scope_name}"
                assert (
                    asyncio.run(get_current_subject(credentials)) == storage.DEFAULT_ADMIN_USERNAME
                ), f"{credential_name} stopped working on {label}/{scope_name}"

    # and the credentials that must NOT work still do not, at the widest scope
    set_keyless_api_access("full")
    storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
    expired, _row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "expired",
        expires_at = (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat(),
    )
    for dead in (raw_key, expired):
        with pytest.raises(HTTPException):
            asyncio.run(
                get_current_subject(
                    resolve(request_for(headers = {"Authorization": f"Bearer {dead}"}))
                )
            )


def test_a_rebound_hostname_cannot_pose_as_a_local_client():
    """DNS rebinding produces every local signal the socket checks look at.

    A page on `evil.example` whose record is re-pointed at 127.0.0.1 keeps its own
    origin, so the fetch is same-origin: no `Origin`, `Sec-Fetch-Site: same-origin`,
    loopback peer on a loopback socket. Unlike the `no-cors` case the response is
    readable, so under `full` this reads local admin data. `Host` is the one header
    that still names the page's own domain.
    """
    seed_user()
    for scope_name in ("inference", "full"):
        set_keyless_api_access(scope_name)
        path = "/v1/models" if scope_name == "inference" else "/api/chat/threads"
        for hostile in (
            "evil.example:8888",
            "localhost.evil.example:8888",
            "evil.example",
            "127.0.0.1.evil.example:8888",
            "[::1]evil.example",
            "studio.internal:8888",
        ):
            assert (
                keyless_request_allowed(
                    request_for(
                        path = path,
                        method = "GET",
                        headers = {"Host": hostile, "Sec-Fetch-Site": "same-origin"},
                    )
                )
                is False
            ), f"{hostile} was admitted under {scope_name}"

        # a client that addressed the socket directly is unaffected
        for direct in (
            "127.0.0.1:8888",
            "localhost:8888",
            "[::1]:8888",
            "127.0.0.1",
            "LOCALHOST:8888",
            "localhost.:8888",
        ):
            assert (
                keyless_request_allowed(
                    request_for(path = path, method = "GET", headers = {"Host": direct})
                )
                is True
            ), f"{direct} was refused under {scope_name}"

        # and no Host at all stays admitted: the merged suite sends none
        assert keyless_request_allowed(request_for(path = path, method = "GET")) is True


# ── the reported race, pinned deterministically ──────────────────────────────


def test_revoking_a_key_after_the_admission_snapshot_never_yields_the_admin():
    """Regression for the PR #9102 race reported by @Imagineer99.

    Reproduced on 99091aba7 and fixed by 10ecfe9d4; the reporter's own repro can
    no longer run on head because it monkeypatches a function the classifier no
    longer calls. This pins the interleaving directly instead: revoke between the
    middleware snapshot and the credential check.
    """
    from state.tool_policy import (
        get_tool_policy_default,
        reset_tool_policy,
        set_tool_policy_default,
    )

    seed_user()
    set_keyless_api_access("inference", tools = False)
    reset_tool_policy()
    set_tool_policy_default(True)
    raw_key, row = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME,
        name = "race",
        expires_at = None,
    )
    scope = asgi_scope(headers = {"Authorization": f"Bearer {raw_key}"})
    observed = {}

    async def downstream(asgi, receive, send):
        observed["snapshot"] = asgi.get("state", {}).get(KEYLESS_ADMISSION_STATE_KEY)
        observed["tools"] = get_tool_policy_default()
        # the race: the key dies after the middleware classified the request
        storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
        credentials = await security(Request(asgi, receive))
        observed["scheme"] = credentials.scheme
        try:
            await get_current_credential(credentials)
            observed["subject_resolved"] = True
        except HTTPException as error:
            observed["status"] = error.status_code

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        return None

    try:
        asyncio.run(KeylessToolPolicyMiddleware(downstream)(scope, receive, send))
    finally:
        reset_tool_policy()

    assert observed["snapshot"] is False, "a request carrying a real key was classified keyless"
    assert observed["scheme"] not in (KEYLESS_SCHEME, KEYLESS_FALLBACK_SCHEME)
    assert observed["scheme"] == "Bearer"
    assert observed.get("subject_resolved") is not True, "revoked key resolved to a subject"
    assert observed["status"] == 401
    # tools stay at the caller's own policy: an API-key client is not keyless, so
    # the grant it already had is not taken away. It never reaches a handler anyway.
    assert observed["tools"] is True
