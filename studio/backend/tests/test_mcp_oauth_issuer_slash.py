# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import types

import httpx2
import pytest
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.client.auth.utils import validate_authorization_response_iss
from mcp.shared.auth import OAuthClientInformationFull

from core.inference import mcp_client

URL = "https://calendarmcp.googleapis.com/mcp/v1"
PRM_URL = "https://calendarmcp.googleapis.com/.well-known/oauth-protected-resource/mcp/v1"
BARE = "https://accounts.google.com"
SLASH = "https://accounts.google.com/"


class _ReachedSignIn(Exception):
    pass


def _auth(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_client, "_oauth_token_store", None)
    auth = mcp_client._oauth(URL, "client-id", "client-secret")

    async def no_browser(authorization_url):
        raise _ReachedSignIn(authorization_url)

    auth.context.redirect_handler = no_browser
    return auth


async def _discover(auth, advertised, issuer):
    flow = auth.async_auth_flow(httpx2.Request("POST", URL, json = {}))
    try:
        request = await flow.__anext__()
        challenge = {"WWW-Authenticate": f'Bearer resource_metadata="{PRM_URL}"'}
        request = await flow.asend(httpx2.Response(401, headers = challenge, request = request))
        resource = {"resource": URL, "authorization_servers": [advertised]}
        request = await flow.asend(httpx2.Response(200, json = resource, request = request))
        metadata = {
            "issuer": issuer,
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
        }
        await flow.asend(httpx2.Response(200, json = metadata, request = request))
    finally:
        await flow.aclose()


@pytest.mark.parametrize(
    ("advertised", "issuer"),
    [(SLASH, BARE), (BARE, SLASH), (BARE, BARE), (SLASH, SLASH)],
)
def test_root_issuer_reaches_sign_in(tmp_path, monkeypatch, advertised, issuer):
    auth = _auth(tmp_path, monkeypatch)
    with pytest.raises(_ReachedSignIn, match = "accounts.google.com/o/oauth2/v2/auth"):
        asyncio.run(_discover(auth, advertised, issuer))
    assert auth.context.auth_server_url == advertised
    assert str(auth.context.oauth_metadata.issuer) == issuer


@pytest.mark.parametrize(
    ("advertised", "issuer"),
    [
        ("https://login.example.com/tenant/", "https://login.example.com/tenant"),
        ("https://login.example.com/tenant", "https://login.example.com/tenant/"),
        (SLASH, "https://accounts.example.com"),
        (SLASH, "http://accounts.google.com"),
        (SLASH, "https://accounts.google.com:8443"),
    ],
)
def test_other_issuer_differences_stay_rejected(tmp_path, monkeypatch, advertised, issuer):
    auth = _auth(tmp_path, monkeypatch)
    with pytest.raises(OAuthFlowError, match = "issuer mismatch"):
        asyncio.run(_discover(auth, advertised, issuer))
    assert auth.context.oauth_metadata is None


def test_authorization_response_iss_stays_exact(tmp_path, monkeypatch):
    auth = _auth(tmp_path, monkeypatch)
    with pytest.raises(_ReachedSignIn):
        asyncio.run(_discover(auth, SLASH, BARE))
    metadata = auth.context.oauth_metadata

    validate_authorization_response_iss(BARE, metadata)
    with pytest.raises(OAuthFlowError, match = "iss mismatch"):
        validate_authorization_response_iss(SLASH, metadata)


def test_issuer_stamp_is_stored_as_plain_text(tmp_path, monkeypatch):
    auth = _auth(tmp_path, monkeypatch)
    auth.context.auth_server_url = SLASH
    info = OAuthClientInformationFull(
        client_id = "client-id",
        redirect_uris = auth.context.client_metadata.redirect_uris,
        issuer = auth._expected_issuer(),
    )
    stored = OAuthClientInformationFull.model_validate_json(info.model_dump_json())

    assert type(stored.issuer) is str
    assert stored.issuer == SLASH


def test_provider_without_the_hook_is_left_alone():
    auth = types.SimpleNamespace()
    mcp_client._accept_root_issuer_slash(auth)

    assert not hasattr(auth, "_expected_issuer")


def test_unpatchable_provider_does_not_break_oauth():
    class Frozen:
        def _expected_issuer(self):
            return BARE

        def __setattr__(self, name, value):
            raise AttributeError("frozen")

    auth = Frozen()
    mcp_client._accept_root_issuer_slash(auth)

    assert type(auth._expected_issuer()) is str


def test_non_string_expected_issuer_passes_through():
    auth = types.SimpleNamespace(_expected_issuer = lambda: None)
    mcp_client._accept_root_issuer_slash(auth)

    assert auth._expected_issuer() is None


SAME = [
    (BARE, BARE),
    (SLASH, SLASH),
    (BARE, SLASH),
    ("http://localhost:8080", "http://localhost:8080/"),
    ("https://[::1]:8443", "https://[::1]:8443/"),
    ("https://login.example.com/tenant", "https://login.example.com/tenant"),
]

DIFFERENT = [
    ("https://login.example.com/tenant", "https://login.example.com/tenant/"),
    ("https://login.example.com/", "https://login.example.com/tenant"),
    (BARE, "https://accounts.example.com/"),
    (BARE, "https://accounts.google.com.example.com/"),
    (BARE, "http://accounts.google.com/"),
    (BARE, "https://accounts.google.com:8443/"),
    (BARE, "https://Accounts.google.com/"),
    (SLASH, "https://accounts.google.com//"),
    ("https://accounts.google.com?x=1", "https://accounts.google.com?x=1/"),
    ("https://accounts.google.com#x", "https://accounts.google.com#x/"),
    ("https://[::1", "https://[::1/"),
    ("", "/"),
]


@pytest.mark.parametrize(("a", "b"), SAME)
def test_same_server(a, b):
    for x, y in ((a, b), (b, a)):
        assert mcp_client._RootSlashIssuer(x) == y
        assert y == mcp_client._RootSlashIssuer(x)
        assert not (y != mcp_client._RootSlashIssuer(x))


@pytest.mark.parametrize(("a", "b"), DIFFERENT)
def test_different_server(a, b):
    for x, y in ((a, b), (b, a)):
        assert mcp_client._RootSlashIssuer(x) != y
        assert y != mcp_client._RootSlashIssuer(x)
