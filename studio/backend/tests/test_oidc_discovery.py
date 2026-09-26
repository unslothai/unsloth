# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import httpx
import pytest

from auth.oidc_config import load_oidc_config
from auth.oidc_discovery import OIDCDiscoveryClient, OIDCProviderUnavailable


def _config():
    return load_oidc_config(
        {
            "UNSLOTH_OIDC_ENABLED": "true",
            "UNSLOTH_OIDC_ISSUER": "https://auth.example.com/realms/company",
            "UNSLOTH_OIDC_CLIENT_ID": "unsloth",
            "UNSLOTH_OIDC_CLIENT_SECRET": "test-only-secret",
            "UNSLOTH_OIDC_REDIRECT_URI": "https://studio.example.com/api/auth/oidc/callback",
        }
    )


def _metadata(**overrides):
    result = {
        "issuer": "https://auth.example.com/realms/company",
        "authorization_endpoint": "https://auth.example.com/realms/company/protocol/openid-connect/auth",
        "token_endpoint": "https://auth.example.com/realms/company/protocol/openid-connect/token",
        "jwks_uri": "https://auth.example.com/realms/company/protocol/openid-connect/certs",
    }
    result.update(overrides)
    return result


def test_discovery_and_jwks_are_cached_until_ttl_expires():
    calls = []
    now = [100.0]

    def handler(request):
        calls.append(str(request.url))
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json = _metadata())
        return httpx.Response(200, json = {"keys": [{"kid": "key-1", "kty": "RSA"}]})

    client = OIDCDiscoveryClient(
        _config(),
        cache_ttl_seconds = 60,
        clock = lambda: now[0],
        transport = httpx.MockTransport(handler),
    )

    assert client.metadata().issuer == _config().issuer
    assert client.metadata().issuer == _config().issuer
    assert client.jwks()["keys"][0]["kid"] == "key-1"
    assert client.jwks()["keys"][0]["kid"] == "key-1"
    assert len(calls) == 2

    now[0] += 61
    client.metadata()
    client.jwks()
    assert len(calls) == 4


def test_discovery_rejects_issuer_mismatch():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json = _metadata(issuer = "https://evil.example"))
    )

    with pytest.raises(OIDCProviderUnavailable, match = "issuer"):
        OIDCDiscoveryClient(_config(), transport = transport).metadata()


def test_discovery_rejects_insecure_endpoint_scheme():
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json = _metadata(token_endpoint = "http://auth.example.com/token"),
        )
    )

    with pytest.raises(OIDCProviderUnavailable, match = "different URL scheme"):
        OIDCDiscoveryClient(_config(), transport = transport).metadata()


def test_provider_failure_is_wrapped_without_response_details():
    transport = httpx.MockTransport(lambda _request: httpx.Response(503, text = "secret details"))

    with pytest.raises(OIDCProviderUnavailable, match = "discovery is unavailable") as error:
        OIDCDiscoveryClient(_config(), transport = transport).metadata()

    assert "secret details" not in str(error.value)


def test_jwks_requires_a_nonempty_key_set():
    def handler(request):
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json = _metadata())
        return httpx.Response(200, json = {"keys": []})

    client = OIDCDiscoveryClient(_config(), transport = httpx.MockTransport(handler))
    with pytest.raises(OIDCProviderUnavailable, match = "invalid signing keys"):
        client.jwks()
