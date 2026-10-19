# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from auth.oidc_config import load_oidc_config
from auth.oidc_tokens import OIDCAuthenticationError, validate_id_token


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


class _Discovery:
    def __init__(self, jwks):
        self.value = jwks
        self.refreshes = 0

    def jwks(self, *, force_refresh = False):
        if force_refresh:
            self.refreshes += 1
        return self.value


@pytest.fixture
def signing_material():
    private = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key(), as_dict = True)
    jwk["kid"] = "test-key"
    return private, {"keys": [jwk]}


def _token(private, **overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "iss": "https://auth.example.com/realms/company",
        "aud": "unsloth",
        "sub": "subject-1",
        "nonce": "expected-nonce",
        "iat": now,
        "exp": now + timedelta(minutes = 5),
    }
    claims.update(overrides)
    return jwt.encode(claims, private, algorithm = "RS256", headers = {"kid": "test-key"})


def test_valid_id_token_is_cryptographically_verified(signing_material):
    private, jwks = signing_material
    claims = validate_id_token(
        _token(private),
        expected_nonce = "expected-nonce",
        config = _config(),
        discovery = _Discovery(jwks),
    )
    assert claims["sub"] == "subject-1"


@pytest.mark.parametrize(
    "overrides",
    (
        {"iss": "https://evil.example"},
        {"aud": "another-client"},
        {"exp": datetime.now(timezone.utc) - timedelta(seconds = 1)},
    ),
)
def test_invalid_issuer_audience_or_expiry_is_rejected(signing_material, overrides):
    private, jwks = signing_material
    with pytest.raises(OIDCAuthenticationError):
        validate_id_token(
            _token(private, **overrides),
            expected_nonce = "expected-nonce",
            config = _config(),
            discovery = _Discovery(jwks),
        )


def test_invalid_nonce_is_rejected(signing_material):
    private, jwks = signing_material
    with pytest.raises(OIDCAuthenticationError, match = "nonce"):
        validate_id_token(
            _token(private, nonce = "wrong"),
            expected_nonce = "expected-nonce",
            config = _config(),
            discovery = _Discovery(jwks),
        )
