# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OIDC authorization-code exchange and cryptographic ID-token validation."""

from __future__ import annotations

import secrets
from typing import Optional

import httpx
import jwt

from .oidc_config import OIDCConfig
from .oidc_discovery import OIDCDiscoveryClient, OIDCProviderUnavailable


_ALLOWED_ID_TOKEN_ALGORITHMS = frozenset(
    ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512")
)
_MAX_TOKEN_RESPONSE_BYTES = 1024 * 1024


class OIDCAuthenticationError(RuntimeError):
    """The provider response cannot establish an authenticated identity."""


def _matching_signing_key(id_token: str, jwks: dict):
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.InvalidTokenError as exc:
        raise OIDCAuthenticationError("The identity token is invalid") from exc
    algorithm = header.get("alg")
    key_id = header.get("kid")
    if algorithm not in _ALLOWED_ID_TOKEN_ALGORITHMS or not isinstance(key_id, str):
        raise OIDCAuthenticationError("The identity token uses an unsupported signing key")
    matches = [key for key in jwks.get("keys", ()) if key.get("kid") == key_id]
    if len(matches) != 1:
        raise OIDCAuthenticationError("The identity token signing key is unavailable")
    try:
        signing_key = jwt.PyJWK.from_dict(matches[0], algorithm = algorithm)
    except (jwt.PyJWTError, ValueError) as exc:
        raise OIDCAuthenticationError("The identity token signing key is invalid") from exc
    return algorithm, signing_key.key


def validate_id_token(
    id_token: str, *, expected_nonce: str, config: OIDCConfig, discovery: OIDCDiscoveryClient
) -> dict:
    """Verify signature and all identity-bearing protocol claims before returning claims."""

    claims = None
    last_error: Exception | None = None
    for force_refresh in (False, True):
        try:
            algorithm, signing_key = _matching_signing_key(
                id_token, discovery.jwks(force_refresh = force_refresh)
            )
            claims = jwt.decode(
                id_token,
                signing_key,
                algorithms = [algorithm],
                audience = config.client_id,
                issuer = config.issuer,
                options = {"require": ["exp", "iat", "iss", "aud", "sub", "nonce"]},
            )
            break
        except OIDCProviderUnavailable:
            raise
        except (jwt.InvalidTokenError, OIDCAuthenticationError) as exc:
            last_error = exc
    if claims is None:
        raise OIDCAuthenticationError("The identity token could not be validated") from last_error

    nonce = claims.get("nonce")
    if not isinstance(nonce, str) or not secrets.compare_digest(nonce, expected_nonce):
        raise OIDCAuthenticationError("The identity token nonce is invalid")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise OIDCAuthenticationError("The identity token subject is invalid")
    audience = claims.get("aud")
    if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != config.client_id:
        raise OIDCAuthenticationError("The identity token authorized party is invalid")
    return claims


def exchange_authorization_code(
    code: str,
    *,
    code_verifier: str,
    config: OIDCConfig,
    discovery: OIDCDiscoveryClient,
    transport: Optional[httpx.BaseTransport] = None,
) -> dict:
    """Exchange one authorization code using confidential-client auth and PKCE."""

    try:
        with httpx.Client(timeout = 10.0, follow_redirects = False, transport = transport) as client:
            response = client.post(
                discovery.metadata().token_endpoint,
                auth = httpx.BasicAuth(config.client_id, config.client_secret),
                data = {
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": config.redirect_uri,
                    "code_verifier": code_verifier,
                },
                headers = {"Accept": "application/json"},
            )
            response.raise_for_status()
            if len(response.content) > _MAX_TOKEN_RESPONSE_BYTES:
                raise OIDCAuthenticationError("The provider token response is too large")
            result = response.json()
    except OIDCAuthenticationError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise OIDCProviderUnavailable("OIDC provider token endpoint is unavailable") from exc
    if not isinstance(result, dict) or not isinstance(result.get("id_token"), str):
        raise OIDCAuthenticationError("The provider did not return an identity token")
    return result
