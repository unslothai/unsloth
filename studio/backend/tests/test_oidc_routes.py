# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from urllib.parse import parse_qs, urlsplit

from fastapi import Response
from auth.oidc_config import OIDCConfig
from auth.oidc_discovery import OIDCProviderMetadata
from auth.oidc_handoff import OIDCSessionHandoffManager
from auth.oidc_state import OIDCStateManager
from models.auth import OIDCHandoffRequest
from routes import oidc


def _config(enabled = True):
    return OIDCConfig(
        enabled = enabled,
        issuer = "https://auth.example/realms/company",
        client_id = "unsloth",
        client_secret = "secret",
        redirect_uri = "https://studio.example/api/auth/oidc/callback",
        display_name = "Company SSO",
    )


class _Discovery:
    def metadata(self):
        return OIDCProviderMetadata(
            issuer = "https://auth.example/realms/company",
            authorization_endpoint = "https://auth.example/authorize",
            token_endpoint = "https://auth.example/token",
            jwks_uri = "https://auth.example/jwks",
        )


def test_public_config_never_contains_backend_secrets(monkeypatch):
    monkeypatch.setattr(oidc, "oidc_config", lambda: _config())
    http_response = Response()
    response = oidc.public_oidc_config(http_response)

    assert response.model_dump() == {"enabled": True, "display_name": "Company SSO"}
    assert "secret" not in str(response.model_dump())
    assert http_response.headers["cache-control"] == "no-store"


def test_login_redirect_contains_state_nonce_and_pkce(monkeypatch):
    state_manager = OIDCStateManager()
    monkeypatch.setattr(oidc, "oidc_config", lambda: _config())
    monkeypatch.setattr(oidc, "oidc_state_manager", state_manager)
    monkeypatch.setattr(oidc, "_discovery", lambda _config: _Discovery())

    response = oidc.oidc_login()
    query = parse_qs(urlsplit(response.headers["location"]).query)

    assert response.status_code == 302
    assert query["response_type"] == ["code"]
    assert query["nonce"][0]
    assert query["state"][0]
    assert query["code_challenge_method"] == ["S256"]
    assert state_manager.consume(query["state"][0]) is not None


def test_provider_error_consumes_and_validates_state(monkeypatch):
    state_manager = OIDCStateManager()
    attempt = state_manager.create()
    monkeypatch.setattr(oidc, "oidc_config", lambda: _config())
    monkeypatch.setattr(oidc, "oidc_state_manager", state_manager)

    try:
        oidc.oidc_callback(state = attempt.state, error = "access_denied")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("provider error was accepted")

    assert state_manager.consume(attempt.state) is None

    try:
        oidc.oidc_callback(state = "unknown", error = "access_denied")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 400
    else:
        raise AssertionError("provider error with invalid state was accepted")


def test_session_handoff_is_single_use(monkeypatch):
    handoffs = OIDCSessionHandoffManager()
    monkeypatch.setattr(oidc, "oidc_config", lambda: _config())
    monkeypatch.setattr(oidc, "oidc_session_handoffs", handoffs)
    code = handoffs.create(access_token = "access", refresh_token = "refresh", account_id = "account")

    token = oidc.oidc_handoff(OIDCHandoffRequest(handoff = code), Response())

    assert token.access_token == "access"
    assert token.refresh_token == "refresh"
    assert token.account_id == "account"

    try:
        oidc.oidc_handoff(OIDCHandoffRequest(handoff = code), Response())
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("replayed handoff was accepted")
