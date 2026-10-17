# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from auth.oidc_config import OIDCConfigurationError, load_oidc_config


def _enabled(**overrides: str) -> dict[str, str]:
    values = {
        "UNSLOTH_OIDC_ENABLED": "true",
        "UNSLOTH_OIDC_ISSUER": "https://auth.example.com/realms/company/",
        "UNSLOTH_OIDC_CLIENT_ID": "unsloth",
        "UNSLOTH_OIDC_CLIENT_SECRET": "test-only-secret",
        "UNSLOTH_OIDC_REDIRECT_URI": "https://studio.example.com/api/auth/oidc/callback",
    }
    values.update(overrides)
    return values


def test_oidc_is_disabled_by_default_and_ignores_partial_settings():
    config = load_oidc_config({"UNSLOTH_OIDC_ISSUER": "not a URL"})

    assert config.enabled is False
    assert config.client_secret == ""
    assert config.display_name == "SSO"


def test_enabled_oidc_configuration_uses_safe_defaults_and_normalizes_issuer():
    config = load_oidc_config(_enabled())

    assert config.enabled is True
    assert config.issuer == "https://auth.example.com/realms/company"
    assert config.client_id == "unsloth"
    assert config.client_secret == "test-only-secret"
    assert config.scope == "openid profile email"
    assert config.display_name == "SSO"
    assert config.auto_create_users is True
    assert config.username_claim == "preferred_username"
    assert config.email_claim == "email"


def test_optional_oidc_configuration_is_parsed_without_provider_coupling():
    config = load_oidc_config(
        _enabled(
            UNSLOTH_OIDC_SCOPES = "openid profile groups groups",
            UNSLOTH_OIDC_DISPLAY_NAME = "Company SSO",
            UNSLOTH_OIDC_AUTO_CREATE_USERS = "no",
            UNSLOTH_OIDC_USERNAME_CLAIM = "nickname",
            UNSLOTH_OIDC_EMAIL_CLAIM = "mail",
            UNSLOTH_OIDC_ALLOWED_GROUPS = "employees, contractors, employees",
        )
    )

    assert config.scopes == ("openid", "profile", "groups")
    assert config.display_name == "Company SSO"
    assert config.auto_create_users is False
    assert config.username_claim == "nickname"
    assert config.email_claim == "mail"
    assert config.allowed_groups == ("employees", "contractors")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("UNSLOTH_OIDC_ISSUER", "relative/issuer", r"absolute http\(s\) URL"),
        ("UNSLOTH_OIDC_ISSUER", "https://user:pass@auth.example.com", "user information"),
        ("UNSLOTH_OIDC_REDIRECT_URI", "https://studio.example.com/callback#token", "fragment"),
        ("UNSLOTH_OIDC_SCOPES", "profile email", "must include openid"),
        ("UNSLOTH_OIDC_AUTO_CREATE_USERS", "sometimes", "must be true or false"),
        ("UNSLOTH_OIDC_USERNAME_CLAIM", "preferred username", "claim name"),
    ),
)
def test_enabled_oidc_rejects_invalid_security_configuration(name, value, message):
    with pytest.raises(OIDCConfigurationError, match = message):
        load_oidc_config(_enabled(**{name: value}))


@pytest.mark.parametrize(
    "missing",
    (
        "UNSLOTH_OIDC_ISSUER",
        "UNSLOTH_OIDC_CLIENT_ID",
        "UNSLOTH_OIDC_CLIENT_SECRET",
        "UNSLOTH_OIDC_REDIRECT_URI",
    ),
)
def test_enabled_oidc_requires_backend_configuration(missing):
    environ = _enabled()
    del environ[missing]

    with pytest.raises(OIDCConfigurationError, match = missing):
        load_oidc_config(environ)
