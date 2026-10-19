# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Environment-backed configuration for external OpenID Connect authentication."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional
from urllib.parse import urlsplit, urlunsplit


_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("", "0", "false", "no", "off"))


class OIDCConfigurationError(ValueError):
    """An enabled OIDC configuration is incomplete or unsafe."""


def _boolean(environ: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise OIDCConfigurationError(f"{name} must be true or false")


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise OIDCConfigurationError(f"{name} is required when OIDC is enabled")
    return value


def _absolute_url(
    value: str,
    name: str,
    *,
    strip_trailing_slash: bool = False,
) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OIDCConfigurationError(f"{name} must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise OIDCConfigurationError(f"{name} must not contain user information")
    if parsed.query or parsed.fragment:
        raise OIDCConfigurationError(f"{name} must not contain a query string or fragment")
    path = parsed.path.rstrip("/") if strip_trailing_slash else parsed.path
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _claim_name(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name, default).strip()
    if not value or any(character.isspace() for character in value):
        raise OIDCConfigurationError(f"{name} must be a non-empty claim name without whitespace")
    return value


def _csv(environ: Mapping[str, str], name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in environ.get(name, "").split(",") if item.strip())
    )


@dataclass(frozen = True, slots = True)
class OIDCConfig:
    """Validated settings for one standards-compatible OIDC provider."""

    enabled: bool = False
    issuer: str = ""
    client_id: str = ""
    client_secret: str = ""
    scopes: tuple[str, ...] = ("openid", "profile", "email")
    redirect_uri: str = ""
    display_name: str = "SSO"
    auto_create_users: bool = True
    username_claim: str = "preferred_username"
    email_claim: str = "email"
    allowed_groups: tuple[str, ...] = ()

    @property
    def scope(self) -> str:
        return " ".join(self.scopes)


def load_oidc_config(environ: Optional[Mapping[str, str]] = None) -> OIDCConfig:
    """Load OIDC settings without retaining or exposing the client secret elsewhere.

    Disabled mode deliberately ignores every other OIDC variable. This keeps existing installs
    byte-for-byte equivalent in behavior and lets an operator disable a partially edited setup.
    """

    source = os.environ if environ is None else environ
    enabled = _boolean(source, "UNSLOTH_OIDC_ENABLED", default = False)
    if not enabled:
        return OIDCConfig()

    scopes = tuple(dict.fromkeys(source.get("UNSLOTH_OIDC_SCOPES", "openid profile email").split()))
    if "openid" not in scopes:
        raise OIDCConfigurationError("UNSLOTH_OIDC_SCOPES must include openid")
    display_name = source.get("UNSLOTH_OIDC_DISPLAY_NAME", "SSO").strip()
    if not display_name:
        raise OIDCConfigurationError("UNSLOTH_OIDC_DISPLAY_NAME must not be empty")

    return OIDCConfig(
        enabled = True,
        issuer = _absolute_url(
            _required(source, "UNSLOTH_OIDC_ISSUER"),
            "UNSLOTH_OIDC_ISSUER",
            strip_trailing_slash = True,
        ),
        client_id = _required(source, "UNSLOTH_OIDC_CLIENT_ID"),
        client_secret = _required(source, "UNSLOTH_OIDC_CLIENT_SECRET"),
        scopes = scopes,
        redirect_uri = _absolute_url(
            _required(source, "UNSLOTH_OIDC_REDIRECT_URI"),
            "UNSLOTH_OIDC_REDIRECT_URI",
        ),
        display_name = display_name,
        auto_create_users = _boolean(source, "UNSLOTH_OIDC_AUTO_CREATE_USERS", default = True),
        username_claim = _claim_name(source, "UNSLOTH_OIDC_USERNAME_CLAIM", "preferred_username"),
        email_claim = _claim_name(source, "UNSLOTH_OIDC_EMAIL_CLAIM", "email"),
        allowed_groups = _csv(source, "UNSLOTH_OIDC_ALLOWED_GROUPS"),
    )


_config: Optional[OIDCConfig] = None


def oidc_config() -> OIDCConfig:
    """Return the process configuration, loading and validating it once."""

    global _config
    if _config is None:
        _config = load_oidc_config()
    return _config


def reset_oidc_config_cache() -> None:
    """Forget the process cache. Intended for tests that replace the environment."""

    global _config
    _config = None
