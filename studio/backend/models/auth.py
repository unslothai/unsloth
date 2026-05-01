# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Authentication API
"""

from typing import Optional

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login payload: username/password to obtain a JWT."""

    username: str = Field(..., description = "Username")
    password: str = Field(..., description = "Password")


class DesktopLoginRequest(BaseModel):
    """Desktop-only local secret exchange payload."""

    secret: str = Field(..., description = "Desktop local auth secret")


class RefreshTokenRequest(BaseModel):
    """Refresh token payload to obtain new access + refresh tokens."""

    refresh_token: str = Field(
        ..., description = "Refresh token from a previous login or refresh"
    )


class AuthStatusResponse(BaseModel):
    """Indicate whether the seeded admin auth flow is ready."""

    initialized: bool = Field(
        ..., description = "True if the auth database contains a login user"
    )
    default_username: str = Field(..., description = "Default seeded admin username")
    requires_password_change: bool = Field(
        ...,
        description = "True if the seeded admin must still change the default password",
    )


class ChangePasswordRequest(BaseModel):
    """Change the current user's password, typically on first login."""

    current_password: str = Field(
        ..., min_length = 8, description = "Existing password for the authenticated user"
    )
    new_password: str = Field(
        ..., min_length = 8, description = "Replacement password (minimum 8 characters)"
    )


# ---------------------------------------------------------------------------
# API key schemas
# ---------------------------------------------------------------------------


class CreateApiKeyRequest(BaseModel):
    """Request body to create a new API key."""

    name: str = Field(..., description = "Human-readable label for this key")
    expires_in_days: Optional[int] = Field(
        None, description = "Number of days until the key expires (None = never)"
    )


class ApiKeyResponse(BaseModel):
    """Public representation of an API key (never contains the raw key)."""

    id: int
    name: str
    key_prefix: str = Field(
        ..., description = "First 8 characters after sk-unsloth- for display"
    )
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_active: bool


class CreateApiKeyResponse(BaseModel):
    """Returned once when a key is created -- ``key`` is never shown again."""

    key: str = Field(..., description = "Full API key (shown once)")
    api_key: ApiKeyResponse


class ApiKeyListResponse(BaseModel):
    """List of API keys for the authenticated user."""

    api_keys: list[ApiKeyResponse]
