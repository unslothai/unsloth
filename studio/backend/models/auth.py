# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Pydantic schemas for Authentication API
"""

from pydantic import BaseModel, Field


class AuthLoginRequest(BaseModel):
    """Login payload: username/password to obtain a JWT."""

    username: str = Field(..., description = "Username")
    password: str = Field(..., description = "Password")


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
