# SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
# Copyright © 2025 Unsloth AI

"""
Pydantic schemas for Authentication API
"""

from pydantic import BaseModel, Field


class AuthSetupRequest(BaseModel):
    """First-time setup: create the initial admin user + password."""

    setup_token: str = Field(
        ..., description = "One-time setup token printed to the server console"
    )
    username: str = Field(..., description = "Admin username")
    password: str = Field(
        ..., min_length = 8, description = "Admin password (minimum 8 characters)"
    )


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
    """Indicate whether auth has been initialized."""

    initialized: bool = Field(..., description = "True if auth setup has been completed")
