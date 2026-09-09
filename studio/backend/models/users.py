# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic models for authentication tokens."""

from typing import Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    """Authentication response model for session credentials."""

    access_token: str = Field(
        ..., description = "Session access credential used for authenticated API requests"
    )
    refresh_token: str = Field(
        ...,
        description = "Session refresh credential used to renew an expired access credential",
    )
    token_type: str = Field(
        ..., description = "Credential type for the Authorization header, always 'bearer'"
    )
    must_change_password: bool = Field(
        ..., description = "True when the user must change the seeded default password"
    )
    account_id: Optional[str] = Field(
        None,
        description = (
            "Immutable id of the signed-in account, the same key storage is partitioned by. "
            "Usernames are reusable, so a client that keeps per-account state must compare this "
            "instead. Null only when an older server issued the credential."
        ),
    )
