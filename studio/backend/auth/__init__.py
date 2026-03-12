# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Authentication module for JWT-based auth with SQLite storage.
"""

from .authentication import (
    create_access_token,
    create_refresh_token,
    refresh_access_token,
    get_current_subject,
    reload_secret,
)
from .storage import (
    is_initialized,
    create_initial_user,
    get_user_and_secret,
    load_jwt_secret,
    save_setup_token,
    consume_setup_token,
    has_pending_setup_token,
    save_refresh_token,
    verify_refresh_token,
    revoke_user_refresh_tokens,
)
from .hashing import hash_password, verify_password

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "refresh_access_token",
    "get_current_subject",
    "reload_secret",
    "is_initialized",
    "create_initial_user",
    "get_user_and_secret",
    "load_jwt_secret",
    "save_setup_token",
    "consume_setup_token",
    "has_pending_setup_token",
    "save_refresh_token",
    "verify_refresh_token",
    "revoke_user_refresh_tokens",
    "hash_password",
    "verify_password",
]
