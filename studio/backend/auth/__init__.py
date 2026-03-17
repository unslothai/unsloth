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
    get_current_subject_allow_password_change,
    reload_secret,
)
from .storage import (
    DEFAULT_ADMIN_USERNAME,
    clear_bootstrap_password,
    generate_bootstrap_password,
    get_bootstrap_password,
    is_initialized,
    create_initial_user,
    ensure_default_admin,
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    requires_password_change,
    save_refresh_token,
    update_password,
    verify_refresh_token,
    revoke_user_refresh_tokens,
)
from .hashing import hash_password, verify_password

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "refresh_access_token",
    "get_current_subject",
    "get_current_subject_allow_password_change",
    "reload_secret",
    "DEFAULT_ADMIN_USERNAME",
    "clear_bootstrap_password",
    "generate_bootstrap_password",
    "get_bootstrap_password",
    "is_initialized",
    "create_initial_user",
    "ensure_default_admin",
    "get_jwt_secret",
    "get_user_and_secret",
    "load_jwt_secret",
    "requires_password_change",
    "save_refresh_token",
    "update_password",
    "verify_refresh_token",
    "revoke_user_refresh_tokens",
    "hash_password",
    "verify_password",
]
