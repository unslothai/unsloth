# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Password hashing utilities using PBKDF2.
"""

import hashlib
import hmac
import secrets
from typing import Tuple


def hash_password(password: str, salt: str | None = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2-HMAC-SHA256.

    Returns (salt, hex_hash) tuple.
    """
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,  # 100k iterations
    )
    return salt, dk.hex()


def verify_password(password: str, salt: str, hashed: str) -> bool:
    """
    Verify a password against a stored salt and hash.

    Uses constant-time comparison to prevent timing attacks.
    """
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    )
    return hmac.compare_digest(dk.hex(), hashed)
