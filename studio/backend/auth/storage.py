# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SQLite storage for authentication data (user credentials + JWT secret).
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple

from utils.paths import auth_db_path, ensure_dir

DB_PATH = auth_db_path()


def _hash_token(token: str) -> str:
    """SHA-256 hash a setup token for safe storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_connection() -> sqlite3.Connection:
    """Get a connection to the auth database, creating tables if needed."""
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            jwt_secret TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS setup_tokens (
            id INTEGER PRIMARY KEY,
            token_hash TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY,
            token_hash TEXT NOT NULL,
            username TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def is_initialized() -> bool:
    """Check if auth has been set up (user exists in DB)."""
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) AS c FROM auth_user")
    row = cur.fetchone()
    conn.close()
    return bool(row["c"])


def create_initial_user(username: str, password: str, jwt_secret: str) -> None:
    """
    Create the initial admin user in the database.

    Raises sqlite3.IntegrityError if username already exists.
    """
    from .hashing import hash_password

    salt, pwd_hash = hash_password(password)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret)
            VALUES (?, ?, ?, ?)
            """,
            (username, salt, pwd_hash, jwt_secret),
        )
        conn.commit()
    finally:
        conn.close()


def delete_user(username: str) -> None:
    """
    Delete a user from the database.

    Used for rollback when setup fails after user creation.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def get_user_and_secret(username: str) -> Optional[Tuple[str, str, str]]:
    """
    Get user's password salt, hash, and JWT secret.

    Returns (password_salt, password_hash, jwt_secret) or None if user not found.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT password_salt, password_hash, jwt_secret
            FROM auth_user
            WHERE username = ?
            """,
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return row["password_salt"], row["password_hash"], row["jwt_secret"]
    finally:
        conn.close()


def load_jwt_secret() -> str:
    """
    Load the JWT secret from the database.

    Raises RuntimeError if auth is not initialized.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT jwt_secret FROM auth_user LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise RuntimeError(
                "Auth is not initialized. Please set up a password first."
            )
        return row["jwt_secret"]
    finally:
        conn.close()


def save_setup_token(token: str) -> None:
    """
    Store a hashed setup token, replacing any existing one.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        conn.execute("DELETE FROM setup_tokens")
        conn.execute("INSERT INTO setup_tokens (token_hash) VALUES (?)", (token_hash,))
        conn.commit()
    finally:
        conn.close()


def consume_setup_token(token: str) -> bool:
    """
    Verify a setup token and delete it if valid.

    Returns True if the token was valid (and is now consumed), False otherwise.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id FROM setup_tokens WHERE token_hash = ?", (token_hash,)
        )
        row = cur.fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM setup_tokens WHERE id = ?", (row["id"],))
        conn.commit()
        return True
    finally:
        conn.close()


def has_pending_setup_token() -> bool:
    """Check if a setup token is waiting to be consumed."""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT COUNT(*) AS c FROM setup_tokens")
        row = cur.fetchone()
        return bool(row["c"])
    finally:
        conn.close()


def save_refresh_token(token: str, username: str, expires_at: str) -> None:
    """
    Store a hashed refresh token with its associated username and expiry.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO refresh_tokens (token_hash, username, expires_at)
            VALUES (?, ?, ?)
            """,
            (token_hash, username, expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def verify_refresh_token(token: str) -> Optional[str]:
    """
    Verify a refresh token and return the username.

    Returns the username if valid and not expired, None otherwise.
    The token is NOT consumed — it stays valid until it expires.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        # Clean up any expired tokens while we're here
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()

        cur = conn.execute(
            """
            SELECT id, username, expires_at FROM refresh_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        # Check expiry
        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            conn.execute("DELETE FROM refresh_tokens WHERE id = ?", (row["id"],))
            conn.commit()
            return None

        return row["username"]
    finally:
        conn.close()


def revoke_user_refresh_tokens(username: str) -> None:
    """Revoke all refresh tokens for a user (e.g. on logout)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()
