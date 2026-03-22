# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SQLite storage for authentication data (user credentials + JWT secret).
"""

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple

from utils.paths import auth_db_path, ensure_dir

DB_PATH = auth_db_path()
DEFAULT_ADMIN_USERNAME = "unsloth"

# Plaintext bootstrap password file — lives beside auth.db, deleted on
# first password change so the credential never lingers on disk.
_BOOTSTRAP_PW_PATH = DB_PATH.parent / ".bootstrap_password"

# In-process cache so we don't re-read the file on every HTML serve.
_bootstrap_password: Optional[str] = None


def generate_bootstrap_password() -> str:
    """Generate a 4-word diceware passphrase and persist it to disk.

    The passphrase is written to ``_BOOTSTRAP_PW_PATH`` so that it
    survives server restarts (the DB only stores the *hash*).  On
    subsequent calls / restarts, the persisted value is returned.
    """
    global _bootstrap_password

    # 1. Already cached in this process?
    if _bootstrap_password is not None:
        return _bootstrap_password

    # 2. Already persisted from a previous run?
    if _BOOTSTRAP_PW_PATH.is_file():
        _bootstrap_password = _BOOTSTRAP_PW_PATH.read_text().strip()
        if _bootstrap_password:
            return _bootstrap_password

    # 3. First-ever startup — generate a fresh passphrase.
    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the *same* passphrase is used if the server restarts
    # before the user changes the password.
    ensure_dir(_BOOTSTRAP_PW_PATH.parent)
    _BOOTSTRAP_PW_PATH.write_text(_bootstrap_password)
    try:
        os.chmod(_BOOTSTRAP_PW_PATH, 0o600)
    except OSError:
        pass

    return _bootstrap_password


def get_bootstrap_password() -> Optional[str]:
    """Return the cached bootstrap password, or None if not yet generated."""
    return _bootstrap_password


def clear_bootstrap_password() -> None:
    """Delete the persisted bootstrap password file (called after password change)."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper used for refresh token storage."""
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
            jwt_secret TEXT NOT NULL,
            must_change_password INTEGER NOT NULL DEFAULT 0
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
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()
    return conn


def is_initialized() -> bool:
    """Check if auth is ready for login (at least one user exists in DB)."""
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) AS c FROM auth_user")
    row = cur.fetchone()
    conn.close()
    return bool(row["c"])


def create_initial_user(
    username: str,
    password: str,
    jwt_secret: str,
    *,
    must_change_password: bool = False,
) -> None:
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
            INSERT INTO auth_user (
                username,
                password_salt,
                password_hash,
                jwt_secret,
                must_change_password
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, salt, pwd_hash, jwt_secret, int(must_change_password)),
        )
        conn.commit()
    finally:
        conn.close()


def delete_user(username: str) -> None:
    """
    Delete a user from the database.

    Used for rollback when user creation fails partway through bootstrap.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def get_user_and_secret(username: str) -> Optional[Tuple[str, str, str, bool]]:
    """
    Get user's password salt, hash, and JWT secret.

    Returns (password_salt, password_hash, jwt_secret, must_change_password)
    or None if user not found.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT password_salt, password_hash, jwt_secret, must_change_password
            FROM auth_user
            WHERE username = ?
            """,
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return (
            row["password_salt"],
            row["password_hash"],
            row["jwt_secret"],
            bool(row["must_change_password"]),
        )
    finally:
        conn.close()


def get_jwt_secret(username: str) -> Optional[str]:
    """Return the current JWT signing secret for a user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT jwt_secret FROM auth_user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return row["jwt_secret"] if row else None
    finally:
        conn.close()


def requires_password_change(username: str) -> bool:
    """Return whether the user must change the seeded default password."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT must_change_password FROM auth_user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return bool(row and row["must_change_password"])
    finally:
        conn.close()


def load_jwt_secret() -> str:
    """
    Load the JWT secret from the database.

    Raises RuntimeError if no auth user has been created yet.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT jwt_secret FROM auth_user LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise RuntimeError(
                "Auth is not initialized. Wait for the seeded admin bootstrap to complete."
            )
        return row["jwt_secret"]
    finally:
        conn.close()


def ensure_default_admin() -> bool:
    """Seed the default admin account on first startup.

    Uses a randomly generated diceware passphrase as the bootstrap password.
    Returns True when the default admin was created in this call.
    """
    bootstrap_pw = generate_bootstrap_password()
    try:
        create_initial_user(
            username = DEFAULT_ADMIN_USERNAME,
            password = bootstrap_pw,
            jwt_secret = secrets.token_urlsafe(64),
            must_change_password = True,
        )
        return True
    except sqlite3.IntegrityError:
        return False


def update_password(username: str, new_password: str) -> bool:
    """Update password, clear first-login requirement, rotate JWT secret."""
    from .hashing import hash_password

    salt, pwd_hash = hash_password(new_password)
    jwt_secret = secrets.token_urlsafe(64)
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE auth_user
            SET password_salt = ?, password_hash = ?, jwt_secret = ?, must_change_password = 0
            WHERE username = ?
            """,
            (salt, pwd_hash, jwt_secret, username),
        )
        conn.commit()
        if cursor.rowcount > 0:
            clear_bootstrap_password()
        return cursor.rowcount > 0
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
