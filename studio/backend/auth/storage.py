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


def _load_bootstrap_password() -> Optional[str]:
    """Load an existing bootstrap password without creating one."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        bootstrap_password = _BOOTSTRAP_PW_PATH.read_text().strip()
        if bootstrap_password:
            _bootstrap_password = bootstrap_password
    return _bootstrap_password


def clear_bootstrap_password() -> None:
    """Delete the persisted bootstrap password file (called after password change)."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper used for refresh token storage.

    Plain SHA-256 is intentional here: refresh tokens are high-entropy
    random strings from ``secrets.token_urlsafe(48)`` (384 bits of
    entropy), so a slow KDF (Argon2 / bcrypt / PBKDF2) provides zero
    additional security — no attacker can brute-force 2^384 regardless
    of hash speed — while adding tens of ms of CPU to every refresh.
    See the OWASP Password Storage Cheat Sheet on fast-vs-slow hashing
    of high-entropy inputs.

    API keys use the separate ``_pbkdf2_api_key`` helper below, which
    runs PBKDF2-HMAC-SHA256 with a persistent server-side salt — not
    for cryptographic reasons (128-bit random tokens don't need slow
    hashing), but because CodeQL's ``py/weak-sensitive-data-hashing``
    query mislabels API keys as passwords and demands a KDF.
    """
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
            expires_at TEXT NOT NULL,
            is_desktop INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            key_prefix TEXT NOT NULL,
            key_hash   TEXT NOT NULL UNIQUE,
            name       TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            expires_at TEXT,
            is_active  INTEGER NOT NULL DEFAULT 1,
            is_internal INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    api_key_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(api_keys)")
    }
    if "is_internal" not in api_key_columns:
        conn.execute(
            "ALTER TABLE api_keys ADD COLUMN is_internal INTEGER NOT NULL DEFAULT 0"
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_secrets (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    refresh_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")
    }
    if "is_desktop" not in refresh_columns:
        conn.execute(
            "ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()
    return conn


# ── API-key PBKDF2 salt ────────────────────────────────────────────────
#
# Module-level cache for the persistent API-key PBKDF2 salt. Populated
# lazily on first use via ``_get_or_create_api_key_pbkdf2_salt``. Not
# protected by a lock because (a) the ``INSERT OR IGNORE`` provides
# atomicity at the SQLite layer and (b) concurrent populations converge
# on the same value, so the worst case is a harmless duplicate read on
# startup.
_api_key_pbkdf2_salt_cache: Optional[bytes] = None


def _get_or_create_api_key_pbkdf2_salt() -> bytes:
    """Return the persistent API-key PBKDF2 salt, generating it once if missing.

    Stored as a hex-encoded 32-byte random value in the ``app_secrets``
    table under key ``"api_key_pbkdf2_salt"``. Regenerated only if the row
    is missing (i.e. fresh install, or operator manually deleted the row
    and accepts invalidating existing API keys).
    """
    global _api_key_pbkdf2_salt_cache
    if _api_key_pbkdf2_salt_cache is not None:
        return _api_key_pbkdf2_salt_cache

    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            ("api_key_pbkdf2_salt",),
        )
        row = cur.fetchone()
        if row is None:
            new_value = secrets.token_hex(32)  # 32 bytes -> 64 hex chars
            conn.execute(
                "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
                ("api_key_pbkdf2_salt", new_value),
            )
            conn.commit()
            cur = conn.execute(
                "SELECT value FROM app_secrets WHERE key = ?",
                ("api_key_pbkdf2_salt",),
            )
            row = cur.fetchone()
        salt = bytes.fromhex(row["value"])
    finally:
        conn.close()

    _api_key_pbkdf2_salt_cache = salt
    return salt


_API_KEY_PBKDF2_ITERATIONS = 100_000
DESKTOP_SECRET_PREFIX = "desktop-"
_DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
_DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"


def _pbkdf2_api_key(raw_key: str) -> str:
    """PBKDF2-HMAC-SHA256 an API key with a persistent server-side salt.

    Used for API-key storage ONLY, not refresh tokens. Matches the
    PBKDF2 algorithm + iteration count used by the password hasher in
    ``auth/hashing.py`` so the codebase is consistent on which KDF it
    uses for credential storage.

    Notes on why a slow KDF here is *only* a CodeQL appeasement and
    *not* a cryptographic requirement: API keys are cryptographically
    random 128-bit tokens (via ``secrets.token_hex``), so brute force
    against 2^128 is infeasible regardless of hash speed. CodeQL's
    ``py/weak-sensitive-data-hashing`` query mislabels these tokens as
    "password" sensitive data and then demands a KDF from its
    allowlist (Argon2 / scrypt / bcrypt / PBKDF2). Per the query's
    own recommendation page we use PBKDF2. The persistent salt is
    still loaded from ``app_secrets`` so an attacker dumping the
    ``api_keys`` table alone cannot derive hashes for candidate
    tokens without also obtaining the salt row.
    """
    salt = _get_or_create_api_key_pbkdf2_salt()
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode("utf-8"),
        salt,
        _API_KEY_PBKDF2_ITERATIONS,
    )
    return dk.hex()


def _pbkdf2_desktop_secret(raw_secret: str) -> str:
    return _pbkdf2_api_key(raw_secret)


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
    if get_user_and_secret(DEFAULT_ADMIN_USERNAME) is not None:
        _load_bootstrap_password()
        return False

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
            clear_desktop_secret()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_refresh_token(
    token: str,
    username: str,
    expires_at: str,
    *,
    is_desktop: bool = False,
) -> None:
    """
    Store a hashed refresh token with its associated username and expiry.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO refresh_tokens (token_hash, username, expires_at, is_desktop)
            VALUES (?, ?, ?, ?)
            """,
            (token_hash, username, expires_at, int(is_desktop)),
        )
        conn.commit()
    finally:
        conn.close()


def consume_refresh_token(token: str) -> Optional[Tuple[str, bool]]:
    """Atomically validate-and-delete a refresh token for single-use rotation.

    DELETE RETURNING fuses validate and delete into one statement so two
    concurrent refresh requests cannot both consume the same token.
    """
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (now,),
        )
        cur = conn.execute(
            """
            DELETE FROM refresh_tokens
            WHERE token_hash = ? AND expires_at >= ?
            RETURNING username, is_desktop
            """,
            (token_hash, now),
        )
        row = cur.fetchone()
        conn.commit()
        if row is None:
            return None
        return row["username"], bool(row["is_desktop"])
    finally:
        conn.close()


def verify_refresh_token(token: str) -> Optional[Tuple[str, bool]]:
    """
    Verify a refresh token and return the username plus desktop marker.

    Returns the username and desktop marker if valid and not expired, None otherwise.
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
            SELECT id, username, expires_at, is_desktop FROM refresh_tokens
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

        return row["username"], bool(row["is_desktop"])
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


def create_desktop_secret() -> str:
    """Create/rotate the local desktop credential and return it once."""
    ensure_default_admin()
    raw_secret = DESKTOP_SECRET_PREFIX + secrets.token_urlsafe(48)
    secret_hash = _pbkdf2_desktop_secret(raw_secret)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (_DESKTOP_SECRET_HASH_KEY, secret_hash),
        )
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (_DESKTOP_SECRET_CREATED_AT_KEY, now),
        )
        conn.commit()
        return raw_secret
    finally:
        conn.close()


def validate_desktop_secret(raw_secret: str) -> Optional[str]:
    """Return the real admin username when the desktop secret matches."""
    if not raw_secret.startswith(DESKTOP_SECRET_PREFIX):
        return None
    if get_user_and_secret(DEFAULT_ADMIN_USERNAME) is None:
        return None

    secret_hash = _pbkdf2_desktop_secret(raw_secret)
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (_DESKTOP_SECRET_HASH_KEY,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        if not secrets.compare_digest(row["value"], secret_hash):
            return None
        return DEFAULT_ADMIN_USERNAME
    finally:
        conn.close()


def clear_desktop_secret() -> None:
    """Remove backend-side desktop auth state."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM app_secrets WHERE key IN (?, ?)",
            (_DESKTOP_SECRET_HASH_KEY, _DESKTOP_SECRET_CREATED_AT_KEY),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

API_KEY_PREFIX = "sk-unsloth-"


def create_api_key(
    username: str,
    name: str,
    expires_at: Optional[str] = None,
    internal: bool = False,
) -> Tuple[str, dict]:
    """Create a new API key for *username*.

    Returns ``(raw_key, row_dict)`` where *raw_key* is shown to the user
    exactly once.  The database only stores the PBKDF2 hash.

    Pass ``internal=True`` for keys minted by workflows (e.g. data-recipe
    runs) that should not appear in user-facing key listings.
    """
    raw_key = API_KEY_PREFIX + secrets.token_hex(16)
    key_hash = _pbkdf2_api_key(raw_key)
    key_prefix = raw_key[len(API_KEY_PREFIX) : len(API_KEY_PREFIX) + 8]
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at, expires_at, is_internal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                key_prefix,
                key_hash,
                name,
                now,
                expires_at,
                1 if internal else 0,
            ),
        )
        conn.commit()
        cur = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,))
        row = cur.fetchone()
        return raw_key, dict(row)
    finally:
        conn.close()


def list_api_keys(username: str, include_internal: bool = False) -> list:
    """Return API keys for *username*. Internal workflow keys are hidden
    by default so they do not clutter user-facing UIs."""
    conn = get_connection()
    try:
        if include_internal:
            cur = conn.execute(
                """
                SELECT id, username, key_prefix, name, created_at, last_used_at,
                       expires_at, is_active, is_internal
                FROM api_keys
                WHERE username = ?
                ORDER BY created_at DESC
                """,
                (username,),
            )
        else:
            cur = conn.execute(
                """
                SELECT id, username, key_prefix, name, created_at, last_used_at,
                       expires_at, is_active, is_internal
                FROM api_keys
                WHERE username = ? AND is_internal = 0
                ORDER BY created_at DESC
                """,
                (username,),
            )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def revoke_api_key(username: str, key_id: int) -> bool:
    """Soft-delete an API key.  Returns True if a matching row was found."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND username = ?",
            (key_id, username),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def revoke_internal_api_key(key_id: int) -> bool:
    """Revoke an internal workflow-minted key without requiring a username.

    Used by the recipe runner to retire its sk-unsloth-* key once the job
    terminates, shrinking the window a leaked key could be abused.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND is_internal = 1",
            (key_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def validate_api_key(raw_key: str) -> Optional[str]:
    """Validate *raw_key* and return the owning username, or ``None``.

    Also updates ``last_used_at`` on success.
    """
    key_hash = _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, username, is_active, expires_at FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        if not row["is_active"]:
            return None
        if row["expires_at"] is not None:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires:
                return None
        conn.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), row["id"]),
        )
        conn.commit()
        return row["username"]
    finally:
        conn.close()
