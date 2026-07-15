# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for auth data (user credentials + JWT secret)."""

import hashlib
import hmac
import ipaddress
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple

from utils.paths import auth_db_path, ensure_dir

DB_PATH = auth_db_path()
DEFAULT_ADMIN_USERNAME = "unsloth"

# Single source for the password policy; models/auth.py ChangePasswordRequest
# and the terminal prompt both enforce it. Keep the unsloth_cli mirror in sync.
MIN_PASSWORD_LENGTH = 8

# Plaintext bootstrap password file beside auth.db, deleted on first password
# change so the credential never lingers on disk.
_BOOTSTRAP_PW_PATH = DB_PATH.parent / ".bootstrap_password"

# In-process cache to avoid re-reading the file on every HTML serve.
_bootstrap_password: Optional[str] = None


def generate_bootstrap_password() -> str:
    """Generate a 4-word diceware passphrase and persist it to disk.

    Persisted (the DB stores only the hash) so it survives restarts; later
    calls return the persisted value.
    """
    global _bootstrap_password

    # Cached in this process?
    if _bootstrap_password is not None:
        return _bootstrap_password

    # Persisted from a previous run?
    if _BOOTSTRAP_PW_PATH.is_file():
        _bootstrap_password = _BOOTSTRAP_PW_PATH.read_text().strip()
        if _bootstrap_password:
            return _bootstrap_password

    # First startup: generate a fresh passphrase.
    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the same passphrase survives restarts until password change.
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
    """Delete the persisted bootstrap password file (after a password change).

    Best-effort: the new hash is already committed, so a locked/undeletable file
    (Windows AV, read-only auth dir) must not fail the change.
    """
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        try:
            _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)
        except OSError as e:
            # Removal failed (Windows AV, read-only auth dir). The hash is already
            # committed, so don't fail the change -- but truncate the file so its
            # stale plaintext can't be re-seeded by generate_bootstrap_password()
            # if a later reset-password deletes auth.db and re-validates it.
            try:
                _BOOTSTRAP_PW_PATH.write_text("")
                cleared = True
            except OSError:
                cleared = False
            import sys

            if cleared:
                message = (
                    f"Warning: could not delete {_BOOTSTRAP_PW_PATH.name} ({e}); "
                    "cleared its contents so the old bootstrap password cannot be reused."
                )
            else:
                # Neither removed nor truncated: stale plaintext is still on disk
                # and would be reused if auth.db is reset. Don't claim otherwise.
                message = (
                    f"Warning: could not delete or clear {_BOOTSTRAP_PW_PATH.name} ({e}); "
                    "its old bootstrap password is still on disk. Remove it manually to "
                    "prevent reuse after a reset."
                )
            print(message, file = sys.stderr, flush = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper for refresh token storage.

    Plain SHA-256 is intentional: refresh tokens are 384-bit random strings, so
    a slow KDF adds no security while costing per-refresh latency. API keys use
    the separate ``_pbkdf2_api_key`` helper, only to satisfy CodeQL's
    ``py/weak-sensitive-data-hashing`` query, not for crypto reasons.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_connection() -> sqlite3.Connection:
    """Get a connection to the auth database, creating tables if needed."""
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    # Keep the auth dir + DB private (they hold the JWT/identity secrets and
    # password hashes); sqlite3.connect would otherwise create the DB 0644 under
    # a 022 umask, letting another OS user read the identity secret and forge proofs.
    for _path, _mode in ((DB_PATH.parent, 0o700), (DB_PATH, 0o600)):
        try:
            os.chmod(_path, _mode)
        except OSError:
            pass
    conn.row_factory = sqlite3.Row
    # WAL lets token reads run concurrently with refresh-token writes;
    # busy_timeout bounds lock waits. Matches the other Studio SQLite stores.
    # Set busy_timeout first: switching journal_mode needs a lock, so if a
    # refresh-token write already holds one, journal_mode=WAL raises SQLITE_BUSY;
    # with busy_timeout already in effect it waits instead of failing and leaving
    # this connection on SQLite's default zero lock wait.
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
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
    api_key_columns = {row["name"] for row in conn.execute("PRAGMA table_info(api_keys)")}
    if "is_internal" not in api_key_columns:
        conn.execute("ALTER TABLE api_keys ADD COLUMN is_internal INTEGER NOT NULL DEFAULT 0")
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
    refresh_columns = {row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
    if "is_desktop" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    return conn


# ── API-key PBKDF2 salt ────────────────────────────────────────────────
#
# Module-level cache for the persistent API-key PBKDF2 salt, populated lazily
# via ``_get_or_create_api_key_pbkdf2_salt``. No lock needed: (a) ``INSERT OR
# IGNORE`` is atomic at the SQLite layer and (b) concurrent populations
# converge on the same value, so the worst case is a harmless duplicate read
# on startup.
_api_key_pbkdf2_salt_cache: Optional[bytes] = None


def _get_or_create_api_key_pbkdf2_salt() -> bytes:
    """Return the persistent API-key PBKDF2 salt, generating it once if missing.

    Hex-encoded 32-byte random value in ``app_secrets``. Regenerated only when
    the row is missing (fresh install, or operator deleted it).
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


# Secret answering the /api/auth/identity challenge (HMAC(secret, nonce)). Lives
# in this same-user DB so a port squatter or remote/fake server can't forge a
# proof. Separate from the per-user JWT secret.
_IDENTITY_SECRET_DB_KEY = "studio_identity_secret"
_identity_secret_cache: Optional[bytes] = None


def get_or_create_identity_secret() -> bytes:
    """Return the identity secret (hex 32-byte row in app_secrets), creating it once."""
    global _identity_secret_cache
    if _identity_secret_cache is not None:
        return _identity_secret_cache

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (_IDENTITY_SECRET_DB_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
                (_IDENTITY_SECRET_DB_KEY, secrets.token_hex(32)),
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM app_secrets WHERE key = ?",
                (_IDENTITY_SECRET_DB_KEY,),
            ).fetchone()
        secret = bytes.fromhex(row["value"])
    finally:
        conn.close()

    _identity_secret_cache = secret
    return secret


def compute_identity_proof(nonce: bytes, host: str, port: int) -> str:
    """HMAC-SHA256 proof that the caller holds this install's identity secret,
    bound to the loopback address and port the connection landed on. A proof
    relayed from a Studio on a different address/port (a squatter proxying to the
    real one, e.g. localhost resolving to ::1 while Studio is on 127.0.0.1) was
    computed for that other endpoint and won't match the one the client dialed."""
    try:
        host = ipaddress.ip_address(host).compressed  # normalise 127.0.0.1 / ::1 forms
    except ValueError:
        host = (host or "").lower()
    msg = b"|".join([nonce, host.encode(), str(int(port)).encode()])
    return hmac.new(get_or_create_identity_secret(), msg, hashlib.sha256).hexdigest()


# Capability secret for public ``/p`` preview share links. HMAC(secret, ref)
# turns the deterministic preview ref into an unguessable bearer capability, so a
# guessed run/checkpoint name can't reach inference. Dedicated (not the per-user
# JWT secret) so rotating it revokes every shared link without touching logins.
_PREVIEW_LINK_SECRET_DB_KEY = "preview_link_secret"
_preview_link_secret_cache: Optional[bytes] = None


def get_or_create_preview_link_secret() -> bytes:
    """Return the preview-link signing secret (hex 32-byte row in app_secrets), creating it once."""
    global _preview_link_secret_cache
    if _preview_link_secret_cache is not None:
        return _preview_link_secret_cache

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (_PREVIEW_LINK_SECRET_DB_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
                (_PREVIEW_LINK_SECRET_DB_KEY, secrets.token_hex(32)),
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM app_secrets WHERE key = ?",
                (_PREVIEW_LINK_SECRET_DB_KEY,),
            ).fetchone()
        secret = bytes.fromhex(row["value"])
    finally:
        conn.close()

    _preview_link_secret_cache = secret
    return secret


def rotate_preview_link_secret() -> bytes:
    """Rotate the preview-link secret, immediately revoking every outstanding ``/p`` share link."""
    global _preview_link_secret_cache
    new_secret_hex = secrets.token_hex(32)
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO app_secrets (key, value) VALUES (?, ?)",
            (_PREVIEW_LINK_SECRET_DB_KEY, new_secret_hex),
        )
        conn.commit()
    finally:
        conn.close()

    secret = bytes.fromhex(new_secret_hex)
    _preview_link_secret_cache = secret
    return secret


_API_KEY_PBKDF2_ITERATIONS = 100_000
DESKTOP_SECRET_PREFIX = "desktop-"
_DESKTOP_SECRET_HASH_KEY = "desktop_secret_hash"
_DESKTOP_SECRET_CREATED_AT_KEY = "desktop_secret_created_at"


def _pbkdf2_api_key(raw_key: str) -> str:
    """PBKDF2-HMAC-SHA256 an API key with a persistent server-side salt.

    For API-key storage ONLY, not refresh tokens. The slow KDF is only to
    appease CodeQL's ``py/weak-sensitive-data-hashing`` query, not a crypto
    requirement (API keys are random 128-bit tokens). The salt lives in
    ``app_secrets`` so dumping ``api_keys`` alone can't derive hashes.
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


# Memoize the deterministic raw-key -> PBKDF2-hash derivation so the 100k-round
# KDF runs once per key instead of on every authenticated request. Keyed by a
# salted HMAC of the key (not the key itself); revocation/expiry are still
# enforced by the SQLite read on every call, so a cache hit only skips the KDF.
# Only keys present in the DB are cached, so unknown-key spam can't grow it.
_api_key_hash_cache: dict[str, str] = {}
_API_KEY_HASH_CACHE_MAX = 4096
_api_key_hash_cache_lock = threading.Lock()


def _api_key_cache_id(raw_key: str) -> str:
    """Cache id for a raw key: salted HMAC-SHA256 (not the key itself)."""
    return hmac.new(
        _get_or_create_api_key_pbkdf2_salt(), raw_key.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _reset_api_key_hash_cache() -> None:
    """Drop memoized derivations (tests / salt change)."""
    with _api_key_hash_cache_lock:
        _api_key_hash_cache.clear()


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


def update_password(
    username: str,
    new_password: str,
    *,
    revoke_refresh_tokens: bool = False,
) -> bool:
    """Update password, clear first-login requirement, rotate JWT secret.

    ``revoke_refresh_tokens`` deletes the user's refresh tokens in the SAME
    transaction: a separate delete could fail after the password commit and
    leave a pre-change token still able to mint access tokens.
    """
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
        if revoke_refresh_tokens and cursor.rowcount > 0:
            conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
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
        # Opportunistically clean up expired tokens
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
    cache_id = _api_key_cache_id(raw_key)
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT id, username, is_active, expires_at FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        # Real key: memoize so later requests skip the KDF. Bounded; clear on overflow.
        if cached_hash is None:
            with _api_key_hash_cache_lock:
                if len(_api_key_hash_cache) >= _API_KEY_HASH_CACHE_MAX:
                    _api_key_hash_cache.clear()
                _api_key_hash_cache[cache_id] = key_hash
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
