# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for auth data (user credentials + JWT secret)."""

from contextlib import contextmanager

import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional, Tuple

from utils.paths import auth_db_path, ensure_dir

logger = logging.getLogger(__name__)

DB_PATH = auth_db_path()
DEFAULT_ADMIN_USERNAME = "unsloth"

# Single source for the password policy; models/auth.py ChangePasswordRequest
# and the terminal prompt both enforce it. Keep the unsloth_cli mirror in sync.
MIN_PASSWORD_LENGTH = 8

# Managed-account setup codes are high-entropy initial passwords. They are
# intentionally short-lived because an owner may need to send one out of band.
SETUP_CODE_TTL_MINUTES = 60
_SETUP_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

# Plaintext bootstrap password file beside auth.db, deleted on first password
# change so the credential never lingers on disk.
_BOOTSTRAP_PW_PATH = DB_PATH.parent / ".bootstrap_password"

# In-process cache to avoid re-reading the file on every HTML serve.
_bootstrap_password: Optional[str] = None

# Shown when a deleted account's files could not be moved aside, so its name is
# still reserved. States the remedy, because a retry normally succeeds.
_RETIRED_USERNAME_MESSAGE = (
    "That username still has files from the deleted account that could not be "
    "released. Close anything using them and try again."
)


def _bootstrap_file_bytes(password: str) -> bytes:
    """Exact on-disk form: the secret plus one LF.

    Bytes, not text: text mode writes CRLF on Windows, and `$(cat ...)` strips
    the LF but leaves the CR attached to the credential.
    """
    return (password + "\n").encode("utf-8")


def _persist_bootstrap_password(password: str) -> None:
    """Atomically write the bootstrap password 0600, LF terminated on every OS.

    A partial write would destroy the only plaintext recovery credential.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix = f".{_BOOTSTRAP_PW_PATH.name}.", dir = _BOOTSTRAP_PW_PATH.parent
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_bootstrap_file_bytes(password))
        try:
            os.chmod(tmp_name, 0o600)
        except OSError:
            pass
        os.replace(tmp_name, _BOOTSTRAP_PW_PATH)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _normalise_bootstrap_file(raw: bytes, password: str) -> None:
    """Append the LF a pre-newline release left off.

    Append-only, and only when the file is exactly the credential:
    clear_bootstrap_password() may unlink or (when unlink fails, notably on
    Windows while this descriptor is open) truncate through another descriptor
    after we read, so a rewrite could restore revoked plaintext. An append
    cannot: worst case is a lone "\\n" over a cleared file, which strips back to
    no bootstrap password. Pre-newline releases wrote no terminator at all, so
    that is the only shape in the wild; anything else reads fine, since every
    reader strips, and is left alone.
    """
    if raw != password.encode("utf-8"):
        return

    # O_BINARY: without it Windows opens in text mode and turns the LF straight
    # back into CRLF, the bug being fixed.
    fd = os.open(
        _BOOTSTRAP_PW_PATH,
        os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0),
    )
    try:
        os.write(fd, b"\n")
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            # fchmod only reached Windows in 3.13.
            pass
    finally:
        os.close(fd)


def _read_persisted_bootstrap_password() -> Optional[str]:
    """Read the persisted password, normalising the file if it is malformed."""
    if not _BOOTSTRAP_PW_PATH.is_file():
        return None

    # No caller handles a raise, so an unreadable file has to mean "no bootstrap
    # password", not a dead backend. We write UTF-8, so undecodable bytes are
    # damage whose plaintext is worthless anyway.
    try:
        raw = _BOOTSTRAP_PW_PATH.read_bytes()
        password = raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not password:
        return None

    # Older releases wrote no terminator; best-effort, a read-only auth dir must
    # not fail startup.
    if raw != _bootstrap_file_bytes(password):
        try:
            _normalise_bootstrap_file(raw, password)
        except OSError:
            pass
    return password


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
    persisted = _read_persisted_bootstrap_password()
    if persisted:
        _bootstrap_password = persisted
        return _bootstrap_password

    # First startup: generate a fresh passphrase.
    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the same passphrase survives restarts until password change.
    ensure_dir(_BOOTSTRAP_PW_PATH.parent)
    _persist_bootstrap_password(_bootstrap_password)

    return _bootstrap_password


def get_bootstrap_password() -> Optional[str]:
    """Return the cached bootstrap password, or None if not yet generated."""
    return _bootstrap_password


def _load_bootstrap_password() -> Optional[str]:
    """Load an existing bootstrap password without creating one.

    Upgrades take this path, not generate_bootstrap_password()
    (ensure_default_admin short-circuits once the admin row exists), so it has
    to normalise too.
    """
    global _bootstrap_password
    _bootstrap_password = _read_persisted_bootstrap_password()
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
            # if auth.db is ever recreated.
            try:
                _BOOTSTRAP_PW_PATH.write_text("", encoding = "utf-8")
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


class CredentialRotated(Exception):
    """A password reset revoked the credential this request authenticated with."""


def credential_generation(jwt_secret: str) -> str:
    """Marker for the credential version a refresh token was issued under.

    Every password change rotates ``jwt_secret``, so a token stamped with the
    previous one is rejected even if it was inserted after the revoking DELETE.
    """
    return hashlib.sha256(jwt_secret.encode("utf-8")).hexdigest()


def _current_secret(conn: sqlite3.Connection, username: str) -> Optional[str]:
    row = conn.execute(
        "SELECT jwt_secret FROM auth_user WHERE username = ?", (username,)
    ).fetchone()
    return row["jwt_secret"] if row else None


def _current_generation(conn: sqlite3.Connection, username: str) -> Optional[str]:
    secret = _current_secret(conn, username)
    return credential_generation(secret) if secret is not None else None


@contextmanager
def credential_generation_guard(username: str, expect_gen: Optional[str]) -> Iterator[None]:
    """Hold the auth write lock while a credential-derived write commits elsewhere."""
    conn = get_connection()
    try:
        if expect_gen is not None:
            conn.execute("BEGIN IMMEDIATE")
            if _current_generation(conn, username) != expect_gen:
                raise CredentialRotated(
                    "The credential this request authenticated with was revoked."
                )
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
    # busy_timeout bounds lock waits. Matches the other Unsloth SQLite stores.
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
            must_change_password INTEGER NOT NULL DEFAULT 0,
            is_admin INTEGER NOT NULL DEFAULT 0,
            setup_code_expires_at TEXT
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
            is_desktop INTEGER NOT NULL DEFAULT 0,
            secret_gen TEXT
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
    # A username whose workspace files could not be renamed away on delete. It
    # stays unusable until they are, so a namesake cannot inherit them.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retired_usernames (
            username   TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        """
    )
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    if "is_admin" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
        # The seeded legacy account is the installation owner. Upgrade it in
        # the migration transaction; doing this on every read would turn all
        # auth lookups into SQLite writers.
        conn.execute(
            "UPDATE auth_user SET is_admin = 1 WHERE username = ?",
            (DEFAULT_ADMIN_USERNAME,),
        )
    if "setup_code_expires_at" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN setup_code_expires_at TEXT")
    refresh_columns = {row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
    if "is_desktop" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0")
    if "secret_gen" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN secret_gen TEXT")
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


# Dedicated AES-256 key used to encrypt Unsloth credentials in studio.db.
# It intentionally lives in auth.db so copying studio.db alone does not expose
# provider or Hugging Face tokens, and survives password changes/resets.
_CREDENTIAL_ENCRYPTION_KEY_DB_KEY = "credential_encryption_key_v1"
_credential_encryption_key_cache: Optional[bytes] = None


def get_or_create_credential_encryption_key() -> bytes:
    """Return the install-local credential encryption key, creating it once."""
    global _credential_encryption_key_cache
    if _credential_encryption_key_cache is not None:
        return _credential_encryption_key_cache

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (_CREDENTIAL_ENCRYPTION_KEY_DB_KEY,),
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
                (_CREDENTIAL_ENCRYPTION_KEY_DB_KEY, secrets.token_hex(32)),
            )
            conn.commit()
            row = conn.execute(
                "SELECT value FROM app_secrets WHERE key = ?",
                (_CREDENTIAL_ENCRYPTION_KEY_DB_KEY,),
            ).fetchone()
        secret = bytes.fromhex(row["value"])
        if len(secret) != 32:
            raise ValueError("Invalid credential encryption key")
    finally:
        conn.close()

    _credential_encryption_key_cache = secret
    return secret


def compute_identity_proof(nonce: bytes, host: str, port: int) -> str:
    """HMAC-SHA256 proof that the caller holds this install's identity secret,
    bound to the loopback address and port the connection landed on. A proof
    relayed from an Unsloth on a different address/port (a squatter proxying to the
    real one, e.g. localhost resolving to ::1 while Unsloth is on 127.0.0.1) was
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


_PREVIEW_INCARNATION_KEY_PREFIX = "preview_incarnation:"


def preview_link_incarnation(subject: str, *, create: bool = True) -> str:
    """A value identifying this account, which a recreated namesake does not share.

    The signed payload named only the reusable username, so a deleted account's
    links resolved against its replacement's checkpoints. Dropped on retirement,
    so the replacement mints a different one.

    ``create = False`` when verifying an account that no longer exists: minting
    there hands the caller a fresh identity to verify against.
    """
    key = f"{_PREVIEW_INCARNATION_KEY_PREFIX}{subject}"
    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM app_secrets WHERE key = ?", (key,)).fetchone()
        if row is None:
            if not create:
                return ""
            conn.execute(
                "INSERT OR IGNORE INTO app_secrets (key, value) VALUES (?, ?)",
                (key, secrets.token_hex(16)),
            )
            conn.commit()
            row = conn.execute("SELECT value FROM app_secrets WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else ""
    finally:
        conn.close()


def clear_preview_link_incarnation(subject: str) -> None:
    """Retire an account's preview identity, revoking every link it had shared."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM app_secrets WHERE key = ?",
            (f"{_PREVIEW_INCARNATION_KEY_PREFIX}{subject}",),
        )
        conn.commit()
    finally:
        conn.close()


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
# Whether each memoized key was minted internally; set once, since minting decides it.
_api_key_internal_cache: dict[str, bool] = {}
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
        _api_key_internal_cache.clear()


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
    is_admin: bool = False,
    setup_code_expires_at: Optional[str] = None,
    reject_if_retired: bool = False,
) -> None:
    """
    Create the initial admin user in the database.

    Raises sqlite3.IntegrityError if username already exists.

    ``reject_if_retired`` reads the tombstone inside this insert's own write
    transaction. Checking first and inserting second can have the read answered
    from the pre-delete snapshot, so the replacement briefly shares a directory
    with the account it replaces; BEGIN IMMEDIATE puts this behind the delete.
    """
    from .hashing import hash_password

    salt, pwd_hash = hash_password(password)
    conn = get_connection()
    try:
        if reject_if_retired:
            conn.execute("BEGIN IMMEDIATE")
            reserved = conn.execute(
                "SELECT 1 FROM retired_usernames WHERE username = ?",
                (username,),
            ).fetchone()
            if reserved is not None:
                conn.rollback()
                raise ValueError(_RETIRED_USERNAME_MESSAGE)
        try:
            conn.execute(
                """
                INSERT INTO auth_user (
                    username,
                    password_salt,
                    password_hash,
                    jwt_secret,
                    must_change_password,
                    is_admin,
                    setup_code_expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    salt,
                    pwd_hash,
                    jwt_secret,
                    int(must_change_password),
                    int(is_admin),
                    setup_code_expires_at,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
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


def is_installation_owner(subject: str | None = None) -> bool:
    """Whether this account administers the install, for capability decisions.

    The seeded owner short-circuits without touching auth.db, so the single-user
    path cannot be demoted by a half-applied is_admin migration. Fails CLOSED,
    so an unreachable auth.db withholds a capability rather than granting one.
    """
    from utils.workspace_context import (
        LEGACY_WORKSPACE_SUBJECT,
        current_workspace_subject,
    )

    resolved = subject or current_workspace_subject()
    if resolved == LEGACY_WORKSPACE_SUBJECT or resolved == DEFAULT_ADMIN_USERNAME:
        return True
    try:
        return bool(is_admin(resolved))
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning("Could not check admin status for %s", resolved, exc_info = True)
        return False


def subject_may_reach_private_hosts(subject: str | None = None) -> bool:
    """Whether this account may point the backend at a loopback or LAN address.

    The owner may, since a local Ollama or vLLM endpoint is the ordinary reason
    to run Unsloth. A managed account may not: it cannot reach those services
    from its browser, so naming one turns the backend into a probe of that
    network. Single-user installs have only the owner.
    """
    return is_installation_owner(subject)


def is_admin(username: str) -> bool:
    """Return whether ``username`` may manage installation accounts."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT is_admin FROM auth_user WHERE username = ?",
            (username,),
        ).fetchone()
        return bool(row and row["is_admin"])
    finally:
        conn.close()


def list_users() -> list[dict]:
    """List public account metadata; password and signing data never leave storage."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT username, must_change_password, is_admin, setup_code_expires_at
            FROM auth_user
            ORDER BY is_admin DESC, username COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "username": row["username"],
                "must_change_password": bool(row["must_change_password"]),
                "is_admin": bool(row["is_admin"]),
                "setup_code_expires_at": row["setup_code_expires_at"],
                "setup_code_expired": _setup_code_expired(row["setup_code_expires_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def _new_setup_code() -> str:
    """Return an 80-bit, human-readable setup code without ambiguous glyphs."""
    raw = "".join(secrets.choice(_SETUP_CODE_ALPHABET) for _ in range(16))
    return "-".join(raw[index : index + 4] for index in range(0, len(raw), 4))


def _new_setup_code_expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes = SETUP_CODE_TTL_MINUTES)).isoformat()


def _setup_code_expired(expires_at: Optional[str]) -> bool:
    if expires_at is None:
        return False
    try:
        return datetime.fromisoformat(expires_at) <= datetime.now(timezone.utc)
    except (TypeError, ValueError):
        # Fail closed if a managed credential's expiry was corrupted.
        return True


def create_managed_user(username: str) -> dict:
    """Create a standard account and return its one-time-visible initial password."""
    # Retries the retirement as a side effect: the usual blocker is a file handle
    # that has since been released, and clearing it here lets the create proceed.
    if username_is_retired(username):
        raise ValueError(_RETIRED_USERNAME_MESSAGE)
    setup_code = _new_setup_code()
    expires_at = _new_setup_code_expiry()
    create_initial_user(
        username = username,
        password = setup_code,
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = True,
        is_admin = False,
        setup_code_expires_at = expires_at,
        reject_if_retired = True,
    )
    return {"setup_code": setup_code, "setup_code_expires_at": expires_at}


def regenerate_managed_user_setup_code(username: str) -> dict:
    """Replace a pending managed account's setup code and revoke its sessions."""
    from .hashing import hash_password

    setup_code = _new_setup_code()
    expires_at = _new_setup_code_expiry()
    salt, pwd_hash = hash_password(setup_code)
    jwt_secret = secrets.token_urlsafe(64)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT is_admin, must_change_password FROM auth_user WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            conn.rollback()
            raise KeyError(username)
        if bool(row["is_admin"]):
            conn.rollback()
            raise ValueError("Administrator setup codes cannot be regenerated")
        if not bool(row["must_change_password"]):
            conn.rollback()
            raise RuntimeError("Account setup is already complete")
        conn.execute(
            """
            UPDATE auth_user
            SET password_salt = ?, password_hash = ?, jwt_secret = ?,
                setup_code_expires_at = ?, must_change_password = 1
            WHERE username = ?
            """,
            (salt, pwd_hash, jwt_secret, expires_at, username),
        )
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.commit()
        return {"setup_code": setup_code, "setup_code_expires_at": expires_at}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def setup_code_login_allowed(username: str, password_hash: str) -> bool:
    """Check expiry against the exact credential hash that login verified."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT setup_code_expires_at
            FROM auth_user
            WHERE username = ? AND password_hash = ?
            """,
            (username, password_hash),
        ).fetchone()
        return row is not None and not _setup_code_expired(row["setup_code_expires_at"])
    finally:
        conn.close()


def _resolve_subject_owned_roots(username: str) -> tuple[list, bool]:
    """``(roots, complete)`` for every directory whose path derives from ``username``.

    The workspace tree, the projects tree (a separate Documents root), the tool
    sandbox tree and the scoped temporary root each key on
    workspace_key(username), so retiring only the first still hands a recycled
    name the rest.

    ``complete`` is False when a root could not be resolved at all, which is not
    "there was nothing to move": the caller must treat it as a failed retirement
    rather than renaming what it found and releasing the name.
    """
    from pathlib import Path

    from utils.paths.storage_roots import project_workspaces_root, studio_root, tmp_root
    from utils.workspace_context import run_in_workspace, workspace_key

    def _scoped() -> list:
        from core.inference.tools import sandbox_root

        # tmp_root keys on workspace_key like the rest, and unstructured seed
        # processing leaves plaintext parquet chunks under it.
        return [project_workspaces_root(), Path(sandbox_root()), tmp_root()]

    roots = [studio_root() / "workspaces" / workspace_key(username)]
    try:
        roots += run_in_workspace(username, _scoped)
    except Exception:
        logger.warning("Could not resolve every workspace root for %s", username)
        return roots, False
    return roots, True


def _subject_owned_roots(username: str) -> list:
    """The roots alone, for callers that do not act on an incomplete list."""
    return _resolve_subject_owned_roots(username)[0]


def _clear_username_tombstone(username: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM retired_usernames WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def username_is_retired(username: str) -> bool:
    """Whether ``username`` still has files a recreated account would inherit.

    Retried rather than permanent: the usual cause is a Windows handle held by a
    worker that has since exited, so a later attempt normally clears it.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM retired_usernames WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    if _retire_workspace_directory(username):
        _clear_username_tombstone(username)
        return False
    return True


# Probed only where the process already imported one: a module never imported
# cannot hold a render, and importing here would pull the diffusion stack into a
# deletion, where a failed optional import holds the name reserved for good.
_MEDIA_ENGINES = (
    ("core.inference.diffusion", "get_diffusion_backend"),
    ("core.inference.sd_cpp_backend", "get_sd_cpp_backend"),
    ("core.inference.video", "get_video_backend"),
)


def _loaded_media_backends() -> list:
    """Media backends already live in this process."""
    import sys

    backends = []
    for module_name, accessor in _MEDIA_ENGINES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        getter = getattr(module, accessor, None)
        if getter is None:
            continue
        backends.append(getter())
    return backends


# The engines report idle as "ready" or None and failed as "error"; any other
# phase names a stage of a load still running.
_MEDIA_LOAD_IDLE_PHASES = (None, "ready", "error")


def _media_load_active(backend, username: str) -> bool:
    """Whether ``backend`` is loading a model for this account right now.

    A load is not a render, and it holds the account's identity the same way: the
    loading state carries the subject, and the payload names the repo or local
    directory it is pulling. A username released while one is in flight lets a
    recreated namesake match that subject and drive the load through the ordinary
    progress and unload paths.
    """
    probe = getattr(backend, "load_progress", None)
    if not callable(probe):
        return False
    return probe(subject = username).get("phase") not in _MEDIA_LOAD_IDLE_PHASES


# Fields a media backend reports a loaded model under. A repo id is install-wide
# and shared by design; only a filesystem path can name one account's private
# weights.
_MEDIA_STATUS_PATH_FIELDS = ("repo_id", "base_repo", "model_path", "local_path", "resolved")


def _status_names_a_path_under(status: dict, root: str) -> bool:
    """Whether any model field in ``status`` resolves inside ``root``."""
    import os

    for field in _MEDIA_STATUS_PATH_FIELDS:
        value = status.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            real = os.path.realpath(os.path.expanduser(value.strip()))
        except (OSError, RuntimeError, ValueError):
            continue
        if real == root or real.startswith(root + os.sep):
            return True
    return False


def _workspace_jobs_active(username: str) -> bool:
    """Whether anything is still running under this account's workspace.

    Quiescing signals, it does not wait, and a worker still unwinding recreates
    the pathname on its next lookup. So the tombstone is held until nothing is
    running, and the create path's existing retry releases it.

    Fails CLOSED: guessing wrong costs one reserved name, against a live worker
    writing into somebody else's files.
    """

    def _training_active() -> bool:
        from core.training.training import get_training_backend
        backend = get_training_backend()
        return bool(backend.is_training_active() and backend.owns_workspace(username))

    def _diffusion_active() -> bool:
        from core.training.diffusion_training_service import get_diffusion_training_service
        service = get_diffusion_training_service()
        return bool(service.is_active() and service.owns_workspace(username))

    def _export_active() -> bool:
        from core.export import get_export_backend
        orchestrator = get_export_backend()
        return bool(orchestrator.is_export_active() and orchestrator.owns_workspace(username))

    def _generations_active() -> bool:
        from state import active_generations
        return bool(active_generations.active_thread_ids(username))

    def _media_renders_active() -> bool:
        return any(
            backend.generate_progress(subject = username).get("active")
            or _media_load_active(backend, username)
            for backend in _loaded_media_backends()
        )

    def _recipe_job_active() -> bool:
        from core.data_recipe.jobs.manager import get_job_manager
        manager = get_job_manager()
        return bool(manager.is_active() and manager.owns_workspace(username))

    def _mcp_sessions_cached() -> bool:
        from core.inference.mcp_client import workspace_has_cached_sessions
        return bool(workspace_has_cached_sessions(username))

    def _research_runs_active() -> bool:
        # A supervisor between model calls holds no visible lease, but its row is
        # non-terminal and the run reopens this account's databases.
        from storage import research_runs_db
        return bool(research_runs_db.unfinished_run_ids())

    def _rag_workers_active() -> bool:
        # Workspace-bound threads whose next rag_db_path() recreates the directory.
        from core.rag import folder_sync
        from storage import rag_db

        if folder_sync.workspace_sync_worker_active(username):
            return True
        try:
            if not rag_db.rag_available():
                return False
        except Exception:  # noqa: BLE001 - an unreadable rag.db answers nothing
            return True
        return bool(rag_db.live_ingestion_or_sync_jobs())

    from utils.workspace_context import run_in_workspace

    for what, probe in (
        ("training", _training_active),
        ("diffusion training", _diffusion_active),
        ("export", _export_active),
        ("chat generations", _generations_active),
        ("media renders", _media_renders_active),
        ("data recipe job", _recipe_job_active),
        ("cached MCP sessions", _mcp_sessions_cached),
        ("research runs", _research_runs_active),
        ("RAG workers", _rag_workers_active),
    ):
        try:
            if run_in_workspace(username, probe):
                logger.info("Holding %s reserved: %s still running", username, what)
                return True
        except Exception:  # noqa: BLE001 - unanswerable means busy; see the docstring
            logger.warning("Could not check %s for %s", what, username, exc_info = True)
            return True
    return False


def _retire_workspace_directory(username: str) -> bool:
    """Move a deleted account's directories aside so a recreated name cannot
    inherit them: the keys are a pure function of the username. Renamed rather
    than deleted, so they stay recoverable by hand."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    directories, retired_all = _resolve_subject_owned_roots(username)
    # Before the rename: Windows refuses to rename a directory holding an open
    # file, and the WAL keeper holds studio.db open for the life of the process.
    try:
        from storage.studio_db import close_wal_keeper_for
        from utils.paths.storage_roots import studio_db_path
        from utils.workspace_context import run_in_workspace

        close_wal_keeper_for(run_in_workspace(username, studio_db_path))
    except Exception:  # noqa: BLE001 - a keeper we cannot name is one we cannot close
        logger.warning("Could not release the studio.db keeper for %s", username, exc_info = True)
    for directory in directories:
        try:
            if not directory.is_dir():
                continue
            retired = directory.with_name(f"{directory.name}-deleted-{stamp}")
            suffix = 1
            while retired.exists():
                retired = directory.with_name(f"{directory.name}-deleted-{stamp}-{suffix}")
                suffix += 1
            directory.rename(retired)
        except OSError:
            # Never let a locked file block the account revocation itself; the
            # caller tombstones the name instead so nobody inherits the files.
            logger.warning("Could not retire %s for %s", directory, username)
            retired_all = False
    # The ready-path caches key on the pathname a namesake reuses, so its fresh
    # database would be served without a schema.
    from storage import schema_cache

    schema_cache.forget_all()
    _forget_workspace_registries(username)
    # Moving the files is not enough: a worker still bound to this subject
    # recreates the pathname on its next lookup.
    return retired_all and not _workspace_jobs_active(username)


# Every registry that keys on the username and outlives the account's files, as
# (what, module, function taking the subject).
#
# A table rather than seven copies of the same lazy import and try/except. Each of
# these was reported separately, one per review round, because nothing said out
# loud that a per-account registry has to be retired with the account: this list
# is that statement, and it is the one place a new one has to be added.
#
# Imported by name, at call time. A reduced install where one of these modules
# will not import must still be able to delete an account.
_WORKSPACE_REGISTRIES: tuple[tuple[str, str, str], ...] = (
    # The preview token names the username, so a namesake's run at the same ref
    # would answer links the previous holder shared.
    ("preview links", "auth.storage", "clear_preview_link_incarnation"),
    # Process-lifetime memos, which would index into the previous embedding space.
    ("embedding memos", "utils.embedding_model_settings", "forget_workspace"),
    ("dataset grants", "hub.services.datasets.cache_access", "forget_workspace"),
    # The scan records authorize discarding remote code, so a namesake could
    # delete a dependency another account's model still needs.
    ("remote-code grants", "routes.models", "forget_scan_created_remote_code"),
    # Downloads are not workspace jobs, so the sweep never saw them.
    ("download initiators", "hub.services.download_lifecycle", "forget_workspace_initiators"),
    ("dictation grants", "routes.inference", "forget_stt_model_downloader"),
    # The video route's own job map, beside the backend state the sweep reached.
    ("video jobs", "routes.video", "forget_workspace_jobs"),
)


def _forget_workspace_registries(username: str) -> None:
    """Drop what every per-account registry still holds for a deleted account.

    Never fatal: a registry that cannot be reached must not block the deletion
    itself, which would leave an account the owner cannot remove.
    """
    import importlib

    for what, module_path, attr in _WORKSPACE_REGISTRIES:
        try:
            getattr(importlib.import_module(module_path), attr)(username)
        except Exception:  # noqa: BLE001 - see the docstring
            logger.warning("Could not clear %s for %s", what, username, exc_info = True)


def _quiesce_workspace_jobs(username: str) -> None:
    """Stop the jobs this account owns, before its files are moved aside.

    A spawned worker outlives the row that authorised it, and the ownership
    guards then hide it from the owner while the deleted account can no longer
    sign in to stop it: a multi-hour GPU job would burn the card until restart.

    Best effort and never fatal, since the alternative is an account the owner
    cannot remove at all. Each stop is guarded by the subsystem's ownership
    predicate AND its "is something running" check, so an unclaimed singleton is
    never mistaken for this account's.
    """

    def _stop_training() -> None:
        from core.training.training import get_training_backend
        backend = get_training_backend()
        if backend.is_training_active() and backend.owns_workspace(username):
            backend.stop_training(save = False)

    def _stop_diffusion_training() -> None:
        from core.training.diffusion_training_service import get_diffusion_training_service
        service = get_diffusion_training_service()
        if service.is_active() and service.owns_workspace(username):
            service.stop(save = False)

    def _stop_export() -> None:
        from core.export import get_export_backend
        orchestrator = get_export_backend()
        if orchestrator.is_export_active() and orchestrator.owns_workspace(username):
            orchestrator.cancel_export()

    def _stop_generations() -> None:
        from state import active_generations
        for thread_id in active_generations.active_thread_ids(username):
            active_generations.cancel_thread(thread_id, subject = username)

    def _stop_media_renders() -> None:
        # The galleries resolve the workspace root when the render FINISHES, so a
        # running one writes into whoever takes the name next. A load counts too
        # and cancel_generate does not reach it, hence the engine's own path.
        for backend in _loaded_media_backends():
            if backend.generate_progress(subject = username).get("active"):
                backend.cancel_generate(subject = username)
            if _media_load_active(backend, username):
                backend.unload(subject = username)

    def _close_mcp_sessions() -> None:
        # The session key holds the reusable username, so a namesake created
        # inside the idle TTL could check the session out and inherit its state.
        from core.inference.mcp_client import close_mcp_sessions
        close_mcp_sessions()

    def _stop_research_runs() -> None:
        # Through the run row, not the in-memory event: the row is what both the
        # supervisor and a restart consult.
        from storage import research_runs_db
        for run_id in research_runs_db.unfinished_run_ids():
            try:
                research_runs_db.request_cancel(run_id)
            except KeyError:
                continue

    def _stop_rag_workers() -> None:
        # Only this account's: stop_auto_sync() is a process shutdown. A running
        # ingestion finishes, and the probe holds the tombstone until it does.
        from core.rag import folder_sync
        folder_sync.stop_workspace_auto_sync(username)

    def _shutdown_idle_export_worker() -> None:
        # is_export_active() is false once a checkpoint has loaded, so the cancel
        # left it resident for a namesake that passes owns_workspace().
        from core.export import get_export_backend

        orchestrator = get_export_backend()
        if not orchestrator.owns_workspace(username):
            return
        if orchestrator.is_export_active():
            return
        if orchestrator.is_worker_alive() or orchestrator.current_checkpoint:
            orchestrator._shutdown_subprocess()
        orchestrator.current_checkpoint = None
        orchestrator.is_vision = False
        orchestrator.is_peft = False
        with orchestrator._workspace_guard():
            orchestrator._workspace_subject = None

    def _reset_training() -> None:
        # A terminal run is not active, so nothing above stopped it, and the
        # singleton kept the subject beside what a namesake then read back.
        from core.training.training import get_training_backend
        backend = get_training_backend()
        if backend.owns_workspace(username):
            backend.reset_retained_state(username)

    def _reset_recipe_state() -> None:
        # cancel() on a terminal job clears neither _job nor the subject, so
        # /jobs/current handed a namesake the old job and its rows.
        from core.data_recipe.jobs.manager import get_job_manager
        get_job_manager().reset_retained_state(username)

    def _forget_terminal_video() -> None:
        # A completed record holds the whole recipe, and reports inactive, so the
        # cancel left it for whoever takes the name next.
        from core.inference.video import get_video_backend
        get_video_backend().forget_terminal_video(subject = username)

    def _clear_api_monitor() -> None:
        # Fifty entries of prompts and replies, authorized only by the subject string.
        from core.inference.api_monitor import api_monitor
        api_monitor.clear(subject = username)

    def _unload_private_resident_media() -> None:
        # A namesake derives the same workspace root, so an idle pipeline loaded
        # from the old one reads as theirs.
        from utils.paths.storage_roots import workspace_root
        try:
            private_root = str(run_in_workspace(username, workspace_root).resolve())
        except (OSError, RuntimeError, ValueError):
            return
        for backend in _loaded_media_backends():
            try:
                status = backend.status()
            except Exception:  # noqa: BLE001 - an engine that cannot answer is left alone
                continue
            if not isinstance(status, dict) or not status.get("loaded"):
                continue
            if not _status_names_a_path_under(status, private_root):
                continue
            backend.unload()

    def _unload_private_resident_text() -> None:
        # Process-wide backends plus a username-keyed ownership record, so a
        # namesake inherits both. Dropping the record alone leaves the weights.
        from routes.inference import (
            forget_text_model_owner,
            resident_text_model_workspace,
            retire_text_model_owner,
        )

        if resident_text_model_workspace() != username:
            return
        # Fenced first, dropped only once the weights are gone: the unloads are
        # best effort, and an unowned model passes the containment fallback.
        retire_text_model_owner(username)
        try:
            from routes.inference import get_llama_cpp_backend
            get_llama_cpp_backend().unload_model()
        except Exception:  # noqa: BLE001 - an engine that cannot answer is left alone
            logger.warning("Could not unload the resident GGUF for %s", username, exc_info = True)
        try:
            from routes.inference import _peek_inference_backend

            backend = _peek_inference_backend()
            active = getattr(backend, "active_model_name", None) if backend is not None else None
            if backend is not None and active:
                backend.unload_model(active)
        except Exception:  # noqa: BLE001 - same
            logger.warning("Could not unload the resident model for %s", username, exc_info = True)
        # Nothing resident means nothing left to fence.
        from routes.inference import _resident_text_model_identifiers

        try:
            if not _resident_text_model_identifiers():
                forget_text_model_owner()
        except Exception:  # noqa: BLE001 - the fence stays if we cannot tell
            pass

    def _reset_diffusion_training() -> None:
        # is_active() is false for a terminal run, so the stop skipped it and the
        # singleton kept the subject beside the paths /diffusion/status returns.
        from core.training.diffusion_training_service import get_diffusion_training_service
        service = get_diffusion_training_service()
        if service.owns_workspace(username):
            service.reset_retained_state(username)

    def _stop_recipe_job() -> None:
        # The worker keeps the artifact root it was given, so it can recreate the
        # retired pathname.
        from core.data_recipe.jobs.manager import get_job_manager

        manager = get_job_manager()
        if not manager.owns_workspace(username):
            return
        job_id = manager.get_current_job_id()
        if job_id is not None:
            manager.cancel(job_id)

    from utils.workspace_context import run_in_workspace

    for what, stop in (
        ("training", _stop_training),
        ("diffusion training", _stop_diffusion_training),
        ("export", _stop_export),
        ("chat generations", _stop_generations),
        ("media renders", _stop_media_renders),
        ("data recipe job", _stop_recipe_job),
        ("research runs", _stop_research_runs),
        ("RAG folder sync worker", _stop_rag_workers),
        ("cached MCP sessions", _close_mcp_sessions),
        # Below: state that OUTLIVES the work. Nothing above stops it, because by
        # then there is nothing left to stop.
        ("retained diffusion training state", _reset_diffusion_training),
        ("retained training state", _reset_training),
        ("idle export worker", _shutdown_idle_export_worker),
        ("retained recipe job", _reset_recipe_state),
        ("completed video record", _forget_terminal_video),
        ("API monitor entries", _clear_api_monitor),
        ("private resident media models", _unload_private_resident_media),
        ("private resident text models", _unload_private_resident_text),
    ):
        try:
            run_in_workspace(username, stop)
        except Exception:  # noqa: BLE001 - see the docstring; never fatal
            logger.warning("Could not stop %s for %s", what, username, exc_info = True)


def delete_managed_user(username: str) -> bool:
    """Revoke and delete a non-admin account, retiring its workspace files."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT is_admin FROM auth_user WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            conn.rollback()
            return False
        if bool(row["is_admin"]):
            conn.rollback()
            raise ValueError("Administrator accounts cannot be deleted")
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.execute("DELETE FROM api_keys WHERE username = ?", (username,))
        conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        # Same transaction as the delete: both must never be absent at once, or a
        # racing create binds to a workspace this call is about to rename.
        conn.execute(
            "INSERT OR IGNORE INTO retired_usernames (username, created_at) VALUES (?, ?)",
            (username, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    # Before the sweep, which only sees work that has already started: a request
    # admitted a moment ago is invisible to it.
    from utils.workspace_context import note_workspace_retired

    note_workspace_retired(username)
    # After the credentials are gone and before the rename below.
    _quiesce_workspace_jobs(username)
    # Cleared only once every root is out of the way; a failure leaves the name
    # reserved and the next create retries the retirement.
    if _retire_workspace_directory(username):
        _clear_username_tombstone(username)
    return True


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
            is_admin = True,
        )
        return True
    except sqlite3.IntegrityError:
        return False


def update_password(
    username: str,
    new_password: str,
    *,
    revoke_refresh_tokens: bool = False,
    expect_password_hash: Optional[str] = None,
    preserve_desktop_secret: bool = False,
) -> Optional[str]:
    """Update password, clear first-login requirement, rotate JWT secret.

    Returns the new JWT secret, or None when nothing was updated. Callers that
    mint tokens for the caller must sign with the returned secret: re-reading it
    would pick up a reset that landed between this commit and the mint.

    ``revoke_refresh_tokens`` deletes the user's refresh tokens in the SAME
    transaction: a separate delete could fail after the password commit and
    leave a pre-change token still able to mint access tokens.

    ``expect_password_hash`` makes the write conditional on the credential the
    caller verified still being current, so a request that checked the old
    password cannot overwrite a reset that landed while it was in flight.
    Returns None when the credential moved underneath it.

    ``preserve_desktop_secret`` keeps the local desktop credential valid. It is
    for a caller that already authenticated as the desktop app: revoking the
    secret it is currently using would break desktop auto-auth for a change the
    desktop itself made.
    """
    from .hashing import hash_password

    salt, pwd_hash = hash_password(new_password)
    jwt_secret = secrets.token_urlsafe(64)
    # app_secrets is install-wide and holds the OWNER's desktop credential, so a
    # managed account's password change must not revoke it. Name as well as flag:
    # an install seeded before the is_admin column still owns it. Read before the
    # write transaction opens, so the lookup never contends with the UPDATE below
    # on its own connection.
    owns_install_secrets = username == DEFAULT_ADMIN_USERNAME or is_admin(username)

    conn = get_connection()
    try:
        # setup_code_expires_at is cleared with the write: a managed account that
        # swaps its one-time setup code for a real password must not keep an
        # expiring credential alongside it.
        if expect_password_hash is None:
            cursor = conn.execute(
                """
                UPDATE auth_user
                SET password_salt = ?, password_hash = ?, jwt_secret = ?,
                    must_change_password = 0, setup_code_expires_at = NULL
                WHERE username = ?
                """,
                (salt, pwd_hash, jwt_secret, username),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE auth_user
                SET password_salt = ?, password_hash = ?, jwt_secret = ?,
                    must_change_password = 0, setup_code_expires_at = NULL
                WHERE username = ? AND password_hash = ?
                """,
                (salt, pwd_hash, jwt_secret, username, expect_password_hash),
            )
        if revoke_refresh_tokens and cursor.rowcount > 0:
            conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.commit()
        if cursor.rowcount > 0:
            clear_bootstrap_password()
            if not preserve_desktop_secret and owns_install_secrets:
                clear_desktop_secret()
            return jwt_secret
        return None
    finally:
        conn.close()


def save_refresh_token(
    token: str,
    username: str,
    expires_at: str,
    *,
    is_desktop: bool = False,
    secret_gen: Optional[str] = None,
) -> None:
    """
    Store a hashed refresh token with its associated username and expiry.

    ``secret_gen`` binds the token to a credential version; it defaults to the
    current one, and callers that already verified a credential must pass the
    version they verified rather than let this re-read a rotated one.
    """
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        if secret_gen is None:
            secret_gen = _current_generation(conn, username)
        conn.execute(
            """
            INSERT INTO refresh_tokens (token_hash, username, expires_at, is_desktop, secret_gen)
            VALUES (?, ?, ?, ?, ?)
            """,
            (token_hash, username, expires_at, int(is_desktop), secret_gen),
        )
        conn.commit()
    finally:
        conn.close()


def consume_refresh_token(token: str) -> Optional[Tuple[str, bool, str]]:
    """Atomically validate-and-delete a refresh token for single-use rotation.

    DELETE RETURNING fuses validate and delete into one statement so two
    concurrent refresh requests cannot both consume the same token. Returns
    ``(username, is_desktop, jwt_secret)``; the caller must mint the replacement
    tokens against that secret so a rotation landing mid-refresh cannot issue a
    post-rotation session from a pre-rotation token.
    """
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        # One transaction with the delete: an unstamped legacy row has no
        # generation to compare, so reading the credential after committing would
        # hand a reset's new secret to a token issued before it.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (now,),
        )
        cur = conn.execute(
            """
            DELETE FROM refresh_tokens
            WHERE token_hash = ? AND expires_at >= ?
            RETURNING username, is_desktop, secret_gen
            """,
            (token_hash, now),
        )
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        secret = _current_secret(conn, row["username"])
        conn.commit()
        if secret is None:
            return None
        if row["secret_gen"] is not None and row["secret_gen"] != credential_generation(secret):
            return None
        return row["username"], bool(row["is_desktop"]), secret
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
            SELECT id, username, expires_at, is_desktop, secret_gen FROM refresh_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        if row["secret_gen"] is not None and row["secret_gen"] != _current_generation(
            conn, row["username"]
        ):
            conn.execute("DELETE FROM refresh_tokens WHERE id = ?", (row["id"],))
            conn.commit()
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


def validate_desktop_secret_with_credential(raw_secret: str) -> Optional[Tuple[str, str]]:
    """Validate the desktop secret and return ``(username, jwt_secret)``.

    Both reads share one transaction so the returned secret is the credential
    version the desktop secret was checked against; a reset landing mid-request
    then invalidates the tokens minted from it rather than blessing them.
    """
    if not raw_secret.startswith(DESKTOP_SECRET_PREFIX):
        return None

    secret_hash = _pbkdf2_desktop_secret(raw_secret)
    conn = get_connection()
    try:
        conn.execute("BEGIN")
        row = conn.execute(
            "SELECT value FROM app_secrets WHERE key = ?",
            (_DESKTOP_SECRET_HASH_KEY,),
        ).fetchone()
        if row is None or not secrets.compare_digest(row["value"], secret_hash):
            return None
        jwt_secret = _current_secret(conn, DEFAULT_ADMIN_USERNAME)
        if jwt_secret is None:
            return None
        return DEFAULT_ADMIN_USERNAME, jwt_secret
    finally:
        conn.rollback()
        conn.close()


def validate_desktop_secret(raw_secret: str) -> Optional[str]:
    """Return the real admin username when the desktop secret matches."""
    verified = validate_desktop_secret_with_credential(raw_secret)
    return verified[0] if verified else None


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

# The ``name`` a workflow mints its internal key under. Unsloth mints internal
# keys for more than one workflow and they do not carry the same authority, so
# the name is the only thing that tells them apart after the fact. Deep Research
# is durable: its hops outlive the session that started them, so they carry a key
# instead of a JWT and still have to reach the saved connection the run was
# created with. A data-recipe key is handed to a user-authored recipe subprocess
# and needs nothing but this host's local /v1, so it must never gain that reach.
DEEP_RESEARCH_WORKFLOW_KEY_NAME = "deep-research workflow"


def create_api_key(
    username: str,
    name: str,
    expires_at: Optional[str] = None,
    internal: bool = False,
    expect_gen: Optional[str] = None,
) -> Tuple[str, dict]:
    """Create a new API key for *username*.

    Returns ``(raw_key, row_dict)`` where *raw_key* is shown to the user
    exactly once.  The database only stores the PBKDF2 hash.

    Pass ``internal=True`` for keys minted by workflows (e.g. data-recipe
    runs) that should not appear in user-facing key listings.

    ``expect_gen`` ties the insert to the credential generation the request
    authenticated under, so a session revoked by a concurrent password reset
    cannot mint a key that outlives it. Raises ``CredentialRotated`` if it moved.
    """
    raw_key = API_KEY_PREFIX + secrets.token_hex(16)
    key_hash = _pbkdf2_api_key(raw_key)
    key_prefix = raw_key[len(API_KEY_PREFIX) : len(API_KEY_PREFIX) + 8]
    now = datetime.now(timezone.utc).isoformat()

    conn = get_connection()
    try:
        if expect_gen is not None:
            conn.execute("BEGIN IMMEDIATE")
            if _current_generation(conn, username) != expect_gen:
                raise CredentialRotated(
                    "The credential this request authenticated with was revoked."
                )
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


def is_internal_api_key(raw_key: str) -> bool:
    """Whether *raw_key* is a workflow-minted internal key rather than a user's own.

    Lets request-scoped code (the API monitor) tell Unsloth's own background work from a
    third party using Unsloth as an API server. The answer is memoized because this runs on
    the event loop for every API-key request and a key's origin is fixed when it is minted.
    """
    if not raw_key.startswith(API_KEY_PREFIX):
        return False
    cache_id = _api_key_cache_id(raw_key)
    cached_internal = _api_key_internal_cache.get(cache_id)
    if cached_internal is not None:
        return cached_internal
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT is_internal FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    internal = bool(row["is_internal"])
    with _api_key_hash_cache_lock:
        if len(_api_key_hash_cache) >= _API_KEY_HASH_CACHE_MAX:
            _api_key_hash_cache.clear()
            _api_key_internal_cache.clear()
        _api_key_hash_cache[cache_id] = key_hash
        _api_key_internal_cache[cache_id] = internal
    return internal


def internal_api_key_name(raw_key: str) -> Optional[str]:
    """The workflow name *raw_key* was minted under, or ``None`` if it is not internal.

    ``is_internal_api_key`` answers "is this Unsloth's own key", which is the right
    question for a monitor label but far too coarse for authorization: a
    data-recipe key runs inside a recipe the user authored, so treating it as
    equal to the Deep Research hop would let that recipe spend any saved cloud
    credential. The name is fixed when the key is minted and is the only durable
    thing that separates the two.

    Deliberately not memoized: this is read on the external-provider path only,
    once per request that carries an API key, and a stale answer here would be a
    stale authorization. The PBKDF2 derivation is taken from the shared hash
    cache when it is warm, so the cost is one indexed lookup.
    """
    if not raw_key.startswith(API_KEY_PREFIX):
        return None
    cache_id = _api_key_cache_id(raw_key)
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM api_keys WHERE key_hash = ? AND is_internal = 1 AND is_active = 1",
            (key_hash,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    name = row["name"]
    return name if isinstance(name, str) else None


def validate_api_key(raw_key: str) -> Optional[str]:
    """Validate *raw_key* and return the owning username, or ``None``."""
    verified = validate_api_key_with_credential(raw_key)
    return verified[0] if verified else None


def validate_api_key_with_credential(
    raw_key: str, *, touch: bool = True
) -> Optional[Tuple[str, str]]:
    """Validate *raw_key* and return ``(username, jwt_secret)``, or ``None``.

    Also updates ``last_used_at`` on success. The key check and the credential
    read share one write transaction, so the returned version is the one the key
    was actually valid under: a reset committing right after cannot have its new
    generation handed to a request the key it revoked authenticated.

    ``touch=False`` drops that stamp, and with it the write transaction, for a caller
    that only asks whether the key authenticates and never binds a write to the
    generation. One request must not count as two uses, and on sqlite the write lock
    is global, so an advisory check has no business taking it.
    """
    cache_id = _api_key_cache_id(raw_key)
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        if touch:
            conn.execute("BEGIN IMMEDIATE")
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
                    # is_internal_api_key sizes itself against the hash cache, so clearing one
                    # without the other lets the origin cache grow past the bound. Deep Research
                    # mints a fresh internal key per model call, so that adds up.
                    _api_key_internal_cache.clear()
                _api_key_hash_cache[cache_id] = key_hash
        if not row["is_active"]:
            return None
        if row["expires_at"] is not None:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires:
                return None
        secret = _current_secret(conn, row["username"])
        if secret is None:
            return None
        if touch:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row["id"]),
            )
            conn.commit()
        return row["username"], secret
    finally:
        conn.rollback()
        conn.close()
