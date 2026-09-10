# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for auth data (user credentials + JWT secret)."""

import contextlib
from contextlib import contextmanager

import hashlib
import hmac
import ipaddress
import os
import re
import secrets
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta, timezone
import uuid
from typing import Iterator, Optional, Tuple

from utils.paths import auth_db_path, ensure_dir

DB_PATH = auth_db_path()
DEFAULT_ADMIN_USERNAME = "unsloth"

# Single source for the password policy; models/auth.py ChangePasswordRequest and the terminal
# prompt both enforce it. Keep the unsloth_cli mirror in sync.
MIN_PASSWORD_LENGTH = 8

# Plaintext bootstrap password file, deleted on first password change.
_BOOTSTRAP_PW_PATH = DB_PATH.parent / ".bootstrap_password"

# In-process cache to avoid re-reading the file on every HTML serve.
_bootstrap_password: Optional[str] = None


def _bootstrap_file_bytes(password: str) -> bytes:
    """Exact on-disk form: the secret plus one LF. Bytes, not text: text mode writes CRLF on Windows,
    and `$(cat ...)` strips the LF but leaves the CR attached to the credential."""
    return (password + "\n").encode("utf-8")


def _persist_bootstrap_password(password: str) -> None:
    """Atomically write the bootstrap password 0600, LF terminated on every OS: a partial write would
    destroy the only plaintext recovery credential."""
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
    """Append the LF a pre-newline release left off. Append-only, and only when the file is exactly the
    credential: clear_bootstrap_password() may unlink or (when unlink fails, notably on Windows
    while this descriptor is open) truncate through another descriptor after we read, so a rewrite
    could restore revoked plaintext. An append cannot: worst case is a lone terminator over a
    cleared file, which strips back to no bootstrap password. Pre-newline releases wrote no
    terminator at all, so that is the only shape in the wild."""
    if raw != password.encode("utf-8"):
        return

    # O_BINARY: Windows text mode turns the LF back into CRLF, the bug being fixed.
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

    # An unreadable file must mean "no bootstrap password"; no caller handles a raise.
    try:
        raw = _BOOTSTRAP_PW_PATH.read_bytes()
        password = raw.decode("utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not password:
        return None

    # Older releases wrote no terminator; a read-only auth dir must not fail startup.
    if raw != _bootstrap_file_bytes(password):
        try:
            _normalise_bootstrap_file(raw, password)
        except OSError:
            pass
    return password


def generate_bootstrap_password() -> str:
    """Generate a 4-word diceware passphrase and persist it to disk. Persisted (the DB stores only the
    hash) so it survives restarts; later calls return it."""
    global _bootstrap_password

    if _bootstrap_password is not None:
        return _bootstrap_password

    persisted = _read_persisted_bootstrap_password()
    if persisted:
        _bootstrap_password = persisted
        return _bootstrap_password

    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the same passphrase survives restarts until password change.
    ensure_dir(_BOOTSTRAP_PW_PATH.parent)
    _persist_bootstrap_password(_bootstrap_password)

    return _bootstrap_password


def get_bootstrap_password() -> Optional[str]:
    return _bootstrap_password


def _load_bootstrap_password() -> Optional[str]:
    """Load an existing bootstrap password without creating one. Upgrades take this path, not
    generate_bootstrap_password() (ensure_default_admin short-circuits once the admin row exists),
    so it has to normalise too."""
    global _bootstrap_password
    _bootstrap_password = _read_persisted_bootstrap_password()
    return _bootstrap_password


def clear_bootstrap_password() -> None:
    """Delete the persisted bootstrap password file (after a password change). Best-effort: the new
    hash is already committed, so a locked or undeletable file must not fail the change."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        try:
            _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)
        except OSError as e:
            # Truncate when removal fails: stale plaintext would otherwise be re-seeded if auth.db is recreated.
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
                # Stale plaintext is still on disk and would be reused if auth.db is reset.
                message = (
                    f"Warning: could not delete or clear {_BOOTSTRAP_PW_PATH.name} ({e}); "
                    "its old bootstrap password is still on disk. Remove it manually to "
                    "prevent reuse after a reset."
                )
            print(message, file = sys.stderr, flush = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper for refresh token storage. Plain SHA-256 is intentional: refresh tokens are
    384-bit random strings, so a slow KDF adds no security while costing per-refresh latency. API
    keys use the separate ``_pbkdf2_api_key`` helper, only to satisfy CodeQL's
    ``py/weak-sensitive-data-hashing`` query."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class CredentialRotated(Exception):
    """A password reset revoked the credential this request authenticated with."""


def credential_generation(jwt_secret: str) -> str:
    """Marker for the credential version a refresh token was issued under. Every password change
    rotates ``jwt_secret``, so a token stamped with the previous one is rejected even if it was
    inserted after the revoking DELETE."""
    return hashlib.sha256(jwt_secret.encode("utf-8")).hexdigest()


# Downgrade fence: managed creds live in ``account_*`` with prefixed hashes, so a build without
# account support 401s a managed login.
_FENCE_PREFIX = "account:"
_FENCED_HASH_SQL = "IN (?, ?)"
_LEGACY_PASSWORD_HASH_SENTINEL = "managed-account"
_SECRET_SQL = "COALESCE(account_jwt_secret, jwt_secret)"
_PASSWORD_HASH_SQL = "COALESCE(account_password_hash, password_hash)"
_PASSWORD_SALT_SQL = "COALESCE(account_password_salt, password_salt)"


def _is_owner_name(username: str) -> bool:
    return username == DEFAULT_ADMIN_USERNAME


def _fenced_hash(digest: str, username: str) -> str:
    return digest if _is_owner_name(username) else _FENCE_PREFIX + digest


def _hash_candidates(digest: str) -> tuple[str, str]:
    return digest, _FENCE_PREFIX + digest


def _legacy_dummies() -> tuple[str, str, str]:
    return secrets.token_hex(16), _LEGACY_PASSWORD_HASH_SENTINEL, secrets.token_urlsafe(64)


def _current_secret(conn: sqlite3.Connection, username: str) -> Optional[str]:
    row = conn.execute(
        f"SELECT {_SECRET_SQL} AS jwt_secret FROM auth_user WHERE username = ?", (username,)
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


_auth_schema_ready: set[tuple[str, int, int, int]] = set()


def get_connection() -> sqlite3.Connection:
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    # sqlite3.connect would create the DB 0644 under a 022 umask, exposing identity secrets and password hashes.
    for _path, _mode in ((DB_PATH.parent, 0o700), (DB_PATH, 0o600)):
        try:
            os.chmod(_path, _mode)
        except OSError:
            pass
    conn.row_factory = sqlite3.Row
    # Set busy_timeout before journal_mode=WAL: switching journal mode needs a lock and would otherwise
    # raise SQLITE_BUSY.
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    file_stat = DB_PATH.stat()
    schema_key = (
        str(DB_PATH),
        file_stat.st_dev,
        file_stat.st_ino,
        conn.execute("PRAGMA schema_version").fetchone()[0],
    )
    if schema_key in _auth_schema_ready:
        return conn
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
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(auth_user)")}
    if "must_change_password" not in columns:
        conn.execute(
            "ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
        )
    _ensure_account_columns(conn, columns)
    _ensure_account_api_keys(conn, api_key_columns)
    refresh_columns = {row["name"] for row in conn.execute("PRAGMA table_info(refresh_tokens)")}
    if "is_desktop" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN is_desktop INTEGER NOT NULL DEFAULT 0")
    if "secret_gen" not in refresh_columns:
        conn.execute("ALTER TABLE refresh_tokens ADD COLUMN secret_gen TEXT")
    conn.commit()
    _auth_schema_ready.add(
        (
            *schema_key[:3],
            conn.execute("PRAGMA schema_version").fetchone()[0],
        )
    )
    return conn


# No lock needed: INSERT OR IGNORE is atomic and concurrent populations converge on the same value.
# ── API-key PBKDF2 salt ────────────────────────────────────────────────
_api_key_pbkdf2_salt_cache: Optional[bytes] = None


_ACCOUNT_COLUMNS = (
    ("account_id", "TEXT"),
    ("role", "TEXT NOT NULL DEFAULT 'user'"),
    ("is_active", "INTEGER NOT NULL DEFAULT 1"),
    ("created_at", "TEXT"),
    ("setup_code_hash", "TEXT"),
    ("setup_code_expires_at", "TEXT"),
    ("account_password_salt", "TEXT"),
    ("account_password_hash", "TEXT"),
    ("account_jwt_secret", "TEXT"),
)


_owner_id_repaired: set[str] = set()


def _repair_owner_account_id(conn: sqlite3.Connection) -> None:
    db_key = str(DB_PATH)
    if db_key in _owner_id_repaired:
        return
    from utils.account_context import OWNER_ACCOUNT_ID, ROLE_OWNER

    repaired = conn.execute(
        "UPDATE auth_user SET account_id = ?, role = ?, created_at = COALESCE(created_at, ?) "
        "WHERE username = ? AND account_id IS NULL",
        (
            OWNER_ACCOUNT_ID,
            ROLE_OWNER,
            datetime.now(timezone.utc).isoformat(),
            DEFAULT_ADMIN_USERNAME,
        ),
    ).rowcount
    if repaired:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS auth_user_account_id ON auth_user(account_id)"
        )
    conn.commit()
    _owner_id_repaired.add(db_key)


def _ensure_account_columns(conn: sqlite3.Connection, existing: set) -> None:
    if all(name in existing for name, _decl in _ACCOUNT_COLUMNS):
        _repair_owner_account_id(conn)
        return
    # Both connections can see the columns missing: re-read under the write lock, where a losing
    # ALTER is the other side's, not an error.
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(auth_user)")}
        added = False
        fence_added = False
        for name, decl in _ACCOUNT_COLUMNS:
            if name in existing:
                continue
            try:
                conn.execute(f"ALTER TABLE auth_user ADD COLUMN {name} {decl}")
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
                continue
            added = True
            fence_added = fence_added or name == "account_jwt_secret"
        if added:
            _backfill_account_columns(conn, fence_added)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


_ACCOUNT_API_KEY_COLUMNS = "username, key_prefix, key_hash, name, created_at, expires_at, is_active, is_internal, account_id"


_account_keys_synced: set[str] = set()


def _ensure_account_api_keys(conn: sqlite3.Connection, existing: set) -> None:
    """Pin managed API keys to the immutable ``account_id`` (a namesake inherits none) and mirror
    them into ``account_api_keys``. Idempotent."""
    db_key = str(DB_PATH)
    if db_key in _account_keys_synced and "account_id" in existing:
        return
    if "account_id" not in existing:
        try:
            conn.execute("ALTER TABLE api_keys ADD COLUMN account_id TEXT")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
        conn.execute(
            """UPDATE api_keys SET account_id = (
                   SELECT account_id FROM auth_user
                   WHERE auth_user.username = api_keys.username AND auth_user.role != 'owner'
               ) WHERE account_id IS NULL"""
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_api_keys (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT NOT NULL,
            key_prefix  TEXT NOT NULL,
            key_hash    TEXT NOT NULL UNIQUE,
            name        TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            expires_at  TEXT,
            is_active   INTEGER NOT NULL DEFAULT 1,
            is_internal INTEGER NOT NULL DEFAULT 0,
            account_id  TEXT NOT NULL
        );
        """
    )
    conn.execute(
        f"""INSERT INTO account_api_keys ({_ACCOUNT_API_KEY_COLUMNS})
            SELECT {_ACCOUNT_API_KEY_COLUMNS} FROM api_keys k
            WHERE k.account_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM account_api_keys b WHERE b.key_hash = k.key_hash)"""
    )
    conn.execute(
        f"""INSERT INTO api_keys ({_ACCOUNT_API_KEY_COLUMNS})
            SELECT {_ACCOUNT_API_KEY_COLUMNS} FROM account_api_keys b
            WHERE NOT EXISTS (SELECT 1 FROM api_keys k WHERE k.key_hash = b.key_hash)
              AND EXISTS (SELECT 1 FROM auth_user u WHERE u.account_id = b.account_id)"""
    )
    conn.commit()
    _account_keys_synced.add(db_key)


def _backfill_account_columns(conn: sqlite3.Connection, fence_added: bool) -> None:
    from utils.account_context import OWNER_ACCOUNT_ID, ROLE_OWNER, ROLE_USER
    import uuid

    now = datetime.now(timezone.utc).isoformat()
    for row_id, username, account_id in conn.execute(
        "SELECT id, username, account_id FROM auth_user"
    ).fetchall():
        if account_id:
            continue
        if username == DEFAULT_ADMIN_USERNAME:
            account_id, role = OWNER_ACCOUNT_ID, ROLE_OWNER
        else:
            account_id, role = uuid.uuid4().hex, ROLE_USER
        conn.execute(
            "UPDATE auth_user SET account_id = ?, role = ?, created_at = COALESCE(created_at, ?) WHERE id = ?",
            (account_id, role, now, row_id),
        )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS auth_user_account_id ON auth_user(account_id)")
    if fence_added:
        _fence_managed_credentials(conn)


def _fence_managed_credentials(conn: sqlite3.Connection) -> None:
    # Positional reads: the caller's connection may lack a row factory.
    rows = conn.execute(
        "SELECT id, password_salt, password_hash, jwt_secret FROM auth_user "
        "WHERE role = 'user' AND account_jwt_secret IS NULL"
    ).fetchall()
    for row_id, real_salt, real_hash, real_secret in rows:
        salt, pwd_hash, secret = _legacy_dummies()
        conn.execute(
            """UPDATE auth_user
               SET account_password_salt = ?, account_password_hash = ?, account_jwt_secret = ?,
                   password_salt = ?, password_hash = ?, jwt_secret = ?
               WHERE id = ?""",
            (real_salt, real_hash, real_secret, salt, pwd_hash, secret, row_id),
        )
    managed = "SELECT username FROM auth_user WHERE role = 'user'"
    conn.execute(
        f"UPDATE api_keys SET key_hash = ? || key_hash WHERE username IN ({managed}) AND key_hash NOT LIKE ?",
        (_FENCE_PREFIX, _FENCE_PREFIX + "%"),
    )
    conn.execute(
        f"UPDATE refresh_tokens SET token_hash = ? || token_hash WHERE username IN ({managed}) AND token_hash NOT LIKE ?",
        (_FENCE_PREFIX, _FENCE_PREFIX + "%"),
    )


def get_user_record(username: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            f"""
            SELECT username, {_PASSWORD_SALT_SQL} AS password_salt,
                   {_PASSWORD_HASH_SQL} AS password_hash, {_SECRET_SQL} AS jwt_secret,
                   must_change_password, account_id, role, is_active
            FROM auth_user WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_account(username: str):
    from utils.account_context import AccountContext

    record = get_user_record(username)
    if record is None:
        return None
    return AccountContext(record["account_id"], record["username"], record["role"])


def get_account_by_id(account_id: str):
    from utils.account_context import AccountContext

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT account_id, username, role, is_active FROM auth_user WHERE account_id = ?",
            (account_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or not row["is_active"]:
        return None
    return AccountContext(row["account_id"], row["username"], row["role"])


def count_active_accounts() -> int:
    return account_counts()[0]


def account_counts() -> tuple[int, int]:
    """``(active, managed-of-any-state)``; the second counts deactivated accounts too."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END), 0) AS active,
                   COALESCE(SUM(CASE WHEN role IS NOT NULL AND role != 'owner' THEN 1 ELSE 0 END), 0) AS managed
            FROM auth_user
            """
        ).fetchone()
        return int(row["active"]), int(row["managed"])
    finally:
        conn.close()


def validate_account_username(username: str) -> str:
    username = username.casefold()
    if re.fullmatch(r"[a-z0-9_-]{3,32}", username) is None:
        raise ValueError("Username must contain 3 to 32 letters, digits, underscores or hyphens")
    if username in {"unsloth", "owner", "admin", "root", "system"}:
        raise ValueError("This username is reserved")
    return username


def _public_account(row) -> dict:
    return {
        "account_id": row["account_id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "setup_code_pending": bool(row["setup_code_hash"]),
    }


def list_accounts() -> list[dict]:
    conn = get_connection()
    try:
        return [
            _public_account(row)
            for row in conn.execute("SELECT * FROM auth_user ORDER BY created_at, account_id")
        ]
    finally:
        conn.close()


def _managed_account(conn: sqlite3.Connection, account_id: str):
    row = conn.execute("SELECT * FROM auth_user WHERE account_id = ?", (account_id,)).fetchone()
    if row is None:
        raise LookupError("Account not found")
    if row["role"] == "owner" or account_id == "owner":
        raise ValueError("The installation owner cannot be modified here")
    return row


def _revoke_account_credentials(conn: sqlite3.Connection, row) -> None:
    # These frozen legacy tables key on username; resolve it from the account id under the same
    # write lock as the mutation.
    conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (row["username"],))
    conn.execute("DELETE FROM api_keys WHERE username = ?", (row["username"],))
    conn.execute("DELETE FROM account_api_keys WHERE account_id = ?", (row["account_id"],))
    conn.execute(
        "UPDATE auth_user SET account_jwt_secret = ? WHERE account_id = ?",
        (secrets.token_urlsafe(64), row["account_id"]),
    )


def issue_account_setup_code(
    *, username: Optional[str] = None, account_id: Optional[str] = None
) -> dict:
    from auth.hashing import hash_password
    from auth.policy import account_mutation

    if (username is None) == (account_id is None):
        raise ValueError("Specify either a username or an account id")
    if username is not None:
        username = validate_account_username(username)
    code = secrets.token_urlsafe(32)
    salt, pwd_hash = hash_password(code)
    # Before the transaction: a first-run salt creation opens its own connection.
    code_hash = _hash_setup_code(code)
    now = datetime.now(timezone.utc)
    expires_at = (now + timedelta(minutes = 60)).isoformat()
    with account_mutation():
        conn = get_connection()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if username is not None:
                    account_id = uuid.uuid4().hex
                    legacy_salt, legacy_hash, legacy_secret = _legacy_dummies()
                    conn.execute(
                        """INSERT INTO auth_user
                        (username, account_id, role, is_active, created_at, password_salt,
                         password_hash, jwt_secret, account_password_salt, account_password_hash,
                         account_jwt_secret, must_change_password, setup_code_hash, setup_code_expires_at)
                        VALUES (?, ?, 'user', 1, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                        (
                            username,
                            account_id,
                            now.isoformat(),
                            legacy_salt,
                            legacy_hash,
                            legacy_secret,
                            salt,
                            pwd_hash,
                            secrets.token_urlsafe(64),
                            code_hash,
                            expires_at,
                        ),
                    )
                else:
                    row = _managed_account(conn, account_id)
                    _revoke_account_credentials(conn, row)
                    conn.execute(
                        """UPDATE auth_user SET account_password_salt = ?, account_password_hash = ?,
                           must_change_password = 1, setup_code_hash = ?, setup_code_expires_at = ?
                           WHERE account_id = ?""",
                        (salt, pwd_hash, code_hash, expires_at, account_id),
                    )
                account = _public_account(_managed_account(conn, account_id))
        finally:
            conn.close()
    return {"account": account, "setup_code": code, "setup_code_expires_at": expires_at}


def authenticate_account_login(
    username: str, password: str
) -> Optional[Tuple[str, str, str, bool]]:
    """Managed login. A setup code is consumed once; must_change_password with no pending code
    admits only the issued session's password change."""
    from auth.hashing import equalize_login_work, verify_password

    def miss():
        # A miss without a hash to check costs what a wrong password costs.
        equalize_login_work(password)
        return None

    conn = get_connection()
    try:
        row = conn.execute(
            f"""SELECT account_id, is_active, must_change_password, setup_code_hash,
                       {_PASSWORD_SALT_SQL} AS password_salt, {_PASSWORD_HASH_SQL} AS password_hash,
                       {_SECRET_SQL} AS jwt_secret
                FROM auth_user WHERE username = ?""",
            (username,),
        ).fetchone()
        if row is None or not row["is_active"]:
            return miss()
        if row["must_change_password"]:
            # The setup-code hash is the PBKDF2 round a miss would otherwise spend.
            if not row["setup_code_hash"] or not hmac.compare_digest(
                row["setup_code_hash"], _hash_setup_code(password)
            ):
                return None
            # Compare-and-swap on expiry, activity and generation: no code is spent twice.
            with conn:
                cursor = conn.execute(
                    f"""UPDATE auth_user SET setup_code_hash = NULL, setup_code_expires_at = NULL
                       WHERE account_id = ? AND setup_code_hash = ? AND {_SECRET_SQL} = ?
                       AND is_active = 1 AND setup_code_expires_at > ?""",
                    (
                        row["account_id"],
                        row["setup_code_hash"],
                        row["jwt_secret"],
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                if cursor.rowcount != 1:
                    return None
        elif not verify_password(password, row["password_salt"], row["password_hash"]):
            return None
        return (
            row["password_salt"],
            row["password_hash"],
            row["jwt_secret"],
            bool(row["must_change_password"]),
        )
    finally:
        conn.close()


def update_account_password(
    username: str, new_password: str, *, expect_password_hash: str, expect_secret: str
) -> Optional[str]:
    from auth.hashing import hash_password

    salt, pwd_hash = hash_password(new_password)
    secret = secrets.token_urlsafe(64)
    conn = get_connection()
    try:
        with conn:
            cursor = conn.execute(
                f"""UPDATE auth_user SET account_password_salt = ?, account_password_hash = ?,
                   account_jwt_secret = ?, must_change_password = 0,
                   setup_code_hash = NULL, setup_code_expires_at = NULL
                   WHERE username = ? AND role = 'user' AND is_active = 1
                   AND {_PASSWORD_HASH_SQL} = ? AND {_SECRET_SQL} = ?""",
                (salt, pwd_hash, secret, username, expect_password_hash, expect_secret),
            )
            if cursor.rowcount != 1:
                return None
            conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        return secret
    finally:
        conn.close()


def set_account_active(account_id: str, is_active: bool) -> dict:
    from auth.policy import account_mutation
    with account_mutation():
        conn = get_connection()
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = _managed_account(conn, account_id)
                if not is_active:
                    _revoke_account_credentials(conn, row)
                conn.execute(
                    "UPDATE auth_user SET is_active = ? WHERE account_id = ?",
                    (int(is_active), account_id),
                )
                result = _public_account(_managed_account(conn, account_id))
        finally:
            conn.close()
    return result


def delete_account(account_id: str, retire) -> None:
    """Revoke, retire files, then drop the identity under a write lock. ``retire()`` must not write
    auth.db under that lock; a failed retire leaves the account disabled for a retry."""
    from auth.policy import account_mutation
    from utils.account_context import AccountContext

    set_account_active(account_id, False)
    with account_mutation():
        conn = get_connection()
        restore_roots = None
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = _managed_account(conn, account_id)
                _revoke_account_credentials(conn, row)
                restore_roots = retire(
                    AccountContext(row["account_id"], row["username"], row["role"])
                )
                conn.execute("DELETE FROM auth_user WHERE account_id = ?", (account_id,))
        except Exception:
            # The identity survives the rollback, so the roots must come back too; a failed
            # restore is the error to report, the account staying disabled either way.
            try:
                if restore_roots is not None:
                    restore_roots()
            finally:
                # An owner may reactivate between revocation and this lock; stay disabled anyway.
                with contextlib.suppress(sqlite3.Error):
                    set_account_active(account_id, False)
            raise
        finally:
            conn.close()


def _get_or_create_api_key_pbkdf2_salt() -> bytes:
    """Return the persistent API-key PBKDF2 salt, generating it once if missing. Hex-encoded 32-byte
    random value in ``app_secrets``, regenerated only when the row is missing."""
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
            new_value = secrets.token_hex(32)
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


# Identity-challenge secret lives in auth.db so a port squatter cannot forge a proof; separate from the JWT secret.
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


# Dedicated AES-256 key: lives in auth.db so copying studio.db alone does not expose provider/HF
# tokens, and survives password resets.
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
    """HMAC-SHA256 proof that the caller holds this install's identity secret, bound to the loopback
    address and port the connection landed on. A proof relayed from an Unsloth on a different
    address or port was computed for that other endpoint and will not match the one the client
    dialed."""
    try:
        host = ipaddress.ip_address(host).compressed
    except ValueError:
        host = (host or "").lower()
    msg = b"|".join([nonce, host.encode(), str(int(port)).encode()])
    return hmac.new(get_or_create_identity_secret(), msg, hashlib.sha256).hexdigest()


# Dedicated secret so rotating it revokes every shared preview link without touching logins.
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
    """PBKDF2-HMAC-SHA256 an API key with a persistent server-side salt. For API-key storage ONLY, not
    refresh tokens: the slow KDF is only to appease CodeQL's ``py/weak-sensitive-data-hashing``
    query, since API keys are random 128-bit tokens. The salt lives in ``app_secrets`` so dumping
    ``api_keys`` alone cannot derive hashes."""
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


def _hash_setup_code(code: str) -> str:
    """A setup code is typed where a password goes, so it is stored like an API key."""
    return _pbkdf2_api_key(code)


# Keyed by a salted HMAC, not the key; revocation/expiry are still enforced by the SQLite read, and
# only known keys are cached.
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
    with _api_key_hash_cache_lock:
        _api_key_hash_cache.clear()
        _api_key_internal_cache.clear()


def is_initialized() -> bool:
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
    """Create the initial admin user in the database. Raises sqlite3.IntegrityError if username already
    exists."""
    from .hashing import hash_password
    from utils.account_context import OWNER_ACCOUNT_ID, ROLE_OWNER, ROLE_USER
    import uuid

    salt, pwd_hash = hash_password(password)
    if username == DEFAULT_ADMIN_USERNAME:
        account_id, role = OWNER_ACCOUNT_ID, ROLE_OWNER
        legacy = (salt, pwd_hash, jwt_secret)
        fenced = (None, None, None)
    else:
        account_id, role = uuid.uuid4().hex, ROLE_USER
        legacy = _legacy_dummies()
        fenced = (salt, pwd_hash, jwt_secret)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO auth_user (
                username,
                password_salt,
                password_hash,
                jwt_secret,
                account_password_salt,
                account_password_hash,
                account_jwt_secret,
                must_change_password,
                account_id,
                role,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                *legacy,
                *fenced,
                int(must_change_password),
                account_id,
                role,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    from auth.policy import invalidate_account_cache

    invalidate_account_cache()


def delete_user(username: str) -> None:
    """Delete a user from the database, for rollback when user creation fails partway through
    bootstrap."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()
    from auth.policy import invalidate_account_cache

    invalidate_account_cache()


def get_user_and_secret(username: str) -> Optional[Tuple[str, str, str, bool]]:
    """Get user's password salt, hash, and JWT secret. Returns (password_salt, password_hash,
    jwt_secret, must_change_password) or None."""
    conn = get_connection()
    try:
        cur = conn.execute(
            f"""
            SELECT {_PASSWORD_SALT_SQL} AS password_salt, {_PASSWORD_HASH_SQL} AS password_hash,
                   {_SECRET_SQL} AS jwt_secret, must_change_password
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
    conn = get_connection()
    try:
        cur = conn.execute(
            f"SELECT {_SECRET_SQL} AS jwt_secret FROM auth_user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return row["jwt_secret"] if row else None
    finally:
        conn.close()


def requires_password_change(username: str) -> bool:
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
    """Load the JWT secret from the database. Raises RuntimeError if no auth user has been created yet."""
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
    """Seed the default admin account on first startup, using a randomly generated diceware passphrase
    as the bootstrap password. Returns True when it was created in this call."""
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
    expect_password_hash: Optional[str] = None,
    preserve_desktop_secret: bool = False,
) -> Optional[str]:
    """Update password, clear first-login requirement, rotate JWT secret. Returns the new JWT secret,
    or None when nothing was updated. Callers that mint tokens must sign with the returned secret:
    re-reading it would pick up a reset that landed between this commit and the mint.
    ``revoke_refresh_tokens`` deletes the user's refresh tokens in the SAME transaction, since a
    separate delete could fail after the password commit and leave a pre-change token able to mint
    access tokens. ``expect_password_hash`` makes the write conditional on the credential the caller
    verified still being current, returning False when it moved underneath.
    ``preserve_desktop_secret`` keeps the local desktop credential valid, for a caller that already
    authenticated as the desktop app and would otherwise break its own auto-auth."""
    from .hashing import hash_password

    salt, pwd_hash = hash_password(new_password)
    jwt_secret = secrets.token_urlsafe(64)
    columns = (
        "password_salt = ?, password_hash = ?, jwt_secret = ?"
        if _is_owner_name(username)
        else "account_password_salt = ?, account_password_hash = ?, account_jwt_secret = ?"
    )
    conn = get_connection()
    try:
        if expect_password_hash is None:
            cursor = conn.execute(
                f"""
                UPDATE auth_user
                SET {columns}, must_change_password = 0
                WHERE username = ?
                """,
                (salt, pwd_hash, jwt_secret, username),
            )
        else:
            cursor = conn.execute(
                f"""
                UPDATE auth_user
                SET {columns}, must_change_password = 0
                WHERE username = ? AND {_PASSWORD_HASH_SQL} = ?
                """,
                (salt, pwd_hash, jwt_secret, username, expect_password_hash),
            )
        if revoke_refresh_tokens and cursor.rowcount > 0:
            conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.commit()
        if cursor.rowcount > 0:
            clear_bootstrap_password()
            if not preserve_desktop_secret:
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
    token_hash = _fenced_hash(_hash_token(token), username)
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
    """Atomically validate-and-delete a refresh token for single-use rotation. DELETE RETURNING fuses
    validate and delete into one statement so two concurrent refresh requests cannot both consume
    the same token. Returns ``(username, is_desktop, jwt_secret)``; the caller must mint the
    replacement tokens against that secret so a rotation landing mid-refresh cannot issue a
    post-rotation session from a pre-rotation token."""
    token_hash = _hash_token(token)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        # One transaction with the delete: an unstamped legacy row has no generation, so a later read could
        # hand a reset's new secret to an older token.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (now,),
        )
        cur = conn.execute(
            f"""
            DELETE FROM refresh_tokens
            WHERE token_hash {_FENCED_HASH_SQL} AND expires_at >= ?
            RETURNING username, is_desktop, secret_gen
            """,
            (*_hash_candidates(token_hash), now),
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
    """Verify a refresh token and return the username plus desktop marker, or None when invalid or
    expired. The token is NOT consumed: it stays valid until it expires."""
    token_hash = _hash_token(token)
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?",
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()

        cur = conn.execute(
            f"""
            SELECT id, username, expires_at, is_desktop, secret_gen FROM refresh_tokens
            WHERE token_hash {_FENCED_HASH_SQL}
            """,
            _hash_candidates(token_hash),
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

        expires_at = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            conn.execute("DELETE FROM refresh_tokens WHERE id = ?", (row["id"],))
            conn.commit()
            return None

        return row["username"], bool(row["is_desktop"])
    finally:
        conn.close()


def revoke_user_refresh_tokens(username: str, *, account_id: Optional[str] = None) -> None:
    """Revoke a user's refresh tokens. ``account_id`` pins the username-keyed table to the
    immutable identity, so a recreated namesake keeps its sessions."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            if account_id is not None:
                row = conn.execute(
                    "SELECT username FROM auth_user WHERE account_id = ?", (account_id,)
                ).fetchone()
                if row is None or row["username"] != username:
                    return
            conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
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
    """Validate the desktop secret and return ``(username, jwt_secret)``. Both reads share one
    transaction so the returned secret is the credential version the desktop secret was checked
    against; a reset landing mid-request then invalidates the tokens minted from it rather than
    blessing them."""
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
    verified = validate_desktop_secret_with_credential(raw_secret)
    return verified[0] if verified else None


def clear_desktop_secret() -> None:
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM app_secrets WHERE key IN (?, ?)",
            (_DESKTOP_SECRET_HASH_KEY, _DESKTOP_SECRET_CREATED_AT_KEY),
        )
        conn.commit()
    finally:
        conn.close()


API_KEY_PREFIX = "sk-unsloth-"

# The name is the only thing distinguishing internal keys by authority: Deep Research keys must
# reach the saved connection, data-recipe keys only local /v1.
DEEP_RESEARCH_WORKFLOW_KEY_NAME = "deep-research workflow"


def create_api_key(
    username: str,
    name: str,
    expires_at: Optional[str] = None,
    internal: bool = False,
    expect_gen: Optional[str] = None,
    account_id: Optional[str] = None,
) -> Tuple[str, dict]:
    """Create a new API key for *username*. Returns ``(raw_key, row_dict)`` where *raw_key* is shown to
    the user exactly once; the database only stores the PBKDF2 hash. Pass ``internal=True`` for keys
    minted by workflows that should not appear in user-facing listings. ``expect_gen`` ties the
    insert to the credential generation the request authenticated under, so a session revoked by a
    concurrent password reset cannot mint a key that outlives it. Raises ``CredentialRotated`` if it
    moved."""
    raw_key = API_KEY_PREFIX + secrets.token_hex(16)
    key_hash = _fenced_hash(_pbkdf2_api_key(raw_key), username)
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
        if account_id is None:
            account_id = _managed_account_id(conn, username)
        conn.execute(
            """
            INSERT INTO api_keys (username, key_prefix, key_hash, name, created_at, expires_at, is_internal, account_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                key_prefix,
                key_hash,
                name,
                now,
                expires_at,
                1 if internal else 0,
                account_id,
            ),
        )
        if account_id is not None:
            conn.execute(
                f"""INSERT INTO account_api_keys ({_ACCOUNT_API_KEY_COLUMNS})
                    SELECT {_ACCOUNT_API_KEY_COLUMNS} FROM api_keys WHERE key_hash = ?""",
                (key_hash,),
            )
        conn.commit()
        cur = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,))
        row = cur.fetchone()
        return raw_key, dict(row)
    finally:
        conn.close()


def list_api_keys(
    username: str,
    include_internal: bool = False,
    account_id: Optional[str] = None,
) -> list:
    conn = get_connection()
    try:
        scope, params = _key_scope(username, account_id)
        internal = "" if include_internal else " AND is_internal = 0"
        cur = conn.execute(
            f"""
            SELECT id, username, key_prefix, name, created_at, last_used_at,
                   expires_at, is_active, is_internal
            FROM api_keys
            WHERE {scope}{internal}
            ORDER BY created_at DESC
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _managed_account_id(conn: sqlite3.Connection, username: str) -> Optional[str]:
    row = conn.execute(
        "SELECT account_id, role FROM auth_user WHERE username = ?", (username,)
    ).fetchone()
    if row is None or row["role"] in (None, "owner"):
        return None
    return row["account_id"]


def _revoke_key_copy(conn: sqlite3.Connection, key_id: int) -> None:
    conn.execute(
        "UPDATE account_api_keys SET is_active = 0 "
        "WHERE key_hash = (SELECT key_hash FROM api_keys WHERE id = ?)",
        (key_id,),
    )


def _key_scope(username: str, account_id: Optional[str]) -> Tuple[str, tuple]:
    if account_id is None:
        return "username = ?", (username,)
    return "username = ? AND account_id = ?", (username, account_id)


def revoke_api_key(
    username: str,
    key_id: int,
    account_id: Optional[str] = None,
) -> bool:
    """Soft-delete an API key.  Returns True if a matching row was found."""
    conn = get_connection()
    try:
        scope, params = _key_scope(username, account_id)
        cursor = conn.execute(
            f"UPDATE api_keys SET is_active = 0 WHERE id = ? AND {scope}",
            (key_id, *params),
        )
        if cursor.rowcount:
            _revoke_key_copy(conn, key_id)
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def revoke_internal_api_key(key_id: int) -> bool:
    """Revoke an internal workflow-minted key without requiring a username. Used by the recipe runner
    to retire its key once the job terminates, shrinking the window a leaked key could be abused."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND is_internal = 1",
            (key_id,),
        )
        if cursor.rowcount:
            _revoke_key_copy(conn, key_id)
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def is_internal_api_key(raw_key: str) -> bool:
    """Whether *raw_key* is a workflow-minted internal key rather than a user's own. Lets
    request-scoped code tell Unsloth's own background work from a third party using Unsloth as an
    API server. Memoized, because this runs on the event loop for every API-key request and a key's
    origin is fixed when it is minted."""
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
            f"SELECT is_internal FROM api_keys WHERE key_hash {_FENCED_HASH_SQL}",
            _hash_candidates(key_hash),
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
    ``is_internal_api_key`` answers "is this Unsloth's own key", which is far too coarse for
    authorization: a data-recipe key runs inside a recipe the user authored, so treating it as equal
    to the Deep Research hop would let that recipe spend any saved cloud credential. The name is
    fixed when the key is minted and is the only durable thing separating the two. Deliberately not
    memoized: this is read on the external-provider path once per request, and a stale answer would
    be a stale authorization. The PBKDF2 derivation comes from the shared hash cache when it is
    warm."""
    if not raw_key.startswith(API_KEY_PREFIX):
        return None
    cache_id = _api_key_cache_id(raw_key)
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT name FROM api_keys WHERE key_hash {_FENCED_HASH_SQL} AND is_internal = 1 AND is_active = 1",
            _hash_candidates(key_hash),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    name = row["name"]
    return name if isinstance(name, str) else None


def validate_api_key(raw_key: str) -> Optional[str]:
    verified = validate_api_key_with_credential(raw_key)
    return verified[0] if verified else None


def validate_api_key_with_credential(
    raw_key: str, *, touch: bool = True
) -> Optional[Tuple[str, str]]:
    verified = validate_api_key_account(raw_key, touch = touch)
    return (verified[0]["username"], verified[1]) if verified else None


def validate_api_key_account(raw_key: str, *, touch: bool = True) -> Optional[Tuple[dict, str]]:
    """Validate *raw_key* -> ``(account record, jwt_secret)``, or ``None``. The record comes from
    the matching statement, so a namesake cannot bind; ``touch=False`` skips the last-used write."""
    cache_id = _api_key_cache_id(raw_key)
    cached_hash = _api_key_hash_cache.get(cache_id)
    key_hash = cached_hash if cached_hash is not None else _pbkdf2_api_key(raw_key)
    conn = get_connection()
    try:
        if touch:
            conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            f"""
            SELECT k.id, k.username, k.is_active, k.expires_at,
                   u.account_id, u.role, u.is_active AS account_active,
                   COALESCE(u.account_jwt_secret, u.jwt_secret) AS jwt_secret
            FROM api_keys k JOIN auth_user u ON u.username = k.username
              AND (k.account_id = u.account_id OR (k.account_id IS NULL AND u.role = 'owner'))
            WHERE k.key_hash {_FENCED_HASH_SQL}
            """,
            _hash_candidates(key_hash),
        )
        row = cur.fetchone()
        if row is None:
            return None
        # Real key: memoize so later requests skip the KDF. Bounded; clear on overflow.
        if cached_hash is None:
            with _api_key_hash_cache_lock:
                if len(_api_key_hash_cache) >= _API_KEY_HASH_CACHE_MAX:
                    _api_key_hash_cache.clear()
                    # Clear both caches together: the origin cache is sized against the hash cache and would otherwise
                    # grow past its bound.
                    _api_key_internal_cache.clear()
                _api_key_hash_cache[cache_id] = key_hash
        if not row["is_active"]:
            return None
        if row["expires_at"] is not None:
            expires = datetime.fromisoformat(row["expires_at"])
            if datetime.now(timezone.utc) > expires:
                return None
        if not row["account_active"]:
            return None
        secret = row["jwt_secret"]
        if secret is None:
            return None
        if touch:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), row["id"]),
            )
            conn.commit()
        record = {
            "account_id": row["account_id"],
            "username": row["username"],
            "role": row["role"],
            "is_active": int(row["account_active"]),
        }
        return record, secret
    finally:
        conn.rollback()
        conn.close()
