# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Encrypted installation-wide credential persistence in ``studio.db``.

Studio is a single-user local application. Credentials belong to the installation,
not to an authenticated subject. The AES key lives separately in auth.db and the
credential kind/scope are authenticated so ciphertext rows cannot be swapped.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from auth.storage import get_or_create_credential_encryption_key
from utils.paths import ensure_dir, studio_db_path

logger = logging.getLogger(__name__)

HF_TOKEN_KIND = "hf_token"
HF_TOKEN_SCOPE = "default"
PROVIDER_API_KEY_KIND = "provider_api_key"

OPENAI_CODEX_OAUTH_KIND = "openai_codex_oauth"

OPENAI_CODEX_OAUTH_FLOW_KIND = "openai_codex_oauth_flow"
_FORMAT_VERSION = 1
_NONCE_BYTES = 12

_schema_lock = threading.Lock()
_schema_ready = False


def _associated_data(credential_kind: str, scope_id: str) -> bytes:
    return f"unsloth-studio-credential\0{credential_kind}\0{scope_id}".encode("utf-8")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS credential_secrets (
            credential_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            nonce BLOB NOT NULL,
            ciphertext BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (credential_kind, scope_id)
        ) WITHOUT ROWID
        """
    )
    conn.commit()


def get_connection() -> sqlite3.Connection:
    global _schema_ready
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), timeout = 5.0)
    conn.row_factory = sqlite3.Row
    try:
        os.chmod(db_path.parent, 0o700)
        os.chmod(db_path, 0o600)
    except OSError:
        pass
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready = True
                except Exception:
                    conn.close()
                    raise
    return conn


def ensure_schema() -> None:
    """Ensure the credential table exists before a shared transaction starts."""
    conn = get_connection()
    conn.close()


def _encrypted_secret(
    credential_kind: str, scope_id: str, plaintext: str
) -> tuple[bytes, bytes, str]:
    if not credential_kind or not scope_id:
        raise ValueError("Credential kind and scope are required")
    if not plaintext:
        raise ValueError("Credential value cannot be empty")
    key = get_or_create_credential_encryption_key()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce,
        plaintext.encode("utf-8"),
        _associated_data(credential_kind, scope_id),
    )
    return nonce, ciphertext, datetime.now(timezone.utc).isoformat()


def upsert_secret(
    credential_kind: str,
    scope_id: str,
    plaintext: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    """Encrypt and atomically insert or replace one installation credential."""
    nonce, ciphertext, now = _encrypted_secret(credential_kind, scope_id, plaintext)
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        conn.execute(
            """
            INSERT INTO credential_secrets (
                credential_kind, scope_id, format_version,
                nonce, ciphertext, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(credential_kind, scope_id) DO UPDATE SET
                format_version = excluded.format_version,
                nonce = excluded.nonce,
                ciphertext = excluded.ciphertext,
                updated_at = excluded.updated_at
            """,
            (credential_kind, scope_id, _FORMAT_VERSION, nonce, ciphertext, now, now),
        )
        if owns_connection:
            conn.commit()
    finally:
        if owns_connection:
            conn.close()


def insert_secret_if_absent(credential_kind: str, scope_id: str, plaintext: str) -> bool:
    """Atomically insert a migration credential without replacing an existing value."""
    nonce, ciphertext, now = _encrypted_secret(credential_kind, scope_id, plaintext)
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO credential_secrets (
                credential_kind, scope_id, format_version,
                nonce, ciphertext, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (credential_kind, scope_id, _FORMAT_VERSION, nonce, ciphertext, now, now),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_secret(credential_kind: str, scope_id: str) -> Optional[str]:
    """Return a decrypted credential, or ``None`` if absent or unreadable."""
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT format_version, nonce, ciphertext
            FROM credential_secrets
            WHERE credential_kind = ? AND scope_id = ?
            """,
            (credential_kind, scope_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["format_version"] != _FORMAT_VERSION:
        return None
    try:
        plaintext = AESGCM(get_or_create_credential_encryption_key()).decrypt(
            bytes(row["nonce"]),
            bytes(row["ciphertext"]),
            _associated_data(credential_kind, scope_id),
        )
        return plaintext.decode("utf-8")
    except Exception:
        logger.warning(
            "Saved credential is unreadable; re-entry is required (kind=%s)",
            credential_kind,
        )
        return None


def has_secret(credential_kind: str, scope_id: str) -> bool:
    return get_secret(credential_kind, scope_id) is not None


def delete_secret(
    credential_kind: str,
    scope_id: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> bool:
    """Idempotently delete one credential; return whether a row existed."""
    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM credential_secrets WHERE credential_kind = ? AND scope_id = ?",
            (credential_kind, scope_id),
        )
        if owns_connection:
            conn.commit()
        return cursor.rowcount > 0
    finally:
        if owns_connection:
            conn.close()


def get_hf_token() -> Optional[str]:
    return get_secret(HF_TOKEN_KIND, HF_TOKEN_SCOPE)


def save_hf_token(token: str) -> None:
    upsert_secret(HF_TOKEN_KIND, HF_TOKEN_SCOPE, token)


def save_hf_token_if_absent(token: str) -> bool:
    return insert_secret_if_absent(HF_TOKEN_KIND, HF_TOKEN_SCOPE, token)


def delete_hf_token() -> bool:
    return delete_secret(HF_TOKEN_KIND, HF_TOKEN_SCOPE)


def get_provider_api_key(provider_id: str) -> Optional[str]:
    return get_secret(PROVIDER_API_KEY_KIND, provider_id)


def save_provider_api_key(
    provider_id: str,
    api_key: str,
    *,
    connection: sqlite3.Connection | None = None,
) -> None:
    upsert_secret(PROVIDER_API_KEY_KIND, provider_id, api_key, connection = connection)


def save_provider_api_key_if_absent(provider_id: str, api_key: str) -> bool:
    return insert_secret_if_absent(PROVIDER_API_KEY_KIND, provider_id, api_key)


def delete_provider_api_key(
    provider_id: str, *, connection: sqlite3.Connection | None = None
) -> bool:
    return delete_secret(PROVIDER_API_KEY_KIND, provider_id, connection = connection)


def resolve_provider_api_key(provider_id: Optional[str], encrypted_api_key: Optional[str]) -> str:
    """Resolve an explicit request key first, then the installation's saved key."""
    if encrypted_api_key:
        from core.inference.key_exchange import decrypt_api_key
        return decrypt_api_key(encrypted_api_key)
    if provider_id:
        return get_provider_api_key(provider_id) or ""
    return ""
