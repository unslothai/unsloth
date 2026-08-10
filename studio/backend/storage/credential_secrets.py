# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Encrypted, owner-scoped credential persistence in ``studio.db``.

This is deliberately a small application-layer envelope rather than a secrets
service. The AES key lives separately in auth.db; authenticated metadata binds
each ciphertext to its owner, kind, and scope so rows cannot be swapped.
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
_FORMAT_VERSION = 1
_NONCE_BYTES = 12

_schema_lock = threading.Lock()
_schema_ready = False


def _associated_data(owner_subject: str, credential_kind: str, scope_id: str) -> bytes:
    return f"unsloth-studio-credential\0{owner_subject}\0{credential_kind}\0{scope_id}".encode(
        "utf-8"
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS credential_secrets (
            owner_subject TEXT NOT NULL,
            credential_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            nonce BLOB NOT NULL,
            ciphertext BLOB NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (owner_subject, credential_kind, scope_id)
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


def upsert_secret(
    owner_subject: str,
    credential_kind: str,
    scope_id: str,
    plaintext: str,
) -> None:
    """Encrypt and atomically insert or replace one credential."""
    if not owner_subject or not credential_kind or not scope_id:
        raise ValueError("Credential owner, kind, and scope are required")
    if not plaintext:
        raise ValueError("Credential value cannot be empty")

    key = get_or_create_credential_encryption_key()
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(
        nonce,
        plaintext.encode("utf-8"),
        _associated_data(owner_subject, credential_kind, scope_id),
    )
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO credential_secrets (
                owner_subject, credential_kind, scope_id, format_version,
                nonce, ciphertext, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_subject, credential_kind, scope_id) DO UPDATE SET
                format_version = excluded.format_version,
                nonce = excluded.nonce,
                ciphertext = excluded.ciphertext,
                updated_at = excluded.updated_at
            """,
            (
                owner_subject,
                credential_kind,
                scope_id,
                _FORMAT_VERSION,
                nonce,
                ciphertext,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_secret(
    owner_subject: str,
    credential_kind: str,
    scope_id: str,
) -> Optional[str]:
    """Return a decrypted credential, or ``None`` if absent or unreadable.

    Corrupt rows, unknown envelope versions, and a replaced auth.db key fail
    closed for this credential only and never prevent Studio startup.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT format_version, nonce, ciphertext
            FROM credential_secrets
            WHERE owner_subject = ? AND credential_kind = ? AND scope_id = ?
            """,
            (owner_subject, credential_kind, scope_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["format_version"] != _FORMAT_VERSION:
        return None
    try:
        plaintext = AESGCM(get_or_create_credential_encryption_key()).decrypt(
            bytes(row["nonce"]),
            bytes(row["ciphertext"]),
            _associated_data(owner_subject, credential_kind, scope_id),
        )
        return plaintext.decode("utf-8")
    except Exception:
        logger.warning(
            "Saved credential is unreadable; re-entry is required (kind=%s)",
            credential_kind,
        )
        return None


def has_secret(owner_subject: str, credential_kind: str, scope_id: str) -> bool:
    """True only when a saved credential is present and decryptable."""
    return get_secret(owner_subject, credential_kind, scope_id) is not None


def delete_secret(owner_subject: str, credential_kind: str, scope_id: str) -> bool:
    """Idempotently delete one credential; return whether a row existed."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            DELETE FROM credential_secrets
            WHERE owner_subject = ? AND credential_kind = ? AND scope_id = ?
            """,
            (owner_subject, credential_kind, scope_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_hf_token(owner_subject: str) -> Optional[str]:
    return get_secret(owner_subject, HF_TOKEN_KIND, HF_TOKEN_SCOPE)


def save_hf_token(owner_subject: str, token: str) -> None:
    upsert_secret(owner_subject, HF_TOKEN_KIND, HF_TOKEN_SCOPE, token)


def delete_hf_token(owner_subject: str) -> bool:
    return delete_secret(owner_subject, HF_TOKEN_KIND, HF_TOKEN_SCOPE)


def get_provider_api_key(owner_subject: str, provider_id: str) -> Optional[str]:
    return get_secret(owner_subject, PROVIDER_API_KEY_KIND, provider_id)


def save_provider_api_key(owner_subject: str, provider_id: str, api_key: str) -> None:
    upsert_secret(owner_subject, PROVIDER_API_KEY_KIND, provider_id, api_key)


def delete_provider_api_key(owner_subject: str, provider_id: str) -> bool:
    return delete_secret(owner_subject, PROVIDER_API_KEY_KIND, provider_id)


def resolve_provider_api_key(
    owner_subject: str,
    provider_id: Optional[str],
    encrypted_api_key: Optional[str],
) -> str:
    """Resolve explicit legacy/request key first, then an owner-scoped saved key."""
    if encrypted_api_key:
        from core.inference.key_exchange import decrypt_api_key

        return decrypt_api_key(encrypted_api_key)
    if provider_id:
        return get_provider_api_key(owner_subject, provider_id) or ""
    return ""
