# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic external OIDC identity to internal account mapping."""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import policy, storage


class OIDCIdentityConflict(RuntimeError):
    """The identity was concurrently mapped and should be resolved again."""


def _account_record(row) -> dict:
    return {
        "account_id": row["account_id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
    }


def get_external_identity(issuer: str, subject: str) -> Optional[dict]:
    """Return the exact mapped account, never an email or username match."""

    conn = storage.get_connection()
    try:
        row = conn.execute(
            """SELECT u.account_id, u.username, u.role, u.is_active
               FROM external_identities e
               JOIN auth_user u ON u.account_id = e.account_id
               WHERE e.provider_type = 'oidc' AND e.issuer = ? AND e.subject = ?""",
            (issuer, subject),
        ).fetchone()
        return _account_record(row) if row is not None else None
    finally:
        conn.close()


def _username_base(preferred_username: object, email: object, subject: str) -> str:
    candidate = preferred_username if isinstance(preferred_username, str) else ""
    if not candidate and isinstance(email, str):
        candidate = email.partition("@")[0]
    candidate = re.sub(r"[^a-z0-9_-]+", "-", candidate.casefold()).strip("-_")
    if len(candidate) < 3 or candidate in {"unsloth", "owner", "admin", "root", "system"}:
        candidate = "oidc-user"
    return candidate[:32]


def _available_username(conn: sqlite3.Connection, base: str, issuer: str, subject: str) -> str:
    if conn.execute("SELECT 1 FROM auth_user WHERE username = ?", (base,)).fetchone() is None:
        return base
    suffix = hashlib.sha256(f"{issuer}\0{subject}".encode("utf-8")).hexdigest()[:8]
    stem = base[: 32 - len(suffix) - 1]
    candidate = f"{stem}-{suffix}"
    counter = 2
    while conn.execute("SELECT 1 FROM auth_user WHERE username = ?", (candidate,)).fetchone():
        tail = f"-{suffix}-{counter}"
        candidate = f"{base[:32 - len(tail)]}{tail}"
        counter += 1
    return candidate


def create_external_account(
    *,
    issuer: str,
    subject: str,
    preferred_username: object = None,
    email: object = None,
    display_name: object = None,
) -> dict:
    """Atomically create a non-owner account and its issuer+subject mapping.

    A username collision always creates a distinct generated name. It never links to a local
    account or to another external identity merely because mutable profile claims are equal.
    """

    email_value = email.strip() if isinstance(email, str) and email.strip() else None
    display_value = (
        display_name.strip() if isinstance(display_name, str) and display_name.strip() else None
    )
    preferred_value = (
        preferred_username.strip()
        if isinstance(preferred_username, str) and preferred_username.strip()
        else None
    )
    now = datetime.now(timezone.utc).isoformat()
    with policy.account_mutation():
        conn = storage.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """SELECT u.account_id, u.username, u.role, u.is_active
                   FROM external_identities e JOIN auth_user u ON u.account_id = e.account_id
                   WHERE e.provider_type = 'oidc' AND e.issuer = ? AND e.subject = ?""",
                (issuer, subject),
            ).fetchone()
            if existing is not None:
                conn.rollback()
                return _account_record(existing)

            username = _available_username(
                conn, _username_base(preferred_value, email_value, subject), issuer, subject
            )
            account_id = uuid.uuid4().hex
            legacy_salt, legacy_hash, legacy_secret = storage.passwordless_account_credentials()
            account_salt, account_hash, _unused_secret = storage.passwordless_account_credentials()
            conn.execute(
                """INSERT INTO auth_user
                   (username, account_id, role, is_active, created_at, password_salt,
                    password_hash, jwt_secret, account_password_salt, account_password_hash,
                    account_jwt_secret, must_change_password)
                   VALUES (?, ?, 'user', 1, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    username,
                    account_id,
                    now,
                    legacy_salt,
                    legacy_hash,
                    legacy_secret,
                    account_salt,
                    account_hash,
                    secrets.token_urlsafe(64),
                ),
            )
            conn.execute(
                """INSERT INTO external_identities
                   (account_id, provider_type, issuer, subject, email, username,
                    display_name, created_at, last_login_at)
                   VALUES (?, 'oidc', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    account_id,
                    issuer,
                    subject,
                    email_value,
                    preferred_value,
                    display_value,
                    now,
                    now,
                ),
            )
            conn.commit()
            return {
                "account_id": account_id,
                "username": username,
                "role": "user",
                "is_active": True,
            }
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            # The only expected race is the unique issuer+subject key. Avoid reporting a
            # mutable username collision as an identity match.
            mapped = get_external_identity(issuer, subject)
            if mapped is not None:
                return mapped
            raise OIDCIdentityConflict("Could not create the external identity mapping") from exc
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def record_external_login(
    *,
    issuer: str,
    subject: str,
    email: object = None,
    preferred_username: object = None,
) -> Optional[dict]:
    """Update mutable profile metadata and return the immutable mapped account."""

    email_value = email.strip() if isinstance(email, str) and email.strip() else None
    username_value = (
        preferred_username.strip()
        if isinstance(preferred_username, str) and preferred_username.strip()
        else None
    )
    conn = storage.get_connection()
    try:
        with conn:
            row = conn.execute(
                """SELECT u.account_id, u.username, u.role, u.is_active
                   FROM external_identities e JOIN auth_user u ON u.account_id = e.account_id
                   WHERE e.provider_type = 'oidc' AND e.issuer = ? AND e.subject = ?""",
                (issuer, subject),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """UPDATE external_identities SET email = ?, username = ?, last_login_at = ?
                   WHERE provider_type = 'oidc' AND issuer = ? AND subject = ?""",
                (
                    email_value,
                    username_value,
                    datetime.now(timezone.utc).isoformat(),
                    issuer,
                    subject,
                ),
            )
            return _account_record(row)
    finally:
        conn.close()
