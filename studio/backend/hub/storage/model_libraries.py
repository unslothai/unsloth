# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistence for user-registered model library locations.

A model library is a *cache home* directory holding ``hub/`` and ``xet/`` subtrees
exactly like the active Hugging Face cache. The default library is always the active
HF cache (:mod:`utils.hf_cache_settings`), so this table stores only the additional
libraries the user registered; ``is_default`` is derived against the active cache
home and never persisted.

Libraries are owner-scoped on purpose: they name physical drives the machine owns,
and the HF cache setting plus every download worker resolve against the owner's
scope. Every public call here therefore runs under :func:`run_as(OWNER, ...)`.
"""

from __future__ import annotations

import os
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.studio_db import get_connection

from utils.account_context import OWNER, run_as


_LIBRARY_COLLATION = "COLLATE NOCASE" if platform.system() == "Windows" else ""

_CACHE_WORDINGS = (
    ("Choose a cache folder.", "Choose a library folder."),
    (
        "The Hugging Face cache folder must be an absolute path.",
        "The library folder must be an absolute path.",
    ),
    ("The Hugging Face cache folder is invalid.", "The library folder is invalid."),
    ("Unsloth cannot use this cache folder:", "Unsloth cannot use this library folder:"),
)


def _library_wording(message: str) -> str:
    """Relabel :func:`utils.hf_cache_settings._validate_cache_home` errors so they
    read as library guidance rather than cache-settings guidance."""
    for cache_phrase, library_phrase in _CACHE_WORDINGS:
        message = message.replace(cache_phrase, library_phrase)
    return message


def _row_by_id(conn, library_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT id, path, label, created_at FROM model_libraries WHERE id = ?",
        (library_id,),
    ).fetchone()


def list_model_libraries() -> list[dict]:
    """Every registered library row, newest first. ``is_default`` is derived by
    the caller; the active HF cache home is never materialised as a row."""

    def _impl() -> list[dict]:
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT id, path, label, created_at FROM model_libraries ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    return run_as(OWNER, _impl)


def model_library_homes() -> list[Path]:
    """Cache homes of every registered library, for scan unification.

    Deliberately not resolved here: callers canonicalise against their own notion
    of the filesystem, and a dead volume must not raise while enumerating scans.
    """
    return [Path(row["path"]) for row in list_model_libraries()]


def add_model_library(path_value: str, label: Optional[str] = None) -> dict:
    """Register a new library location, validating it like a HF cache home.

    Raises ``ValueError`` when the path cannot be used; returns the existing row
    when the same location is already registered.
    """

    def _impl() -> dict:
        value = (path_value or "").strip()
        if not value:
            raise ValueError("Choose a library folder.")
        from utils.hf_cache_settings import _validate_cache_home

        try:
            resolved = _validate_cache_home(value)
        except ValueError as exc:
            raise ValueError(_library_wording(str(exc))) from exc

        normalized = os.path.realpath(resolved)
        conn = get_connection()
        try:
            existing = conn.execute(
                "SELECT id, path, label, created_at FROM model_libraries WHERE path = ? "
                f"{_LIBRARY_COLLATION}",
                (normalized,),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            try:
                conn.execute(
                    "INSERT INTO model_libraries (path, label, created_at) VALUES (?, ?, ?)",
                    (
                        normalized,
                        (label or "").strip() or None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass
            row = conn.execute(
                "SELECT id, path, label, created_at FROM model_libraries WHERE path = ? "
                f"{_LIBRARY_COLLATION}",
                (normalized,),
            ).fetchone()
            if row is None:
                raise ValueError("Library was concurrently removed")
            return dict(row)
        finally:
            conn.close()

    return run_as(OWNER, _impl)


def remove_model_library(library_id: int) -> bool:
    """Unregister a library. Unregistering the default library's location is refused."""

    def _impl() -> bool:
        conn = get_connection()
        try:
            row = _row_by_id(conn, library_id)
            if row is None:
                return False
            from utils.hf_cache_settings import get_hf_cache_paths

            active = str(get_hf_cache_paths().cache_home)
            if os.path.normcase(os.path.realpath(row["path"])) == os.path.normcase(
                os.path.realpath(active)
            ):
                raise ValueError(
                    "The default library cannot be removed. Set another library as default first."
                )
            cursor = conn.execute("DELETE FROM model_libraries WHERE id = ?", (library_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    return run_as(OWNER, _impl)


def set_default_model_library(library_id: int) -> None:
    """Make the registered library the default by pointing the active HF cache at
    its home. Uses the same validation, history and invalidation as the settings
    route's cache change."""

    def _impl() -> None:
        conn = get_connection()
        try:
            row = _row_by_id(conn, library_id)
        finally:
            conn.close()
        if row is None:
            raise ValueError("Library not found")
        from utils.hf_cache_settings import set_hf_cache_home

        set_hf_cache_home(row["path"])

    run_as(OWNER, _impl)
