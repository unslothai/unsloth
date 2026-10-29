# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite for LLM providers; API keys stay in the browser and are sent encrypted per request"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from models.providers import ProviderReasoningConfig, normalize_provider_reasoning_config
from storage.studio_db import connect_studio_db
from utils.paths import studio_db_path, ensure_dir

_schema_lock = threading.Lock()
_schema_ready: set[Path] = set()
_UNSET = object()

_LEGACY_CUSTOM_PRESET_TYPES = {
    "custom": "custom",
    "llama.cpp": "llama_cpp",
    "vllm": "vllm",
    "ollama": "ollama",
}


def _migrate_legacy_custom_provider_types(conn: sqlite3.Connection) -> None:
    """Rows saved as openai change type only on an exact built-in label; a URL may be an OpenAI proxy."""
    updates: list[tuple[str, str]] = []
    rows = conn.execute(
        "SELECT id, display_name, base_url FROM llm_providers WHERE provider_type = 'openai'"
    ).fetchall()
    for provider_id, display_name, base_url in rows:
        label = str(display_name or "").strip().lower()
        url = str(base_url or "").strip()
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
        except ValueError:
            host = ""

        # Display name is user-editable; never let it turn an OpenAI endpoint into a custom one.
        if host == "api.openai.com" or host.endswith(
            (".openai.azure.com", ".services.ai.azure.com")
        ):
            continue

        migrated_type = _LEGACY_CUSTOM_PRESET_TYPES.get(label)
        if migrated_type is not None:
            updates.append((migrated_type, str(provider_id)))

    conn.executemany(
        "UPDATE llm_providers SET provider_type = ? WHERE id = ?",
        updates,
    )


def _encode_models_json(models: Optional[list[str]]) -> str:
    if not models:
        return "[]"
    return json.dumps([str(model).strip() for model in models if str(model).strip()])


def _decode_models_json(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(model).strip() for model in parsed if str(model).strip()]


def _encode_reasoning_config(config) -> Optional[str]:
    if config is None:
        return None
    if isinstance(config, ProviderReasoningConfig):
        config = config.model_dump()
    return ProviderReasoningConfig.model_validate(config).model_dump_json()


def _decode_reasoning_config(raw) -> Optional[dict]:
    if not raw:
        return None
    try:
        return normalize_provider_reasoning_config(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return None


def _row_models(row: sqlite3.Row) -> tuple[list[str], list[str]]:
    return (
        _decode_models_json(row["models_json"] if "models_json" in row.keys() else None),
        _decode_models_json(
            row["available_models_json"] if "available_models_json" in row.keys() else None
        ),
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_providers (
            id TEXT NOT NULL PRIMARY KEY,
            provider_type TEXT NOT NULL,
            display_name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            is_enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_providers)").fetchall()}
    if "models_json" not in existing_cols:
        conn.execute("ALTER TABLE llm_providers ADD COLUMN models_json TEXT NOT NULL DEFAULT '[]'")
    if "available_models_json" not in existing_cols:
        conn.execute(
            "ALTER TABLE llm_providers ADD COLUMN available_models_json TEXT NOT NULL DEFAULT '[]'"
        )
    added_api_type = "api_type" not in existing_cols
    if added_api_type:
        conn.execute(
            "ALTER TABLE llm_providers ADD COLUMN api_type TEXT NOT NULL DEFAULT 'chat_completions'"
        )
        _migrate_legacy_custom_provider_types(conn)
    if "max_output_tokens" not in existing_cols:
        conn.execute("ALTER TABLE llm_providers ADD COLUMN max_output_tokens INTEGER")
    if "reasoning_config_json" not in existing_cols:
        conn.execute("ALTER TABLE llm_providers ADD COLUMN reasoning_config_json TEXT")
    if "shared" not in existing_cols:
        conn.execute("ALTER TABLE llm_providers ADD COLUMN shared INTEGER NOT NULL DEFAULT 0")
    # ALTER TABLE persists independently in SQLite, but its one-time data migration does not.
    conn.commit()


def reset_schema_state_for_tests() -> None:
    with _schema_lock:
        _schema_ready.clear()


def get_connection() -> sqlite3.Connection:
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = connect_studio_db(db_path)
    conn.row_factory = sqlite3.Row
    if db_path not in _schema_ready:
        with _schema_lock:
            schema_path = db_path.resolve()
            if schema_path not in _schema_ready:
                try:
                    _ensure_schema(conn)
                    _schema_ready.add(schema_path)
                except Exception:
                    conn.close()
                    raise
    return conn


@contextmanager
def provider_bundle_transaction() -> Iterator[sqlite3.Connection]:
    """One SQLite write transaction, so no other process sees a new endpoint paired with the old key."""
    # Both tables must exist before the transaction; credential_secrets commits its own schema.
    from storage import credential_secrets

    credential_secrets.ensure_schema()
    conn = get_connection()
    try:
        conn.commit()
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_provider(
    id: str,
    provider_type: str,
    display_name: str,
    base_url: str,
    models: Optional[list[str]] = None,
    available_models: Optional[list[str]] = None,
    max_output_tokens: Optional[int] = None,
    api_type: str = "chat_completions",
    reasoning_config: Optional[dict] = None,
    shared: bool = False,
) -> None:
    """Insert a new provider configuration."""
    reasoning_json = _encode_reasoning_config(reasoning_config)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO llm_providers (
                id, provider_type, display_name, base_url,
                models_json, available_models_json, max_output_tokens, api_type,
                reasoning_config_json, shared, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                provider_type,
                display_name,
                base_url,
                _encode_models_json(models),
                _encode_models_json(available_models),
                max_output_tokens,
                api_type,
                reasoning_json,
                1 if shared else 0,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def update_provider(
    id: str,
    display_name: Optional[str] = None,
    base_url: Optional[str] = None,
    is_enabled: Optional[bool] = None,
    models: Optional[list[str]] = None,
    available_models: Optional[list[str]] = None,
    max_output_tokens: int | None | object = _UNSET,
    *,
    connection: sqlite3.Connection | None = None,
    api_type: str | None = None,
    reasoning_config: dict | None | object = _UNSET,
    shared: bool | None = None,
) -> bool:
    """Update fields on an existing provider. Returns True if a row was updated."""
    updates = []
    params = []
    if display_name is not None:
        updates.append("display_name = ?")
        params.append(display_name)
    if base_url is not None:
        updates.append("base_url = ?")
        params.append(base_url)
    if api_type is not None:
        updates.append("api_type = ?")
        params.append(api_type)
    if is_enabled is not None:
        updates.append("is_enabled = ?")
        params.append(1 if is_enabled else 0)
    if models is not None:
        updates.append("models_json = ?")
        params.append(_encode_models_json(models))
    if available_models is not None:
        updates.append("available_models_json = ?")
        params.append(_encode_models_json(available_models))
    if max_output_tokens is not _UNSET:
        updates.append("max_output_tokens = ?")
        params.append(max_output_tokens)
    if reasoning_config is not _UNSET:
        updates.append("reasoning_config_json = ?")
        params.append(_encode_reasoning_config(reasoning_config))
    if shared is not None:
        updates.append("shared = ?")
        params.append(1 if shared else 0)
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(id)

    owns_connection = connection is None
    conn = connection or get_connection()
    try:
        cursor = conn.execute(
            f"UPDATE llm_providers SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        if owns_connection:
            conn.commit()
        return cursor.rowcount > 0
    finally:
        if owns_connection:
            conn.close()


def delete_provider(id: str) -> bool:
    """Delete a provider by ID. Returns True if a row was deleted."""
    conn = get_connection()
    try:
        cursor = conn.execute("DELETE FROM llm_providers WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _row_to_provider(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["reasoning_config"] = _decode_reasoning_config(data.pop("reasoning_config_json", None))
    models, available_models = _row_models(row)
    data["models"] = models
    data["available_models"] = available_models
    data["shared"] = bool(data.get("shared"))
    return data


def _own_rows(where: str = "", params: tuple = ()) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM llm_providers {where} ORDER BY created_at", params
        ).fetchall()
        return [_row_to_provider(row) for row in rows]
    finally:
        conn.close()


def base_url_carries_credential(base_url: Optional[str]) -> bool:
    try:
        parts = urlparse(base_url or "")
        return bool(parts.username or parts.password or parts.query)
    except ValueError:
        return True


def _owner_shared_rows(provider_id: Optional[str] = None) -> list[dict]:
    # A ChatGPT subscription is never shared: it is one person's sign-in, not an API key.
    from utils.account_context import OWNER, is_owner_context, run_as

    if is_owner_context():
        return []
    where = "WHERE shared = 1 AND provider_type != 'openai_codex'"
    params: tuple = ()
    if provider_id is not None:
        where += " AND id = ?"
        params = (provider_id,)

    def _read() -> list[dict]:
        # Reading must never create the owner's database.
        if not studio_db_path().exists():
            return []
        return _own_rows(where, params)

    try:
        rows = run_as(OWNER, _read)
    except Exception:  # noqa: BLE001 - an unreadable owner store shares nothing
        logger.warning("Could not read the owner's shared connections", exc_info = True)
        return []
    from hub.services.models.account_access import shared_connection_view

    # Re-checked on read: an older build can edit the URL of a row it does not know is shared.
    rows = [row for row in rows if not base_url_carries_credential(row.get("base_url"))]
    return [view for view in (shared_connection_view(row) for row in rows) if view is not None]


def is_owner_shared_provider(id: str) -> bool:
    return not _own_rows("WHERE id = ?", (id,)) and bool(_owner_shared_rows(id))


def get_own_provider(id: str) -> Optional[dict]:
    rows = _own_rows("WHERE id = ?", (id,))
    return rows[0] if rows else None


def get_provider(id: str) -> Optional[dict]:
    own = get_own_provider(id)
    if own is not None:
        return own
    shared = _owner_shared_rows(id)
    return shared[0] if shared else None


def list_providers() -> list[dict]:
    own = _own_rows()
    own_ids = {row["id"] for row in own}
    return own + [row for row in _owner_shared_rows() if row["id"] not in own_ids]
