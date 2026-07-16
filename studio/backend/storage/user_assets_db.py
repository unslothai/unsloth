# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-scoped SQLite persistence for Studio recipes and executions."""

from __future__ import annotations

import base64
import binascii
import json
import sqlite3
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

from core.user_assets_validation import (
    UserAssetValidationError,
    canonical_json,
    project_execution_metadata,
    redact_secret_fields,
    validate_id,
    validate_legacy_batch_size,
    validate_name,
    validate_recipe_payload,
    validate_timestamp,
    MAX_EXECUTION_JSON_BYTES,
    MAX_ID_CHARS,
    MAX_RECIPE_JSON_BYTES,
)
from storage import studio_db


DEFAULT_EXECUTION_PAGE_LIMIT = 100
MAX_EXECUTION_PAGE_LIMIT = 100


class UserAssetStorageError(RuntimeError):
    """Safe storage error that route handlers can map without inspecting SQL."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        current_resource: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.current_resource = current_resource
        self.current_revision = (
            current_resource.get("revision") if current_resource is not None else None
        )


class AssetAlreadyExistsError(UserAssetStorageError):
    def __init__(self, current_resource: dict[str, Any]) -> None:
        super().__init__(
            "already_exists",
            "An asset with this id already exists",
            current_resource = current_resource,
        )


class RevisionConflictError(UserAssetStorageError):
    def __init__(self, current_resource: dict[str, Any]) -> None:
        super().__init__(
            "revision_conflict",
            "The asset was changed by another writer",
            current_resource = current_resource,
        )


class RetiredAssetError(UserAssetStorageError):
    def __init__(self) -> None:
        super().__init__("id_retired", "This asset id has been retired")


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _require_owner(owner_subject: str) -> str:
    if not isinstance(owner_subject, str) or not owner_subject:
        raise ValueError("owner_subject must be a non-empty string")
    return owner_subject


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UserAssetValidationError("invalid_json_object", f"{field_name} must be a JSON object")
    return value


def _validate_expected_revision(value: Any, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise UserAssetValidationError("invalid_revision", "revision must be an integer")
    return value


def _validate_recipe_links(value: Mapping[str, Any]) -> tuple[str | None, str | None]:
    learning_id = value.get("learningRecipeId")
    learning_title = value.get("learningRecipeTitle")
    return (
        validate_id(learning_id, "learning recipe id") if learning_id is not None else None,
        validate_name(learning_title, "learning recipe title")
        if learning_title is not None
        else None,
    )


def _ledger_safe_id(value: Any) -> str | None:
    if isinstance(value, str) and value and len(value) <= MAX_ID_CHARS:
        return value
    return None


def _recipe_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "payload": json.loads(row["payload_json"]),
        "learningRecipeId": row["learning_recipe_id"],
        "learningRecipeTitle": row["learning_recipe_title"],
        "revision": row["revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _execution_from_row(row: sqlite3.Row) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"])
    metadata.update(
        {
            "id": row["id"],
            "recipeId": row["recipe_id"],
            "revision": row["revision"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "finishedAt": row["finished_at"],
        }
    )
    return metadata


def list_recipes(owner_subject: str) -> list[dict[str, Any]]:
    owner = _require_owner(owner_subject)
    conn = studio_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM data_recipes
            WHERE owner_subject = ? AND deleted_at IS NULL
            ORDER BY updated_at DESC, id
            """,
            (owner,),
        ).fetchall()
        return [_recipe_from_row(row) for row in rows]
    finally:
        conn.close()


def get_recipe(owner_subject: str, recipe_id: str) -> dict[str, Any] | None:
    owner = _require_owner(owner_subject)
    asset_id = validate_id(recipe_id, "recipe id")
    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM data_recipes
            WHERE owner_subject = ? AND id = ? AND deleted_at IS NULL
            """,
            (owner, asset_id),
        ).fetchone()
        return _recipe_from_row(row) if row is not None else None
    finally:
        conn.close()


def create_recipe(owner_subject: str, recipe: Mapping[str, Any]) -> dict[str, Any]:
    owner = _require_owner(owner_subject)
    value = _require_mapping(recipe, "recipe")
    asset_id = validate_id(value.get("id"), "recipe id")
    name = validate_name(value.get("name"), "recipe name")
    payload = validate_recipe_payload(value.get("payload"))
    learning_id, learning_title = _validate_recipe_links(value)
    payload_json = canonical_json(payload, MAX_RECIPE_JSON_BYTES, "recipe payload")
    now = _now_ms()
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT * FROM data_recipes WHERE owner_subject = ? AND id = ?",
            (owner, asset_id),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            if existing["deleted_at"] is not None:
                raise RetiredAssetError()
            raise AssetAlreadyExistsError(_recipe_from_row(existing))
        conn.execute(
            """
            INSERT INTO data_recipes
                (owner_subject, id, name, payload_json, learning_recipe_id,
                 learning_recipe_title, revision, created_at, updated_at, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)
            """,
            (
                owner,
                asset_id,
                name,
                payload_json,
                learning_id,
                learning_title,
                now,
                now,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM data_recipes WHERE owner_subject = ? AND id = ?",
            (owner, asset_id),
        ).fetchone()
        assert row is not None
        return _recipe_from_row(row)
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def update_recipe(
    owner_subject: str, recipe_id: str, recipe: Mapping[str, Any], expected_revision: int
) -> dict[str, Any] | None:
    owner = _require_owner(owner_subject)
    asset_id = validate_id(recipe_id, "recipe id")
    value = _require_mapping(recipe, "recipe")
    name = validate_name(value.get("name"), "recipe name")
    payload = validate_recipe_payload(value.get("payload"))
    learning_id, learning_title = _validate_recipe_links(value)
    has_learning_id = "learningRecipeId" in value
    has_learning_title = "learningRecipeTitle" in value
    payload_json = canonical_json(payload, MAX_RECIPE_JSON_BYTES, "recipe payload")
    expected_revision = _validate_expected_revision(expected_revision)
    now = _now_ms()
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        changed = conn.execute(
            """
            UPDATE data_recipes
            SET name = ?, payload_json = ?,
                learning_recipe_id = CASE WHEN ? THEN ? ELSE learning_recipe_id END,
                learning_recipe_title = CASE WHEN ? THEN ? ELSE learning_recipe_title END,
                revision = revision + 1,
                updated_at = MAX(updated_at + 1, created_at, ?)
            WHERE owner_subject = ? AND id = ? AND deleted_at IS NULL AND revision = ?
            """,
            (
                name,
                payload_json,
                has_learning_id,
                learning_id,
                has_learning_title,
                learning_title,
                now,
                owner,
                asset_id,
                expected_revision,
            ),
        ).rowcount
        if not changed:
            current = conn.execute(
                "SELECT * FROM data_recipes WHERE owner_subject = ? AND id = ?",
                (owner, asset_id),
            ).fetchone()
            conn.rollback()
            if current is None:
                return None
            if current["deleted_at"] is not None:
                raise RetiredAssetError()
            raise RevisionConflictError(_recipe_from_row(current))
        conn.commit()
        row = conn.execute(
            "SELECT * FROM data_recipes WHERE owner_subject = ? AND id = ?",
            (owner, asset_id),
        ).fetchone()
        assert row is not None
        return _recipe_from_row(row)
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def delete_recipe(owner_subject: str, recipe_id: str, expected_revision: int) -> bool:
    owner = _require_owner(owner_subject)
    asset_id = validate_id(recipe_id, "recipe id")
    expected_revision = _validate_expected_revision(expected_revision)
    now = _now_ms()
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute(
            "SELECT * FROM data_recipes WHERE owner_subject = ? AND id = ?",
            (owner, asset_id),
        ).fetchone()
        if current is None:
            conn.rollback()
            return False
        if current["deleted_at"] is not None:
            conn.commit()
            return True
        if current["revision"] != expected_revision:
            conn.rollback()
            raise RevisionConflictError(_recipe_from_row(current))
        conn.execute(
            """
            UPDATE data_recipes
            SET revision = revision + 1,
                updated_at = MAX(updated_at + 1, created_at, ?),
                deleted_at = MAX(updated_at + 1, created_at, ?)
            WHERE owner_subject = ? AND id = ?
            """,
            (now, now, owner, asset_id),
        )
        conn.commit()
        return True
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _encode_execution_cursor(created_at: int, asset_id: str) -> str:
    payload = json.dumps(
        {"v": 1, "createdAt": created_at, "id": asset_id},
        separators = (",", ":"),
        ensure_ascii = True,
    ).encode("ascii")
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _decode_execution_cursor(cursor: str) -> tuple[int, str]:
    try:
        if not isinstance(cursor, str) or not cursor or len(cursor) > 512:
            raise ValueError
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            cursor + padding,
            altchars = b"-_",
            validate = True,
        )
        value = json.loads(decoded.decode("ascii"))
        if not isinstance(value, dict) or value.get("v") != 1:
            raise ValueError
        created_at = validate_timestamp(value.get("createdAt"), "cursor createdAt")
        asset_id = validate_id(value.get("id"), "cursor execution id")
        return created_at, asset_id
    except (
        binascii.Error,
        UnicodeDecodeError,
        json.JSONDecodeError,
        UserAssetValidationError,
        ValueError,
    ) as error:
        raise UserAssetValidationError("invalid_cursor", "execution cursor is invalid") from error


def list_recipe_executions(
    owner_subject: str,
    recipe_id: str,
    *,
    cursor: str | None = None,
    limit: int = DEFAULT_EXECUTION_PAGE_LIMIT,
) -> dict[str, Any] | None:
    owner = _require_owner(owner_subject)
    parent_id = validate_id(recipe_id, "recipe id")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_EXECUTION_PAGE_LIMIT
    ):
        raise UserAssetValidationError(
            "invalid_page_limit",
            f"execution page limit must be between 1 and {MAX_EXECUTION_PAGE_LIMIT}",
        )
    cursor_values = _decode_execution_cursor(cursor) if cursor is not None else None
    conn = studio_db.get_connection()
    try:
        parent = conn.execute(
            """
            SELECT 1 FROM data_recipes
            WHERE owner_subject = ? AND id = ? AND deleted_at IS NULL
            """,
            (owner, parent_id),
        ).fetchone()
        if parent is None:
            return None
        if cursor_values is None:
            rows = conn.execute(
                """
                SELECT * FROM data_recipe_executions
                WHERE owner_subject = ? AND recipe_id = ?
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                """,
                (owner, parent_id, limit + 1),
            ).fetchall()
        else:
            cursor_created_at, cursor_id = cursor_values
            rows = conn.execute(
                """
                SELECT * FROM data_recipe_executions
                WHERE owner_subject = ? AND recipe_id = ?
                  AND (created_at < ? OR (created_at = ? AND id > ?))
                ORDER BY created_at DESC, id ASC
                LIMIT ?
                """,
                (
                    owner,
                    parent_id,
                    cursor_created_at,
                    cursor_created_at,
                    cursor_id,
                    limit + 1,
                ),
            ).fetchall()
        page_rows = rows[:limit]
        resumable_row = conn.execute(
            """
            SELECT * FROM data_recipe_executions
            WHERE owner_subject = ? AND recipe_id = ? AND finished_at IS NULL
              AND json_extract(metadata_json, '$.status') NOT IN
                  ('cancelled', 'completed', 'error')
            ORDER BY created_at DESC, id ASC
            LIMIT 1
            """,
            (owner, parent_id),
        ).fetchone()
        next_cursor = None
        if len(rows) > limit and page_rows:
            last = page_rows[-1]
            next_cursor = _encode_execution_cursor(last["created_at"], last["id"])
        return {
            "executions": [_execution_from_row(row) for row in page_rows],
            "nextCursor": next_cursor,
            "resumable": _execution_from_row(resumable_row) if resumable_row is not None else None,
        }
    finally:
        conn.close()


def upsert_recipe_execution(
    owner_subject: str,
    recipe_id: str,
    execution_id: str,
    metadata: Mapping[str, Any],
    expected_revision: int | None = None,
) -> dict[str, Any] | None:
    owner = _require_owner(owner_subject)
    parent_id = validate_id(recipe_id, "recipe id")
    asset_id = validate_id(execution_id, "execution id")
    expected_revision = _validate_expected_revision(expected_revision, optional = True)
    projected = project_execution_metadata(metadata)
    metadata_json = canonical_json(projected, MAX_EXECUTION_JSON_BYTES, "execution metadata")
    created_at = validate_timestamp(projected.get("createdAt"), "createdAt")
    finished_at = projected.get("finishedAt")
    if finished_at is not None:
        finished_at = validate_timestamp(finished_at, "finishedAt")
        if finished_at < created_at:
            raise UserAssetValidationError(
                "invalid_timestamp", "finishedAt must not be earlier than createdAt"
            )
    now = _now_ms()
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        parent = conn.execute(
            """
            SELECT 1 FROM data_recipes
            WHERE owner_subject = ? AND id = ? AND deleted_at IS NULL
            """,
            (owner, parent_id),
        ).fetchone()
        if parent is None:
            conn.rollback()
            return None
        current = conn.execute(
            """
            SELECT * FROM data_recipe_executions
            WHERE owner_subject = ? AND id = ?
            """,
            (owner, asset_id),
        ).fetchone()
        if current is None:
            if expected_revision not in (None, 0):
                conn.rollback()
                return None
            conn.execute(
                """
                INSERT INTO data_recipe_executions
                    (owner_subject, id, recipe_id, metadata_json, revision,
                     created_at, updated_at, finished_at)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (
                    owner,
                    asset_id,
                    parent_id,
                    metadata_json,
                    created_at,
                    max(now, created_at),
                    finished_at,
                ),
            )
        else:
            if current["recipe_id"] != parent_id:
                conn.rollback()
                raise AssetAlreadyExistsError(_execution_from_row(current))
            if expected_revision is None or current["revision"] != expected_revision:
                conn.rollback()
                raise RevisionConflictError(_execution_from_row(current))
            conn.execute(
                """
                UPDATE data_recipe_executions
                SET metadata_json = ?, revision = revision + 1,
                    updated_at = MAX(updated_at + 1, created_at, ?),
                    finished_at = ?
                WHERE owner_subject = ? AND id = ? AND revision = ?
                """,
                (
                    metadata_json,
                    now,
                    finished_at,
                    owner,
                    asset_id,
                    expected_revision,
                ),
            )
        conn.commit()
        row = conn.execute(
            """
            SELECT * FROM data_recipe_executions
            WHERE owner_subject = ? AND id = ?
            """,
            (owner, asset_id),
        ).fetchone()
        assert row is not None
        return _execution_from_row(row)
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def list_legacy_imports(owner_subject: str, source: str) -> dict[str, list[str]]:
    owner = _require_owner(owner_subject)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    conn = studio_db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT entity_kind, legacy_id FROM user_asset_legacy_imports
            WHERE owner_subject = ? AND source = ?
              AND outcome <> 'missing_parent'
              AND (
                outcome <> 'rejected'
                OR reason IN ('already_exists', 'parent_retired')
              )
            ORDER BY entity_kind, legacy_id
            """,
            (owner, source),
        ).fetchall()
        return {
            "recipes": [row["legacy_id"] for row in rows if row["entity_kind"] == "recipe"],
            "executions": [row["legacy_id"] for row in rows if row["entity_kind"] == "execution"],
        }
    finally:
        conn.close()


def _legacy_result(
    legacy_id: Any,
    outcome: str,
    *,
    reason: str | None = None,
    redacted_paths: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": legacy_id if isinstance(legacy_id, str) else "",
        "outcome": outcome,
    }
    if reason is not None:
        result["reason"] = reason
    if redacted_paths:
        result["redactedPaths"] = redacted_paths
    return result


def _ledger_outcome(
    conn: sqlite3.Connection,
    owner: str,
    source: str,
    kind: str,
    legacy_id: str,
    outcome: str,
    reason: str | None,
    imported_at: int,
) -> None:
    conn.execute(
        """
        INSERT INTO user_asset_legacy_imports
            (owner_subject, source, entity_kind, legacy_id, outcome, reason, imported_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (owner, source, kind, legacy_id, outcome, reason, imported_at),
    )


def _already_imported(
    conn: sqlite3.Connection, owner: str, source: str, kind: str, legacy_id: str
) -> bool:
    row = conn.execute(
        """
        SELECT outcome, reason FROM user_asset_legacy_imports
        WHERE owner_subject = ? AND source = ? AND entity_kind = ? AND legacy_id = ?
        """,
        (owner, source, kind, legacy_id),
    ).fetchone()
    if row is None:
        return False
    if row["outcome"] == "missing_parent" or (
        row["outcome"] == "rejected" and row["reason"] not in {"already_exists", "parent_retired"}
    ):
        # Drop old validation rejections so corrected input can retry.
        conn.execute(
            """
            DELETE FROM user_asset_legacy_imports
            WHERE owner_subject = ? AND source = ? AND entity_kind = ? AND legacy_id = ?
            """,
            (owner, source, kind, legacy_id),
        )
        return False
    return True


def import_legacy_assets(
    owner_subject: str, source: str, recipes: list[Any], executions: list[Any]
) -> dict[str, Any]:
    """Import a bounded batch and ledger only terminal outcomes."""

    owner = _require_owner(owner_subject)
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")
    validate_legacy_batch_size(recipes, executions)
    now = _now_ms()
    recipe_results: list[dict[str, Any]] = []
    execution_results: list[dict[str, Any]] = []
    conn = studio_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for raw in recipes:
            item = raw if isinstance(raw, Mapping) else {}
            raw_id = item.get("id")
            ledger_id = _ledger_safe_id(raw_id)
            if ledger_id and _already_imported(conn, owner, source, "recipe", ledger_id):
                recipe_results.append(_legacy_result(ledger_id, "already_imported"))
                continue
            try:
                asset_id = validate_id(raw_id, "recipe id")
                if _already_imported(conn, owner, source, "recipe", asset_id):
                    recipe_results.append(_legacy_result(asset_id, "already_imported"))
                    continue
                existing = conn.execute(
                    "SELECT deleted_at FROM data_recipes WHERE owner_subject = ? AND id = ?",
                    (owner, asset_id),
                ).fetchone()
                if existing is not None:
                    outcome = "id_retired" if existing["deleted_at"] is not None else "rejected"
                    reason = "id_retired" if outcome == "id_retired" else "already_exists"
                    _ledger_outcome(conn, owner, source, "recipe", asset_id, outcome, reason, now)
                    recipe_results.append(_legacy_result(asset_id, outcome, reason = reason))
                    continue
                name = validate_name(item.get("name", "Unnamed"), "recipe name")
                clean_payload, paths = validate_recipe_payload(item.get("payload"), legacy = True)
                learning_id, learning_title = _validate_recipe_links(item)
                payload_json = canonical_json(
                    clean_payload, MAX_RECIPE_JSON_BYTES, "recipe payload"
                )
                created_at = item.get("createdAt", now)
                created_at = validate_timestamp(created_at, "createdAt")
                if "updatedAt" in item:
                    updated_at = validate_timestamp(item["updatedAt"], "updatedAt")
                    updated_at = max(created_at, updated_at)
                else:
                    updated_at = max(now, created_at)
                outcome = "redacted" if paths else "imported"
                conn.execute(
                    """
                    INSERT INTO data_recipes
                        (owner_subject, id, name, payload_json, learning_recipe_id,
                         learning_recipe_title, revision, created_at, updated_at, deleted_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, NULL)
                    """,
                    (
                        owner,
                        asset_id,
                        name,
                        payload_json,
                        learning_id,
                        learning_title,
                        created_at,
                        updated_at,
                    ),
                )
                _ledger_outcome(conn, owner, source, "recipe", asset_id, outcome, None, now)
                recipe_results.append(_legacy_result(asset_id, outcome, redacted_paths = paths))
            except UserAssetValidationError as error:
                result = _legacy_result(raw_id, "rejected", reason = error.code)
                recipe_results.append(result)

        for raw in executions:
            item = raw if isinstance(raw, Mapping) else {}
            raw_id = item.get("id")
            ledger_id = _ledger_safe_id(raw_id)
            if ledger_id and _already_imported(conn, owner, source, "execution", ledger_id):
                execution_results.append(_legacy_result(ledger_id, "already_imported"))
                continue
            try:
                asset_id = validate_id(raw_id, "execution id")
                parent_id = validate_id(item.get("recipeId"), "recipe id")
                parent = conn.execute(
                    """
                    SELECT deleted_at FROM data_recipes
                    WHERE owner_subject = ? AND id = ?
                    """,
                    (owner, parent_id),
                ).fetchone()
                if parent is None:
                    execution_results.append(
                        _legacy_result(asset_id, "missing_parent", reason = "missing_parent")
                    )
                    continue
                if parent["deleted_at"] is not None:
                    _ledger_outcome(
                        conn,
                        owner,
                        source,
                        "execution",
                        asset_id,
                        "rejected",
                        "parent_retired",
                        now,
                    )
                    execution_results.append(
                        _legacy_result(asset_id, "rejected", reason = "parent_retired")
                    )
                    continue
                existing = conn.execute(
                    """
                    SELECT 1 FROM data_recipe_executions
                    WHERE owner_subject = ? AND id = ?
                    """,
                    (owner, asset_id),
                ).fetchone()
                if existing is not None:
                    _ledger_outcome(
                        conn,
                        owner,
                        source,
                        "execution",
                        asset_id,
                        "rejected",
                        "already_exists",
                        now,
                    )
                    execution_results.append(
                        _legacy_result(asset_id, "rejected", reason = "already_exists")
                    )
                    continue
                clean_item, paths = redact_secret_fields(item)
                projected = project_execution_metadata(clean_item)
                created_at = validate_timestamp(projected.get("createdAt"), "createdAt")
                finished_at = projected.get("finishedAt")
                if finished_at is not None:
                    finished_at = validate_timestamp(finished_at, "finishedAt")
                    if finished_at < created_at:
                        raise UserAssetValidationError(
                            "invalid_timestamp",
                            "finishedAt must not be earlier than createdAt",
                        )
                metadata_json = canonical_json(
                    projected, MAX_EXECUTION_JSON_BYTES, "execution metadata"
                )
                conn.execute(
                    """
                    INSERT INTO data_recipe_executions
                        (owner_subject, id, recipe_id, metadata_json, revision,
                         created_at, updated_at, finished_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        owner,
                        asset_id,
                        parent_id,
                        metadata_json,
                        created_at,
                        max(now, created_at),
                        finished_at,
                    ),
                )
                outcome = "redacted" if paths else "imported"
                _ledger_outcome(conn, owner, source, "execution", asset_id, outcome, None, now)
                execution_results.append(_legacy_result(asset_id, outcome, redacted_paths = paths))
            except UserAssetValidationError as error:
                execution_results.append(_legacy_result(raw_id, "rejected", reason = error.code))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    outcomes = Counter(result["outcome"] for result in (*recipe_results, *execution_results))
    return {
        "recipes": recipe_results,
        "executions": execution_results,
        "summary": dict(sorted(outcomes.items())),
    }


list_executions = list_recipe_executions
upsert_execution = upsert_recipe_execution
