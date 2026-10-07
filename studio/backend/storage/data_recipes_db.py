# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data Recipe definitions and run history in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Iterable

from storage.studio_db import get_connection


def _recipe_from_row(row: sqlite3.Row) -> dict:
    record = {
        "id": row["id"],
        "name": row["name"],
        "payload": json.loads(row["payload_json"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }
    if row["learning_recipe_id"] is not None:
        record["learningRecipeId"] = row["learning_recipe_id"]
    if row["learning_recipe_title"] is not None:
        record["learningRecipeTitle"] = row["learning_recipe_title"]
    return record


def _recipe_params(recipe: dict) -> tuple:
    return (
        recipe["id"],
        recipe["name"],
        json.dumps(recipe["payload"]),
        recipe.get("learningRecipeId"),
        recipe.get("learningRecipeTitle"),
        int(recipe["createdAt"]),
        int(recipe["updatedAt"]),
    )


def _execution_params(execution: dict) -> tuple:
    return (
        execution["id"],
        execution["recipeId"],
        int(execution["createdAt"]),
        json.dumps(execution),
    )


def list_recipes() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM data_recipes ORDER BY updated_at DESC").fetchall()
        return [_recipe_from_row(r) for r in rows]
    finally:
        conn.close()


def get_recipe(recipe_id: str) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM data_recipes WHERE id = ?", (recipe_id,)).fetchone()
        return _recipe_from_row(row) if row else None
    finally:
        conn.close()


class RecipeDeleted(Exception):
    pass


class RecipeConflict(Exception):
    pass


def upsert_recipe(recipe: dict, base_updated_at: int | None = None) -> dict:
    """``base_updated_at`` is the version the caller edited; a newer stored row is a conflict."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO data_recipes
                (id, name, payload_json, learning_recipe_id, learning_recipe_title,
                 created_at, updated_at)
            SELECT ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM data_recipe_tombstones WHERE id = ?1)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                payload_json = excluded.payload_json,
                learning_recipe_id = excluded.learning_recipe_id,
                learning_recipe_title = excluded.learning_recipe_title,
                updated_at = excluded.updated_at
            WHERE ?8 IS NULL OR data_recipes.updated_at = ?8
            """,
            (*_recipe_params(recipe), base_updated_at),
        )
        row = conn.execute("SELECT * FROM data_recipes WHERE id = ?", (recipe["id"],)).fetchone()
        conn.commit()
        if cur.rowcount:
            return _recipe_from_row(row)
        if row is None:
            raise RecipeDeleted(recipe["id"])
        raise RecipeConflict(recipe["id"])
    finally:
        conn.close()


def delete_recipe(recipe_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM data_recipes WHERE id = ?", (recipe_id,))
        conn.execute(
            "INSERT OR IGNORE INTO data_recipe_tombstones (id, deleted_at) VALUES (?, ?)",
            (recipe_id, int(time.time() * 1000)),
        )
        conn.commit()
    finally:
        conn.close()


def list_executions(recipe_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT record_json FROM data_recipe_executions"
            " WHERE recipe_id = ? ORDER BY created_at DESC",
            (recipe_id,),
        ).fetchall()
        return [json.loads(r["record_json"]) for r in rows]
    finally:
        conn.close()


_TERMINAL = ("completed", "cancelled", "error")


def upsert_execution(execution: dict) -> bool:
    """Returns False when the run's recipe does not exist or the id belongs to another recipe.

    A snapshot older than the stored one (two tabs tracking one run) is accepted but dropped:
    a lower lastEventId, a snapshot without analysis at the same lastEventId as one with it, or a
    non-terminal status over a terminal one, never replaces it.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO data_recipe_executions (id, recipe_id, created_at, record_json)
            SELECT ?, ?, ?, ?
            WHERE EXISTS (SELECT 1 FROM data_recipes WHERE id = ?2)
            ON CONFLICT(id) DO UPDATE SET record_json = excluded.record_json
            WHERE data_recipe_executions.recipe_id = excluded.recipe_id
              AND COALESCE(json_extract(excluded.record_json, '$.lastEventId'), -1)
                  >= COALESCE(json_extract(data_recipe_executions.record_json, '$.lastEventId'), -1)
              AND NOT (
                  COALESCE(json_extract(excluded.record_json, '$.lastEventId'), -1)
                      = COALESCE(json_extract(data_recipe_executions.record_json, '$.lastEventId'), -1)
                  AND json_extract(data_recipe_executions.record_json, '$.analysis') IS NOT NULL
                  AND json_extract(excluded.record_json, '$.analysis') IS NULL
              )
              AND NOT (
                  json_extract(data_recipe_executions.record_json, '$.status') IN (?5, ?6, ?7)
                  AND COALESCE(json_extract(excluded.record_json, '$.status'), '')
                      NOT IN (?5, ?6, ?7)
              )
            """,
            (*_execution_params(execution), *_TERMINAL),
        )
        stored = conn.execute(
            "SELECT recipe_id FROM data_recipe_executions WHERE id = ?", (execution["id"],)
        ).fetchone()
        conn.commit()
        return cur.rowcount > 0 or (
            stored is not None and stored["recipe_id"] == execution["recipeId"]
        )
    finally:
        conn.close()


def import_legacy(recipes: Iterable[dict], executions: Iterable[dict]) -> dict:
    """Insert-only: never overwrites a server record or revives a deleted recipe."""
    conn = get_connection()
    try:
        recipe_count = 0
        for recipe in recipes:
            recipe_count += conn.execute(
                """
                INSERT OR IGNORE INTO data_recipes
                    (id, name, payload_json, learning_recipe_id, learning_recipe_title,
                     created_at, updated_at)
                SELECT ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM data_recipe_tombstones WHERE id = ?1)
                """,
                _recipe_params(recipe),
            ).rowcount
        execution_count = 0
        for execution in executions:
            execution_count += conn.execute(
                """
                INSERT OR IGNORE INTO data_recipe_executions
                    (id, recipe_id, created_at, record_json)
                SELECT ?, ?, ?, ?
                WHERE EXISTS (SELECT 1 FROM data_recipes WHERE id = ?2)
                """,
                _execution_params(execution),
            ).rowcount
        conn.commit()
        return {"recipes": recipe_count, "executions": execution_count}
    finally:
        conn.close()
