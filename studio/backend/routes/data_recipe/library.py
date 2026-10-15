# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Saved Data Recipes and their run history, stored per account in studio.db."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import (
    authenticated_via_api_key,
    require_ui_session_for_local_commands,
)
from storage import data_recipes_db as db

router = APIRouter(prefix = "/recipes")
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]

_ID = Field(min_length = 1, max_length = 128)
# JS millisecond timestamps; also keeps values inside SQLite INTEGER.
_TIME = Field(ge = 0, le = 2**53)
_TIME_OPTIONAL = Field(default = None, ge = 0, le = 2**53)


class RecipeRecord(BaseModel):
    id: str = _ID
    name: str = Field(max_length = 10_000)
    payload: dict[str, Any]
    createdAt: int = _TIME
    updatedAt: int = _TIME
    learningRecipeId: str | None = Field(default = None, max_length = 128)
    learningRecipeTitle: str | None = Field(default = None, max_length = 10_000)


class SaveRecipeRequest(RecipeRecord):
    # The updatedAt the client last read; omitted for a brand-new recipe.
    baseUpdatedAt: int | None = _TIME_OPTIONAL


class ExecutionRecord(BaseModel):
    # The run record is UI state with many optional fields; only the keys used for storage are typed.
    model_config = ConfigDict(extra = "allow")
    id: str = _ID
    recipeId: str = _ID
    createdAt: int = _TIME


class LegacyImportRequest(BaseModel):
    recipes: list[RecipeRecord] = Field(default_factory = list, max_length = 10_000)
    executions: list[ExecutionRecord] = Field(default_factory = list, max_length = 50_000)


def _payload_has_stdio_mcp(payload: Any) -> bool:
    """True when the recipe holds a stdio MCP provider anywhere; running it later starts that command."""
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("provider_type") == "stdio":
                return True
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return False


@router.get("")
def get_recipes():
    return {"recipes": db.list_recipes()}


@router.post("/import")
def import_legacy_recipes(req: LegacyImportRequest, via_api_key: ViaApiKey = False):
    if any(_payload_has_stdio_mcp(r.payload) for r in req.recipes):
        require_ui_session_for_local_commands(via_api_key)
    return db.import_legacy(
        [r.model_dump(exclude_none = True) for r in req.recipes],
        [e.model_dump() for e in req.executions],
    )


@router.get("/{recipe_id}")
def get_recipe(recipe_id: str):
    record = db.get_recipe(recipe_id)
    if record is None:
        raise HTTPException(status_code = 404, detail = "Recipe not found")
    return record


@router.put("/{recipe_id}")
def put_recipe(
    recipe_id: str,
    recipe: SaveRecipeRequest,
    via_api_key: ViaApiKey = False,
):
    if recipe.id != recipe_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    if _payload_has_stdio_mcp(recipe.payload):
        require_ui_session_for_local_commands(via_api_key)
    try:
        return db.upsert_recipe(
            recipe.model_dump(exclude_none = True, exclude = {"baseUpdatedAt"}),
            recipe.baseUpdatedAt,
        )
    except db.RecipeDeleted:
        raise HTTPException(status_code = 410, detail = "Recipe was deleted")
    except db.RecipeConflict:
        raise HTTPException(
            status_code = 409,
            detail = "This recipe was changed in another window. Reload it to keep editing.",
        )


@router.delete("/{recipe_id}", status_code = 204)
def remove_recipe(recipe_id: str):
    db.delete_recipe(recipe_id)


@router.get("/{recipe_id}/executions")
def get_executions(recipe_id: str):
    return {"executions": db.list_executions(recipe_id)}


@router.put("/{recipe_id}/executions/{execution_id}", status_code = 204)
def put_execution(recipe_id: str, execution_id: str, execution: ExecutionRecord):
    if execution.id != execution_id or execution.recipeId != recipe_id:
        raise HTTPException(status_code = 400, detail = "ID mismatch")
    if not db.upsert_execution(execution.model_dump()):
        raise HTTPException(status_code = 404, detail = "Recipe not found")
