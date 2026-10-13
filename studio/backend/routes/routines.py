# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated API for local, scheduled Studio workflows."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auth.authentication import (
    authenticated_via_api_key,
    get_current_subject,
    require_ui_session_for_local_commands,
)
from core.data_recipe.service import recipe_has_stdio_mcp
from models.data_recipe import RecipePayload
from storage import routines_db as db

router = APIRouter(dependencies = [Depends(get_current_subject)])
ViaApiKey = Annotated[bool, Depends(authenticated_via_api_key)]


class RoutineStep(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    kind: Literal["recipe", "folder_sync"]
    recipe: RecipePayload | None = None
    folder_id: str | None = Field(default = None, min_length = 1, max_length = 200)
    label: str = Field(min_length = 1, max_length = 120)

    @model_validator(mode = "after")
    def valid_input(self):
        if self.kind == "recipe" and (self.recipe is None or not self.recipe.recipe.get("columns")):
            raise ValueError("A recipe step needs a configured Data Recipe")
        if self.kind == "folder_sync" and not self.folder_id:
            raise ValueError("A folder sync step needs a linked folder")
        if (self.kind == "recipe" and self.folder_id) or (self.kind == "folder_sync" and self.recipe):
            raise ValueError("Step input does not match its kind")
        return self


class RoutineInput(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    name: str = Field(min_length = 1, max_length = 120)
    steps: list[RoutineStep] = Field(min_length = 1, max_length = 10)
    interval_minutes: int = Field(ge = 1, le = 10080)
    enabled: bool = True


class EnabledInput(BaseModel):
    enabled: bool


def _validate_steps(payload: RoutineInput, via_api_key: bool) -> None:
    if not payload.name.strip():
        raise HTTPException(status_code = 422, detail = "Routine name cannot be blank")
    for step in payload.steps:
        if step.kind == "recipe" and step.recipe is not None:
            if recipe_has_stdio_mcp(step.recipe.recipe):
                require_ui_session_for_local_commands(via_api_key)
        elif step.kind == "folder_sync":
            from core.rag.folder_sync import get_folder
            if get_folder(step.folder_id) is None:
                raise HTTPException(status_code = 404, detail = "Linked folder not found")


@router.get("")
def list_routines():
    return {"routines": db.list_routines()}


@router.post("")
def create_routine(payload: RoutineInput, via_api_key: ViaApiKey = False):
    _validate_steps(payload, via_api_key)
    return db.save_routine(
        routine_id = None, name = payload.name.strip(),
        steps = [step.model_dump(exclude_none = True) for step in payload.steps],
        interval_minutes = payload.interval_minutes, enabled = payload.enabled,
    )


@router.put("/{routine_id}")
def update_routine(routine_id: str, payload: RoutineInput, via_api_key: ViaApiKey = False):
    _validate_steps(payload, via_api_key)
    try:
        return db.save_routine(
            routine_id = routine_id, name = payload.name.strip(),
            steps = [step.model_dump(exclude_none = True) for step in payload.steps],
            interval_minutes = payload.interval_minutes, enabled = payload.enabled,
        )
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = "Routine not found") from exc


@router.patch("/{routine_id}/enabled")
def change_enabled(routine_id: str, payload: EnabledInput):
    routine = db.set_enabled(routine_id, payload.enabled)
    if routine is None:
        raise HTTPException(status_code = 404, detail = "Routine not found")
    return routine


@router.post("/{routine_id}/run")
def run_routine(routine_id: str, via_api_key: ViaApiKey = False):
    routine = db.get_routine(routine_id)
    if routine is None:
        raise HTTPException(status_code = 404, detail = "Routine not found")
    if any(
        step["kind"] == "recipe" and recipe_has_stdio_mcp(step["recipe"]["recipe"])
        for step in routine["steps"]
    ):
        require_ui_session_for_local_commands(via_api_key)
    try:
        return {"run_id": db.run_now(routine_id)}
    except KeyError as exc:
        raise HTTPException(status_code = 404, detail = "Routine not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.get("/{routine_id}/runs")
def list_routine_runs(routine_id: str):
    if db.get_routine(routine_id) is None:
        raise HTTPException(status_code = 404, detail = "Routine not found")
    return {"runs": db.list_runs(routine_id)}
