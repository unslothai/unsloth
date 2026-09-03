# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool

from auth.authentication import get_current_subject
from core.inference.skills import SkillError, SkillNotFoundError, list_skills, set_skill_enabled


router = APIRouter()


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    description: str
    source: Literal["agents", "claude"]
    enabled: bool
    valid: bool
    shadowed: bool
    shadowed_by: Optional[Literal["agents", "claude"]] = None
    error: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[str] = None


class SkillEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: StrictBool


@router.get("", response_model = list[SkillRecord])
def get_skills(current_subject: str = Depends(get_current_subject)) -> list[dict[str, Any]]:
    try:
        return list_skills()
    except SkillError as exc:
        raise HTTPException(status_code = 500, detail = "Could not read Agent Skills.") from exc


@router.put("/{name}/enabled", response_model = SkillRecord)
def update_skill_enabled(
    name: str,
    payload: SkillEnabledRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        return set_skill_enabled(name, payload.enabled)
    except SkillNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
