# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from auth.authentication import get_current_subject
from core.inference.skills import (
    SkillError,
    SkillExistsError,
    SkillNotFoundError,
    create_skill,
    delete_skill,
    list_skills,
    read_skill_manifest,
    set_skill_enabled,
    update_skill,
)


router = APIRouter()


class SkillRecord(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    description: str
    source: Literal["agents", "claude", "bundled"]
    enabled: bool
    valid: bool
    shadowed: bool
    linked: bool = False
    shadowed_by: Optional[Literal["agents", "claude", "bundled"]] = None
    error: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[str] = None
    # Only on a freshly created skill: where it landed, as the user would name it.
    path: Optional[str] = None


class SkillManifest(SkillRecord):
    instructions: str


class SkillEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: StrictBool


class SkillDraft(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    description: StrictStr
    instructions: StrictStr


class SkillCreateRequest(SkillDraft):
    name: StrictStr


def _invalidate_catalog() -> None:
    # The next inference scan must not serve the snapshot from before this change.
    from routes.inference import _invalidate_agent_skills_cache

    _invalidate_agent_skills_cache()


def _http_error(exc: SkillError) -> HTTPException:
    if isinstance(exc, SkillNotFoundError):
        return HTTPException(status_code = 404, detail = str(exc))
    if isinstance(exc, SkillExistsError):
        return HTTPException(status_code = 409, detail = str(exc))
    return HTTPException(status_code = 400, detail = str(exc))


@router.get("", response_model = list[SkillRecord])
def get_skills(current_subject: str = Depends(get_current_subject)) -> list[dict[str, Any]]:
    try:
        records = list_skills()
    except SkillError as exc:
        raise HTTPException(status_code = 500, detail = "Could not read Agent Skills.") from exc
    # The client just saw the folders; the next inference scan must not serve an older snapshot.
    _invalidate_catalog()
    return records


@router.post("", response_model = SkillRecord, status_code = 201)
def create_skill_route(
    payload: SkillCreateRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        record = create_skill(payload.name, payload.description, payload.instructions)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return record


@router.get("/{name}", response_model = SkillManifest)
def get_skill(name: str, current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    try:
        return read_skill_manifest(name)
    except SkillError as exc:
        raise _http_error(exc) from exc


@router.put("/{name}", response_model = SkillRecord)
def update_skill_route(
    name: str,
    payload: SkillDraft,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        record = update_skill(name, payload.description, payload.instructions)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return record


@router.delete("/{name}", status_code = 204, response_class = Response)
def delete_skill_route(name: str, current_subject: str = Depends(get_current_subject)) -> Response:
    try:
        delete_skill(name)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return Response(status_code = 204)


@router.put("/{name}/enabled", response_model = SkillRecord)
def update_skill_enabled(
    name: str,
    payload: SkillEnabledRequest,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    try:
        updated = set_skill_enabled(name, payload.enabled)
        _invalidate_catalog()
        return updated
    except SkillNotFoundError as exc:
        raise HTTPException(status_code = 404, detail = str(exc)) from exc
    except SkillError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
