# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from auth.authentication import get_current_subject
from core.inference.skills import (
    MAX_SKILL_ARCHIVE_BYTES,
    SkillError,
    SkillExistsError,
    SkillNotFoundError,
    create_skill,
    delete_skill,
    export_skill,
    import_skill_archive,
    list_skills,
    read_skill_manifest,
    set_all_skills_enabled,
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
    path: Optional[str] = None


class SkillManifest(SkillRecord):
    instructions: str


class SkillEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: StrictBool


class AllSkillsEnabledRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    # null restores the defaults a fresh install has.
    enabled: Optional[StrictBool]


class SkillDraft(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    description: StrictStr
    instructions: StrictStr


class SkillCreateRequest(SkillDraft):
    name: StrictStr


class SkillImportFailure(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    error: str


class SkillImportResult(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    imported: list[SkillRecord]
    failed: list[SkillImportFailure]


def _invalidate_catalog() -> None:
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


@router.put("", response_model = list[SkillRecord])
def update_all_skills_enabled(
    payload: AllSkillsEnabledRequest, current_subject: str = Depends(get_current_subject)
) -> list[dict[str, Any]]:
    try:
        records = set_all_skills_enabled(payload.enabled)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return records


@router.post("", response_model = SkillRecord, status_code = 201)
def create_skill_route(
    payload: SkillCreateRequest, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    try:
        record = create_skill(payload.name, payload.description, payload.instructions)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return record


# 200, not 201: results can be partial.
@router.post("/import", response_model = SkillImportResult)
def import_skill_route(
    file: UploadFile = File(...), current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    data = file.file.read(MAX_SKILL_ARCHIVE_BYTES + 1)
    if len(data) > MAX_SKILL_ARCHIVE_BYTES:
        raise HTTPException(status_code = 413, detail = "Skill archive exceeds the 20 MB limit.")
    try:
        result = import_skill_archive(data)
    except SkillError as exc:
        raise _http_error(exc) from exc
    _invalidate_catalog()
    return result


@router.get("/{name}", response_model = SkillManifest)
def get_skill(name: str, current_subject: str = Depends(get_current_subject)) -> dict[str, Any]:
    try:
        return read_skill_manifest(name)
    except SkillError as exc:
        raise _http_error(exc) from exc


@router.get("/{name}/export", response_class = Response)
def export_skill_route(name: str, current_subject: str = Depends(get_current_subject)) -> Response:
    try:
        skill_name, archive = export_skill(name)
    except SkillError as exc:
        raise _http_error(exc) from exc
    return Response(
        content = archive,
        media_type = "application/zip",
        headers = {
            "Content-Disposition": f'attachment; filename="{skill_name}.zip"',
            "Cache-Control": "no-store",
        },
    )


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
