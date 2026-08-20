# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP API for installing and managing portable Agent Skills bundles."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from auth.authentication import get_current_subject
from core.inference.skills import (
    MAX_ARCHIVE_BYTES,
    SkillError,
    delete_skill,
    import_skill_archive,
    list_skills,
    set_skill_enabled,
)


router = APIRouter()


class SkillEnabledRequest(BaseModel):
    enabled: bool


def _bad_skill_request(exc: SkillError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("")
async def get_skills(current_subject: str = Depends(get_current_subject)):
    try:
        skills = await asyncio.to_thread(list_skills)
    except SkillError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"skills": skills}


@router.post("/import")
async def import_skill(
    file: UploadFile = File(...),
    replace: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".zip":
        raise HTTPException(status_code=400, detail="Skill bundle must be a ZIP archive.")

    fd, temporary_name = tempfile.mkstemp(prefix="unsloth-skill-", suffix=".zip")
    total = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_ARCHIVE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Skill archive exceeds the 100 MB upload limit.",
                    )
                output.write(chunk)
        try:
            skill = await asyncio.to_thread(
                import_skill_archive,
                Path(temporary_name),
                replace=replace,
            )
        except SkillError as exc:
            raise _bad_skill_request(exc) from exc
        return {"skill": skill}
    finally:
        await file.close()
        try:
            os.unlink(temporary_name)
        except OSError:
            pass


@router.put("/{name}/enabled")
async def update_skill_enabled(
    name: str,
    request: SkillEnabledRequest,
    current_subject: str = Depends(get_current_subject),
):
    try:
        skill = await asyncio.to_thread(set_skill_enabled, name, request.enabled)
    except SkillError as exc:
        raise _bad_skill_request(exc) from exc
    return {"skill": skill}


@router.delete("/{name}", status_code=204)
async def remove_skill(name: str, current_subject: str = Depends(get_current_subject)):
    try:
        await asyncio.to_thread(delete_skill, name)
    except SkillError as exc:
        raise _bad_skill_request(exc) from exc
