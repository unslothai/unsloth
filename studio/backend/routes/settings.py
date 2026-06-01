# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from utils.upload_limits import (
    MAX_UPLOAD_LIMIT_MB,
    MIN_UPLOAD_LIMIT_MB,
    default_upload_limit_mb,
    get_upload_limit_mb,
    set_upload_limit_mb,
    upload_limit_bytes,
    upload_limit_label,
)

router = APIRouter()


class UploadLimitPayload(BaseModel):
    max_upload_size_mb: int = Field(
        ..., ge = MIN_UPLOAD_LIMIT_MB, le = MAX_UPLOAD_LIMIT_MB
    )


class UploadLimitResponse(BaseModel):
    max_upload_size_mb: int
    max_upload_size_bytes: int
    max_upload_size_label: str
    default_upload_size_mb: int
    min_upload_size_mb: int = MIN_UPLOAD_LIMIT_MB
    max_allowed_upload_size_mb: int = MAX_UPLOAD_LIMIT_MB


def _upload_limit_response(limit_mb: int) -> UploadLimitResponse:
    return UploadLimitResponse(
        max_upload_size_mb = limit_mb,
        max_upload_size_bytes = upload_limit_bytes(limit_mb),
        max_upload_size_label = upload_limit_label(limit_mb),
        default_upload_size_mb = default_upload_limit_mb(),
    )


@router.get("/upload-limit", response_model = UploadLimitResponse)
def get_upload_limit(
    current_subject: str = Depends(get_current_subject),
) -> UploadLimitResponse:
    return _upload_limit_response(get_upload_limit_mb())


@router.put("/upload-limit", response_model = UploadLimitResponse)
def update_upload_limit(
    payload: UploadLimitPayload,
    current_subject: str = Depends(get_current_subject),
) -> UploadLimitResponse:
    try:
        limit_mb = set_upload_limit_mb(payload.max_upload_size_mb)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    return _upload_limit_response(limit_mb)
