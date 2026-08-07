# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deprecated aliases for dataset routes now served from /api/hub/datasets."""

import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from hub.schemas.datasets import (
    AiAssistMappingRequest as HubAiAssistMappingRequest,
    CheckFormatRequest as HubCheckFormatRequest,
)
from hub.services.datasets import downloads, formatting, local
from models.datasets import (
    AiAssistMappingRequest,
    AiAssistMappingResponse,
    CheckFormatRequest,
    CheckFormatResponse,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)

router = APIRouter()


@router.post("/upload", response_model = UploadDatasetResponse, deprecated = True)
async def upload_dataset(
    file: Optional[UploadFile] = File(None),
    native_path_lease: Optional[str] = Form(None, alias = "nativePathLease"),
    current_subject: str = Depends(get_current_subject),
) -> UploadDatasetResponse:
    return await local.upload_dataset_response(file, native_path_lease)


@router.get("/local", response_model = LocalDatasetsResponse, deprecated = True)
def list_local_datasets(
    current_subject: str = Depends(get_current_subject),
) -> LocalDatasetsResponse:
    result = local.list_local_datasets_response()
    return LocalDatasetsResponse.model_validate(
        {
            "datasets": [
                item.model_dump(exclude = {"source"})
                for item in result.datasets
                if item.source == "recipe"
            ]
        }
    )


@router.get("/download-progress", deprecated = True)
async def get_dataset_download_progress(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID, e.g. 'unsloth/LaTeX_OCR'"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_dataset_download_progress_response(
        repo_id,
        hf_token = hf_token,
    )


@router.post("/check-format", response_model = CheckFormatResponse, deprecated = True)
def check_format(
    request: CheckFormatRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
) -> CheckFormatResponse:
    hub_request = HubCheckFormatRequest.model_validate(request.model_dump(exclude = {"hf_token"}))
    return formatting.check_format_response(
        hub_request,
        request.hf_token or hf_token,
        allow_unlabeled_tier1_fallback = True,
    )


@router.post(
    "/ai-assist-mapping",
    response_model = AiAssistMappingResponse,
    deprecated = True,
)
def ai_assist_mapping(
    request: AiAssistMappingRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
) -> AiAssistMappingResponse:
    hub_request = HubAiAssistMappingRequest.model_validate(request.model_dump(exclude = {"hf_token"}))
    return formatting.ai_assist_mapping_response(
        hub_request,
        request.hf_token or hf_token,
    )
