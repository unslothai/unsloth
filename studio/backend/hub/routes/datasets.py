# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Endpoints mounted at /api/hub/datasets/*."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends, Query, UploadFile

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from hub.schemas.datasets import (
    AiAssistMappingRequest,
    AiAssistMappingResponse,
    CachedDatasetsResponse,
    CheckFormatRequest,
    CheckFormatResponse,
    DeleteCachedDatasetResponse,
    LocalDatasetsResponse,
    UploadDatasetResponse,
)
from hub.schemas.downloads import (
    ActiveDownloadsResponse,
    CancelDatasetDownloadRequest,
    CancelDatasetDownloadResponse,
    DatasetDownloadJobStatus,
    DatasetDownloadStartResponse,
    DownloadProgressResponse,
    DownloadDatasetRequest,
    TransportStatusResponse,
)
from hub.services.datasets import cache_inventory, downloads, formatting, local

router = APIRouter()


@router.post("/upload", response_model = UploadDatasetResponse)
async def upload_dataset(
    file: UploadFile, current_subject: str = Depends(get_current_subject)
) -> UploadDatasetResponse:
    return await local.upload_dataset_response(file)


@router.get("/local", response_model = LocalDatasetsResponse)
def list_local_datasets(
    current_subject: str = Depends(get_current_subject),
) -> LocalDatasetsResponse:
    return local.list_local_datasets_response()


@router.get(
    "/cached",
    response_model = CachedDatasetsResponse,
    response_model_exclude_unset = True,
)
async def list_cached_datasets(current_subject: str = Depends(get_current_subject)):
    return await cache_inventory.list_cached_datasets_response()


@router.delete("/cached", response_model = DeleteCachedDatasetResponse)
async def delete_cached_dataset(
    repo_id: str = Body(..., embed = True),
    cache_path: Optional[str] = Body(None, embed = True),
    current_subject: str = Depends(get_current_subject),
):
    return await cache_inventory.delete_cached_dataset_response(repo_id, cache_path)


@router.get("/download-progress", response_model = DownloadProgressResponse)
async def get_dataset_download_progress(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID, e.g. 'unsloth/LaTeX_OCR'"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_dataset_download_progress_response(
        repo_id,
        expected_bytes = expected_bytes,
        hf_token = hf_token,
    )


@router.post("/download", response_model = DatasetDownloadStartResponse, status_code = 202)
async def download_dataset(
    body: DownloadDatasetRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.download_dataset_response(body, hf_token)


@router.post("/download/cancel", response_model = CancelDatasetDownloadResponse, status_code = 202)
async def cancel_dataset_download(
    body: CancelDatasetDownloadRequest, current_subject: str = Depends(get_current_subject)
):
    return await downloads.cancel_dataset_download_response(body)


@router.get("/download-status", response_model = DatasetDownloadJobStatus)
async def get_dataset_download_status(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_dataset_download_status_response(repo_id)


@router.get("/active-downloads", response_model = ActiveDownloadsResponse)
async def get_active_dataset_downloads(
    repo_id: str = Query("", description = "HuggingFace dataset repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_active_dataset_downloads_response(repo_id)


@router.get("/transport-status", response_model = TransportStatusResponse)
async def get_dataset_transport_status(
    repo_id: str = Query(..., description = "HuggingFace dataset repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_dataset_transport_status_response(repo_id)


@router.post("/check-format", response_model = CheckFormatResponse)
def check_format(
    request: CheckFormatRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return formatting.check_format_response(request, hf_token)


@router.post("/ai-assist-mapping", response_model = AiAssistMappingResponse)
def ai_assist_mapping(
    request: AiAssistMappingRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return formatting.ai_assist_mapping_response(request, hf_token)
