# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Endpoints mounted at /api/hub/* for the model inventory."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends, Query

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from hub.schemas.downloads import (
    ActiveDownloadsResponse,
    CancelDownloadResponse,
    CancelDownloadRequest,
    DownloadProgressResponse,
    DownloadJobStatus,
    DownloadModelRequest,
    DownloadStartResponse,
    TransportStatusResponse,
)
from hub.schemas.inventory import (
    AddScanFolderRequest,
    BrowseFoldersResponse,
    CachedGgufResponse,
    CachedModelsResponse,
    DeleteCachedModelResponse,
    GgufVariantsResponse,
    HiddenModelsResponse,
    LocalModelListResponse,
    ModelsFolderResponse,
    RecommendedFoldersResponse,
    RemoveScanFolderResponse,
    ScanFolderInfo,
    ScanFoldersResponse,
)
from hub.services.models import (
    cache_inventory,
    deletion,
    downloads,
    folder_browser,
    gguf_variants,
    local_inventory,
)

router = APIRouter()


@router.get("/local", response_model = LocalModelListResponse)
async def list_local_models(
    models_dir: str = Query(
        default = "./models", description = "Directory to scan for local model folders"
    ),
    current_subject: str = Depends(get_current_subject),
):
    return await local_inventory.list_local_models_response(models_dir)


# Plain `def` (not async): synchronous SQLite + filesystem work runs in
# FastAPI's thread-pool instead of blocking the event loop.
@router.get("/scan-folders", response_model = ScanFoldersResponse)
def get_scan_folders(current_subject: str = Depends(get_current_subject)):
    return local_inventory.get_scan_folders_response()


@router.post("/scan-folders", response_model = ScanFolderInfo, status_code = 201)
def add_scan_folder_endpoint(
    body: AddScanFolderRequest, current_subject: str = Depends(get_current_subject)
):
    return local_inventory.add_scan_folder_response(body.path)


@router.delete("/scan-folders/{folder_id}", response_model = RemoveScanFolderResponse)
def remove_scan_folder_endpoint(
    folder_id: int, current_subject: str = Depends(get_current_subject)
):
    return local_inventory.remove_scan_folder_response(folder_id)


@router.get("/recommended-folders", response_model = RecommendedFoldersResponse)
def get_recommended_folders(current_subject: str = Depends(get_current_subject)):
    return folder_browser.get_recommended_folders_response()


@router.get("/browse-folders", response_model = BrowseFoldersResponse)
def browse_folders(
    path: Optional[str] = Query(None),
    show_hidden: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    return folder_browser.browse_folders_response(path, show_hidden)


@router.get("/models-folder", response_model = ModelsFolderResponse)
def get_models_folder(current_subject: str = Depends(get_current_subject)):
    return local_inventory.get_models_folder_response()


@router.get("/gguf-variants", response_model = GgufVariantsResponse)
async def get_gguf_variants(
    repo_id: str = Query(
        ..., description = "HuggingFace repo ID (e.g. 'unsloth/gemma-3-4b-it-GGUF')"
    ),
    prefer_local_cache: bool = Query(False),
    offline: bool = Query(False),
    local_path: Optional[str] = Query(None),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await gguf_variants.get_gguf_variants_response(
        repo_id,
        prefer_local_cache = prefer_local_cache,
        offline = offline,
        local_path = local_path,
        hf_token = hf_token,
    )


@router.post("/download", response_model = DownloadStartResponse, status_code = 202)
async def download_model(
    body: DownloadModelRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.download_model_response(body, hf_token)


@router.post("/download/cancel", response_model = CancelDownloadResponse, status_code = 202)
async def cancel_download_model(
    body: CancelDownloadRequest, current_subject: str = Depends(get_current_subject)
):
    return await downloads.cancel_download_model_response(body)


@router.get("/download-status", response_model = DownloadJobStatus)
async def get_download_status(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    gguf_variant: str = Query("", description = "Quantization variant (empty for safetensors)"),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_download_status_response(repo_id, gguf_variant)


@router.get("/active-downloads", response_model = ActiveDownloadsResponse)
async def get_active_downloads(
    repo_id: str = Query("", description = "HuggingFace repo ID"),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_active_downloads_response(repo_id)


@router.get("/transport-status", response_model = TransportStatusResponse)
async def get_model_transport_status(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    gguf_variant: str = Query("", description = "Quantization variant (empty for safetensors)"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_model_transport_status_response(
        repo_id,
        gguf_variant,
        hf_token,
    )


@router.get(
    "/gguf-download-progress",
    response_model = DownloadProgressResponse,
    response_model_exclude_none = True,
)
async def get_gguf_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    variant: str = Query("", description = "Quantization variant (e.g. UD-TQ1_0)"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_gguf_download_progress_response(
        repo_id,
        variant = variant,
        expected_bytes = expected_bytes,
        hf_token = hf_token,
    )


@router.get("/download-progress", response_model = DownloadProgressResponse)
async def get_download_progress(
    repo_id: str = Query(..., description = "HuggingFace repo ID"),
    expected_bytes: int = Query(0, description = "Expected total download size in bytes"),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await downloads.get_download_progress_response(
        repo_id,
        expected_bytes = expected_bytes,
        hf_token = hf_token,
    )


@router.get("/cached-gguf", response_model = CachedGgufResponse)
async def list_cached_gguf(
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await cache_inventory.list_cached_gguf_response(hf_token)


@router.get("/cached-models", response_model = CachedModelsResponse)
async def list_cached_models(
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await cache_inventory.list_cached_models_response(hf_token)


@router.get("/hidden-models", response_model = HiddenModelsResponse)
async def list_hidden_models(current_subject: str = Depends(get_current_subject)):
    import asyncio

    from routes.models import hidden_model_matchers

    needles, exact_paths = await asyncio.to_thread(hidden_model_matchers)
    return HiddenModelsResponse(needles = needles, exact_paths = exact_paths)


@router.delete(
    "/delete-cached",
    response_model = DeleteCachedModelResponse,
    response_model_exclude_none = True,
)
async def delete_cached_model(
    repo_id: str = Body(...),
    variant: Optional[str] = Body(None),
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    return await deletion.delete_cached_model_response(repo_id, variant, hf_token)
