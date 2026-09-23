# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU detection, setup and model management.

Loading and chat use the inference routes. All routes here except status are owner-only.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from auth import policy
from auth.authentication import get_current_subject
from core.inference.npu_backend import NpuError, get_npu_backend

router = APIRouter()


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


@router.get("/status")
async def npu_status(current_subject: str = Depends(get_current_subject)):
    """Whether this machine has a supported NPU, and the runtime's state. Never starts anything."""
    return await asyncio.to_thread(get_npu_backend().status)


@router.post("/enable", dependencies = [Depends(_require_installation_owner)])
async def enable_npu():
    """Install and start Lemonade, install FastFlowLM and validate the NPU. Idempotent."""
    try:
        return await asyncio.to_thread(get_npu_backend().enable)
    except NpuError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None


@router.get("/models", dependencies = [Depends(_require_installation_owner)])
async def list_npu_models():
    try:
        models = await asyncio.to_thread(get_npu_backend().catalog)
    except NpuError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    return {"models": [model.to_json() for model in models]}


@router.post("/models/{model_id}/download", dependencies = [Depends(_require_installation_owner)])
async def download_npu_model(model_id: str):
    """Stream download progress through a final complete or error event.

    Disconnecting stops progress updates, not the download.
    """
    npu = get_npu_backend()

    def _events():
        try:
            for event in npu.download(model_id):
                yield f"data: {json.dumps(event)}\n\n"
        except NpuError as exc:
            yield f"data: {json.dumps({'event': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        iterate_in_threadpool(_events()),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/models/{model_id}", dependencies = [Depends(_require_installation_owner)])
async def delete_npu_model(model_id: str):
    try:
        await asyncio.to_thread(get_npu_backend().delete, model_id)
    except NpuError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from None
    return {"status": "deleted", "model": model_id}
