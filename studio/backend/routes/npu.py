# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NPU detection, setup and model management.

Loading and chat use the inference routes. All routes here except status are owner-only.
"""

from __future__ import annotations

import asyncio
import json
import threading

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from auth import policy
from auth.authentication import get_current_subject
from core.inference.npu_backend import NpuError, get_npu_backend
from hub.services.models import account_access

router = APIRouter()

_MANAGED_ACCOUNT_STATUS = {
    "supported": False,
    "hardware": {"present": False, "supported": False},
    "runtime_installed": False,
    "runtime_running": False,
    "state": "idle",
    "ready": False,
    "error": None,
    "validation": None,
    "help_url": None,
    "loaded_model": None,
    "context_length": None,
    "loading_model": None,
}


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


class _Download:
    """One pull, run off the request so a dropped progress stream does not stop it."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.finished = False
        self.changed = threading.Condition()

    def publish(self, event: dict) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def follow(self):
        """Every event so far, then each new one, until the pull ends."""
        seen = 0
        while True:
            with self.changed:
                while seen >= len(self.events) and not self.finished:
                    self.changed.wait()
                batch, seen = self.events[seen:], len(self.events)
                finished = self.finished
            yield from batch
            if finished:
                return


_downloads: dict[str, _Download] = {}
_downloads_lock = threading.Lock()


def _start_download(npu, model_id: str) -> _Download:
    """The model's running pull, or a new one; a second request follows the first."""
    with _downloads_lock:
        job = _downloads.get(model_id)
        if job is not None and not job.finished:
            return job
        job = _Download()
        _downloads[model_id] = job

    def _run() -> None:
        try:
            for event in npu.download(model_id):
                job.publish(event)
        except NpuError as exc:
            job.publish({"event": "error", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- reported on the stream, not lost in a thread
            job.publish({"event": "error", "error": f"Downloading {model_id} failed: {exc}"})
        finally:
            with job.changed:
                job.finished = True
                job.changed.notify_all()

    threading.Thread(target = _run, daemon = True, name = "npu-pull").start()
    return job


@router.get("/status")
async def npu_status(current_subject: str = Depends(get_current_subject)):
    """Whether this machine has a supported NPU, and the runtime's state. Never starts anything."""
    if account_access.managed_account():
        return dict(_MANAGED_ACCOUNT_STATUS)
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
    job = _start_download(get_npu_backend(), model_id)

    def _events():
        for event in job.follow():
            yield f"data: {json.dumps(event)}\n\n"

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
