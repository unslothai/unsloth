# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Owner-controlled installation of optional local serving engines."""

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from auth import policy
from auth.authentication import get_current_subject
from core.inference import engine_install

Engine = Literal["vllm", "sglang"]
router = APIRouter(dependencies = [Depends(get_current_subject)])


def _reap_crashed_engine() -> None:
    """A crashed engine holds its lease until reaped, blocking repair and removal."""
    from routes.inference import _peek_inference_backend

    backend = _peek_inference_backend()
    if getattr(backend, "_managed_engine", None) is not None:
        backend.reap_dead_managed_engine()


@router.get("")
async def list_engines():
    def rows():
        _reap_crashed_engine()
        return [engine_install.status(name) for name in engine_install.PROFILES]

    return await asyncio.to_thread(rows)


@router.post("/{engine}/install", dependencies = [Depends(policy.require_owner)])
async def install(engine: Engine):
    try:
        await asyncio.to_thread(_reap_crashed_engine)
        return await asyncio.to_thread(engine_install.start_install, engine)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.post("/{engine}/cancel", dependencies = [Depends(policy.require_owner)])
async def cancel(engine: Engine):
    return await asyncio.to_thread(engine_install.cancel_install, engine)


@router.delete("/{engine}", dependencies = [Depends(policy.require_owner)])
async def remove(engine: Engine):
    try:
        await asyncio.to_thread(_reap_crashed_engine)
        return await asyncio.to_thread(engine_install.remove, engine)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code = 409,
            detail = "Unload the engine and wait for installation to finish before removing it.",
        ) from exc


@router.post("/{engine}/rollback", dependencies = [Depends(policy.require_owner)])
async def rollback(engine: Engine):
    try:
        await asyncio.to_thread(_reap_crashed_engine)
        return await asyncio.to_thread(engine_install.rollback, engine)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.delete("/wsl/environment", dependencies = [Depends(policy.require_owner)])
async def remove_wsl_environment():
    """Windows only: delete Studio's private WSL distro with every engine inside it."""
    try:
        await asyncio.to_thread(_reap_crashed_engine)
        return await asyncio.to_thread(engine_install.remove_wsl_environment)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
