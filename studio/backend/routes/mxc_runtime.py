# SPDX-License-Identifier: AGPL-3.0-only
"""Authenticated owner controls for Studio's fixed Windows MXC runtime package."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import sys

from fastapi import APIRouter, Depends, HTTPException

from auth import policy
from auth.authentication import get_current_subject


router = APIRouter(dependencies=[Depends(get_current_subject)])


async def _require_owner() -> None:
    await policy.require_owner()


def _runtime_module():
    if sys.platform != "win32":
        raise HTTPException(status_code=409, detail="MXC runtime lifecycle is Windows-only")
    from core.inference import mxc_runtime

    return mxc_runtime


def _status_payload(status) -> dict:
    payload = asdict(status)
    payload["state"] = status.state.value
    return payload


@router.get("/status")
def get_runtime_status() -> dict:
    runtime = _runtime_module()
    return _status_payload(runtime.runtime_status())


@router.post("/{operation}", dependencies=[Depends(_require_owner)])
async def mutate_runtime(operation: str) -> dict:
    runtime = _runtime_module()
    operations = {
        "install": runtime.install_approved_runtime,
        "repair": runtime.repair_runtime,
        "update": runtime.update_runtime,
        "rollback": runtime.rollback_runtime,
        "gc": runtime.garbage_collect,
        "uninstall": runtime.uninstall_runtime,
    }
    action = operations.get(operation)
    if action is None:
        raise HTTPException(status_code=404, detail="unknown MXC runtime operation")
    try:
        result = await asyncio.to_thread(action)
    except runtime.MxcRuntimeUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if operation == "gc":
        return {"state": "ready", "retired": result}
    if operation == "uninstall":
        return {"state": "not_installed" if result else "retirement_pending"}
    return {
        "state": "ready",
        "generation": result.generation,
        "runnerSha256": result.runner_sha256,
    }
