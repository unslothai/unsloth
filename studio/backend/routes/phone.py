# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import socket
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from loggers import get_logger
from auth.authentication import (
    create_phone_token,
    get_current_subject,
    get_phone_viewer,
)
from routes.training import build_training_status
from utils.hardware import get_gpu_utilization
from core.training import get_training_backend
from utils.utils import log_and_http_error

router = APIRouter()
logger = get_logger(__name__)


class PhoneShareResponse(BaseModel):
    page_url: str
    expires_at: str


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _lan_ip() -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return None
    finally:
        sock.close()


def _phone_host() -> Optional[str]:
    bind_host = os.environ.get("UNSLOTH_BIND_HOST", "127.0.0.1")
    if bind_host in _LOOPBACK_HOSTS:
        return None
    if bind_host in ("0.0.0.0", "::"):
        return _lan_ip()
    return bind_host


def _require_run_active(viewer: Tuple[str, str]) -> None:
    # Exact match only; an empty run_id is not a wildcard for future runs.
    run_id = viewer[1]
    backend = get_training_backend()
    if run_id != (getattr(backend, "current_job_id", "") or ""):
        raise HTTPException(
            status_code = 410,
            detail = "This phone link's training run has ended.",
        )


@router.post("/share", response_model = PhoneShareResponse)
async def share_to_phone(request: Request, current_subject: str = Depends(get_current_subject)):
    try:
        scheme = request.url.scheme or "http"
        port = request.url.port or (443 if scheme == "https" else 80)

        host = _phone_host()
        if not host:
            # Loopback bind: hand the UI a code + relaunch command, not a raw error.
            raise HTTPException(
                status_code = 409,
                detail = {
                    "code": "loopback_only",
                    "command": f"unsloth studio -H 0.0.0.0 -p {port}",
                },
            )

        backend = get_training_backend()
        run_id = getattr(backend, "current_job_id", "") or ""
        token, expires_at = create_phone_token(current_subject, run_id)

        # Fragment, not path — keeps the token out of server logs.
        page_url = f"{scheme}://{host}:{port}/m#{token}"

        return PhoneShareResponse(
            page_url = page_url,
            expires_at = expires_at.isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to create phone share link",
            event = "phone.share_failed",
            log = logger,
        )


@router.get("/status")
async def phone_status(viewer: Tuple[str, str] = Depends(get_phone_viewer)):
    try:
        _require_run_active(viewer)
        return build_training_status()
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get training status",
            event = "phone.status_failed",
            log = logger,
        )


@router.get("/hardware")
async def phone_hardware(viewer: Tuple[str, str] = Depends(get_phone_viewer)):
    try:
        _require_run_active(viewer)
        return get_gpu_utilization()
    except HTTPException:
        raise
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get hardware utilization",
            event = "phone.hardware_failed",
            log = logger,
        )
