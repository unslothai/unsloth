# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only "watch training on your phone" routes for the QR dashboard."""

import socket
from typing import Tuple

from fastapi import APIRouter, Depends, Request
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


def _lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


@router.post("/share", response_model = PhoneShareResponse)
async def share_to_phone(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    try:
        backend = get_training_backend()
        run_id = getattr(backend, "current_job_id", "") or ""
        token, expires_at = create_phone_token(current_subject, run_id)

        scheme = request.url.scheme or "http"
        port = request.url.port or (443 if scheme == "https" else 80)
        page_url = f"{scheme}://{_lan_ip()}:{port}/m/{token}"

        return PhoneShareResponse(
            page_url = page_url,
            expires_at = expires_at.isoformat(),
        )
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
        return build_training_status()
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
        return get_gpu_utilization()
    except Exception as e:
        raise log_and_http_error(
            e,
            500,
            "Failed to get hardware utilization",
            event = "phone.hardware_failed",
            log = logger,
        )
