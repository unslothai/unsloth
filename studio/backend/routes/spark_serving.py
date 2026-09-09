# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-Spark serving status, so the UI can show both nodes working. One endpoint, which
answers ``{"enabled": false, ...}`` off a paired DGX Spark. Studio and Desktop share this
backend, so both read the same field."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from auth.authentication import get_current_subject
from core.inference import spark_serving

router = APIRouter()


@router.get("/status")
async def spark_serving_status(
    current_subject: str = Depends(get_current_subject),
) -> Dict[str, Any]:
    """Topology, router queue depth and per-backend in-flight counts, peer process state."""
    return spark_serving.status()
