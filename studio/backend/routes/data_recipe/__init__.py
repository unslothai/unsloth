# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data Recipe route package."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends

from auth.authentication import get_current_subject

backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from .jobs import router as jobs_router
from .mcp import router as mcp_router
from .seed import router as seed_router
from .validate import router as validate_router

router = APIRouter(dependencies = [Depends(get_current_subject)])
router.include_router(seed_router)
router.include_router(validate_router)
router.include_router(jobs_router)
router.include_router(mcp_router)

__all__ = ["router"]
