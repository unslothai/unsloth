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

from .jobs import download_router as jobs_download_router
from .jobs import router as jobs_router
from .mcp import router as mcp_router
from .seed import router as seed_router
from .validate import router as validate_router

_header_authenticated = APIRouter(dependencies = [Depends(get_current_subject)])
_header_authenticated.include_router(seed_router)
_header_authenticated.include_router(validate_router)
_header_authenticated.include_router(jobs_router)
_header_authenticated.include_router(mcp_router)

router = APIRouter()
# Kept out of the group above: the download link is fetched without a header, so it brings its
# own guard.
router.include_router(jobs_download_router)
router.include_router(_header_authenticated)

__all__ = ["router"]
