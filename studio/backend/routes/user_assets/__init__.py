# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import APIRouter

from .legacy_import import router as legacy_import_router
from .recipes import router as recipes_router


router = APIRouter()
router.include_router(legacy_import_router)
router.include_router(recipes_router)

__all__ = ["router"]
