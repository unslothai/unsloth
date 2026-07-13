# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import get_current_subject
from core.user_assets_validation import UserAssetValidationError
from models.user_assets import BootstrapResponse, LegacyImportRequest, LegacyImportResponse
from storage import user_assets_db

from .errors import UserAssetsRoute, raise_validation


DEFAULT_LEGACY_SOURCE = "recipe-indexeddb-v1"

router = APIRouter(route_class = UserAssetsRoute)


@router.get("/bootstrap", response_model = BootstrapResponse)
def bootstrap(current_subject: str = Depends(get_current_subject)):
    ledger = user_assets_db.list_legacy_imports(current_subject, DEFAULT_LEGACY_SOURCE)
    return {
        "subject": current_subject,
        "importLedger": {"source": DEFAULT_LEGACY_SOURCE, **ledger},
    }


@router.post("/legacy-import", response_model = LegacyImportResponse)
def import_legacy_assets(
    payload: LegacyImportRequest, current_subject: str = Depends(get_current_subject)
):
    if payload.confirmSubject != current_subject:
        raise HTTPException(
            status_code = 422,
            detail = {
                "code": "confirmation_mismatch",
                "message": "confirmSubject must match the authenticated account",
            },
        )
    if payload.source != DEFAULT_LEGACY_SOURCE:
        raise HTTPException(
            status_code = 422,
            detail = {
                "code": "unsupported_import_source",
                "message": "The legacy import source is not supported",
            },
        )
    try:
        return user_assets_db.import_legacy_assets(
            current_subject,
            payload.source,
            payload.recipes,
            payload.executions,
        )
    except UserAssetValidationError as error:
        raise_validation(error)
