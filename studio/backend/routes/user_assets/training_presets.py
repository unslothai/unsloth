# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status

from auth.authentication import get_current_subject
from core.user_assets_validation import UserAssetValidationError
from models.user_assets import (
    TrainingPresetCreateRequest,
    TrainingPresetListResponse,
    TrainingPresetRecord,
    TrainingPresetUpdateRequest,
)
from storage import user_assets_db
from storage.user_assets_db import UserAssetStorageError

from .errors import (
    UserAssetsRoute,
    ensure_path_id,
    raise_not_found,
    raise_storage,
    raise_validation,
)


router = APIRouter(route_class = UserAssetsRoute)


def _preset_input(payload: TrainingPresetCreateRequest | TrainingPresetUpdateRequest) -> dict:
    value = payload.model_dump(exclude = {"revision", "createdAt", "updatedAt"})
    return value


@router.get("/training-presets", response_model = TrainingPresetListResponse)
def list_training_presets(current_subject: str = Depends(get_current_subject)):
    return {"presets": user_assets_db.list_training_presets(current_subject)}


@router.post(
    "/training-presets",
    response_model = TrainingPresetRecord,
    status_code = status.HTTP_201_CREATED,
)
def create_training_preset(
    payload: TrainingPresetCreateRequest, current_subject: str = Depends(get_current_subject)
):
    try:
        return user_assets_db.create_training_preset(current_subject, _preset_input(payload))
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)


@router.get("/training-presets/{preset_id}", response_model = TrainingPresetRecord)
def get_training_preset(preset_id: str, current_subject: str = Depends(get_current_subject)):
    try:
        record = user_assets_db.get_training_preset(current_subject, preset_id)
    except UserAssetValidationError as error:
        raise_validation(error)
    if record is None:
        raise_not_found()
    return record


@router.put("/training-presets/{preset_id}", response_model = TrainingPresetRecord)
def update_training_preset(
    preset_id: str,
    payload: TrainingPresetUpdateRequest,
    current_subject: str = Depends(get_current_subject),
):
    ensure_path_id(payload.id, preset_id, "preset id")
    try:
        record = user_assets_db.update_training_preset(
            current_subject,
            preset_id,
            _preset_input(payload),
            payload.revision,
        )
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)
    if record is None:
        raise_not_found()
    return record


@router.delete(
    "/training-presets/{preset_id}",
    status_code = status.HTTP_204_NO_CONTENT,
)
def delete_training_preset(
    preset_id: str,
    revision: int = Query(ge = 1),
    current_subject: str = Depends(get_current_subject),
):
    try:
        deleted = user_assets_db.delete_training_preset(current_subject, preset_id, revision)
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)
    if not deleted:
        raise_not_found()
    return Response(status_code = status.HTTP_204_NO_CONTENT)
