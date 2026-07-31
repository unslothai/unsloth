# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response, status

from auth.authentication import get_current_subject
from core.data_recipe.jobs.manager import JobManager, get_job_manager
from core.user_assets_validation import UserAssetValidationError
from models.user_assets import (
    ExecutionListResponse,
    ExecutionRecord,
    ExecutionUpsertRequest,
    RecipeCreateRequest,
    RecipeListResponse,
    RecipeRecord,
    RecipeUpdateRequest,
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


def _recipe_input(payload: RecipeCreateRequest | RecipeUpdateRequest) -> dict:
    return payload.model_dump(
        exclude = {"revision", "createdAt", "updatedAt"},
        exclude_none = False,
        exclude_unset = isinstance(payload, RecipeUpdateRequest),
    )


@router.get("/recipes", response_model = RecipeListResponse)
def list_recipes(
    cursor: str | None = Query(
        default = None,
        min_length = 1,
        max_length = user_assets_db.MAX_CURSOR_CHARS,
    ),
    limit: int = Query(default = 100, ge = 1, le = 100),
    current_subject: str = Depends(get_current_subject),
):
    try:
        return user_assets_db.list_recipe_summaries(
            current_subject,
            cursor = cursor,
            limit = limit,
        )
    except UserAssetValidationError as error:
        raise_validation(error)


@router.post(
    "/recipes",
    response_model = RecipeRecord,
    status_code = status.HTTP_201_CREATED,
)
def create_recipe(
    payload: RecipeCreateRequest, current_subject: str = Depends(get_current_subject)
):
    try:
        return user_assets_db.create_recipe(current_subject, _recipe_input(payload))
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)


@router.get("/recipes/{recipe_id}", response_model = RecipeRecord)
def get_recipe(recipe_id: str, current_subject: str = Depends(get_current_subject)):
    try:
        record = user_assets_db.get_recipe(current_subject, recipe_id)
    except UserAssetValidationError as error:
        raise_validation(error)
    if record is None:
        raise_not_found()
    return record


@router.put("/recipes/{recipe_id}", response_model = RecipeRecord)
def update_recipe(
    recipe_id: str,
    payload: RecipeUpdateRequest,
    current_subject: str = Depends(get_current_subject),
):
    ensure_path_id(payload.id, recipe_id, "recipe id")
    try:
        record = user_assets_db.update_recipe(
            current_subject,
            recipe_id,
            _recipe_input(payload),
            payload.revision,
        )
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)
    if record is None:
        raise_not_found()
    return record


@router.delete("/recipes/{recipe_id}", status_code = status.HTTP_204_NO_CONTENT)
def delete_recipe(
    recipe_id: str,
    revision: int = Query(ge = 1),
    current_subject: str = Depends(get_current_subject),
):
    try:
        deleted = user_assets_db.delete_recipe(current_subject, recipe_id, revision)
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)
    if not deleted:
        raise_not_found()
    return Response(status_code = status.HTTP_204_NO_CONTENT)


@router.get(
    "/recipes/{recipe_id}/executions",
    response_model = ExecutionListResponse,
)
def list_recipe_executions(
    recipe_id: str,
    cursor: str | None = Query(
        default = None,
        min_length = 1,
        max_length = user_assets_db.MAX_CURSOR_CHARS,
    ),
    limit: int = Query(default = 100, ge = 1, le = 100),
    current_subject: str = Depends(get_current_subject),
):
    try:
        page = user_assets_db.list_recipe_executions(
            current_subject,
            recipe_id,
            cursor = cursor,
            limit = limit,
        )
    except UserAssetValidationError as error:
        raise_validation(error)
    if page is None:
        raise_not_found()
    return page


@router.get("/recipes/{recipe_id}/executions/{execution_id}/dataset")
def get_persisted_execution_dataset(
    recipe_id: str,
    execution_id: str,
    limit: int = Query(default = 20, ge = 1, le = 500),
    offset: int = Query(default = 0, ge = 0),
    current_subject: str = Depends(get_current_subject),
):
    try:
        execution = user_assets_db.get_recipe_execution(current_subject, recipe_id, execution_id)
    except UserAssetValidationError as error:
        raise_validation(error)
    if execution is None:
        raise_not_found()
    artifact_path = execution.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise_not_found()
    result = JobManager.get_dataset_from_artifact(artifact_path, limit = limit, offset = offset)
    if "error" in result:
        raise_not_found()
    return {**result, "limit": limit, "offset": offset}


@router.put(
    "/recipes/{recipe_id}/executions/{execution_id}",
    response_model = ExecutionRecord,
)
def upsert_recipe_execution(
    recipe_id: str,
    execution_id: str,
    payload: ExecutionUpsertRequest,
    current_subject: str = Depends(get_current_subject),
):
    ensure_path_id(payload.id, execution_id, "execution id")
    ensure_path_id(payload.recipeId, recipe_id, "recipe id")
    metadata = payload.model_dump(
        exclude = {"id", "recipeId", "revision", "updatedAt"},
        exclude_none = False,
        exclude_unset = True,
    )
    manager = get_job_manager()
    if payload.artifact_path is not None:
        verified_artifact_path = manager.get_owned_completed_artifact_path(
            payload.jobId or "", current_subject
        )
        if verified_artifact_path != payload.artifact_path:
            # A subsequent job can replace the manager's sole live job after a
            # run has completed but before its browser persists the terminal
            # snapshot. Keep the terminal history in that race, but never
            # trust an artifact path that is no longer server-verifiable.
            if payload.status == "completed":
                existing = user_assets_db.get_recipe_execution(
                    current_subject, recipe_id, execution_id
                )
                if existing is None or existing.get("artifact_path") != payload.artifact_path:
                    metadata.pop("artifact_path", None)
                elif "jobId" in existing:
                    metadata["jobId"] = existing["jobId"]
                else:
                    metadata.pop("jobId", None)
            else:
                raise_validation(
                    UserAssetValidationError(
                        "invalid_execution_artifact",
                        "execution artifact is not owned by this completed job",
                    )
                )
    try:
        record = user_assets_db.upsert_recipe_execution(
            current_subject,
            recipe_id,
            execution_id,
            metadata,
            payload.revision,
        )
    except UserAssetValidationError as error:
        raise_validation(error)
    except UserAssetStorageError as error:
        raise_storage(error)
    if record is None:
        raise_not_found()
    if payload.artifact_path is not None and record.get("artifact_path") == payload.artifact_path:
        record_job_id = record.get("jobId")
        if isinstance(record_job_id, str) and record_job_id:
            manager.release_completed_artifact_path(
                record_job_id, current_subject, payload.artifact_path
            )
    return record
