# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation endpoints for data recipe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from core.data_recipe.service import (
    build_config_builder,
    create_data_designer,
    validate_recipe,
)
from models.data_recipe import RecipePayload, ValidateError, ValidateResponse

router = APIRouter()


def _collect_validation_errors(recipe: dict[str, Any]) -> list[ValidateError]:
    try:
        from data_designer.engine.compiler import (
            _add_internal_row_id_column_if_needed,
            _get_allowed_references,
            _resolve_and_add_seed_columns,
        )
        from data_designer.engine.validation import (
            ViolationLevel,
            validate_data_designer_config,
        )
    except ImportError:
        return []

    try:
        builder = build_config_builder(recipe)
        designer = create_data_designer(recipe)
        resource_provider = designer._create_resource_provider(  # type: ignore[attr-defined]
            "validate-configuration",
            builder,
        )
        config = builder.build()
        _resolve_and_add_seed_columns(config, resource_provider.seed_reader)
        _add_internal_row_id_column_if_needed(config)
        violations = validate_data_designer_config(
            columns = config.columns,
            processor_configs = config.processors or [],
            allowed_references = _get_allowed_references(config),
        )
    except (TypeError, ValueError, AttributeError):
        return []

    errors: list[ValidateError] = []
    for violation in violations:
        if violation.level != ViolationLevel.ERROR:
            continue
        code = getattr(violation.type, "value", None)
        path = violation.column if violation.column else None
        message = str(violation.message).strip() or "Validation failed."
        errors.append(
            ValidateError(
                message = message,
                path = path,
                code = code,
            )
        )
    return errors


def _patch_local_providers(recipe: dict[str, Any]) -> None:
    """Strip is_local and fill a dummy endpoint so validation doesn't choke.

    Uses a strict `is True` check to match _inject_local_providers in
    jobs.py - malformed payloads with truthy but non-boolean is_local
    values should not be treated as local.
    """
    for provider in recipe.get("model_providers", []):
        if not isinstance(provider, dict):
            continue
        if provider.pop("is_local", None) is True:
            provider["endpoint"] = "http://127.0.0.1"


@router.post("/validate", response_model = ValidateResponse)
def validate(payload: RecipePayload) -> ValidateResponse:
    recipe = payload.recipe
    if not recipe.get("columns"):
        return ValidateResponse(
            valid = False,
            errors = [ValidateError(message = "Recipe must include columns.")],
        )

    _patch_local_providers(recipe)

    try:
        validate_recipe(recipe)
    except RuntimeError as exc:
        raise HTTPException(status_code = 503, detail = str(exc)) from exc
    except Exception as exc:
        detail = str(exc).strip() or "Validation failed."
        parsed_errors = _collect_validation_errors(recipe)
        return ValidateResponse(
            valid = False,
            errors = parsed_errors or [ValidateError(message = detail)],
            raw_detail = detail,
        )

    return ValidateResponse(valid = True)
