# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist account-specific Model Hub annotations without changing inventory records."""

from typing import Optional

from models.model_annotations import ModelAnnotation

MODEL_ANNOTATIONS_KEY = "model_annotations"


def get_model_annotations() -> dict[str, dict]:
    """Read the account's committed details, surfacing storage errors instead of an empty success."""
    from storage.studio_db import get_app_setting
    return get_app_setting(MODEL_ANNOTATIONS_KEY, {})


def set_model_annotation(model_id: str, annotation: ModelAnnotation) -> Optional[ModelAnnotation]:
    """Atomically replace one row's details; all-empty details remove that preference.

    Inventory ids are opaque keys. They are never used as paths or load ids, and
    other rows survive the shared map update. Notes retain their written spacing.
    """
    from storage.studio_db import upsert_app_setting_map_entry

    present = bool(annotation.display_name or annotation.labels or annotation.notes.strip())
    upsert_app_setting_map_entry(
        MODEL_ANNOTATIONS_KEY, model_id, annotation.model_dump() if present else None
    )
    return annotation if present else None
