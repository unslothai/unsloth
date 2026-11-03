# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted, feature-specific opt-ins for loading ML libraries during startup."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Literal

MlWarmupFeature = Literal["hardware", "text", "datasets", "media", "quantization"]
ML_WARMUP_FEATURES: tuple[MlWarmupFeature, ...] = (
    "hardware",
    "text",
    "datasets",
    "media",
    "quantization",
)
ML_WARMUP_FEATURES_SETTING_KEY = "ml_library_warmup_features"
_HARDWARE_DEPENDENTS = frozenset({"text", "media", "quantization"})


def ml_warmup_disabled_by_env() -> bool:
    """Preserve the existing operator override for every optional warmup stage."""
    return os.environ.get("UNSLOTH_STUDIO_DISABLE_TORCH_WARM") == "1"


def _validate_features(features: Iterable[str]) -> frozenset[str]:
    if isinstance(features, (str, bytes, dict)):
        raise ValueError("Startup warmup features must be a list of supported features.")
    try:
        selected = list(features)
    except TypeError as exc:
        raise ValueError("Startup warmup features must be a list of supported features.") from exc
    if any(
        not isinstance(feature, str) or feature not in ML_WARMUP_FEATURES for feature in selected
    ):
        raise ValueError("Unknown startup warmup feature.")
    return frozenset(selected)


def get_ml_warmup_features() -> frozenset[str]:
    """Missing, invalid or unreadable preferences keep optional imports cold."""
    try:
        from storage.studio_db import get_app_setting

        stored = get_app_setting(ML_WARMUP_FEATURES_SETTING_KEY, None)
        if not isinstance(stored, list):
            return frozenset()
        return _validate_features(stored)
    except Exception:
        return frozenset()


def set_ml_warmup_features(features: Iterable[str]) -> frozenset[str]:
    """Save explicit selections in canonical order; dependencies stay implicit."""
    selected = _validate_features(features)
    from storage.studio_db import upsert_app_settings

    upsert_app_settings(
        {
            ML_WARMUP_FEATURES_SETTING_KEY: [
                feature for feature in ML_WARMUP_FEATURES if feature in selected
            ],
        }
    )
    return selected


def effective_ml_warmup_features(features: Iterable[str]) -> frozenset[str]:
    """Text, media and quantization share hardware setup; datasets do not need it."""
    selected = _validate_features(features)
    if selected & _HARDWARE_DEPENDENTS:
        return selected | {"hardware"}
    return selected


def get_startup_ml_warmup_features() -> frozenset[str]:
    """Resolve the launch's feature selection, respecting the broad disable flag."""
    if ml_warmup_disabled_by_env():
        return frozenset()
    return effective_ml_warmup_features(get_ml_warmup_features())
