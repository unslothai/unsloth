# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Unsloth upload/request size limits."""

from __future__ import annotations

import os
from typing import Any

UPLOAD_LIMIT_SETTING_KEY = "max_upload_size_mb"
DEFAULT_UPLOAD_LIMIT_MB = 500
MIN_UPLOAD_LIMIT_MB = 1
MAX_UPLOAD_LIMIT_MB = 8192
_BYTES_PER_MB = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 10 * _BYTES_PER_MB
STT_AUDIO_RAW_MAX_BYTES = 25 * _BYTES_PER_MB
VIDEO_INPUT_REFERENCE_MAX_BYTES = 32 * _BYTES_PER_MB
STT_AUDIO_B64_MAX_CHARS = ((STT_AUDIO_RAW_MAX_BYTES + 2) // 3) * 4
STT_AUDIO_JSON_MAX_BYTES = STT_AUDIO_B64_MAX_CHARS + 64 * 1024
VIDEO_INPUT_REFERENCE_B64_MAX_CHARS = ((VIDEO_INPUT_REFERENCE_MAX_BYTES + 2) // 3) * 4
VIDEO_INPUT_REFERENCE_JSON_MAX_BYTES = VIDEO_INPUT_REFERENCE_B64_MAX_CHARS + 64 * 1024

LOCAL_SEED_UPLOAD_MAX_BYTES = 100 * _BYTES_PER_MB
LOCAL_SEED_UPLOAD_MAX_LABEL = "100MB"
UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES = 500 * _BYTES_PER_MB
UNSTRUCTURED_RECIPE_UPLOAD_MAX_LABEL = "500MB"
UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES = 1024 * _BYTES_PER_MB
UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_LABEL = "1GB"


def _coerce_upload_limit_mb(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < MIN_UPLOAD_LIMIT_MB or parsed > MAX_UPLOAD_LIMIT_MB:
        return None
    return parsed


def default_upload_limit_mb() -> int:
    env_value = _coerce_upload_limit_mb(os.environ.get("UNSLOTH_STUDIO_MAX_BODY_MB"))
    return env_value or DEFAULT_UPLOAD_LIMIT_MB


def validate_upload_limit_mb(value: Any) -> int:
    parsed = _coerce_upload_limit_mb(value)
    if parsed is None:
        raise ValueError(
            f"Upload limit must be a whole number from {MIN_UPLOAD_LIMIT_MB} to {MAX_UPLOAD_LIMIT_MB} MB."
        )
    return parsed


def get_upload_limit_mb() -> int:
    """The upload cap, which is installation-wide.

    MaxBodyMiddleware asks for this before anything has authenticated, so it can
    only ever read the owner's database. Stored per account, a managed user could
    raise their own limit in Settings, watch it save, and still be refused at the
    owner's value by the middleware, with no way to tell why. Installation-wide is
    the honest shape for a limit only the install can enforce, so the write is
    gated on the owner and the read goes to the same place the middleware reads.
    """
    try:
        from storage.studio_db import get_install_setting
        stored = get_install_setting(UPLOAD_LIMIT_SETTING_KEY, None)
    except Exception:
        stored = None
    return _coerce_upload_limit_mb(stored) or default_upload_limit_mb()


def set_upload_limit_mb(value: Any) -> int:
    parsed = validate_upload_limit_mb(value)
    from storage.studio_db import upsert_install_settings

    upsert_install_settings({UPLOAD_LIMIT_SETTING_KEY: parsed})
    return parsed


def upload_limit_bytes(limit_mb: int | None = None) -> int:
    return (limit_mb if limit_mb is not None else get_upload_limit_mb()) * _BYTES_PER_MB


def get_upload_limit_bytes() -> int:
    return upload_limit_bytes()


def upload_limit_label(limit_mb: int | None = None) -> str:
    return f"{limit_mb if limit_mb is not None else get_upload_limit_mb()}MB"


def get_upload_limit_label() -> str:
    return upload_limit_label()


def default_request_body_limit_bytes() -> int:
    """Default protected-route body cap for non-upload requests."""

    return default_upload_limit_mb() * _BYTES_PER_MB


def upload_request_limit_bytes(file_limit_bytes: int | None = None) -> int:
    """Request cap for upload routes, including multipart field overhead."""

    return (
        file_limit_bytes if file_limit_bytes is not None else get_upload_limit_bytes()
    ) + MULTIPART_OVERHEAD_BYTES
