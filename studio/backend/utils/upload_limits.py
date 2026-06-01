# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Studio upload/request size limits."""

from __future__ import annotations

import math
import os
from typing import Any

UPLOAD_LIMIT_SETTING_KEY = "max_upload_size_mb"
DEFAULT_UPLOAD_LIMIT_MB = 500
MIN_UPLOAD_LIMIT_MB = 1
MAX_UPLOAD_LIMIT_MB = 8192
_BYTES_PER_MB = 1024 * 1024
_MULTIPART_OVERHEAD_BYTES = 8 * _BYTES_PER_MB
_BASE64_BODY_OVERHEAD_RATIO = 1.4


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
    try:
        from storage.studio_db import get_app_setting

        stored = get_app_setting(UPLOAD_LIMIT_SETTING_KEY, None)
    except Exception:
        stored = None
    return _coerce_upload_limit_mb(stored) or default_upload_limit_mb()


def set_upload_limit_mb(value: Any) -> int:
    parsed = validate_upload_limit_mb(value)
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({UPLOAD_LIMIT_SETTING_KEY: parsed})
    return parsed


def upload_limit_bytes(limit_mb: int | None = None) -> int:
    return (limit_mb if limit_mb is not None else get_upload_limit_mb()) * _BYTES_PER_MB


def get_upload_limit_bytes() -> int:
    return upload_limit_bytes()


def upload_limit_label(limit_mb: int | None = None) -> str:
    return f"{limit_mb if limit_mb is not None else get_upload_limit_mb()}MB"


def get_upload_limit_label() -> str:
    return upload_limit_label()


def request_body_limit_bytes(limit_mb: int | None = None) -> int:
    """Allow transport overhead for the configured max file upload size.

    Multipart uploads add boundaries/headers, and the legacy recipe local-file
    path sends base64 JSON. The user-facing setting remains a file-size limit;
    the raw request-body guard gets enough headroom to carry that file.
    """

    file_limit = upload_limit_bytes(limit_mb)
    multipart_limit = file_limit + _MULTIPART_OVERHEAD_BYTES
    base64_limit = math.ceil(file_limit * _BASE64_BODY_OVERHEAD_RATIO)
    return max(multipart_limit, base64_limit)


def get_request_body_limit_bytes() -> int:
    return request_body_limit_bytes()
