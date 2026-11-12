# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in (off by default) for Q4NX export on a machine without a supported AMD NPU. The
converter is CPU-only, but it is a download nobody without an NPU should get unasked."""

from __future__ import annotations

from typing import Any

Q4NX_EXPORT_SETTING_KEY = "npu_q4nx_export_anywhere"
DEFAULT_Q4NX_EXPORT_ENABLED = False


def get_q4nx_export_setting() -> bool:
    """The persisted owner setting; missing or unreadable is off."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        stored = run_as(OWNER, get_app_setting, Q4NX_EXPORT_SETTING_KEY, None)
    except Exception:
        stored = None
    return stored if isinstance(stored, bool) else DEFAULT_Q4NX_EXPORT_ENABLED


def set_q4nx_export_setting(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Q4NX export must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({Q4NX_EXPORT_SETTING_KEY: value})
    return value


def npu_supported() -> bool:
    """Never raises: a failed probe counts as no NPU."""
    try:
        from core.inference.npu_backend import get_npu_backend
        return bool(get_npu_backend().hardware().get("supported"))
    except Exception:
        return False


def q4nx_export_allowed() -> bool:
    return npu_supported() or get_q4nx_export_setting()


Q4NX_EXPORT_DISABLED_DETAIL = (
    "Q4NX export is off on machines without an AMD NPU. "
    "Turn on Settings > Developer mode > Q4NX export to use it."
)
