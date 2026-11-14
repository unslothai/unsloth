# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Q4NX export is an AMD NPU feature: only a machine with a supported NPU and FastFlowLM installed
gets it, so NVIDIA and other machines never download the converter."""

from __future__ import annotations

Q4NX_EXPORT_UNAVAILABLE_DETAIL = (
    "Q4NX export needs a supported AMD Ryzen AI NPU with FastFlowLM installed. "
    "Set the NPU up from the Model Hub first."
)


def q4nx_export_allowed() -> bool:
    """Never raises: a failed probe counts as no NPU. A managed account is refused like a machine
    without one, since /api/npu/status hides the NPU from it and this answer must not reveal it."""
    try:
        from hub.services.models import account_access

        if account_access.managed_account():
            return False
        from core.inference.npu_backend import get_npu_backend

        status = get_npu_backend().status()
    except Exception:
        return False
    return bool(status.get("supported")) and bool(status.get("ready"))
