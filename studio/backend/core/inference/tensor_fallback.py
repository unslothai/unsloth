# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tensor-parallel -> layer-split auto-fallback for GGUF loads.

Kept in its own module (no FastAPI / httpx deps) so the orchestration can be
unit-tested with a fake loader, without a GPU or a running llama-server.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from core.inference.llama_server_args import (
    _effective_tensor_parallel,
    strip_split_mode_only,
)

logger = logging.getLogger(__name__)


async def load_with_tensor_fallback(
    attempt_load: Callable[[bool, Optional[list[str]]], Awaitable[bool]],
    *,
    requested_tensor: bool,
    extra_args: Optional[list[str]],
    label: str = "",
    cancelled: Optional[Callable[[], bool]] = None,
) -> bool:
    """Retry a failed tensor-split load as forced --split-mode layer; a user cancel is not retried."""
    tensor_requested = _effective_tensor_parallel(extra_args, requested_tensor)
    try:
        success = await attempt_load(requested_tensor, extra_args)
    except Exception as exc:
        if not tensor_requested:
            raise
        logger.warning("Tensor-parallel load raised for '%s': %s", label, exc)
        success = False

    if success or not tensor_requested:
        return success

    # Cancelled, not unsupported: do not relaunch.
    if cancelled is not None and cancelled():
        return success

    logger.warning(
        "Tensor-parallel load failed for '%s'; retrying with layer split "
        "(this model may not support tensor parallelism)",
        label,
    )
    # Force --split-mode layer (CLI beats env) so LLAMA_ARG_SPLIT_MODE=tensor cannot re-crash it.
    layer_extras = strip_split_mode_only(extra_args) or []
    return await attempt_load(False, [*layer_extras, "--split-mode", "layer"])
