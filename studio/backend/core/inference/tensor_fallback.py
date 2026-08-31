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
    """Run a GGUF load with the tensor-parallel -> layer-split auto-fallback.

    ``attempt_load(tensor_parallel, extra_args)`` performs one load and returns
    True on success; it *raises* on a hard crash (llama-server aborts on some
    archs / older builds), which is treated the same as a False return.

    Tensor mode can be requested by the toggle, by a ``--split-mode tensor`` in
    ``extra_args`` (an allowed shadow flag), or by an inherited
    ``LLAMA_ARG_SPLIT_MODE=tensor`` env (load_model engages it the same way), so
    the retry is keyed on whether tensor mode is actually engaged, and it forces
    ``--split-mode layer`` on the retry so neither leftover extras nor the
    inherited tensor env can relaunch the same failing tensor load. A non-tensor
    load keeps its original contract and propagates exceptions.

    ``cancelled()`` distinguishes a real tensor-start failure from a user
    cancellation: ``attempt_load`` also returns False when the load was
    cancelled, so without this the helper would restart a load the user just
    cancelled.
    """
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

    # The first attempt returned False because the user cancelled, not because
    # tensor mode is unsupported -- do not relaunch the cancelled load.
    if cancelled is not None and cancelled():
        return success

    logger.warning(
        "Tensor-parallel load failed for '%s'; retrying with layer split "
        "(this model may not support tensor parallelism)",
        label,
    )
    # Force --split-mode layer (CLI wins over env) so neither leftover extras nor
    # an inherited LLAMA_ARG_SPLIT_MODE=tensor can re-engage tensor and re-crash
    # the retry; load_model and the child both honor the explicit layer override.
    layer_extras = strip_split_mode_only(extra_args) or []
    return await attempt_load(False, [*layer_extras, "--split-mode", "layer"])
