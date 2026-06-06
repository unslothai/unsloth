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
    resolve_tensor_parallel,
    strip_shadowing_flags,
)

logger = logging.getLogger(__name__)


async def load_with_tensor_fallback(
    attempt_load: Callable[[bool, Optional[list[str]]], Awaitable[bool]],
    *,
    requested_tensor: bool,
    extra_args: Optional[list[str]],
    label: str = "",
) -> bool:
    """Run a GGUF load with the tensor-parallel -> layer-split auto-fallback.

    ``attempt_load(tensor_parallel, extra_args)`` performs one load and returns
    True on success; it *raises* on a hard crash (llama-server aborts on some
    archs / older builds), which is treated the same as a False return.

    Tensor mode can be requested by the toggle or by a ``--split-mode tensor``
    in ``extra_args`` (an allowed shadow flag), so the retry is keyed on whether
    tensor mode is actually engaged, and it strips ``--split-mode`` from the
    extras so the layer retry can't relaunch the same failing tensor load. A
    non-tensor load keeps its original contract and propagates exceptions.
    """
    tensor_requested = resolve_tensor_parallel(extra_args, requested_tensor)
    try:
        success = await attempt_load(requested_tensor, extra_args)
    except Exception as exc:
        if not tensor_requested:
            raise
        logger.warning("Tensor-parallel load raised for '%s': %s", label, exc)
        success = False

    if success or not tensor_requested:
        return success

    logger.warning(
        "Tensor-parallel load failed for '%s'; retrying with layer split "
        "(this model may not support tensor parallelism)",
        label,
    )
    layer_extra_args = (
        strip_shadowing_flags(
            extra_args,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = True,
        )
        if extra_args
        else extra_args
    )
    return await attempt_load(False, layer_extra_args)
