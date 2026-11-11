# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM run settings an NPU load passes to Lemonade as its ``flm_args`` recipe option.

Bounds mirror lemonade's flm_arg_resolver.cpp at the pinned v2026.44.1, which refuses anything else.
"""

from __future__ import annotations

from typing import Any, Optional

NPU_POWER_MODES = ("powersaver", "balanced", "performance", "turbo")
NPU_IMG_PRE_RESIZE_MAX = 4
# LoadRequest field -> FastFlowLM flag.
NPU_FLM_FIELDS = {
    "npu_power_mode": "--pmode",
    "npu_prefill_chunk_len": "--prefill-chunk-len",
    "npu_img_pre_resize": "--img-pre-resize",
}


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def build_flm_args(
    *,
    power_mode: Optional[str],
    prefill_chunk_len: Optional[int],
    img_pre_resize: Optional[int],
    ctx_size: int,
) -> Optional[str]:
    """The ``flm_args`` string, or None when every setting is the default. Raises ValueError."""
    args: list[str] = []
    if power_mode not in (None, "", "default"):
        if power_mode not in NPU_POWER_MODES:
            raise ValueError(
                "Power mode must be one of: default, " + ", ".join(NPU_POWER_MODES) + "."
            )
        args += ["--pmode", power_mode]
    if prefill_chunk_len is not None:
        chunk = _int(prefill_chunk_len)
        if chunk is None or chunk < 1:
            raise ValueError("Prefill chunk length must be a positive whole number.")
        # Lemonade refuses a chunk longer than the context.
        args += ["--prefill-chunk-len", str(min(chunk, ctx_size))]
    if img_pre_resize is not None:
        level = _int(img_pre_resize)
        if level is None or not 0 <= level <= NPU_IMG_PRE_RESIZE_MAX:
            raise ValueError(
                f"Image pre-resize must be a level from 0 to {NPU_IMG_PRE_RESIZE_MAX}."
            )
        args += ["--img-pre-resize", str(level)]
    return " ".join(args) or None


def flm_args_status(flm_args: Optional[str]) -> dict[str, Any]:
    """The status fields for a resident load's ``flm_args``."""
    tokens = (flm_args or "").split()
    values = dict(zip(tokens[::2], tokens[1::2]))
    status: dict[str, Any] = {}
    for field, flag in NPU_FLM_FIELDS.items():
        value = values.get(flag)
        status[field] = value if value is None or field == "npu_power_mode" else int(value)
    return status
