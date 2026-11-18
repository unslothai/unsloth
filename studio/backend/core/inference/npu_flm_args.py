# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FastFlowLM run settings an NPU load passes to Lemonade as its ``flm_args`` recipe option.

Bounds mirror lemonade's flm_arg_resolver.cpp at the pinned v2026.44.1, which refuses anything else.
"""

from __future__ import annotations

from typing import Any, Optional

NPU_POWER_MODES = ("powersaver", "balanced", "performance", "turbo")
NPU_IMG_PRE_RESIZE_MAX = 4
NPU_FLM_FIELDS = {
    "npu_power_mode": "--pmode",
    "npu_prefill_chunk_len": "--prefill-chunk-len",
    "npu_img_pre_resize": "--img-pre-resize",
}


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def build_flm_args(
    *, power_mode: Optional[str], prefill_chunk_len: Optional[int], img_pre_resize: Optional[int]
) -> Optional[str]:
    """The ``flm_args`` string, or None when every setting is the default. Raises ValueError.
    The prefill chunk is held to the context by the load (clamp_prefill_chunk)."""
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
        args += ["--prefill-chunk-len", str(chunk)]
    if img_pre_resize is not None:
        level = _int(img_pre_resize)
        if level is None or not 0 <= level <= NPU_IMG_PRE_RESIZE_MAX:
            raise ValueError(
                f"Image pre-resize must be a level from 0 to {NPU_IMG_PRE_RESIZE_MAX}."
            )
        args += ["--img-pre-resize", str(level)]
    return " ".join(args) or None


def clamp_prefill_chunk(flm_args: Optional[str], ctx_size: Optional[int]) -> Optional[str]:
    """``flm_args`` with its prefill chunk held to ``ctx_size``: Lemonade refuses a longer one."""
    if not flm_args or not ctx_size:
        return flm_args or None
    tokens = flm_args.split()
    for i, token in enumerate(tokens[:-1]):
        if token == "--prefill-chunk-len" and tokens[i + 1].isdigit():
            tokens[i + 1] = str(min(int(tokens[i + 1]), ctx_size))
    return " ".join(tokens) or None


def _flag_groups(args: str) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for token in args.split():
        if token.startswith("-") and not token.lstrip("-").replace(".", "", 1).isdigit():
            groups.append((token.split("=", 1)[0], [token]))
        elif groups:
            groups[-1][1].append(token)
    return groups


def merge_flm_args(saved: Optional[str], studio: Optional[str]) -> Optional[str]:
    """``studio`` on top of the model's saved ``flm_args``. A request's ``flm_args`` replaces the saved
    and architecture ones in Lemonade, so each saved flag Unsloth Studio does not set is carried over."""
    if not studio:
        return None
    ours = {flag for flag, _ in _flag_groups(studio)}
    kept = [" ".join(tokens) for flag, tokens in _flag_groups(saved or "") if flag not in ours]
    return " ".join([studio, *kept])


def flm_args_status(flm_args: Optional[str]) -> dict[str, Any]:
    """The status fields for a resident load's ``flm_args``."""
    tokens = (flm_args or "").split()
    values = dict(zip(tokens[::2], tokens[1::2]))
    status: dict[str, Any] = {}
    for field, flag in NPU_FLM_FIELDS.items():
        value = values.get(flag)
        status[field] = value if value is None or field == "npu_power_mode" else int(value)
    return status
