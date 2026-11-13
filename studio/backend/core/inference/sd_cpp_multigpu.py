# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-card sd-cli split via per-module ``--backend``; sized from nvidia-smi so Studio opens no CUDA context."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Callable, Optional, Sequence

# Working memory beside the weights; denoise matches the graph-cut budget's 2 GiB.
DENOISE_HEADROOM_MIB = 2048
ENCODER_HEADROOM_MIB = 1024
VAE_DECODE_HEADROOM_MIB = 1536
# Held back from free MiB: the child's CUDA context (~550 MiB on a T4) plus sd.cpp auto-fit's 512 MiB.
CARD_RESERVE_MIB = 1536


@dataclasses.dataclass(frozen = True)
class SdCppCardSplit:
    primary: int
    secondary: int
    vae_on_secondary: bool
    reason: str
    # the diffusion model's blocks across both cards (sd.cpp's own layer split), each card held to its budget
    layer: bool = False
    budgets_mib: tuple[int, int] = (0, 0)

    def describe(self) -> dict[str, Any]:
        diffusion = f"cuda:{self.primary}" + (f"+cuda:{self.secondary}" if self.layer else "")
        return {
            "chosen": "layer_split" if self.layer else "companion_split",
            "primary": f"cuda:{self.primary}",
            "secondary": f"cuda:{self.secondary}",
            "placement": {
                "diffusion_model": diffusion,
                "text_encoders": f"cuda:{self.secondary}",
                "vae": f"cuda:{self.secondary if self.vae_on_secondary else self.primary}",
            },
            "summary": self.reason,
        }


def plan_sd_cpp_split(
    *,
    diffusion_mib: int,
    text_encoder_mib: int,
    vae_mib: int,
    primary: int,
    primary_limit_mib: Optional[int],
    secondary_limits_mib: dict[int, Optional[int]],
    layer_ok: bool = False,
) -> tuple[Optional[SdCppCardSplit], str]:
    """``(split, reason)``. Splits only when one card cannot hold every module; with ``layer_ok`` blocks are cut across both."""
    if primary_limit_mib is None:
        return None, f"free memory of cuda:{primary} is unknown"
    if text_encoder_mib <= 0:
        return None, "no text encoder to move"
    whole = diffusion_mib + text_encoder_mib + vae_mib + DENOISE_HEADROOM_MIB
    if whole <= primary_limit_mib:
        return (
            None,
            f"not needed: every module fits cuda:{primary} ({whole} of {primary_limit_mib} MiB)",
        )
    limits = {
        int(o): int(v)
        for o, v in secondary_limits_mib.items()
        if v is not None and int(o) != primary
    }
    if not limits:
        return None, "no second card with readable free memory"
    best = max(limits, key = lambda o: (limits[o], -o))
    # each card's room for diffusion blocks; the secondary's is counted after the encoders, the VAE and their working memory
    primary_room = primary_limit_mib - DENOISE_HEADROOM_MIB
    secondary_room = (
        limits[best]
        - text_encoder_mib
        - vae_mib
        - DENOISE_HEADROOM_MIB
        - max(ENCODER_HEADROOM_MIB, VAE_DECODE_HEADROOM_MIB)
    )
    if (
        layer_ok
        and diffusion_mib + DENOISE_HEADROOM_MIB > primary_limit_mib
        and primary_room > 0
        and secondary_room > 0
        and diffusion_mib <= primary_room + secondary_room
    ):
        # no card holds the diffusion model resident: sd.cpp cuts its blocks in proportion to each card's budget
        return (
            SdCppCardSplit(
                primary = int(primary),
                secondary = best,
                vae_on_secondary = vae_mib > 0,
                layer = True,
                budgets_mib = (int(primary_limit_mib), int(limits[best])),
                reason = (
                    f"multi-GPU: diffusion model ({diffusion_mib} MiB) split across cuda:{primary} ({primary_room} MiB "
                    f"room) and cuda:{best} ({secondary_room} MiB beside the text encoders); one card needs {whole} MiB"
                ),
            ),
            "",
        )
    with_vae = text_encoder_mib + vae_mib + max(ENCODER_HEADROOM_MIB, VAE_DECODE_HEADROOM_MIB)
    encoders = text_encoder_mib + ENCODER_HEADROOM_MIB
    options = ((with_vae, True), (encoders, False)) if vae_mib > 0 else ((encoders, False),)
    for need, vae_moves in options:
        if need > limits[best]:
            continue
        on_primary = diffusion_mib + (0 if vae_moves else vae_mib) + DENOISE_HEADROOM_MIB
        moved = "text encoders + VAE" if vae_moves else "text encoders"
        return (
            SdCppCardSplit(
                primary = int(primary),
                secondary = best,
                vae_on_secondary = vae_moves,
                reason = (
                    f"multi-GPU: diffusion model on cuda:{primary} ({on_primary} of {primary_limit_mib} MiB), "
                    f"{moved} on cuda:{best} ({need} of {limits[best]} MiB); one card needs {whole} MiB"
                ),
            ),
            "",
        )
    return None, f"text encoders need {encoders} MiB, cuda:{best} has {limits[best]} MiB"


def file_mib(path: Optional[str]) -> int:
    """MiB of a model file; a ``*.index.json`` counts its shards. 0 when unreadable."""
    if not path:
        return 0
    try:
        if str(path).endswith(".index.json"):
            with open(path, encoding = "utf-8") as f:
                shards = set((json.load(f).get("weight_map") or {}).values())
            base = os.path.dirname(str(path))
            return sum(os.path.getsize(os.path.join(base, s)) for s in shards) >> 20
        return os.path.getsize(str(path)) >> 20
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


TEXT_ENCODER_FIELDS: tuple[str, ...] = ("clip_l", "clip_g", "t5xxl", "llm", "llm_vision", "qwen2vl")


def card_limit_mib(ordinal: int) -> Optional[int]:
    """Free MiB minus max(CARD_RESERVE_MIB, VRAM Budget reserve), via nvidia-smi (no CUDA context here). None when unreadable."""
    from .diffusion_device import smi_free_total_mib
    from .diffusion_memory import _user_budget_reserve_mib

    free, total = smi_free_total_mib(int(ordinal))
    if free is None:
        return None
    reserve = max(CARD_RESERVE_MIB, int(_user_budget_reserve_mib(total or free) or 0))
    return max(0, int(free) - reserve)


def split_for_load(
    files: Any,
    *,
    primary: Optional[int],
    offload: Sequence[str],
    layer_ok: bool = False,
    limit_of: Optional[Callable[[int], Optional[int]]] = None,
    logger: Any = None,
) -> Optional[SdCppCardSplit]:
    """Split for this load, else None. Only without offload flags; balanced / low_vram / cpu_offload stay one card."""
    from .diffusion_multigpu import current_selection, multigpu_all_enabled

    selection = current_selection()
    if selection is None or not multigpu_all_enabled() or not selection.secondaries:
        return None
    if primary is None and selection.automatic:
        primary = selection.primary
    if primary is None or int(primary) != int(selection.primary):
        return None
    limit_of = limit_of or card_limit_mib
    if list(offload):
        if logger is not None:
            logger.info("sd_cpp.multigpu: one card (explicit offload %s)", " ".join(offload))
        return None
    from .diffusion_memory import vram_budget_scope

    limits = vram_budget_scope("image", load = True)(
        lambda: {int(o): limit_of(int(o)) for o in (int(primary), *selection.secondaries)}
    )()
    split, why = plan_sd_cpp_split(
        diffusion_mib = file_mib(getattr(files, "diffusion_model", None)),
        text_encoder_mib = sum(file_mib(getattr(files, f, None)) for f in TEXT_ENCODER_FIELDS),
        vae_mib = file_mib(getattr(files, "vae", None)),
        primary = int(primary),
        primary_limit_mib = limits.get(int(primary)),
        secondary_limits_mib = {int(o): limits.get(int(o)) for o in selection.secondaries},
        layer_ok = layer_ok,
    )
    if logger is not None:
        logger.info(
            "sd_cpp.multigpu: %s", split.reason if split is not None else f"one card ({why})"
        )
    return split
