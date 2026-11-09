# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two CUDA cards for the native (sd.cpp) engine: the load's multi-card selection (diffusion_multigpu) decides, sd-cli
places through its own per-module ``--backend`` assignment (``diffusion=CUDA0,te=CUDA1,vae=CUDA0``). Every card is
sized from nvidia-smi, so the Studio process never opens a CUDA context on a card the child owns. A selection of one
card, an explicit offload mode, an off-torch card or ``UNSLOTH_DIFFUSION_MULTIGPU_ALL=0`` keeps the one-card argv."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Callable, Optional, Sequence

# Working memory beside the weights: a card running the diffusion model (sd.cpp compute buffers at 1024 px plus the
# child's CUDA context and cuBLAS workspaces, the same 2 GiB the graph-cut budget keeps), a card running only the text
# encoders (one forward per prompt), and a VAE decode.
DENOISE_HEADROOM_MIB = 2048
ENCODER_HEADROOM_MIB = 1024
VAE_DECODE_HEADROOM_MIB = 1536


@dataclasses.dataclass(frozen = True)
class SdCppCardSplit:
    """Where sd-cli runs each module: the diffusion model on ``primary``, the text encoders on ``secondary``, the VAE
    on ``secondary`` only when ``vae_on_secondary``."""

    primary: int
    secondary: int
    vae_on_secondary: bool
    reason: str

    def describe(self) -> dict[str, Any]:
        return {
            "chosen": "companion_split",
            "primary": f"cuda:{self.primary}",
            "secondary": f"cuda:{self.secondary}",
            "placement": {
                "diffusion_model": f"cuda:{self.primary}",
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
) -> tuple[Optional[SdCppCardSplit], str]:
    """Pure decision: ``(split, reason)`` or ``(None, why one card)``. A split only when one card cannot hold every
    module resident, where sd.cpp's auto-fit keeps the weights that do not fit in host RAM and streams them: the text
    encoders (and the VAE, when that card also has a decode's room) then run on the roomiest other card, so the
    primary holds as much of the diffusion model as it can (auto-fit still places what remains). Limits are each
    card's free memory after its reserve."""
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
    with_vae = text_encoder_mib + vae_mib + max(ENCODER_HEADROOM_MIB, VAE_DECODE_HEADROOM_MIB)
    encoders = text_encoder_mib + ENCODER_HEADROOM_MIB
    for need, vae_moves in ((with_vae, True), (encoders, False)):
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
    """A card's free memory after the larger of 1 GiB and the user's VRAM Budget reserve, read from nvidia-smi (no CUDA
    context in this process). None when unreadable."""
    from .diffusion_device import smi_free_total_mib
    from .diffusion_memory import _user_budget_reserve_mib

    free, total = smi_free_total_mib(int(ordinal))
    if free is None:
        return None
    reserve = max(1024, int(_user_budget_reserve_mib(total or free) or 0))
    return max(0, int(free) - reserve)


def split_for_load(
    files: Any,
    *,
    primary: Optional[int],
    offload: Sequence[str],
    limit_of: Optional[Callable[[int], Optional[int]]] = None,
    logger: Any = None,
) -> Optional[SdCppCardSplit]:
    """The split for this load under the current multi-card selection, else None (one card). Only for the default
    memory policy (no offload flags: sd.cpp auto-fit); balanced / low_vram / cpu_offload keep their one-card meaning."""
    from .diffusion_multigpu import current_selection, multigpu_all_enabled

    selection = current_selection()
    if selection is None or not multigpu_all_enabled() or primary is None:
        return None
    if int(primary) != int(selection.primary) or not selection.secondaries:
        return None
    limit_of = limit_of or card_limit_mib
    if list(offload):
        if logger is not None:
            logger.info("sd_cpp.multigpu: one card (explicit offload %s)", " ".join(offload))
        return None
    split, why = plan_sd_cpp_split(
        diffusion_mib = file_mib(getattr(files, "diffusion_model", None)),
        text_encoder_mib = sum(file_mib(getattr(files, f, None)) for f in TEXT_ENCODER_FIELDS),
        vae_mib = file_mib(getattr(files, "vae", None)),
        primary = int(primary),
        primary_limit_mib = limit_of(int(primary)),
        secondary_limits_mib = {int(o): limit_of(int(o)) for o in selection.secondaries},
    )
    if logger is not None:
        logger.info(
            "sd_cpp.multigpu: %s", split.reason if split is not None else f"one card ({why})"
        )
    return split
