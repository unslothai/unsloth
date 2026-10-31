# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic llama-server split mode (layer vs tensor) and the MTP draft depth per split mode.

Explicit choices (toggle on, --split-mode / -sm / --tensor-split, LLAMA_ARG_SPLIT_MODE / LLAMA_ARG_TENSOR_SPLIT)
always win. The UI always sends tensor_parallel = false, so False means "not chosen".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

AUTO_TENSOR_SPLIT_DEFAULT = True
AUTO_TENSOR_SPLIT_ENV = "UNSLOTH_STUDIO_AUTO_TENSOR_SPLIT"
# Widen a dense model that fits one GPU onto two idle ones: 2x T4 Qwen3.5-9B Q8_0 decode 18.8 -> 41.5 tok/s.
TP_WIDEN_FITS_ONE_GPU = True
# The prebuilt's internal AllReduce (no NCCL) covers exactly 2 GPUs; explicit tensor requests are not capped.
AUTO_TENSOR_MAX_GPUS = 2


@dataclass(frozen = True)
class SplitPolicyThresholds:
    """Gates for the auto tensor split."""

    # MoE: tensor only when resident on the pair with a draft head (2x T4: decode +14..+21%, prompt -1..+4%);
    # without a draft the prompt lost 10..17%.
    dense_only: bool = True
    moe_resident_with_mtp: bool = True
    min_gpus: int = 2
    max_gpus: int = 2
    require_same_name: bool = True
    total_mem_match_frac: float = 0.02
    # Tensor split syncs every layer, so one busy card stalls all. 0 / None disables a check.
    min_free_frac: float = 0.80
    max_util_pct: Optional[float] = 10.0
    refuse_shared_gpus: bool = True


SPLIT_POLICY = SplitPolicyThresholds()

# Default --spec-draft-n-max per split; None keeps the platform default. 2 beat 4 on 2x T4 under both splits.
MTP_DRAFT_DEPTH_BY_SPLIT: dict[str, Optional[int]] = {"layer": 2, "tensor": 2, "none": None}


@dataclass(frozen = True)
class GpuInfo:
    """One candidate GPU; memory in MiB."""

    index: int
    name: Optional[str]
    total_mib: int
    free_mib: int
    util_pct: Optional[float] = None


@dataclass(frozen = True)
class SplitDecision:
    """``mode``: "tensor", "layer" or "none"; ``auto``: chosen by this policy, not the user or a precondition."""

    mode: str
    auto: bool
    reason: str
    gpu_indices: Optional[tuple] = None


def auto_tensor_split_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    value = (env if env is not None else os.environ).get(AUTO_TENSOR_SPLIT_ENV)
    if value is None or not str(value).strip():
        return AUTO_TENSOR_SPLIT_DEFAULT
    return str(value).strip().lower() not in ("0", "false", "off", "no")


def binary_supports_tensor_split(caps: Optional[Mapping[str, object]]) -> bool:
    """Whether ``--split-mode`` lists ``tensor`` in the probed help. Fails closed."""
    if not caps:
        return False
    flags = caps.get("flags")
    if not isinstance(flags, Mapping):
        return False
    desc = flags.get("--split-mode") or flags.get("-sm") or ""
    # "tensor" as a listed value, not part of "--tensor-split" / "tensor-split" prose.
    return re.search(r"(?<![-\w])tensor(?![-\w])", str(desc).lower()) is not None


def _gpus_match(gpus: Sequence[GpuInfo], thresholds: SplitPolicyThresholds) -> Optional[str]:
    if thresholds.require_same_name:
        names = {(g.name or "").strip().lower() for g in gpus}
        if "" in names:
            return "a GPU name is unknown"
        if len(names) > 1:
            return "the GPUs are different models (" + ", ".join(sorted(names)) + ")"
    totals = [g.total_mib for g in gpus]
    if any(t <= 0 for t in totals):
        return "a GPU total memory is unknown"
    largest = max(totals)
    if (largest - min(totals)) > thresholds.total_mem_match_frac * largest:
        return "the GPUs differ in total memory (" + ", ".join(f"{t} MiB" for t in totals) + ")"
    return None


def choose_split_mode(
    *,
    is_moe: bool,
    gpus: Sequence[GpuInfo],
    fits_one_gpu: bool,
    binary_supports_tensor: bool,
    p2p_veto: Optional[str] = None,
    draft_active: bool = False,
    user_tensor_parallel: bool = False,
    split_mode_override: Optional[str] = None,
    user_tensor_split: bool = False,
    env: Optional[Mapping[str, str]] = None,
    shared_gpus: frozenset = frozenset(),
    thresholds: SplitPolicyThresholds = SPLIT_POLICY,
    widen_fits_one_gpu: Optional[bool] = None,
    fits_on: Optional[Callable[[tuple], bool]] = None,
    platform_reason: Optional[str] = None,
    single_gpu_index: Optional[int] = None,
) -> SplitDecision:
    """Decide the split mode for an Auto GPU placement.

    ``p2p_veto`` does NOT block tensor (the AllReduce stages through host memory); it is only logged.
    """
    env = env if env is not None else os.environ
    n = len(gpus)
    single = "none" if n <= 1 else "layer"

    if user_tensor_parallel:
        return SplitDecision("tensor", False, "Tensor Parallelism toggle is on")
    if split_mode_override is not None:
        mode = str(split_mode_override).strip().lower()
        return SplitDecision(mode, False, f"user --split-mode {mode}")
    if user_tensor_split:
        return SplitDecision(single, False, "user --tensor-split keeps the layer split")
    inherited = (env.get("LLAMA_ARG_SPLIT_MODE") or "").strip().lower()
    if inherited:
        return SplitDecision(inherited, False, f"inherited LLAMA_ARG_SPLIT_MODE={inherited}")
    if (env.get("LLAMA_ARG_TENSOR_SPLIT") or "").strip():
        return SplitDecision(
            single, False, "inherited LLAMA_ARG_TENSOR_SPLIT keeps the layer split"
        )

    if not auto_tensor_split_enabled(env):
        return SplitDecision(
            single, False, f"auto tensor split off ({AUTO_TENSOR_SPLIT_ENV}=0 or default off)"
        )
    if platform_reason:
        return SplitDecision(
            single, False, f"auto tensor split not enabled here: {platform_reason}"
        )
    if n < max(2, thresholds.min_gpus):
        return SplitDecision(
            single,
            False,
            f"{n} GPU(s) available, tensor split needs >= {max(2, thresholds.min_gpus)}",
        )
    widen = TP_WIDEN_FITS_ONE_GPU if widen_fits_one_gpu is None else widen_fits_one_gpu
    if fits_one_gpu and (not widen or is_moe):
        return SplitDecision(
            "none",
            False,
            "the model fits one GPU; not widened onto more"
            + (" (MoE: not measured)" if is_moe and widen else ""),
        )
    if is_moe and (
        thresholds.dense_only and not (thresholds.moe_resident_with_mtp and draft_active)
    ):
        return SplitDecision(
            "none" if fits_one_gpu else "layer",
            fits_one_gpu is False,
            "MoE model without an MTP / draft head: layer split",
        )
    if not binary_supports_tensor:
        return SplitDecision(
            "none" if fits_one_gpu else "layer",
            not fits_one_gpu,
            "this llama-server build has no --split-mode tensor",
        )
    pair, why_not = _pick_pair(gpus, thresholds, shared_gpus)
    if pair is None:
        return SplitDecision(
            "none" if fits_one_gpu else "layer", not fits_one_gpu, f"layer split: {why_not}"
        )
    indices = tuple(g.index for g in pair)
    single = next((g for g in gpus if g.index == single_gpu_index), None)
    if (
        fits_one_gpu
        and single is not None
        and single.index not in indices
        and _gpu_idle_status(single, thresholds, shared_gpus) == "idle"
        and _gpus_match([single, pair[0]], thresholds) is not None
    ):
        # A widen keeps the idle card the one-GPU placement picked when the pair is another (smaller) class.
        return SplitDecision(
            "none",
            False,
            f"the model fits GPU {single_gpu_index}, outside the matching idle GPUs {list(indices)}; one GPU",
        )
    if fits_on is not None and not fits_on(indices):
        if fits_one_gpu:
            return SplitDecision("none", False, "the model does not fit the GPU pair; one GPU")
        return SplitDecision(
            "layer",
            True,
            f"layer split: the model needs more than {len(pair)} GPUs "
            f"(tensor split is capped at {min(AUTO_TENSOR_MAX_GPUS, thresholds.max_gpus)}: "
            "the internal AllReduce covers 2)",
        )
    if is_moe:
        reason = (
            f"MoE model resident on GPUs {list(indices)} with an MTP / draft head: tensor split"
        )
    elif fits_one_gpu:
        reason = f"dense model fits one GPU, widened onto matching idle GPUs {list(indices)}: tensor split"
    else:
        reason = f"dense model on matching idle GPUs {list(indices)}: tensor split"
    if draft_active:
        reason += f" (MTP depth {MTP_DRAFT_DEPTH_BY_SPLIT['tensor']})"
    if p2p_veto:
        reason += f"; no GPU peer path ({p2p_veto}), the AllReduce stages through host memory"
    return SplitDecision("tensor", True, reason, indices)


def _gpu_idle_status(g: GpuInfo, thresholds: SplitPolicyThresholds, shared_gpus: frozenset) -> str:
    """One card's state ("idle", "busy" or "unknown"), by the rules the pair is picked with."""
    if thresholds.max_util_pct is not None:
        # Free memory does not prove a card idle: an unread / N/A utilisation is not idle.
        if g.util_pct is None:
            return "unknown"
        if g.util_pct > thresholds.max_util_pct:
            return "busy"
    if thresholds.refuse_shared_gpus and g.index in shared_gpus:
        return "busy"
    if (
        thresholds.min_free_frac > 0
        and g.total_mib > 0
        and g.free_mib < thresholds.min_free_frac * g.total_mib
    ):
        return "busy"
    return "idle"


def _pick_pair(
    gpus: Sequence[GpuInfo], thresholds: SplitPolicyThresholds, shared_gpus: frozenset
) -> tuple[Optional[list], str]:
    """Up to max_gpus matching idle cards with the most free memory, else (None, why)."""
    idle = []
    busy = []
    unknown = []
    for g in gpus:
        status = _gpu_idle_status(g, thresholds, shared_gpus)
        if status == "unknown":
            unknown.append(g.index)
        elif status == "busy":
            busy.append(g.index)
        else:
            idle.append(g)
    need = max(2, thresholds.min_gpus)
    if len(idle) < need:
        why = []
        if busy:
            why.append(
                f"GPU(s) {sorted(busy)} are busy (another model, under {thresholds.min_free_frac:.0%} free, "
                f"or over {thresholds.max_util_pct}% utilisation)"
            )
        if unknown:
            why.append(f"GPU(s) {sorted(unknown)} utilisation unknown, not proven idle")
        return None, ("; ".join(why) or f"{len(idle)} idle GPU(s), tensor split needs {need}")
    groups: list[list] = []
    for g in sorted(idle, key = lambda x: -x.free_mib):
        for grp in groups:
            if _gpus_match([grp[0], g], thresholds) is None:
                grp.append(g)
                break
        else:
            groups.append([g])
    groups = [grp for grp in groups if len(grp) >= need]
    if not groups:
        return None, (_gpus_match(idle, thresholds) or "no two idle GPUs match")
    take = min(AUTO_TENSOR_MAX_GPUS, max(need, thresholds.max_gpus))
    if take < need:
        return (
            None,
            f"tensor split needs {need} GPUs, the auto policy is capped at {AUTO_TENSOR_MAX_GPUS}",
        )
    best = max(groups, key = lambda grp: sum(g.free_mib for g in grp[:take]))
    return best[:take], ""


def mtp_draft_depth_for_split(split_mode: Optional[str], default: int) -> int:
    if split_mode is None:
        return default
    depth = MTP_DRAFT_DEPTH_BY_SPLIT.get(str(split_mode).strip().lower())
    return default if depth is None else int(depth)
