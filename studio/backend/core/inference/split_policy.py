# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic llama-server split mode (layer vs tensor) and the MTP draft depth per split mode.

Pure functions and constants only (no torch, no FastAPI), so the decision table is unit-tested without a GPU.

Explicit choices always win and never reach this module's decision: the Tensor Parallelism toggle (True),
``--split-mode`` / ``-sm`` / ``--tensor-split`` in the advanced arguments, and an inherited
``LLAMA_ARG_SPLIT_MODE`` / ``LLAMA_ARG_TENSOR_SPLIT``. The UI always sends ``tensor_parallel = false``, so False
is read as "not chosen"; a user who wants layer split passes ``--split-mode layer`` or sets
``UNSLOTH_STUDIO_AUTO_TENSOR_SPLIT=0``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Single switch for the auto policy. False = today's behaviour (layer split unless the user asks for tensor).
AUTO_TENSOR_SPLIT_DEFAULT = True
# Kill switch / opt-in env: "0" keeps today's behaviour, "1" enables the policy when the default is off.
AUTO_TENSOR_SPLIT_ENV = "UNSLOTH_STUDIO_AUTO_TENSOR_SPLIT"
# Whether a model that fits one GPU may be widened onto 2+ GPUs for tensor split. Off: the fewest-GPUs selection
# stays the authority, the auto policy only changes HOW an already multi-GPU load is split.
TP_WIDEN_FITS_ONE_GPU = False


@dataclass(frozen = True)
class SplitPolicyThresholds:
    """Data-driven gates for the auto tensor split. Provisional values; the measured table replaces them."""

    # MoE models stay on layer split (expert weights do not shard well under -sm tensor, and the MoE cache /
    # expert offload paths are layer-split only).
    dense_only: bool = True
    # Smallest GPU count a tensor split is chosen for.
    min_gpus: int = 2
    # Every selected GPU must report the same device name ...
    require_same_name: bool = True
    # ... and the same total memory within this fraction of the largest card.
    total_mem_match_frac: float = 0.02
    # Each GPU must have at least this fraction of its total memory free (idle enough: tensor split synchronises
    # every layer, so one busy card stalls all of them). 0 disables the check.
    min_free_frac: float = 0.80
    # Cards another loaded model already runs on are not idle.
    refuse_shared_gpus: bool = True


SPLIT_POLICY = SplitPolicyThresholds()

# MTP draft depth (--spec-draft-n-max) Studio picks by itself, per split mode. "none" = one GPU / no split: None
# keeps today's platform default (2 on a GPU, 3 on CPU, PR #5582). Never applied over a user value, a trained
# head's depth or DSpark's fixed 3.
MTP_DRAFT_DEPTH_BY_SPLIT: dict[str, Optional[int]] = {"layer": 2, "tensor": 4, "none": None}


@dataclass(frozen = True)
class GpuInfo:
    """One candidate GPU: physical index, device name (lowercased or not), total and free memory in MiB."""

    index: int
    name: Optional[str]
    total_mib: int
    free_mib: int


@dataclass(frozen = True)
class SplitDecision:
    """``mode``: "tensor", "layer" or "none" (single GPU). ``auto``: whether this module chose it (False when an
    explicit user choice or a precondition left today's behaviour in place). ``reason``: one line for the log."""

    mode: str
    auto: bool
    reason: str


def auto_tensor_split_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """AUTO_TENSOR_SPLIT_DEFAULT, overridden by UNSLOTH_STUDIO_AUTO_TENSOR_SPLIT (0/false/off or 1/true/on)."""
    value = (env if env is not None else os.environ).get(AUTO_TENSOR_SPLIT_ENV)
    if value is None or not str(value).strip():
        return AUTO_TENSOR_SPLIT_DEFAULT
    return str(value).strip().lower() not in ("0", "false", "off", "no")


def binary_supports_tensor_split(caps: Optional[Mapping[str, object]]) -> bool:
    """Whether the llama-server ``--help`` (as parsed by probe_server_capabilities into ``caps["flags"]``) lists
    ``tensor`` among the ``--split-mode`` values. Fails closed on a failed or missing probe."""
    if not caps:
        return False
    flags = caps.get("flags")
    if not isinstance(flags, Mapping):
        return False
    desc = flags.get("--split-mode") or flags.get("-sm") or ""
    return "tensor" in str(desc).lower()


def _gpus_match(gpus: Sequence[GpuInfo], thresholds: SplitPolicyThresholds) -> Optional[str]:
    """None when every GPU matches, else why not."""
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
) -> SplitDecision:
    """Decide the split mode for an Auto GPU placement.

    ``gpus``: the candidate GPUs the placement may use (after any GPU picker filter). ``fits_one_gpu``: the
    fewest-GPUs selection would put the model on one card. ``p2p_veto``: _p2p_veto_reason's answer; it does NOT
    block tensor (the internal AllReduce stages through pinned host memory and NCCL picks its own transport), it is
    only recorded in the reason. ``draft_active``: a draft model / MTP head will run (only the depth depends on it).
    """
    env = env if env is not None else os.environ
    n = len(gpus)
    single = "none" if n <= 1 else "layer"

    # Explicit choices first: never auto.
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
    if n < max(2, thresholds.min_gpus):
        return SplitDecision(
            single,
            False,
            f"{n} GPU(s) available, tensor split needs >= {max(2, thresholds.min_gpus)}",
        )
    widen = TP_WIDEN_FITS_ONE_GPU if widen_fits_one_gpu is None else widen_fits_one_gpu
    if fits_one_gpu and not widen:
        return SplitDecision("none", False, "the model fits one GPU; not widened onto more")
    if thresholds.dense_only and is_moe:
        return SplitDecision("layer", True, "MoE model: layer split")
    if not binary_supports_tensor:
        return SplitDecision("layer", True, "this llama-server build has no --split-mode tensor")
    mismatch = _gpus_match(gpus, thresholds)
    if mismatch is not None:
        return SplitDecision("layer", True, f"layer split: {mismatch}")
    if thresholds.refuse_shared_gpus:
        busy = sorted(g.index for g in gpus if g.index in shared_gpus)
        if busy:
            return SplitDecision(
                "layer", True, f"layer split: GPU(s) {busy} already run another model"
            )
    if thresholds.min_free_frac > 0:
        busy = sorted(g.index for g in gpus if g.free_mib < thresholds.min_free_frac * g.total_mib)
        if busy:
            return SplitDecision(
                "layer",
                True,
                f"layer split: GPU(s) {busy} have under {thresholds.min_free_frac:.0%} of their memory free",
            )
    reason = f"dense model on {n} matching idle GPUs: tensor split"
    if draft_active:
        reason += f" (MTP depth {MTP_DRAFT_DEPTH_BY_SPLIT['tensor']})"
    if p2p_veto:
        reason += f"; no GPU peer path ({p2p_veto}), the AllReduce stages through host memory"
    return SplitDecision("tensor", True, reason)


def mtp_draft_depth_for_split(split_mode: Optional[str], default: int) -> int:
    """Studio's own MTP draft depth for a split mode: the table entry for "layer" / "tensor" / "none", else
    ``default`` (today's platform rule). Callers apply it only when the user set no depth."""
    if split_mode is None:
        return default
    depth = MTP_DRAFT_DEPTH_BY_SPLIT.get(str(split_mode).strip().lower())
    return default if depth is None else int(depth)
