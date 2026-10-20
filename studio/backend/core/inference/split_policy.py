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
from typing import Callable, Mapping, Optional, Sequence

# Single switch for the auto policy. False = today's behaviour (layer split unless the user asks for tensor).
AUTO_TENSOR_SPLIT_DEFAULT = True
# Kill switch / opt-in env: "0" keeps today's behaviour, "1" enables the policy when the default is off.
AUTO_TENSOR_SPLIT_ENV = "UNSLOTH_STUDIO_AUTO_TENSOR_SPLIT"
# Whether a dense model that fits one GPU is widened onto two matching idle GPUs for tensor split. Measured on
# Kaggle 2x T4 (plans/multi_gpu_dense.md): Qwen3.5-9B Q8_0 decode 18.8 -> 41.5 tok/s (one GPU -> tensor over two),
# 2k prompt 934 -> 1388 tok/s. Only idle cards are taken (SPLIT_POLICY.min_free_frac / refuse_shared_gpus).
TP_WIDEN_FITS_ONE_GPU = True


@dataclass(frozen = True)
class SplitPolicyThresholds:
    """Data-driven gates for the auto tensor split. Provisional values; the measured table replaces them."""

    # MoE models stay on layer split, except (moe_resident_with_mtp) a model whose weights all fit the GPU pair while
    # an MTP / draft head runs: there tensor decode was +14..+21% and the prompt within -1..+4% (2k / 8k) on 2x T4
    # (Qwen3.6-35B-A3B UD-Q4_K_XL, depth 2). Without a draft the prompt cost -10..-17% (2k / 8k) against +15%
    # decode, and expert offload / the MoE cache are layer-split only.
    dense_only: bool = True
    moe_resident_with_mtp: bool = True
    # Smallest GPU count a tensor split is chosen for.
    min_gpus: int = 2
    # Largest: the internal AllReduce (the only one in the prebuilt, which does not link NCCL) covers exactly 2
    # GPUs; with more, llama.cpp falls back to the meta backend's generic AllReduce, which was not measured.
    max_gpus: int = 2
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
# head's depth or DSpark's fixed 3. Measured on 2x T4 (plans/multi_gpu_dense.md): depth 2 beat depth 4 under layer split
# everywhere (+4..+40%); under tensor split depth 2 won for Qwen3.6-35B-A3B and was mixed for Qwen3.5-9B / Qwen3.6-27B,
# so both stay at 2.
MTP_DRAFT_DEPTH_BY_SPLIT: dict[str, Optional[int]] = {"layer": 2, "tensor": 2, "none": None}


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
    # The GPUs a tensor split runs on (physical indices), None = the caller's selection unchanged.
    gpu_indices: Optional[tuple] = None


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
    fits_on: Optional[Callable[[tuple], bool]] = None,
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
    if fits_on is not None and not fits_on(indices):
        if fits_one_gpu:
            return SplitDecision("none", False, "the model does not fit the GPU pair; one GPU")
        return SplitDecision(
            "layer",
            True,
            f"layer split: the model needs more than {len(pair)} GPUs "
            f"(tensor split is capped at {thresholds.max_gpus}: the internal AllReduce covers 2)",
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


def _pick_pair(
    gpus: Sequence[GpuInfo], thresholds: SplitPolicyThresholds, shared_gpus: frozenset
) -> tuple[Optional[list], str]:
    """The tensor-split GPUs: up to thresholds.max_gpus idle cards of one model (same name, total within
    total_mem_match_frac), the group with the most free memory first. (None, why) when no such pair exists."""
    idle = []
    busy = []
    for g in gpus:
        if thresholds.refuse_shared_gpus and g.index in shared_gpus:
            busy.append(g.index)
            continue
        if (
            thresholds.min_free_frac > 0
            and g.total_mib > 0
            and g.free_mib < thresholds.min_free_frac * g.total_mib
        ):
            busy.append(g.index)
            continue
        idle.append(g)
    need = max(2, thresholds.min_gpus)
    if len(idle) < need:
        return None, (
            f"GPU(s) {sorted(busy)} are busy (another model, or under {thresholds.min_free_frac:.0%} free)"
            if busy
            else f"{len(idle)} idle GPU(s), tensor split needs {need}"
        )
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
    best = max(
        groups, key = lambda grp: sum(g.free_mib for g in grp[: max(need, thresholds.max_gpus)])
    )
    return best[: max(need, thresholds.max_gpus)], ""


def mtp_draft_depth_for_split(split_mode: Optional[str], default: int) -> int:
    """Studio's own MTP draft depth for a split mode: the table entry for "layer" / "tensor" / "none", else
    ``default`` (today's platform rule). Callers apply it only when the user set no depth."""
    if split_mode is None:
        return default
    depth = MTP_DRAFT_DEPTH_BY_SPLIT.get(str(split_mode).strip().lower())
    return default if depth is None else int(depth)
