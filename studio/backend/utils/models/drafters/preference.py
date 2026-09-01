# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ranking keys that decide which sidecar a load prefers.

The download, the snapshot reuse and the offline cache lookup share them, so a
repo resolves the same way whichever reaches it first. The local scan does not:
detect_mtp_file ranks size-first (``_smallest_first``) because a folder holding
several copies costs disk, while these rank speed-first because the hub path is
choosing what to spend a download on. Same family, sometimes a different copy.
"""

from pathlib import Path
from typing import Iterable, Optional

from utils.models.drafters.common import (
    _drafter_matches_weight,
    _drafter_names_other_weight,
    _drafter_stem_rank,
)


def dspark_precision_rank(name: str) -> int:
    """Sidecar precision preference: Q8_0 first, the precision the DSpark model
    card recommends. Shared with the hub download and VRAM-sizing paths so the
    file Unsloth budgets for is the file it fetches and launches."""
    base = Path(name).name.lower()
    if "-q8_0" in base:
        return 0
    if "-q4_0" in base:
        return 1
    if "-bf16" in base or "-f16" in base:
        return 2
    return 3


def dspark_preference_key(name: str) -> tuple[int, str]:
    """Sort key picking the preferred sidecar by name alone (no filesystem)."""
    return dspark_precision_rank(name), Path(name).name.lower()


def mtp_precision_rank(name: str) -> int:
    """MTP head precision preference: Q8_0 first. Correctness does not enter into
    it, since the target verifies every drafted token, so this is purely which
    head drafts fastest. Q8_0 measured both quicker and more accepted than bf16 on
    Qwen3.8-Flash-Next: a draft step is dominated by the LM head, and that head is
    cheaper to execute at 8 bits, so bf16 is larger and slower for no gain."""
    base = Path(name).name.lower()
    if "-q8_0" in base:
        return 0
    if "-q6_k" in base:
        return 1
    if "-q5_k" in base:
        return 2
    if "-q4_k" in base or "-q4_0" in base:
        return 3
    if "-bf16" in base or "-f16" in base:
        return 4
    return 5


def mtp_preference_key(name: str) -> tuple[int, int, str]:
    """Sort key picking the preferred MTP head by name alone.

    A head borrowing the target's token_embd/output wins the tie: 1.35 GB smaller
    at Q8_0 and no worse, accepting identically (159 of 284) on the shipped
    prebuilt. Only the qwen4exp path reaches a repo publishing both forms, and
    qwen4exp MTP and the borrow ship in one fork, so the borrow always resolves.
    """
    borrows = 0 if "shared" in Path(name).name.lower() else 1
    return mtp_precision_rank(name), borrows, Path(name).name.lower()


# DFlash publishes the same precision vocabulary (and the published sidecar
# carries no precision token at all, which lands in the catch-all rank), so the
# ordering is shared rather than duplicated.
dflash_precision_rank = dspark_precision_rank


def dflash_preference_key(name: str) -> tuple[int, str]:
    """Sort key picking the preferred DFlash sidecar by name alone."""
    return dflash_precision_rank(name), Path(name).name.lower()


def dflash_repo_preference_key(
    name: str,
    weight_name: Optional[str] = None,
    other_weight_names: Iterable[str] = (),
) -> tuple[int, int, int, str]:
    """Order DFlash sidecars in a repo listing / cache snapshot against the
    weight actually being loaded.

    dflash_preference_key ranks by precision and name alone, which is all a
    single-model repo needs. A repo hosting more than one family also has to be
    told which weight each sidecar belongs to, or ``dflash-model-A-Q8_0.gguf``
    outranks the generic ``dflash-kquant.gguf`` on precision and model B is
    launched with model A's drafter. Same rule the local scan applies in
    detect_dflash_file, kept in one place so the download, the snapshot reuse
    and the offline cache all pick the same file.

    Three buckets: a sidecar naming this weight's family (most specific stem
    first, as detect_mtp_file does), then one naming no weight present here,
    then one naming a neighbour. The last is demoted rather than dropped, so a
    repo whose only sidecar looks foreign still gets a fallback and today's
    single-sidecar behaviour is unchanged.
    """
    precision, sort_name = dflash_preference_key(name)
    if weight_name is not None and _drafter_matches_weight(name, weight_name, kind = "dflash"):
        return 0, _drafter_stem_rank(name, kind = "dflash"), precision, sort_name
    foreign = _drafter_names_other_weight(name, weight_name, other_weight_names)
    return 2 if foreign else 1, 0, precision, sort_name
