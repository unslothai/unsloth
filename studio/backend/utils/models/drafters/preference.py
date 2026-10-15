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
    """Q8_0 ranks first for draft speed only: the target verifies every drafted token anyway."""
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
    """Prefers the self-contained MTP head over a ``-shared-`` one, which cannot load alone for --fit."""
    borrows = 1 if "shared" in Path(name).name.lower() else 0
    return mtp_precision_rank(name), borrows, Path(name).name.lower()


# DFlash shares DSpark's precision vocabulary (the sidecar has none, landing in the catch-all).
dflash_precision_rank = dspark_precision_rank


def dflash_preference_key(name: str) -> tuple[int, str]:
    """Sort key picking the preferred DFlash sidecar by name alone."""
    return dflash_precision_rank(name), Path(name).name.lower()


def dflash_repo_preference_key(
    name: str,
    weight_name: Optional[str] = None,
    other_weight_names: Iterable[str] = (),
) -> tuple[int, int, int, str]:
    """Demotes sidecars for another family below those for this weight, but keeps them as a fallback"""
    precision, sort_name = dflash_preference_key(name)
    if weight_name is not None and _drafter_matches_weight(name, weight_name, kind = "dflash"):
        return 0, _drafter_stem_rank(name, kind = "dflash"), precision, sort_name
    foreign = _drafter_names_other_weight(name, weight_name, other_weight_names)
    return 2 if foreign else 1, 0, precision, sort_name
