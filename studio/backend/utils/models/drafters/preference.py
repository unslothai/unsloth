# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ranking keys that decide which sidecar a load prefers.

The download, the snapshot reuse, the offline cache lookup and the local scan
all order candidates with these, so a repo resolves to the same file whichever
path reaches it first.
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
