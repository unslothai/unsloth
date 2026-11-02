# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Drafter cost for the training coexistence guard; sees only repo listings, so it bounds the sidecar."""

from typing import Callable, Mapping

from utils.models.drafters.common import split_listing_is_complete


def dflash_budget_bytes(
    sizes: Mapping[str, int],
    extra_shards: Callable[[Mapping[str, int], str], list],
    target_bytes: int = 0,
    *,
    require_full_sizes: bool = False,
) -> int:
    """Over-estimate of the DFlash sidecar: largest candidate, whole shard sets; target_bytes drops ordinary weights; require_full_sizes drops partly sized families."""

    def _family(name: str, size: int) -> tuple[int, bool]:
        shards = list(extra_shards(sizes, name))
        total = size + sum(sizes.get(shard, 0) for shard in shards)
        sized = bool(size) and all(sizes.get(shard) for shard in shards)
        return total, sized

    totals = (
        total
        for name, size in sizes.items()
        if split_listing_is_complete(sizes, name)
        for total, sized in (_family(name, size),)
        if sized or not require_full_sizes
    )
    return max(
        (total for total in totals if not target_bytes or total < target_bytes),
        default = 0,
    )
