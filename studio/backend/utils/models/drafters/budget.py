# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a drafter costs the training coexistence guard.

The guard admits an inference load only if it fits beside a running training
job, so it has to price the drafter that load will actually make resident. It
sees a repository listing, never a header, which is what makes this its own
problem rather than a detail of discovery: the rules that decide WHICH sidecar
lands cannot all be evaluated here, so the budget bounds them instead.
"""

from typing import Callable, Mapping

from utils.models.drafters.common import split_listing_is_complete


def dflash_budget_bytes(
    sizes: Mapping[str, int],
    extra_shards: Callable[[Mapping[str, int], str], list],
    target_bytes: int = 0,
    *,
    require_full_sizes: bool = False,
) -> int:
    """A safe bound on the DFlash sidecar a load may end up resident on.

    The largest candidate the fetch could end up on, not the best-ranked one.
    The download can only read a candidate's header once it has paid for the
    bytes, and a rejection falls through to the next name in the ranking, so any
    candidate can be the file that lands, and the whole point of the fallback is
    the case where it is a different, bigger one. Headers are unreadable from a
    listing, so the ranking cannot narrow that down here, and over-estimating is
    the established safe direction for a guard protecting a running training
    job.

    Each entry summed is a whole shard SET, not one file: a split sidecar is
    picked as its first shard and the companion download then fetches every
    sibling, all of which llama-server keeps resident. Sizing one shard would
    halve a two-shard sidecar, and under-estimating is the direction that waves
    a load through and then exhausts VRAM.

    ``target_bytes`` drops what the fetch itself refuses: a drafter is a few layers of
    its target, so a set at least that large is an ordinary weight wearing the prefix.
    Zero means unknown and keeps every candidate.

    An incomplete split set is refused for the same reason: the fetch turns those
    families away on the shard count, so charging their listed part is a 409 for a
    load that fits, which is what a mid-publication listing looks like.

    ``require_full_sizes`` drops a loadable family whose listing did not size every
    shard, instead of summing the shards it did size. A two-shard sidecar listed as
    3 GiB plus an unknown is not a 3 GiB sidecar; llama-server maps both. Callers
    that have somewhere else to go -- a cache measurement, then a flat reserve --
    want that family excluded so they get there. Callers with no fallback are
    better off with the partial sum than with nothing, so this is off by default.
    """

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
