# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Full gc.collect() passes that stop once a pass that really ran finds nothing. Imports only gc."""

import gc

__all__ = ["collect_garbage"]


def collect_garbage(
    passes = 3,
    before = None,
    after = None,
):
    """Stops after an empty pass unless gc.callbacks are registered (they may drop refs) or gen 2
    stats lack one more empty collection (CPython returns 0 while another thread collects)."""
    for _ in range(passes):
        if before is not None:
            before()
        seen = gc.get_stats()[2]
        found = gc.collect()
        ran = gc.get_stats()[2]
        if after is not None:
            after()
        if (
            found == 0
            and not gc.callbacks
            and ran["collections"] == seen["collections"] + 1
            and ran["collected"] == seen["collected"]
            and ran["uncollectable"] == seen["uncollectable"]
        ):
            break
