# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memoized family-capability snapshots for media status responses (no heavy imports)."""

from __future__ import annotations

import functools
import time


def memoize_capability_snapshot(is_complete, retry_after_s: float = 60.0):
    """Memoize a family-capability snapshot; an incomplete one expires.

    A strict probe that races the startup import warm fails transiently (a half-imported
    transformers raised "cannot import name 'CLIPImageProcessor'" for FluxPipeline), so a
    snapshot missing a family is re-probed after ``retry_after_s`` instead of pinned for the
    process lifetime."""

    def decorate(fn):
        cache: dict = {}

        @functools.wraps(fn)
        def wrapper(*args):
            now = time.monotonic()
            hit = cache.get(args)
            if hit is not None and (hit[0] is None or now < hit[0]):
                return hit[1]
            value = fn(*args)
            cache[args] = (None if is_complete(value) else now + retry_after_s, value)
            return value

        wrapper.cache_clear = cache.clear
        return wrapper

    return decorate
