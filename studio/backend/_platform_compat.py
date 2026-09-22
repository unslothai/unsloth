# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import before structlog/rich: Anaconda's ``|...|`` metadata in sys.version makes
platform._sys_version() raise (cpython#102396, wontfix), so seed its cache first."""

import platform
import re
import sys


def _seed_sys_version_cache() -> None:
    raw = sys.version

    cleaned = re.sub(r"\s*\|[^|]*\|\s*", " ", raw).strip()

    # Pipe-strip can leave two consecutive (...) groups; drop the second.
    cleaned = re.sub(r"(\([^)]*\))\s+\([^)]*\)", r"\1", cleaned)

    if "|" in cleaned:
        m = re.match(r"([\w.+]+)\s*", cleaned)
        p = cleaned.find("(")
        if m and p > 0:
            cleaned = m.group(0) + cleaned[p:]

    if cleaned == raw:
        return

    try:
        result = platform._sys_version(cleaned)
    except ValueError:
        return

    cache = getattr(platform, "_sys_version_cache", None)
    if isinstance(cache, dict):
        cache[raw] = result


if "|" in sys.version:
    _seed_sys_version_cache()
