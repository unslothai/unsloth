# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Compatibility shim for Anaconda/conda-forge Python builds.

Anaconda modifies sys.version to include distributor metadata between pipe
characters, e.g. '3.12.4 | packaged by Anaconda, Inc. | (main, ...) [MSC ...]'.
Python's platform._sys_version() has a hardcoded regex that cannot parse this,
raising ValueError. CPython closed this as "not planned" (cpython#102396).

This module patches platform._sys_version() to retry with a cleaned version
string on ValueError, fixing the import chain:
    structlog -> rich.pretty -> attrs._compat -> platform.python_implementation()

Import this module before any library imports that may trigger the above chain.
The patch is idempotent and safe to import multiple times.
"""

import platform
import re
import sys


def _patch_platform_sys_version() -> None:
    if getattr(platform, "_unsloth_sys_version_patched", False):
        return

    _original = platform._sys_version

    def _patched(sys_version = None):
        try:
            return _original(sys_version)
        except ValueError:
            target = sys.version if sys_version is None else sys_version
            if "|" not in target:
                raise
            # Strip paired |...| segments (Anaconda, conda-forge)
            cleaned = re.sub(r"\s*\|[^|]*\|\s*", " ", target).strip()
            if "|" in cleaned:
                # Unpaired pipes -- keep version + everything from "(" onward
                m = re.match(r"([\w.+]+)\s*", cleaned)
                p = cleaned.find("(")
                if m and p > 0:
                    cleaned = m.group(0) + cleaned[p:]
            if cleaned == target:
                raise
            result = _original(cleaned)
            # Cache under original key so future calls are fast
            cache = getattr(platform, "_sys_version_cache", None)
            if isinstance(cache, dict):
                cache[target] = result
            return result

    platform._sys_version = _patched
    platform._unsloth_sys_version_patched = True


if "|" in sys.version:
    _patch_platform_sys_version()
