# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Reuse `importlib.metadata.packages_distributions()` while `import unsloth` runs.

transformers (>= 5) and TRL (>= 1) each call it at import time. Every call walks every installed
distribution and stats every file listed in RECORD for wheels without top_level.txt: about
0.5 s here and 0.8 s on Colab, paid twice. A repeat call returns a copy of the first result while
sys.path, the mtime of each sys.path entry and the distribution finders on sys.meta_path are
unchanged. importlib.metadata keys its own directory listing cache (FastPath.lookup) on the same
mtimes, so a distribution installed or removed in between is seen exactly as without the cache.

Stdlib only: Studio loads this file by path before transformers is imported, without running
unsloth/__init__.py.
"""

import importlib.metadata as _metadata
import os
import sys
import threading

__all__ = ["install", "uninstall"]

_lock = threading.Lock()


def _fingerprint():
    entries = []
    for entry in list(sys.path):
        root = entry
        try:
            if not entry:
                # "" is the current directory, which can change between two calls.
                root = os.getcwd()
            mtime = os.stat(root).st_mtime_ns
        except (OSError, TypeError, ValueError):
            mtime = None
        entries.append((entry, root, mtime))
    finders = tuple(
        id(finder)
        for finder in list(sys.meta_path)
        if getattr(finder, "find_distributions", None) is not None
    )
    return (tuple(entries), finders)


def _copy(mapping):
    return {name: list(dists) for name, dists in mapping.items()}


def install():
    """Wrap `importlib.metadata.packages_distributions`; returns a token for `uninstall`, else None."""
    with _lock:
        original = getattr(_metadata, "packages_distributions", None)
        if original is None:
            return None
        cache = [None]

        def packages_distributions():
            key = _fingerprint()
            with _lock:
                hit = cache[0]
            if hit is not None and hit[0] == key:
                return _copy(hit[1])
            result = original()
            # Keyed on the state before the call: a change while it ran misses next time.
            with _lock:
                cache[0] = (key, _copy(result))
            return result

        packages_distributions.__wrapped__ = original
        packages_distributions.__doc__ = getattr(original, "__doc__", None)
        packages_distributions._unsloth_metadata_cache = cache
        _metadata.packages_distributions = packages_distributions
        return packages_distributions


def uninstall(token):
    """Put back what `install` replaced, unless something wrapped it since (then leave both)."""
    if token is None:
        return
    with _lock:
        token._unsloth_metadata_cache[0] = None
        if getattr(_metadata, "packages_distributions", None) is token:
            _metadata.packages_distributions = token.__wrapped__
