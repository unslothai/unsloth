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


def _live(scope):
    """True while the window `install` opened is still open."""
    if not scope["open"]:
        return False
    owner = scope["owner"]
    # A failed import drops its module from sys.modules; uninstall never ran.
    return owner is None or sys.modules.get(scope["name"]) is owner


def _close(wrapper):
    """End the window: forget the memo and put the original back if nothing wrapped it since."""
    wrapper._unsloth_metadata_scope["open"] = False
    wrapper._unsloth_metadata_cache[0] = None
    if getattr(_metadata, "packages_distributions", None) is wrapper:
        restored = wrapper.__wrapped__
        _metadata.packages_distributions = restored
        # An outer window that already closed (its uninstall found this one on top): unwrap it too.
        scope = getattr(restored, "_unsloth_metadata_scope", None)
        if scope is not None and not _live(scope):
            _close(restored)


def install(owner = None):
    """Wrap `importlib.metadata.packages_distributions`; returns a token for `uninstall`, else None.

    `owner` is the module whose import opens the window: if that import raises, the wrapper closes
    itself on its next call (and the next `install` unwraps it) instead of staying for good.
    """
    with _lock:
        original = getattr(_metadata, "packages_distributions", None)
        # Left behind by an earlier attempt whose import raised: take it out, never stack on it.
        scope = getattr(original, "_unsloth_metadata_scope", None)
        if scope is not None and not _live(scope):
            _close(original)
            original = getattr(_metadata, "packages_distributions", None)
        if original is None:
            return None
        cache = [None]
        scope = {
            "open": True,
            "owner": owner,
            "name": getattr(owner, "__name__", None),
        }

        def packages_distributions():
            if not _live(scope):
                # Window over (or its import failed): behave exactly like the original.
                with _lock:
                    _close(packages_distributions)
                return original()
            key = _fingerprint()
            with _lock:
                hit = cache[0]
            if hit is not None and hit[0] == key:
                return _copy(hit[1])
            result = original()
            # Keyed on the state before the call: a change while it ran misses next time.
            with _lock:
                if scope["open"]:
                    cache[0] = (key, _copy(result))
            return result

        packages_distributions.__wrapped__ = original
        packages_distributions.__doc__ = getattr(original, "__doc__", None)
        packages_distributions._unsloth_metadata_cache = cache
        packages_distributions._unsloth_metadata_scope = scope
        _metadata.packages_distributions = packages_distributions
        return packages_distributions


def uninstall(token):
    """Put back what `install` replaced, unless something wrapped it since (then leave both).

    Either way the wrapper stops caching: a reference kept past the window reaches the original.
    """
    if token is None:
        return
    with _lock:
        _close(token)
