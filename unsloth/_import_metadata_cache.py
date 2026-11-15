# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Reuse `importlib.metadata.packages_distributions()` while `import unsloth` runs.

transformers >= 5 and TRL >= 1 both scan every distribution at import. A repeat call returns a copy
of the first result while sys.path, each entry's mtime and the distribution finders are unchanged:
the same mtimes key importlib.metadata's own FastPath.lookup cache, so installs are still seen.
Stdlib only: Studio loads this file by path, without running unsloth/__init__.py.
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
    if not scope["open"]:
        return False
    owner = scope["owner"]
    # A failed import drops its module from sys.modules; uninstall never ran.
    return owner is None or sys.modules.get(scope["name"]) is owner


def _close(wrapper):
    wrapper._unsloth_metadata_scope["open"] = False
    wrapper._unsloth_metadata_cache[0] = None
    if getattr(_metadata, "packages_distributions", None) is wrapper:
        restored = wrapper.__wrapped__
        _metadata.packages_distributions = restored
        # An outer window that closed while this one was on top.
        scope = getattr(restored, "_unsloth_metadata_scope", None)
        if scope is not None and not _live(scope):
            _close(restored)


def install(owner = None):
    """Returns a token for `uninstall`, else None. If `owner`'s import raises, the wrapper closes itself."""
    with _lock:
        original = getattr(_metadata, "packages_distributions", None)
        # Left by an earlier import that raised: never stack on it.
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
    """Restore the original unless something wrapped it since; the wrapper stops caching either way."""
    if token is None:
        return
    with _lock:
        _close(token)
