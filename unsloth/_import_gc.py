# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Raise the gen 0 GC threshold while `import unsloth` runs: torch / transformers / trl allocate
~700k long-lived objects, so the default triggers full collections that find almost nothing.
A failed import leaves sys.modules; a meta path hook then restores it on the next import."""

import gc
import sys

IMPORT_THRESHOLD = 100_000
_saved = None


class _RestoreAfterFailedImport:
    """Never finds anything; once `import unsloth` is no longer in flight, gives the threshold back."""

    __slots__ = ("module", "name")

    def __init__(self, module):
        self.arm(module)

    def arm(self, module):
        self.module = module
        self.name = module.__name__

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        module = self.module
        if module is not None and sys.modules.get(self.name) is not module:
            # Not removed here: sys.meta_path is iterated live, removal skips the next finder.
            self.module = None
            restore_gc_threshold(remove_hook = False)
        return None


_hook = None


def raise_gc_threshold(module = None):
    global _saved, _hook
    current = gc.get_threshold()
    if _saved is not None and current[0] == IMPORT_THRESHOLD:
        pass  # Still raised by an earlier attempt that failed.
    # 0 means automatic collection is off; a higher threshold was chosen on purpose.
    elif current[0] == 0 or current[0] >= IMPORT_THRESHOLD:
        _saved = None
        return
    else:
        _saved = current[0]
        gc.set_threshold(IMPORT_THRESHOLD, *current[1:])
    if module is not None:
        if _hook is None:
            _hook = _RestoreAfterFailedImport(module)
            sys.meta_path.insert(0, _hook)
        else:
            _hook.arm(module)


def restore_gc_threshold(remove_hook = True):
    global _saved, _hook
    if remove_hook:
        hook, _hook = _hook, None
        if hook is not None:
            try:
                sys.meta_path.remove(hook)
            except ValueError:
                pass
    saved, _saved = _saved, None
    # Leave it alone if something changed it during the import; generations 1 and 2 are the caller's.
    current = gc.get_threshold()
    if saved is not None and current[0] == IMPORT_THRESHOLD:
        gc.set_threshold(saved, *current[1:])
