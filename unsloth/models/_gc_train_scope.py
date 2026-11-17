# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""gc.freeze setup heap in outermost train() (nested calls no-op), and again after its first step
(the heap torch.compile left behind); unfreeze on exit unless caller already froze;
UNSLOTH_GC_FREEZE=0 disables both, UNSLOTH_GC_FREEZE_AFTER_STEP=0 only the second."""

import functools
import gc
import os
import threading

__all__ = [
    "unsloth_gc_train_scope",
    "patch_trainer_train_gc_scope",
    "freeze_after_first_step",
]

_lock = threading.Lock()
_depth = 0
_owns_freeze = False


def _gc_freeze_enabled():
    value = os.environ.get("UNSLOTH_GC_FREEZE", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _enter():
    global _depth, _owns_freeze
    with _lock:
        _depth += 1
        if _depth != 1:
            return
        _owns_freeze = False
        try:
            if not _gc_freeze_enabled() or gc.get_freeze_count() > 0:
                return
            gc.collect()
            gc.freeze()
            _owns_freeze = True
        except BaseException:
            # KeyboardInterrupt inside gc.collect(): leave the counter as we found it.
            _depth -= 1
            raise


def _exit():
    global _depth, _owns_freeze
    with _lock:
        _depth -= 1
        if _depth != 0:
            return
        if _owns_freeze:
            _owns_freeze = False
            gc.unfreeze()


def _after_step_enabled():
    value = os.environ.get("UNSLOTH_GC_FREEZE_AFTER_STEP", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def freeze_after_first_step():
    """Freeze what the first step left alive (compiled graphs, guards, Inductor state): otherwise each
    automatic full collection rescans it, and a warm start, which builds less of it, sets a lower
    collection threshold and pays that rescan more often. Only inside a freeze this scope owns."""
    with _lock:
        if _depth < 1 or not _owns_freeze or not _after_step_enabled():
            return False
        gc.collect()
        gc.freeze()
        return True


def _first_step_callback():
    try:
        from transformers import TrainerCallback
    except Exception:
        return None

    class _FreezeAfterFirstStep(TrainerCallback):
        def __init__(self):
            self.done = False

        def on_step_end(self, args, state, control, **kwargs):
            if not self.done:
                self.done = True
                freeze_after_first_step()

    return _FreezeAfterFirstStep()


class unsloth_gc_train_scope:
    """Context manager: `gc.collect(); gc.freeze()` on the outermost entry, `gc.unfreeze()` on exit."""

    __slots__ = ()

    def __enter__(self):
        _enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _exit()
        return False


def patch_trainer_train_gc_scope(trainer_class):
    """Wrap `trainer_class.train` once so every call runs inside `unsloth_gc_train_scope`."""
    original_train = trainer_class.__dict__.get("train")
    if original_train is None or getattr(original_train, "_unsloth_gc_train_scope", False):
        return False

    @functools.wraps(original_train)
    def train(self, *args, **kwargs):
        with unsloth_gc_train_scope():
            callback = None
            if _depth == 1 and _owns_freeze and hasattr(self, "add_callback"):
                callback = _first_step_callback()
                if callback is not None:
                    self.add_callback(callback)
            try:
                return original_train(self, *args, **kwargs)
            finally:
                if callback is not None:
                    try:
                        self.remove_callback(callback)
                    except Exception:
                        pass

    train._unsloth_gc_train_scope = True
    trainer_class.train = train
    return True
