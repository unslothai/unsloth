# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Freeze the setup-time heap for the duration of `Trainer.train()`.

Libraries call `gc.collect()` several times inside `train()`. Each full collection walks every
tracked object, and after loading a model, tokenizer, dataset and compiled code that heap is
large: about 0.5 s per pass, 2 to 3 s per `train()` on a 0.6B / 1B LoRA run. `gc.freeze()` moves
everything alive at train start into the permanent generation, so those passes only see objects
created during training.

The freeze must be undone when `train()` exits (normally, by exception or by KeyboardInterrupt):
frozen objects are never collected, so a cyclic object dropped after training (the usual
`del model; gc.collect()` in a notebook) would otherwise keep its memory forever.

Rules:
- Only the outermost `train()` freezes; nested calls (resume guards, per-instance wrappers) are
  counted and do nothing.
- If something is already frozen at entry, the caller owns the freeze: Python cannot unfreeze
  only our objects, so we leave the heap alone and do not unfreeze on exit.
- `gc.isenabled()` is never touched.
- `UNSLOTH_GC_FREEZE=0` turns it off (read on each outermost entry).
"""

import functools
import gc
import os
import threading

__all__ = [
    "unsloth_gc_train_scope",
    "patch_trainer_train_gc_scope",
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
            return original_train(self, *args, **kwargs)

    train._unsloth_gc_train_scope = True
    trainer_class.train = train
    return True
