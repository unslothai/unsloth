# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A generation counter for "Studio changed what is resident on the GPUs".

Bumped when Studio loads or unloads a model, or starts or stops training. The cached
nvidia-smi reads in utils/hardware/gpu_query.py refuse to serve a live memory sample taken
under an older generation to anything that decides placement or fit. Kept free of imports
so the model backends can decorate their load/unload methods at class-definition time
without pulling in utils.hardware.
"""

import functools
import threading
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound = Callable[..., Any])

_lock = threading.Lock()
_generation = 0


def generation() -> int:
    return _generation


def invalidate_gpu_memory(reason: str = "") -> int:
    """Mark every GPU memory reading taken before now as unusable for a fit decision."""
    global _generation
    with _lock:
        _generation += 1
        return _generation


def invalidates_gpu_memory(reason: str) -> Callable[[_F], _F]:
    """Decorator: invalidate on entry and again on exit (success or failure), so neither
    a check made during the call nor one made right after it reuses an earlier sample."""

    def decorate(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            invalidate_gpu_memory(reason)
            try:
                return fn(*args, **kwargs)
            finally:
                invalidate_gpu_memory(reason)

        return wrapper  # type: ignore[return-value]

    return decorate
