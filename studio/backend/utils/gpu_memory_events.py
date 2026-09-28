# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generation counter bumped when Studio changes what is resident on the GPUs.
Import-free so backends can decorate load/unload without pulling in utils.hardware."""

import functools
import threading
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound = Callable[..., Any])

_lock = threading.Lock()
_generation = 0


def generation() -> int:
    return _generation


def invalidate_gpu_memory(reason: str = "") -> int:
    global _generation
    with _lock:
        _generation += 1
        return _generation


def invalidates_gpu_memory(reason: str) -> Callable[[_F], _F]:
    """Invalidate on entry and on exit (success or failure)."""

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
