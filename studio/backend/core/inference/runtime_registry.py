# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Process-local handles for inference runtimes owned by another core layer.

The GGUF backend is constructed by the inference router today. Internal
workflows use this registry instead of importing a FastAPI route or fabricating
an HTTP request. Registration carries no credentials and does not load a model.
"""

from __future__ import annotations

import threading
from typing import Any, Optional


_LOCK = threading.Lock()
_llama_cpp_backend: Optional[Any] = None


def register_llama_cpp_backend(backend: Optional[Any]) -> None:
    global _llama_cpp_backend
    with _LOCK:
        _llama_cpp_backend = backend


def peek_llama_cpp_backend() -> Optional[Any]:
    with _LOCK:
        return _llama_cpp_backend


__all__ = ["peek_llama_cpp_backend", "register_llama_cpp_backend"]
