# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inference submodule - backend for model loading and generation.

get_inference_backend() returns an InferenceOrchestrator that delegates to a
subprocess; the original InferenceBackend runs inside it and can be imported
from .inference directly.

Symbols are lazy (PEP 562 ``__getattr__``) so importing a stdlib-only helper
(e.g. ``core.inference._html_to_md``) never pulls in the orchestrator or the
GGUF backend - the document-extractor HTML path must work even when the
inference extras are broken or missing.
"""

from typing import Any

__all__ = [
    "InferenceBackend",
    "InferenceOrchestrator",
    "get_inference_backend",
    "get_llama_cpp_backend",
    "LlamaCppBackend",
]


def __getattr__(name: str) -> Any:
    if name in ("InferenceOrchestrator", "get_inference_backend", "InferenceBackend"):
        from .orchestrator import InferenceOrchestrator, get_inference_backend

        globals()["InferenceOrchestrator"] = InferenceOrchestrator
        globals()["get_inference_backend"] = get_inference_backend
        globals()["InferenceBackend"] = InferenceOrchestrator
        return globals()[name]
    if name in ("LlamaCppBackend", "get_llama_cpp_backend"):
        from .llama_cpp import LlamaCppBackend, get_llama_cpp_backend

        globals()["LlamaCppBackend"] = LlamaCppBackend
        globals()["get_llama_cpp_backend"] = get_llama_cpp_backend
        return globals()[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
