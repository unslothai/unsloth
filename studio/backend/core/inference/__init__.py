# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference submodule - Inference backend for model loading and generation

The default get_inference_backend() returns an InferenceOrchestrator that
delegates to a subprocess. The original InferenceBackend runs inside
the subprocess and can be imported directly from .inference when needed.

Symbols are exposed lazily through ``__getattr__`` (PEP 562) so that
importing a stdlib-only helper from this package (e.g.
``from core.inference._html_to_md import html_to_markdown``) does not
eagerly pull in the orchestrator or the GGUF/llama-server backend.
That matters for the document-extractor HTML path which must keep
working in environments where the inference extras are unavailable or
broken.
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
