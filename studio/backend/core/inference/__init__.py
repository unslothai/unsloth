# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference submodule - backend for model loading and generation.

The default get_inference_backend() returns an InferenceOrchestrator that
delegates to a subprocess. The original InferenceBackend runs inside the
subprocess and can be imported directly from .inference when needed.

Imports are LAZY (via __getattr__, PEP 562) so dependency-light leaf modules
(e.g. chat_template_helpers) can be imported without pulling in the heavy
orchestrator / llama_cpp stack (loggers -> structlog, httpx, ...). Mirrors the
lazy-import pattern in core/__init__.py.
"""

__all__ = [
    "InferenceBackend",
    "InferenceOrchestrator",
    "get_inference_backend",
    "LlamaCppBackend",
]


def __getattr__(name):
    # Expose InferenceOrchestrator as InferenceBackend for backward compat.
    if name in ("InferenceBackend", "InferenceOrchestrator", "get_inference_backend"):
        from .orchestrator import InferenceOrchestrator, get_inference_backend

        globals()["InferenceOrchestrator"] = InferenceOrchestrator
        globals()["InferenceBackend"] = InferenceOrchestrator
        globals()["get_inference_backend"] = get_inference_backend
        return globals()[name]

    if name == "LlamaCppBackend":
        from .llama_cpp import LlamaCppBackend
        globals()["LlamaCppBackend"] = LlamaCppBackend
        return LlamaCppBackend

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
