# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference submodule - backend for model loading and generation.

The default get_inference_backend() returns an InferenceOrchestrator that
delegates to a subprocess. The original InferenceBackend runs inside the
subprocess and can be imported directly from .inference when needed.

Public names are resolved lazily (PEP 562): importing this package -- or a
dependency-light leaf like ``core.inference.chat_eos`` -- must NOT eagerly pull
the orchestrator / llama_cpp import chain (httpx, subprocess plumbing, the ML
backend and its Studio dependencies). Those load only when a public name is
actually accessed, so standalone helpers stay unit-testable without the full
inference stack.
"""

from typing import TYPE_CHECKING

__all__ = [
    "InferenceBackend",
    "InferenceOrchestrator",
    "get_inference_backend",
    "LlamaCppBackend",
]

# name -> (submodule, attribute); InferenceBackend aliases InferenceOrchestrator.
_LAZY_ATTRS = {
    "InferenceOrchestrator": ("orchestrator", "InferenceOrchestrator"),
    "InferenceBackend": ("orchestrator", "InferenceOrchestrator"),
    "get_inference_backend": ("orchestrator", "get_inference_backend"),
    "LlamaCppBackend": ("llama_cpp", "LlamaCppBackend"),
}


def __getattr__(name):
    try:
        submodule, attr = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{submodule}"), attr)
    globals()[name] = value  # cache so later access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # keep static analysers / IDEs aware of the lazy names
    from .llama_cpp import LlamaCppBackend
    from .orchestrator import InferenceOrchestrator, get_inference_backend
    InferenceBackend = InferenceOrchestrator
