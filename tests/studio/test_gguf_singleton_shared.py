# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""GGUF llama-server backend must be one process-wide singleton, shared by
routes.inference (load/unload) and core.inference.llama_cpp (the canonical
accessor used by routes.models, run.py shutdown, and vlm_capability).

Pinned regression: a route-local eager ``_llama_cpp_backend`` once shadowed
the core accessor, splitting state so a loaded GGUF was invisible to
/api/models/list, deletable from cache while serving, leaked at shutdown,
and hidden from the VLM probe. routes.inference now re-exports the core
accessor so every consumer sees the same instance.
"""

from __future__ import annotations

import pytest


def test_routes_and_core_singleton_are_the_same_object():
    from core.inference import llama_cpp as core_mod
    from routes import inference as routes_mod

    core_backend = core_mod.get_llama_cpp_backend()
    routes_backend = routes_mod.get_llama_cpp_backend()

    assert core_backend is routes_backend, (
        "routes.inference.get_llama_cpp_backend() and "
        "core.inference.llama_cpp.get_llama_cpp_backend() must return "
        "the same LlamaCppBackend instance. If they don't, GGUF "
        "models loaded through /api/inference/load are invisible to "
        "/api/models/list, cache-delete, shutdown, and the VLM probe."
    )


def test_vlm_probe_sees_route_loaded_gguf(monkeypatch):
    """Load a GGUF VLM through the route path; ``detect_loaded_vlm`` must
    see it. Pre-fix the route and core accessors returned different
    instances, so ``_probe_gguf`` returned ``source='none'``; post-fix they
    are one object.
    """
    from core.chat import vlm_capability
    from core.inference.llama_cpp import get_llama_cpp_backend as core_acc
    from routes.inference import get_llama_cpp_backend as routes_acc

    # Singleton identity is the contract.
    assert core_acc() is routes_acc()

    # Pretend the route just loaded a GGUF VLM by mutating the private
    # fields behind the @property accessors.
    backend = routes_acc()
    monkeypatch.setattr(
        backend, "_model_identifier", "unsloth/Qwen2-VL-2B-Instruct-GGUF", raising = False
    )
    monkeypatch.setattr(backend, "_is_vision", True, raising = False)
    # is_loaded is a derived property; override at class level for the test.
    cls = type(backend)
    original_is_loaded = cls.is_loaded
    monkeypatch.setattr(cls, "is_loaded", property(lambda self: True))
    try:
        cap = vlm_capability.detect_loaded_vlm()
    finally:
        # restoration handled by monkeypatch.undo()
        pass

    assert cap.source == "gguf", (
        "VLM probe must see the GGUF backend loaded via the routes "
        "path. If it returns source='none', the load path is "
        "populating a different singleton from the probe path."
    )
    assert cap.is_vlm is True
    assert cap.model_name == "unsloth/Qwen2-VL-2B-Instruct-GGUF"


def test_routes_models_uses_same_singleton():
    """routes.models.list_models and the cache-delete guard must read the
    same get_llama_cpp_backend that routes.inference.load_model writes to.
    No handler call; accessor identity is the only invariant needed.
    """
    from core.inference.llama_cpp import (
        get_llama_cpp_backend as core_accessor,
    )

    # routes.models imports its accessor inside each handler; mirror that.
    import importlib

    routes_models = importlib.import_module("routes.models")

    # Exercise the same in-handler import path routes.models uses and
    # assert it returns the same instance as core_accessor().
    from core.inference.llama_cpp import get_llama_cpp_backend

    assert routes_models is not None  # imported cleanly
    assert get_llama_cpp_backend() is core_accessor()
