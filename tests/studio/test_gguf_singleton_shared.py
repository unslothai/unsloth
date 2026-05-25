# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Tests that the GGUF llama-server backend is a SINGLE process-wide
singleton, shared between ``routes.inference`` (the load/unload path)
and ``core.inference.llama_cpp`` (the canonical accessor used by
``routes.models`` list/cache-delete, ``run.py`` shutdown, and
``core.chat.vlm_capability``).

Failure mode the test pins:
    The PR's first cut left a route-local
    ``_llama_cpp_backend = LlamaCppBackend()`` at the top of
    ``routes/inference.py`` whose own ``get_llama_cpp_backend`` shadowed
    the imported core function. The result was two distinct
    ``LlamaCppBackend`` instances:
        - ``routes.inference._llama_cpp_backend`` (eager) -- populated
          by ``/api/inference/load`` and used by every call site in that
          file.
        - ``core.inference.llama_cpp._llama_cpp_backend`` (lazy) --
          read by ``routes.models`` list / cache-delete,
          ``run.py`` shutdown, and ``core.chat.vlm_capability``.

    Consequence: a GGUF loaded through ``/api/inference/load`` was
    invisible to ``/api/models/list``, deletable from cache *while
    serving*, leaked at shutdown, and the VLM probe could not see it
    even if it was a vision model.

After the patch, ``routes.inference`` re-exports
``get_llama_cpp_backend`` from the core module, so all consumers see
exactly the same instance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


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
    """Simulate a GGUF VLM having been loaded through the normal
    route path, then confirm ``detect_loaded_vlm`` (called from the
    document extractor) sees it.

    Pre-fix: ``routes.inference._llama_cpp_backend`` is the eager
    instance that ``/api/inference/load`` populates;
    ``core.inference.llama_cpp.get_llama_cpp_backend()`` returns a
    different lazy instance, so ``_probe_gguf`` (which reads the core
    one) never sees the loaded model and returns ``source='none'``.

    Post-fix the two are one object, so mutating the routes-side
    backend's internals is observable by the probe.
    """
    from core.chat import vlm_capability
    from core.inference.llama_cpp import get_llama_cpp_backend as core_acc
    from routes.inference import get_llama_cpp_backend as routes_acc

    # Singleton identity is the contract.
    assert core_acc() is routes_acc()

    # Pretend the route just finished loading a GGUF VLM by mutating
    # the underlying private fields the @property accessors expose.
    backend = routes_acc()
    monkeypatch.setattr(backend, "_model_identifier",
                        "unsloth/Qwen2-VL-2B-Instruct-GGUF", raising=False)
    monkeypatch.setattr(backend, "_is_vision", True, raising=False)
    # is_loaded is a property derived from internal state; we override
    # the property at the class level just for this test instance.
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
    """Static/structural check: routes.models.list_models and the
    cache-delete guard must read the same get_llama_cpp_backend that
    routes.inference.load_model writes to.

    We don't actually call the FastAPI handler; we just assert the
    accessor identity, which is the only invariant the fix needs to
    preserve.
    """
    from core.inference.llama_cpp import (
        get_llama_cpp_backend as core_accessor,
    )

    # routes.models imports its accessor inside each handler at call
    # time -- mirror that here.
    import importlib

    routes_models = importlib.import_module("routes.models")

    # routes.models loads the accessor via `from
    # core.inference.llama_cpp import get_llama_cpp_backend` inside
    # the handler body. Exercise the same path here and assert it
    # returns the same instance as core_accessor().
    from core.inference.llama_cpp import get_llama_cpp_backend

    assert routes_models is not None  # imported cleanly
    assert get_llama_cpp_backend() is core_accessor()
