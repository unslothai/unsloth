# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``vram_fit_context_length`` names only a ceiling a fit priced inside the budget (#12571).

``max_context_length`` also carries anchors and floors nobody measured (the Auto offload
context, the Metal fit floor, native when no fit ran), so mirroring it would advertise a
no-spill capacity the machine does not have.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    # The matrix module declares dataclasses, which look their module up by name.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_matrix = _load("_matrix_for_vram_fit_ctx", "test_auto_offload_ctx_platform_matrix.py")
_metal = _load("_metal_for_vram_fit_ctx", "test_metal_explicit_context_guard.py")

from core.inference.llama_cpp import _AUTO_OFFLOAD_CTX, _FIT_MIN_CTX  # noqa: E402

GB = 1024**3


def _loaded(tmp_path, monkeypatch, platform, accelerator, **kwargs):
    backend, gguf = _matrix.cell_backend(tmp_path, monkeypatch, platform, accelerator, **kwargs)
    _matrix._launch(backend, gguf, n_ctx = 0)
    return backend


@pytest.mark.parametrize("platform,accelerator", _matrix.MATRIX)
def test_a_model_that_fits_publishes_its_fitted_ceiling(
    tmp_path, monkeypatch, platform, accelerator
):
    backend = _loaded(tmp_path, monkeypatch, platform, accelerator, model_fraction = _matrix.FITS)
    if accelerator.memory or accelerator.apple_budget_bytes:
        assert backend.vram_fit_context_length == backend.max_context_length
        assert backend.vram_fit_context_length > _AUTO_OFFLOAD_CTX
    else:
        # No device and no Metal budget: nothing was fitted, so nothing is claimed.
        assert backend.vram_fit_context_length is None


@pytest.mark.parametrize("platform,accelerator", _matrix.MATRIX)
def test_an_overflowing_model_claims_no_fit(tmp_path, monkeypatch, platform, accelerator):
    backend = _loaded(tmp_path, monkeypatch, platform, accelerator)
    assert backend.vram_fit_context_length is None
    if accelerator.memory:
        assert backend.max_context_length == _AUTO_OFFLOAD_CTX
    elif accelerator.apple_budget_bytes:
        assert backend.max_context_length == _FIT_MIN_CTX


def test_an_unmeasurable_metal_load_claims_no_fit(tmp_path, monkeypatch):
    captured = _metal._launch(
        tmp_path,
        monkeypatch,
        n_ctx = 0,
        metal = True,
        can_estimate_kv = False,
        real_fit = True,
        budget_bytes = 24 * GB,
    )
    backend = captured["backend"]
    assert backend.max_context_length == _FIT_MIN_CTX
    assert backend.vram_fit_context_length is None


def test_unload_forgets_the_fit(tmp_path, monkeypatch):
    accelerator = next(a for a in _matrix.ACCELERATORS if a.label == "nvidia-single")
    platform = _matrix.PLATFORMS[0]
    backend = _loaded(tmp_path, monkeypatch, platform, accelerator, model_fraction = _matrix.FITS)
    assert backend.vram_fit_context_length is not None
    backend.unload_model()
    assert backend.vram_fit_context_length is None


@pytest.mark.parametrize(
    "state",
    [{"_gpu_offload_active": False}, {"_cpu_fallback_reason": "vulkan_startup_crash"}],
    ids = ["landed-on-cpu", "vulkan-cpu-fallback"],
)
def test_a_child_on_cpu_claims_no_fit(state):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._vram_fit_context_length = 65536
    assert backend.vram_fit_context_length == 65536
    for name, value in state.items():
        setattr(backend, name, value)
    assert backend.vram_fit_context_length is None
