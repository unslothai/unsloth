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
_REAL_POPEN = __import__("subprocess").Popen


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
    [
        {"_gpu_offload_active": False},
        {"_cpu_fallback_reason": "vulkan_startup_crash"},
        {"_arch_gate_forced_cpu": True},
        {"_gpu_memory_mode": "manual", "_gpu_layers": 0},
    ],
    ids = ["landed-on-cpu", "vulkan-cpu-fallback", "arch-gated-cpu", "manual-zero-layers"],
)
def test_a_child_on_cpu_claims_no_fit(state):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend()
    backend._vram_fit_context_length = 65536
    assert backend.vram_fit_context_length == 65536
    for name, value in state.items():
        setattr(backend, name, value)
    assert backend.vram_fit_context_length is None


def test_a_recovered_crash_drops_the_ceiling_its_plan_priced(tmp_path, monkeypatch):
    import subprocess

    accelerator = next(a for a in _matrix.ACCELERATORS if a.label == "nvidia-single")
    backend, gguf = _matrix.cell_backend(
        tmp_path, monkeypatch, _matrix.PLATFORMS[0], accelerator, model_fraction = _matrix.FITS
    )
    healthy = iter([False, True])
    backend._wait_for_health = lambda timeout, **_kw: next(healthy)
    launches = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        launches.append(list(cmd))
        code = 1 if len(launches) == 1 else None
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": code,
                "poll": lambda self: code,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    from unittest.mock import patch

    from core.inference.llama_cpp import GgufLoadIntent

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 0)
        )
    assert len(launches) == 2, launches
    assert backend.max_context_length > _AUTO_OFFLOAD_CTX
    assert backend.vram_fit_context_length is None


def test_a_zero_layer_metal_load_claims_no_fit(tmp_path, monkeypatch):
    def load(sub, **kwargs):
        (tmp_path / sub).mkdir()
        return _metal._launch(
            tmp_path / sub, monkeypatch, n_ctx = 0, metal = True, real_fit = True, **kwargs
        )["backend"]

    assert load("auto").vram_fit_context_length is not None
    manual = load("manual", gpu_memory_mode = "manual", gpu_layers = 0)
    assert manual.vram_fit_context_length is None


def test_a_placement_that_raises_claims_no_fit(tmp_path, monkeypatch):
    accelerator = next(a for a in _matrix.ACCELERATORS if a.label == "nvidia-single")
    backend, gguf = _matrix.cell_backend(
        tmp_path, monkeypatch, _matrix.PLATFORMS[0], accelerator, model_fraction = _matrix.FITS
    )

    def boom(*_a, **_kw):
        raise RuntimeError("selection failed")

    # Raises after the subset sweep priced a ceiling, inside the placement try.
    backend._select_gpus_split_aware = boom
    backend._select_gpus = boom
    captured = _matrix._launch(backend, gguf, n_ctx = 4096)
    assert ("--fit", "on") in zip(captured["cmd"], captured["cmd"][1:])
    assert backend.vram_fit_context_length is None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--device", "none"],
        ["-ngl", "8"],
        ["-ot", "exps=CPU"],
        ["--lora", "adapter.gguf"],
        ["--no-kv-offload"],
    ],
    ids = ["cpu-device", "user-layers", "tensors-on-cpu", "lora-adapter", "kv-on-host"],
)
def test_a_user_placement_override_claims_no_fit(tmp_path, monkeypatch, extra_args):
    (tmp_path / "metal").mkdir()
    (tmp_path / "cuda").mkdir()
    metal = _metal._launch(
        tmp_path / "metal",
        monkeypatch,
        n_ctx = 0,
        metal = True,
        real_fit = True,
        extra_args = extra_args,
    )["backend"]
    assert metal.vram_fit_context_length is None

    accelerator = next(a for a in _matrix.ACCELERATORS if a.label == "nvidia-single")
    backend, gguf = _matrix.cell_backend(
        tmp_path / "cuda",
        monkeypatch,
        _matrix.PLATFORMS[0],
        accelerator,
        model_fraction = _matrix.FITS,
    )
    _matrix._launch(backend, gguf, n_ctx = 0, extra_args = tuple(extra_args))
    assert backend.vram_fit_context_length is None


# Needs both cards, so a selection or split that leaves one empty strands half the credit.
_TWO_CARD_FRACTION = 0.7


def _two_card_load(tmp_path, monkeypatch, **load_kwargs):
    accelerator = next(a for a in _matrix.ACCELERATORS if a.label == "nvidia-multi")
    backend, gguf = _matrix.cell_backend(
        tmp_path, monkeypatch, _matrix.PLATFORMS[0], accelerator, model_fraction = _TWO_CARD_FRACTION
    )
    load_kwargs.setdefault("n_ctx", 0)
    _matrix._launch(backend, gguf, **load_kwargs)
    return backend


def test_a_two_card_fit_is_published(tmp_path, monkeypatch):
    assert _two_card_load(tmp_path, monkeypatch).vram_fit_context_length is not None


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--device", "CUDA0"],
        ["--split-mode", "none"],
        ["--tensor-split", "100,1"],
        ["--fit", "on", "--fit-target", "8192"],
    ],
    ids = ["narrower-device", "split-mode-none", "skewed-split", "user-fitter"],
)
def test_a_placement_narrower_than_the_fit_claims_no_fit(tmp_path, monkeypatch, extra_args):
    backend = _two_card_load(tmp_path, monkeypatch, extra_args = tuple(extra_args))
    assert backend.vram_fit_context_length is None


def test_an_inherited_projector_claims_no_fit(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_MMPROJ_URL", "https://example.invalid/mmproj.gguf")
    assert _two_card_load(tmp_path, monkeypatch).vram_fit_context_length is None


def test_a_gpu_ids_pin_ignores_the_device_env_it_clears(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
    backend = _two_card_load(tmp_path, monkeypatch, gpu_ids = [0, 1])
    assert backend.vram_fit_context_length is not None


def test_a_one_card_device_pin_under_a_two_card_cap_claims_no_fit(tmp_path, monkeypatch):
    # The explicit context fits one card, so the selection credits one; the cap pooled two.
    backend = _two_card_load(tmp_path, monkeypatch, n_ctx = 2048, extra_args = ("--device", "CUDA0"))
    assert backend.vram_fit_context_length is None
