# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export checkpoint loading must shard across every visible GPU (#7053): the
``device_map="sequential"`` loader default stacks the whole model on GPU0 and OOMs
while the other GPUs sit empty. The loader now passes ``device_map="balanced"``, but
only on a real multi-GPU CUDA/ROCm host, so single-GPU, CPU and MLX are untouched."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Reuse the absolute-paths test's stub harness for loading core/export/export.py
# without torch/unsloth.
from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    return _load_module("test_core_export_backend_device_map", "core/export/export.py", monkeypatch)


def _stub_hardware(monkeypatch, visible, device_map):
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: visible, raising = False)
    monkeypatch.setattr(hw, "get_device_map", lambda ids: device_map, raising = False)


# ── _multi_gpu_device_map_kwargs ──


def test_multi_gpu_host_gets_balanced(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0, 1, 2], "balanced")
    assert mod._multi_gpu_device_map_kwargs() == {"device_map": "balanced"}


def test_single_gpu_host_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0], "sequential")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_non_balanced_resolution_keeps_loader_default(monkeypatch):
    # >1 visible id but a non-CUDA device resolves to "sequential": pass nothing.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    _stub_hardware(monkeypatch, [0, 1], "sequential")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_uuid_mig_mask_falls_back_to_count_detection(monkeypatch):
    # UUID/MIG masks resolve to NO numeric ids ([]), but get_device_map(None) still
    # detects >1 GPU, so the empty list must route there, not to the loader default.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [], raising = False)
    monkeypatch.setattr(
        hw,
        "get_device_map",
        lambda ids: "balanced" if ids is None else "sequential",
        raising = False,
    )
    assert mod._multi_gpu_device_map_kwargs() == {"device_map": "balanced"}


def test_no_visible_gpus_keeps_loader_default(monkeypatch):
    # Empty mask / CPU host: get_device_map(None) resolves "sequential" -> {}.
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [], raising = False)
    monkeypatch.setattr(hw, "get_device_map", lambda ids: "sequential", raising = False)
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_mlx_host_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    # The stubs set _IS_MLX = True; even a multi-GPU view must yield no device_map.
    _stub_hardware(monkeypatch, [0, 1], "balanced")
    assert mod._multi_gpu_device_map_kwargs() == {}


def test_hardware_probe_failure_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    hw = sys.modules["utils.hardware"]

    def _boom():
        raise RuntimeError("no GPUs")

    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", _boom, raising = False)
    assert mod._multi_gpu_device_map_kwargs() == {}


# ── load_checkpoint forwards the kwargs to from_pretrained ──


class _RecordingLoader:
    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        return types.SimpleNamespace(), types.SimpleNamespace()


def _load_text_checkpoint(monkeypatch, tmp_path, device_map_kwargs):
    mod = _export_mod(monkeypatch)
    _RecordingLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _RecordingLoader)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: device_map_kwargs)

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    assert len(_RecordingLoader.calls) == 1
    return _RecordingLoader.calls[0]


def test_load_checkpoint_forwards_balanced_device_map(monkeypatch, tmp_path):
    kwargs = _load_text_checkpoint(monkeypatch, tmp_path, {"device_map": "balanced"})
    assert kwargs["device_map"] == "balanced"


def test_load_checkpoint_omits_device_map_on_single_gpu(monkeypatch, tmp_path):
    kwargs = _load_text_checkpoint(monkeypatch, tmp_path, {})
    assert "device_map" not in kwargs  # loader default (sequential) untouched


# ── a load that succeeds but offloads to CPU/disk ──


def test_cpu_offloaded_modules_counts_cpu_and_disk(monkeypatch):
    mod = _export_mod(monkeypatch)
    model = types.SimpleNamespace(hf_device_map = {"a": 0, "b": "cpu", "c": 1, "d": "disk"})
    assert mod._cpu_offloaded_modules(model) == 2


def test_cpu_offloaded_modules_ignores_gpu_only_and_missing_maps(monkeypatch):
    mod = _export_mod(monkeypatch)
    assert mod._cpu_offloaded_modules(types.SimpleNamespace(hf_device_map = {"a": 0})) == 0
    assert mod._cpu_offloaded_modules(types.SimpleNamespace(hf_device_map = None)) == 0
    assert mod._cpu_offloaded_modules(types.SimpleNamespace()) == 0


class _SpillThenCleanLoader:
    """First call offloads to CPU (bf16 accepts it silently), second is clean."""

    calls: list[dict] = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        device_map = {"model.layers.0": 0} if len(cls.calls) > 1 else {"model.layers.0": "cpu"}
        return types.SimpleNamespace(hf_device_map = device_map), types.SimpleNamespace()


def _run_spill_loader(monkeypatch, tmp_path, device_map_kwargs):
    mod = _export_mod(monkeypatch)
    _SpillThenCleanLoader.calls = []
    monkeypatch.setattr(mod, "FastLanguageModel", _SpillThenCleanLoader)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: device_map_kwargs)

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    return ok, message, _SpillThenCleanLoader.calls


def test_successful_load_that_offloads_to_cpu_retries_single_device(monkeypatch, tmp_path):
    # Nothing raises, so only hf_device_map catches it; the parameters would otherwise
    # stay on meta and kill the export inside safetensors.
    ok, message, calls = _run_spill_loader(monkeypatch, tmp_path, {"device_map": "balanced"})
    assert ok, message
    assert len(calls) == 2
    assert calls[0]["device_map"] == "balanced"
    assert "device_map" not in calls[1]


def test_single_gpu_offload_is_left_alone(monkeypatch, tmp_path):
    # No multi-GPU map was requested, so there is nothing to retry on.
    ok, message, calls = _run_spill_loader(monkeypatch, tmp_path, {})
    assert ok, message
    assert len(calls) == 1


def test_retry_result_is_kept_even_if_it_also_offloads(monkeypatch, tmp_path):
    # The retry runs with _device_map_override set, so it must never recurse again.
    mod = _export_mod(monkeypatch)

    class _AlwaysSpills:
        calls: list[dict] = []

        @classmethod
        def from_pretrained(cls, **kwargs):
            cls.calls.append(kwargs)
            return types.SimpleNamespace(hf_device_map = {"a": "cpu"}), types.SimpleNamespace()

    monkeypatch.setattr(mod, "FastLanguageModel", _AlwaysSpills)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {"device_map": "balanced"})

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    assert len(_AlwaysSpills.calls) == 2
