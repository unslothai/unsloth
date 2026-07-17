# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export checkpoint loading must shard across every visible GPU on a multi-GPU
host (#7053): unsloth's ``from_pretrained`` defaults to ``device_map="sequential"``,
which stacks the whole model on GPU0 and OOMs while the other GPUs sit empty.
The export loader now passes ``device_map="balanced"`` -- but only on a real
multi-GPU CUDA/ROCm host, so single-GPU, CPU, and MLX loads keep the loader
default untouched."""

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

# Reuse the export-backend stub harness from the absolute-paths test so the two
# stay in lockstep about how core/export/export.py is loaded without torch/unsloth.
from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    return _load_module(
        "test_core_export_backend_device_map", "core/export/export.py", monkeypatch
    )


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


def test_mlx_host_keeps_loader_default(monkeypatch):
    mod = _export_mod(monkeypatch)
    # _install_export_backend_stubs sets _IS_MLX = True already; even a stubbed
    # multi-GPU view must not produce a device_map on MLX.
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
