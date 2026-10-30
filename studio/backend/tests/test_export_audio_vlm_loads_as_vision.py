# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An audio_vlm checkpoint that takes images exports through the vision load, which keeps its processor."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


class _Loader:
    def __init__(self):
        self.calls = []

    def from_pretrained(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(), types.SimpleNamespace()


def _load(monkeypatch, tmp_path, audio_type, is_vision):
    _install_export_backend_stubs(monkeypatch)
    mod = _load_module("test_core_export_audio_vlm_vision", "core/export/export.py", monkeypatch)
    text, vision = _Loader(), _Loader()
    monkeypatch.setattr(mod, "FastLanguageModel", text)
    monkeypatch.setattr(mod, "FastVisionModel", vision)
    monkeypatch.setattr(mod, "_IS_MLX", False)
    for name in ("PeftModel", "PeftModelForCausalLM"):
        monkeypatch.setattr(mod, name, type(name, (), {}), raising = False)
    monkeypatch.setattr(mod, "detect_audio_type", lambda *a, **k: audio_type)
    monkeypatch.setattr(mod, "is_vision_model", lambda *a, **k: is_vision)
    monkeypatch.setattr(mod, "_hf_offline", lambda *a, **k: False)
    monkeypatch.setattr(mod, "_offline_window_if", lambda flag: contextlib.nullcontext())
    monkeypatch.setattr(mod, "_multi_gpu_device_map_kwargs", lambda: {})
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    backend = mod.ExportBackend.__new__(mod.ExportBackend)
    backend.cleanup_memory = lambda: None
    ok, message = backend.load_checkpoint(str(checkpoint))
    assert ok, message
    return backend, text, vision


@pytest.mark.parametrize(
    "audio_type, is_vision, expect_vision",
    [
        ("audio_vlm", True, True),  # LFM3-Omni, Gemma 3n/4
        ("audio_vlm", False, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_audio_vlm_with_vision_takes_the_vision_load(
    monkeypatch, tmp_path, audio_type, is_vision, expect_vision
):
    backend, text, vision = _load(monkeypatch, tmp_path, audio_type, is_vision)
    assert backend.is_vision is expect_vision
    assert (len(vision.calls), len(text.calls)) == ((1, 0) if expect_vision else (0, 1))
