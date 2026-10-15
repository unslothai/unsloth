# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where an audio model's weights go: the accelerator, or plain CPU RAM.

Audio loads take an accelerator whenever one exists. That is right until the GPU
is the scarce resource (a resident chat model, a training run, a card too small
for the checkpoint), and Whisper and the smaller TTS models run fine on CPU.

Values match ``RAG_EMBED_DEVICE`` (``core/rag/config.py``):

``auto``  detect as before.
``cpu``   force CPU RAM, even with a working accelerator.
``gpu``   prefer the accelerator. The existing CPU retry after a failed load
          still applies, so this is a preference and not a guarantee.

``UNSLOTH_AUDIO_DEVICE`` supplies the default for a request that names none.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = [
    "AUDIO_DEVICE_CHOICES",
    "audio_device_default",
    "audio_device_forces_cpu",
    "mask_accelerators_for_cpu_audio",
    "normalize_audio_device",
]

AUDIO_DEVICE_CHOICES = ("auto", "cpu", "gpu")

_CPU_ALIASES = frozenset({"cpu", "ram", "cpu_ram", "system", "system_ram"})
_GPU_ALIASES = frozenset(
    {"gpu", "cuda", "rocm", "hip", "xpu", "mps", "metal", "accelerator", "accel"}
)


def normalize_audio_device(value: Optional[str]) -> str:
    """Unknown values become auto, but HTTP models pin the canonical values so a typo is a 422."""
    text = str(value or "").strip().lower()
    if not text:
        return "auto"
    if text in _CPU_ALIASES:
        return "cpu"
    if text in _GPU_ALIASES:
        return "gpu"
    if text == "auto":
        return "auto"
    return "auto"


def audio_device_default() -> str:
    """Does not apply to GGUF TTS models, which take placement from gpu_memory_mode and gpu_layers."""
    return normalize_audio_device(os.environ.get("UNSLOTH_AUDIO_DEVICE"))


def audio_device_forces_cpu(value: Optional[str]) -> bool:
    """None falls back to the environment default, so older callers still honour a server-wide setting."""
    if value is None:
        return audio_device_default() == "cpu"
    return normalize_audio_device(value) == "cpu"


def audio_load_runs_on_cpu(audio_type: Optional[str], value: Optional[str]) -> bool:
    """True when a native audio load of this type ends up in CPU RAM: asked for, or a GGUF
    audio model on an audio.cpp runtime that only runs on the CPU (Auto cannot place it on a GPU)."""
    if audio_device_forces_cpu(value):
        return True
    from core.inference.audio_cpp_models import AUDIO_CPP_AUDIO_TYPES

    if audio_type not in AUDIO_CPP_AUDIO_TYPES:
        return False
    from core.inference.audio_cpp_server import runtime_runs_on_cpu

    return runtime_runs_on_cpu()


def mask_accelerators_for_cpu_audio(env: dict) -> None:
    """Call before importing torch: masking first stops detect_hardware() creating a CUDA context."""
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = "-1"
