# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the compiled GGUF dequant accelerator (``diffusion_gguf_compile.py``).

Covers install/uninstall idempotency + exact reversibility, the kill-switch, and the
on-by-default behaviour. Runs on CPU -- patching the module attribute is lazy
(torch.compile only traces on the first real call).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_compile as gc  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    # Always start and end from a clean, unpatched state so tests do not leak the
    # process-wide patch into each other.
    gc.uninstall_all()
    yield
    gc.uninstall_all()


def test_compiled_dequant_install_uninstall_reversible():
    orig = gguf_utils.dequantize_gguf_tensor
    assert gc.is_compiled_dequant_installed() is False

    assert gc.install_compiled_dequant() is True
    assert gc.is_compiled_dequant_installed() is True
    # The module attribute is now a different (compiled) callable...
    assert gguf_utils.dequantize_gguf_tensor is not orig
    # ...idempotent: a second install is a no-op, attribute unchanged.
    patched = gguf_utils.dequantize_gguf_tensor
    assert gc.install_compiled_dequant() is True
    assert gguf_utils.dequantize_gguf_tensor is patched

    gc.uninstall_compiled_dequant()
    assert gc.is_compiled_dequant_installed() is False
    # Exact original restored.
    assert gguf_utils.dequantize_gguf_tensor is orig
    # Uninstall is idempotent.
    gc.uninstall_compiled_dequant()
    assert gguf_utils.dequantize_gguf_tensor is orig


def test_compiled_dequant_kill_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT", "0")
    orig = gguf_utils.dequantize_gguf_tensor
    assert gc.install_compiled_dequant() is False
    assert gc.is_compiled_dequant_installed() is False
    assert gguf_utils.dequantize_gguf_tensor is orig


def test_compiled_dequant_on_by_default(monkeypatch):
    # The compiled dequant is the real win, so it is ON without any env opt-in.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_GGUF_COMPILE_DEQUANT", raising=False)
    assert gc.install_compiled_dequant() is True
    assert gc.is_compiled_dequant_installed() is True


def test_uninstall_all(monkeypatch):
    orig = gguf_utils.dequantize_gguf_tensor
    gc.install_compiled_dequant()
    assert gc.is_installed() is True
    gc.uninstall_all()
    assert gc.is_installed() is False
    assert gguf_utils.dequantize_gguf_tensor is orig
