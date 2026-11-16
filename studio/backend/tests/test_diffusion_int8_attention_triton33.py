# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""NVIDIA Triton 3.3 (torch 2.7) cannot compile the INT8 attention quantiser, so the Triton path is skipped there."""

import types

import pytest

pytest.importorskip("torch")

from core.inference import diffusion_int8_attention as I  # noqa: E402
from core.inference import diffusion_int8_attention_triton as T  # noqa: E402


@pytest.mark.parametrize(
    "version, rocm, available",
    [
        ("3.3.1", False, False),
        ("3.3.0", False, False),
        ("3.3.1", True, True),
        ("3.2.0", False, True),
        ("3.4.0", False, True),
        ("3.7.1", False, True),
        ("dev", False, True),
    ],
)
def test_triton_available_skips_only_nvidia_triton_33(monkeypatch, version, rocm, available):
    monkeypatch.setattr(T, "triton", types.SimpleNamespace(__version__ = version))
    monkeypatch.setattr(T, "_rocm_build", lambda: rocm)
    I.triton_available.cache_clear()
    try:
        assert I.triton_available() is available
    finally:
        I.triton_available.cache_clear()


def test_no_triton_is_unavailable(monkeypatch):
    monkeypatch.setattr(T, "triton", None)
    I.triton_available.cache_clear()
    try:
        assert I.triton_available() is False
    finally:
        I.triton_available.cache_clear()
