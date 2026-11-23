# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch-only gate tests for core.inference.diffusion_ref_preprocess (no diffusers / Pillow needed)."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_ref_preprocess as rp  # noqa: E402


def test_env_switch(monkeypatch):
    monkeypatch.setenv(rp.ENV, "0")
    assert not rp.enabled()
    monkeypatch.setenv(rp.ENV, "1")
    assert rp.enabled()
    monkeypatch.delenv(rp.ENV)
    assert rp.enabled()


def test_only_flux2_processor(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)

    class P:
        image_processor = object()

    assert rp.install(P()) is False


def test_kernel_crop_dims_not_specialized():
    src = (BACKEND / "core" / "inference" / "diffusion_lanczos_triton.py").read_text(
        encoding = "utf-8"
    )
    assert "crop_width: tl.constexpr" not in src and "crop_height: tl.constexpr" not in src
    assert "Apache License 2.0" in src and "SGLang project" in src


def test_installed_at_load():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    assert "from .diffusion_ref_preprocess import install as install_ref_preprocess" in src


def test_rocm_torch_never_takes_the_gpu_path(monkeypatch):
    import torch

    from core.inference import rocm_bf16

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda _t: True)
    assert rp._cuda_ok() is False
