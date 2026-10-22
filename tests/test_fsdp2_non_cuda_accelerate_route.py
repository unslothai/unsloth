# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FSDP2=1 on an XPU / NPU / ... install (no CUDA / ROCm torch build) keeps accelerate's FSDP2
wrap, as it did before the native backend existed, instead of being refused: native FSDP2 runs on CUDA /
ROCm or the CPU only. The decision is made from the torch build and the installed vendor packages (no
torch import), so the launch step and Trainer setup agree. An explicit UNSLOTH_FSDP2_BACKEND=native is
still native (and refused at Trainer setup when that device is present)."""

import importlib.util
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CLEAN = (
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_CPU",
    "FSDP_VERSION",
    "FSDP_AUTO_WRAP_POLICY",
    "FSDP_STATE_DICT_TYPE",
    "UNSLOTH_FSDP2_BACKEND",
)


@pytest.fixture
def launch(monkeypatch):
    for key in _CLEAN:
        monkeypatch.delenv(key, raising = False)
    for key, value in dict(UNSLOTH_FSDP2 = "1", WORLD_SIZE = "2", RANK = "0").items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(
        "_fsdp2_launch_under_test", os.path.join(REPO, "unsloth", "fsdp2_launch.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build(monkeypatch, cuda = None, hip = None, xpu = None):
    monkeypatch.setattr(torch.version, "cuda", cuda, raising = False)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.version, "xpu", xpu, raising = False)


def test_xpu_build_routes_to_accelerate_fsdp2(launch, monkeypatch):
    _build(monkeypatch, xpu = "20250101")
    assert launch.unsloth_fsdp2_native_requested() is False
    assert launch.apply_unsloth_fsdp2_launch_env() is True
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert os.environ["FSDP_VERSION"] == "2"
    assert launch.fsdp_version_from_env() == 2


def test_vendor_package_on_a_non_cuda_build_routes_to_accelerate(launch, monkeypatch):
    _build(monkeypatch)
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "torch_npu" else real(name, *a, **k),
    )
    assert launch.unsloth_fsdp2_native_requested() is False
    assert launch.apply_unsloth_fsdp2_launch_env() is True


@pytest.mark.parametrize("build", [dict(cuda = "12.8"), dict(hip = "6.3"), dict(cuda = "12.8", xpu = "1")])
def test_cuda_and_rocm_builds_stay_native(launch, monkeypatch, build):
    _build(monkeypatch, **build)
    assert launch.unsloth_fsdp2_native_requested() is True
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "ACCELERATE_USE_FSDP" not in os.environ


def test_explicit_native_backend_and_cpu_launch_stay_native(launch, monkeypatch):
    _build(monkeypatch, xpu = "20250101")
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "native")
    assert launch.unsloth_fsdp2_native_requested() is True
    monkeypatch.delenv("UNSLOTH_FSDP2_BACKEND")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    assert launch.unsloth_fsdp2_native_requested() is True


def test_torch_build_is_read_without_importing_torch(launch, monkeypatch, tmp_path):
    pkg = tmp_path / "torch"
    pkg.mkdir()
    (pkg / "version.py").write_text(
        "from typing import Optional\n__version__ = '2.11.0+xpu'\n"
        "cuda: Optional[str] = None\nhip: Optional[str] = None\nxpu: Optional[str] = '20250101'\n"
    )
    spec = importlib.machinery.ModuleSpec("torch", None, is_package = True)
    spec.submodule_search_locations = [str(pkg)]
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: spec if name == "torch" else real(name, *a, **k),
    )
    monkeypatch.delitem(sys.modules, "torch")
    try:
        assert launch.unsloth_fsdp2_native_requested() is False
    finally:
        sys.modules["torch"] = torch
