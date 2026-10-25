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
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_AT_IMPORT = dict(os.environ)

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
    # apply_unsloth_fsdp2_launch_env writes os.environ directly; monkeypatch.delenv records nothing for
    # an absent key, so restore the whole environment or the FSDP launch env leaks into later tests.
    saved = dict(os.environ)
    yield _launch_module(monkeypatch)
    os.environ.clear()
    os.environ.update(saved)


def _launch_module(monkeypatch):
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


def _build(
    monkeypatch,
    cuda = None,
    hip = None,
    xpu = None,
):
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


def test_xpu_version_label_with_a_null_runtime_marker(launch, monkeypatch):
    # A +xpu wheel can carry torch.version.xpu = None: the version label is the build marker.
    _build(monkeypatch)
    monkeypatch.setattr(torch.version, "__version__", "2.9.1+xpu", raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.1+xpu")
    assert launch.non_cuda_install() == "Intel XPU"
    assert launch.unsloth_fsdp2_native_requested() is False
    assert launch.apply_unsloth_fsdp2_launch_env() is True


def test_xpu_version_label_read_without_importing_torch(launch, monkeypatch, tmp_path):
    pkg = tmp_path / "torch"
    pkg.mkdir()
    (pkg / "version.py").write_text(
        "from typing import Optional\n__version__ = '2.9.1+xpu'\n"
        "cuda: Optional[str] = None\nhip: Optional[str] = None\nxpu: Optional[str] = None\n"
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
        assert launch.non_cuda_install() == "Intel XPU"
    finally:
        sys.modules["torch"] = torch


def test_cpu_wheel_label_is_not_xpu(launch, monkeypatch):
    _build(monkeypatch)
    monkeypatch.setattr(torch.version, "__version__", "2.9.1+cpu", raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.1+cpu")
    assert launch.non_cuda_install() is None
    assert launch.unsloth_fsdp2_native_requested() is True


def test_unsloth_fsdp2_native_on_xpu_stays_native_and_is_refused(launch, monkeypatch):
    _build(monkeypatch, xpu = "20250101")
    monkeypatch.setenv("UNSLOTH_FSDP2", "native")
    assert launch.unsloth_fsdp2_native_requested() is True
    assert launch.apply_unsloth_fsdp2_launch_env() is False
    assert "ACCELERATE_USE_FSDP" not in os.environ
    # Trainer setup with a live XPU device: refused with the accelerate alternative named.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "xpu", types.SimpleNamespace(is_available = lambda: True), raising = False
    )
    with pytest.raises(RuntimeError, match = "UNSLOTH_FSDP2_BACKEND=accelerate"):
        launch.refuse_native_backend_conflicts(None, uses_cpu = False)
    # The same explicit request with a CPU Trainer runs natively on the CPU.
    launch.refuse_native_backend_conflicts(None, uses_cpu = True)


@pytest.mark.parametrize("flag", ["use_cpu", "no_cuda"])
def test_cpu_trainer_on_xpu_install_takes_native_back(launch, monkeypatch, flag):
    _build(monkeypatch, xpu = "20250101")
    monkeypatch.setenv("FSDP_STATE_DICT_TYPE", "SHARDED_STATE_DICT")
    assert launch.apply_unsloth_fsdp2_launch_env() is True
    assert launch.reclaim_native_fsdp2_for_cpu_trainer(types.SimpleNamespace()) is False
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"
    assert launch.reclaim_native_fsdp2_for_cpu_trainer(types.SimpleNamespace(**{flag: True}))
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION", "FSDP_AUTO_WRAP_POLICY"):
        assert key not in os.environ, key
    assert os.environ["FSDP_STATE_DICT_TYPE"] == "SHARDED_STATE_DICT"  # the user's value stays
    assert launch.unsloth_fsdp2_native_requested() is True
    assert launch.native_backend_conflicts(None, uses_cpu = True) == []


def test_cpu_trainer_reclaim_needs_the_non_cuda_route(launch, monkeypatch):
    _build(monkeypatch, cuda = "12.8")
    monkeypatch.setenv("UNSLOTH_FSDP2_BACKEND", "accelerate")
    assert launch.apply_unsloth_fsdp2_launch_env() is True
    assert launch.reclaim_native_fsdp2_for_cpu_trainer(types.SimpleNamespace(use_cpu = True)) is False
    assert os.environ["ACCELERATE_USE_FSDP"] == "true"


def test_zz_launch_env_does_not_leak_out_of_this_module():
    # Runs last in this file: every key the launch tests wrote is back to its value at import.
    for key in _CLEAN + ("UNSLOTH_FSDP2", "WORLD_SIZE", "RANK"):
        assert os.environ.get(key) == _ENV_AT_IMPORT.get(key), key
