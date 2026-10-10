# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""install.sh's Intel GPU route survives `unsloth studio update`: _ensure_xpu_torch acts on the
resolved xpu backend, or the xpu flavor the install recorded, and on nothing else unpinned."""

import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_xpu_auto", ROOT / "studio" / "install_python_stack.py"
)
stack = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stack
_SPEC.loader.exec_module(stack)


def _pci(monkeypatch, tmp_path, *devices):
    """A fake PCI tree of (vendor, device, class) functions."""
    root = tmp_path / "pci"
    root.mkdir(exist_ok = True)
    for old in root.iterdir():
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    for i, (vendor, device, cls) in enumerate(devices):
        d = root / f"0000_{i:02x}_00.0"  # ":" is not legal in Windows paths
        d.mkdir()
        (d / "vendor").write_text(vendor + "\n")
        (d / "device").write_text(device + "\n")
        (d / "class").write_text(cls + "\n")
    monkeypatch.setattr(stack, "_PCI_DEVICES_ROOT", str(root), raising = False)


@pytest.fixture(autouse = True)
def _linux(monkeypatch, tmp_path):
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    for var in (
        "ZE_AFFINITY_MASK",
        "ONEAPI_DEVICE_SELECTOR",
        "SYCL_DEVICE_FILTER",
        "UNSLOTH_DISABLE_XPU_AUTO",
        "UNSLOTH_ROCM_GFX_ARCH",
    ):
        monkeypatch.delenv(var, raising = False)
    _pci(monkeypatch, tmp_path, ("0x8086", "0x56a0", "0x030000"))
    stack._invalidate_torch_runtime_probe()
    yield
    stack._invalidate_torch_runtime_probe()


def _run(
    backend,
    recorded,
    version = "2.11.0+cpu",
    probe_only = False,
):
    with (
        patch.object(stack, "_TORCH_BACKEND", backend),
        patch.object(stack, "_RECORDED_TORCH_TAG", recorded),
        patch.object(stack, "_probe_torch_runtime", return_value = (True, True, version, None, None)),
        patch.object(stack, "pip_install") as pip,
    ):
        pip.result = (
            stack._ensure_xpu_torch(probe_only = probe_only)
            if probe_only
            else stack._ensure_xpu_torch()
        )
    return pip


@pytest.mark.parametrize(
    ("backend", "recorded"),
    [("xpu", None), ("", "xpu")],
    ids = ["install.sh route", "standalone update, recorded xpu"],
)
def test_the_auto_route_repairs_a_cpu_wheel_from_the_xpu_index(backend, recorded):
    pip = _run(backend, recorded)
    assert pip.called
    assert pip.call_args.args[-2:] == ("--index-url", "https://download.pytorch.org/whl/xpu")


@pytest.mark.parametrize(
    ("backend", "recorded"),
    [("", None), ("", "cpu"), ("cpu", "xpu"), ("cuda", None), ("rocm", "xpu")],
)
def test_no_xpu_signal_or_another_stated_backend_leaves_torch_alone(backend, recorded):
    assert not _run(backend, recorded).called


def test_another_family_pin_wins_over_a_recorded_xpu(monkeypatch):
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cpu")
    assert not _run("", "xpu").called


def test_a_supported_xpu_wheel_is_left_alone():
    assert not _run("xpu", None, version = "2.9.1+xpu").called


def test_install_sh_exports_the_xpu_backend():
    source = (ROOT / "install.sh").read_text(encoding = "utf-8")
    assert re.search(r'^\s*xpu\)\s*export UNSLOTH_TORCH_BACKEND="xpu"', source, re.M)


@pytest.mark.parametrize("vendor", ["_has_usable_nvidia_gpu", "_has_rocm_gpu"])
def test_a_recorded_xpu_flavor_yields_to_a_gpu_added_since(monkeypatch, vendor):
    monkeypatch.setattr(stack, vendor, lambda: True)
    assert not _run("", "xpu").called


def test_install_sh_route_still_acts_beside_another_vendor_probe(monkeypatch):
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    assert _run("xpu", None).called


@pytest.mark.parametrize(("backend", "recorded"), [("xpu", None), ("", "xpu")])
def test_an_unexpressible_repair_is_reported(monkeypatch, backend, recorded):
    monkeypatch.setattr(stack, "_pytorch_whl_leaf_url", lambda leaf: None)
    pip = _run(backend, recorded)
    assert pip.result is False and not pip.called
    assert _run(backend, recorded, version = "2.9.1+xpu").result is None


@pytest.mark.parametrize(
    ("backend", "recorded", "version", "needs"),
    [
        ("", "xpu", "2.11.0+cpu", True),
        ("", "xpu", "2.9.1+xpu", False),
        ("", "cpu", "2.11.0+cpu", False),
        ("", None, "2.11.0+cpu", False),
    ],
)
def test_the_fast_path_probe_matches_the_repair(backend, recorded, version, needs):
    pip = _run(backend, recorded, version = version, probe_only = True)
    assert bool(pip.result) is needs and not pip.called


def test_setup_sh_forces_the_pass_on_a_stale_xpu_wheel():
    setup = (ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    stack_src = (ROOT / "studio" / "install_python_stack.py").read_text(encoding = "utf-8")
    assert setup.count("--xpu-torch-needs-dependency-pass") == 2
    assert 'sys.argv[1:] == ["--xpu-torch-needs-dependency-pass"]' in stack_src


@pytest.mark.parametrize("backend, recorded", [("xpu", None), ("", "xpu")])
@pytest.mark.parametrize(
    "case",
    [
        "intel gone",
        "non-Arc iGPU only",
        "Cedar Trail 0x0be0",
        "Cedar Trail 0x0be5",
        "AMD beside Arc",
        "mask set",
        "emptied mask",
        "oneAPI selector",
        "SYCL filter",
        "opt-out",
    ],
)
def test_the_unpinned_route_is_revalidated(monkeypatch, tmp_path, backend, recorded, case):
    if case == "intel gone":
        _pci(monkeypatch, tmp_path)
    elif case == "non-Arc iGPU only":
        _pci(monkeypatch, tmp_path, ("0x8086", "0x46a6", "0x030000"))
    elif case.startswith("Cedar Trail"):
        _pci(monkeypatch, tmp_path, ("0x8086", case.split()[-1], "0x030000"))
    elif case == "oneAPI selector":
        monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    elif case == "SYCL filter":
        monkeypatch.setenv("SYCL_DEVICE_FILTER", "level_zero:gpu:0")
    elif case == "AMD beside Arc":
        _pci(
            monkeypatch,
            tmp_path,
            ("0x8086", "0x56a0", "0x030000"),
            ("0x1002", "0x744c", "0x030000"),
        )
    elif case == "mask set":
        monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    elif case == "emptied mask":
        monkeypatch.setenv("ZE_AFFINITY_MASK", "")
    else:
        monkeypatch.setenv("UNSLOTH_DISABLE_XPU_AUTO", "1")
    assert not _run(backend, recorded).called


def test_an_explicit_pin_stays_authoritative(monkeypatch, tmp_path):
    _pci(monkeypatch, tmp_path)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "xpu")
    assert _run("", None).called


def test_the_allowlist_matches_hardware_py():
    import ast

    def tables(path):
        out = {}
        for node in ast.parse(path.read_text(encoding = "utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("_INTEL_XPU_PCI_ID_RANGES", "_INTEL_XPU_PCI_IDS"):
                    value = node.value.args[0] if isinstance(node.value, ast.Call) else node.value
                    out[name] = set(ast.literal_eval(value))
        return out

    hardware = tables(ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py")
    assert tables(ROOT / "studio" / "install_python_stack.py") == hardware
    assert len(hardware) == 2


@pytest.mark.parametrize("device", ["0x0bd0", "0x0bd5", "0x0b69", "0x0b6e"])
def test_pvc_ids_still_route(monkeypatch, tmp_path, device):
    _pci(monkeypatch, tmp_path, ("0x8086", device, "0x030000"))
    assert stack._intel_xpu_auto_route_holds()


def test_an_explicit_pin_wins_over_a_sycl_selector(monkeypatch, tmp_path):
    monkeypatch.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "xpu")
    assert _run("", None).called
