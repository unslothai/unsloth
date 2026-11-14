# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mixed NVIDIA cards with CUDA_DEVICE_ORDER unset: warn once at import, change nothing else."""

import functools
import inspect
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

import unsloth.device_type as dt

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"

if not hasattr(dt, "torch"):
    pytest.skip(
        "device_type stub or MLX host; the real probes are not loaded", allow_module_level = True
    )

A100, L4, T4 = "NVIDIA A100-SXM4-80GB", "NVIDIA L4", "Tesla T4"
CAPS = {A100: (8, 0), L4: (8, 9), T4: (7, 5)}
MARK = "Unsloth: visible GPUs differ"


@pytest.fixture(autouse = True)
def _nvidia_host(monkeypatch):
    # is_hip() is cached from the real host; these cases simulate NVIDIA.
    monkeypatch.setattr(dt, "is_hip", lambda: False)


CASES = [
    pytest.param([A100], None, False, id = "single"),
    pytest.param([A100, A100], None, False, id = "homogeneous"),
    pytest.param([A100, L4], None, True, id = "mixed-unset"),
    pytest.param([L4, A100, A100], None, True, id = "mixed3-unset"),
    pytest.param([A100, L4], "FASTEST_FIRST", False, id = "mixed-fastest-first-explicit"),
    pytest.param([A100, L4], " ", True, id = "mixed-blank"),
    pytest.param([A100, L4], "PCI_BUS_ID", False, id = "mixed-pci"),
    pytest.param([A100, L4], " pci_bus_id ", False, id = "mixed-pci-lower"),
    pytest.param([A100, T4], None, True, id = "mixed-with-pre-ampere"),
    pytest.param([A100, None], None, False, id = "mixed-one-unreadable"),
    pytest.param([], None, False, id = "no-devices"),
]


def _fake_torch(names):
    def get_device_name(device = None):
        if names[device or 0] is None:
            raise RuntimeError("unreadable")
        return names[device or 0]

    def get_device_capability(device = None):
        return CAPS.get(names[device or 0], (8, 0))

    return types.SimpleNamespace(
        device = __import__("torch").device,
        version = types.SimpleNamespace(hip = None, cuda = "12.8"),
        cuda = types.SimpleNamespace(
            device_count = lambda: len(names),
            current_device = lambda: 0,
            is_available = lambda: True,
            get_device_name = get_device_name,
            get_device_capability = get_device_capability,
            get_device_properties = lambda device = None: types.SimpleNamespace(
                name = get_device_name(device)
            ),
            is_bf16_supported = lambda including_emulation = True: True,
        ),
    )


def _run_cuda_arm(fake, monkeypatch):
    import unsloth._gpu_init as gpu_init

    source = GPU_INIT.read_text(encoding = "utf-8")
    start = 'if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():'
    body = start + source.split(start, 1)[1].split('\nelif DEVICE_TYPE == "hip":', 1)[0]
    monkeypatch.setattr(dt, "torch", fake)
    namespace = dict(vars(gpu_init))
    namespace.update(torch = fake, DEVICE_TYPE = "cuda", inspect = inspect, functools = functools)
    exec(compile(body, str(GPU_INIT), "exec"), namespace)


def _set_order(monkeypatch, order):
    if order is None:
        monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    else:
        monkeypatch.setenv("CUDA_DEVICE_ORDER", order)


@pytest.mark.parametrize("names,order,warn", CASES)
def test_import_warns_once_on_mixed_nvidia_without_pci_order(
    monkeypatch, capsys, names, order, warn
):
    _set_order(monkeypatch, order)
    if names:
        _run_cuda_arm(_fake_torch(names), monkeypatch)
    else:
        monkeypatch.setattr(dt, "torch", _fake_torch(names))
        assert dt.cuda_order_mismatch_names() == []
        return
    out = capsys.readouterr().out
    assert out.count(MARK) == (1 if warn else 0)
    if warn:
        for name in set(names):
            assert name in out


@pytest.mark.parametrize("names,order,warn", CASES)
def test_helper_matches_the_warning(monkeypatch, names, order, warn):
    _set_order(monkeypatch, order)
    monkeypatch.setattr(dt, "torch", _fake_torch(names))
    assert bool(dt.cuda_order_mismatch_names()) is warn


def test_hip_is_silent(monkeypatch):
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    monkeypatch.setattr(dt, "torch", _fake_torch([A100, L4]))
    monkeypatch.setattr(dt, "is_hip", lambda: True)
    assert dt.cuda_order_mismatch_names() == []


def test_unreadable_count_is_silent(monkeypatch):
    def boom():
        raise RuntimeError("no driver")

    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    monkeypatch.setattr(
        dt, "torch", types.SimpleNamespace(cuda = types.SimpleNamespace(device_count = boom))
    )
    assert dt.cuda_order_mismatch_names() == []


def test_warning_changes_nothing_else(monkeypatch):
    import unsloth._gpu_init as gpu_init

    results = []
    for order in (None, "PCI_BUS_ID"):
        _set_order(monkeypatch, order)
        fake = _fake_torch([A100, L4])
        source = GPU_INIT.read_text(encoding = "utf-8")
        start = 'if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():'
        body = start + source.split(start, 1)[1].split('\nelif DEVICE_TYPE == "hip":', 1)[0]
        monkeypatch.setattr(dt, "torch", fake)
        namespace = dict(vars(gpu_init))
        namespace.update(torch = fake, DEVICE_TYPE = "cuda", inspect = inspect, functools = functools)
        exec(compile(body, str(GPU_INIT), "exec"), namespace)
        results.append((namespace["SUPPORTS_BFLOAT16"], fake.cuda.is_bf16_supported()))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "os_name,command",
    [
        ("posix", "export CUDA_DEVICE_ORDER=PCI_BUS_ID"),
        ("nt", "$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'"),
    ],
)
def test_hint_command_fits_the_shell(monkeypatch, capsys, os_name, command):
    _set_order(monkeypatch, None)
    import unsloth._gpu_init as gpu_init

    # Restore os.name before pytest reports the call: Python 3.13 pathlib reads it there.
    with monkeypatch.context() as patch:
        patch.setattr(gpu_init.os, "name", os_name)
        _run_cuda_arm(_fake_torch([A100, L4]), monkeypatch)
    assert f"`{command}`" in capsys.readouterr().out


def test_repeated_names_listed_once(monkeypatch):
    _set_order(monkeypatch, None)
    monkeypatch.setattr(dt, "torch", _fake_torch([A100, A100, L4, L4]))
    assert dt.cuda_order_mismatch_names() == [A100, L4]
