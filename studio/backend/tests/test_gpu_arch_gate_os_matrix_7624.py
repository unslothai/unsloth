# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OS x GPU-vendor matrix for the #7624 ROCm arch gate in ``_get_gpu_memory``.

The gate drops a device whose gfx arch is missing from the prebuilt's
``mapped_targets``, and runs on exactly one shape of host (ROCm torch build,
``for_llama_server = True``). Every other cell asserts it is *inert*, which
matching output alone cannot show (a gate that ran and kept everything looks
identical), so the marker reader is spied and asserted un-called.

Matrix: [Windows, Linux, WSL, macOS] x [NVIDIA, AMD, CPU-only]. macOS has no ROCm,
so its "AMD" cell is the two real Apple shapes (Apple Silicon MPS, Intel Mac
Radeon), neither enumerating a ``torch.cuda`` device. WSL is its own cell: ROCm
reports through the same torch path yet ``sys.platform`` is "linux", so the
Windows-only free-VRAM cap (#8403) must NOT engage.

No AMD GPU or ROCm runtime exists here, so every AMD result is mock-based: torch
and its arch attributes, ``sys.platform`` / ``platform.system``,
``utils.hardware.IS_ROCM``, the marker and the HIP/ROCR/CUDA masks are faked in the
shapes #7072 / #7624 / #8403 document (AMD SDK wheels leaving ``torch.version.hip``
unset; the ``gcn_arch_name`` / ``arch_name`` / ``gfx_arch_name`` spellings;
``gfx103X`` omitting the gfx1033/1035/1036 iGPUs; Windows over-reporting free VRAM).
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

# Imported eagerly so a fake torch can never be in place when the real module
# graph is first loaded.
from core.inference import llama_cpp
from core.inference.llama_cpp import (
    _IGPU_HOST_RESERVE_MIB,
    GgufLoadIntent,
    LlamaCppBackend,
)
from utils.hardware import hardware as hw

MiB = 1024 * 1024
_TOTAL_BYTES = 32 * 1024**3

# Concrete tokens the published ROCm manifest records for these bundles. gfx103X
# omits the gfx1033/1035/1036 iGPUs, which is the whole of #7624: they enumerate,
# outrank the dGPU on "free memory", and have no kernels in the binary.
GFX103X = ["gfx1030", "gfx1031", "gfx1032", "gfx1034"]
GFX110X = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]
GFX120X = ["gfx1200", "gfx1201"]

# platform.system() / sys.platform pair per simulated host.
_OS_CELLS = {
    "windows": ("win32", "Windows"),
    "linux": ("linux", "Linux"),
    "wsl": ("linux", "Linux"),
    "macos": ("darwin", "Darwin"),
}
OS_KEYS = list(_OS_CELLS)


def _device(
    arch = "",
    *,
    free_mib = 12000,
    total_bytes = _TOTAL_BYTES,
    name = "AMD Radeon RX 6800",
    is_integrated = 0,
    arch_attr = "gcnArchName",
    describe_error = None,
):
    """One fake enumerated device. ``describe_error`` makes
    ``get_device_properties`` raise for it (a card torch cannot describe)."""
    return {
        "arch": arch,
        "free_mib": free_mib,
        "total_bytes": total_bytes,
        "name": name,
        "is_integrated": is_integrated,
        "arch_attr": arch_attr,
        "describe_error": describe_error,
    }


class _Props:
    """hipDeviceProp_t / cudaDeviceProp stand-in. Only the arch attribute the
    device asks for is set, so the AMD SDK wheels that populate none of the
    canonical spellings can be reproduced."""

    def __init__(self, spec):
        self.name = spec["name"]
        self.total_memory = spec["total_bytes"]
        self.is_integrated = spec["is_integrated"]
        if spec["arch"]:
            setattr(self, spec["arch_attr"], spec["arch"])


def _fake_torch(
    devices,
    *,
    vendor = "amd",
    cuda_available = None,
    reserved_bytes = 0,
):
    """A fake ``torch``.

    vendor:
      "amd"      -- ROCm wheel (``version.hip`` set).
      "amd_sdk"  -- AMD SDK / Radeon wheel: ``version.hip`` unset, "rocm" only
                    in ``__version__`` (the shape ``_torch_is_rocm`` exists for).
      "nvidia"   -- CUDA wheel.
      "cpu"      -- CPU-only wheel.
      "mps"      -- Apple Silicon: a Metal backend, no ``torch.cuda`` devices.
    """
    devices = list(devices)
    torch = types.ModuleType("torch")
    if vendor == "amd":
        torch.version = types.SimpleNamespace(hip = "7.1.0", cuda = None)
        torch.__version__ = "2.9.0+rocm7.1"
    elif vendor == "amd_sdk":
        torch.version = types.SimpleNamespace()
        torch.__version__ = "2.6.0+rocm6.4"
    elif vendor == "nvidia":
        torch.version = types.SimpleNamespace(hip = None, cuda = "12.4")
        torch.__version__ = "2.6.0+cu124"
    else:
        torch.version = types.SimpleNamespace(hip = None, cuda = None)
        torch.__version__ = "2.6.0+cpu"
    if vendor == "mps":
        torch.backends = types.SimpleNamespace(mps = types.SimpleNamespace(is_available = lambda: True))

    available = bool(devices) if cuda_available is None else cuda_available

    def _get_device_properties(ordinal):
        spec = devices[ordinal]
        if spec["describe_error"] is not None:
            raise spec["describe_error"]
        return _Props(spec)

    torch.cuda = types.SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: len(devices),
        mem_get_info = lambda o: (devices[o]["free_mib"] * MiB, devices[o]["total_bytes"]),
        memory_reserved = lambda o = None: reserved_bytes,
        get_device_properties = _get_device_properties,
    )
    return torch


def _binary_with_marker(tmp_path, payload):
    """Lay out ``<root>/UNSLOTH_PREBUILT_INFO.json`` with the binary path below
    it, matching the managed install layout the marker walk-up covers."""
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload), encoding = "utf-8")
    return str(tmp_path / "build" / "bin" / "llama-server")


def _apply_os(
    monkeypatch,
    os_key,
    *,
    is_rocm = False,
):
    """Pin the simulated host OS. ``IS_ROCM`` is the backend's own detection
    flag and gates the Windows free-VRAM cap, so it travels with the vendor."""
    platform_name, system_name = _OS_CELLS[os_key]
    monkeypatch.setattr(llama_cpp.sys, "platform", platform_name)
    monkeypatch.setattr(hw.sys, "platform", platform_name)
    monkeypatch.setattr(hw.platform, "system", lambda: system_name)
    monkeypatch.setattr(hw, "IS_ROCM", is_rocm)
    if os_key == "wsl":
        # The WSL prebuilts load the system ROCm libs before the bundled HIP.
        # Nothing in the probe branches on it; pinned so the cell is a real WSL
        # host rather than a relabelled Linux one.
        monkeypatch.setattr(llama_cpp, "_wsl_system_rocm_lib_dirs", lambda: ["/opt/rocm/lib"])


@pytest.fixture(autouse = True)
def _clear_marker_cache():
    """read_install_marker memoizes per binary path with no invalidation, so a
    stale entry from another test would answer for this one."""
    import utils.llama_cpp_freshness as freshness

    freshness._marker_cache.clear()
    yield
    freshness._marker_cache.clear()


@pytest.fixture
def marker_spy(monkeypatch):
    """Records every install-marker read. The inertness claim for non-ROCm
    hosts is "the gate never ran", which only this can show."""
    import utils.llama_cpp_freshness as freshness

    real = freshness.read_install_marker
    calls: list = []

    def _spy(binary_path):
        calls.append(binary_path)
        return real(binary_path)

    monkeypatch.setattr(freshness, "read_install_marker", _spy)
    return calls


@pytest.fixture
def arch_map_spy(monkeypatch):
    """Records every per-device arch enumeration (the gate's second input)."""
    real = LlamaCppBackend._rocm_arch_by_physical_id
    calls: list = []

    def _spy():
        calls.append(True)
        return real()

    monkeypatch.setattr(LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(_spy))
    return calls


@pytest.fixture
def probe_env(tmp_path, monkeypatch):
    """Force ``_get_gpu_memory`` down the torch fallback, hermetically: no
    nvidia-smi, a binary under tmp_path so a marker can be planted beside it,
    and no inherited visibility mask."""

    def _no_nvidia_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    fake_binary = str(tmp_path / "build" / "bin" / "llama-server")
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: fake_binary)
    )
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    return fake_binary


def _install_cell(monkeypatch, tmp_path, os_key, vendor):
    """Set up one matrix cell and return the expected ``_get_gpu_free_memory``
    result. Every cell plants a marker covering NOTHING its devices report, so a
    gate that runs where it should not shows up as a dropped device."""
    if vendor == "nvidia":
        _apply_os(monkeypatch, os_key, is_rocm = False)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX110X})
        if os_key == "macos":
            # No CUDA on any Mac torch build: the probe enumerates nothing.
            torch = _fake_torch([], vendor = "nvidia", cuda_available = False)
            expected = []
        else:
            torch = _fake_torch(
                [
                    _device(name = "NVIDIA GeForce RTX 4090", free_mib = 22000),
                    _device(name = "NVIDIA GeForce RTX 3090", free_mib = 18000),
                ],
                vendor = "nvidia",
            )
            expected = [(0, 22000), (1, 18000)]
    elif vendor == "amd":
        if os_key == "macos":
            # Apple ships no ROCm. The Apple Silicon shape is Metal-only; the
            # Intel shape is a Radeon behind a CPU/MPS torch build. Neither
            # enumerates a torch.cuda device, so neither can reach the gate.
            _apply_os(monkeypatch, os_key, is_rocm = False)
            _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
            torch = _fake_torch([], vendor = "mps", cuda_available = False)
            expected = []
        else:
            # The #7624 shape: a covered dGPU plus an iGPU the gfx103X bundle
            # has no kernels for, whose shared-RAM "free memory" outranks it.
            _apply_os(monkeypatch, os_key, is_rocm = True)
            _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
            torch = _fake_torch(
                [
                    _device("gfx1030", free_mib = 12049, name = "AMD Radeon RX 6800"),
                    _device(
                        "gfx1036",
                        free_mib = 12176,
                        name = "AMD Radeon Graphics",
                        is_integrated = 1,
                    ),
                ],
                vendor = "amd",
            )
            # Plenty of system RAM, so the iGPU's shared-pool cap never binds
            # and the only thing that can change its figure is the host reserve.
            monkeypatch.setattr(
                LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
            )
            expected = [(0, 12049)]
    else:
        _apply_os(monkeypatch, os_key, is_rocm = False)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        torch = _fake_torch([], vendor = "cpu", cuda_available = False)
        expected = []
    monkeypatch.setitem(sys.modules, "torch", torch)
    return expected


CELLS = [(os_key, vendor) for os_key in OS_KEYS for vendor in ("nvidia", "amd", "cpu")]


class TestOsVendorMatrix:
    """The 12 cells, against the llama-server (gated) probe."""

    @pytest.mark.parametrize("os_key,vendor", CELLS)
    def test_cell_outcome(
        self, os_key, vendor, tmp_path, monkeypatch, probe_env, marker_spy, arch_map_spy
    ):
        expected = _install_cell(monkeypatch, tmp_path, os_key, vendor)
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == expected
        if vendor == "amd" and os_key != "macos":
            # The only cells where the gate is allowed to do anything.
            assert marker_spy, "the ROCm cell must consult the install marker"
            assert arch_map_spy, "the ROCm cell must enumerate device archs"
        else:
            assert marker_spy == [], f"{os_key}/{vendor} read the install marker"
            assert arch_map_spy == [], f"{os_key}/{vendor} enumerated device archs"

    @pytest.mark.parametrize("os_key,vendor", CELLS)
    def test_cell_is_unfiltered_by_default(
        self, os_key, vendor, tmp_path, monkeypatch, probe_env, marker_spy, arch_map_spy
    ):
        """The backwards-compatibility guarantee: every pre-existing caller
        (``for_llama_server`` off, the PyTorch RAG embedding picker included) keeps
        every device on every cell, the AMD one whose iGPU the gate drops too."""
        expected = _install_cell(monkeypatch, tmp_path, os_key, vendor)
        if vendor == "amd" and os_key != "macos":
            # The iGPU is kept, still with its unified-memory host reserve: the
            # gate is what must not apply here, not the APU accounting.
            expected = [(0, 12049), (1, 12176 - _IGPU_HOST_RESERVE_MIB)]
        assert LlamaCppBackend._get_gpu_free_memory() == expected
        assert marker_spy == [], f"{os_key}/{vendor} read the install marker unasked"
        assert arch_map_spy == [], f"{os_key}/{vendor} enumerated device archs unasked"

    @pytest.mark.parametrize("os_key", OS_KEYS)
    def test_nvidia_smi_path_never_reaches_the_gate(
        self, os_key, tmp_path, monkeypatch, marker_spy, arch_map_spy
    ):
        """The common NVIDIA host answers from nvidia-smi and returns before the
        torch fallback the gate lives in. Separate from the cell above, which
        forces the fallback so "inert" is proven on both NVIDIA routes."""
        _apply_os(monkeypatch, os_key, is_rocm = False)
        binary = _binary_with_marker(tmp_path, {"mapped_targets": GFX110X})
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: binary)
        )
        for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(var, raising = False)

        class _Result:
            returncode = 0
            stdout = "0, 22000, 24564\n1, 18000, 24564\n"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())
        # No torch at all: a bare nvidia-smi host must not need one.
        monkeypatch.setitem(sys.modules, "torch", None)
        assert LlamaCppBackend._get_gpu_memory(for_llama_server = True) == [
            (0, 22000, 24564),
            (1, 18000, 24564),
        ]
        assert marker_spy == []
        assert arch_map_spy == []

    @pytest.mark.parametrize("os_key", OS_KEYS)
    def test_cpu_only_host_without_torch_is_empty_and_quiet(
        self, os_key, tmp_path, monkeypatch, probe_env, marker_spy
    ):
        """CPU-only with no torch installed at all: no GPU, no traceback,
        empty list (the import failure is swallowed by the probe)."""
        _apply_os(monkeypatch, os_key, is_rocm = False)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` -> ImportError
        assert LlamaCppBackend._get_gpu_memory(for_llama_server = True) == []
        assert marker_spy == []

    def test_macos_intel_radeon_shape_is_inert(self, tmp_path, monkeypatch, probe_env, marker_spy):
        """The second Apple shape: an Intel Mac with a Radeon, i.e. a CPU torch
        build that still reports an AMD device name. No ROCm exists there, so
        ``_torch_is_rocm`` is false and the marker stays unread."""
        _apply_os(monkeypatch, "macos", is_rocm = False)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("", name = "AMD Radeon Pro 5500M", free_mib = 8000)],
                vendor = "cpu",
                cuda_available = False,
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == []
        assert marker_spy == []


class TestAmdCoverageCases:
    """The substantive ROCm cells. Mock-based: no AMD hardware here."""

    @pytest.mark.parametrize("os_key", ["windows", "linux", "wsl"])
    def test_every_device_covered_drops_nothing(self, os_key, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, os_key, is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1032", free_mib = 7000)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 7000),
        ]

    @pytest.mark.parametrize("os_key", ["windows", "linux", "wsl"])
    def test_uncovered_igpu_is_dropped_and_dgpu_kept(
        self, os_key, tmp_path, monkeypatch, probe_env
    ):
        # #7624 verbatim: gfx1030 dGPU + gfx1036 iGPU against the real gfx103X
        # bundle, which maps gfx1030/1031/1032/1034 only.
        _apply_os(monkeypatch, os_key, is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1030", free_mib = 12049),
                    _device("gfx1036", free_mib = 12176, is_integrated = 1),
                ],
                vendor = "amd",
            ),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    @pytest.mark.parametrize("os_key", ["windows", "linux", "wsl"])
    def test_missing_marker_fails_open(self, os_key, tmp_path, monkeypatch, probe_env):
        # Source build / custom link: coverage unknown, so keep every device.
        _apply_os(monkeypatch, os_key, is_rocm = True)
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    @pytest.mark.parametrize("os_key", ["windows", "linux", "wsl"])
    def test_empty_mapped_targets_fails_open(self, os_key, tmp_path, monkeypatch, probe_env):
        # Non-ROCm bundles record []: unknown coverage, not "covers nothing".
        _apply_os(monkeypatch, os_key, is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": []})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    @pytest.mark.parametrize("os_key", ["windows", "linux", "wsl"])
    def test_every_device_uncovered_empties_the_pool(
        self, os_key, tmp_path, monkeypatch, probe_env
    ):
        """A marker that covers none of the present cards leaves the llama-server
        probe with nothing. Pinned deliberately: this is the input to the
        downstream behaviour asserted in TestEveryDeviceUncoveredDownstream."""
        _apply_os(monkeypatch, os_key, is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX120X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == []
        # ... while the unfiltered probe, and therefore every torch caller,
        # still sees both cards.
        assert LlamaCppBackend._get_gpu_free_memory() == [(0, 12049), (1, 12176)]

    def test_amd_sdk_wheel_is_gated_on_every_os(self, tmp_path, monkeypatch, probe_env):
        """AMD SDK / Radeon wheels leave ``version.hip`` unset and only encode
        "rocm" in ``__version__``; a bare ``version.hip`` test would skip the
        gate on exactly the hosts #7624 was reported from."""
        _apply_os(monkeypatch, "windows", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd_sdk",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]


class TestWslIsNotWindows:
    """ROCm under WSL reports through the same torch path as native Linux, but
    ``sys.platform`` is "linux", so the Windows-only free-VRAM cap (#8403) must not
    engage: same arch gate, different memory accounting."""

    def _probe(self, monkeypatch, tmp_path, os_key):
        _apply_os(monkeypatch, os_key, is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1030", free_mib = 20000),
                    _device("gfx1036", free_mib = 12176),
                ],
                vendor = "amd",
                # 24 GiB reserved by this process's own allocator, against a
                # 32 GiB card: on Windows the driver's "free" is an over-report
                # and gets capped to total - reserved = 8192 MiB.
                reserved_bytes = 24 * 1024**3,
            ),
        )
        return LlamaCppBackend._get_gpu_free_memory(for_llama_server = True)

    def test_windows_caps_free_against_the_allocator(self, tmp_path, monkeypatch, probe_env):
        assert self._probe(monkeypatch, tmp_path, "windows") == [(0, 8192)]

    def test_wsl_keeps_the_driver_reading(self, tmp_path, monkeypatch, probe_env):
        assert self._probe(monkeypatch, tmp_path, "wsl") == [(0, 20000)]

    def test_linux_keeps_the_driver_reading(self, tmp_path, monkeypatch, probe_env):
        assert self._probe(monkeypatch, tmp_path, "linux") == [(0, 20000)]


class TestVisibilityMaskMapping:
    """The gate keys on PHYSICAL ids while torch enumerates visible ordinals.
    Every case below is built so that confusing the two drops the wrong card,
    rather than merely relabelling the surviving one."""

    def test_hip_mask_reversing_the_order_drops_the_right_card(
        self, tmp_path, monkeypatch, probe_env
    ):
        # HIP_VISIBLE_DEVICES=1,0, so ordinal 0 IS physical 1. Unsupported gfx1036
        # is physical 1, supported gfx1030 physical 0, with different free VRAM, so
        # an ordinal/physical mix-up returns (1, 12176) instead of (0, 5000).
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1036", free_mib = 12176),  # ordinal 0 = physical 1
                    _device("gfx1030", free_mib = 5000),  # ordinal 1 = physical 0
                ],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._rocm_arch_by_physical_id() == {1: "gfx1036", 0: "gfx1030"}
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 5000)]

    def test_rocr_mask_with_a_gap_maps_by_physical_id(self, tmp_path, monkeypatch, probe_env):
        # ROCR_VISIBLE_DEVICES=2,3 on Linux: an off-by-one on either side would
        # report GPU 3 (the unsupported card) or drop GPU 2.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX110X})
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1100", free_mib = 9000), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(2, 9000)]

    def test_windows_ignores_a_stray_rocr_mask(self, tmp_path, monkeypatch, probe_env):
        # Windows HIP has no ROCr layer, so ROCR_VISIBLE_DEVICES masks nothing and
        # must not be read as the ordinal->physical map: doing so would label the
        # surviving card GPU 2 and pin a device that does not exist.
        _apply_os(monkeypatch, "windows", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX110X})
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1100", free_mib = 9000), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 9000)]

    def test_hip_mask_wins_over_cuda_mask(self, tmp_path, monkeypatch, probe_env):
        # Both masks set with different contents: HIP is the one the ROCm
        # runtime honors, so it decides the physical ids the gate drops by.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1036", free_mib = 12176), _device("gfx1030", free_mib = 5000)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(1, 5000)]

    def test_empty_mask_reports_no_gpu(self, tmp_path, monkeypatch, probe_env):
        # HIP_VISIBLE_DEVICES="" hides every agent, so the runtime enumerates
        # nothing. The gate still resolves its inputs first, and must return an
        # empty list rather than raise on the empty ordinal->physical map.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch([], vendor = "amd"))
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == []


class TestArchStringRobustness:
    """Both sides of the comparison are normalised: the manifest token and the
    driver's arch string. A device must never be dropped over spelling."""

    @pytest.mark.parametrize(
        "reported",
        [
            "gfx1030",
            "GFX1030",
            "  gfx1030  ",
            "gfx1030:xnack-",
            "gfx1030:sramecc+:xnack-",
            "GFX1030:SRAMECC+:XNACK-",
        ],
        ids = [
            "plain",
            "uppercase",
            "whitespace",
            "xnack",
            "sramecc_xnack",
            "uppercase_suffixed",
        ],
    )
    def test_covered_device_survives_every_spelling(
        self, reported, tmp_path, monkeypatch, probe_env
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device(reported, free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    @pytest.mark.parametrize(
        "token", ["GFX1030", " gfx1030 ", "gfx1030:xnack-", "gfx1030:sramecc+:xnack-"]
    )
    def test_marker_token_spellings_are_normalised_too(
        self, token, tmp_path, monkeypatch, probe_env
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": [token]})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    @pytest.mark.parametrize("attr", ["gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"])
    def test_every_arch_attribute_spelling_is_read(self, attr, tmp_path, monkeypatch, probe_env):
        # AMD SDK / Radeon wheels populate only one of these. Reading a single
        # spelling would leave the map empty and fail the gate open -- the crash
        # this exists to prevent.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1030", free_mib = 12049, arch_attr = attr),
                    _device("gfx1036", free_mib = 12176, arch_attr = attr),
                ],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    def test_blank_arch_fails_open(self, tmp_path, monkeypatch, probe_env):
        # A device reporting no arch at all is unknown, not unsupported.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 12049), _device("", free_mib = 12176)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    def test_undescribable_device_fails_open(self, tmp_path, monkeypatch, probe_env):
        # get_device_properties raising (a card the runtime cannot query) must
        # neither drop the device nor abort the probe for the other cards.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1030", free_mib = 12049),
                    _device(
                        "gfx1036",
                        free_mib = 12176,
                        describe_error = RuntimeError("hipGetDeviceProperties failed"),
                    ),
                    _device("gfx1036", free_mib = 3000),
                ],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    def test_marker_junk_tokens_are_ignored_not_matched(self, tmp_path, monkeypatch, probe_env):
        # Blank / whitespace-only entries must not become a "" arch that some
        # device's blank arch could match against.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        _binary_with_marker(tmp_path, {"mapped_targets": ["", "   ", "gfx1030"]})
        assert LlamaCppBackend._installed_llama_gfx_archs(
            str(tmp_path / "build" / "bin" / "llama-server")
        ) == frozenset({"gfx1030"})


def _run_auto_load(
    monkeypatch,
    tmp_path,
    torch,
    marker_targets,
    *,
    returncode = 1,
    output = "",
    env_extra = None,
    model_bytes = 1024,
    capture = None,
    intent_kwargs = None,
    apu_ram_stub = None,
    host_offload_stub = None,
    backend = None,
):
    """Drive a real automatic (no explicit GPU pick) llama-server load with the real
    ``_get_gpu_memory`` behind it, and return the spawned (cmd, env) list.

    Everything below the placement decision is faked: header-only GGUF, Popen never
    runs, health wait answers from ``returncode``. The point is what placement handed
    the child, not that llama-server works."""
    if marker_targets is not None:
        _binary_with_marker(tmp_path, {"mapped_targets": marker_targets})
    binary = str(tmp_path / "build" / "bin" / "llama-server")
    monkeypatch.setitem(sys.modules, "torch", torch)

    def _gguf_string(value):
        encoded = value.encode()
        return struct.pack("<Q", len(encoded)) + encoded

    metadata = _gguf_string("general.architecture") + struct.pack("<I", 8) + _gguf_string("llama")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)

    # ``backend`` replays a SECOND load onto the state the first one left, which is
    # the only way to see per-load state that must not be inherited.
    backend = backend if backend is not None else LlamaCppBackend()
    # ``capture`` hands the post-launch backend back for state the argv and env
    # cannot show (e.g. _gpu_offload_active).
    if capture is not None:
        capture["backend"] = backend
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    # model_bytes drives the placement decision: a model no GPU can hold makes
    # _select_gpus return (None, True), so `--fit on` owns placement.
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **_kwargs: None
    # Off by default: the APU RAM preflight is not what most of these cells are
    # about. A test that IS about it passes its own recording stub.
    backend._apu_ram_shortfall_message = apu_ram_stub or (lambda *_args, **_kwargs: None)
    # same, off: model_bytes here is sized to force --fit on, not to describe a host
    backend._host_offload_shortfall_message = host_offload_stub or (lambda *_args, **_kwargs: None)
    backend._find_llama_server_binary = lambda include_denied = False: binary
    backend._fit_off_retry_eligible = lambda *_args, **_kwargs: False
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    # env_extra seeds the child env the way an inherited / user-set variable
    # would, so "did THIS launch set it" can be told from "it was already there".
    _base_env = {"PATH": os.environ.get("PATH", ""), **(env_extra or {})}
    backend._llama_server_env_for_binary = lambda _binary: dict(_base_env)
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: False)
    )

    launches = []

    class _Process:
        pid = 123
        stdout = ()

        def __init__(self, code):
            self.returncode = code

        def poll(self):
            return returncode

        def terminate(self):
            return None

        def wait(self, timeout = None):
            return returncode

        def kill(self):
            return None

    def _popen(cmd, **kwargs):
        launches.append((list(cmd), dict(kwargs["env"])))
        return _Process(returncode)

    def _wait_for_health(timeout):
        backend._stdout_lines = [output]
        return returncode is None

    backend._wait_for_health = _wait_for_health
    backend._prepare_cpu_fallback_launch = lambda *_a, **_kw: None
    monkeypatch.setattr(subprocess, "Popen", _popen)
    try:
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "owner/model",
                **(intent_kwargs or {}),
            )
        )
    except Exception as exc:
        # A crashing child is one of the cases under test; the launches are the
        # evidence either way. A refusal that PREVENTS a spawn has no launch to
        # point at, so hand it back through ``capture`` for those tests.
        if capture is not None:
            capture["error"] = exc
    return launches


def _visibility(env):
    return {
        name: env.get(name)
        for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        if env.get(name) is not None
    }


class TestEveryDeviceUncoveredDownstream:
    """What the gate emptying the pool does to a load. Mock-based (no ROCm here).
    The comparison case is the #7624 shape, one covered card surviving; the case
    under test is the same host with a marker covering neither."""

    def test_one_covered_device_is_pinned(self, tmp_path, monkeypatch, probe_env):
        """The end-to-end #7624 fix. The iGPU's shared-RAM "free memory" outranks
        the dGPU's VRAM even after the host reserve, so before the gate automatic
        placement pinned the iGPU and llama-server died with "device kernel image is
        invalid". Measured on origin/main with these inputs: ROCR_VISIBLE_DEVICES=1
        (the gfx1036 iGPU); with the gate, 0."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 30000, is_integrated = 1),
            ],
            vendor = "amd",
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        launches = _run_auto_load(monkeypatch, tmp_path, torch, GFX103X, returncode = None)
        assert len(launches) == 1
        _cmd, env = launches[0]
        # The covered dGPU, and only it, is exposed to the child. Masked at the
        # ROCr layer, because a HIP-only mask still enumerates every agent first
        # and that enumeration can segfault on the unsupported card.
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}

    def test_all_uncovered_degrades_to_cpu(self, tmp_path, monkeypatch, probe_env):
        """Every device gated out must mask the child onto the CPU (#7624).

        Before this the empty pool left ``gpu_indices`` None, the launch took the
        ``--fit on`` arm, the pin block never ran and no mask was written, so the
        child enumerated both unsupported cards and died, with the reactive retry
        unable to help (its guard needs a truthy ``gpu_indices``)."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX120X,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
        )
        assert len(launches) == 1, "the arch-crash retry cannot fire without a pinned set"
        _cmd, env = launches[0]
        # "-1" keeps the HIP spelling: the sentinel has no portable ROCR form.
        assert _visibility(env) == {"HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1"}

    def test_all_uncovered_keeps_an_inherited_rocr_mask(self, tmp_path, monkeypatch, probe_env):
        """The forced-CPU mask must not widen what the child can see. ROCr filters
        at topology build, below HIP, so a parent that hid a segfaulting agent keeps
        hiding it: HIP "-1" already means zero devices, and clearing ROCR would hand
        the HSA enumeration the dropped agents. The embedding CPU launch states the
        rule; the chat one went through the default HIP arm, which clears ROCR."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX120X,  # covers neither card
            returncode = None,
            env_extra = {"ROCR_VISIBLE_DEVICES": "1"},
        )
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert _visibility(env) == {
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }

    def test_the_forced_cpu_server_reports_zero_vram(self, tmp_path, monkeypatch, probe_env):
        """A masked-off child holds no VRAM, so the flag training reads must be
        exactly False: routes/training_vram.py spares a server only on
        ``is not False``, so the counted classifier's None (the gated probe left the
        detected list empty) would unload one whose death frees nothing."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch, tmp_path, torch, GFX120X, returncode = None, capture = capture
        )
        _cmd, env = launches[0]
        assert _visibility(env) == {"HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1"}
        assert capture["backend"]._gpu_offload_active is False

    def test_a_covered_host_still_classifies_normally(self, tmp_path, monkeypatch, probe_env):
        """The zero-VRAM verdict is the gate's doing, not a blanket False: a host
        the build covers keeps the counted classifier's answer."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1031", free_mib = 12176)],
            vendor = "amd",
        )
        capture: dict = {}
        _run_auto_load(monkeypatch, tmp_path, torch, GFX103X, returncode = None, capture = capture)
        assert capture["backend"]._gpu_offload_active is not False

    def test_all_uncovered_names_the_devices_in_the_warning(self, tmp_path, monkeypatch, probe_env):
        """The CPU fallback is a large, silent-looking performance cliff, so the
        log has to say why: which devices are present, and that the installed
        build covers none of them (#7624)."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        # structlog, so the stdlib caplog fixture cannot see these records.
        warnings = []
        monkeypatch.setattr(
            llama_cpp.logger,
            "warning",
            lambda msg, *a, **kw: warnings.append(msg % a if a else msg),
        )
        _run_auto_load(monkeypatch, tmp_path, torch, GFX120X, returncode = 1)
        _hits = [w for w in warnings if "falls back to CPU" in w]
        assert len(_hits) == 1, warnings
        assert "0 (gfx1030)" in _hits[0] and "1 (gfx1036)" in _hits[0]

    def test_partial_coverage_never_reaches_the_cpu_fallback(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The normal #7624 host (one covered card) must be untouched by the CPU
        fallback: it still pins the survivor, and never pays for the second,
        ungated probe the fallback needs to tell "all gated out" from "no GPU"."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        _ungated = []
        _real = LlamaCppBackend._get_gpu_memory

        def _spy(binary = None, *, for_llama_server = False):
            if not for_llama_server:
                _ungated.append(binary)
            return _real(binary, for_llama_server = for_llama_server)

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_spy))
        launches = _run_auto_load(monkeypatch, tmp_path, torch, GFX103X, returncode = None)
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
        assert _ungated == [], "a covered device was found; the ungated re-probe is dead weight"

    def test_a_model_too_large_to_pin_still_masks_the_uncovered_card(
        self, tmp_path, monkeypatch, probe_env
    ):
        """One card short of the forced-CPU case: the gate drops the iGPU but keeps
        the dGPU, so the pool is not empty -- and a model too large for the planner
        makes `_select_gpus` answer (None, True) with `gpu_indices` still None.
        Nothing else writes a mask on that arm, so the child would enumerate the
        dropped card and die, the reactive retry needing `gpu_indices` to help."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1036", free_mib = 12176)],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,
            returncode = None,
            model_bytes = 400 * 1024**3,
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        # --fit on is the arm under test: placement was handed to llama.cpp.
        assert cmd[cmd.index("--fit") + 1] == "on"
        # gfx1036 is index 1 and has no kernels in a gfx103X build, so only 0
        # may reach the child. ROCr, not HIP: a HIP mask still enumerates first.
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}

    def test_full_coverage_leaves_a_fit_owned_launch_unmasked(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The same unpinned arm on a host the build fully covers writes no mask
        at all, so the pin is the gate's doing and not a blanket change to every
        `--fit on` launch."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1031", free_mib = 12176)],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,
            returncode = None,
            model_bytes = 400 * 1024**3,
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        assert cmd[cmd.index("--fit") + 1] == "on"
        assert _visibility(env) == {}

    def test_cuda_host_with_no_gpu_does_not_reprobe(self, tmp_path, monkeypatch, probe_env):
        """The GPU-less path is the common case, so the ROCm guard has to come
        BEFORE the ungated re-probe: on a CUDA or CPU-only host an empty probe
        just means "no GPU", and probing again would cost every such load."""
        _apply_os(monkeypatch, "linux", is_rocm = False)
        torch = _fake_torch([], vendor = "nvidia")
        _probes = []
        _real = LlamaCppBackend._get_gpu_memory

        def _spy(binary = None, *, for_llama_server = False):
            _probes.append(for_llama_server)
            return _real(binary, for_llama_server = for_llama_server)

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_spy))
        launches = _run_auto_load(monkeypatch, tmp_path, torch, GFX110X, returncode = None)
        assert len(launches) == 1
        _cmd, env = launches[0]
        # No mask at all: this is an ordinary CPU load, not the gated fallback.
        assert _visibility(env) == {}
        assert _probes == [True], "the ungated re-probe must not run off ROCm"


class TestArchCrashRetryEnv:
    """What the arch-crash respawn inherits from the crashed launch (#7624). The
    canonical shape: the shared-pool APU outranks the dGPU, is pinned, and
    llama-server dies with "device kernel image is invalid"; the retry moves to the
    discrete card. Mock-based, no ROCm here."""

    def _apu_then_dgpu(self, monkeypatch):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _fake_torch(
            [
                _device("gfx1151", free_mib = 40000, is_integrated = 1),
                _device("gfx1030", free_mib = 12049),
            ],
            vendor = "amd",
        )

    def test_retry_on_a_discrete_card_drops_unified_memory_env(
        self, tmp_path, monkeypatch, probe_env
    ):
        """GGML_CUDA_ENABLE_UNIFIED_MEMORY is decided for the CRASHED set. Left
        in place, the respawn on the dGPU keeps env that
        _amd_apu_wants_unified_memory's own docstring says hurts discrete GPUs."""
        torch = self._apu_then_dgpu(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
        )
        # The APU is pinned first, so the crashed spawn carries the env and the
        # respawns are masked onto the discrete card. The unrelated --fit off retry
        # spawns each launch twice, so select by the mask, not by position.
        assert launches[0][1].get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all("GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env for env in _retry)

    def test_the_no_binary_for_gpu_spelling_also_fires_the_retry(
        self, tmp_path, monkeypatch, probe_env
    ):
        """Same mismatch, HIP's other error code: hipErrorNoBinaryForGpu, documented
        as code compiled for a different arch. Neither field log showed it, so keying
        recovery on the InvalidImage wording alone leaves those builds on the
        misleading GGUF error."""
        torch = self._apu_then_dgpu(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: no kernel image is available for execution on the device",
        )
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the retry did not fire on the NoBinaryForGpu wording"

    def test_a_user_set_unified_memory_value_survives_the_retry(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The first spawn uses setdefault, so a value already in env is the
        user's. Popping unconditionally would clobber it; ownership is tracked
        instead, and only a value this launch set is withdrawn."""
        torch = self._apu_then_dgpu(monkeypatch)
        monkeypatch.setenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1")
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            env_extra = {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"},
        )
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all(env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1" for env in _retry)

    def _big_then_small_discrete(self, monkeypatch):
        """Both cards discrete, so neither pool reading is capped against system RAM and
        the survivor is genuinely too small to hold what the crashed card held."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(set))
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 20_000)
        )
        return _fake_torch(
            [_device("gfx1030", free_mib = 40_000), _device("gfx900", free_mib = 4_000)],
            vendor = "amd",
        )

    def test_the_retry_reprices_the_spill_against_the_narrowed_pool(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The host guard ran against the aggregate pool, and the retry masks the child onto
        the survivor. When the crashed card supplied most of that credit the narrowed launch
        spills far more into RAM than the preflight allowed, which is the OOM this guard
        exists to stop. A 30 GB model is held by the 40000 MiB card the launch pins; the
        4000 MiB survivor leaves about 26 GB for a host with 20 GB."""
        torch = self._big_then_small_discrete(monkeypatch)
        capture = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 30 * 1024**3,
            host_offload_stub = LlamaCppBackend._host_offload_shortfall_message,
            capture = capture,
        )

        assert launches, "the first launch never ran, so the retry is not what was tested"
        # The repricing is the point, not a block: the respawn goes ahead and carries
        # the advisory the narrowed pool produced.
        assert "does not fit in GPU memory" in (
            capture["backend"].last_load_warning or ""
        ), "the retry did not reprice the spill against the narrowed pool"

    def _arch_retry_launches(self, tmp_path, monkeypatch, capture, **kwargs):
        """The narrowed-pool retry above, plus whatever the caller asks the load for.

        Returns ``(first_argv, retry_argv)``: the crashed launch and the respawn, told
        apart by the mask, since the unrelated --fit off retry spawns each twice."""
        torch = self._big_then_small_discrete(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            host_offload_stub = LlamaCppBackend._host_offload_shortfall_message,
            capture = capture,
            **kwargs,
        )
        assert launches, "the first launch never ran, so the retry is not what was tested"
        _retry = [cmd for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        return launches[0][0], _retry[0]

    @pytest.mark.parametrize(
        "extra_args",
        [["--no-mmap"], ["--load-mode", "none"], ["--load-mode=none"], ["--no-direct-io"]],
        ids = ["no-mmap", "load-mode-none", "load-mode-none-equals", "no-direct-io"],
    )
    def test_the_narrowed_retry_pages_an_unmapped_respawn(
        self, tmp_path, monkeypatch, probe_env, extra_args
    ):
        """The shortfall the retry discovers is its FIRST one, so nothing upstream
        remapped the load: the original placement held the model in VRAM and the
        discrete guard abstained. "none" and "mlock" do not mmap, so respawning that
        argv unchanged allocates the whole 30 GB in a 20 GB host and is OOM-killed
        rather than paged. The override belongs to the condition, so it runs here too,
        before the respawn."""
        capture = {}
        first, retry = self._arch_retry_launches(
            tmp_path,
            monkeypatch,
            capture,
            model_bytes = 30 * 1024**3,
            intent_kwargs = {"extra_args": list(extra_args)},
        )

        # The first launch fit the card it pinned, so it is left exactly as asked --
        # which is what makes the retry the first place the shortfall can be found.
        assert _unmapped_tokens(first) == list(extra_args), f"the fitting launch changed: {first}"
        assert not _unmapped_tokens(retry), f"the respawn still loads unmapped: {retry}"
        assert "memory mapping instead" in (capture["backend"].last_load_warning or "")

    @pytest.mark.parametrize(
        "extra_args",
        [["--no-mmap"], ["--load-mode", "none"], ["--load-mode=none"], ["--no-direct-io"]],
        ids = ["no-mmap", "load-mode-none", "load-mode-none-equals", "no-direct-io"],
    )
    def test_a_narrowed_retry_that_fits_keeps_the_mode_it_was_given(
        self, tmp_path, monkeypatch, probe_env, extra_args
    ):
        """The control. Same crash and same narrowing, on a model the 4000 MiB survivor
        plus a 20 GB host holds comfortably: no shortfall, so both argvs keep the
        unmapped mode the user chose."""
        capture = {}
        first, retry = self._arch_retry_launches(
            tmp_path,
            monkeypatch,
            capture,
            model_bytes = 2 * 1024**3,
            intent_kwargs = {"extra_args": list(extra_args)},
        )

        assert _unmapped_tokens(first) == list(extra_args), f"the first launch changed: {first}"
        assert _unmapped_tokens(retry) == list(extra_args), f"the respawn changed: {retry}"
        assert capture["backend"].last_load_warning is None

    def test_the_retry_pages_an_unmapped_respawn_the_opt_out_silenced(
        self, tmp_path, monkeypatch, probe_env
    ):
        """UNSLOTH_ALLOW_HOST_OFFLOAD hides the retry's warning. It must not also hand
        the respawn back the load mode that cannot complete."""
        monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
        capture = {}
        _first, retry = self._arch_retry_launches(
            tmp_path,
            monkeypatch,
            capture,
            model_bytes = 30 * 1024**3,
            intent_kwargs = {"extra_args": ["--no-mmap"]},
        )

        assert not _unmapped_tokens(retry), f"the silenced respawn still loads unmapped: {retry}"
        assert capture["backend"].last_load_warning is None


def _unmapped_tokens(cmd):
    """The tokens in ``cmd`` that select a mode llama.cpp does not mmap.

    Copied from test_llama_cpp_placement.py rather than imported: these two files are
    separate harnesses and neither imports the other."""
    out = []
    for i, token in enumerate(cmd):
        if token in ("--no-mmap", "-no-mmap", "--no-direct-io", "-ndio"):
            out.append(token)
        elif token in ("--load-mode", "-lm") and i + 1 < len(cmd):
            if cmd[i + 1].strip().lower() in ("none", "mlock"):
                out.extend([token, cmd[i + 1]])
        elif token.split("=", 1)[0] in ("--load-mode", "-lm") and "=" in token:
            if token.split("=", 1)[1].strip().lower() in ("none", "mlock"):
                out.append(token)
    return out


class TestManualSplitLaunchesRespectTheGate:
    """Manual memory mode is not an explicit GPU pick, so the probe still opts into
    the gate (``for_llama_server = not gpu_ids``) -- but a manual per-GPU ratio took
    its own env branch, re-emitting the WHOLE visible set and handing the child the
    card the gate had just dropped."""

    def _manual_split(
        self,
        monkeypatch,
        tmp_path,
        targets,
        *,
        devices,
        capture = None,
        backend = None,
        tensor_split = (1.0, 1.0),
        gpu_layers = 20,
        tensor_parallel = False,
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(devices, vendor = "amd")
        return _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            targets,
            returncode = None,
            capture = capture,
            backend = backend,
            intent_kwargs = {
                "gpu_memory_mode": "manual",
                "gpu_layers": gpu_layers,
                "tensor_split": tensor_split,
                "tensor_parallel": tensor_parallel,
            },
        )

    def test_the_dropped_ratio_is_recorded_for_the_duplicate_load_check(
        self, tmp_path, monkeypatch, probe_env
    ):
        """Dropping the ratio from the argv is half the job: the UI re-sends the same
        request on every Apply and the duplicate-load check compares the live
        ``_tensor_split`` against it, so a launch that drops the ratio and records
        nothing respawns the same already-normalized server every time."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
            capture = capture,
        )
        backend = capture["backend"]
        assert backend._tensor_split is None  # gone from the argv
        assert backend._arch_gate_dropped_tensor_split == (1.0, 1.0)  # but not forgotten

    def test_a_covered_host_records_no_drop(self, tmp_path, monkeypatch, probe_env):
        """The record is the gate's doing. With every card covered the ratio is
        still live, so nothing may be recorded as dropped -- that entry is what
        excuses a mismatch, and an unearned one would dedupe a genuine change."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1031", free_mib = 12176),
            ],
            capture = capture,
        )
        backend = capture["backend"]
        assert list(backend._tensor_split) == [1.0, 1.0]
        assert backend._arch_gate_dropped_tensor_split is None

    def test_a_later_load_does_not_inherit_the_dropped_ratio(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The record excuses a mismatch, so it must not outlive the launch that
        earned it: after a gated split load, a load carrying no ratio would leave the
        stale entry excusing a split request against a server running none, and Apply
        would silently do nothing."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
            capture = capture,
        )
        backend = capture["backend"]
        assert backend._arch_gate_dropped_tensor_split == (1.0, 1.0)
        # Same backend, second load: covered host, no ratio, so nothing is dropped.
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1031", free_mib = 12176),
            ],
            backend = backend,
            tensor_split = None,
            gpu_layers = 21,  # differs, so this really relaunches rather than dedupes
        )
        assert backend._tensor_split is None
        assert backend._arch_gate_dropped_tensor_split is None

    def test_the_dropped_tensor_mode_is_recorded_too(self, tmp_path, monkeypatch, probe_env):
        """The ratio is half the normalization: narrowing to one survivor also drops
        --split-mode tensor, and ``_tensor_parallel_matches_loaded`` compares the
        unchanged request against the layer-split server, so an unrecorded drop reloads
        the same multi-GB model on every Apply."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
            capture = capture,
            tensor_parallel = True,
        )
        backend = capture["backend"]
        assert backend._tensor_parallel is False  # gone from the argv
        assert backend._arch_gate_dropped_tensor_parallel is True  # but not forgotten

    def test_a_covered_host_records_no_mode_drop(self, tmp_path, monkeypatch, probe_env):
        """The record excuses a mismatch, so an unearned one would dedupe a genuine
        layer-to-tensor change away."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1031", free_mib = 12176),
            ],
            capture = capture,
            tensor_parallel = True,
        )
        # Live, not normalized: the record would be unearned.
        assert capture["backend"]._tensor_parallel is True
        assert capture["backend"]._arch_gate_dropped_tensor_parallel is False

    def test_a_forced_cpu_launch_records_both_drops(self, tmp_path, monkeypatch, probe_env):
        """The forced-CPU arm strips the mode AND the ratio, so it records both: it is
        the one arm that normalizes a manual tensor request all the way down to a server
        with no visible device."""
        capture: dict = {}
        self._manual_split(
            monkeypatch,
            tmp_path,
            ["gfx908"],  # covers neither card
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
            capture = capture,
            tensor_parallel = True,
        )
        backend = capture["backend"]
        assert backend._arch_gate_forced_cpu is True
        assert backend._tensor_parallel is False and backend._tensor_split is None
        assert backend._arch_gate_dropped_tensor_parallel is True
        assert backend._arch_gate_dropped_tensor_split == (1.0, 1.0)

    def test_a_narrowed_host_masks_and_drops_the_ratio(self, tmp_path, monkeypatch, probe_env):
        launches = self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        # The ratio was sized for both cards; the mask re-indexes the survivor to
        # ordinal 0, so keeping it would weight the wrong device.
        assert "--tensor-split" not in cmd
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}

    def test_a_covered_host_keeps_its_ratio_and_its_order_pin(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The drop is the gate's doing. With every card covered the manual ratio
        survives untouched and the launch keeps the PCI order pin it always had."""
        launches = self._manual_split(
            monkeypatch,
            tmp_path,
            GFX103X,
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1031", free_mib = 12176),
            ],
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        assert cmd[cmd.index("--tensor-split") + 1] == "1,1"
        assert env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID"


class TestForcedCpuDropsTensorMode:
    """``--split-mode tensor`` with no visible device aborts the server instead of
    loading on CPU (the file says so twice: the manual gpu_layers=0 guard, and the
    paravirtual pin's note on "LLAMA_SPLIT_MODE_TENSOR not implemented for
    architecture"). Manual mode admits tensor parallelism on the FULL device count,
    so the forced-CPU mask could be reached with the flag still in the argv."""

    def test_the_forced_cpu_launch_carries_no_split_flags(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1031", free_mib = 12176)],
            vendor = "amd",
        )
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX120X,  # covers neither card
            returncode = None,
            capture = capture,
            intent_kwargs = {
                "gpu_memory_mode": "manual",
                "gpu_layers": 20,
                "tensor_parallel": True,
            },
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        assert _visibility(env) == {"HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1"}
        assert "--split-mode" not in cmd and "--tensor-split" not in cmd
        # /status must not advertise a mode the child was never given.
        assert capture["backend"]._tensor_parallel is False

    def test_a_covered_host_keeps_tensor_mode(self, tmp_path, monkeypatch, probe_env):
        """The normalisation is the gate's doing, not a blanket drop of tensor
        mode from every manual launch."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(
            [_device("gfx1030", free_mib = 12049), _device("gfx1031", free_mib = 12176)],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,
            returncode = None,
            intent_kwargs = {
                "gpu_memory_mode": "manual",
                "gpu_layers": 20,
                "tensor_parallel": True,
            },
        )
        cmd, _env = launches[0]
        assert cmd[cmd.index("--split-mode") + 1] == "tensor"


class TestGatedNarrowingDropsUnifiedMemory:
    """The unified-memory decision is made against ``gpu_indices``, None on the
    fit-owned and manual-split arms, so an uncovered APU anywhere on the host turns
    it on -- and the survivor mask then hands the child only discrete cards, where
    the same code calls it harmful."""

    def _apu_and_dgpu(self, monkeypatch):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _fake_torch(
            [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1151", free_mib = 40000, is_integrated = 1),
            ],
            vendor = "amd",
        )

    def test_a_fit_owned_narrowing_withdraws_it(self, tmp_path, monkeypatch, probe_env):
        torch = self._apu_and_dgpu(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,  # covers the dGPU, not the gfx1151 APU
            returncode = None,
            model_bytes = 400 * 1024**3,
        )
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
        assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env

    def test_an_inherited_user_value_still_stands(self, tmp_path, monkeypatch, probe_env):
        """Ownership, not presence: the withdrawal only takes back what this
        launch set, so a deliberate user value survives the narrowing."""
        torch = self._apu_and_dgpu(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,
            returncode = None,
            model_bytes = 400 * 1024**3,
            env_extra = {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"},
        )
        _cmd, env = launches[0]
        assert env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"

    def test_a_surviving_apu_keeps_it(self, tmp_path, monkeypatch, probe_env):
        """The withdrawal is scoped to a narrowing that leaves only discrete
        cards; an APU that survives the gate still wants the shared pool."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        torch = _fake_torch(
            [
                _device("gfx1151", free_mib = 40000, is_integrated = 1),
                _device("gfx1200", free_mib = 12049),
            ],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            ["gfx1151"],  # covers the APU, not the gfx1200 dGPU
            returncode = None,
            model_bytes = 400 * 1024**3,
        )
        _cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
        assert env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"


class TestGatedNarrowingDropsDeadTensorMode:
    """The proactive twin of TestArchCrashRetryDropsDeadTensorMode. Manual mode
    admits tensor parallelism on the FULL device count, so the survivor pin can leave
    one device running ``--split-mode tensor`` -- a no-op /status still advertises
    and that arms the MTP watchdog. ``_without_tensor_split`` takes only the ratio."""

    def _narrowed_tensor_load(
        self,
        monkeypatch,
        tmp_path,
        targets,
        *,
        devices,
        capture = None,
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        torch = _fake_torch(devices, vendor = "amd")
        return _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            targets,
            returncode = None,
            capture = capture,
            intent_kwargs = {
                "gpu_memory_mode": "manual",
                "gpu_layers": 20,
                "tensor_parallel": True,
            },
        )

    def test_one_survivor_drops_split_mode_tensor(self, tmp_path, monkeypatch, probe_env):
        capture: dict = {}
        launches = self._narrowed_tensor_load(
            monkeypatch,
            tmp_path,
            GFX103X,  # covers the dGPU, not the gfx1036 iGPU
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1036", free_mib = 12176),
            ],
            capture = capture,
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}
        assert "--split-mode" not in cmd and "-sm" not in cmd
        # /status must not advertise a mode one device cannot be running.
        assert capture["backend"]._tensor_parallel is False

    def test_two_survivors_keep_it(self, tmp_path, monkeypatch, probe_env):
        """Scoped to a narrowing that leaves ONE card, not to every narrowing:
        two survivors still split by tensor."""
        capture: dict = {}
        launches = self._narrowed_tensor_load(
            monkeypatch,
            tmp_path,
            GFX103X,  # covers 0 and 1, not the gfx1036 iGPU
            devices = [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1031", free_mib = 12176),
                _device("gfx1036", free_mib = 30000),
            ],
            capture = capture,
        )
        cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0,1", "CUDA_VISIBLE_DEVICES": "0,1"}
        assert cmd[cmd.index("--split-mode") + 1] == "tensor"
        assert capture["backend"]._tensor_parallel is True


class TestGatedNarrowingRechecksTheApuRamGuard:
    """The proactive twin of TestArchCrashRetryRechecksTheApuRamGuard, other
    direction. The preflight runs with ``gpu_indices`` None on an unpinned launch, so
    an uncovered APU anywhere on the host prices the load -- while the gate hands the
    child only the discrete survivor, whose VRAM the weights fit. Refusing there
    rejects a load that would have run (#7624)."""

    def _apu_and_dgpu(
        self,
        monkeypatch,
        *,
        avail_mib = 8000,
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
        )
        return _fake_torch(
            [
                _device("gfx1030", free_mib = 12049),
                _device("gfx1151", free_mib = 40000, is_integrated = 1),
            ],
            vendor = "amd",
        )

    @staticmethod
    def _shortfall_stub(calls):
        # Counted, not raised: this runs inside the launch path's own
        # except-Exception arms, which would swallow an AssertionError.
        def _shortfall(model_size_bytes, avail_mib, *_a, **_kw):
            calls.append((model_size_bytes, avail_mib))
            return "This model needs about 20 GB but only about 8 GB of memory is available."

        return _shortfall

    def test_a_discrete_survivor_waives_the_refusal(self, tmp_path, monkeypatch, probe_env):
        torch = self._apu_and_dgpu(monkeypatch)
        calls: list = []
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX103X,  # covers the dGPU, not the gfx1151 APU
            returncode = None,
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = self._shortfall_stub(calls),
        )
        assert len(calls) == 1, f"the RAM guard ran {len(calls)} times, expected once"
        assert capture.get("error") is None, capture.get("error")
        assert len(launches) == 1, "the load was refused for a device it never uses"
        _cmd, env = launches[0]
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"}

    def test_a_surviving_apu_still_warns(self, tmp_path, monkeypatch, probe_env):
        """The waiver is scoped to survivors that are all discrete. A build
        covering both cards leaves the APU in play, so the warning stands."""
        torch = self._apu_and_dgpu(monkeypatch)
        calls: list = []
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            [*GFX103X, "gfx1151"],
            returncode = None,
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = self._shortfall_stub(calls),
        )
        assert len(calls) == 1, f"the RAM guard ran {len(calls)} times, expected once"
        assert launches, "the oversized APU load never reached the child"
        assert "only about 8 GB" in (capture["backend"].last_load_warning or "")

    def test_the_forced_cpu_host_still_warns(self, tmp_path, monkeypatch, probe_env):
        """Every card gated out means the weights load into system RAM after all, so
        the guard the gate has no survivor to answer with still prices it."""
        torch = self._apu_and_dgpu(monkeypatch)
        calls: list = []
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            GFX120X,  # covers neither card
            returncode = None,
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = self._shortfall_stub(calls),
        )
        assert len(calls) == 1, f"the RAM guard ran {len(calls)} times, expected once"
        assert launches, "the oversized CPU-bound load never reached the child"
        assert "only about 8 GB" in (capture["backend"].last_load_warning or "")


class TestArchCrashRetryOntoAnApu:
    """The mirror of TestArchCrashRetryEnv (#7624). A markerless mixed host can as
    easily crash on the DISCRETE card and land on the APU, where the first launch
    correctly left GGML_CUDA_ENABLE_UNIFIED_MEMORY unset -- so handling only the
    withdrawal direction respawns onto a shared pool without the setting
    _amd_apu_wants_unified_memory says it needs. Mock-based, no ROCm here."""

    def _dgpu_then_apu(self, monkeypatch):
        # Device 1 is the APU, so the free-VRAM rank pins the dGPU first and the
        # retry's prefer-never-selected branch hands back exactly the APU.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _fake_torch(
            [
                _device("gfx1030", free_mib = 40000),
                _device("gfx1151", free_mib = 12000, is_integrated = 1),
            ],
            vendor = "amd",
        )

    def test_retry_onto_an_apu_sets_unified_memory_env(self, tmp_path, monkeypatch, probe_env):
        torch = self._dgpu_then_apu(monkeypatch)
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
        )
        assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in launches[0][1]
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all(env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1" for env in _retry)


class TestUnifiedMemoryOptOut:
    """Turning GGML_CUDA_ENABLE_UNIFIED_MEMORY off has to make it ABSENT (#8651).

    ggml gates on ``getenv(...) != nullptr``, so "0" is still on and the reporter's
    only route was patching the source. The host below is the reported one: a
    gfx1151 Strix Halo APU whose pool ROCm reports in full. Mock-based, no ROCm."""

    def _strix_halo(self, monkeypatch):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _fake_torch(
            [_device("gfx1151", free_mib = 47000, is_integrated = 1)],
            vendor = "amd",
        )

    def _load(
        self,
        tmp_path,
        monkeypatch,
        env_extra = None,
    ):
        return _run_auto_load(
            monkeypatch,
            tmp_path,
            self._strix_halo(monkeypatch),
            None,
            returncode = None,
            env_extra = env_extra,
        )

    def test_the_apu_still_gets_it_by_default(self, tmp_path, monkeypatch, probe_env):
        """Baseline: #5301 added the variable for exactly this hardware."""
        _cmd, env = self._load(tmp_path, monkeypatch)[0]
        assert env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"

    @pytest.mark.parametrize("value", ["0", "", "false", "FALSE", "no", "off", " 0 "])
    def test_a_falsy_user_value_is_not_passed_through(
        self, tmp_path, monkeypatch, probe_env, value
    ):
        """setdefault kept the user's "0", which ggml then read as enabled."""
        _cmd, env = self._load(tmp_path, monkeypatch, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": value})[0]
        assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env

    @pytest.mark.parametrize("value", ["1", "2", "true", "on"])
    def test_a_truthy_user_value_still_wins(self, tmp_path, monkeypatch, probe_env, value):
        """Only off spellings are intercepted; anything else passes through."""
        _cmd, env = self._load(tmp_path, monkeypatch, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": value})[0]
        assert env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == value

    def test_the_disable_switch_keeps_it_unset(self, tmp_path, monkeypatch, probe_env):
        """The switch users can find, mirroring UNSLOTH_DISABLE_DC_TUNING."""
        _cmd, env = self._load(tmp_path, monkeypatch, {"UNSLOTH_DISABLE_UNIFIED_MEMORY": "1"})[0]
        assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env

    def test_the_disable_switch_also_clears_an_inherited_value(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The switch has to beat a stale inherited "1" too."""
        _cmd, env = self._load(
            tmp_path,
            monkeypatch,
            {"UNSLOTH_DISABLE_UNIFIED_MEMORY": "1", "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"},
        )[0]
        assert "GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env

    def test_a_non_one_disable_value_does_nothing(self, tmp_path, monkeypatch, probe_env):
        """Exact "1", like the DC switch: firing on any value is the same trap."""
        _cmd, env = self._load(tmp_path, monkeypatch, {"UNSLOTH_DISABLE_UNIFIED_MEMORY": "0"})[0]
        assert env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1"

    def test_the_opt_out_survives_a_retry_onto_an_apu(self, tmp_path, monkeypatch, probe_env):
        """The retry re-adds the variable on an APU; it must not undo the opt-out."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        torch = _fake_torch(
            [
                _device("gfx1030", free_mib = 40000),
                _device("gfx1151", free_mib = 12000, is_integrated = 1),
            ],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            env_extra = {"UNSLOTH_DISABLE_UNIFIED_MEMORY": "1"},
        )
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all("GGML_CUDA_ENABLE_UNIFIED_MEMORY" not in env for _c, env in launches)


class TestArchCrashRetryDropsDeadTensorMode:
    """Narrowing to one device makes --split-mode tensor a no-op still REPORTED as
    active: tensor_parallel drives the UI and the MTP crash watchdog. Dropping
    --tensor-split alone left both behind (#7624)."""

    def test_a_single_gpu_retry_drops_split_mode_tensor(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        # Both cards are selected for the tensor split, so the retry cannot prefer
        # an unselected device and falls back to dropping the unified-memory one.
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        torch = _fake_torch(
            [
                _device("gfx1151", free_mib = 40000, is_integrated = 1),
                _device("gfx1030", free_mib = 30000),
            ],
            vendor = "amd",
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            intent_kwargs = {"tensor_parallel": True},
        )
        _retry = [cmd for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        for cmd in _retry:
            _modes = [
                cmd[i + 1]
                for i, tok in enumerate(cmd)
                if tok in ("--split-mode", "-sm") and i + 1 < len(cmd)
            ]
            assert "tensor" not in _modes, f"the narrowed respawn kept tensor mode: {cmd}"

    def test_the_retry_records_what_it_normalized(self, tmp_path, monkeypatch, probe_env):
        """The reactive path keeps the same record the proactive gate does: a markerless
        build only ever reaches the gate reactively, so without it every identical Apply
        reads the normalized server as new."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        capture: dict = {}
        _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(
                [
                    _device("gfx1151", free_mib = 40000, is_integrated = 1),
                    _device("gfx1030", free_mib = 30000),
                ],
                vendor = "amd",
            ),
            None,  # markerless: only the reactive retry can help
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            capture = capture,
            intent_kwargs = {"tensor_parallel": True},
        )
        backend = capture["backend"]
        assert backend._tensor_parallel is False  # normalized by the retry
        assert backend._arch_gate_dropped_tensor_parallel is True  # and recorded


class TestArchCrashRetryRechecksTheApuRamGuard:
    """The APU RAM preflight runs once, against the FIRST spawn's selection. On the
    mirror shape (crash on the dGPU, retry on the unified-memory sibling) the respawn
    is the first launch onto system RAM and skipped the guard entirely, so an
    oversized GGUF was OOM-killed mid-load instead of getting the refusal the same
    host gives when the APU is picked first (#7624). Mock-based, no ROCm here."""

    def _dgpu_then_apu(self, monkeypatch):
        # Device 1 is the APU, so the free-VRAM rank pins the dGPU first and the
        # retry's prefer-never-selected branch hands back exactly the APU.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 8000)
        )
        return _fake_torch(
            [
                _device("gfx1030", free_mib = 40000),
                _device("gfx1151", free_mib = 12000, is_integrated = 1),
            ],
            vendor = "amd",
        )

    def test_an_oversized_model_is_respawned_with_a_warning(self, tmp_path, monkeypatch, probe_env):
        torch = self._dgpu_then_apu(monkeypatch)
        # Counted, not raised: this runs inside the launch path's own
        # except-Exception arms, which would swallow an AssertionError and leave
        # the test green on a guard that never ran.
        calls: list = []

        def _shortfall(model_size_bytes, avail_mib, *_a, **_kw):
            calls.append((model_size_bytes, avail_mib))
            return "This model needs about 20 GB but only about 8 GB is available."

        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = _shortfall,
        )
        # The dGPU is pinned first, so the pre-launch guard never asks (it is
        # gated on the selection wanting unified memory) and the crash happens.
        assert launches, "the first spawn never ran"
        # The preflight still runs against the APU the retry lands on and still prices
        # the projector with the weights; it just no longer stops the respawn.
        assert len(calls) == 1, f"the RAM guard ran {len(calls)} times, expected once"
        assert calls[0][0] == 20 * 1024**3
        assert "only about 8 GB" in (capture["backend"].last_load_warning or "")
        # capture["error"] is now the simulated HIP crash, not the RAM guard, so
        # asserting the guard's text there would only re-test the mock.

    def test_a_model_that_fits_still_retries_onto_the_apu(self, tmp_path, monkeypatch, probe_env):
        """The guard warns on a shortfall, it does not block the fallback. With
        room to spare the retry runs exactly as before."""
        torch = self._dgpu_then_apu(monkeypatch)
        calls: list = []

        def _no_shortfall(model_size_bytes, avail_mib, *_a, **_kw):
            calls.append((model_size_bytes, avail_mib))
            return None

        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 1024,
            apu_ram_stub = _no_shortfall,
        )
        assert len(calls) == 1, f"the RAM guard ran {len(calls)} times, expected once"
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all(env.get("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == "1" for env in _retry)

    @staticmethod
    def _override_log(monkeypatch):
        """Collect the pageable-override log line. structlog, so caplog cannot see it."""
        lines: list = []
        monkeypatch.setattr(
            llama_cpp.logger,
            "warning",
            lambda msg, *a, **kw: lines.append(msg % a if a else msg),
        )
        return lines

    def _respawn_unmapped_and_oversized(self, tmp_path, monkeypatch, capture):
        """The respawn lands on the APU, is oversized there, and was asked to load
        without mmap -- the one shape where the override is what lets it finish."""
        torch = self._dgpu_then_apu(monkeypatch)
        return _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = lambda *_a, **_kw: (
                "This model needs about 20 GB but only about 8 GB of memory is available."
            ),
            intent_kwargs = {"extra_args": ["--no-mmap"]},
        )

    def test_the_opt_out_silences_the_retrys_apu_advisory_and_keeps_the_override(
        self, tmp_path, monkeypatch, probe_env
    ):
        """UNSLOTH_ALLOW_HOST_OFFLOAD is documented as silencing the warning and
        nothing else, but this site recorded the APU advisory without consulting it, so
        a user who opted out still got a memory_warning back. The verdict stays: the
        respawn is still remapped, or "none" would allocate the whole model in the RAM
        that cannot hold it."""
        monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
        capture: dict = {}
        logged = self._override_log(monkeypatch)

        launches = self._respawn_unmapped_and_oversized(tmp_path, monkeypatch, capture)

        _retry = [cmd for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all("--no-mmap" not in cmd for cmd in _retry), _retry
        assert capture["backend"].last_load_warning is None, (
            "the opt-out left the retry's APU advisory in memory_warning: "
            f"{capture['backend'].last_load_warning}"
        )
        assert [line for line in logged if "Overriding the unmapped load mode" in line], logged

    def test_without_the_opt_out_the_retrys_advisory_still_names_the_override(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The control. Nothing silenced, so the same respawn warns as before and the
        override is named in the text the route hands back."""
        monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
        capture: dict = {}

        launches = self._respawn_unmapped_and_oversized(tmp_path, monkeypatch, capture)

        _retry = [cmd for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert all("--no-mmap" not in cmd for cmd in _retry), _retry
        warning = capture["backend"].last_load_warning or ""
        assert "only about 8 GB" in warning, warning
        assert "memory mapping instead" in warning, warning


class TestArchCrashRetryReplacesTheCrashedSelectionsWarning:
    """The canonical #7624 shape, priced. The APU's shared-pool "free memory"
    outranks the dGPU, so auto pins it; the APU RAM guard warns that system RAM
    cannot hold the weights; the child then dies with a kernel-image error and the
    retry respawns on the discrete sibling, which holds the model in VRAM.

    ``_record_load_warning`` keeps the FIRST notice, so the crashed selection's
    verdict outlived the placement it described: the served response told the user
    the OS might stop a load running entirely on a discrete card. The retry prices
    the set it actually reaches, so that answer -- not the dead one -- is the load's.
    """

    def _apu_then_dgpu(self, monkeypatch, *, dgpu_free_mib):
        # Device 0 is the APU and outranks the dGPU on free memory, so auto pins it and
        # the pre-launch APU guard DOES ask (the mirror of
        # TestArchCrashRetryRechecksTheApuRamGuard, where the dGPU is pinned first and
        # the guard never asks). The APU's usable figure is its shared pool capped by
        # available system RAM minus a host reserve, so the RAM stub has to stay high
        # enough to leave the APU on top; the shortfall itself is the stubbed verdict,
        # which is how every APU-guard cell in this file drives it -- the arithmetic
        # belongs to test_host_offload_ram_guard.py, the plumbing is what is at stake.
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _fake_torch(
            [
                _device("gfx1151", free_mib = 47000, is_integrated = 1),
                _device("gfx1030", free_mib = dgpu_free_mib),
            ],
            vendor = "amd",
        )

    @staticmethod
    def _apu_stub(calls):
        # Counted, not raised: this runs inside the launch path's own
        # except-Exception arms, which would swallow an AssertionError.
        def _shortfall(model_size_bytes, avail_mib, *_a, **_kw):
            calls.append((model_size_bytes, avail_mib))
            return "This model needs about 20 GB but only about 8 GB of memory is available."

        return _shortfall

    def test_a_comfortable_retry_clears_the_dead_selections_warning(
        self, tmp_path, monkeypatch, probe_env
    ):
        # 30000MiB free holds the 20GB model outright, so the respawn spills nothing.
        torch = self._apu_then_dgpu(monkeypatch, dgpu_free_mib = 30000)
        calls: list = []
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,  # no marker: the proactive gate fails open, so the crash path runs
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = self._apu_stub(calls),
        )
        assert calls, "the pre-launch APU RAM guard never ran, so nothing was warned"
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert capture["backend"].last_load_warning is None, (
            "the response still carries the crashed APU's shortfall for a child that "
            "runs on the discrete card"
        )

    def test_a_shortfall_that_still_stands_survives_the_retry(
        self, tmp_path, monkeypatch, probe_env
    ):
        """Scoped: clearing the dead verdict must not silence a live one. The respawn
        prices the same weights against a NARROWER pool, so a real spill is re-warned.
        """
        # 4000MiB free leaves most of the 20GB model in host RAM on the respawn.
        torch = self._apu_then_dgpu(monkeypatch, dgpu_free_mib = 4000)
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            torch,
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            model_bytes = 20 * 1024**3,
            capture = capture,
            apu_ram_stub = self._apu_stub([]),
            host_offload_stub = (
                lambda *_a, **_kw: "About 16 GB of this model does not fit in GPU memory."
            ),
        )
        _retry = [env for _c, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert _retry, "the arch-crash retry did not fire"
        assert "does not fit in GPU memory" in (capture["backend"].last_load_warning or "")


class TestHsaOverrideGfxVersion:
    """``HSA_OVERRIDE_GFX_VERSION`` is the long-standing AMD workaround for an arch
    the ROCm stack does not build for: the user sets it, ROCr reports the SPOOFED
    arch, and code compiled for it really does run on the card. The gate reads the
    arch through the same device properties HIP will act on, so it sees the override
    too. Pinned because this is the shape most likely to read as "your fix broke my
    working setup": raw silicon uncovered, presented arch covered, and the launch has
    to follow the presented one."""

    def test_a_spoofed_arch_is_gated_on_what_the_runtime_reports(
        self, tmp_path, monkeypatch, probe_env
    ):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        _binary_with_marker(tmp_path, {"mapped_targets": GFX103X})
        # A gfx1035 laptop iGPU presenting itself as gfx1030 under the override,
        # beside a card the bundle does not cover at all.
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [_device("gfx1030", free_mib = 8000), _device("gfx1036", free_mib = 30000)],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(
            binary = str(tmp_path / "build" / "bin" / "llama-server"), for_llama_server = True
        ) == [(0, 8000)]


class TestAnInstallFromBeforeThisPr:
    """An install written by an older Unsloth has no ``mapped_targets``, and the
    fingerprint deliberately does not cover the field, so it is never refreshed for
    that reason alone. Such a host must behave EXACTLY as it did before the PR: the
    fix arrives with the next llama.cpp update, and until then nothing may change,
    least of all a drop to CPU."""

    OLD_MARKER = {
        "release_tag": "b10107",
        "asset": "app-b10107-windows-x64-rocm-gfx110X.zip",
        "install_kind": "app",
        "bundle_profile": "rocm",
        "runtime_line": "rocm",
        "coverage_class": "gfx110X",
        "install_fingerprint": "unchanged-by-this-pr",
        "installed_at_utc": "2026-07-01T00:00:00Z",
    }

    def test_the_probe_keeps_every_device(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        # The APU's usable figure is its shared pool minus a host reserve, and the
        # pool is capped by the REAL free system RAM unless this is stubbed. Without
        # it the expected 11152 silently assumes the machine has >12176MiB free, so
        # the test passes on a large runner and fails on a small one (measured: it
        # failed inside a WSL VM at 7340->6316MiB). Same stub as the sibling below.
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps(self.OLD_MARKER), encoding = "utf-8"
        )
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                [
                    _device("gfx1101", free_mib = 12049),
                    _device("gfx1036", free_mib = 12176, is_integrated = 1),
                ],
                vendor = "amd",
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(
            binary = str(tmp_path / "build" / "bin" / "llama-server"), for_llama_server = True
        ) == [(0, 12049), (1, 11152)]

    def test_the_launch_is_not_masked_onto_the_cpu(self, tmp_path, monkeypatch, probe_env):
        """The failure that would matter most: an old install that used to run
        on the GPU being gated to CPU by a marker the gate cannot read."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps(self.OLD_MARKER), encoding = "utf-8"
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(
                [
                    _device("gfx1101", free_mib = 12049),
                    _device("gfx1036", free_mib = 30000, is_integrated = 1),
                ],
                vendor = "amd",
            ),
            None,  # the marker above stands; do not overwrite it
            returncode = None,
        )
        assert len(launches) == 1
        _cmd, env = launches[0]
        assert env.get("HIP_VISIBLE_DEVICES") != "-1", "an old install was gated onto the CPU"
        # Pre-PR placement, reproduced: the shared-pool iGPU still outranks the dGPU
        # (#7669) until the install is refreshed. CUDA_VISIBLE_DEVICES is 0 because
        # ROCr re-indexes the visible agents from zero; the physical pick is ROCR's.
        assert _visibility(env) == {"ROCR_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "0"}


class TestArchForcedCpuFlagLifecycle:
    """``_arch_gate_forced_cpu`` drives ``holds_no_vram``, so it has to be
    per-load state: a CPU-masked launch that kept the flag would go on claiming
    zero VRAM for the GPU load that replaces it."""

    def _host(self):
        return [
            _device("gfx1101", free_mib = 12049),
            _device("gfx1036", free_mib = 30000, is_integrated = 1),
        ]

    def test_every_device_uncovered_sets_it(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        capture: dict = {}
        _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(self._host(), vendor = "amd"),
            GFX103X,  # covers neither card
            returncode = None,
            capture = capture,
        )
        backend = capture["backend"]
        assert backend._arch_gate_forced_cpu is True
        # The GPU arbiter reads this: a child masked onto the CPU must not keep
        # the CHAT claim or block an image/video pipeline.
        assert backend.holds_no_vram is True

    def test_a_covered_card_leaves_it_alone(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        capture: dict = {}
        _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(self._host(), vendor = "amd"),
            GFX110X,  # the dGPU survives
            returncode = None,
            capture = capture,
        )
        backend = capture["backend"]
        assert backend._arch_gate_forced_cpu is False
        assert backend.holds_no_vram is False

    def test_unload_clears_it(self):
        backend = LlamaCppBackend()
        backend._arch_gate_forced_cpu = True
        backend.unload_model()
        assert backend._arch_gate_forced_cpu is False


class TestTheForcedCpuFlagIsNotSticky:
    """``load_model`` phase 1 only kills the old process, so per-load state only ever
    set TRUE outlives the launch that set it. The dangerous direction for this flag:
    a host that gains coverage (a llama.cpp update, or just the next model) would
    report a VRAM-holding server as holding none, and the arbiter leave it unclaimed
    beside a competing workload."""

    def _load(self, monkeypatch, tmp_path, targets, capture):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        return _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(
                [
                    _device("gfx1101", free_mib = 12049),
                    _device("gfx1036", free_mib = 30000, is_integrated = 1),
                ],
                vendor = "amd",
            ),
            targets,
            returncode = None,
            capture = capture,
        )

    def test_a_covered_load_after_a_gated_one_clears_it(self, tmp_path, monkeypatch, probe_env):
        """The llama.cpp-update shape: the SAME backend instance loads again on
        a build that now covers the dGPU, with no unload in between (load_model
        phase 1 only kills the process)."""
        gated: dict = {}
        self._load(monkeypatch, tmp_path, GFX103X, gated)  # covers neither card
        assert gated["backend"]._arch_gate_forced_cpu is True

        covered: dict = {}
        second = tmp_path / "second"
        second.mkdir()
        self._load(monkeypatch, second, GFX110X, covered)
        # Carry the stale flag onto the instance the second load ran on: the
        # harness builds a fresh backend per drive, so without this the assert
        # would pass on a default rather than on the publish.
        backend = covered["backend"]
        assert backend._arch_gate_forced_cpu is False
        assert backend.holds_no_vram is False

    def test_the_publish_overwrites_a_stale_true(self, tmp_path, monkeypatch, probe_env):
        """The mutation the class exists for, on one instance: pre-set the flag,
        run a covered load, and the launch must publish False over it."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        _real_init = LlamaCppBackend.__init__

        def _init_with_stale_flag(self, *args, **kwargs):
            _real_init(self, *args, **kwargs)
            self._arch_gate_forced_cpu = True

        monkeypatch.setattr(LlamaCppBackend, "__init__", _init_with_stale_flag)
        covered: dict = {}
        self._load(monkeypatch, tmp_path, GFX110X, covered)
        assert covered["backend"]._arch_gate_forced_cpu is False
        assert covered["backend"].holds_no_vram is False

    def test_the_diffusion_state_path_clears_it(self):
        """A diffusion runner does hold VRAM, and its state block already resets
        the sibling chat fields for exactly this reason."""
        source = (
            Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        ).read_text(encoding = "utf-8")
        block = source.split("Diffusion is never tensor-parallel")[1].split("def ")[0]
        assert "self._arch_gate_forced_cpu = False" in block

    def test_the_flag_is_published_not_only_set(self):
        """Source-level, because the miss was structural: a branch that only
        assigns True can never clear a previous launch's value."""
        source = (
            Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        ).read_text(encoding = "utf-8")
        assert "self._arch_gate_forced_cpu = bool(_arch_gate_forced_cpu)" in source
        assert "self._arch_gate_forced_cpu = True" not in source


class TestInheritedSplitEnvGoesWithTheArgvStrip:
    """LLAMA_ARG_SPLIT_MODE / LLAMA_ARG_TENSOR_SPLIT are the env spelling of
    --split-mode / --tensor-split, so dropping the tokens alone leaves the inherited
    value in force -- and the forced-CPU arm strips --split-mode precisely because a
    tensor split aborts a child with no visible device."""

    def _host(self):
        return [
            _device("gfx1101", free_mib = 12049),
            _device("gfx1036", free_mib = 30000, is_integrated = 1),
        ]

    def test_the_forced_cpu_launch_clears_both(self, tmp_path, monkeypatch, probe_env):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(self._host(), vendor = "amd"),
            GFX103X,  # covers neither card
            returncode = None,
            # Split mode unset on purpose. The existing tensor->layer
            # reconciliation only clears the pair when an inherited mode is
            # present and non-layer, so this is the shape that survives it.
            env_extra = {"LLAMA_ARG_TENSOR_SPLIT": "3,1"},
        )
        assert len(launches) == 1
        cmd, env = launches[0]
        assert "LLAMA_ARG_TENSOR_SPLIT" not in env
        assert "--split-mode" not in cmd and "--tensor-split" not in cmd
        assert env.get("HIP_VISIBLE_DEVICES") == "-1"

    def test_an_inherited_tensor_mode_cannot_survive_the_forced_cpu_strip(
        self, tmp_path, monkeypatch, probe_env
    ):
        """The abort this strip prevents: llama.cpp fails the load with
        "LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices" once the mask hides every device,
        and an inherited tensor mode reinstates it after the argv flag is gone."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(self._host(), vendor = "amd"),
            GFX103X,  # covers neither card
            returncode = None,
            intent_kwargs = {"tensor_parallel": True},
            env_extra = {"LLAMA_ARG_SPLIT_MODE": "tensor"},
        )
        assert launches
        cmd, env = launches[0]
        assert "LLAMA_ARG_SPLIT_MODE" not in env
        assert "--split-mode" not in cmd

    def test_the_narrowed_pin_clears_both(self, tmp_path, monkeypatch, probe_env):
        """A survivor pin re-indexes the visible devices, so an inherited
        positional ratio would land on the wrong card."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_arch_gate_survivors", staticmethod(lambda _b = None: [0])
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(self._host(), vendor = "amd"),
            GFX110X,
            returncode = None,
            model_bytes = 400 * 1024**3,  # too large to place: --fit on owns it
            env_extra = {"LLAMA_ARG_TENSOR_SPLIT": "3,1"},
        )
        assert launches
        _cmd, env = launches[0]
        assert "LLAMA_ARG_TENSOR_SPLIT" not in env

    def test_an_ordinary_load_keeps_them(self, tmp_path, monkeypatch, probe_env):
        """Only the gate's own branches clear these. An unrelated launch must
        not lose an inherited setting the existing reconciliation allows."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000)
        )
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch([_device("gfx1101", free_mib = 20000)], vendor = "amd"),
            GFX110X,  # covers the only card: nothing to narrow
            returncode = None,
            env_extra = {"LLAMA_ARG_TENSOR_SPLIT": "3,1"},
        )
        assert launches
        _cmd, env = launches[0]
        assert env.get("LLAMA_ARG_TENSOR_SPLIT") == "3,1"


class TestTheForcedCpuLaunchAppliesThePageLock:
    """The gate masks every device away, so the child runs from host RAM, but
    ``_weights_in_host_memory`` answered for the ORIGINAL placement, where a manual full
    offload onto discrete cards reads as not host-resident. Left alone the page-lock is
    skipped AND recorded as deliberate, which no relaunch undoes."""

    def _run(self, tmp_path, monkeypatch, *, targets):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 200000)
        )
        import utils.model_memory_settings as _mem_settings

        monkeypatch.setattr(_mem_settings, "get_model_memory_settings", lambda: (True, False))
        capture: dict = {}
        backend = LlamaCppBackend()
        # A known block count is what lets _offloads_every_layer answer True for the
        # manual maximum below; without it every launch reads as host-resident and the
        # skip this test is about could never happen.
        backend._n_layers = 32
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch(
                # Two DISCRETE cards: _amd_apu_wants_unified_memory answers False, so
                # a full offload really does read as not host-resident.
                [
                    _device("gfx1030", free_mib = 40000),
                    _device("gfx1031", free_mib = 30000),
                ],
                vendor = "amd",
            ),
            targets,
            returncode = None,
            capture = capture,
            backend = backend,
            # Manual full offload: every layer on the GPU.
            intent_kwargs = {"gpu_memory_mode": "manual", "gpu_layers": 33},  # n_layers + 1
        )
        return launches, capture

    def test_the_masked_child_gets_the_lock(self, tmp_path, monkeypatch, probe_env):
        launches, capture = self._run(tmp_path, monkeypatch, targets = ["gfx908"])
        assert launches
        cmd, env = launches[0]
        assert env.get("HIP_VISIBLE_DEVICES") == "-1", _visibility(env)
        assert "--mlock" in cmd or "mmap+mlock" in " ".join(cmd), cmd
        assert capture["backend"]._memory_mlock_applicable is True

    def test_a_covered_host_still_skips_it(self, tmp_path, monkeypatch, probe_env):
        """The recompute is the gate's doing. With the cards covered the launch really
        is fully offloaded, so the pre-existing skip has to stand."""
        launches, capture = self._run(tmp_path, monkeypatch, targets = GFX103X)
        assert launches
        cmd, env = launches[0]
        assert env.get("HIP_VISIBLE_DEVICES") != "-1"
        assert "--mlock" not in cmd
        assert capture["backend"]._memory_mlock_applicable is False


class TestForcedCpuNeedsRealArchEvidence:
    """``_get_gpu_memory`` turns any probe error into [], so "gated empty, ungated not"
    also describes a one-shot failure of the FIRST probe on a host the gate never
    filters. Masking that onto the CPU is a silent, permanent cliff, so the branch
    re-derives the gate's verdict instead of inferring it."""

    def _run(self, tmp_path, monkeypatch, *, targets, flaky):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        real = LlamaCppBackend._get_gpu_memory
        calls = {"n": 0}

        def _probe(binary = None, *, for_llama_server = False):
            calls["n"] += 1
            if flaky and calls["n"] == 1:
                return []  # the transient failure, on the gated call
            return real(binary, for_llama_server = for_llama_server)

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_probe))
        capture: dict = {}
        _run_auto_load(
            monkeypatch,
            tmp_path,
            _fake_torch([_device("gfx1030", free_mib = 12049)], vendor = "amd"),
            targets,
            returncode = None,
            capture = capture,
        )
        return capture["backend"]

    def test_a_transient_probe_failure_does_not_force_cpu(self, tmp_path, monkeypatch, probe_env):
        """The marker covers the only card, so nothing was filtered: an empty gated
        result here is the probe, not the gate."""
        backend = self._run(tmp_path, monkeypatch, targets = GFX103X, flaky = True)
        assert backend._arch_gate_forced_cpu is False

    def test_an_unmarked_install_does_not_force_cpu(self, tmp_path, monkeypatch, probe_env):
        """No marker at all means unknown coverage, which the filter fails open on, so
        the gate cannot have emptied anything."""
        backend = self._run(tmp_path, monkeypatch, targets = None, flaky = True)
        assert backend._arch_gate_forced_cpu is False

    def test_a_genuinely_uncovered_host_still_forces_cpu(self, tmp_path, monkeypatch, probe_env):
        backend = self._run(tmp_path, monkeypatch, targets = ["gfx908"], flaky = False)
        assert backend._arch_gate_forced_cpu is True


class TestTheApuRetryRecomputesThePageLock:
    """Residency is a property of the DEVICES, which the arch-crash retry changes:
    crash on the discrete card, land on the unified-memory APU, and the weights are
    host-backed after all, so the lock the first launch skipped is the one the user
    asked for. Left alone the respawn runs unlocked and records the missing lock as
    deliberate, deduping away the reload that would apply it."""

    def _dgpu_then_apu(self, monkeypatch):
        # The dGPU is picked first (more free VRAM after the APU host reserve),
        # then crashes; the APU is what remains.
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {1})
        )
        return _fake_torch(
            [
                # The dGPU wins the free-memory rank (the APU's shared pool is
                # reported minus the host reserve), so it is picked, crashes, and
                # the APU is what remains.
                _device("gfx1201", free_mib = 40000),
                _device("gfx1151", free_mib = 30000, is_integrated = 1),
            ],
            vendor = "amd",
        )

    def _run(self, tmp_path, monkeypatch, *, mlock):
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 200000)
        )
        import utils.model_memory_settings as _mem_settings

        # (keep_resident, no_ram_reserve): both should_mlock and
        # apply_model_memory_policy read this one snapshot.
        monkeypatch.setattr(_mem_settings, "get_model_memory_settings", lambda: (mlock, False))

        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            self._dgpu_then_apu(monkeypatch),
            None,  # markerless: only the reactive retry can help
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            capture = capture,
        )
        return launches, capture

    def test_the_respawn_onto_the_apu_gets_the_lock(self, tmp_path, monkeypatch, probe_env):
        launches, capture = self._run(tmp_path, monkeypatch, mlock = True)
        retry = [(cmd, env) for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert (
            retry
        ), f"the arch-crash retry never targeted the APU: {[_visibility(e) for _c, e in launches]}"
        cmd, _env = retry[0]
        assert "--mlock" in cmd or "mmap+mlock" in " ".join(cmd), cmd
        # And the record agrees with the child, or the reload that would apply
        # the policy is deduplicated away.
        assert capture["backend"]._memory_mlock_applicable is True

    def test_page_locking_off_changes_nothing(self, tmp_path, monkeypatch, probe_env):
        launches, _capture = self._run(tmp_path, monkeypatch, mlock = False)
        assert launches
        for cmd, _env in launches:
            assert "--mlock" not in cmd

    def test_the_users_own_extra_args_survive_the_recompute(self, tmp_path, monkeypatch, probe_env):
        """The recompute may only ADD the lock: taking the crashed launch's memory
        flags back off would mean scanning for --mlock / --no-mmap, valueless in
        llama.cpp's parser, so the scan drops the argv entry after them -- here the
        user's own -c 8192."""
        _apply_os(monkeypatch, "linux", is_rocm = True)
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 200000)
        )
        import utils.model_memory_settings as _mem_settings

        monkeypatch.setattr(_mem_settings, "get_model_memory_settings", lambda: (True, False))
        capture: dict = {}
        launches = _run_auto_load(
            monkeypatch,
            tmp_path,
            self._dgpu_then_apu(monkeypatch),
            None,
            returncode = 1,
            output = "ROCm error: device kernel image is invalid",
            capture = capture,
            # --no-mmap is a memory flag apply_model_memory_policy keeps when it
            # emits the legacy --mlock (this build reports no --load-mode), and
            # -c 8192 is the entry a valueless-flag scan would eat with it.
            intent_kwargs = {"extra_args": ["--no-mmap", "-c", "8192"]},
        )
        retry = [(cmd, env) for cmd, env in launches if env.get("ROCR_VISIBLE_DEVICES") == "1"]
        assert retry, [_visibility(e) for _c, e in launches]
        cmd, _env = retry[0]
        assert "--no-mmap" in cmd, f"a user memory flag was stripped by the recompute: {cmd}"
        assert cmd[cmd.index("-c") + 1] == "8192", f"the extra after --no-mmap was eaten: {cmd}"
        assert "--mlock" in cmd, cmd
        # The record has to describe the argv actually launched, or the reload
        # comparator fights a child it already agrees with.
        assert capture["backend"]._memory_mlock_applicable is True
        assert capture["backend"]._memory_state[0] is True  # (mlock, reserves_ram)
