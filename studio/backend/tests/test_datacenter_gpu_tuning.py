# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data-center llama.cpp env tuning: datacenter NVIDIA parts only, never consumer
GeForce, ROCm, CPU or macOS. User values win; UNSLOTH_DISABLE_DC_TUNING=1 disables.

P2P has a second, stricter gate (#10613): a datacenter NAME is not evidence of an
NVLink fabric, and on a non-NVLink box the peer copy is silently discarded while
still reporting success, so every model emits garbage. It needs a confirmed NV#
link across the selection and fails CLOSED on unknowns.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    names,
    *,
    hip = None,
    cuda_ok = True,
):
    """torch stub: version.hip, cuda.*, get_device_properties(i).name."""
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_ok,
        device_count = lambda: len(names),
        get_device_properties = lambda i: types.SimpleNamespace(name = names[i]),
    )
    return t


# Real `nvidia-smi topo -m` from an 8x B200 NVLink host: ANSI-underlined header, NIC
# rows/columns, affinity columns and a Legend, none of which a toy string exercises.
TOPO_NVLINK_8X = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tNIC0\tNIC1\t"
    "CPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tSYS\tSYS\t0-47,96-143\t0\t\tN/A\n"
    "GPU1\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tSYS\tSYS\t0-47,96-143\t0\t\tN/A\n"
    "GPU2\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tNV18\tSYS\tSYS\t0-47,96-143\t0\t\tN/A\n"
    "GPU3\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tNV18\tSYS\tSYS\t0-47,96-143\t0\t\tN/A\n"
    "GPU4\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tNV18\tPIX\tPIX\t48-95,144-191\t1\t\tN/A\n"
    "GPU5\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNV18\tPIX\tPIX\t48-95,144-191\t1\t\tN/A\n"
    "GPU6\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNV18\tNODE\tNODE\t48-95,144-191\t1\t\tN/A\n"
    "GPU7\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\tNV18\t X \tNODE\tNODE\t48-95,144-191\t1\t\tN/A\n"
    "NIC0\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\tNODE\tNODE\t X \tPIX\t\t\t\n"
    "NIC1\tSYS\tSYS\tSYS\tSYS\tPIX\tPIX\tNODE\tNODE\tPIX\t X \t\t\t\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes\n"
    "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges\n"
    "  NV#  = Connection traversing a bonded set of # NVLinks\n"
)

# The reporter's host (#10613): 2x RTX 6000 Ada, NODE (PCIe via a host bridge), no
# NVLink. The driver still advertises P2P as available.
TOPO_PCIE_2X = (
    "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tNODE\t0-23\t0\t\tN/A\n"
    "GPU1\tNODE\t X \t0-23\t0\t\tN/A\n"
    "\n"
    "Legend:\n"
    "\n"
    "  X    = Self\n"
    "  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges\n"
)


@pytest.fixture(autouse = True)
def _clear_cuda_visible_devices(monkeypatch):
    """Run unmasked by default (physical id == ordinal); masked tests set it."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)


# Captured before the autouse fixture stubs it, so the probe can run for real.
_REAL_IOMMU_IS_TRANSLATING = LlamaCppBackend.__dict__["_iommu_is_translating"].__func__
_REAL_NVML_LIBRARY = LlamaCppBackend.__dict__["_nvml_library"].__func__


def _no_nvidia_smi(*a, **k):
    raise FileNotFoundError("nvidia-smi")


@pytest.fixture(autouse = True)
def _isolate_host_topology(monkeypatch):
    """Keep the P2P gate off the real host: only nvidia-smi's output is stubbed (the
    parser always runs), caches are dropped either side, and the platform probes are
    pinned so a CI box with GPUs cannot colour the results.

    NVML is stubbed absent by default so the topo tests still exercise the parser;
    with a real driver the fast path would answer first and the canned `nvidia-smi`
    output would never be read. The NVML tests install their own fake."""
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    LlamaCppBackend._NVLINK_TOPO_GENERATION = 0
    LlamaCppBackend._IOMMU_CACHE = None
    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    monkeypatch.setattr(LlamaCppBackend, "_nvml_library", staticmethod(lambda: None))
    monkeypatch.delenv("UNSLOTH_P2P_TOPO_CROSSCHECK", raising = False)
    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: False))
    monkeypatch.setattr(LlamaCppBackend, "_running_virtualized", staticmethod(lambda: False))
    # Overrides off and the warn-once latch reset: no host env, no ordering leak.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_P2P", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_DC_P2P", raising = False)
    # Most tests are about the topology verdict, not device ordering, so default to
    # a pinned order; the unpinned-path tests delete this themselves.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    LlamaCppBackend._warned_no_nvlink = False
    yield
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    LlamaCppBackend._NVLINK_TOPO_GENERATION = 0
    LlamaCppBackend._IOMMU_CACHE = None
    LlamaCppBackend._warned_no_nvlink = False


def _use_topo(
    monkeypatch,
    text,
    returncode = 0,
):
    """Feed the real parser canned `nvidia-smi topo -m` output."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode = returncode, stdout = text, stderr = ""),
    )
    LlamaCppBackend._NVLINK_TOPO_CACHE = None


# ---------------------------------------------------------------------------
# _is_datacenter_gpu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names,expected",
    [
        # Datacenter / professional parts.
        (["NVIDIA A100-SXM4-80GB"], True),
        (["NVIDIA A30"], True),
        (["NVIDIA H100 80GB HBM3"], True),
        (["NVIDIA H200"], True),
        (["NVIDIA H800"], True),
        (["NVIDIA GH200 480GB"], True),
        (["NVIDIA B200"], True),
        (["NVIDIA GB200"], True),
        (["NVIDIA L40S"], True),
        (["NVIDIA L4"], True),
        (["NVIDIA RTX PRO 6000 Blackwell Server Edition"], True),
        (["NVIDIA RTX 6000 Ada Generation"], True),
        # Consumer GeForce: never.
        (["NVIDIA GeForce RTX 4090"], False),
        (["NVIDIA GeForce RTX 5090"], False),
        (["NVIDIA GeForce RTX 3090"], False),
        (["NVIDIA GeForce RTX 2080 Ti"], False),
        (["NVIDIA GeForce GTX 1080"], False),
        # Short markers must not match as substrings ("a100" in "A1000").
        (["NVIDIA RTX A1000 Laptop GPU"], False),
        (["NVIDIA RTX A1000 6GB Laptop GPU"], False),
        (["NVIDIA RTX A3000 Laptop GPU"], False),
        # Homogeneous multi-DC: all must match.
        (["NVIDIA B200", "NVIDIA B200"], True),
        (["NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3"], True),
        # Mixed: non-DC, so tuning never lands on the GeForce.
        (["NVIDIA B200", "NVIDIA GeForce RTX 4090"], False),
        (["NVIDIA GeForce RTX 4090", "NVIDIA B200"], False),
    ],
)
def test_is_datacenter_gpu(monkeypatch, names, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(names))
    assert LlamaCppBackend._is_datacenter_gpu() is expected


def test_is_datacenter_gpu_respects_selection(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(["NVIDIA B200", "NVIDIA GeForce RTX 4090"]),
    )
    assert LlamaCppBackend._is_datacenter_gpu([0]) is True
    assert LlamaCppBackend._is_datacenter_gpu([1]) is False
    assert LlamaCppBackend._is_datacenter_gpu([0, 1]) is False


def test_is_datacenter_gpu_out_of_range_indices_skipped(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    # Invalid indices are skipped; all invalid -> nothing seen -> False.
    assert LlamaCppBackend._is_datacenter_gpu([0, 5, -1]) is True
    assert LlamaCppBackend._is_datacenter_gpu([5, 9]) is False


def test_is_datacenter_gpu_masked_host_physical_ids(monkeypatch):
    # PHYSICAL selection [4,5] must resolve (the pre-fix bug: 4 >= device_count).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._is_datacenter_gpu([4, 5]) is True
    assert LlamaCppBackend._is_datacenter_gpu([4, 5, 6, 7]) is True
    assert LlamaCppBackend._is_datacenter_gpu(None) is True
    assert LlamaCppBackend._is_datacenter_gpu([0, 1]) is False  # not visible -> skip


def test_is_datacenter_gpu_masked_host_reordered(monkeypatch):
    # Reordered mask preserves order: ordinal 0 -> physical 7.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,4,5,6")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100 80GB HBM3"] * 4))
    assert LlamaCppBackend._is_datacenter_gpu([7, 4]) is True


def test_is_datacenter_gpu_masked_host_mixed_class(monkeypatch):
    # Detection must follow the selected physical GPU, not a same-numbered ordinal.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(["NVIDIA GeForce RTX 4090", "NVIDIA B200"]),
    )
    assert LlamaCppBackend._is_datacenter_gpu([4]) is False
    assert LlamaCppBackend._is_datacenter_gpu([5]) is True
    assert LlamaCppBackend._is_datacenter_gpu([4, 5]) is False


def test_is_datacenter_gpu_unparsable_mask_falls_back(monkeypatch):
    # Unparsable (UUID) mask falls back to physical id == ordinal.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef12")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    assert LlamaCppBackend._is_datacenter_gpu([0]) is True


def test_is_datacenter_gpu_rocm_is_false(monkeypatch):
    # ROCm reuses torch.cuda.*.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(["AMD Instinct MI300X"], hip = "6.2.0"),
    )
    assert LlamaCppBackend._is_datacenter_gpu() is False


def test_is_datacenter_gpu_no_cuda_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], cuda_ok = False))
    assert LlamaCppBackend._is_datacenter_gpu() is False


def test_is_datacenter_gpu_missing_torch_is_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._is_datacenter_gpu() is False


# ---------------------------------------------------------------------------
# _effective_gpu_count
# ---------------------------------------------------------------------------


def test_effective_gpu_count_explicit_selection(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._effective_gpu_count([0]) == 1
    assert LlamaCppBackend._effective_gpu_count([0, 1, 2]) == 3


def test_effective_gpu_count_none_uses_visible(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._effective_gpu_count(None) == 4


def test_effective_gpu_count_no_cuda_is_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], cuda_ok = False))
    assert LlamaCppBackend._effective_gpu_count(None) == 0


def test_effective_gpu_count_missing_torch_is_zero(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert LlamaCppBackend._effective_gpu_count(None) == 0


# ---------------------------------------------------------------------------
# _apply_datacenter_env (the env-injection decision)
# ---------------------------------------------------------------------------


def test_apply_env_single_dc_gpu_sets_only_fp32(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is True
    assert env == {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"}
    assert "GGML_CUDA_P2P" not in env  # no multi-GPU flags on one GPU
    assert "CUDA_SCALE_LAUNCH_QUEUES" not in env


def test_apply_env_multi_dc_gpu_sets_all(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"
    assert env["GGML_CUDA_P2P"] == "1"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"


def test_apply_env_none_indices_uses_visible_count(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100", "NVIDIA H100"]))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, None) is True
    assert env["GGML_CUDA_P2P"] == "1"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"


def test_apply_env_consumer_gpu_is_noop(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA GeForce RTX 4090"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is False
    assert env == {}


def test_apply_env_user_value_wins(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env = {
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "0",  # user disabled
        "CUDA_SCALE_LAUNCH_QUEUES": "8x",  # user override
    }
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    # setdefault must not clobber user values; the unset one defaults.
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "0"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "8x"
    assert env["GGML_CUDA_P2P"] == "1"


def test_apply_env_disable_flag_respected(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_TUNING", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is False
    assert env == {}


def test_apply_env_fail_open_on_detection_error(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", None)  # detection raises -> False
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is False
    assert env == {}


def test_apply_env_masked_host_multi_dc(monkeypatch):
    # Masked host end-to-end: pre-fix this applied no tuning at all.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [4, 5]) is True
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"
    assert env["GGML_CUDA_P2P"] == "1"
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"


# ---------------------------------------------------------------------------
# nvidia-smi topo -m parsing
# ---------------------------------------------------------------------------


def test_topo_parses_real_nvlink_table(monkeypatch):
    # 8 GPUs -> 8*7 ordered pairs; NIC rows, affinity columns and legend ignored.
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    matrix = LlamaCppBackend._nvlink_topology()
    assert len(matrix) == 56
    assert set(matrix.values()) == {"NV18"}
    assert (0, 0) not in matrix  # self ("X") is not a pair
    assert matrix[(6, 7)] == "NV18"


def test_topo_parses_pcie_table(monkeypatch):
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    assert LlamaCppBackend._nvlink_topology() == {(0, 1): "NODE", (1, 0): "NODE"}


def test_topo_is_cached_then_refreshable(monkeypatch):
    calls = []

    def _once(*a, **k):
        calls.append(1)
        return types.SimpleNamespace(returncode = 0, stdout = TOPO_PCIE_2X, stderr = "")

    monkeypatch.setattr(subprocess, "run", _once)
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    LlamaCppBackend._nvlink_topology()
    LlamaCppBackend._nvlink_topology()
    assert len(calls) == 1  # one shell-out per process
    LlamaCppBackend._nvlink_topology(refresh = True)
    assert len(calls) == 2


def test_topo_unavailable_paths_are_none(monkeypatch):
    # Missing binary, non-zero exit and a GPU-less table mean "unknown", never NVLink.
    assert LlamaCppBackend._nvlink_topology() is None
    _use_topo(monkeypatch, TOPO_NVLINK_8X, returncode = 9)
    assert LlamaCppBackend._nvlink_topology() is None
    _use_topo(monkeypatch, "Legend:\n  X = Self\n")
    assert LlamaCppBackend._nvlink_topology() is None


def test_topo_truncated_row_refuses_to_guess(monkeypatch):
    _use_topo(
        monkeypatch,
        "\tGPU0\tGPU1\tCPU Affinity\n"
        "GPU0\t X \tNV18\t0-23\n"
        "GPU1\tNV18\n",  # row cut short: fewer labels than columns
    )
    assert LlamaCppBackend._nvlink_topology() is None


# ---------------------------------------------------------------------------
# _p2p_veto_reason (the #10613 gate)
# ---------------------------------------------------------------------------


def test_p2p_vetoed_on_rtx_6000_ada(monkeypatch):
    # A datacenter NAME with no NVLink.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert reason is not None and "NVLink-capable" in reason


@pytest.mark.parametrize(
    "name",
    ["NVIDIA L40S", "NVIDIA L40", "NVIDIA L4", "NVIDIA RTX PRO 6000 Blackwell Server Edition"],
)
def test_p2p_vetoed_on_other_connectorless_parts(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([name] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)  # even a lying matrix must not rescue them
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None


def test_p2p_allowed_on_confirmed_nvlink(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 8))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


def test_p2p_vetoed_when_topology_unknown(monkeypatch):
    # NVLink-capable parts but no nvidia-smi: fail CLOSED (the old code trusted the
    # name alone).
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert reason is not None and "interconnect matrix" in reason


def test_p2p_vetoed_on_pcie_pair_between_nvlink_parts(monkeypatch):
    # A100s with no bridge fitted is real, and the name gate would allow it.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert reason is not None and "NODE" in reason


def test_p2p_veto_names_the_iommu_on_bare_metal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: True))
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert "translating IOMMU" in reason
    # Under a hypervisor CUDA supports pass-through P2P, so the IOMMU is not it.
    monkeypatch.setattr(LlamaCppBackend, "_running_virtualized", staticmethod(lambda: True))
    assert "IOMMU" not in LlamaCppBackend._p2p_veto_reason([0, 1])


def test_p2p_exact_mapping_consults_only_the_selected_pair(monkeypatch):
    # PCI_BUS_ID makes the index spaces the same, so a partially linked box can still
    # use its NVLinked pair.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100"] * 3))
    _use_topo(
        monkeypatch,
        "\tGPU0\tGPU1\tGPU2\tCPU Affinity\n"
        "GPU0\t X \tNV18\tNODE\t0-23\n"
        "GPU1\tNV18\t X \tNODE\t0-23\n"
        "GPU2\tNODE\tNODE\t X \t0-23\n",
    )
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None
    assert LlamaCppBackend._p2p_veto_reason([0, 2]) is not None
    assert LlamaCppBackend._p2p_veto_reason([0, 1, 2]) is not None


def test_p2p_checks_the_selected_pair_on_a_partially_linked_box(monkeypatch):
    # Selection and matrix are both nvidia-smi indices, so a linked pair is allowed
    # and an unlinked one refused on the same box, whatever CUDA_DEVICE_ORDER says.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100"] * 3))
    _use_topo(
        monkeypatch,
        "\tGPU0\tGPU1\tGPU2\tCPU Affinity\n"
        "GPU0\t X \tNV18\tNODE\t0-23\n"
        "GPU1\tNV18\t X \tNODE\t0-23\n"
        "GPU2\tNODE\tNODE\t X \t0-23\n",
    )
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None
    assert LlamaCppBackend._p2p_veto_reason([0, 2]) is not None
    # No selection: the whole visible box has to qualify.
    assert LlamaCppBackend._p2p_veto_reason(None) is not None
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


# ---------------------------------------------------------------------------
# End-to-end: the reported bug, and the opt-out that did not work
# ---------------------------------------------------------------------------


def test_apply_env_rtx_6000_ada_gets_fp32_but_not_p2p(monkeypatch):
    """#10613 exactly: 2x RTX 6000 Ada keeps FP32 accum, harmless per the reporter's
    own isolation, and loses the P2P that garbled every model."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    # Launch-queue depth moves no data across the bus and #10613 measured it clean on
    # that host, so it is deliberately NOT gated with P2P.
    assert env == {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1", "CUDA_SCALE_LAUNCH_QUEUES": "4x"}


def test_apply_env_l40s_multi_gpu_gets_fp32_but_not_p2p(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA L40S"] * 4))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1, 2, 3]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_sanitize_falsy_user_p2p_is_removed():
    """ggml tests GGML_CUDA_P2P for presence, so passing a user's "0" through ENABLES
    peer copies; the opt-out has to unset it."""
    for value in ("0", "false", "OFF", "no", "", " 0 "):
        env = {"GGML_CUDA_P2P": value, "OTHER": "kept"}
        assert LlamaCppBackend._sanitize_p2p_env(env) == value
        assert env == {"OTHER": "kept"}, value


def test_sanitize_leaves_a_truthy_user_p2p_alone():
    env = {"GGML_CUDA_P2P": "1"}
    assert LlamaCppBackend._sanitize_p2p_env(env) is None
    assert env == {"GGML_CUDA_P2P": "1"}


def test_opted_out_p2p_is_not_reintroduced_by_the_default(monkeypatch):
    # The call site strips it; the DC block must not put it back. The rest stands.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1], p2p_opted_out = True) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_disable_dc_p2p_drops_peer_flag_but_keeps_fp32(monkeypatch):
    # The surgical opt-out: UNSLOTH_DISABLE_DC_TUNING is all-or-nothing and discards
    # a tuning that is not implicated.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_force_dc_p2p_opts_back_in_over_an_unreadable_topology(monkeypatch):
    # For a real but unparsable fabric, once p2p_integrity_probe.py passes.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_P2P", raising = False)
    monkeypatch.setenv("UNSLOTH_FORCE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert env["GGML_CUDA_P2P"] == "1"


def test_disable_dc_p2p_beats_force_dc_p2p(monkeypatch):
    # Both set: the safe direction wins.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_P2P", "1")
    monkeypatch.setenv("UNSLOTH_FORCE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env


def test_sanitize_applies_to_a_consumer_box_that_never_reaches_the_dc_gate():
    # A 2x RTX 3090 never matches the allowlist, so only the call-site strip helps.
    env = {"GGML_CUDA_P2P": "0"}
    assert LlamaCppBackend._sanitize_p2p_env(env) == "0"
    assert env == {}


def test_apply_env_truthy_user_p2p_still_wins(monkeypatch):
    # An explicit opt-in survives the veto.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env = {"GGML_CUDA_P2P": "1"}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert env["GGML_CUDA_P2P"] == "1"


def test_apply_env_single_nvlink_gpu_still_skips_p2p(monkeypatch):
    # One GPU has no peer.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is True
    assert env == {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"}


def test_apply_env_multi_dc_without_nvidia_smi_withholds_p2p(monkeypatch):
    # Unknown topology must cost the optimisation, not the correctness.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


# ---------------------------------------------------------------------------
# Platform probes
# ---------------------------------------------------------------------------


def _iommu_tree(tmp_path, types_by_group):
    root = tmp_path / "iommu_groups"
    for group, kind in types_by_group.items():
        (root / str(group)).mkdir(parents = True)
        if kind is not None:
            (root / str(group) / "type").write_text(f"{kind}\n")
    return str(root)


def test_iommu_identity_groups_are_not_translating(tmp_path):
    root = _iommu_tree(tmp_path, {0: "identity", 1: "identity"})
    assert _REAL_IOMMU_IS_TRANSLATING(root) is False


def test_iommu_dma_group_is_translating(tmp_path):
    # The reporter's host: 175 groups in DMA-FQ (translating) mode.
    root = _iommu_tree(tmp_path, {0: "identity", 1: "DMA-FQ"})
    assert _REAL_IOMMU_IS_TRANSLATING(root) is True


def test_iommu_absent_or_empty_is_not_translating(tmp_path):
    assert _REAL_IOMMU_IS_TRANSLATING(str(tmp_path / "nope")) is False
    (tmp_path / "empty").mkdir()
    assert _REAL_IOMMU_IS_TRANSLATING(str(tmp_path / "empty")) is False


def test_iommu_unreadable_types_are_unknown(tmp_path):
    # Pre-5.x kernels expose groups with no `type`: unknown, not "safe".
    root = _iommu_tree(tmp_path, {0: None, 1: None})
    assert _REAL_IOMMU_IS_TRANSLATING(root) is None


# ---------------------------------------------------------------------------
# Gaps found reviewing the #10613 fix
# ---------------------------------------------------------------------------


def test_disable_dc_p2p_also_drops_an_inherited_truthy_value(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env = {"GGML_CUDA_P2P": "1"}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    # Presence is truth upstream, so "disabled" has to mean absent.
    assert "GGML_CUDA_P2P" not in env


def test_apply_env_does_not_leave_a_falsy_value_present_when_called_directly(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env = {"GGML_CUDA_P2P": "0"}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    assert "GGML_CUDA_P2P" not in env


def test_shared_llama_server_env_builder_sanitizes_p2p(monkeypatch, tmp_path):
    """The STT sidecar and the embedding probe build from this, so stripping here
    covers every llama-server child, not just chat."""
    monkeypatch.setenv("GGML_CUDA_P2P", "0")
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding = "utf-8")
    env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert "GGML_CUDA_P2P" not in env


def test_stt_sidecar_env_sanitizes_p2p(monkeypatch, tmp_path):
    # Spawns llama-server with -ngl 99, so it is exposed like chat.
    from core.inference import stt_mtmd_sidecar

    monkeypatch.setenv("GGML_CUDA_P2P", "0")
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding = "utf-8")
    assert "GGML_CUDA_P2P" not in stt_mtmd_sidecar._llama_server_child_env(str(binary))


def test_embedding_server_env_sanitizes_p2p(monkeypatch, tmp_path):
    """Bypasses the shared builder, and a corrupt embedding degrades retrieval with
    no garbled text to notice."""
    from core.rag.embed_llama_server import LlamaServerBackend

    monkeypatch.setenv("GGML_CUDA_P2P", "0")
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding = "utf-8")
    server = LlamaServerBackend.__new__(LlamaServerBackend)
    env = server._build_env(str(binary), use_gpu = False)
    assert "GGML_CUDA_P2P" not in env


# ---------------------------------------------------------------------------
# Found by the platform/hardware simulation matrix (temp/sim_10613)
# ---------------------------------------------------------------------------


# 4x A100 bridged over (0,1) and (2,3), PCIe between the islands: the standard
# bridged build, which the whole-matrix fallback used to veto outright.
TOPO_BRIDGED_4X = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\x1b[0m\n"
    "GPU0\t X \tNV12\tSYS\tSYS\t0-23\t0\n"
    "GPU1\tNV12\t X \tSYS\tSYS\t0-23\t0\n"
    "GPU2\tSYS\tSYS\t X \tNV12\t24-47\t1\n"
    "GPU3\tSYS\tSYS\tNV12\t X \t24-47\t1\n"
    "\nLegend:\n\n  X    = Self\n"
)


def test_bridged_pair_keeps_p2p_on_a_partially_bridged_box(monkeypatch):
    """A genuinely NVLinked pair on a partially bridged box keeps P2P: gpu_indices
    are already nvidia-smi indices, so the pair checked is the pair selected."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None
    assert LlamaCppBackend._p2p_veto_reason([2, 3]) is None
    # Across the islands the copy really would cross PCIe.
    assert LlamaCppBackend._p2p_veto_reason([0, 2]) is not None
    assert LlamaCppBackend._p2p_veto_reason([1, 3]) is not None


def test_selection_is_not_remapped_out_of_the_nvidia_smi_index_space(monkeypatch):
    """The selection indexes the topology matrix VERBATIM: both come from nvidia-smi,
    one enumeration. Remapping them as CUDA ordinals would, under FASTEST_FIRST on a
    bridged box, turn a PCIe-crossing selection into an NVLinked-looking one and
    enable the flag this gate exists to withhold."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    for pair in ([0, 1], [2, 3]):
        assert LlamaCppBackend._p2p_veto_reason(pair) is None, pair
    for pair in ([0, 2], [0, 3], [1, 2], [1, 3]):
        assert LlamaCppBackend._p2p_veto_reason(pair) is not None, pair
    # An unset CUDA_DEVICE_ORDER must not change any of the above.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason([0, 2]) is not None


def test_name_gate_veto_still_names_the_iommu(monkeypatch):
    """The reporter's host fails the NAME gate first, so without this the one
    actionable diagnosis never reaches them (#10613)."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: True))
    LlamaCppBackend._IOMMU_CACHE = None
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert "translating IOMMU" in reason


def test_a_raising_probe_never_fails_the_model_load(monkeypatch):
    # Losing the tuning is acceptable; taking the model load down is not.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))

    def boom(*a, **k):
        raise OSError("sysfs exploded")

    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(boom))
    monkeypatch.setattr(LlamaCppBackend, "_nvlink_topology", classmethod(boom))
    LlamaCppBackend._IOMMU_CACHE = None
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_iommu_scan_is_cached_across_loads(monkeypatch):
    # A boot-time property, and 175 groups on the reporter's host.
    calls = []

    def counted(*a, **k):
        calls.append(1)
        return True

    monkeypatch.setattr(LlamaCppBackend, "_iommu_is_translating", staticmethod(counted))
    LlamaCppBackend._IOMMU_CACHE = None
    for _ in range(5):
        LlamaCppBackend._iommu_is_translating_cached()
    assert len(calls) == 1


def _capture_warnings(monkeypatch):
    """Collect logger.warning calls: caplog does not see structlog here, so asserting
    on it would pass vacuously."""
    from core.inference import llama_cpp as _mod

    seen: list[str] = []
    real = _mod.logger.warning
    monkeypatch.setattr(
        _mod.logger,
        "warning",
        lambda msg, *a, **k: (seen.append(str(msg)), real(msg, *a, **k))[0],
    )
    return seen


def test_no_pcie_warning_when_the_user_opts_in_on_a_verified_fabric(monkeypatch):
    """A deliberate GGML_CUDA_P2P=1 on a confirmed NV# pair is the benchmarked
    configuration, so warning there would push users off a working optimisation."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None
    seen = _capture_warnings(monkeypatch)
    env = {"GGML_CUDA_P2P": "1"}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    assert env["GGML_CUDA_P2P"] == "1"
    assert not [m for m in seen if "peer copies" in m], seen


def test_pcie_warning_still_fires_on_an_unverified_fabric(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    seen = _capture_warnings(monkeypatch)
    env = {"GGML_CUDA_P2P": "1"}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    assert [m for m in seen if "peer copies stay" in m], seen


def test_masked_visible_devices_filter_needs_a_shared_index_space(monkeypatch):
    """nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the matrix covers cards the child
    never touches and filtering it to the mask recovers a clean NVLinked pair. Sound
    only under PCI_BUS_ID: numeric mask entries are CUDA ordinals, and under
    FASTEST_FIRST "0,1" can mean physical 0,2."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 2))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert LlamaCppBackend._p2p_veto_reason(None) is None
    # A mask spanning the islands is still correctly refused.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason(None) is not None
    # Without a shared index space the mask is not trusted: the whole box has to
    # qualify, which the bridged fixture does not.
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason(None) is not None


def test_explicit_p2p_opt_out_does_not_warn_about_corruption(monkeypatch):
    """UNSLOTH_DISABLE_DC_P2P=1 returns before the topology is inspected, so calling
    the fabric unconfirmed would warn about corruption on a healthy box."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    seen = _capture_warnings(monkeypatch)
    env: dict = {}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    assert "GGML_CUDA_P2P" not in env
    assert not [m for m in seen if "without a confirmed NVLink" in m], seen


def test_auto_fit_selection_without_a_pinned_order_refuses_p2p(monkeypatch):
    """The gate verifies nvidia-smi physical ids, and the child resolves the same
    cards only under a pinned CUDA_DEVICE_ORDER. On a box that is NOT uniformly
    NVLinked, [0,1] here can be [0,2] there, so refuse rather than confirm NVLink for
    a pair that is not the one about to run."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    # Auto-fit (the launch will not pin the order): withheld.
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None
    # Explicit user pick, so the launch pins PCI_BUS_ID: allowed.
    assert LlamaCppBackend._p2p_veto_reason([0, 1], launch_order_pinned = True) is None
    # Already pinned in this process: allowed either way.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


def test_auto_fit_launch_does_not_rewrite_an_inherited_device_order(monkeypatch):
    """Forcing PCI_BUS_ID for an auto-fit launch would re-read an inherited numeric
    mask: a scheduler's CUDA_VISIBLE_DEVICES=0,1 meaning physical 2,0 would become
    physical 0,1, running on a GPU that was hidden on purpose."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    branch = src[src.index("elif gpu_indices is not None and not is_vulkan_backend") :]
    branch = branch[: branch.index("_launch_pinned_ids")]
    pin = 'env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"'
    assert pin in branch
    # The pin must stay behind a guard, never unconditional.
    assert "if _p2p_launch_order_pinned:" in branch[: branch.index(pin)]
    # The guard must require an ABSENT inherited mask: an inherited numeric mask is
    # the thing that must not be re-read.
    start = src.index("_p2p_launch_order_pinned = ")
    guard = src[start : src.index("\n\n", start)]
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") is None' in guard, guard


def test_datacenter_box_warns_once_not_twice(monkeypatch):
    """The call-site warning is for hosts that never reach _apply_datacenter_env; a
    datacenter box does, and its veto branch warns about the same variable."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    assert LlamaCppBackend._is_datacenter_gpu([0, 1]) is True
    seen = _capture_warnings(monkeypatch)
    env = {"GGML_CUDA_P2P": "1"}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1])
    assert len([m for m in seen if "peer copies stay" in m]) == 1, seen


def test_uniform_nvlink_box_keeps_p2p_without_a_pinned_order(monkeypatch):
    """The NVSwitch case PR #6098 benchmarked. When every pair on the box is NV# the
    index mapping is irrelevant, since whichever cards the child resolves are linked,
    and requiring a pinned order would strip P2P from every DGX/HGX auto-fit load."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 8))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    for sel in ([0, 1], [0, 7], list(range(8)), None):
        assert LlamaCppBackend._p2p_veto_reason(sel) is None, sel


def test_partially_bridged_box_still_needs_a_pinned_order(monkeypatch):
    # Not uniform, so a permutation really can change which pair runs.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None


def test_unmasked_auto_fit_keeps_p2p(monkeypatch):
    """An Auto load that fits on a subset sets gpu_indices with gpu_ids empty. With no
    inherited CUDA_VISIBLE_DEVICES there is nothing to reinterpret, so the launch pins
    PCI_BUS_ID and the flag is kept, sparing ordinary Auto loads on NVLink boxes."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 4))
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    # Not uniform, so this only passes because the launch can pin the order.
    assert LlamaCppBackend._p2p_veto_reason([0, 1], launch_order_pinned = True) is None


def test_p2p_opt_out_skips_the_topology_probe(monkeypatch):
    """A user who turned P2P off should not pay the nvidia-smi probe, up to a 10s
    timeout, to decide something they already decided."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    probed = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_nvlink_topology",
        classmethod(lambda cls, *a, **k: (probed.append(1), None)[1]),
    )
    env: dict = {}
    LlamaCppBackend._apply_datacenter_env(env, [0, 1], p2p_opted_out = True)
    assert "GGML_CUDA_P2P" not in env
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"
    assert probed == [], "topology probed despite an explicit opt-out"


def test_row_truncated_matrix_is_rejected(monkeypatch):
    """Row-truncated output still exits 0 and parses cleanly, and the rows that
    arrived can be uniformly NV#, which would read as "the whole box is NVLinked" and
    enable P2P on cards whose links were never seen. Partial must mean none."""
    header, *rows = TOPO_NVLINK_8X.splitlines()
    truncated = "\n".join([header] + rows[:4]) + "\n"
    _use_topo(monkeypatch, truncated)
    assert LlamaCppBackend._nvlink_topology() is None

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 8))
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    # The pairs that DID parse are all NV18, so without the completeness check the
    # uniform escape would allow this.
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None


def test_complete_matrix_is_still_accepted(monkeypatch):
    # The completeness check must not reject the real thing.
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    assert len(LlamaCppBackend._nvlink_topology()) == 8 * 7
    _use_topo(monkeypatch, TOPO_BRIDGED_4X)
    assert len(LlamaCppBackend._nvlink_topology()) == 4 * 3
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    assert len(LlamaCppBackend._nvlink_topology()) == 2 * 1


def test_pin_requires_ids_to_be_pci_indices(monkeypatch):
    """When the nvidia-smi query fails, _get_gpu_memory falls back to torch CUDA
    ordinals, and pinning PCI order re-emits those numbers as a different set of
    cards, so Auto could run on GPUs other than the ones it measured."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    start = src.index("_p2p_launch_order_pinned = ")
    guard = src[start : src.index("\n\n", start)]
    assert "_GPU_IDS_ARE_PCI_INDICES is True" in guard, guard
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") is None' in guard, guard


def test_gpu_id_provenance_is_recorded(monkeypatch):
    # The nvidia-smi branch yields PCI indices; the torch fallback yields ordinals.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode = 0,
            stdout = "0, 1000, 2000\n1, 1000, 2000\n",
            stderr = "",
        ),
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
    )
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = None
    assert LlamaCppBackend._get_gpu_memory("llama-server")
    assert LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is True

    # nvidia-smi absent -> torch fallback -> ordinals.
    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = None
    LlamaCppBackend._get_gpu_memory("llama-server")
    assert LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES is not True


# ---------------------------------------------------------------------------
# NVML fast path: the same matrix from the driver's own NVML instead of
# `nvidia-smi topo -m`. These drive a fake libnvidia-ml so the failure modes are
# reachable without hardware.
# ---------------------------------------------------------------------------

_NVML_OK = 0
_NVML_ERROR = 999


class _FakeNvml:
    """Stand-in for libnvidia-ml.so.1 with the real calling convention: rc out,
    values written through byref pointers."""

    def __init__(
        self,
        count = 2,
        linked_pairs = None,
        active_links = None,
        status_rc = _NVML_OK,
        status_value = None,
        handle_rc = _NVML_OK,
        init_rc = _NVML_OK,
        count_rc = _NVML_OK,
        missing = (),
        fail_status_after = None,
    ):
        self.count = count
        # None = every pair NVLinked.
        self.linked_pairs = linked_pairs
        # None = every device has links.
        self.active_links = active_links
        self.status_rc = status_rc
        self.status_value = status_value
        self.handle_rc = handle_rc
        self.init_rc = init_rc
        self.count_rc = count_rc
        self.missing = set(missing)
        self.fail_status_after = fail_status_after
        self.status_calls = 0
        self.shutdown_calls = 0

    def _fn(self, name, impl):
        if name in self.missing:
            raise AttributeError(name)
        return impl

    # ctypes attribute lookups happen once, up front, in the probe.
    @property
    def nvmlInit_v2(self):
        return self._fn("nvmlInit_v2", lambda: self.init_rc)

    @property
    def nvmlShutdown(self):
        def _shutdown():
            self.shutdown_calls += 1
            return _NVML_OK

        return _shutdown

    @property
    def nvmlDeviceGetCount_v2(self):
        def _count(ref):
            ref._obj.value = self.count
            return self.count_rc

        return self._fn("nvmlDeviceGetCount_v2", _count)

    @property
    def nvmlDeviceGetHandleByIndex_v2(self):
        def _handle(index, ref):
            ref._obj.value = 1000 + int(getattr(index, "value", index))
            return self.handle_rc

        return self._fn("nvmlDeviceGetHandleByIndex_v2", _handle)

    @property
    def nvmlDeviceGetNvLinkState(self):
        def _state(handle, link, ref):
            device = int(handle.value) - 1000
            links = 4 if self.active_links is None else self.active_links[device]
            if int(getattr(link, "value", link)) >= links:
                return _NVML_ERROR
            ref._obj.value = 1  # NVML_FEATURE_ENABLED
            return _NVML_OK

        return self._fn("nvmlDeviceGetNvLinkState", _state)

    @property
    def nvmlDeviceGetP2PStatus(self):
        def _status(a, b, index, ref):
            self.status_calls += 1
            if self.fail_status_after is not None and self.status_calls > self.fail_status_after:
                return _NVML_ERROR
            if self.status_rc != _NVML_OK:
                return self.status_rc
            pair = (int(a.value) - 1000, int(b.value) - 1000)
            linked = self.linked_pairs is None or pair in self.linked_pairs
            if self.status_value is not None:
                ref._obj.value = self.status_value
            else:
                ref._obj.value = 0 if linked else 5  # OK / NOT_SUPPORTED
            return _NVML_OK

        return self._fn("nvmlDeviceGetP2PStatus", _status)


def _use_nvml(monkeypatch, fake):
    monkeypatch.setattr(LlamaCppBackend, "_nvml_library", staticmethod(lambda: fake))
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    return fake


def test_nvml_all_pairs_nvlinked_is_a_complete_matrix(monkeypatch):
    _use_nvml(monkeypatch, _FakeNvml(count = 4))
    matrix = LlamaCppBackend._probe_nvml_nvlink_topology()
    assert set(matrix) == {(a, b) for a in range(4) for b in range(4) if a != b}
    assert all(LlamaCppBackend._label_is_nvlink(v) for v in matrix.values())


def test_nvml_is_preferred_over_the_shell_out(monkeypatch):
    """The whole point: on a host where both answer, nvidia-smi is never spawned."""
    _use_nvml(monkeypatch, _FakeNvml(count = 2))
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: calls.append(a)
        or types.SimpleNamespace(returncode = 0, stdout = TOPO_PCIE_2X, stderr = ""),
    )
    matrix = LlamaCppBackend._nvlink_topology()
    assert LlamaCppBackend._matrix_is_nvml(matrix)
    assert calls == []


def test_nvml_absent_falls_back_to_topo(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_nvml_library", staticmethod(lambda: None))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    matrix = LlamaCppBackend._nvlink_topology()
    assert matrix and not LlamaCppBackend._matrix_is_nvml(matrix)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"init_rc": _NVML_ERROR},
        {"count_rc": _NVML_ERROR},
        {"handle_rc": _NVML_ERROR},
        {"status_rc": _NVML_ERROR},
        {"missing": ("nvmlDeviceGetP2PStatus",)},
        {"missing": ("nvmlDeviceGetNvLinkState",)},
        {"count": 1},
    ],
)
def test_nvml_unknown_falls_through_to_topo(monkeypatch, kwargs):
    """Every way NVML can fail to answer must reach the shell-out, not a verdict."""
    _use_nvml(monkeypatch, _FakeNvml(**kwargs))
    assert LlamaCppBackend._probe_nvml_nvlink_topology() is None
    _use_topo_after_nvml(monkeypatch, _FakeNvml(**kwargs), TOPO_NVLINK_8X)
    matrix = LlamaCppBackend._nvlink_topology()
    assert matrix and not LlamaCppBackend._matrix_is_nvml(matrix)


def _use_topo_after_nvml(monkeypatch, fake, text):
    monkeypatch.setattr(LlamaCppBackend, "_nvml_library", staticmethod(lambda: fake))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = text, stderr = ""),
    )
    LlamaCppBackend._NVLINK_TOPO_CACHE = None


def test_nvml_partial_walk_is_not_a_partial_matrix(monkeypatch):
    """A failure mid-walk must discard the successes, not approve the pairs that
    happened to answer first."""
    _use_nvml(monkeypatch, _FakeNvml(count = 4, fail_status_after = 3))
    assert LlamaCppBackend._probe_nvml_nvlink_topology() is None


def test_nvml_positive_without_active_links_is_a_contradiction(monkeypatch):
    """Guard for hardware not available here: an OK on a box with no live NVLinks is
    unknown, not a green light."""
    _use_nvml(monkeypatch, _FakeNvml(count = 2, active_links = [0, 0]))
    assert LlamaCppBackend._probe_nvml_nvlink_topology() is None


def test_nvml_unlinked_pair_vetoes_p2p(monkeypatch):
    """Two islands, 0-1 and 2-3. A cross-island selection must not get P2P."""
    linked = {(0, 1), (1, 0), (2, 3), (3, 2)}
    _use_nvml(monkeypatch, _FakeNvml(count = 4, linked_pairs = linked))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
    assert LlamaCppBackend._p2p_veto_reason([0, 2]) is not None
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


def test_nvml_matrix_needs_nvidia_smi_provenance(monkeypatch):
    """`topo -m` answering proves nvidia-smi enumerated the selection; NVML does not,
    so torch-ordinal ids must not be matched against an NVML matrix."""
    _use_nvml(monkeypatch, _FakeNvml(count = 4))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = False
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = True
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


def test_nvml_shuts_down_what_it_started(monkeypatch):
    fake = _use_nvml(monkeypatch, _FakeNvml(count = 2))
    LlamaCppBackend._probe_nvml_nvlink_topology()
    assert fake.shutdown_calls == 1
    # A failed init owns no session, so it must not shut one down.
    failed = _use_nvml(monkeypatch, _FakeNvml(count = 2, init_rc = _NVML_ERROR))
    LlamaCppBackend._probe_nvml_nvlink_topology()
    assert failed.shutdown_calls == 0


def test_nvml_single_gpu_gives_no_verdict(monkeypatch):
    """One device cannot make a pair, and an empty matrix must never read as
    'everything is NVLinked' through a vacuous all()."""
    _use_nvml(monkeypatch, _FakeNvml(count = 1))
    assert LlamaCppBackend._probe_nvml_nvlink_topology() is None


def test_crosscheck_prefers_topo_on_disagreement(monkeypatch):
    """When both answer and differ, the slow one wins and it is logged."""
    monkeypatch.setenv("UNSLOTH_P2P_TOPO_CROSSCHECK", "1")
    _use_topo_after_nvml(monkeypatch, _FakeNvml(count = 2), TOPO_PCIE_2X)
    records = _capture_warnings(monkeypatch)
    matrix = LlamaCppBackend._probe_interconnect_matrix()
    assert not LlamaCppBackend._matrix_is_nvml(matrix)
    assert any("cross-check disagreement" in r for r in records)


def test_nvml_unknown_status_is_not_a_denial(monkeypatch):
    """UNKNOWN (6), or a status from a newer driver, is not evidence of no NVLink:
    recording it as NO-NVLINK would deny topo -m the chance to confirm a real
    fabric."""
    for status in (6, 42):
        _use_nvml(monkeypatch, _FakeNvml(count = 2, status_value = status))
        assert LlamaCppBackend._probe_nvml_nvlink_topology() is None
    # A documented "not supported, and here is why" IS a denial, and stays one.
    for status in (1, 2, 3, 4, 5):
        _use_nvml(monkeypatch, _FakeNvml(count = 2, status_value = status))
        matrix = LlamaCppBackend._probe_nvml_nvlink_topology()
        assert matrix and not any(LlamaCppBackend._label_is_nvlink(v) for v in matrix.values())


def test_a_failed_prime_does_not_poison_the_cache(monkeypatch):
    """The prime runs early, possibly mid driver init. A miss cached there would keep
    P2P off for the whole process even once the topology is readable."""
    monkeypatch.setattr(
        LlamaCppBackend, "_probe_interconnect_matrix", classmethod(lambda cls: None)
    )
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._nvlink_topology(cache_failure = False) is None
    assert LlamaCppBackend._NVLINK_TOPO_CACHE is None, "a failed prime cached its miss"
    # The load path still caches its own miss, as it did before this change.
    assert LlamaCppBackend._nvlink_topology() is None
    assert LlamaCppBackend._NVLINK_TOPO_CACHE == (None,)


def test_a_successful_prime_is_still_cached(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_probe_interconnect_matrix",
        classmethod(lambda cls: {(0, 1): "NVLINK", (1, 0): "NVLINK"}),
    )
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._nvlink_topology(cache_failure = False) is not None
    assert LlamaCppBackend._NVLINK_TOPO_CACHE is not None


def test_windows_nvml_candidates_include_the_nvsmi_directory(monkeypatch):
    """A driver install can leave nvml.dll in NVSMI rather than a DLL search path,
    as it does nvidia-smi.exe."""
    tried = []
    # From the module-level capture, since the autouse fixture already replaced the
    # class attribute with the absent stub.
    monkeypatch.setattr(LlamaCppBackend, "_nvml_library", staticmethod(_REAL_NVML_LIBRARY))
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("ProgramFiles", r"C:\Program Files")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")

    def _fake_cdll(name):
        tried.append(name)
        raise OSError("not here")

    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    assert LlamaCppBackend._nvml_library() is None
    assert any("NVSMI" in t for t in tried), tried
    assert tried[0] == "nvml.dll", "the search path should still be tried first"


def test_an_explicit_pick_keeps_its_pci_provenance(monkeypatch):
    """The UI's picker hands back PCI-ordered ids. A torch fallback in the unrelated
    memory query sets _GPU_IDS_ARE_PCI_INDICES False process-wide, and that must not
    cost an explicitly selected NVLinked pair its P2P."""
    _use_nvml(monkeypatch, _FakeNvml(count = 4))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    LlamaCppBackend._GPU_IDS_ARE_PCI_INDICES = False

    # Caller says nothing: fall back to the process-wide flag, which vetoes.
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None

    # Caller knows these ids came from the PCI-ordered picker: no veto.
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason([0, 1], True, ids_are_pci_indices = True) is None

    # And an explicit False from the caller still vetoes.
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    assert LlamaCppBackend._p2p_veto_reason([0, 1], True, ids_are_pci_indices = False) is not None


def test_a_stalled_nvml_call_cannot_hang_the_load(monkeypatch):
    """ctypes has no timeout and the shell-out it replaced had one, so a wedged
    driver must cost a fallback rather than the load."""
    import threading as _threading

    release = _threading.Event()

    def _hang(cls):
        release.wait(30)
        return {(0, 1): "NVLINK", (1, 0): "NVLINK"}

    monkeypatch.setattr(LlamaCppBackend, "_probe_nvml_nvlink_topology_inner", classmethod(_hang))
    monkeypatch.setattr(LlamaCppBackend, "_NVML_PROBE_TIMEOUT_SECONDS", 0.2)
    try:
        assert LlamaCppBackend._probe_nvml_nvlink_topology() is None
        # And the caller still reaches the shell-out.
        _use_topo(monkeypatch, TOPO_NVLINK_8X)
        matrix = LlamaCppBackend._probe_interconnect_matrix()
        assert matrix and not LlamaCppBackend._matrix_is_nvml(matrix)
    finally:
        release.set()


def test_a_racing_failure_does_not_erase_a_published_matrix(monkeypatch):
    """The prime publishes a good matrix; a load whose own probe fails transiently
    must not overwrite it with a miss and disable P2P for every later load."""
    good = {(0, 1): "NVLINK", (1, 0): "NVLINK"}
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    monkeypatch.setattr(
        LlamaCppBackend, "_probe_interconnect_matrix", classmethod(lambda cls: good)
    )
    assert LlamaCppBackend._nvlink_topology() == good

    # Now a pass that probes and fails while the good matrix is already published.
    monkeypatch.setattr(
        LlamaCppBackend, "_probe_interconnect_matrix", classmethod(lambda cls: None)
    )
    assert LlamaCppBackend._nvlink_topology(refresh = True) == good
    assert LlamaCppBackend._NVLINK_TOPO_CACHE == (good,)


def test_the_prime_skips_work_the_opt_outs_make_useless(monkeypatch):
    """UNSLOTH_DISABLE_DC_TUNING=1, UNSLOTH_DISABLE_DC_P2P=1 or a falsy GGML_CUDA_P2P
    all mean the answer can never be used, so the prime must not pay for it."""
    from utils import torch_warmup

    probed = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_probe_interconnect_matrix",
        classmethod(lambda cls: (probed.append(1), {(0, 1): "NVLINK", (1, 0): "NVLINK"})[1]),
    )
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 8))

    for var, value in (
        ("UNSLOTH_DISABLE_DC_TUNING", "1"),
        ("UNSLOTH_DISABLE_DC_P2P", "1"),
        ("GGML_CUDA_P2P", "0"),
    ):
        monkeypatch.setenv(var, value)
        LlamaCppBackend._NVLINK_TOPO_CACHE = None
        probed.clear()
        torch_warmup._prime_nvlink_topology().join(10)
        assert probed == [], f"{var}={value} still primed"
        monkeypatch.delenv(var, raising = False)

    # Without an opt-out it still primes.
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    probed.clear()
    torch_warmup._prime_nvlink_topology().join(10)
    assert probed == [1]


def test_the_prime_never_blocks_the_warm_sequence(monkeypatch):
    """The probe can spend its NVML bound plus the shell-out timeout. A warm stage
    that waits that long delays every stage behind it, and on a loaded CI worker that
    is enough to tip unrelated timing-sensitive tests over."""
    from utils import torch_warmup
    import threading as _threading
    import time as _time

    release = _threading.Event()

    def _slow(cls, **kw):
        release.wait(30)
        return {(0, 1): "NVLINK", (1, 0): "NVLINK"}

    monkeypatch.setattr(LlamaCppBackend, "_nvlink_topology", classmethod(_slow))
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 8))
    try:
        started = _time.perf_counter()
        worker = torch_warmup._prime_nvlink_topology()
        elapsed = _time.perf_counter() - started
        assert elapsed < 1.0, f"the prime blocked its caller for {elapsed:.2f}s"
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(10)
