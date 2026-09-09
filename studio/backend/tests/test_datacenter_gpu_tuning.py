# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data-center llama.cpp env tuning: FP32 accum (+ P2P / launch queues for
multi-GPU) must apply only to datacenter NVIDIA parts, never consumer GeForce,
AMD/ROCm, CPU or macOS. User values win; UNSLOTH_DISABLE_DC_TUNING=1 disables.

P2P carries a second, stricter gate (#10613): a datacenter NAME is not evidence
of an NVLink fabric, and on a non-NVLink multi-GPU box the peer copy is silently
discarded while still reporting success, so every model emits garbage. The flags
now need a confirmed NV# link across the selection and fail CLOSED on unknowns.
"""

from __future__ import annotations

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
    """torch stub: version.hip, cuda.is_available/device_count, get_device_properties(i).name."""
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: cuda_ok,
        device_count = lambda: len(names),
        get_device_properties = lambda i: types.SimpleNamespace(name = names[i]),
    )
    return t


# Real `nvidia-smi topo -m` from an 8x B200 NVLink host, verbatim: nvidia-smi
# underlines the header with ANSI escapes, the table carries NIC rows/columns and
# trailing affinity columns, and a Legend section follows. A toy two-line string
# would not exercise any of that.
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

# The reporter's host (#10613): 2x RTX 6000 Ada, NODE (PCIe through a host
# bridge), no NVLink anywhere. The driver still advertises P2P as available.
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
    """Detection reads CUDA_VISIBLE_DEVICES, so clear it by default (run unmasked,
    physical id == ordinal) regardless of host; masked tests set it explicitly."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)


# Captured before the autouse fixture stubs the class attribute, so the platform
# probes can still be exercised for real against a fixture tree.
_REAL_IOMMU_IS_TRANSLATING = LlamaCppBackend.__dict__["_iommu_is_translating"].__func__


def _no_nvidia_smi(*a, **k):
    raise FileNotFoundError("nvidia-smi")


@pytest.fixture(autouse = True)
def _isolate_host_topology(monkeypatch):
    """Keep the P2P gate off the real host. The parser itself always runs (only
    what nvidia-smi "returns" is stubbed), the process-lifetime cache is dropped
    either side, and the platform probes are pinned so a CI box that happens to
    have GPUs or a translating IOMMU cannot colour the results."""
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    monkeypatch.setattr(
        LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: False)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_running_virtualized", staticmethod(lambda: False)
    )
    # Both explicit overrides off by default, and the once-per-process warning
    # latch reset, so neither the host's environment nor test ordering leaks in.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_P2P", raising = False)
    monkeypatch.delenv("UNSLOTH_FORCE_DC_P2P", raising = False)
    LlamaCppBackend._warned_no_nvlink = False
    yield
    LlamaCppBackend._NVLINK_TOPO_CACHE = None
    LlamaCppBackend._warned_no_nvlink = False


def _use_topo(monkeypatch, text, returncode = 0):
    """Feed the real parser canned `nvidia-smi topo -m` output."""
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(
            returncode = returncode, stdout = text, stderr = ""
        ),
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
        # Workstation/laptop: short markers must not match as substrings
        # ("a100" in "A1000", "a30" in "A3000").
        (["NVIDIA RTX A1000 Laptop GPU"], False),
        (["NVIDIA RTX A1000 6GB Laptop GPU"], False),
        (["NVIDIA RTX A3000 Laptop GPU"], False),
        # Homogeneous multi-DC: all must match.
        (["NVIDIA B200", "NVIDIA B200"], True),
        (["NVIDIA H100 80GB HBM3", "NVIDIA H100 80GB HBM3"], True),
        # Mixed DC + consumer: non-DC, so tuning never lands on the GeForce.
        (["NVIDIA B200", "NVIDIA GeForce RTX 4090"], False),
        (["NVIDIA GeForce RTX 4090", "NVIDIA B200"], False),
    ],
)
def test_is_datacenter_gpu(monkeypatch, names, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(names))
    assert LlamaCppBackend._is_datacenter_gpu() is expected


def test_is_datacenter_gpu_respects_selection(monkeypatch):
    # A mixed box where only the DC GPU is selected -> True; only consumer -> False.
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
    # Out-of-range / negative indices are skipped; the one valid DC GPU still wins.
    assert LlamaCppBackend._is_datacenter_gpu([0, 5, -1]) is True
    # Only invalid indices -> nothing seen -> False (fail closed for the flag).
    assert LlamaCppBackend._is_datacenter_gpu([5, 9]) is False


def test_is_datacenter_gpu_masked_host_physical_ids(monkeypatch):
    # Mask 4,5,6,7 -> ordinals 0..3 == physical 4..7. PHYSICAL selection [4,5]
    # must resolve, not index out of range (the pre-fix bug: 4 >= device_count).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    assert LlamaCppBackend._is_datacenter_gpu([4, 5]) is True
    assert LlamaCppBackend._is_datacenter_gpu([4, 5, 6, 7]) is True
    assert LlamaCppBackend._is_datacenter_gpu(None) is True
    assert LlamaCppBackend._is_datacenter_gpu([0, 1]) is False  # not visible -> skip


def test_is_datacenter_gpu_masked_host_reordered(monkeypatch):
    # Reordered mask preserves order: ordinal 0 -> physical 7, 1 -> 4, ...
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,4,5,6")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100 80GB HBM3"] * 4))
    assert LlamaCppBackend._is_datacenter_gpu([7, 4]) is True


def test_is_datacenter_gpu_masked_host_mixed_class(monkeypatch):
    # Mask 4,5: physical 4 = GeForce, physical 5 = B200. Detection must follow the
    # selected physical GPU, not a same-numbered ordinal.
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
    # Unparsable (UUID) mask falls back to physical id == ordinal (mirrors
    # _get_gpu_free_memory), so ordinal lookup still classifies the device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef12")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"]))
    assert LlamaCppBackend._is_datacenter_gpu([0]) is True


def test_is_datacenter_gpu_rocm_is_false(monkeypatch):
    # ROCm reuses torch.cuda.*; an MI300X must not qualify.
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
    # None -> visible device count.
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
    # None on a 2x DC box -> multi-GPU flags applied.
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
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "0",  # user explicitly disabled
        "CUDA_SCALE_LAUNCH_QUEUES": "8x",  # user override
    }
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    # setdefault must not clobber user values; the unset one still defaults.
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
    # End-to-end masked host (mask 4,5,6,7, physical selection [4,5]): pre-fix
    # applied no tuning; now all three multi-GPU flags must be set.
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
    # 8 GPUs -> 8*7 ordered pairs. NIC rows/columns, the affinity columns, the
    # ANSI-underlined header and the legend must all be ignored.
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
    assert len(calls) == 1  # one shell-out per process, not per launch
    LlamaCppBackend._nvlink_topology(refresh = True)
    assert len(calls) == 2


def test_topo_unavailable_paths_are_none(monkeypatch):
    # Missing binary (the autouse default), non-zero exit, and a table with no
    # GPU rows all mean "unknown", never "assume NVLink".
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
    # The reported host: 2x RTX 6000 Ada, a datacenter NAME with no NVLink.
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
    # NVLink-capable parts but no nvidia-smi: fail CLOSED, the whole point of the
    # fix. The old code set the flag on the name alone.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert reason is not None and "interconnect matrix" in reason


def test_p2p_vetoed_on_pcie_pair_between_nvlink_parts(monkeypatch):
    # A100s wired over PCIe (no NVLink bridge fitted) is a real configuration and
    # the name gate alone would wave it through.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert reason is not None and "NODE" in reason


def test_p2p_veto_names_the_iommu_on_bare_metal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA A100-SXM4-80GB"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    monkeypatch.setattr(
        LlamaCppBackend, "_iommu_is_translating", staticmethod(lambda *a: True)
    )
    reason = LlamaCppBackend._p2p_veto_reason([0, 1])
    assert "translating IOMMU" in reason
    # Under a hypervisor CUDA supports pass-through P2P, so the IOMMU is not the
    # diagnosis and must not be blamed.
    monkeypatch.setattr(
        LlamaCppBackend, "_running_virtualized", staticmethod(lambda: True)
    )
    assert "IOMMU" not in LlamaCppBackend._p2p_veto_reason([0, 1])


def test_p2p_exact_mapping_consults_only_the_selected_pair(monkeypatch):
    # CUDA_DEVICE_ORDER=PCI_BUS_ID makes physical ids and nvidia-smi indices the
    # same space, so a partially linked box can still use its NVLinked pair.
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


def test_p2p_inexact_mapping_demands_a_uniform_matrix(monkeypatch):
    # Without PCI_BUS_ID ordering, "GPU 0" means different cards to CUDA and to
    # nvidia-smi. A uniformly NVLinked box is safe under any permutation; a
    # partially linked one is not, so it is refused even for its linked pair.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA H100"] * 3))
    _use_topo(
        monkeypatch,
        "\tGPU0\tGPU1\tGPU2\tCPU Affinity\n"
        "GPU0\t X \tNV18\tNODE\t0-23\n"
        "GPU1\tNV18\t X \tNODE\t0-23\n"
        "GPU2\tNODE\tNODE\t X \t0-23\n",
    )
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is not None
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    assert LlamaCppBackend._p2p_veto_reason([0, 1]) is None


# ---------------------------------------------------------------------------
# End-to-end: the reported bug, and the opt-out that did not work
# ---------------------------------------------------------------------------


def test_apply_env_rtx_6000_ada_gets_fp32_but_not_p2p(monkeypatch):
    """#10613 exactly: 2x RTX 6000 Ada kept FP32 accum (which the reporter's own
    isolation shows is harmless) and lost the P2P pair that garbled every model."""
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    # The launch-queue depth moves no data across the bus and #10613 measured it
    # clean on the affected host, so it is deliberately NOT gated with P2P.
    assert env == {
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1",
        "CUDA_SCALE_LAUNCH_QUEUES": "4x",
    }


def test_apply_env_l40s_multi_gpu_gets_fp32_but_not_p2p(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA L40S"] * 4))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1, 2, 3]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_sanitize_falsy_user_p2p_is_removed():
    """ggml tests GGML_CUDA_P2P for presence, so passing a user's "0" through
    ENABLES peer copies. The documented opt-out has to unset it instead."""
    for value in ("0", "false", "OFF", "no", "", " 0 "):
        env = {"GGML_CUDA_P2P": value, "OTHER": "kept"}
        assert LlamaCppBackend._sanitize_p2p_env(env) == value
        assert env == {"OTHER": "kept"}, value


def test_sanitize_leaves_a_truthy_user_p2p_alone():
    env = {"GGML_CUDA_P2P": "1"}
    assert LlamaCppBackend._sanitize_p2p_env(env) is None
    assert env == {"GGML_CUDA_P2P": "1"}


def test_opted_out_p2p_is_not_reintroduced_by_the_default(monkeypatch):
    # The call site strips the falsy value; the DC block must not put it back on
    # an NVLink box that would otherwise qualify. The rest of the tuning stands.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(
        env, [0, 1], p2p_opted_out = True
    ) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["CUDA_SCALE_LAUNCH_QUEUES"] == "4x"
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_disable_dc_p2p_drops_peer_flag_but_keeps_fp32(monkeypatch):
    # UNSLOTH_DISABLE_DC_TUNING is all-or-nothing and throws away a tuning that
    # is not implicated; this is the surgical opt-out the #10613 reporter wanted.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setenv("UNSLOTH_DISABLE_DC_P2P", "1")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 2))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert "GGML_CUDA_P2P" not in env
    assert env["GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F"] == "1"


def test_force_dc_p2p_opts_back_in_over_an_unreadable_topology(monkeypatch):
    # For the host whose fabric is real but whose topology we cannot parse, after
    # they have confirmed it with scripts/p2p_integrity_probe.py.
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
    # A 2x RTX 3090 user who set GGML_CUDA_P2P=0 by hand never matches the
    # datacenter allowlist, so only the call-site sanitizer protects them.
    env = {"GGML_CUDA_P2P": "0"}
    assert LlamaCppBackend._sanitize_p2p_env(env) == "0"
    assert env == {}


def test_apply_env_truthy_user_p2p_still_wins(monkeypatch):
    # An explicit opt-in survives the veto: the user asked for it by name.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA RTX 6000 Ada Generation"] * 2))
    _use_topo(monkeypatch, TOPO_PCIE_2X)
    env = {"GGML_CUDA_P2P": "1"}
    assert LlamaCppBackend._apply_datacenter_env(env, [0, 1]) is True
    assert env["GGML_CUDA_P2P"] == "1"


def test_apply_env_single_nvlink_gpu_still_skips_p2p(monkeypatch):
    # One GPU has no peer, so the topology is never consulted.
    monkeypatch.delenv("UNSLOTH_DISABLE_DC_TUNING", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(["NVIDIA B200"] * 4))
    _use_topo(monkeypatch, TOPO_NVLINK_8X)
    env: dict = {}
    assert LlamaCppBackend._apply_datacenter_env(env, [0]) is True
    assert env == {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F": "1"}


def test_apply_env_multi_dc_without_nvidia_smi_withholds_p2p(monkeypatch):
    # The regression that matters: unknown topology must cost the optimisation,
    # not the correctness of the output.
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
    # The reporter's host: all 175 groups in DMA-FQ (translating) mode.
    root = _iommu_tree(tmp_path, {0: "identity", 1: "DMA-FQ"})
    assert _REAL_IOMMU_IS_TRANSLATING(root) is True


def test_iommu_absent_or_empty_is_not_translating(tmp_path):
    assert _REAL_IOMMU_IS_TRANSLATING(str(tmp_path / "nope")) is False
    (tmp_path / "empty").mkdir()
    assert _REAL_IOMMU_IS_TRANSLATING(str(tmp_path / "empty")) is False


def test_iommu_unreadable_types_are_unknown(tmp_path):
    # Pre-5.x kernels expose groups without a `type` file: unknown, not "safe".
    root = _iommu_tree(tmp_path, {0: None, 1: None})
    assert _REAL_IOMMU_IS_TRANSLATING(root) is None
