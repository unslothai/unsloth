# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On a unified-memory NVIDIA host (integrated GPU: CPU and GPU share one RAM
pool -- GB10 / DGX Spark, Jetson) nvidia-smi reports [N/A] free, so the probe
falls through to torch.cuda.mem_get_info, whose "free" tracks raw MemFree and
ignores reclaimable page cache. That misleading ~1.5 GB floored the context fit
at min_ctx=4096 even with ~60 GB really available (#6757). The torch fallback
must lift the free figure to system MemAvailable for integrated GPUs only, never
for discrete cards.
"""

from __future__ import annotations

import subprocess
import sys
import types
from unittest import mock

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    *,
    integrated,
    free_mib,
    total_mib,
    hip = None,
):
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    props = types.SimpleNamespace(is_integrated = 1 if integrated else 0)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        mem_get_info = lambda i: (free_mib * 1024 * 1024, total_mib * 1024 * 1024),
        get_device_properties = lambda i: props,
    )
    return t


def _mock_nvidia_smi_run(fake_output: str, returncode: int = 0):
    """Patch subprocess.run so only the nvidia-smi probe is faked; on a unified
    GPU it returns [N/A] columns, which the probe skips -> torch fallback."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and "nvidia-smi" in cmd[0]:
            return subprocess.CompletedProcess(
                args = cmd, returncode = returncode, stdout = fake_output, stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


@pytest.fixture(autouse = True)
def _clear_visibility_masks(monkeypatch):
    for _m in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(_m, raising = False)


def _fixed_avail(
    monkeypatch,
    mib,
    total = None,
):
    # Pin the unified-memory budget (available, total) that _get_gpu_memory reads.
    monkeypatch.setattr(
        LlamaCppBackend, "_system_memory_budget_mib", staticmethod(lambda: (mib, total))
    )


# ── _get_gpu_memory: unified-memory ceiling ──


def test_integrated_gpu_uses_system_available_ram(monkeypatch):
    # GB10 reproducer: nvidia-smi -> [N/A], mem_get_info free a misleading ~1.5 GB,
    # ~60 GB actually available. Free must be lifted to system-available; total,
    # already correct from mem_get_info on a unified device, stays put.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610)
    )
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("0, [N/A], [N/A]\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 61850, 124610)]


def test_discrete_gpu_keeps_mem_get_info_free(monkeypatch):
    # Discrete card: dedicated VRAM is the real ceiling, system RAM is irrelevant,
    # so the reported free must be left exactly as mem_get_info gives it.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = False, free_mib = 20000, total_mib = 24576)
    )
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("", returncode = 1):  # force the torch fallback
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24576)]


def test_integrated_gpu_unknown_available_keeps_free(monkeypatch):
    # System RAM unreadable (psutil + /proc both fail): fail safe to mem_get_info.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610)
    )
    _fixed_avail(monkeypatch, None)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 1590, 124610)]


def test_integrated_gpu_override_never_lowers_free(monkeypatch):
    # The override only ever raises the ceiling: if mem_get_info free already
    # exceeds system-available, keep the larger figure.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = True, free_mib = 8000, total_mib = 124610)
    )
    _fixed_avail(monkeypatch, 2000)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 8000, 124610)]


def test_rocm_integrated_apu_not_overridden(monkeypatch):
    # PyTorch flags AMD APUs as integrated too, but their unified-memory handling
    # is scoped to gfx1150/gfx1151 in _amd_apu_wants_unified_memory. Raising the
    # budget here without the matching launch-path support risks OOM, so ROCm
    # (torch.version.hip set) must keep mem_get_info's free untouched.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610, hip = "6.2.0"),
    )
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 1590, 124610)]


def test_integrated_gpu_clamps_total_to_container_budget(monkeypatch):
    # 8 GiB container cap on a 128 GiB host: total must be clamped to the budget
    # too. The fit reserves (1 - frac) * total, so leaving total at 124610 while
    # free is the 7 GiB allowance would zero the budget and floor the context.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610)
    )
    _fixed_avail(monkeypatch, 7000, total = 8192)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 7000, 8192)]


# ── _gpu_is_integrated flag ──


def test_gpu_is_integrated_true_and_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(integrated = True, free_mib = 1, total_mib = 1))
    assert LlamaCppBackend._gpu_is_integrated(0) is True
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = False, free_mib = 1, total_mib = 1)
    )
    assert LlamaCppBackend._gpu_is_integrated(0) is False


def test_gpu_is_integrated_missing_attr_is_false(monkeypatch):
    # Older torch without cudaDeviceProp.integrated must fail closed, not raise.
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = None)
    t.cuda = types.SimpleNamespace(get_device_properties = lambda i: types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", t)
    assert LlamaCppBackend._gpu_is_integrated(0) is False


# ── cgroup: read the process's own cgroup, discount reclaimable cache ──


def _make_v2(tmp_path, rel, *, limit, current, inactive_file):
    """Fake cgroup v2 tree: memory files at <root><rel>, /proc/self/cgroup -> rel."""
    root = tmp_path / "cgroup"
    d = root if rel in ("", "/") else root / rel.lstrip("/")
    d.mkdir(parents = True, exist_ok = True)
    (d / "memory.max").write_text(str(limit))
    (d / "memory.current").write_text(str(current))
    (d / "memory.stat").write_text(f"anon 4096\ninactive_file {inactive_file}\nactive_file 8192\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text(f"0::{rel or '/'}\n")
    return str(proc), str(root)


def test_cgroup_v2_available_discounts_reclaimable(tmp_path):
    # 8 GiB cap, 3 GiB used but 2 GiB of that is reclaimable file cache -> 1 GiB
    # working set -> 7 GiB (7168 MiB) available; limit 8192 MiB.
    proc, root = _make_v2(
        tmp_path, "/pod123", limit = 8 * 1024**3, current = 3 * 1024**3, inactive_file = 2 * 1024**3
    )
    assert LlamaCppBackend._cgroup_memory_mib(proc, root) == (7168, 8192)


def test_cgroup_v2_max_is_unlimited(tmp_path):
    proc, root = _make_v2(tmp_path, "/pod123", limit = "max", current = 2 * 1024**3, inactive_file = 0)
    assert LlamaCppBackend._cgroup_memory_mib(proc, root) == (None, None)


def test_cgroup_v2_reads_process_path_not_root(tmp_path):
    # The process's own /pod123 cgroup caps at 8 GiB; the mount root has a bogus
    # 1 GiB. Resolving the path from /proc/self/cgroup must read the pod, not root.
    proc, root = _make_v2(tmp_path, "/pod123", limit = 8 * 1024**3, current = 0, inactive_file = 0)
    root_dir = tmp_path / "cgroup"
    (root_dir / "memory.max").write_text(str(1 * 1024**3))
    (root_dir / "memory.current").write_text("0")
    (root_dir / "memory.stat").write_text("inactive_file 0\n")
    assert LlamaCppBackend._cgroup_memory_mib(proc, root) == (8192, 8192)


def test_cgroup_v2_falls_back_to_mount_root(tmp_path):
    # cgroup-namespaced: /proc/self/cgroup names /pod123 but the files live at the
    # (namespaced) mount root. The root files must still be read.
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(4 * 1024**3))
    (root / "memory.current").write_text(str(1024**3))
    (root / "memory.stat").write_text("inactive_file 0\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/pod123\n")  # path absent under root
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(root)) == (3072, 4096)


def test_cgroup_v1_available_discounts_reclaimable(tmp_path):
    root = tmp_path / "cgroup"
    memdir = root / "memory" / "docker" / "abc"
    memdir.mkdir(parents = True)
    (memdir / "memory.limit_in_bytes").write_text(str(4 * 1024**3))
    (memdir / "memory.usage_in_bytes").write_text(str(2 * 1024**3))
    (memdir / "memory.stat").write_text(f"cache 9\ntotal_inactive_file {1024**3}\n")
    proc = tmp_path / "proc_cgroup"
    proc.write_text("12:memory:/docker/abc\n11:cpu:/docker/abc\n")
    # usage 2 GiB - reclaimable 1 GiB = 1 GiB working set -> 3 GiB avail, 4 GiB limit.
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(root)) == (3072, 4096)


def test_cgroup_none_when_unreadable(tmp_path):
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/pod\n")
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(tmp_path / "absent")) == (None, None)


# ── budget composition: clamp both available and total to the cgroup ──


def test_system_budget_clamps_available_and_total(monkeypatch):
    # Big host (60/128 GB) but an 8 GB container cap: both figures clamp to the cap.
    monkeypatch.setattr(LlamaCppBackend, "_host_memory_mib", staticmethod(lambda: (60000, 128000)))
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_mib", staticmethod(lambda: (7000, 8192)))
    assert LlamaCppBackend._system_memory_budget_mib() == (7000, 8192)
    assert LlamaCppBackend._available_system_memory_mib() == 7000


def test_system_budget_uses_host_without_cgroup(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_host_memory_mib", staticmethod(lambda: (60000, 128000)))
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_mib", staticmethod(lambda: (None, None)))
    assert LlamaCppBackend._system_memory_budget_mib() == (60000, 128000)
    assert LlamaCppBackend._available_system_memory_mib() == 60000


def test_system_budget_v1_unlimited_folds_to_host(monkeypatch):
    # An unlimited v1 limit reads as a huge sentinel; min() with the host wins.
    huge = 0x7FFFFFFFFFFFF000 // (1024 * 1024)
    monkeypatch.setattr(LlamaCppBackend, "_host_memory_mib", staticmethod(lambda: (60000, 128000)))
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_memory_mib", staticmethod(lambda: (huge, huge)))
    assert LlamaCppBackend._system_memory_budget_mib() == (60000, 128000)
