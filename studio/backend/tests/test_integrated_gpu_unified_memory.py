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

from core.inference.llama_cpp import (
    _INTEGRATED_GPU_HOST_RESERVE_MIB as _RESERVE,
    LlamaCppBackend,
)


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


def _fake_torch_multi(devices, *, hip = None):
    # devices: list of dicts {integrated, free_mib, total_mib}, one per ordinal.
    t = types.ModuleType("torch")
    t.version = types.SimpleNamespace(hip = hip)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(devices),
        mem_get_info = lambda i: (
            devices[i]["free_mib"] * 1024 * 1024,
            devices[i]["total_mib"] * 1024 * 1024,
        ),
        get_device_properties = lambda i: types.SimpleNamespace(
            is_integrated = 1 if devices[i]["integrated"] else 0
        ),
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
        assert LlamaCppBackend._get_gpu_memory() == [(0, 61850 - _RESERVE, 124610)]


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


def test_integrated_gpu_clamps_down_to_budget(monkeypatch):
    # In a capped container CUDA can report host MemFree (8 GiB) above the cgroup
    # allowance (2 GiB). For a unified device the system budget is authoritative,
    # so free must be clamped DOWN to it, not left at the host reading (else the
    # fit picks a context that exceeds the container cap and gets OOM-killed).
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(integrated = True, free_mib = 8000, total_mib = 124610)
    )
    _fixed_avail(monkeypatch, 2000, total = 4096)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 2000 - _RESERVE, 4096)]


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
        assert LlamaCppBackend._get_gpu_memory() == [(0, 7000 - _RESERVE, 8192)]


# ── mixed-host dispatch: an [N/A] line must defer to torch ──


def test_mixed_host_na_line_defers_to_torch(monkeypatch):
    # GPU 0 discrete (normal), GPU 1 integrated ([N/A] free). nvidia-smi would
    # return only GPU 0; we must defer to torch so the integrated GPU is included
    # with the system-RAM budget instead of being dropped.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch_multi(
            [
                {"integrated": False, "free_mib": 20000, "total_mib": 24576},
                {"integrated": True, "free_mib": 1590, "total_mib": 124610},
            ]
        ),
    )
    _fixed_avail(monkeypatch, 50000, total = 64000)
    with _mock_nvidia_smi_run("0, 20000, 24576\n1, [N/A], [N/A]\n"):
        gpus = LlamaCppBackend._get_gpu_memory()
    # Discrete GPU keeps mem_get_info; integrated GPU gets the budget.
    assert gpus == [(0, 20000, 24576), (1, 50000 - _RESERVE, 64000)]


def test_clean_nvidia_smi_does_not_probe_torch(monkeypatch):
    # All-numeric nvidia-smi is the fast path: torch must not be touched.
    boom = types.ModuleType("torch")

    def _raise(*a, **k):
        raise AssertionError("torch must not be probed on a clean nvidia-smi result")

    boom.cuda = types.SimpleNamespace(is_available = _raise)
    monkeypatch.setitem(sys.modules, "torch", boom)
    with _mock_nvidia_smi_run("0, 20000, 24576\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24576)]


def test_skipped_line_keeps_smi_result_when_torch_absent(monkeypatch):
    # nvidia-smi skipped an [N/A] line but torch is unavailable: keep the parsed
    # discrete GPU rather than losing it.
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    with _mock_nvidia_smi_run("0, 20000, 24576\n1, [N/A], [N/A]\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24576)]


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


def _write_v2_level(
    root,
    rel,
    *,
    limit,
    current,
    inactive_file = 0,
):
    d = root if rel in ("", "/") else root / rel.lstrip("/")
    d.mkdir(parents = True, exist_ok = True)
    (d / "memory.max").write_text(str(limit))
    (d / "memory.current").write_text(str(current))
    (d / "memory.stat").write_text(f"inactive_file {inactive_file}\nactive_file 8192\n")


def test_cgroup_v2_walks_to_ancestor_limit(tmp_path):
    # The process's own /pod/ctr cgroup is uncapped ("max"), but the parent /pod
    # slice caps at 8 GiB (2 GiB used). The ancestor limit must still bind rather
    # than the process being treated as uncapped.
    root = tmp_path / "cgroup"
    _write_v2_level(root, "/pod/ctr", limit = "max", current = 1 * 1024**3)
    _write_v2_level(root, "/pod", limit = 8 * 1024**3, current = 2 * 1024**3)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/pod/ctr\n")
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(root)) == (6144, 8192)


def test_cgroup_v2_most_restrictive_ancestor_wins(tmp_path):
    # child caps 4 GiB (1 GiB used -> 3 GiB), parent caps 8 GiB: the tighter child
    # budget binds via the per-level minimum.
    root = tmp_path / "cgroup"
    _write_v2_level(root, "/pod/ctr", limit = 4 * 1024**3, current = 1 * 1024**3)
    _write_v2_level(root, "/pod", limit = 8 * 1024**3, current = 2 * 1024**3)
    proc = tmp_path / "proc_cgroup"
    proc.write_text("0::/pod/ctr\n")
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(root)) == (3072, 4096)


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


# ── a non-numeric line is NOT proof of unified memory ──
#
# nvidia-smi reports a non-numeric memory column for several reasons that have
# nothing to do with an integrated GPU: a MIG parent reports [N/A] for
# memory.free by design, and so do a vGPU guest, a card sitting in ERR!, and one
# the caller lacks permission to query. Discarding the whole nvidia-smi reading
# on that signal put those devices BACK into the result, priced with
# mem_get_info numbers that do not describe them, and offered them to auto
# placement. These tests pin the per-device rule: keep what nvidia-smi priced,
# fill in a skipped device only when torch says it is integrated.


def _mixed_torch(devices):
    return _fake_torch_multi(devices)


def test_mig_parent_na_does_not_resurrect_the_device(monkeypatch):
    # GPU 0 is a MIG parent (N/A by design, NOT integrated); GPU 1 is healthy.
    # Only the healthy card may be returned, exactly as before this change.
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": False, "free_mib": 1000, "total_mib": 40960},
        {"integrated": False, "free_mib": 20000, "total_mib": 24576},
    ]))
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run("0, [N/A], [N/A]\n1, 20000, 24576\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(1, 20000, 24576)]


def test_err_state_card_does_not_resurrect_and_healthy_cards_survive(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": False, "free_mib": 1, "total_mib": 24576},
        {"integrated": False, "free_mib": 20000, "total_mib": 24576},
        {"integrated": False, "free_mib": 18000, "total_mib": 24576},
    ]))
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run("0, ERR!, ERR!\n1, 20000, 24576\n2, 18000, 24576\n"):
        assert LlamaCppBackend._get_gpu_memory() == [(1, 20000, 24576), (2, 18000, 24576)]


def test_insufficient_permissions_line_is_not_integrated(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": False, "free_mib": 1000, "total_mib": 24576},
        {"integrated": False, "free_mib": 20000, "total_mib": 24576},
    ]))
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run(
        "0, [Insufficient Permissions], [Insufficient Permissions]\n1, 20000, 24576\n"
    ):
        assert LlamaCppBackend._get_gpu_memory() == [(1, 20000, 24576)]


def test_healthy_smi_host_never_initialises_cuda(monkeypatch):
    # Falling through to torch costs the backend process a PERMANENT CUDA
    # context. A host nvidia-smi answered for must not pay that, so the torch
    # module must go untouched when every line parsed.
    touched = {"n": 0}
    fake = _fake_torch(integrated = False, free_mib = 20000, total_mib = 24576)
    real_is_available = fake.cuda.is_available

    def counting_is_available():
        touched["n"] += 1
        return real_is_available()

    fake.cuda.is_available = counting_is_available
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run("0, 20000, 24576\n1, 18000, 24576\n"):
        LlamaCppBackend._get_gpu_memory()
    assert touched["n"] == 0


def test_integrated_device_on_mixed_host_is_still_filled_in(monkeypatch):
    # The case the restructure exists for keeps working: the skipped device IS
    # integrated, so torch prices it and it rejoins the healthy card.
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": False, "free_mib": 20000, "total_mib": 24576},
        {"integrated": True, "free_mib": 1590, "total_mib": 124610},
    ]))
    _fixed_avail(monkeypatch, 50000, total = 64000)
    with _mock_nvidia_smi_run("0, 20000, 24576\n1, [N/A], [N/A]\n"):
        assert LlamaCppBackend._get_gpu_memory() == [
            (0, 20000, 24576), (1, 50000 - _RESERVE, 64000)]


# ── host headroom and the shared pool ──


def test_integrated_free_keeps_host_headroom(monkeypatch):
    # This "VRAM" is the RAM the OS runs in; handing all of it to the fit is how
    # a unified-memory host meets the OOM killer.
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(integrated = True, free_mib = 1590, total_mib = 124610))
    _fixed_avail(monkeypatch, 61850)
    with _mock_nvidia_smi_run("0, [N/A], [N/A]\n"):
        free = LlamaCppBackend._get_gpu_memory()[0][1]
    assert free == 61850 - _RESERVE
    assert free < 61850


def test_two_integrated_devices_do_not_each_claim_the_whole_pool(monkeypatch):
    # One RAM pool, two integrated devices: a caller that sums across cards must
    # not be able to commit the pool twice.
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": True, "free_mib": 1590, "total_mib": 124610},
        {"integrated": True, "free_mib": 1500, "total_mib": 124610},
    ]))
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run("0, [N/A], [N/A]\n1, [N/A], [N/A]\n"):
        gpus = LlamaCppBackend._get_gpu_memory()
    assert sum(free for _idx, free, _total in gpus) <= 61850 - _RESERVE
    assert sum(total for _idx, _free, total in gpus) <= 124610


# ── torch shapes that must never cost every GPU ──


def test_torch_without_version_module_still_reports_gpus(monkeypatch):
    # `torch.version` is a submodule and has been absent on stripped builds.
    # Reading it with attribute access raised into the outer handler and returned
    # [], i.e. "no GPU at all", dropping Studio to CPU.
    fake = _fake_torch(integrated = False, free_mib = 20000, total_mib = 24576)
    del fake.version
    monkeypatch.setitem(sys.modules, "torch", fake)
    _fixed_avail(monkeypatch, 61850, total = 124610)
    with _mock_nvidia_smi_run("", returncode = 1):
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24576)]


def test_integrated_probe_is_empty_on_rocm(monkeypatch):
    # AMD APUs report is_integrated too; the physical-id helper must not claim
    # them, or the merge would price an APU from the CUDA budget.
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(integrated = True, free_mib = 65380,
                                    total_mib = 65536, hip = "7.2.53211"))
    assert LlamaCppBackend._integrated_gpu_physical_ids() == set()


def test_integrated_probe_reports_physical_ids(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _mixed_torch([
        {"integrated": False, "free_mib": 20000, "total_mib": 24576},
        {"integrated": True, "free_mib": 1590, "total_mib": 124610},
    ]))
    assert LlamaCppBackend._integrated_gpu_physical_ids() == {1}


# ── cgroup hardening ──


def test_unreadable_memory_stat_does_not_discard_a_real_limit(tmp_path):
    # Dropping the level here reported the cgroup as uncapped and handed the
    # caller the whole host, which is the direction that gets a container killed.
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "memory.max").write_text(str(8 * 1024 ** 3), encoding = "utf-8")
    (root / "memory.current").write_text(str(2 * 1024 ** 3), encoding = "utf-8")
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text("0::/\n", encoding = "utf-8")
    assert LlamaCppBackend._cgroup_memory_mib(str(proc), str(root)) == (6144, 8192)
