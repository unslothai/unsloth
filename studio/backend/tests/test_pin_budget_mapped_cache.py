# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pin budget credits clean mapped page cache inside a capped cgroup, so it does not flip with LRU aging."""

import sys
import types
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from utils import host_memory  # noqa: E402
from core.inference import diffusion_memory as dm  # noqa: E402

GIB = 1024**3
MIB = 1024**2


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def _stat(directory: Path, **counters) -> None:
    _write(directory / "memory.stat", "".join(f"{k} {v}\n" for k, v in counters.items()))


@pytest.fixture
def cg(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    root.mkdir()
    proc = tmp_path / "proc_self_cgroup"
    monkeypatch.setattr(host_memory, "CGROUP_ROOT", str(root))
    monkeypatch.setattr(host_memory, "PROC_SELF_CGROUP", str(proc))
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(llama_cpp, "_PROC_SELF_CGROUP", str(proc))
    monkeypatch.delenv(dm.PIN_CREDIT_MAPPED_ENV, raising = False)

    import psutil

    host = {"total": 94 * GIB, "available": 80 * GIB}
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(
            total = host["total"], available = host["available"], percent = 20.0
        ),
    )
    return types.SimpleNamespace(root = root, proc = proc, host = host)


def _scope(cg, *, limit, current, **stat) -> Path:
    cg.proc.write_text("0::/app.slice/run-x.scope\n", encoding = "utf-8")
    d = cg.root / "app.slice" / "run-x.scope"
    _write(d / "memory.max", str(limit))
    _write(d / "memory.current", str(current))
    _stat(d, **stat)
    return d


def _reserve(limit_gib: int) -> int:
    return max(dm._PIN_RESERVE_MIN_MIB, int(limit_gib * 1024 * dm._PIN_RESERVE_FRACTION))


def test_same_live_set_reads_the_same_budget_on_either_lru_list(cg):
    """70G scope, 40 GiB anon + 20 GiB mmap'd checkpoint: the tester saw 7.7 GB and 0 for this one state."""
    common = dict(anon = 40 * GIB, file_mapped = 20 * GIB, file_dirty = 0, file_writeback = 0)
    _scope(cg, limit = 70 * GIB, current = 60 * GIB, inactive_file = 20 * GIB, active_file = 0, **common)
    on_inactive = dm._pin_budget_mib()
    _scope(cg, limit = 70 * GIB, current = 60 * GIB, inactive_file = 0, active_file = 20 * GIB, **common)
    on_active = dm._pin_budget_mib()
    assert on_inactive == on_active == 30 * 1024 - _reserve(70)
    # The general reader keeps its stricter active-list rule (GGUF admission, /api/system).
    assert host_memory.cgroup_headroom_mib() == 10 * 1024


def test_kill_switch_restores_the_uncredited_budget(cg, monkeypatch):
    _scope(
        cg,
        limit = 70 * GIB,
        current = 60 * GIB,
        anon = 40 * GIB,
        inactive_file = 0,
        active_file = 20 * GIB,
        file_mapped = 20 * GIB,
    )
    monkeypatch.setenv(dm.PIN_CREDIT_MAPPED_ENV, "0")
    assert dm._pin_budget_mib() == 0  # 10 GiB headroom is under the 10.5 GiB reserve
    monkeypatch.delenv(dm.PIN_CREDIT_MAPPED_ENV)
    assert dm._pin_budget_mib() == 30 * 1024 - _reserve(70)


def test_an_earlier_reading_is_not_added_to_the_credited_one(cg, monkeypatch):
    """The pages moved to the active list after the first reading credited them from the inactive list."""
    _scope(
        cg,
        limit = 70 * GIB,
        current = 60 * GIB,
        anon = 40 * GIB,
        inactive_file = 0,
        active_file = 20 * GIB,
        file_mapped = 20 * GIB,
    )
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 30 * 1024)
    assert dm._pin_budget_mib() == 30 * 1024 - _reserve(70)


def test_dirty_and_writeback_pages_stay_uncredited(cg):
    _scope(
        cg,
        limit = 32 * GIB,
        current = 30 * GIB,
        inactive_file = 2 * GIB,
        active_file = 10 * GIB,
        file_mapped = 4 * GIB,
        file_dirty = 1 * GIB,
        file_writeback = 1 * GIB,
    )
    assert host_memory.cgroup_headroom_mib() == 8 * 1024
    assert host_memory.credited_cgroup_headroom_mib() == 12 * 1024
    assert dm._pin_budget_mib() == 12 * 1024 - _reserve(32)


def test_shmem_is_never_credited(cg):
    """tmpfs pages count in file_mapped when mapped but live on the anon LRU, so they are in neither *_file list."""
    _scope(
        cg,
        limit = 32 * GIB,
        current = 24 * GIB,
        shmem = 12 * GIB,
        inactive_file = 2 * GIB,
        active_file = 0,
        file_mapped = 12 * GIB,
    )
    assert host_memory.credited_cgroup_headroom_mib() == 10 * 1024
    assert dm._pin_budget_mib() == 10 * 1024 - _reserve(32)


def test_host_memavailable_still_bounds_the_credit(cg):
    _scope(
        cg,
        limit = 70 * GIB,
        current = 60 * GIB,
        inactive_file = 0,
        active_file = 20 * GIB,
        file_mapped = 20 * GIB,
    )
    cg.host["available"] = 15 * GIB
    assert dm._pin_budget_mib() == 15 * 1024 - _reserve(70)
    cg.host["available"] = (
        6 * GIB
    )  # tighter than the cgroup reading: no credit, low-RAM reserve applies
    assert dm._pin_budget_mib() == 0


def test_tightest_level_binds_with_the_credit(cg):
    """An ancestor limit with sibling usage still binds after the leaf's mapped pages are credited."""
    leaf = _scope(
        cg,
        limit = 70 * GIB,
        current = 60 * GIB,
        inactive_file = 0,
        active_file = 20 * GIB,
        file_mapped = 20 * GIB,
    )
    parent = leaf.parent
    _write(parent / "memory.max", str(72 * GIB))
    _write(parent / "memory.current", str(68 * GIB))
    _stat(parent, inactive_file = 0, active_file = 20 * GIB, file_mapped = 20 * GIB)
    # leaf 70 - 40 = 30, parent 72 - 48 = 24 GiB.
    assert dm._pin_budget_mib() == 24 * 1024 - _reserve(70)


def test_no_limit_is_unchanged(cg):
    cg.proc.write_text("0::/\n", encoding = "utf-8")
    assert host_memory.credited_cgroup_headroom_mib() is None
    assert dm._pin_budget_mib() == 80 * 1024 - int(94 * 1024 * dm._PIN_RESERVE_FRACTION)


def test_v1_hierarchical_mapped_pages_are_credited(cg):
    cg.proc.write_text("12:memory:/docker/abc\n", encoding = "utf-8")
    d = cg.root / "memory" / "docker" / "abc"
    _write(d / "memory.limit_in_bytes", str(64 * GIB))
    _write(d / "memory.usage_in_bytes", str(60 * GIB))
    _stat(
        d,
        inactive_file = 50 * GIB,
        active_file = 50 * GIB,
        mapped_file = 50 * GIB,
        total_inactive_file = 2 * GIB,
        total_active_file = 18 * GIB,
        total_mapped_file = 16 * GIB,
        total_dirty = 1 * GIB,
        total_writeback = 0,
    )
    # Uncredited: 64 - (60 - 2 - 1) = 7 GiB. Credited: 64 - (60 - 2 - 17) = 23 GiB.
    assert host_memory.cgroup_headroom_mib() == 7 * 1024
    assert host_memory.credited_cgroup_headroom_mib() == 23 * 1024
    assert dm._pin_budget_mib() == 23 * 1024 - _reserve(64)


def test_v1_level_counters_without_totals(cg):
    cg.proc.write_text("12:memory:/docker/abc\n", encoding = "utf-8")
    d = cg.root / "memory" / "docker" / "abc"
    _write(d / "memory.limit_in_bytes", str(16 * GIB))
    _write(d / "memory.usage_in_bytes", str(14 * GIB))
    _stat(d, inactive_file = 1 * GIB, active_file = 6 * GIB, mapped_file = 2 * GIB)
    assert host_memory.cgroup_headroom_mib() == 7 * 1024
    assert host_memory.credited_cgroup_headroom_mib() == 9 * 1024


def test_unreadable_stat_never_raises(cg):
    d = _scope(cg, limit = 32 * GIB, current = 20 * GIB)
    _write(d / "memory.stat", "garbage\nactive_file x\n")
    assert host_memory.credited_cgroup_headroom_mib() == 12 * 1024
    assert dm._pin_budget_mib() == 12 * 1024 - _reserve(32)


def test_integrated_gpu_still_never_pins(cg, monkeypatch):
    """The credit only sizes the budget; the integrated-GPU refusal ahead of it is untouched."""
    _scope(
        cg,
        limit = 70 * GIB,
        current = 60 * GIB,
        inactive_file = 0,
        active_file = 20 * GIB,
        file_mapped = 20 * GIB,
    )
    monkeypatch.setattr(dm, "host_pinning_allowed", lambda device = None: False)
    monkeypatch.delenv(dm.GROUP_OFFLOAD_PIN_ENV, raising = False)
    assert dm._streamed_pin_plan(1024, 1024) == (False, False)
