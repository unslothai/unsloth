# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fake cgroup trees drive the single usable-host-RAM reader that all memory sizing consumers share."""

import os
import sys
import types
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

try:
    from utils import host_memory  # noqa: E402
except ImportError:  # a tree without the shared reader: only the consumer contract tests apply
    host_memory = None

reader = pytest.mark.skipif(host_memory is None, reason = "utils.host_memory not in this tree")

GIB = 1024**3
MIB = 1024**2
# What the kernel writes to memory.limit_in_bytes for an unlimited v1 group on 4 KiB pages.
V1_UNLIMITED = 9223372036854771712


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def _v2(
    root: Path,
    relative: str,
    *,
    limit,
    current = None,
    inactive_file = None,
) -> Path:
    d = root / relative.strip("/") if relative.strip("/") else root
    d.mkdir(parents = True, exist_ok = True)
    if limit is not None:
        _write(d / "memory.max", limit if isinstance(limit, str) else str(limit))
    if current is not None:
        _write(d / "memory.current", current if isinstance(current, str) else str(current))
    if inactive_file is not None:
        _write(d / "memory.stat", f"anon 1\nfile 2\ninactive_file {inactive_file}\nactive_file 3\n")
    return d


@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    """Point every cgroup reader (the shared one and llama.cpp's patch points) at a fake tree."""
    root = tmp_path / "cgroup"
    root.mkdir()
    proc = tmp_path / "proc_self_cgroup"
    if host_memory is not None:
        monkeypatch.setattr(host_memory, "CGROUP_ROOT", str(root))
        monkeypatch.setattr(host_memory, "PROC_SELF_CGROUP", str(proc))
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_CGROUP_ROOT", str(root))
    monkeypatch.setattr(llama_cpp, "_PROC_SELF_CGROUP", str(proc))

    def membership(text: str) -> None:
        proc.write_text(text, encoding = "utf-8")

    return types.SimpleNamespace(root = root, membership = membership)


@pytest.fixture
def host(monkeypatch):
    """A fixed host-wide psutil reading: 1.5 TiB total, 1.2 TiB available."""
    import psutil

    state = {"total": 1536 * GIB, "available": 1229 * GIB}
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(
            total = state["total"], available = state["available"], percent = 20.0
        ),
    )
    return state


@reader
def test_no_cgroup_keeps_the_host_reading(fake_cgroup, host):
    fake_cgroup.membership("0::/\n")
    assert host_memory.cgroup_memory_budgets() == []
    assert host_memory.cgroup_headroom_mib() is None
    assert host_memory.cgroup_limit_mib() is None
    assert host_memory.usable_host_ram_mib() == host["available"] // MIB
    assert host_memory.host_ram_capacity_mib() == host["total"] // MIB


@reader
def test_unlimited_v2_and_v1_levels_are_no_limit(fake_cgroup, host):
    fake_cgroup.membership("0::/user.slice/app.scope\n5:memory:/docker/abc\n")
    _v2(fake_cgroup.root, "user.slice/app.scope", limit = "max", current = 10 * GIB)
    _v2(fake_cgroup.root, "user.slice", limit = "max", current = 20 * GIB)
    v1 = fake_cgroup.root / "memory" / "docker" / "abc"
    _write(v1 / "memory.limit_in_bytes", str(V1_UNLIMITED))
    _write(v1 / "memory.usage_in_bytes", str(5 * GIB))
    assert host_memory.cgroup_memory_budgets() == []
    assert host_memory.usable_host_ram_mib() == host["available"] // MIB


@reader
def test_v2_headroom_is_limit_minus_working_set(fake_cgroup, host):
    fake_cgroup.membership("0::/user.slice/run-x.scope\n")
    _v2(
        fake_cgroup.root,
        "user.slice/run-x.scope",
        limit = 58 * GIB,
        current = 30 * GIB,
        inactive_file = 6 * GIB,
    )
    # 58 - (30 - 6 reclaimable inactive file) = 34 GiB.
    assert host_memory.cgroup_headroom_mib() == 34 * 1024
    assert host_memory.cgroup_limit_mib() == 58 * 1024
    assert host_memory.usable_host_ram_mib() == 34 * 1024
    assert host_memory.host_ram_capacity_mib() == 58 * 1024


@reader
def test_v1_headroom_uses_hierarchical_inactive_file(fake_cgroup, host):
    fake_cgroup.membership("12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc\n")
    d = fake_cgroup.root / "memory" / "docker" / "abc"
    _write(d / "memory.limit_in_bytes", str(16 * GIB))
    _write(d / "memory.usage_in_bytes", str(10 * GIB))
    _write(d / "memory.stat", f"inactive_file {1 * GIB}\ntotal_inactive_file {2 * GIB}\n")
    assert host_memory.cgroup_headroom_mib() == 8 * 1024
    assert host_memory.cgroup_limit_mib() == 16 * 1024
    assert host_memory.usable_host_ram_mib() == 8 * 1024


@reader
def test_v1_container_root_mount(fake_cgroup, host):
    # A v1 container sees its own group at the mount root while /proc/self/cgroup names the host path.
    fake_cgroup.membership("12:memory:/docker/abc\n")
    d = fake_cgroup.root / "memory"
    _write(d / "memory.limit_in_bytes", str(8 * GIB))
    _write(d / "memory.usage_in_bytes", str(3 * GIB))
    assert host_memory.cgroup_headroom_mib() == 5 * 1024
    assert host_memory.cgroup_limit_mib() == 8 * 1024


@reader
def test_nested_parent_slice_binds_with_sibling_usage(fake_cgroup, host):
    # The leaf scope allows 70 GiB but its parent slice caps at 64 GiB and siblings already use 40.
    fake_cgroup.membership("0::/workload.slice/studio.scope\n")
    _v2(fake_cgroup.root, "workload.slice/studio.scope", limit = 70 * GIB, current = 10 * GIB)
    _v2(fake_cgroup.root, "workload.slice", limit = 64 * GIB, current = 50 * GIB)
    assert host_memory.cgroup_headroom_mib() == 14 * 1024  # parent remainder, not the leaf's 60
    assert host_memory.cgroup_limit_mib() == 64 * 1024  # tightest limit
    assert host_memory.usable_host_ram_mib() == 14 * 1024


@reader
def test_nested_unlimited_leaf_under_limited_ancestor(fake_cgroup, host):
    fake_cgroup.membership("0::/a.slice/b.slice/c.scope\n")
    _v2(fake_cgroup.root, "a.slice/b.slice/c.scope", limit = "max", current = 5 * GIB)
    _v2(fake_cgroup.root, "a.slice/b.slice", limit = "max", current = 6 * GIB)
    _v2(fake_cgroup.root, "a.slice", limit = 32 * GIB, current = 12 * GIB)
    assert host_memory.cgroup_headroom_mib() == 20 * 1024
    assert host_memory.cgroup_limit_mib() == 32 * 1024


@reader
def test_host_reading_wins_when_it_is_tighter(fake_cgroup, host):
    host["available"] = 10 * GIB
    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 1 * GIB)
    assert host_memory.usable_host_ram_mib() == 10 * 1024


@reader
def test_over_limit_usage_clamps_headroom_to_zero(fake_cgroup, host):
    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 8 * GIB, current = 9 * GIB)
    assert host_memory.cgroup_headroom_mib() == 0
    assert host_memory.usable_host_ram_mib() == 0


@reader
@pytest.mark.parametrize(
    "limit, current, expect_headroom_mib",
    [
        ("garbage", str(GIB), None),  # unparsable limit: level skipped
        ("", str(GIB), None),  # empty limit file
        ("-5", str(GIB), None),  # negative limit
        (str(4 * GIB), "garbage", 4 * 1024),  # unparsable usage: whole limit is the remainder
        (str(4 * GIB), "-1", 4 * 1024),  # negative usage: same
    ],
)
@reader
def test_malformed_files_never_raise(fake_cgroup, host, limit, current, expect_headroom_mib):
    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = limit, current = current)
    assert host_memory.cgroup_headroom_mib() == expect_headroom_mib
    usable = host_memory.usable_host_ram_mib()
    expected = host["available"] // MIB if expect_headroom_mib is None else expect_headroom_mib
    assert usable == expected


@reader
def test_malformed_stat_and_binary_files_never_raise(fake_cgroup, host):
    fake_cgroup.membership("0::/s.scope\n")
    d = _v2(fake_cgroup.root, "s.scope", limit = 8 * GIB, current = 2 * GIB)
    (d / "memory.stat").write_bytes(b"inactive_file \xff\xfe\nnot a stat line at all\n")
    assert host_memory.cgroup_headroom_mib() == 6 * 1024
    (d / "memory.max").write_bytes(b"\xff\xfe\x00")
    assert host_memory.cgroup_headroom_mib() is None


@reader
def test_unreadable_proc_self_cgroup_reads_the_root_only(fake_cgroup, host):
    # No /proc/self/cgroup (e.g. a sandbox): only the mount root is consulted.
    _v2(fake_cgroup.root, "", limit = 12 * GIB, current = 2 * GIB)
    assert host_memory.cgroup_headroom_mib() == 10 * 1024


@reader
def test_escaping_relative_path_falls_back_to_the_root(fake_cgroup, host):
    fake_cgroup.membership("0::/../../etc\n")
    _v2(fake_cgroup.root, "", limit = 12 * GIB, current = 2 * GIB)
    assert host_memory.cgroup_headroom_mib() == 10 * 1024


@reader
def test_psutil_missing_falls_back_to_proc_meminfo(fake_cgroup, monkeypatch):
    fake_cgroup.membership("0::/\n")
    monkeypatch.setitem(sys.modules, "psutil", None)
    if os.path.exists("/proc/meminfo"):
        assert host_memory.system_available_mib() is not None
        assert host_memory.system_total_mib() is not None


@reader
def test_llama_cpp_readers_delegate_to_the_shared_one(fake_cgroup, host):
    from core.inference.llama_cpp import LlamaCppBackend

    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 20 * GIB)
    assert LlamaCppBackend._cgroup_available_memory_mib() == host_memory.cgroup_headroom_mib()
    assert LlamaCppBackend._cgroup_memory_limit_mib() == host_memory.cgroup_limit_mib()
    assert LlamaCppBackend._available_system_memory_mib() == host_memory.usable_host_ram_mib()
    assert LlamaCppBackend._host_memory_capacity_mib() == host_memory.host_ram_capacity_mib()


def test_pin_budget_sizes_from_the_cgroup_headroom(fake_cgroup, host):
    from core.inference import diffusion_memory as dm

    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 30 * GIB)
    reserve = max(dm._PIN_RESERVE_MIN_MIB, int(58 * 1024 * dm._PIN_RESERVE_FRACTION))
    assert dm._pin_budget_mib() == 28 * 1024 - reserve
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 52 * GIB)
    assert dm._pin_budget_mib() == 0


def test_pin_budget_without_a_limit_is_unchanged(fake_cgroup, host):
    from core.inference import diffusion_memory as dm

    fake_cgroup.membership("0::/\n")
    total_mib, available_mib = host["total"] // MIB, host["available"] // MIB
    reserve = max(dm._PIN_RESERVE_MIN_MIB, int(total_mib * dm._PIN_RESERVE_FRACTION))
    assert dm._pin_budget_mib() == available_mib - reserve


def _h3_held(monkeypatch, held_gb: float) -> None:
    from core.inference import video_minimax_h3 as vmh3
    kb = int(held_gb * 1e9 / 1024)
    monkeypatch.setattr(vmh3, "_proc_status_kb", lambda fields: {"RssAnon": kb, "RssShmem": 0})


def test_h3_guard_capacity_is_capped_by_the_cgroup(fake_cgroup, host, monkeypatch):
    from core.inference import video_minimax_h3 as vmh3

    _h3_held(monkeypatch, 30.0)
    fake_cgroup.membership("0::/s.scope\n")
    # 58 GiB limit, the server already charges 35 GiB (30 GB of it the reusable model).
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 35 * GIB)
    capacity = vmh3.h3_host_capacity_bytes()
    assert capacity == 23 * GIB + int(30.0 * 1e9 / 1024) * 1024
    assert capacity <= 58 * GIB
    kw = dict(text_encoder_gb = 27.2, transformer_gb = 20.3, text_encoder_streamed = True)
    assert vmh3.h3_host_ram_shortfall(11.0, **kw) is None
    # Below the measured floor the cgroup is what refuses, not the 1.2 TiB the host reports.
    floor = vmh3.H3_DIFFUSERS_HOST_RAM_STREAMED_SET_GB
    small = int((floor - 2.0) * 1e9)
    _v2(fake_cgroup.root, "s.scope", limit = small, current = 35 * GIB)
    message = vmh3.h3_host_ram_shortfall(11.0, **kw)
    assert message is not None and f"{floor:.0f} GB" in message


def test_h3_guard_capacity_never_exceeds_the_limit(fake_cgroup, host, monkeypatch):
    from core.inference import video_minimax_h3 as vmh3

    # Held anon charged to ANOTHER cgroup (or counted twice) still cannot lift capacity past the limit.
    _h3_held(monkeypatch, 60.0)
    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 1 * GIB)
    assert vmh3.h3_host_capacity_bytes() == 58 * GIB


def test_h3_guard_without_a_limit_is_unchanged(fake_cgroup, host, monkeypatch):
    from core.inference import video_minimax_h3 as vmh3

    _h3_held(monkeypatch, 40.0)
    fake_cgroup.membership("0::/\n")
    assert vmh3.h3_host_capacity_bytes() == host["available"] + int(40.0 * 1e9 / 1024) * 1024
    assert (
        vmh3.h3_host_ram_shortfall(
            11.0, text_encoder_gb = 27.2, transformer_gb = 20.3, text_encoder_streamed = True
        )
        is None
    )


def test_h3_guard_admits_the_tier_under_a_limit_that_fits(fake_cgroup, host, monkeypatch):
    from core.inference import video_minimax_h3 as vmh3

    _h3_held(monkeypatch, 40.0)
    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 80 * GIB, current = 42 * GIB)
    assert (
        vmh3.h3_host_ram_shortfall(
            11.0, text_encoder_gb = 27.2, transformer_gb = 20.3, text_encoder_streamed = True
        )
        is None
    )


def _system_memory(monkeypatch):
    import main
    monkeypatch.setattr(
        main,
        "_get_cached_system_gpu_info",
        lambda logger, refresh_memory = False: ({"available": False}, {"available": False}),
    )
    return main.get_system_info(current_subject = "test")["memory"]


def test_api_system_memory_reports_the_cgroup_view(fake_cgroup, host, monkeypatch):
    from core.inference import video_minimax_h3 as vmh3

    fake_cgroup.membership("0::/s.scope\n")
    _v2(fake_cgroup.root, "s.scope", limit = 58 * GIB, current = 2 * GIB)
    memory = _system_memory(monkeypatch)
    assert memory["total_gb"] == 58.0
    assert memory["available_gb"] == 56.0
    assert memory["percent_used"] == pytest.approx(3.4, abs = 0.05)
    tiers = vmh3.h3_diffusers_fit_tiers()
    if tiers:
        # The picker's RAM tier is the guard's floor: 58 GiB admits the H3 row, as the guard admits the render.
        assert memory["available_gb"] >= tiers[0]["system_ram_gb"]
        floor_gib = tiers[0]["system_ram_gb"]
        _v2(fake_cgroup.root, "s.scope", limit = int((floor_gib - 1) * GIB), current = 0)
        # A limit under the floor withdraws the row, while the host still reports 1.2 TiB free.
        assert _system_memory(monkeypatch)["available_gb"] < floor_gib


def test_api_system_memory_without_a_limit_is_unchanged(fake_cgroup, host, monkeypatch):
    fake_cgroup.membership("0::/\n")
    memory = _system_memory(monkeypatch)
    assert memory["total_gb"] == round(host["total"] / GIB, 2)
    assert memory["available_gb"] == round(host["available"] / GIB, 2)
    assert memory["percent_used"] == 20.0
