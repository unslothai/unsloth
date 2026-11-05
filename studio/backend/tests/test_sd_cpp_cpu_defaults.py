# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sd.cpp ``--threads``: ``UNSLOTH_CPU_THREADS`` reaches it, and a cgroup CPU quota caps the default."""

from __future__ import annotations

import pytest

from core.inference import sd_cpp_backend as bk


def _cgroups(
    tmp_path,
    quotas,
    own = "/user.slice/app.scope",
):
    """A cgroup v2 tree where ``quotas`` maps a cgroup path to its cpu.max line, plus /proc/self/cgroup."""
    root = tmp_path / "cg"
    for rel, line in quotas.items():
        d = root / rel.lstrip("/")
        d.mkdir(parents = True, exist_ok = True)
        (d / "cpu.max").write_text(line + "\n")
    (root / own.lstrip("/")).mkdir(parents = True, exist_ok = True)
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{own}\n")
    return str(root), str(proc)


@pytest.mark.parametrize(
    "quotas, expected",
    [
        ({}, None),
        ({"/user.slice/app.scope": "max 100000"}, None),
        ({"/user.slice/app.scope": "400000 100000"}, 4),
        ({"/user.slice/app.scope": "250000 100000"}, 3),
        ({"/user.slice": "200000 100000", "/user.slice/app.scope": "800000 100000"}, 2),
        ({"/": "600000 100000"}, 6),
        ({"/user.slice/app.scope": "garbage"}, None),
    ],
)
def test_the_tightest_quota_over_the_own_cgroup_and_its_parents(tmp_path, quotas, expected):
    root, proc = _cgroups(tmp_path, quotas)
    assert bk._cgroup_cpu_limit(root, proc) == expected


def test_a_container_with_its_own_cgroup_namespace_reads_the_root(tmp_path):
    root, proc = _cgroups(tmp_path, {"/": "400000 100000"}, own = "/")
    assert bk._cgroup_cpu_limit(root, proc) == 4


def test_no_cgroup_v2_entry_reads_the_root_only(tmp_path):
    root, _ = _cgroups(tmp_path, {"/": "300000 100000"})
    missing = str(tmp_path / "missing")
    assert bk._cgroup_cpu_limit(root, missing) == 3


def _host(
    monkeypatch,
    *,
    cpus = 192,
    limit = None,
    platform = "linux",
    physical = None,
    allowed = None,
    hybrid = False,
    windows = None,
):
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(bk.sys, "platform", platform)
    monkeypatch.setattr(bk.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: limit)
    monkeypatch.setattr(bk, "_linux_physical_cores", lambda *a, **k: physical)
    monkeypatch.setattr(bk, "_allowed_cpus", lambda: cpus if allowed is None else allowed)
    monkeypatch.setattr(bk, "_linux_hybrid_cpu", lambda *a, **k: hybrid)
    monkeypatch.setattr(bk, "_windows_cores", lambda: windows)
    monkeypatch.delenv(bk.SD_CPP_CPU_SMT_ENV, raising = False)


def test_without_a_quota_the_default_is_unchanged(monkeypatch):
    _host(monkeypatch)
    assert bk._default_threads() == 96
    _host(monkeypatch, cpus = None)
    assert bk._default_threads() == 8


def test_a_quota_caps_the_default(monkeypatch):
    _host(monkeypatch, limit = 4)
    assert bk._default_threads() == 4
    _host(monkeypatch, cpus = 8, limit = 16)
    assert bk._default_threads() == 4


def test_other_platforms_never_read_cgroups(monkeypatch):
    _host(monkeypatch, cpus = 10, platform = "darwin")
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: None)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: pytest.fail("no cgroups off Linux"))
    assert bk._default_threads() == 5


@pytest.mark.parametrize("cpus, p_cores", [(3, 3), (10, 8), (16, 12), (10, 4)])
def test_macos_uses_the_performance_core_count_not_half_the_cores(monkeypatch, cpus, p_cores):
    _host(monkeypatch, cpus = cpus, platform = "darwin")
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: p_cores)
    assert bk._default_threads() == p_cores
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "2")
    assert bk._default_threads() == 2


def test_apple_performance_cores_is_none_off_macos(monkeypatch):
    monkeypatch.setattr(bk.sys, "platform", "linux")
    assert bk._apple_performance_cores() is None


@pytest.mark.parametrize("value, expected", [("12", 12), ("0", 4), ("x", 4), ("", 4), ("-3", 4)])
def test_unsloth_cpu_threads_wins_over_the_default_and_the_quota(monkeypatch, value, expected):
    _host(monkeypatch, limit = 4)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", value)
    assert bk._default_threads() == expected


_TOPOLOGIES = {
    "m1pro": {
        "hw.nperflevels": 2,
        "hw.perflevel0.name": "Performance",
        "hw.perflevel0.physicalcpu": 8,
        "hw.perflevel1.name": "Efficiency",
        "hw.perflevel1.physicalcpu": 2,
        "hw.physicalcpu": 10,
    },
    # ggml-org/llama.cpp#29782
    "m5max": {
        "hw.nperflevels": 2,
        "hw.perflevel0.name": "Super",
        "hw.perflevel0.physicalcpu": 6,
        "hw.perflevel1.name": "Performance",
        "hw.perflevel1.physicalcpu": 12,
        "hw.physicalcpu": 18,
    },
    "vm": {
        "hw.nperflevels": 1,
        "hw.perflevel0.name": "Performance",
        "hw.perflevel0.physicalcpu": 3,
        "hw.physicalcpu": 3,
    },
    "intel": {"hw.physicalcpu": 6},
    "unnamed": {
        "hw.nperflevels": 2,
        "hw.perflevel0.physicalcpu": 4,
        "hw.perflevel1.physicalcpu": 6,
        "hw.physicalcpu": 10,
    },
    # a lower tier with a name this code does not know is left out, as an unnamed one is
    "unknown_tier": {
        "hw.nperflevels": 3,
        "hw.perflevel0.name": "Super",
        "hw.perflevel0.physicalcpu": 4,
        "hw.perflevel1.name": "Performance",
        "hw.perflevel1.physicalcpu": 6,
        "hw.perflevel2.name": "Background",
        "hw.perflevel2.physicalcpu": 4,
        "hw.physicalcpu": 14,
    },
}


@pytest.mark.parametrize(
    "topology, expected",
    [("m1pro", 8), ("m5max", 18), ("vm", 3), ("intel", 6), ("unnamed", 4), ("unknown_tier", 10)],
)
def test_apple_cores_count_perflevel0_and_the_named_fast_tiers(monkeypatch, topology, expected):
    table = _TOPOLOGIES[topology]
    monkeypatch.setattr(bk.sys, "platform", "darwin")
    monkeypatch.setattr(bk, "_sysctl", lambda name, as_str = False: table.get(name))
    assert bk._apple_performance_cores() == expected


def test_the_libc_handle_is_looked_up_once_and_typed(monkeypatch):
    import ctypes
    import ctypes.util

    class _Fn:
        def __call__(self, *args):
            return 0

    class _Libc:
        def __init__(self, path):
            self.sysctlbyname = _Fn()

    calls = []
    monkeypatch.setattr(
        ctypes.util, "find_library", lambda name: calls.append(name) or "libc.dylib"
    )
    monkeypatch.setattr(ctypes, "CDLL", _Libc)
    bk._libc.cache_clear()
    try:
        for _ in range(3):
            bk._sysctl("hw.physicalcpu")
        assert calls == ["c"]
        fn = bk._libc().sysctlbyname
        assert fn.restype is ctypes.c_int and len(fn.argtypes) == 5
    finally:
        bk._libc.cache_clear()


def test_minimax_h3_video_cli_and_server_get_the_default_thread_count(monkeypatch, tmp_path):
    """H3 runs its text encoder on the Mac CPU, so both native video paths pass Studio's count."""
    from core.inference import sd_cpp_engine as eng
    from core.inference import sd_cpp_server as srv
    from core.inference import video_minimax_h3 as h3
    from core.inference.sd_cpp_args import (
        SdCppModelFiles,
        SdCppVideoGenParams,
        build_sd_cpp_video_command,
    )

    files = SdCppModelFiles(
        diffusion_model = "/m/h3.gguf", vae = "/m/vae.safetensors", llm = "/m/te.gguf"
    )
    params = SdCppVideoGenParams(prompt = "a dog", width = 448, height = 256, num_frames = 17, fps = 24)
    cmd = build_sd_cpp_video_command("/bin/sd-cli", files, params, output_path = "/o.mp4", threads = 7)
    assert cmd[cmd.index("--threads") + 1] == "7"
    assert "--threads" not in build_sd_cpp_video_command(
        "/bin/sd-cli", files, params, output_path = "/o.mp4"
    )

    seen = {}
    engine = eng.SdCppEngine.__new__(eng.SdCppEngine)
    monkeypatch.setattr(engine, "_require_binary", lambda: "/bin/sd-cli", raising = False)
    monkeypatch.setattr(
        engine, "_run", lambda cmd, out, **kw: seen.setdefault("cmd", cmd), raising = False
    )
    engine.generate_video(files, params, output_path = str(tmp_path / "o.mp4"), threads = 5)
    assert seen["cmd"][seen["cmd"].index("--threads") + 1] == "5"

    monkeypatch.setattr(bk, "_explicit_threads", lambda: 11)
    started = {}

    class _Srv:
        def __init__(self, binary):
            self.alive = False

        def start(
            self,
            files,
            *,
            offload = None,
            threads = None,
            env = None,
            extra_args = None,
        ):
            started["threads"] = threads
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    slot = h3.H3NativeServerSlot("/bin/sd-server", files, ("--offload-to-cpu",))
    slot.get()
    slot.stop()
    assert started["threads"] == 11


def test_h3_video_keeps_sd_cpps_default_threads_unless_studio_knows_better(monkeypatch):
    # Linux / Windows with no override and no tighter quota: no --threads, as on main (sd.cpp counts physical cores)
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: None)
    monkeypatch.setattr(bk.os, "cpu_count", lambda: 32)
    monkeypatch.setattr(bk.sys, "platform", "linux")
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda *a, **k: None)
    monkeypatch.setattr(bk, "_linux_physical_cores", lambda *a, **k: None)
    assert bk._explicit_threads() is None
    monkeypatch.setattr(
        bk, "_linux_physical_cores", lambda *a, **k: 16
    )  # 2-way SMT: what sd.cpp picks too
    assert bk._explicit_threads() is None
    monkeypatch.setattr(
        bk, "_cgroup_cpu_limit", lambda *a, **k: 16
    )  # equal to the default: still sd.cpp's own
    assert bk._explicit_threads() is None
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda *a, **k: 6)  # a tighter container quota
    assert bk._explicit_threads() == 6
    monkeypatch.setattr(bk.sys, "platform", "win32")
    assert bk._explicit_threads() is None
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "12")
    assert bk._explicit_threads() == 12
    monkeypatch.delenv("UNSLOTH_CPU_THREADS")
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: 10)
    assert bk._explicit_threads() == 10


def _sysfs(tmp_path, siblings):
    """``/sys/devices/system/cpu`` with ``siblings[cpu]`` as each CPU's thread_siblings_list."""
    root = tmp_path / "cpu"
    for cpu, line in siblings.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents = True)
        (d / "thread_siblings_list").write_text(line + "\n")
    return str(root)


_SMT8 = {
    c: f"{c % 4},{c % 4 + 4}" for c in range(8)
}  # 4 cores, 2 threads each (siblings c and c + 4)
_NO_SMT8 = {c: str(c) for c in range(8)}


@pytest.mark.parametrize(
    "siblings, allowed, expected",
    [
        (_SMT8, set(range(8)), 4),
        (_SMT8, {0, 1, 2, 3}, 4),  # one thread of each core
        (_SMT8, {0, 4}, 1),  # both threads of one core
        (_SMT8, {0, 4, 1}, 2),
        (_NO_SMT8, set(range(8)), 8),
        (_NO_SMT8, {2, 5, 7}, 3),
    ],
)
def test_physical_cores_count_distinct_sibling_sets_in_the_affinity_mask(
    monkeypatch, tmp_path, siblings, allowed, expected
):
    monkeypatch.setattr(bk.os, "sched_getaffinity", lambda pid: allowed, raising = False)
    assert bk._linux_physical_cores(_sysfs(tmp_path, siblings)) == expected


def test_physical_cores_is_none_without_topology_or_affinity(monkeypatch, tmp_path):
    monkeypatch.setattr(bk.os, "sched_getaffinity", lambda pid: {0, 1}, raising = False)
    assert bk._linux_physical_cores(_sysfs(tmp_path, {0: "0"})) is None  # cpu1 has no topology
    assert bk._linux_physical_cores(_sysfs(tmp_path / "e", {0: "", 1: "1"})) is None

    def _denied(pid):
        raise OSError("no")

    monkeypatch.setattr(bk.os, "sched_getaffinity", _denied, raising = False)
    assert bk._linux_physical_cores(_sysfs(tmp_path / "x", _NO_SMT8)) is None
    monkeypatch.delattr(bk.os, "sched_getaffinity", raising = False)
    assert bk._linux_physical_cores(_sysfs(tmp_path / "y", _NO_SMT8)) is None


def test_the_linux_default_counts_cores_then_smt_siblings(monkeypatch):
    # No SMT: every core, not half of them
    _host(monkeypatch, cpus = 8, physical = 8)
    assert bk._default_threads() == 8
    # A 4-CPU cpuset (2 cores x 2 SMT) on a 192-CPU host: its 4 CPUs, not 96
    _host(monkeypatch, cpus = 192, allowed = 4, physical = 2)
    assert bk._default_threads() == 4
    _host(monkeypatch, cpus = 192, allowed = 2, physical = 2)
    assert bk._default_threads() == 2
    # 2-way SMT with the whole machine: a many-core server keeps one thread per core
    _host(monkeypatch, cpus = 192, physical = 96)
    assert bk._default_threads() == 96
    # The quota still caps it, and an unreadable topology keeps half the CPUs
    _host(monkeypatch, cpus = 8, physical = 8, limit = 3)
    assert bk._default_threads() == 3
    _host(monkeypatch, cpus = 12, physical = None)
    assert bk._default_threads() == 6


def test_the_physical_count_is_never_read_off_linux(monkeypatch):
    _host(monkeypatch, cpus = 8, platform = "win32")
    monkeypatch.setattr(bk, "_linux_physical_cores", lambda *a, **k: pytest.fail("Linux only"))
    assert bk._default_threads() == 4


@pytest.mark.parametrize(
    "cpus, physical, limit, allowed, expected",
    [
        (32, 16, None, None, None),  # SMT: sd.cpp's own default is the same count
        (8, 8, None, None, 8),  # no SMT: sd.cpp's fallback would run 4
        (192, 2, None, 4, 2),  # cpuset: sd.cpp ignores the affinity mask
        (4, 4, None, None, None),  # sd.cpp keeps every CPU at 4 or fewer
        (32, 16, 6, None, 6),  # a tighter quota, as before
        (
            4,
            2,
            None,
            None,
            None,
        ),  # 4 vCPU / 2 cores, nothing restricting: sd.cpp's 4 threads measured faster than 2
        (2, 1, None, None, None),  # 2 vCPU / 1 core: sd.cpp's 2, not 1
        (4, 2, 2, None, 2),  # the same host under a 2-CPU quota: the quota still wins
        (4, 2, None, 3, 2),  # or under a 3-CPU taskset
    ],
)
def test_explicit_threads_passes_the_count_when_sd_cpp_would_pick_another(
    monkeypatch, cpus, physical, limit, allowed, expected
):
    _host(monkeypatch, cpus = cpus, physical = physical, limit = limit, allowed = allowed)
    monkeypatch.setattr(bk, "_apple_performance_cores", lambda: None)
    assert bk._explicit_threads() == expected


def test_the_real_host_topology_reads(monkeypatch):
    """On this test host: a positive count, never more than the CPUs this process may use."""
    if not hasattr(bk.os, "sched_getaffinity"):
        pytest.skip("no affinity API")
    count = bk._linux_physical_cores()
    if count is None:
        pytest.skip("no sysfs topology here")
    assert 1 <= count <= len(bk.os.sched_getaffinity(0))


def test_a_small_smt_host_runs_one_thread_per_logical_cpu(monkeypatch):
    # 4 cores x 2 SMT (desktop / laptop / cloud VM): all 8 logical CPUs
    _host(monkeypatch, cpus = 8, physical = 4)
    assert bk._default_threads() == 8
    _host(monkeypatch, cpus = 32, physical = 16)
    assert bk._default_threads() == 32
    _host(monkeypatch, cpus = 64, physical = 32)
    assert bk._default_threads() == 32
    # A cpuset holding 3 whole cores plus one lone sibling: the 7 CPUs it may use
    _host(monkeypatch, cpus = 192, allowed = 7, physical = 4)
    assert bk._default_threads() == 7
    _host(monkeypatch, cpus = 8, physical = 4, limit = 6)
    assert bk._default_threads() == 6


def test_smt_threads_skip_hybrid_hosts_the_opt_out_and_h3_video(monkeypatch):
    _host(monkeypatch, cpus = 20, physical = 14, hybrid = True)  # Intel 6P + 8E: one per core
    assert bk._default_threads() == 14
    _host(monkeypatch, cpus = 8, physical = 4)
    monkeypatch.setenv(bk.SD_CPP_CPU_SMT_ENV, "0")
    assert bk._default_threads() == 4
    monkeypatch.delenv(bk.SD_CPP_CPU_SMT_ENV)
    assert bk._default_threads(smt = False) == 4
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "3")
    assert bk._default_threads() == 3


def test_linux_hybrid_cpu_reads_the_cpu_atom_pmu(tmp_path):
    assert bk._linux_hybrid_cpu(str(tmp_path)) is False
    (tmp_path / "cpu_atom").mkdir()
    (tmp_path / "cpu_atom" / "cpus").write_text("\n")
    assert bk._linux_hybrid_cpu(str(tmp_path)) is False
    (tmp_path / "cpu_atom" / "cpus").write_text("12-19\n")
    assert bk._linux_hybrid_cpu(str(tmp_path)) is True


def _win_record(
    relationship,
    efficiency = 0,
    smt = True,
):
    import struct

    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX: Relationship, Size, Flags, EfficiencyClass, ...
    body = struct.pack("<BB20xH", 1 if smt else 0, efficiency, 1) + struct.pack("<QH6x", 0b11, 0)
    return struct.pack("<II", relationship, 8 + len(body)) + body


def test_parse_windows_cores_counts_cores_and_efficiency_classes():
    cache = _win_record(2)  # RelationCache records are skipped
    four = b"".join(_win_record(0) for _ in range(4)) + cache
    assert bk._parse_windows_cores(four) == (4, False)
    hybrid = b"".join(_win_record(0, 1) for _ in range(6)) + b"".join(
        _win_record(0, 0, smt = False) for _ in range(8)
    )
    assert bk._parse_windows_cores(hybrid) == (14, True)
    assert bk._parse_windows_cores(b"") is None
    assert bk._parse_windows_cores(cache) is None
    assert bk._parse_windows_cores(b"\0" * 16) is None  # a zero Size never loops


def test_windows_runs_the_logical_cpus_on_a_small_non_hybrid_smt_host(monkeypatch):
    _host(monkeypatch, cpus = 4, platform = "win32", windows = (2, False))
    assert bk._default_threads() == 4
    _host(monkeypatch, cpus = 16, platform = "win32", windows = (8, False))
    assert bk._default_threads() == 16
    # hybrid (P + E cores), no SMT, or many cores: one per core
    _host(monkeypatch, cpus = 20, platform = "win32", windows = (14, True))
    assert bk._default_threads() == 14
    _host(monkeypatch, cpus = 8, platform = "win32", windows = (8, False))
    assert bk._default_threads() == 8
    _host(monkeypatch, cpus = 64, platform = "win32", windows = (32, False))
    assert bk._default_threads() == 32
    # unreadable topology: half the logical CPUs, as before
    _host(monkeypatch, cpus = 16, platform = "win32", windows = None)
    assert bk._default_threads() == 8
    _host(monkeypatch, cpus = 4, platform = "win32", windows = (2, False))
    monkeypatch.setenv(bk.SD_CPP_CPU_SMT_ENV, "off")
    assert bk._default_threads() == 2
    monkeypatch.delenv(bk.SD_CPP_CPU_SMT_ENV)
    assert bk._default_threads(smt = False) == 2


def test_windows_cores_is_none_off_windows(monkeypatch):
    monkeypatch.setattr(bk.sys, "platform", "linux")
    assert bk._windows_cores() is None


def test_cpu_renders_size_torch_threads_like_sd_cpp(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_device as dd

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(name, raising = False)
    before = torch.get_num_threads()
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(bk, "_default_threads", lambda *a, **k: before + 3)
    assert dd.apply_cpu_render_threads("cuda:0") is None
    assert dd.apply_cpu_render_threads("mps") is None
    assert calls == []
    assert dd.apply_cpu_render_threads("cpu") == before + 3
    assert dd.apply_cpu_render_threads(torch.device("cpu")) == before + 3
    assert calls == [before + 3, before + 3]
    monkeypatch.setattr(bk, "_default_threads", lambda *a, **k: torch.get_num_threads())
    assert dd.apply_cpu_render_threads("cpu") is None
    monkeypatch.setattr(bk, "_default_threads", lambda *a, **k: 1 / 0)
    assert dd.apply_cpu_render_threads("cpu") is None


@pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_cpu_renders_keep_an_explicit_torch_thread_pool(monkeypatch, name):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_device as dd

    for other in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.delenv(other, raising = False)
    monkeypatch.setenv(name, "2")
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "8")
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    monkeypatch.setattr(bk, "_default_threads", lambda *a, **k: torch.get_num_threads() + 3)
    assert dd.apply_cpu_render_threads("cpu") is None
    assert calls == []
    monkeypatch.setenv(name, " ")
    assert dd.apply_cpu_render_threads("cpu") == torch.get_num_threads() + 3
