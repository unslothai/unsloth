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
):
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    monkeypatch.setattr(bk.sys, "platform", platform)
    monkeypatch.setattr(bk.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(bk, "_cgroup_cpu_limit", lambda: limit)


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
