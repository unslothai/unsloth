# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Back-compat for the kept-install path against every marker shape that has shipped.

``UNSLOTH_PREBUILT_INFO.json`` is append-only across twelve shapes with no version field
and no migration, so an absent key is normal for anything older. These call the deciders
directly with the shapes real installs carry, rather than driving ``install_prebuilt``
with a hand-built two-key marker. Platforms are simulated through ``HostInfo``: that
covers the path decisions and payload tables, not macOS dyld.

Run natively on Windows too (the parity workflow's windows-latest row), where the loader
answers for real: ``_binary_image_runs`` sees an actual ``ERROR_BAD_EXE_FORMAT`` instead of
a stubbed ``run_capture``. Two things stay POSIX-only there and are skipped rather than
weakened: ``os.chmod`` cannot clear an execute bit Windows does not have, and
``os.access(X_OK)`` is true for any file that exists.
"""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

WINDOWS_HOST = os.name == "nt"
# os.access(path, os.X_OK) answers "does this exist" on Windows, so the executable=False
# trees below are indistinguishable from healthy ones there.
SKIP_X_OK = pytest.mark.skipif(
    WINDOWS_HOST,
    reason = "os.access(X_OK) is always true on Windows, so the guard is POSIX only",
)


def _windows_runnable_stub() -> bytes | None:
    """Bytes of a real .exe that still starts after being copied somewhere else.

    The keep path execs what it finds, so a Windows row needs a genuine PE. Copying
    python.exe alone loses python3xx.dll and dies 0xC0000135, hence a System32 tool
    whose imports are all KnownDLLs. Verified by running the copy, not assumed: an
    unverifiable stub skips the module instead of reporting the loader's refusal as
    a back-compat failure.
    """
    for name in ("where.exe", "hostname.exe"):
        source = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name
        if not source.is_file():
            continue
        with tempfile.TemporaryDirectory() as probe_dir:
            copy = Path(probe_dir) / "llama-server.exe"
            try:
                shutil.copyfile(source, copy)
                probe = subprocess.run([str(copy)], capture_output = True, timeout = 30)
            except Exception:
                continue
            # A loader failure returns rather than raises, so an unchecked run would
            # accept the very missing-DLL image this is picking a candidate to avoid,
            # and every healthy fixture after it would inherit it.
            if probe.returncode >= 0xC0000000:
                continue
        return source.read_bytes()
    return None


RUNNABLE_STUB = _windows_runnable_stub() if WINDOWS_HOST else None
if WINDOWS_HOST and RUNNABLE_STUB is None:
    pytest.skip(
        "no self-contained System32 .exe to stand in for llama-server",
        allow_module_level = True,
    )


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

HostInfo = ILP.HostInfo


def _host(**kw) -> HostInfo:
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = True,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
    )
    base.update(kw)
    return HostInfo(**base)


LINUX = _host()
WINDOWS = _host(system = "Windows", machine = "AMD64", is_windows = True, is_linux = False)
MACOS = _host(
    system = "Darwin",
    machine = "arm64",
    is_windows = False,
    is_linux = False,
    is_macos = True,
    is_x86_64 = False,
    is_arm64 = True,
    macos_version = (15, 5),
)

# The payload each platform's kinds share, as runtime_payload_health_groups computes it.
_SHARED_PAYLOAD = {
    "linux": [
        "libllama-common.so",
        "libllama.so",
        "libggml.so",
        "libggml-base.so",
        "libggml-cpu.so",
        "libmtmd.so",
    ],
    "windows": ["llama.dll"],
    "macos": ["libllama.dylib", "libggml.dylib", "libmtmd.dylib"],
}
_BACKEND_PAYLOAD = {
    ("linux", "cuda"): ["libggml-cuda.so"],
    ("linux", "rocm"): ["libggml-hip.so"],
    ("linux", "vulkan"): ["libggml-vulkan.so"],
    ("windows", "cuda"): ["ggml-cuda.dll"],
    ("windows", "rocm"): ["ggml-hip.dll"],
    ("windows", "vulkan"): ["ggml-vulkan.dll"],
}
_PUBLISHED_PAYLOAD = {
    "linux": ["llama-diffusion-gemma-visual-server"],
    "windows": ["llama-diffusion-gemma-visual-server.exe"],
    "macos": [],
}
_CUDART_TRIO = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")


def _platform_of(host: HostInfo) -> str:
    return "windows" if host.is_windows else "macos" if host.is_macos else "linux"


def build_install(
    tmp_path,
    *,
    host = LINUX,
    marker = "default",
    executable = True,
    runnable = True,
    runnable_root = None,
    payload = True,
    payload_backend = "unset",
    cudart = False,
    visual_server = True,
):
    """Write an install tree. ``marker`` is the literal object to serialise: ``"default"``
    writes a minimal current-shape marker, ``None`` writes no marker file at all (a source
    build), and a ``str`` is written verbatim (corrupt markers)."""
    install_dir = tmp_path / "llama.cpp"
    platform = _platform_of(host)
    runtime_dir = (
        install_dir / "build" / "bin" / "Release"
        if host.is_windows
        else install_dir / "build" / "bin"
    )
    runtime_dir.mkdir(parents = True)
    ext = ".exe" if host.is_windows else ""

    for path in (
        install_dir / f"llama-server{ext}",
        install_dir / f"llama-quantize{ext}",
        runtime_dir / f"llama-server{ext}",
        runtime_dir / f"llama-quantize{ext}",
    ):
        ok = (
            runnable
            if path.parent != install_dir
            else (runnable if runnable_root is None else runnable_root)
        )
        # The keep path execs these. The not-ok file has to be a bad image: ENOEXEC on
        # POSIX, a non-PE on Windows, where an empty file is a valid do-nothing program.
        if WINDOWS_HOST:
            path.write_bytes(RUNNABLE_STUB if ok else b"not a PE image\n")
        else:
            path.write_text("#!/bin/sh\nexit 0\n" if ok else "", encoding = "utf-8")
        os.chmod(path, 0o755 if executable else 0o644)
    (install_dir / "convert_hf_to_gguf.py").write_text("", encoding = "utf-8")
    (install_dir / "gguf-py").mkdir()

    marker_path = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    if marker == "default":
        marker = {"release_tag": "old-release", "tag": "old-upstream"}
    if marker is not None:
        marker_path.write_text(
            marker if isinstance(marker, str) else json.dumps(marker) + "\n",
            encoding = "utf-8",
        )

    if payload:
        for name in _SHARED_PAYLOAD[platform]:
            (runtime_dir / name).write_text("", encoding = "utf-8")
        if payload_backend != "unset":
            for name in _BACKEND_PAYLOAD.get((platform, payload_backend), ()):
                (runtime_dir / name).write_text("", encoding = "utf-8")
        if visual_server:
            for name in _PUBLISHED_PAYLOAD[platform]:
                (runtime_dir / name).write_text("", encoding = "utf-8")
        if cudart:
            for name in _CUDART_TRIO:
                (runtime_dir / name).write_text("", encoding = "utf-8")
    return install_dir


# The shipped marker shapes, oldest first, trimmed to the keys the keep path reads.

S1 = {  # 2026-03-25 #4562: no release_tag, no backend, no asset_sha256
    "requested_tag": "b6099",
    "tag": "b6099",
    "asset": "llama-b6099-bin-ubuntu-x64.tar.gz",
    "source": "upstream",
    "bundle_profile": "full",
    "runtime_line": None,
    "coverage_class": None,
    "prebuilt_fallback_used": False,
    "installed_at_utc": "2026-03-26T04:11:07Z",
}
S2 = {  # 2026-04-01 #4741: release_tag + fingerprint arrive
    **S1,
    "release_tag": "b6210",
    "published_repo": "unslothai/llama.cpp",
    "asset": "app-b6210-linux-x64-cuda12.tar.gz",
    "asset_sha256": "3f" * 32,
    "source": "published",
    "runtime_line": "cuda12",
    "install_fingerprint": "aa" * 32,
}
S5 = {**S2, "force_cpu": False}  # 2026-07-20 #7228
S6 = {
    **S5,
    "llama_backend": "vulkan",  # 2026-07-27 #7373
    "asset": "llama-b7001-bin-ubuntu-vulkan-x64.tar.gz",
    "runtime_line": None,
}
S7 = {**S5, "ggml_tree": "b7440"}  # 2026-08-04 #7817
S8 = {
    **S7,
    "rocm_gfx": "gfx1151",  # 2026-08-08 #8050
    "asset": "app-b9001-linux-x64-rocm-gfx110X.tar.gz",
    "runtime_line": None,
}
S9 = {**S7, "backend": "cuda", "backend_request": "auto"}  # 2026-08-13 #8520
S10 = {**S9, "gfx_target": None, "mapped_targets": []}  # 2026-08-13 #7670
S11 = {**S10, "supported_sms": ["80", "86", "89", "90"]}  # 2026-08-18 #8841 == main
S12 = {**S11, "runtime_asset": None}  # this PR

# A real marker, produced by actually running studio/install_llama_prebuilt.py.
S12_REAL = {
    "requested_tag": "latest",
    "tag": "b10698",
    "release_tag": "b10698-mix-67dfc8b",
    "published_repo": "unslothai/llama.cpp",
    "asset": "app-b10698-mix-67dfc8b-linux-x64-cuda13-newer.tar.gz",
    "force_cpu": False,
    "llama_backend": None,
    "backend": "cuda",
    "backend_request": "auto",
    "asset_sha256": "d4" * 32,
    "runtime_asset": None,
    "source": "published",
    "ggml_tree": "0034c6eb",
    "bundle_profile": "cuda13-newer",
    "runtime_line": "cuda13",
    "coverage_class": "newer",
    "gfx_target": None,
    "mapped_targets": [],
    "supported_sms": ["86", "89", "90", "100", "103", "120"],
    "install_fingerprint": "36" * 32,
    "prebuilt_fallback_used": False,
    "installed_at_utc": "2026-08-31T06:40:37Z",
}

ALL_SHAPES = [
    ("S1-upstream-cpu", S1, None),
    ("S2-published-cuda", S2, "cuda"),
    ("S5-force-cpu-field", S5, "cuda"),
    ("S6-legacy-vulkan", S6, "vulkan"),
    ("S7-ggml-tree", S7, "cuda"),
    ("S8-rocm-gfx", S8, "rocm"),
    ("S9-backend-key", S9, "cuda"),
    ("S10-gfx-target", S10, "cuda"),
    ("S11-main-today", S11, "cuda"),
    ("S12-this-pr", S12, "cuda"),
    ("S12-real-install", S12_REAL, "cuda"),
]


@pytest.mark.parametrize(("name", "marker", "backend"), ALL_SHAPES, ids = [s[0] for s in ALL_SHAPES])
def test_every_shipped_marker_shape_keeps_a_complete_linux_install(tmp_path, name, marker, backend):
    """Kept whatever release wrote the marker. S1-S8 have no ``backend`` key, so it comes back
    out of the asset name; S1's upstream CPU asset names none either and must fail open."""
    install_dir = build_install(tmp_path, marker = marker, payload_backend = backend)
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True
    assert ILP._existing_install_runs(install_dir, LINUX) is True


@pytest.mark.parametrize(("name", "marker", "backend"), ALL_SHAPES, ids = [s[0] for s in ALL_SHAPES])
def test_no_shipped_marker_shape_is_kept_once_its_backend_payload_is_gutted(
    tmp_path, name, marker, backend
):
    """Deleting the shared payload must be caught for every shape, old or new."""
    install_dir = build_install(tmp_path, marker = marker, payload_backend = backend)
    runtime_dir = install_dir / "build" / "bin"
    for lib in _SHARED_PAYLOAD["linux"]:
        (runtime_dir / lib).unlink()
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is False
    assert ILP._existing_install_runs(install_dir, LINUX) is False


def test_a_pre_runtime_asset_windows_cuda_install_is_not_asked_for_the_cudart_trio(tmp_path):
    """S11 and older cannot record a pairing, so demanding one would loop forever: every
    Windows CUDA install today predates ``runtime_asset`` and would be rejected on every run."""
    install_dir = build_install(
        tmp_path,
        host = WINDOWS,
        marker = S11,
        payload_backend = "cuda",
        cudart = False,
    )
    assert ILP._kept_install_payload_is_healthy(install_dir, WINDOWS) is True


def test_a_paired_windows_cuda_install_still_owes_its_cudart_trio(tmp_path):
    """Once the marker names the pairing, the trio becomes required."""
    paired = {**S12, "runtime_asset": "cudart-llama-bin-win-cuda-13.0-x64.zip"}
    gutted = build_install(
        tmp_path / "gutted",
        host = WINDOWS,
        marker = paired,
        payload_backend = "cuda",
        cudart = False,
    )
    intact = build_install(
        tmp_path / "intact",
        host = WINDOWS,
        marker = paired,
        payload_backend = "cuda",
        cudart = True,
    )
    assert ILP._kept_install_payload_is_healthy(gutted, WINDOWS) is False
    assert ILP._kept_install_payload_is_healthy(intact, WINDOWS) is True


def test_a_source_build_is_never_kept_because_it_has_no_marker(tmp_path):
    """The keep path is for prebuilts only: ``confirm_install_tree`` requires the marker, so a
    source build never reaches the payload check and falls through to the fallback as before."""
    install_dir = build_install(tmp_path, marker = None, payload_backend = "cuda")
    assert (install_dir / "llama-server").exists()
    assert not (install_dir / "UNSLOTH_PREBUILT_INFO.json").exists()
    assert ILP._install_tree_is_usable(install_dir, LINUX) is False
    assert ILP._existing_install_runs(install_dir, LINUX) is False


@pytest.mark.parametrize(
    "corrupt",
    ["not json", "", "[]", "null", '"cuda"', "123", '{"release_tag": "b1"', "﻿{}", "{}"],
    ids = [
        "garbage",
        "empty",
        "list",
        "null",
        "string",
        "number",
        "truncated",
        "bom",
        "empty-object",
    ],
)
def test_a_corrupt_marker_is_still_kept_but_owes_the_whole_platform_payload(tmp_path, corrupt):
    """An unreadable marker cannot name a backend, so it owes every kind's shared set. Unlike a
    missing one it still satisfies ``confirm_install_tree``, so the tree stays eligible and is
    judged on what is on disk."""
    install_dir = build_install(tmp_path, marker = corrupt, payload_backend = "cuda")
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True
    assert ILP._existing_install_runs(install_dir, LINUX) is True

    for lib in _SHARED_PAYLOAD["linux"]:
        (install_dir / "build" / "bin" / lib).unlink()
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is False


def test_a_marker_from_a_newer_unsloth_is_ignored_key_by_key(tmp_path):
    """Forwards compatibility: unknown keys must not disturb the decision."""
    future = {**S12, "install_generation": 3, "unknown_future_field": {"a": [1, 2]}}
    install_dir = build_install(tmp_path, marker = future, payload_backend = "cuda")
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True
    assert ILP._existing_install_runs(install_dir, LINUX) is True


def test_a_backend_this_unsloth_does_not_know_falls_back_to_the_asset_name(tmp_path):
    """An unrecognised ``backend`` is not the end of the derivation: ``marker_backend`` tries
    the recorded backend, then the install kind, then the asset name. Asserted both ways so the
    fallback order is pinned."""
    named = {**S12, "backend": "sycl"}  # asset is ...linux-x64-cuda12.tar.gz
    assert ILP.marker_backend(named) == "cuda"

    without_cuda_lib = build_install(tmp_path / "a", marker = named, payload_backend = None)
    assert ILP._kept_install_payload_is_healthy(without_cuda_lib, LINUX) is False

    with_cuda_lib = build_install(tmp_path / "b", marker = named, payload_backend = "cuda")
    assert ILP._kept_install_payload_is_healthy(with_cuda_lib, LINUX) is True


def test_a_backend_no_source_can_name_falls_open_to_the_shared_payload(tmp_path):
    """When neither the backend nor the asset name resolves, do not reject a good tree."""
    opaque = {**S12, "backend": "sycl", "asset": "bundle.tar.gz"}
    assert ILP.marker_backend(opaque) is None

    install_dir = build_install(tmp_path, marker = opaque, payload_backend = None)
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True


def test_the_macos_keep_path_requires_the_dylibs(tmp_path):
    """macOS reaches the same decider and owes its own shared payload. Simulated through
    ``HostInfo``: this covers the path and the payload table, not dyld."""
    install_dir = build_install(tmp_path / "ok", host = MACOS, marker = S12)
    assert ILP._kept_install_payload_is_healthy(install_dir, MACOS) is True

    gutted = build_install(tmp_path / "gutted", host = MACOS, marker = S12)
    (gutted / "build" / "bin" / "libggml.dylib").unlink()
    assert ILP._kept_install_payload_is_healthy(gutted, MACOS) is False


def test_a_legacy_published_vulkan_install_without_the_visual_server_is_refused(tmp_path):
    """``source`` has been recorded since the first shape, so this reaches old installs: a
    published Vulkan install predating ``ensure_diffusion_visual_server`` is held to a file it
    never had. The one case where an old install is rebuilt rather than kept."""
    install_dir = build_install(
        tmp_path,
        marker = S6 | {"source": "published"},
        payload_backend = "vulkan",
        visual_server = False,
    )
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is False


def test_a_binary_that_dies_on_sigsegv_is_not_a_working_install(tmp_path, monkeypatch):
    """A crashing image must be rejected, not treated as a successful probe."""
    install_dir = build_install(tmp_path, marker = S12, payload_backend = "cuda")
    monkeypatch.setattr(
        ILP,
        "run_capture",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], -signal.SIGSEGV, "", ""),
    )
    assert ILP._existing_install_runs(install_dir, LINUX) is False


def test_a_binary_that_hangs_is_treated_as_healthy(tmp_path, monkeypatch):
    """Pins the deliberate fail-open: a timeout must not spend a source build."""
    install_dir = build_install(tmp_path, marker = S12, payload_backend = "cuda")

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(a[0] if a else ["llama-server"], 60)

    monkeypatch.setattr(ILP, "run_capture", hang)
    assert ILP._existing_install_runs(install_dir, LINUX) is True


def test_a_windows_loader_failure_is_rejected_even_though_it_exits_zero_ish(tmp_path, monkeypatch):
    """0xC0000135 is a positive exit code, not a signal. Simulated: no Windows here."""
    install_dir = build_install(
        tmp_path,
        host = WINDOWS,
        marker = S12,
        payload_backend = "cuda",
    )
    monkeypatch.setattr(
        ILP,
        "run_capture",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0xC0000135, "", ""),
    )
    assert ILP._existing_install_runs(install_dir, WINDOWS) is False


@SKIP_X_OK
def test_a_non_executable_tree_fails_the_same_gate_setup_sh_uses(tmp_path):
    """setup.sh reuses on ``[ -x build/bin/llama-server ]``; the keep path must agree."""
    install_dir = build_install(
        tmp_path,
        marker = S12,
        payload_backend = "cuda",
        executable = False,
    )
    assert ILP._existing_install_runs(install_dir, LINUX) is False


def test_a_rotten_root_entrypoint_is_caught_even_when_build_bin_is_fine(tmp_path):
    """Inference launches the root copy first, so it cannot be excused by build/bin."""
    install_dir = build_install(
        tmp_path,
        marker = S12,
        payload_backend = "cuda",
        runnable_root = False,
    )
    assert ILP._existing_install_runs(install_dir, LINUX) is False


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        (S1, "auto"),
        (S2, "auto"),
        (S5 | {"force_cpu": True}, "cpu"),
        (S6, "vulkan"),
        (S9, "auto"),
        (S9 | {"backend_request": "cuda"}, "cuda"),
        ({}, "auto"),
    ],
    ids = ["s1", "s2", "s5-forced-cpu", "s6-legacy-vulkan", "s9-auto", "s9-pinned", "empty"],
)
def test_the_stored_backend_choice_reads_the_same_from_every_shape(tmp_path, marker, expected):
    """The keep path gates on this, so old shapes must not read as a pinned choice."""
    install_dir = build_install(tmp_path, marker = marker)
    assert ILP.persisted_backend_request(install_dir) == expected


# ---------------------------------------------------------------------------
ARM64_LINUX = _host(machine = "aarch64", is_x86_64 = False, is_arm64 = True)
MACOS_X64 = _host(
    system = "Darwin",
    machine = "x86_64",
    is_windows = False,
    is_linux = False,
    is_macos = True,
    is_x86_64 = True,
    is_arm64 = False,
    macos_version = (14, 6),
)
WINDOWS_ARM64 = _host(
    system = "Windows",
    machine = "ARM64",
    is_windows = True,
    is_linux = False,
    is_x86_64 = False,
    is_arm64 = True,
)
# WSL reports itself as Linux; these flags are what a WSL2 ROCDXG host carries.
WSL_ROCM = _host(has_rocm = True, rocm_gfx_target = "gfx1151")
LINUX_NVIDIA = _host(
    compute_caps = ["10.0"],
    has_physical_nvidia = True,
    has_usable_nvidia = True,
)
LINUX_ROCM = _host(has_rocm = True, rocm_gfx_target = "gfx1100")
WINDOWS_NVIDIA = _host(
    system = "Windows",
    machine = "AMD64",
    is_windows = True,
    is_linux = False,
    compute_caps = ["8.9"],
    has_physical_nvidia = True,
    has_usable_nvidia = True,
)
WINDOWS_ROCM = _host(
    system = "Windows",
    machine = "AMD64",
    is_windows = True,
    is_linux = False,
    has_rocm = True,
    rocm_gfx_target = "gfx1151",
)

MATRIX = [
    ("linux-nvidia", LINUX_NVIDIA, "cuda", "cuda"),
    ("linux-arm64-nvidia", ARM64_LINUX, "cuda", "cuda"),
    ("linux-amd", LINUX_ROCM, "rocm", "rocm"),
    ("linux-cpu", LINUX, "cpu", None),
    ("linux-arm64-cpu", ARM64_LINUX, "cpu", None),
    ("linux-vulkan", LINUX, "vulkan", "vulkan"),
    ("wsl-amd", WSL_ROCM, "rocm", "rocm"),
    ("wsl-cpu", WSL_ROCM, "cpu", None),
    ("windows-nvidia", WINDOWS_NVIDIA, "cuda", "cuda"),
    ("windows-amd", WINDOWS_ROCM, "rocm", "rocm"),
    ("windows-cpu", WINDOWS, "cpu", None),
    ("windows-arm64-cpu", WINDOWS_ARM64, "cpu", None),
    ("windows-vulkan", WINDOWS, "vulkan", "vulkan"),
    ("macos-arm64", MACOS, "metal", None),
    ("macos-x64", MACOS_X64, "metal", None),
]


@pytest.mark.parametrize(
    ("cell", "host", "backend", "payload_backend"),
    MATRIX,
    ids = [m[0] for m in MATRIX],
)
def test_a_complete_install_is_kept_in_every_os_and_accelerator_cell(
    tmp_path, cell, host, backend, payload_backend
):
    marker = {**S12, "backend": backend, "asset": f"app-b1-{cell}.tar.gz"}
    install_dir = build_install(
        tmp_path,
        host = host,
        marker = marker,
        payload_backend = payload_backend,
    )
    assert ILP._kept_install_payload_is_healthy(install_dir, host) is True, cell
    assert ILP._existing_install_runs(install_dir, host) is True, cell


@pytest.mark.parametrize(
    ("cell", "host", "backend", "payload_backend"),
    MATRIX,
    ids = [m[0] for m in MATRIX],
)
def test_a_gutted_install_is_refused_in_every_os_and_accelerator_cell(
    tmp_path, cell, host, backend, payload_backend
):
    """Every cell must fail closed, or the keep path hands back a broken tree."""
    marker = {**S12, "backend": backend, "asset": f"app-b1-{cell}.tar.gz"}
    install_dir = build_install(
        tmp_path,
        host = host,
        marker = marker,
        payload_backend = payload_backend,
    )
    runtime_dir = (
        install_dir / "build" / "bin" / "Release"
        if host.is_windows
        else install_dir / "build" / "bin"
    )
    (runtime_dir / _SHARED_PAYLOAD[_platform_of(host)][0]).unlink()
    assert ILP._kept_install_payload_is_healthy(install_dir, host) is False, cell


@pytest.mark.parametrize(
    ("cell", "host", "backend"),
    [(c, h, b) for c, h, b, p in MATRIX if p is not None],
    ids = [m[0] for m in MATRIX if m[3] is not None],
)
def test_an_accelerator_install_missing_its_own_backend_library_is_refused(
    tmp_path, cell, host, backend
):
    """The shared payload alone must not be enough for a CUDA/ROCm/Vulkan tree. The partial
    extraction case: everything generic is present and only the accelerator library is gone, so
    the binaries start and fail once a model loads."""
    marker = {**S12, "backend": backend, "asset": f"app-b1-{cell}.tar.gz"}
    install_dir = build_install(
        tmp_path,
        host = host,
        marker = marker,
        payload_backend = None,
    )
    assert ILP._kept_install_payload_is_healthy(install_dir, host) is False, cell


def test_wsl_is_treated_exactly_like_linux_by_the_keep_path(tmp_path):
    """Pin the assumption rather than leaving it implicit: ``HostInfo`` has no WSL flag and the
    WSL2 ROCDXG handling is upstream of here, so a WSL install is judged by the Linux tables."""
    marker = {**S12, "backend": "rocm", "asset": "app-b1-linux-x64-rocm-gfx1151.tar.gz"}
    for host in (LINUX_ROCM, WSL_ROCM):
        ok = build_install(
            tmp_path / f"ok-{host.rocm_gfx_target}",
            host = host,
            marker = marker,
            payload_backend = "rocm",
        )
        gutted = build_install(
            tmp_path / f"gutted-{host.rocm_gfx_target}",
            host = host,
            marker = marker,
            payload_backend = None,
        )
        assert ILP._kept_install_payload_is_healthy(ok, host) is True
        assert ILP._kept_install_payload_is_healthy(gutted, host) is False


def test_a_marker_naming_another_platforms_backend_falls_open(tmp_path):
    """A tree carried between machines must not be judged by the wrong table: "metal" on a Linux
    host filters to no linux kind, so the decider falls back to every kind this platform has
    rather than refusing a payload that is actually complete."""
    marker = {**S12, "backend": "metal", "asset": "app-b1-macos-arm64.tar.gz"}
    install_dir = build_install(tmp_path, host = LINUX, marker = marker, payload_backend = None)
    assert ILP.marker_backend(marker) == "metal"
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True


def test_an_install_whose_keys_were_backfilled_onto_an_old_marker_is_kept(tmp_path):
    """Real markers are not the clean shapes: ``sync_marker_selection`` grafts, so an install
    made at S2 and reused since carries an S2 base with later keys added, which is where a
    reader would expect the shapes to disagree."""
    grafted = {
        **S2,
        "backend": "cuda",
        "backend_request": "auto",
        "ggml_tree": "b9415",
        "supported_sms": ["80", "86"],
        "runtime_asset": None,
    }
    install_dir = build_install(tmp_path, marker = grafted, payload_backend = "cuda")
    assert ILP._kept_install_payload_is_healthy(install_dir, LINUX) is True
    assert ILP._existing_install_runs(install_dir, LINUX) is True


def _transient_listing_failure(monkeypatch, host = LINUX):
    """Make the release listing fail the way a flaky network does."""
    import urllib.error

    def boom(*args, **kwargs):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(ILP, "_fork_manifest_release_plans", boom)
    monkeypatch.setattr(ILP, "detect_host", lambda *a, **k: host)
    monkeypatch.setattr(ILP, "collect_system_report", lambda *a, **k: "report")


def test_a_marker_from_a_newer_unsloth_refuses_rather_than_keeping(tmp_path, monkeypatch):
    """A backend choice this build cannot read must stop the install, not be kept:
    ``effective_backend_request`` raises ``UnknownBackendRequest`` and the handler exits
    ``EXIT_ERROR`` before the keep branch runs. That ordering was unasserted."""
    _transient_listing_failure(monkeypatch)
    install_dir = build_install(
        tmp_path,
        marker = {**S12, "backend": "cuda", "backend_request": "sycl"},
        payload_backend = "cuda",
    )
    with pytest.raises(SystemExit) as caught:
        ILP.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")
    assert caught.value.code == ILP.EXIT_ERROR
    assert (install_dir / "llama-server").exists()


def test_a_transient_failure_keeps_each_shipped_shape_and_returns_exit_zero(tmp_path, monkeypatch):
    """End to end: the exit code setup.sh and setup.ps1 branch on, per shape."""
    _transient_listing_failure(monkeypatch)
    for name, marker, backend in ALL_SHAPES:
        install_dir = build_install(
            tmp_path / name,
            marker = marker,
            payload_backend = backend,
        )
        # Returns rather than raising SystemExit: main() turns that into exit 0.
        ILP.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")
        assert (install_dir / "llama-server").exists(), name


def test_a_transient_failure_still_falls_back_when_the_tree_is_not_runnable(tmp_path, monkeypatch):
    """The other half: a broken tree must still reach the source-build fallback. Exit 2 is what
    tells setup.sh it may build from source; swallowing it would leave a user with a
    half-deleted install told everything was fine."""
    _transient_listing_failure(monkeypatch)
    install_dir = build_install(tmp_path, marker = S12, payload_backend = None)
    with pytest.raises(SystemExit) as caught:
        ILP.install_prebuilt(install_dir, "latest", "unslothai/llama.cpp", "")
    assert caught.value.code == ILP.EXIT_FALLBACK


def test_an_explicit_version_request_is_never_answered_with_the_old_install(tmp_path, monkeypatch):
    """Asking for a specific release and getting the one already there is a lie."""
    _transient_listing_failure(monkeypatch)
    install_dir = build_install(tmp_path, marker = S12, payload_backend = "cuda")
    with pytest.raises(SystemExit) as caught:
        ILP.install_prebuilt(install_dir, "b9999", "unslothai/llama.cpp", "")
    assert caught.value.code in (ILP.EXIT_FALLBACK, ILP.EXIT_ERROR)
