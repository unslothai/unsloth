# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Locating llama-server binaries, loader paths, ROCm libraries and runtime probes."""

import contextlib
import os
import re
import shutil
import sys
import threading

from pathlib import Path
from typing import Optional


def _wsl_system_rocm_lib_dirs() -> "list[str]":
    """System ROCm lib dir(s) to load before a prebuilt's bundled HIP, on WSL.

    The bundled bare-metal HIP can't drive WSL's /dev/dxg and segfaults on the
    first GPU call; the system ROCm libs (libamdhip64 + librocdxg) can, while
    the bundle still supplies libggml-hip / librocblas (gfx1151 kernels).
    Mirrors install_llama_prebuilt._wsl_system_rocm_lib_dirs so a prebuilt that
    passed install validation runs the same at serve time. No-op off a ROCDXG
    WSL host (needs /dev/dxg, "microsoft" /proc/version, librocdxg in /opt/rocm).
    """
    try:
        if not os.path.exists("/dev/dxg"):
            return []
        with open("/proc/version", encoding = "utf-8", errors = "replace") as fh:
            if "microsoft" not in fh.read().lower():
                return []
    except (OSError, UnicodeDecodeError):
        return []
    out: "list[str]" = []
    for d in ("/opt/rocm/lib", "/opt/rocm/lib64"):
        if os.path.exists(os.path.join(d, "librocdxg.so")) or os.path.exists(
            os.path.join(d, "librocdxg.so.1")
        ):
            out.append(d)
    return out


def _bundled_hip_present(binary_dir: str) -> bool:
    """True when a prebuilt bundle ships its own HIP backend library."""
    if not binary_dir:
        return False
    try:
        # Glob the version suffix (libggml-hip.so, .so.0, .so.0.11.1) the same
        # way the installer's runtime health check matches libggml-hip.so*.
        return any(Path(str(binary_dir)).glob("libggml-hip.so*"))
    except OSError:
        return False


def _native_linux_system_rocm_lib_dirs(binary_dir: str = "") -> "list[str]":
    """System ROCm lib dir(s) to prepend before a prebuilt's bundled HIP, on native Linux.

    The bundled bare-metal HIP runtime can mismatch the host amdkfd driver and crash
    in hsa_init(); prepending the whole system ROCm lib dir loads a driver-matched,
    version-consistent stack (libhsa-runtime64 / libamdhip64 / librocblas) ahead of it.
    The whole dir is deliberate: mixing the bundle's rocBLAS with a different-version
    system HIP/ROCR risks missing symbols, and when it does load_model retries once
    with the bundle only. UNSLOTH_LLAMA_NO_SYSTEM_ROCM=1 skips the prepend up front,
    for a host whose system ROCm lacks this arch. No-op on WSL / non-Linux.
    """
    if os.environ.get("UNSLOTH_LLAMA_NO_SYSTEM_ROCM") == "1":
        return []
    if sys.platform != "linux" or os.path.exists("/dev/dxg"):
        return []
    if not os.path.exists("/dev/kfd"):
        return []
    if not _bundled_hip_present(binary_dir):
        return []
    # Env-configured ROCm root first; /opt/rocm only as a fallback so a stale
    # /opt/rocm doesn't shadow the driver-matching install these vars point at.
    candidates = []
    for var in ("HIP_PATH", "HIP_PATH_57", "ROCM_PATH"):
        val = os.environ.get(var)
        if val:
            candidates.append(val)
    candidates.append("/opt/rocm")
    out: "list[str]" = []
    seen: "set[str]" = set()
    for base in candidates:
        for lib_sub in ("lib", "lib64"):
            d = os.path.join(base, lib_sub)
            if d in seen:
                continue
            seen.add(d)
            if os.path.exists(os.path.join(d, "libhsa-runtime64.so")) or os.path.exists(
                os.path.join(d, "libhsa-runtime64.so.1")
            ):
                out.append(d)
                # ROCm keeps LLVM's versioned runtime under <root>/lib/llvm, so a
                # lib64 host still finds it under lib. Probe both and keep them
                # ahead of the bundle, else system libamd_comgr binds to the
                # bundle's incompatible libLLVM.so.*.
                for _sub in (lib_sub, "lib"):
                    llvm_lib = os.path.join(base, _sub, "llvm", "lib")
                    if llvm_lib not in seen and os.path.isdir(llvm_lib):
                        seen.add(llvm_lib)
                        out.append(llvm_lib)
    return out


# Host RAM to leave free on an integrated GPU, matching llama.cpp's own --fit
# margin (default 1024 MiB per device). ggml reports an iGPU's "VRAM" as shared
# system RAM, so hold back the same margin rather than inventing a larger one.
_IGPU_HOST_RESERVE_MIB = 1024
_HOST_RAM_HEADROOM_MIB = 2048
# What the advice refuses to take from the host: this many GB or this share of the
# machine, whichever is larger. The carve-out comes out of the RAM the OS sees, so
# ignoring the host turns a slow load into an unusable desktop.
_CARVEOUT_ADVICE_MIN_HOST_GB = 8
# A fifth, not a quarter: the 128 GB Strix Halo firmware offers 96 GB and runs it. A
# quarter would cap at 95.83 GB and rule out a setting we measured.
_CARVEOUT_ADVICE_HOST_FRACTION = 0.20
# A driver reports the pool it kept, not the firmware menu number (95.83 against a
# 96.00 GB setting here). Without slack a model between the two earns the rung the
# user is ALREADY on, so the advice reads "allocate 96 GB" to someone running 96 GB.
# Half a GB is well under the gap between rungs.
_CARVEOUT_NOMINAL_SLACK_GB = 0.5
# Appended to whichever shortfall warning an oversized non-pageable launch produced,
# after _page_an_oversized_unmapped_load rewrote the mode. One string, so the three
# call sites cannot describe the same override differently.
_PAGEABLE_OVERRIDE_NOTE = (
    " This model was set to load without memory mapping, which reads every byte into "
    "RAM up front and cannot complete at this size, so Unsloth loaded it with memory "
    "mapping instead."
)
# What memory_warning says when the text-only fallback removed the CPU-pinned projector
# whose bytes were what made the load oversized, on a launch that had already had its
# unmapped load mode overridden for that shortfall. The override is REPORTED and not
# taken back: the argv was rewritten and spawned, and force_pageable_load rewrote the
# shared child env in place, so the running server is memory-mapped whatever this says.
_PAGEABLE_OVERRIDE_TEXT_ONLY_NOTICE = (
    "This model was set to load without memory mapping, and with its vision projector "
    "in system RAM it was too large to load that way, so Unsloth loaded it with memory "
    "mapping instead. The projector then failed to start and this session is text-only, "
    "which the weights alone would have fit. The running server keeps the memory "
    "mapping it started with; load the model again to run it as you asked."
)
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def _apply_igpu_host_reserve_mib(free_mib: int, is_igpu: bool) -> int:
    """Reserve host headroom on an integrated (shared-memory) Vulkan GPU.

    An iGPU's reported free "VRAM" is really free system RAM, so sizing
    context/offload against all of it would push the host into swap or the OOM
    killer. Leave the same margin llama.cpp's --fit uses. ``is_igpu`` comes from
    ggml's device type, so a discrete card is never touched; only ever reduces.
    """
    if not is_igpu:
        return free_mib
    return max(0, free_mib - _IGPU_HOST_RESERVE_MIB)


def _resolve_llama_binary(binary: str, *, template_only: bool = False) -> Path:
    """Resolve a managed symlink or shell entrypoint to the real server.

    Follows a chain rather than one hop. A wrapper whose target is another
    wrapper used to resolve to the intermediate script, which on macOS is the
    same defect this is here to avoid (SIP drops DYLD_* through the shell) and
    also gives _llama_lib_dir the wrong directory, so the loader path would
    point at the wrapper's folder instead of the one holding the dylibs.
    Bounded, and stops on a repeat, so a wrapper pair pointing at each other
    cannot spin.

    ``template_only`` stops at the first link that is not the installer's own
    template. Two callers, two different questions. Finding the dylibs wants
    the end of the chain whatever it is made of, because that is where they
    sit. Choosing what to LAUNCH must not step over somebody's own script:
    checking only the outer entrypoint meant an installer-shaped symlink whose
    target was a hand-written wrapper had that wrapper's exports skipped, even
    though running the symlink directly would have executed them.
    """
    resolved = Path(binary).resolve()
    seen = {resolved}
    for _ in range(8):
        try:
            with open(resolved, "rb") as _f:
                _head = _f.read(256)
        except OSError:
            break
        if not _head.startswith(b"#!"):
            break
        _m = re.search(r'exec "\$\(dirname "\$0"\)/([^"]+)"', _head.decode("utf-8", "ignore"))
        if not _m:
            break
        if template_only and not _is_installer_entrypoint(str(resolved)):
            break
        nxt = (resolved.parent / _m.group(1)).resolve()
        if nxt in seen:
            break
        seen.add(nxt)
        resolved = nxt
    return resolved


def _llama_lib_dir(binary: str) -> Path:
    return _resolve_llama_binary(binary).parent


# install_llama_prebuilt.write_exec_wrapper emits exactly this and nothing else:
# a shebang, one exec line, a trailing newline. Anything more is somebody's own
# script, whose extra lines are the whole reason not to skip past it.
_INSTALLER_WRAPPER_RE = re.compile(
    r"\A#!/bin/sh\n" r'exec "\$\(dirname "\$0"\)/[^"\n]+" "\$@"\n?\Z'
)


def _is_installer_entrypoint(binary: str) -> bool:
    """Is ``binary`` an entrypoint the installer itself wrote?

    Symlinks and real executables qualify: neither carries setup that could be
    lost by launching the target instead. A shell script only qualifies when it
    matches the installer's template exactly, so a user's wrapper that exports
    variables before its exec line is launched as written.

    Kept separate from _resolve_llama_binary, which must keep resolving ANY
    wrapper: _llama_lib_dir needs the directory holding the dylibs, and that is
    the target's directory whoever wrote the wrapper.
    """
    try:
        path = Path(binary)
        if path.is_symlink():
            return True
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return True
    if not head.startswith(b"#!"):
        return True
    return bool(_INSTALLER_WRAPPER_RE.match(head.decode("utf-8", "ignore")))


def _loader_path_var() -> str:
    """The platform's shared-library search env var.

    dyld ignores LD_LIBRARY_PATH entirely, so a macOS child launched with only
    that variable set has no search path at all. Same mapping as the sibling
    engines (sd_cpp_engine._lib_path_var, stt_ggml_sidecar) and as the
    installer's own staged-binary validation.
    """
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if sys.platform == "win32":
        return "PATH"
    return "LD_LIBRARY_PATH"


def _prepend_loader_dir(existing: str, lib_dir: str) -> str:
    """``lib_dir`` first in a loader search path, with no repeats.

    Deduplicates on the normalised spelling, so an inherited "/x/bin/" or
    "/x/./bin" does not survive alongside the "/x/bin" being prepended and the
    result is stable however many times it is fed back in. Purely textual:
    realpath would touch the filesystem, and a dead network mount on an
    inherited entry must not stall a model load. Entries keep the spelling they
    arrived with, since that is what the loader is given.
    """
    out: list[str] = []
    seen: set[str] = set()
    for entry in (lib_dir, *existing.split(os.pathsep)):
        if not entry:
            continue
        key = os.path.normcase(os.path.normpath(entry))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return os.pathsep.join(out)


_CPU_RUNTIME_OWNER_FILE = "UNSLOTH_OWNER_PID"

# GPU backends the staged CPU-only runtime must not carry over. Module level: compiled once.
# Vulkan rows for the placement decision in progress. THREAD-LOCAL: system-info polling
# reaches the same probe from other threads and wants live VRAM, and a shared dict let
# such a poll seed the load's memo with minutes-old rows.
_LOAD_PROBE_STATE = threading.local()


def _load_probe_memo_armed() -> bool:
    return getattr(_LOAD_PROBE_STATE, "armed", False)


def _load_probe_memo_get(kind, binary):
    rows = getattr(_LOAD_PROBE_STATE, "rows", None)
    return rows.get((kind, binary)) if rows else None


def _load_probe_memo_put(kind, binary, rows) -> None:
    if not _load_probe_memo_armed():
        return
    store = getattr(_LOAD_PROBE_STATE, "rows", None)
    if store is None:
        store = _LOAD_PROBE_STATE.rows = {}
    store[(kind, binary)] = rows


def _arm_load_probe_memo() -> None:
    """Start memoising the launch probes for the placement decision, on this thread
    only. Keyed by kind: the placement asks two different subprocesses, the device
    enumeration and, for a Vulkan target, the discrete-versus-shared probe.

    Called at the placement work rather than at the load call, which also covers the
    Hub download: rows captured before a multi-minute download would price the fit
    against VRAM that has since been allocated. `_load_probe_memo_scope` on the load
    call owns the lifetime, so this never has to be unwound by hand.
    """
    _LOAD_PROBE_STATE.armed = True
    _LOAD_PROBE_STATE.rows = {}


@contextlib.contextmanager
def _load_probe_memo_scope():
    """Guarantee the memo cannot outlive a load however it exits. Arming is the
    narrower `_arm_load_probe_memo`, taken around the placement decision."""
    try:
        yield
    finally:
        _LOAD_PROBE_STATE.armed = False
        _LOAD_PROBE_STATE.rows = None


# "Probed, no usable answer", kept apart from None = "not probed yet".
_MISSING = object()

# Discrete by construction: `_weights_in_host_memory` classifies them upstream. Vulkan
# gets the shared-memory probe; SYCL/OpenCL/MUSA/CANN may be integrated, so they decline.
_SELF_EVIDENTLY_DISCRETE = frozenset({"cuda"})

# AMD needs its classifier to have RUN: the empty set means "no APU" or "no ROCm torch to
# ask", and there is no Windows ROCm wheel, so reading it as discrete gave a Strix Halo dio.
_NEEDS_ROCM_CLASSIFICATION = frozenset({"rocm", "hip"})

# The rungs narrow exactly these, so a memo keyed on the binary alone answered a narrowed
# set with the original set's verdict.
_DEVICE_VISIBILITY_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GGML_BACKEND_PATH",
    "LLAMA_ARG_DEVICE",
    "LLAMA_ARG_MAIN_GPU",
)

# A header, then one indented `<id>: <desc> (<total> MiB, <free> MiB free)` per device;
# `  (none)` or nothing on a CPU-only build. The id has no internal spaces, which excludes
# the `  Device 0: ...` lines ggml_cuda_init writes ABOVE the header.
_LISTED_DEVICE_RE = re.compile(r"^\s+(\S+):\s")
_LIST_DEVICES_HEADER = "Available devices:"


def _parse_listed_devices(text: Optional[str]) -> Optional[list[str]]:
    """Device ids from ``--list-devices`` output, or None when it had no answer.

    Only lines BELOW the header count: backends log their own initialisation above it,
    which is noise that happens to contain colons.
    """
    if not text:
        return None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == _LIST_DEVICES_HEADER:
            break
    else:
        # No header: an old build or unreadable output, neither of which is evidence.
        return None
    devices = []
    for line in lines[index + 1 :]:
        if not line.strip():
            continue
        if not line[:1].isspace():
            break  # dedent ends the block
        match = _LISTED_DEVICE_RE.match(line)
        if match:
            devices.append(match.group(1))
    return devices


# Exact soname or a real versioned form (`libggml-cuda.so.1`), not a stem prefix: that also
# matched `ggml-cuda.dll.disabled`, the name left behind by DISABLING the backend.
_GGML_GPU_BACKEND_RE = re.compile(
    r"^(?:lib)?ggml-(?:cann|cuda|hip|metal|musa|opencl|sycl|virtgpu|vulkan)"
    r"(?:\.dll|\.so(?:\.\d+)*|(?:\.\d+)*\.dylib)$"
)


def _cpu_runtime_owner_alive(staged_dir: Path) -> bool:
    """Whether a live process still owns this staged CPU-fallback runtime."""
    try:
        pid = int((staged_dir / _CPU_RUNTIME_OWNER_FILE).read_text(encoding = "utf-8").strip())
    except (OSError, ValueError):
        # No owner stamp: written by an older Unsloth, so leave it alone.
        return True
    if pid == os.getpid():
        return True
    # 0 and negatives address process groups, never an owner.
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(0x00100000, False, pid)
        if not handle:
            return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _sweep_abandoned_cpu_runtimes(runtime_root: Path) -> None:
    """Delete staged runtimes whose owning Unsloth is gone (kill -9, host crash)."""
    try:
        candidates = list(runtime_root.glob("llama-cpu-*"))
    except OSError:
        return
    for staged_dir in candidates:
        if not staged_dir.is_dir() or _cpu_runtime_owner_alive(staged_dir):
            continue
        with contextlib.suppress(OSError):
            shutil.rmtree(staged_dir, ignore_errors = True)


def _lib_dir_has_ggml_backend(lib_dir: Path, backend: str) -> bool:
    """Match an exact or versioned ggml backend library soname."""
    stem = f"ggml-{backend}.dll" if sys.platform == "win32" else f"libggml-{backend}.so"
    try:
        return any(
            f.name == stem or f.name.startswith(stem + ".")
            for f in lib_dir.iterdir()
            if f.is_file()
        )
    except OSError:
        return False


def _is_external_link(path: Path) -> bool:
    """True when ``path`` is a --with-llama-cpp-dir local link: a POSIX symlink
    or a Windows directory junction / reparse point. Such a link resolves into
    the user's own llama.cpp checkout, which Unsloth does not own."""
    try:
        if os.path.islink(path):
            return True
    except OSError:
        return False
    if os.name == "nt":
        try:
            import stat
            attrs = os.lstat(path).st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, AttributeError):
            return False
    return False
