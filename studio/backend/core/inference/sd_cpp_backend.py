# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native stable-diffusion.cpp diffusion backend (the no-GPU tier).

``SdCppDiffusionBackend`` presents the SAME public surface the image routes use on the diffusers
``DiffusionBackend``, but is backed by the ``sd-cli`` subprocess (``SdCppEngine``) instead of an
in-process diffusers pipeline. The engine router selects it only when no usable CUDA/ROCm/XPU GPU is
present, where it is measurably faster and far lighter on RAM than diffusers.

It reuses the transformer GGUF the diffusers path already downloads and additionally fetches the
per-family single-file VAE + text encoders declared in ``diffusion_families`` (sd-cli cannot read
the sharded diffusers components). The binary is installed lazily on first use; if it is unavailable
or the family has no native mapping, the router falls back to diffusers, so this backend is only
ever asked to run requests it can serve. Import-light on purpose: no torch / diffusers here, so
selecting it on a CPU box does not drag the heavy GPU stack into the process.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import functools
import os
import re
import shutil
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core.inference.inference_gate import holds_inference, pauses_background_embedding
from core.inference.diffusion_auto_policy import build_resolved_record, format_generation_for_log
from core.inference.diffusion_compat import flux2_inner_dim_for_pick
from core.inference.diffusion_device import (
    auto_cuda_ordinal_for_load,
    resolve_diffusion_device_target,
    resolve_selected_cuda_ordinal,
)
from core.inference.diffusion_families import (
    krea2_raw_checkpoint,
    krea2_sd_cpp_flow_shift,
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    DiffusionModelReplacedError,
    family_for_checkpoint,
    LoadIdentity,
    detect_family_for_pick,
    load_identity,
    family_sd_cpp_supported,
    mirror_repo,
    legacy_source_repo,
    prefer_cached_legacy_source,
    prefer_ungated_mirror,
    canonical_base,
    named_variant_base,
    resolve_base_repo,
    resolve_local_gguf_child,
    sd_cpp_text_encoder_cached_alternatives,
    sd_cpp_text_encoders_for,
    supported_family_names,
    _family_override_resolved,
)
from core.inference.diffusion_flow_shift import (
    SAMPLE_SIGMAS_KEY,
    sd_cpp_sample_sigmas,
    valid_sample_sigmas,
)
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)
from core.inference import sd_cpp_args as sd_cpp_args_mod
from core.inference.sd_cpp_args import (
    CPU_BACKEND_FLAGS,
    SdCppGenParams,
    SdCppModelFiles,
    build_img_gen_request,
    device_backend_flags,
    is_ggml_unsupported_op_abort,
    offload_flags,
    sd_cli_output_paths,
    without_device_backend_flags,
)
from core.inference.sd_cpp_engine import (
    NATIVE_GENERATION_TIMEOUT_S,
    SdCppCancelled,
    SdCppEngine,
    find_sd_cpp_binary,
    find_sd_server_binary,
    help_text_identifies_sd_cpp,
    is_managed_binary,
    legacy_sibling_install_root,
    managed_install_root,
    owning_managed_root,
    runtime_env,
)
from core.inference.sd_cpp_server import SdCppServer
from loggers import get_logger
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
from utils.account_context import account_thread, current_account_id
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

# Only a denominator equal to the requested steps is trusted.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Serialises the one-time binary install so concurrent first-loads don't race.
_install_lock = threading.Lock()

# Installs are writers, one-shot sd-cli runs readers: admission must be one decision.
# Held only across the state change; readers never take _install_lock, so no lock cycle.
_tree_state = threading.Condition()
_tree_readers = 0
_tree_installing = False
# A download can legitimately take minutes; wait rather than run a binary that is being replaced
_TREE_WAIT_TIMEOUT_S = 900.0
# Nothing notifies on cancel, so poll rather than one long wait.
_TREE_WAIT_TICK_S = 0.5


@contextlib.contextmanager
def _tree_claimed_for_install():
    """Claim the managed tree for an install. Yields False when something is running in it, in which
    case the caller keeps what is on disk and retries on a later load."""
    global _tree_installing
    with _tree_state:
        if _tree_readers or _tree_installing or _managed_tree_in_use():
            yield False
            return
        _tree_installing = True
    try:
        yield True
    finally:
        with _tree_state:
            _tree_installing = False
            _tree_state.notify_all()


@contextlib.contextmanager
def _tree_reader(
    binary: Optional[str],
    cancel_event: Optional[threading.Event] = None,
    cancelled_message: str = DIFFUSION_CANCELLED_MSG,
):
    """Claims only managed binaries (else unrelated bundle downloads would block them); a timeout is not admission and the wait is cancellable."""
    global _tree_readers
    if not is_managed_binary(binary):
        yield
        return
    with _tree_state:
        if _tree_installing:
            logger.info("waiting for the sd.cpp install to finish before starting a generation")
            deadline = time.monotonic() + _TREE_WAIT_TIMEOUT_S
            while _tree_installing:
                if cancel_event is not None and cancel_event.is_set():
                    # The video path recognises only its own sentinel, and an image message reaching it reads as Video generation failed for a plain cancel.
                    raise RuntimeError(cancelled_message)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"the stable-diffusion.cpp install is still replacing its binaries after "
                        f"{int(_TREE_WAIT_TIMEOUT_S)}s. Try again once it has finished."
                    )
                _tree_state.wait_for(
                    lambda: not _tree_installing, timeout = min(remaining, _TREE_WAIT_TICK_S)
                )
        _tree_readers += 1
    try:
        yield
    finally:
        with _tree_state:
            _tree_readers -= 1
            _tree_state.notify_all()


# Max images per img_gen job; larger Unsloth batches (up to 32) are split into these chunks
_MAX_SERVER_BATCH = 8


# Families whose bases ship a model_index.json ``sample_sigmas`` grid; every other load skips the card and index reads.
_SAMPLE_SIGMAS_FAMILIES = frozenset({"qwen-image-2.1"})


def _base_sample_sigmas(
    repo_id: str,
    base: str,
    hf_token: Optional[str],
    *,
    family: str,
    explicit_base: bool,
    local_files_only: bool,
    named_base: Optional[str] = None,
) -> tuple[Optional[tuple[float, ...]], str]:
    """(the base's model_index.json grid or None, that base). Without an explicit base the card's ``base_model``
    names it, as on the diffusers route, so a community Turbo GGUF does not read 2.1's index."""
    if family not in _SAMPLE_SIGMAS_FAMILIES:
        return None, base
    try:
        from core.inference.diffusion import (
            _hf_base_model,
            _is_trusted_diffusion_repo,
            hub_cache_dir,
        )
        from core.inference.diffusion_comfy_components import read_model_index

        if not explicit_base and not local_files_only:
            tag = _hf_base_model(repo_id, hf_token)
            # Same trust bar as the diffusers resolver: the tag is repo-author metadata.
            if tag and _is_trusted_diffusion_repo(tag):
                base = canonical_base(tag)
            elif tag is None and named_base:
                base = named_base
        elif not explicit_base and named_base:
            # Cache-only (e.g. the OpenAI route's auto-switch) reads no card: the pick's own name decides.
            base = named_base
        # Read from the repo the plan staged it under (_grid_fetch_repo: the base's unsloth mirror unless the
        # upstream already holds it), never the vendor id directly. Cache-only reads try both roots, as the plan's
        # locality check and _fetch_assets do, and the upstream id too, so an index cached under either still counts.
        fetch = _grid_fetch_repo(base, hf_token)
        repos = tuple(dict.fromkeys((fetch, base))) if local_files_only else (fetch,)
        roots = (hub_cache_dir(), None) if local_files_only else (hub_cache_dir(),)
        attempts = [(repo, root) for repo in repos for root in roots]
        for i, (repo, root) in enumerate(attempts):
            try:
                index = read_model_index(
                    repo, hf_token = hf_token, local_files_only = local_files_only, cache_dir = root
                )
                break
            except Exception:  # noqa: BLE001
                if i == len(attempts) - 1:
                    raise
    except Exception as exc:  # noqa: BLE001 - no grid is the pre-grid behaviour
        logger.debug("sd_cpp.sample_sigmas_unavailable: %s", exc)
        return None, base
    grid = valid_sample_sigmas(index.get(SAMPLE_SIGMAS_KEY) if isinstance(index, dict) else None)
    if grid is not None:
        logger.info(
            "sd_cpp: %s ships a %d-step sampling grid; passing it as custom sigmas", base, len(grid)
        )
    return grid, base


def _grid_fetch_repo(base: str, hf_token: Optional[str]) -> str:
    """The repo a base's ``model_index.json`` sampling grid is fetched from: the same per-repo swap the assets take, so
    the plan stages it where the load reads it (a routed base's unsloth mirror, e.g. unsloth/Qwen-Image-2.1-Turbo-FP8
    for Qwen/Qwen-Image-2.1-Turbo, unless the upstream already holds the file)."""
    files = ["model_index.json"]
    return prefer_cached_legacy_source(prefer_ungated_mirror(base, hf_token, files = files), files)


def _cgroup_cpu_limit(
    root: str = "/sys/fs/cgroup", proc_cgroup: str = "/proc/self/cgroup"
) -> Optional[int]:
    """Tightest cgroup v2 cpu.max over this cgroup and its parents, in whole CPUs; None if unreadable."""
    try:
        with open(proc_cgroup, encoding = "utf-8") as f:
            rel = next(line.split("::", 1)[1].strip() for line in f if line.startswith("0::"))
    except (OSError, StopIteration):
        rel = "/"
    limit: Optional[int] = None
    path = os.path.normpath(os.path.join(root, rel.lstrip("/")))
    while True:
        try:
            with open(os.path.join(path, "cpu.max"), encoding = "utf-8") as f:
                quota, period = f.read().split()[:2]
            if quota != "max":
                cpus = max(1, -(-int(quota) // int(period)))
                limit = cpus if limit is None else min(limit, cpus)
        except (OSError, ValueError):
            pass
        if len(path) <= len(root.rstrip("/")) or path == os.path.dirname(path):
            return limit
        path = os.path.dirname(path)


@functools.lru_cache(maxsize = 1)
def _libc() -> Any:
    """Cached: ``find_library`` walks the filesystem."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    libc.sysctlbyname.restype = ctypes.c_int
    return libc


def _sysctl(name: str, as_str: bool = False) -> Any:
    try:
        import ctypes

        libc = _libc()
        if as_str:
            buf = ctypes.create_string_buffer(64)
            size = ctypes.c_size_t(len(buf))
            ok = libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) == 0
            return buf.value.decode(errors = "replace") if ok else None
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        ok = libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) == 0
        return int(value.value) if ok and value.value > 0 else None
    except Exception:  # noqa: BLE001
        return None


# M5 Pro / Max: perflevel0 "Super", perflevel1 "Performance".
_FAST_PERF_LEVELS = ("super", "performance")


def _apple_performance_cores() -> Optional[int]:
    """macOS fast cores: perflevel0 plus lower tiers named Super/Performance (sd.cpp reads perflevel0 only).
    No SMT, so ``cpu_count() // 2`` undercounts."""
    if sys.platform != "darwin":
        return None
    levels = _sysctl("hw.nperflevels") or 0
    total = 0
    for level in range(levels):
        if level > 0:
            name = _sysctl(f"hw.perflevel{level}.name", as_str = True)
            if name is None or name.strip().lower() not in _FAST_PERF_LEVELS:
                continue
        total += _sysctl(f"hw.perflevel{level}.physicalcpu") or 0
    return total or _sysctl("hw.perflevel0.physicalcpu") or _sysctl("hw.physicalcpu")


def _linux_physical_cores(sysfs: str = "/sys/devices/system/cpu") -> Optional[int]:
    """Physical cores among the CPUs this process may run on: distinct ``thread_siblings_list`` sets over the affinity
    mask, so a host without SMT counts every core and a taskset / container cpuset counts only its own. None when the
    mask or the topology is unreadable."""
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    cores: set[str] = set()
    for cpu in allowed:
        try:
            with open(f"{sysfs}/cpu{cpu}/topology/thread_siblings_list", encoding = "utf-8") as f:
                siblings = f.read().strip()
        except OSError:
            return None
        if not siblings:
            return None
        cores.add(siblings)
    return len(cores) or None


def _default_threads() -> int:
    """``UNSLOTH_CPU_THREADS``, else macOS P-cores, else (Linux) the physical cores in the affinity mask, else
    ``os.cpu_count() // 2``; capped by the cgroup quota on Linux. Half the logical CPUs is one thread per core only on
    a 2-way SMT host that may use all of them: it halves a host without SMT and oversubscribes a cpuset."""
    try:
        configured = int(os.environ.get("UNSLOTH_CPU_THREADS") or 0)
    except ValueError:
        configured = 0
    if configured > 0:
        return configured
    performance_cores = _apple_performance_cores()
    if performance_cores:
        return performance_cores
    linux = sys.platform.startswith("linux")
    physical = _linux_physical_cores() if linux else None
    if physical:
        threads = physical
    else:
        cpu = os.cpu_count()
        threads = cpu // 2 if cpu else 8
    limit = _cgroup_cpu_limit() if linux else None
    return max(1, min(threads, limit) if limit else threads)


def _sd_cpp_default_threads(cpu: Optional[int]) -> Optional[int]:
    """What an sd.cpp build without its own topology read runs (``hardware_concurrency`` halved above 4 CPUs)."""
    if not cpu:
        return None
    return cpu if cpu <= 4 else cpu // 2


def _allowed_cpus() -> Optional[int]:
    """CPUs in this process's affinity mask, or None when unreadable."""
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        return None


def _cpus_restricted(cpu: Optional[int]) -> bool:
    """True when a cpuset / taskset or a cgroup quota leaves this process fewer CPUs than the host has."""
    if not cpu:
        return False
    allowed = _allowed_cpus()
    limit = _cgroup_cpu_limit()
    return (allowed is not None and allowed < cpu) or (limit is not None and limit < cpu)


def _explicit_threads() -> Optional[int]:
    """``UNSLOTH_CPU_THREADS``, the macOS fast cores, or (Linux) the thread count above whenever it differs from what
    sd.cpp would pick by itself; None keeps sd.cpp's default. sd.cpp builds before the fork's topology fix count half
    the logical CPUs and ignore the affinity mask, so a host without SMT or a cpuset needs the count passed."""
    try:
        configured = int(os.environ.get("UNSLOTH_CPU_THREADS") or 0)
    except ValueError:
        configured = 0
    if configured > 0:
        return configured
    performance_cores = _apple_performance_cores()
    if performance_cores:
        return performance_cores
    if sys.platform.startswith("linux"):
        cpu = os.cpu_count()
        threads = _default_threads()
        sd_default = _sd_cpp_default_threads(cpu)
        if threads == sd_default:
            return None
        # Fewer threads than sd.cpp's all-CPU default at 4 CPUs or fewer, only because of SMT: the extra siblings
        # measured faster (2 vCPU / 1 core, 4 vCPU / 2 cores), so keep sd.cpp's count unless something restricts us.
        if sd_default is not None and threads < sd_default and not _cpus_restricted(cpu):
            return None
        return threads
    return None


def _server_binary_runnable(binary: str) -> bool:
    """Probes runnability with the server's env before a multi-GB asset download; only OSError or exit 126/127/0xC0000135 means unrunnable."""
    import subprocess

    try:
        proc = subprocess.run(
            [binary, "--help"],
            capture_output = True,
            timeout = 20,
            env = runtime_env(binary),
            **windows_hidden_subprocess_kwargs(),
        )
    except OSError:
        return False  # cannot exec at all (wrong arch / no execute bit / missing loader)
    except Exception:  # noqa: BLE001 -- don't block on a flaky probe
        return True
    # Negative = signal death (e.g. SIGILL); Windows loader failures are large positive codes.
    return (
        proc.returncode >= 0
        and proc.returncode not in (126, 127)
        and proc.returncode not in _WINDOWS_IMAGE_LOAD_FAILURE_STATUSES
    )


def _usable_or_discard_managed(binary: str) -> bool:
    """Removes an unrunnable binary only if the installer owns it; a user's own build is reported as-is."""
    if _server_binary_runnable(binary):
        return True
    if not is_managed_binary(binary):
        logger.warning(
            "sd.cpp binary %s is not runnable; leaving it alone (not an Unsloth-owned install we may "
            "replace). Delete its directory to have Unsloth reinstall the prebuilt.",
            binary,
        )
        return True  # not ours to replace; the router's own probe still refuses it
    logger.warning("managed sd.cpp binary %s is not runnable; removing it so it reinstalls", binary)
    try:
        Path(binary).unlink()
    except OSError as exc:
        logger.warning("could not remove the unusable managed binary %s: %s", binary, exc)
        return True  # cannot repair it; don't spin on a reinstall that will find it again
    return False


def _sd_cpp_probe_output(binary: str, *args: str) -> Optional[str]:
    """Identifying output settles it whatever the exit code (old builds exit 1); else only a clean exit is evidence, and rc 127 from the loader is not."""
    import subprocess

    try:
        proc = subprocess.run(
            [binary, *args],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 20,
            env = runtime_env(binary),
            **windows_hidden_subprocess_kwargs(),
        )
    except Exception:  # noqa: BLE001 -- cannot exec / timeout: "cannot tell"
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


# --ref-video arrived with H3 upstream; --version is "unknown" on release prebuilts.
_H3_HELP_MARKER = "--ref-video"


def sd_cpp_supports_minimax_h3(binary: str) -> bool:
    """Unreadable help counts as supported: the version() gate already refuses unrunnable builds."""
    text = _sd_cpp_probe_output(binary, "--help")
    if text is None:
        return True
    return help_text_supports_minimax_h3(text)


def help_text_supports_minimax_h3(help_text: str) -> bool:
    """``sd_cpp_supports_minimax_h3``'s verdict on ``--help`` output that is already in hand."""
    return _H3_HELP_MARKER in help_text


def sd_cpp_binary_vets_for_h3(binary: str) -> bool:
    """rejects unrelated --ref-video tools; version() rejects binaries an unreadable probe keeps."""
    text = _sd_cpp_probe_output(binary, "--help")
    if text is None:
        return True
    return help_text_identifies_sd_cpp(text) and help_text_supports_minimax_h3(text)


def sd_cpp_graph_cut_options(binary: Optional[str]) -> frozenset[str]:
    """returns advertised graph-cut flags; missing help emits none because sd-cli rejects them."""
    if not binary:
        return frozenset()
    text = _sd_cpp_probe_output(binary, "--help")
    if text is None or "--max-vram" not in text:
        return frozenset()
    return frozenset(flag for flag in ("--max-vram", "--stream-layers") if flag in text)


def sd_cpp_supports_sage_attn(binary: Optional[str]) -> bool:
    """fails closed because sd-cli rejects unknown flags (u13b9d92 predates --sage-attn)."""
    if not binary:
        return False
    text = _sd_cpp_probe_output(binary, "--help")
    return text is not None and "--sage-attn" in text


def sd_cpp_lists_accelerator_device(binary: Optional[str]) -> bool:
    """Unreadable output does not mean no GPU; only a CPU-only listing or no binary returns False."""
    if not binary:
        return False
    return accelerator_verdict_keeps_gpu(sd_cpp_accelerator_device_verdict(binary))


def accelerator_verdict_keeps_gpu(verdict: Optional[bool]) -> bool:
    """ "Could not tell" keeps the GPU: an unreadable probe is not evidence of no accelerator."""
    return True if verdict is None else verdict


def sd_cpp_accelerator_device_verdict(binary: str) -> Optional[bool]:
    """Keeps None apart from True, so an unreadable re-probe is not taken for a changed build."""
    text = _sd_cpp_probe_output(binary, "--list-devices")
    _remember_device_listing(binary, text)
    if text is None:
        return None
    names = [line.split("\t", 1)[0].strip() for line in text.splitlines() if "\t" in line]
    if not names:
        return None
    return any(name.upper() != "CPU" for name in names)


# Vulkan excluded: its ordinals are another namespace.
_PHYSICAL_INDEX_DEVICE_PREFIXES: tuple[str, ...] = ("CUDA", "ROCM")


def sd_cpp_device_name_for_ordinal(binary: Optional[str], ordinal: Optional[int]) -> Optional[str]:
    """Returns None whenever the name is uncertain, since that falls back to sd.cpp's own device choice."""
    if not binary or ordinal is None:
        return None
    text = sd_cpp_device_listing(binary)
    if text is not None:
        for line in text.splitlines():
            name = line.split("\t", 1)[0].strip()
            head = name.rstrip("0123456789")
            if head.upper() not in _PHYSICAL_INDEX_DEVICE_PREFIXES:
                continue
            if name[len(head) :] == str(ordinal):
                return name
    # Warn rather than refuse: sd.cpp treats an unknown argument (old builds) as fatal.
    logger.warning(
        "sd_cpp.device_pin_unresolved: this build does not report a CUDA/ROCm device %s "
        "(--list-devices %s), so the graph runs on its own default device",
        ordinal,
        "was unreadable" if text is None else "does not list it",
    )
    return None


# ggml-cuda's init log on stderr; a HIP build says "ROCm devices", so only CUDA builds match.
_CUDA_INIT_RE = re.compile(r"ggml_cuda_init: found \d+ CUDA devices")
_CUDA_DEVICE_CC_RE = re.compile(
    r"^\s*Device (\d+): .*?, compute capability (\d+)\.(\d+)", re.MULTILINE
)


# Last --list-devices answer per binary. The accelerator verdict always re-probes (a load re-vets the listing on purpose).
_LAST_DEVICE_LISTING: dict = {}
# What ggml reads to pick or hide devices.
_DEVICE_LISTING_ENV_VARS: tuple[str, ...] = (
    "GGML_VK_VISIBLE_DEVICES",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def _binary_stat_identity(binary: str) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(binary)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _hardware_inventory_generation() -> Optional[int]:
    try:
        from utils.hardware.hardware import physical_gpu_inventory_generation
        return physical_gpu_inventory_generation()
    except Exception:  # noqa: BLE001 -- an identity read must never fail the probe
        return None


def _device_listing_identity(binary: str) -> tuple:
    return (
        _binary_stat_identity(binary),
        tuple(os.environ.get(name) for name in _DEVICE_LISTING_ENV_VARS),
        _hardware_inventory_generation(),
    )


def _remember_device_listing(binary: Optional[str], text: Optional[str]) -> None:
    if not binary:
        return
    if text is None:
        # An unreadable probe or a timeout is never cached.
        return
    _LAST_DEVICE_LISTING[binary] = (_device_listing_identity(binary), text)


def _cached_device_listing(binary: str, *, require_file: bool = False) -> Optional[str]:
    seen = _LAST_DEVICE_LISTING.get(binary)
    if seen is None:
        return None
    identity = _device_listing_identity(binary)
    if require_file and identity[0] is None:
        return None
    return seen[1] if seen[0] == identity else None


def sd_cpp_device_listing(binary: str) -> Optional[str]:
    """``binary --list-devices``, reusing the listing already taken of this file under this environment."""
    text = _cached_device_listing(binary, require_file = True)
    if text is None:
        text = _sd_cpp_probe_output(binary, "--list-devices")
        if _binary_stat_identity(binary) is not None:
            _remember_device_listing(binary, text)
    return text


def sd_cpp_cuda_compute_capability(
    binary: Optional[str],
    device_name: Optional[str],
    *,
    probe: bool = True,
) -> Optional[tuple[int, int]]:
    """Compute capability the CUDA build reports for ``device_name`` (``CUDA<i>``); the lowest card
    when unpinned. None when unsure (unreadable, non-CUDA build, unlisted device). ``probe = False``
    only reads the listing the last accelerator verdict took of this file."""
    if not binary:
        return None
    text = _cached_device_listing(binary)
    if text is None and probe:
        text = sd_cpp_device_listing(binary)
    if text is None or not _CUDA_INIT_RE.search(text):
        return None
    caps = {
        int(m.group(1)): (int(m.group(2)), int(m.group(3)))
        for m in _CUDA_DEVICE_CC_RE.finditer(text)
    }
    if not caps:
        return None
    if device_name is None:
        return min(caps.values())
    head = device_name.rstrip("0123456789")
    if head.upper() != "CUDA" or not device_name[len(head) :]:
        return None
    return caps.get(int(device_name[len(head) :]))


def sd_cpp_device_listing_known(binary: Optional[str]) -> bool:
    if not binary:
        return False
    return _cached_device_listing(binary) is not None


# Runs when --list-devices timed out (possibly a stalled driver); CUDA calls have no deadline, so bound the wait.
TORCH_CC_TIMEOUT_S = 10.0


def torch_cuda_compute_capability(
    ordinal: Optional[int], *, timeout_s: Optional[float] = None
) -> Optional[tuple[int, int]]:
    """torch's compute capability for ``ordinal`` (lowest visible when None), or None on ROCm (not sm numbers),
    error or timeout (the daemon thread is left behind). Same runtime and mask as CUDA sd.cpp, so indices agree."""
    import threading

    result: list = []

    def read() -> None:
        result.append(_torch_cuda_compute_capability(ordinal))

    worker = threading.Thread(target = read, daemon = True, name = "sd-cpp-torch-cc")
    worker.start()
    worker.join(TORCH_CC_TIMEOUT_S if timeout_s is None else timeout_s)
    return result[0] if result else None


def _torch_cuda_capabilities(
    ordinal: Optional[int], *, timeout_s: Optional[float] = None
) -> list[tuple[int, int]]:
    """Capability of ``ordinal`` (None = every visible card); a stalled driver call reads as unknown, not a hang."""
    import threading

    result: list = []
    worker = threading.Thread(
        target = lambda: result.append(_read_torch_cuda_capabilities(ordinal)),
        daemon = True,
        name = "sd-cpp-torch-ccs",
    )
    worker.start()
    worker.join(TORCH_CC_TIMEOUT_S if timeout_s is None else timeout_s)
    return result[0] if result else []


def _read_torch_cuda_capabilities(ordinal: Optional[int]) -> list[tuple[int, int]]:
    try:
        import torch

        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            return []
        cards = [int(ordinal)] if ordinal is not None else list(range(torch.cuda.device_count()))
        return [tuple(int(v) for v in torch.cuda.get_device_capability(i)) for i in cards]
    except Exception:  # noqa: BLE001 -- unknown keeps the build's default
        return []


def _sd_cpp_quant_cublas_env(
    ordinal: Optional[int], weight_types: Optional[frozenset[str]]
) -> dict[str, str]:
    if not weight_types or not weight_types <= sd_cpp_args_mod.QUANT_CUBLAS_WEIGHT_TYPES:
        return {}
    return sd_cpp_args_mod.quant_cublas_env(
        _torch_cuda_capabilities(ordinal), weight_types = weight_types
    )


def _sd_cpp_spawn_env(
    off_torch: Any,
    device: str,
    ordinal: Optional[int],
    offload: "tuple[str, ...] | list[str]",
    pinned: "tuple[str, ...] | list[str]",
    weight_types: Any,
) -> Optional[dict[str, str]]:
    """The child env for the FINAL placement and argv. The ordinal counts only when the argv carries its pin;
    without one sd.cpp picks its own default card, so every visible card must qualify."""
    if off_torch is not None:
        return off_torch.child_env()
    if device != "cuda":
        return None
    pin_applied = len(pinned) > len(offload)
    return _sd_cpp_quant_cublas_env(ordinal if pin_applied else None, weight_types()) or None


def _torch_cuda_compute_capability(ordinal: Optional[int]) -> Optional[tuple[int, int]]:
    try:
        import torch

        if getattr(torch.version, "hip", None) or not getattr(torch.version, "cuda", None):
            return None
        if not torch.cuda.is_available():
            return None
        if ordinal is None:
            caps = [
                tuple(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())
            ]
            return min(caps) if caps else None  # type: ignore[return-value]
        if not 0 <= int(ordinal) < torch.cuda.device_count():
            return None
        major, minor = torch.cuda.get_device_capability(int(ordinal))
        return (int(major), int(minor))
    except Exception:  # noqa: BLE001 -- unknown stays unknown
        return None


_VULKAN_DEVICE_UMA_RE = re.compile(r"^ggml_vulkan: (\d+) = .*?\| uma: ([01]) \|", re.MULTILINE)


def sd_cpp_vulkan_device_is_uma(
    binary: Optional[str],
    device_name: Optional[str],
    *,
    probe: bool = True,
) -> Optional[bool]:
    """Whether ggml-vulkan reports ``device_name`` (``Vulkan<i>``; Vulkan0 when unpinned) as unified memory. Its free
    memory for such a device sums every heap, the GTT the pinned weights come from included. None when unsure."""
    if not binary:
        return None
    text = _cached_device_listing(binary)
    if text is None and probe:
        text = sd_cpp_device_listing(binary)
    if text is None:
        return None
    uma = {int(m.group(1)): m.group(2) == "1" for m in _VULKAN_DEVICE_UMA_RE.finditer(text)}
    if device_name is None:
        return uma.get(0)
    head = device_name.rstrip("0123456789")
    if head.upper() != "VULKAN" or not device_name[len(head) :]:
        return None
    return uma.get(int(device_name[len(head) :]))


# Every namespace ggml names a device in; the narrower list above is physical-index schemes only.
_GGML_DEVICE_PREFIXES: tuple[str, ...] = ("CUDA", "ROCM", "VULKAN", "SYCL", "METAL", "OPENCL")


_TRADEMARK_RE = re.compile(r"\((?:tm|r)\)|[\u2122\u00ae]", re.IGNORECASE)


def _normalized_card_name(text: str) -> str:
    """Letters and digits only, lowercased, trademark marks dropped (HIP's `AMD Radeon(TM) Graphics` is RADV's
    `AMD Radeon Graphics`): `AMD Radeon RX 7900 XTX (RADV NAVI31)` then contains `AMD Radeon RX 7900 XTX`."""
    text = _TRADEMARK_RE.sub(" ", text)
    return "".join(character for character in text.lower() if character.isalnum())


_DRIVER_TAG_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _without_driver_tag(text: str) -> str:
    """The ICD's trailing driver tag removed, so the rest can be compared for EQUALITY: containment cannot, `RX 7600` is inside `RX 7600 XT`."""
    return _DRIVER_TAG_RE.sub("", text).strip()


# ROCR filters agents, then HIP (CUDA_VISIBLE_DEVICES) indexes the rest; GPU_DEVICE_ORDINAL is declined.
_ROCR_MASK_VAR = "ROCR_VISIBLE_DEVICES"
_HIP_MASK_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_OPAQUE_MASK_VAR = "GPU_DEVICE_ORDINAL"


def _mask_entries(value: Optional[str]) -> "Optional[list[int]]":
    """A visibility mask as physical indices, or ``None`` when written another way (a UUID mask, `GPU-...`, says nothing about enumeration order)."""
    if value is None:
        return None
    entries = [part.strip() for part in str(value).split(",")]
    entries = [part for part in entries if part]
    if not entries:
        return []
    out: "list[int]" = []
    for part in entries:
        try:
            index = int(part)
        except ValueError:
            return None
        if index < 0:
            # CUDA stops at the first invalid entry; a negative one hides everything after it.
            break
        out.append(index)
    return out


def _physical_index_of(ordinal: int, env: Optional[dict] = None) -> "tuple[Optional[int], bool]":
    """``(HIP device id, a mask was set)``; ``None`` where the masks cannot be composed, so the caller declines the tie-break rather than guessing."""
    source = os.environ if env is None else env
    # Windows HIP has no ROCr layer, so a leftover ROCR_VISIBLE_DEVICES masks nothing there.
    rocr = None if sys.platform == "win32" else source.get(_ROCR_MASK_VAR)
    # ROCm reads a non-empty HIP_VISIBLE_DEVICES over CUDA_VISIBLE_DEVICES; an empty one does not shadow it.
    present = [source.get(var) for var in _HIP_MASK_VARS if source.get(var) is not None]
    hip = next((raw for raw in present if raw.strip()), present[0] if present else None)
    opaque = source.get(_OPAQUE_MASK_VAR)
    if rocr is None and hip is None and opaque is None:
        return ordinal, False
    if opaque is not None:
        return None, True
    visible: "Optional[list[int]]" = None
    for is_hip, raw in ((False, rocr), (True, hip)):
        if raw is None:
            continue
        entries = _mask_entries(raw)
        if entries is None:
            return None, True
        # clr keeps the first of a repeated HIP entry; ROCr stops at one (utils/hardware/amd.py).
        if is_hip:
            entries = list(dict.fromkeys(entries))
        elif len(set(entries)) < len(entries):
            entries = entries[: next(i for i, e in enumerate(entries) if e in entries[:i])]
        if visible is None:
            visible = entries
        else:
            # The inner mask indexes into what the outer one left, not into the physical list.
            try:
                visible = [visible[index] for index in entries]
            except IndexError:
                return None, True
    if visible is None or ordinal >= len(visible):
        return None, True
    return visible[ordinal], True


def _card_lookup_inventory() -> dict:
    """The physical inventory for naming a selected card. Off the event loop a cold cache is read
    blocking: the non-blocking read answers "unknown" until its refresh lands, which on a host whose
    torch works is the first load, so that load's failure was recorded against every card."""
    from utils.hardware.hardware import get_physical_gpu_inventory

    inventory = get_physical_gpu_inventory(block = False)
    if not (inventory or {}).get("unknown"):
        return inventory
    try:
        import asyncio
        asyncio.get_running_loop()
        return inventory
    except RuntimeError:
        return get_physical_gpu_inventory(block = True)


def _amd_inventory_rows(inventory: Optional[dict]) -> list:
    """The inventory's AMD rows. ``index`` is vendor-local, and the ids here come from amd-smi, so an
    Intel iGPU or NVIDIA card at the same index must not answer for an AMD one."""
    return [
        device
        for device in ((inventory or {}).get("devices") or [])
        if isinstance(device, dict)
        and device.get("index") is not None
        and device.get("vendor") == "amd"
    ]


def _physical_position_of(hip_index: int) -> "tuple[Optional[str], Optional[int]]":
    """``(name, position)`` for a HIP device id; a HIP id is NOT the inventory's `index`."""
    try:
        from utils.hardware.amd import get_hip_id_by_gpu_index

        inventory = _card_lookup_inventory()
        if (inventory or {}).get("unknown"):
            return None, None
        devices = _amd_inventory_rows(inventory)
        hip_by_row = get_hip_id_by_gpu_index()
    except Exception:  # noqa: BLE001
        return None, None
    if not hip_by_row:
        return None, None
    physical_index = next(
        (row for row, hip in hip_by_row.items() if hip == hip_index),
        None,
    )
    if physical_index is None:
        return None, None
    devices.sort(key = lambda device: device.get("index"))
    selected = next((d for d in devices if d.get("index") == physical_index), None)
    if selected is None:
        return None, None
    # Counted on the NAME, what `sd_cpp_device_named` indexes by; `_card_identity` carries the gfx target too and would split one name into two groups.
    name = (selected.get("name") or "").strip() or None
    if name is None:
        return None, None
    wanted = _normalized_card_name(name)
    position = 0
    for device in devices:
        if device.get("index") >= physical_index:
            continue
        other = (device.get("name") or "").strip()
        if not other:
            # An unnamed row could be another card of this same name; nothing here can say.
            return name, None
        if _normalized_card_name(other) == wanted:
            position += 1
    return name, position


def physical_card_name(ordinal: Optional[int]) -> "tuple[Optional[str], Optional[int]]":
    """The card at torch-visible *ordinal*, and its place among the PHYSICAL cards of that name
    (see `sd_cpp_device_named`): Vulkan reads no HIP mask, so counting inside the masked list pinned
    card 0 while Studio reserved card 1."""
    if ordinal is None:
        return None, None
    try:
        import torch
        if not torch.cuda.is_available() or ordinal >= torch.cuda.device_count():
            return None, None
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        return None, None
    name = (names[ordinal] or "").strip()
    physical_index, masked = _physical_index_of(ordinal)
    if not masked:
        if not name:
            return None, None
        # Still a HIP id, not an inventory row: map it, and count in torch order only without a mapping.
        physical_name, position = _physical_position_of(physical_index)
        if position is not None:
            return name or physical_name, position
        return name, sum(1 for index in range(ordinal) if (names[index] or "").strip() == name)
    physical_name, position = (None, None)
    if physical_index is not None:
        physical_name, position = _physical_position_of(physical_index)
    name = name or (physical_name or "")
    if not name:
        return None, None
    return name, position


def sd_cpp_device_named(
    binary: Optional[str],
    card_name: Optional[str],
    *,
    position: Optional[int] = None,
    listing: Optional[str] = None,
) -> Optional[str]:
    """The ggml device that IS *card_name*, when exactly one of them is. The Vulkan build names its
    devices in its own namespace, so a physical ordinal names nothing in it. ``position`` breaks a
    tie on the ASSUMPTION that RADV and HIP walk one vendor's GPUs in the same order -- unreadable
    here, so only among devices already agreed to be the right MODEL. ``listing``: tab-separated name and
    description lines read elsewhere, instead of running ``binary --list-devices``."""
    if not binary or not card_name:
        return None
    wanted = _normalized_card_name(card_name)
    if not wanted:
        return None
    text = listing if listing is not None else sd_cpp_device_listing(binary)
    if text is None:
        return None
    # Both readings: containment cannot tell `RX 7600` from `RX 7600 XT`, equality cannot see through a driver tag.
    exact: list[str] = []
    loose: "list[tuple[str, str]]" = []
    for line in text.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        head = name.rstrip("0123456789")
        if head.upper() not in _GGML_DEVICE_PREFIXES:
            continue
        described = _normalized_card_name(parts[1])
        if not described:
            continue
        if _normalized_card_name(_without_driver_tag(parts[1])) == wanted:
            exact.append(name)
        if wanted in described or described in wanted:
            loose.append((name, described))
    if exact:
        matches, same_model = exact, True
    else:
        matches = [name for name, _described in loose]
        same_model = len({described for _name, described in loose}) == 1
    if matches and position is not None and not (same_model and 0 <= position < len(matches)):
        # UNRESOLVED even for a single answer: nothing says that singleton is the card at that position.
        logger.warning(
            "sd_cpp.device_pin_ambiguous: the selection is card %s of the ones answering to "
            "%r, and this build enumerates %s of them%s, so none is pinned and the graph runs "
            "on this build's own default device",
            position,
            card_name,
            len(matches),
            "" if same_model else " across more than one model",
        )
        return None
    if len(matches) == 1:
        return matches[0]
    if matches and same_model and position is not None and 0 <= position < len(matches):
        return matches[position]
    if matches:
        logger.warning(
            "sd_cpp.device_pin_ambiguous: %s devices answer to %r and %s, so none is pinned "
            "and the graph runs on this build's own default device",
            len(matches),
            card_name,
            "they do not all describe the same model"
            if not same_model
            else "the selection's place among them is not known",
        )
    return None


def _h3_replacement_hint(binary: str) -> str:
    """Advises moving a user-owned build at the install path aside, never deleting it; else empty."""
    roots = [managed_install_root(), legacy_sibling_install_root()]
    for root in roots:
        if root is None:
            continue
        try:
            Path(binary).resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        return f", or move {root} aside so Unsloth can install the pinned prebuilt there"
    return ""


def ensure_h3_sd_cpp_binary(
    *, allow_install: bool = True, accelerator: str = "cpu"
) -> Optional[str]:
    """Only this path requires the binary advertise H3 (image generation must keep working on user builds); a stale owned copy is deleted, a user build left alone."""
    binary = ensure_sd_cpp_binary(allow_install = allow_install, accelerator = accelerator)
    if not binary:
        return binary
    # One --help answers identity and H3 (a second spawn could read another build); None keeps the binary, unlike the engine identity probe, which rejects.
    help_text = _sd_cpp_probe_output(binary, "--help")
    if help_text is None:
        return binary
    # Identity before capability: unrelated tools also expose --ref-video (#8507).
    identified = help_text_identifies_sd_cpp(help_text)
    if identified and help_text_supports_minimax_h3(help_text):
        return binary
    # A managed copy that is not sd.cpp is reinstalled, but calling it an old build would be false.
    fault = "does not advertise MiniMax-H3 support" if identified else "is not stable-diffusion.cpp"
    if not is_managed_binary(binary):
        # Not sd.cpp at all (#8507): not ours to overwrite, so say so and stop.
        if not identified:
            raise RuntimeError(
                f"The executable at {binary} is not stable-diffusion.cpp: its --help output does "
                f"not identify the project. Point SD_CLI_PATH at a stable-diffusion.cpp build from "
                f"master-812-ea7f0c8 or newer, or UNSLOTH_SD_CPP_PATH at the directory holding one"
                f"{_h3_replacement_hint(binary)}."
            )
        raise RuntimeError(
            f"The stable-diffusion.cpp binary at {binary} does not advertise MiniMax-H3 support "
            f"(its --help does not list the H3 options), so generation would fail on it. "
            f"Point SD_CLI_PATH at a build from master-812-ea7f0c8 or "
            f"newer, or UNSLOTH_SD_CPP_PATH at the directory holding one"
            f"{_h3_replacement_hint(binary)}."
        )
    if not allow_install:
        # Ours, but replacing it is exactly what auto-install is switched off for.
        logger.warning("managed sd.cpp binary %s %s", binary, fault)
        return None
    # Deleting writes the managed tree, so take the install claim only around the unlink (not reentrant; ensure claims the tree itself).
    with _tree_claimed_for_install() as claimed:
        if not claimed:
            logger.warning(
                "managed sd.cpp binary %s %s, but something is still running out of the "
                "managed install; retrying on a later load",
                binary,
                fault,
            )
            return None
        logger.warning(
            "managed sd.cpp binary %s %s; removing it so it reinstalls",
            binary,
            fault,
        )
        try:
            Path(binary).unlink()
        except OSError as exc:
            logger.warning("could not remove the stale managed sd.cpp binary %s: %s", binary, exc)
            return None
    binary = ensure_sd_cpp_binary(allow_install = True, accelerator = accelerator)
    if binary and not sd_cpp_supports_minimax_h3(binary):
        return None
    return binary


def _installer_module():
    """The installer module, importable from the backend's sys.path. None if unavailable."""
    import sys

    studio_dir = Path(__file__).resolve().parents[3]  # .../studio
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    import install_sd_cpp_prebuilt

    return install_sd_cpp_prebuilt


# Without this, a host with no asset for its GPU would re-download on every load.
_failed_accelerator_upgrades: set[str] = set()


# (pin, accelerator class) upgrades that failed this process: keep the old bundle instead of retrying every load.
_failed_pin_upgrades: set[tuple[str, str]] = set()
_PIN_UPGRADE_ENV = "UNSLOTH_SD_CPP_AUTO_UPGRADE"


def _pin_upgrade_disabled() -> bool:
    return os.environ.get(_PIN_UPGRADE_ENV, "").strip().lower() in ("0", "false", "no", "off")


def _pin_moved(binary: str, accelerator: str) -> bool:
    """True when ``binary`` is a managed install made for an older pin. Unknown answers False."""
    if _pin_upgrade_disabled():
        return False
    root = owning_managed_root(binary)
    if root is None:
        return False
    if _managed_tree_in_use():
        return False
    try:
        mod = _installer_module()
        want = mod._pinned_tag()
        if not want or (want, mod.accelerator_class(accelerator)) in _failed_pin_upgrades:
            return False
        return bool(mod.install_is_stale(root))
    except Exception:  # noqa: BLE001 -- cannot tell -> keep the existing binary
        return False


def _note_failed_pin_upgrade(accelerator: str) -> None:
    try:
        mod = _installer_module()
        want = mod._pinned_tag()
        if want:
            _failed_pin_upgrades.add((want, mod.accelerator_class(accelerator)))
    except Exception:  # noqa: BLE001
        pass


def _needs_reinstall(binary: str, accelerator: str) -> bool:
    return _accelerator_changed(binary, accelerator) or _pin_moved(binary, accelerator)


def _note_failed_upgrade(accelerator: str) -> None:
    """Stop retrying an accelerator upgrade that just failed while a usable binary is kept."""
    try:
        _failed_accelerator_upgrades.add(_installer_module().accelerator_class(accelerator))
    except Exception:  # noqa: BLE001
        pass


# The rung between "the ROCm build does not run here" and CPU. The upstream ROCm archive carries
# kernels for 27 gfx targets but bundles no HIP/BLAS runtime (NEEDED libhipblas.so.3,
# librocblas.so.5, libamdhip64.so.7 from the HOST), so any host without a matching ROCm userspace
# fails it. Linux: exits 0 listing CPU only (caught by the device probe). Windows: exits 0xC0000135
# with no output (caught by exit status). One-way and one-deep; CUDA needs no rung. Vulkan falls
# to CPU: a CPU-torch host is routed to it, and one that cannot start it must not lose sd.cpp.
_ACCELERATOR_FALLBACK: dict[str, str] = {"rocm": "vulkan", "vulkan": "cpu"}


def _vulkan_cpu_rung_applies() -> bool:
    """Whether ambiguous failures can move the Vulkan build to CPU: only on the CPU-torch host routed to Vulkan. An XPU
    host (always Vulkan) and a ROCm host on its Vulkan rung leave the GPU only on decisive evidence. Unknown: as before."""
    try:
        from utils.hardware.hardware import DeviceType, get_device
        return get_device() == DeviceType.CPU
    except Exception:  # noqa: BLE001
        return True


# The sonames the upstream ROCm archive imports and does not ship.
_ROCM_RUNTIME_SONAMES: tuple[str, ...] = (
    "libamdhip64.so.7",
    "libhipblas.so.3",
    "librocblas.so.5",
)


def rocm_runtime_resolvable() -> Optional[bool]:
    """Uses ctypes, not ``ldconfig -p``: its cache can omit ROCm libraries the loader still resolves."""
    if os.name != "posix" or sys.platform == "darwin":
        return None
    try:
        import ctypes
    except Exception:  # noqa: BLE001 - no ctypes, so nothing can be established
        return None
    for soname in _ROCM_RUNTIME_SONAMES:
        try:
            ctypes.CDLL(soname)
        except OSError:
            return False
        except Exception:  # noqa: BLE001 - an unexpected loader failure establishes nothing
            return None
    return True


def rocm_runtime_missing_sonames() -> Optional[list[str]]:
    """The ROCm runtime libraries the loader cannot open here: [] when all load, None when it cannot be asked
    (Windows, macOS, no ctypes, an unexpected loader error), which says nothing either way."""
    if os.name != "posix" or sys.platform == "darwin":
        return None
    try:
        import ctypes
    except Exception:  # noqa: BLE001
        return None
    missing = []
    for soname in _ROCM_RUNTIME_SONAMES:
        try:
            ctypes.CDLL(soname)
        except OSError:
            missing.append(soname)
        except Exception:  # noqa: BLE001 - an unexpected loader failure establishes nothing
            return None
    return missing


ACCELERATOR_FALLBACK_CLEAR_HINT = (
    "to retry the failed GPU build after fixing that, clear the record with "
    "DELETE /api/settings/diffusion-accelerator-fallback (or restart with "
    "UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0 to never fall back)"
)


def accelerator_fallback_reason(accelerator: Optional[str]) -> str:
    """Why the ``accelerator`` build is being replaced, in words a user can act on."""
    if _accelerator_class_of(accelerator) != "rocm":
        return "its build did not list a GPU"
    if os.name == "nt":
        # The loader cannot be asked ahead of time on Windows; a missing DLL is exit 0xC0000135 with no output.
        return (
            "the ROCm build needs the AMD HIP SDK 7.x runtime DLLs (amdhip64_7.dll, hipblas.dll, rocblas.dll) on "
            "PATH; without them Windows stops it before it prints anything (exit 0xC0000135), or the card's gfx "
            "target is missing from the prebuilt"
        )
    missing = rocm_runtime_missing_sonames()
    if missing is None:
        return (
            "the ROCm build listed no GPU (whether the host ROCm 7 runtime loads could not be checked here, so "
            "either it is missing or the card's gfx target is not in the prebuilt)"
        )
    if missing:
        return (
            "the ROCm build needs the host ROCm 7 runtime and the loader cannot find "
            + ", ".join(missing)
            + " (install ROCm 7.x userspace or add /opt/rocm/lib to the loader path)"
        )
    return (
        "the ROCm build listed no GPU (the ROCm runtime loads, so the card's gfx target may be missing from the "
        "prebuilt, or /dev/kfd is not accessible to this user)"
    )


# ggml device-name prefix -> accelerator class, for a build no install record describes.
_LISTED_DEVICE_CLASSES: dict[str, str] = {
    "VULKAN": "vulkan",
    "ROCM": "rocm",
    "CUDA": "cuda",
    "SYCL": "sycl",
    "METAL": "metal",
    "OPENCL": "opencl",
}


def sd_cpp_listed_device(
    binary: Optional[str],
    device_name: Optional[str] = None,
    *,
    probe: bool = True,
) -> "tuple[Optional[str], Optional[str]]":
    """``(accelerator class, description)`` of the device ``binary`` renders on, from its own ``--list-devices``:
    ``device_name`` when pinned, else the first non-CPU device (sd.cpp's own default). ``(None, None)`` when the
    listing is unreadable or names no GPU. What an unowned build (SD_CLI_PATH / UNSLOTH_SD_CPP_PATH) is, since
    no install record says. ``probe = False`` only reads the listing the last accelerator verdict took of this
    file (no subprocess)."""
    if not binary:
        return None, None
    text = _cached_device_listing(binary)
    if text is None and probe:
        text = sd_cpp_device_listing(binary)
    if text is None:
        return None, None
    for line in text.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        klass = _LISTED_DEVICE_CLASSES.get(name.rstrip("0123456789").upper())
        if klass is None:
            continue
        if device_name is not None and name != device_name:
            continue
        return klass, parts[1].strip() or None
    return None, None


def accelerator_probe_failure_is_decisive(accelerator: Optional[str]) -> bool:
    """A CPU-only probe is decisive only if the ROCm runtime is proven absent; else two strikes apply."""
    klass = _accelerator_class_of(accelerator)
    if klass != "rocm" or not fallback_accelerator_for(klass):
        return False
    return rocm_runtime_resolvable() is False


def sd_cpp_vulkan_fallback_enabled() -> bool:
    raw = (os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "auto") or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def fallback_accelerator_for(accelerator: Optional[str]) -> Optional[str]:
    if not sd_cpp_vulkan_fallback_enabled():
        return None
    try:
        want = _installer_module().accelerator_class(accelerator)
    except Exception:  # noqa: BLE001
        want = (accelerator or "").strip().lower()
    nxt = _ACCELERATOR_FALLBACK.get(want)
    return nxt if nxt and nxt != want else None


# Builds that could not RUN here; entries carry a build/cards/runtime fingerprint.
_ACCELERATOR_RUNTIME_FAILURES_KEY = "sd_cpp_accelerator_runtime_failures"
_accelerator_runtime_failures: dict[str, dict] = {}

# AMBIGUOUS failures needed to divert a host (a decisive one is enough alone).
_AMBIGUOUS_FAILURE_STRIKES = 2
# Rule 2 stopped counting Vulkan device-lost errors: an older Vulkan record keeps only its decisive evidence.
_FAILURE_RULE = 2


def _record_under_current_rule(klass: str, record: Optional[dict]) -> Optional[dict]:
    if not isinstance(record, dict) or klass != "vulkan":
        return record
    if int(record.get("rule", 1) or 1) >= _FAILURE_RULE:
        return record
    return _decisive_evidence_only(record)


def _decisive_evidence_only(record: Optional[dict]) -> Optional[dict]:
    """*record* with its ambiguous strikes dropped, or None when nothing decisive is left."""
    if not isinstance(record, dict):
        return record
    proven_cards = {
        name: {"strikes": 0, "proven": True}
        for name, entry in (record.get("per_card") or {}).items()
        if isinstance(entry, dict) and entry.get("proven")
    }
    unscoped = _unscoped_evidence(record)
    # Proof without per-card tallies stays as written: record-wide, and scoped to its ``cards`` when it names any.
    whole = bool(record.get("proven")) and not isinstance(record.get("per_card"), dict)
    if not (proven_cards or unscoped["proven"] or whole):
        return None
    kept = {
        "strikes": 0,
        "proven": True,
        "fingerprint": record.get("fingerprint") or {},
        "rule": _FAILURE_RULE,
    }
    if record.get("cards"):
        # A card left in ``cards`` without its own entry would divert on another card's record-wide proof.
        kept["cards"] = [
            c for c in record["cards"] if whole or unscoped["proven"] or c in proven_cards
        ]
    if proven_cards:
        kept["per_card"] = proven_cards
    kept["unscoped"] = {"strikes": 0, "proven": unscoped["proven"]}
    return kept


def _accelerator_class_of(accelerator: Optional[str]) -> str:
    try:
        return _installer_module().accelerator_class(accelerator)
    except Exception:  # noqa: BLE001
        return (accelerator or "").strip().lower()


def _discovered_managed_root() -> Optional[str]:
    try:
        found = find_sd_cpp_binary()
        return owning_managed_root(found) if found else None
    except Exception:  # noqa: BLE001
        return None


def _accelerator_fingerprint(binary: Optional[str] = None) -> dict:
    """What the note is a fact ABOUT: bundle, GPU runtime, cards. Best-effort; an unreadable component is None, which keeps the record applying."""
    fp: dict = {"bundle": None}
    try:
        # The binary's own root: the finder also serves a legacy tree.
        root = owning_managed_root(binary) if binary else _discovered_managed_root()
        record = _installer_module().read_install_record(root or managed_install_root())
        if isinstance(record, dict):
            tag = record.get("tag")
            fp["bundle"] = str(tag) if tag else None
    except Exception:  # noqa: BLE001
        pass
    fp.update(_host_fingerprint())
    return fp


# Safe to memoise: baked into the torch wheel. The CARDS are deliberately not; see _host_fingerprint.
_RUNTIME_FINGERPRINT_MEMO: Optional[dict] = None
# Kept under its old name, and still wins when set, for a caller that pinned the whole host half.
_HOST_FINGERPRINT_MEMO: Optional[dict] = None


def _reset_host_fingerprint() -> None:
    global _HOST_FINGERPRINT_MEMO, _RUNTIME_FINGERPRINT_MEMO
    _HOST_FINGERPRINT_MEMO = None
    _RUNTIME_FINGERPRINT_MEMO = None


def _runtime_fingerprint() -> dict:
    global _RUNTIME_FINGERPRINT_MEMO
    if _RUNTIME_FINGERPRINT_MEMO is not None:
        return dict(_RUNTIME_FINGERPRINT_MEMO)
    fp: dict = {"runtime": None}
    try:
        import torch  # noqa: PLC0415

        # The ROCm the WHEEL was built against, so a driver or /opt/rocm upgrade does NOT move it.
        runtime = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        fp["runtime"] = str(runtime) if runtime else None
    except Exception:  # noqa: BLE001
        pass
    _RUNTIME_FINGERPRINT_MEMO = dict(fp)
    return fp


def _card_identity(device: dict) -> Optional[str]:
    """How one enumerated card is named in the fingerprint. The marketing name is absent on exactly
    the hosts this feature is about (gfx1151 answers from `sysfs-drm` with ``name = None``), and the
    gfx target is carried ALONGSIDE it: `AMD Radeon(TM) Graphics` is a gfx1103 APU AND a gfx1151."""
    name = device.get("name")
    gfx = device.get("gfx_candidates") or device.get("gfx") or device.get("arch")
    if isinstance(gfx, (list, tuple)):
        # All of it is kept: the family string alone would make gfx1100 and gfx1151 one card.
        gfx = "/".join(str(g) for g in gfx if g)
    if name and gfx:
        return f"{name}@{gfx}"
    if name:
        return str(name)
    if gfx:
        return str(gfx)
    parts = [
        str(device.get(key))
        for key in ("vendor", "index", "memory_total_gb")
        if device.get(key) is not None
    ]
    return ":".join(parts) or None


def selected_card_identity(ordinal: "Optional[int]") -> "Optional[str]":
    """The card at torch-visible *ordinal*, named by the same ``_card_identity`` the fingerprint uses so the two compare. ``None``: every record applies."""
    if ordinal is None:
        return None
    try:
        from utils.hardware.amd import get_hip_id_by_gpu_index

        physical_index, masked = _physical_index_of(ordinal)
        if physical_index is None:
            return None
        inventory = _card_lookup_inventory()
        if (inventory or {}).get("unknown"):
            return None
        devices = _amd_inventory_rows(inventory)
        # A HIP id, masked or not; the inventory is in amd-smi's order, so translate it either way.
        hip_by_row = get_hip_id_by_gpu_index()
        if hip_by_row:
            physical_index = next(
                (row for row, hip in hip_by_row.items() if hip == physical_index), None
            )
            if physical_index is None:
                return None
        elif len(devices) != 1:
            # No mapping (amd-smi missing or pre-6.4): one card is unambiguous, several are a guess.
            return None
        selected = next((d for d in devices if d.get("index") == physical_index), None)
        if selected is None:
            return None
        return _card_identity(selected)
    except Exception:  # noqa: BLE001
        return None


def _host_fingerprint() -> dict:
    """``{"runtime": ..., "gpus": ...}``. Cards re-read every call, NOT memoised with the runtime:
    on a cold cache the non-blocking probe returns the unknown sentinel, and a memo froze it."""
    if _HOST_FINGERPRINT_MEMO is not None:
        return dict(_HOST_FINGERPRINT_MEMO)
    fp: dict = {"gpus": None}
    fp.update(_runtime_fingerprint())
    try:
        from utils.hardware.hardware import get_physical_gpu_inventory

        # Non-blocking: on a load path a wedged driver must never stall this.
        inventory = get_physical_gpu_inventory(block = False)
        if not (inventory or {}).get("unknown"):
            names = sorted(
                identity
                for identity in (
                    _card_identity(d)
                    for d in ((inventory or {}).get("devices") or [])
                    if isinstance(d, dict)
                )
                if identity
            )
            fp["gpus"] = names or None
    except Exception:  # noqa: BLE001
        pass
    return {"runtime": fp.get("runtime"), "gpus": fp.get("gpus")}


def _fingerprint_still_applies(stored: Optional[dict], current: Optional[dict]) -> bool:
    """Whether a record written under ``stored`` still speaks for a host fingerprinted ``current``. Both sides must be known and DIFFER, else a flaky probe flips the host."""
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return True
    for key in ("bundle", "runtime", "gpus"):
        was, now = stored.get(key), current.get(key)
        if was in (None, "", []) or now in (None, "", []):
            continue
        if was != now:
            return False
    return True


def _normalise_failure_record(key: str, value: object) -> Optional[dict]:
    """One persisted entry in the readers' shape, or None. The bare-list shape an early build wrote becomes one decisive strike."""
    if value is True:
        return {"strikes": _AMBIGUOUS_FAILURE_STRIKES, "proven": True, "fingerprint": {}}
    if not isinstance(value, dict):
        return None
    try:
        strikes = int(value.get("strikes", 0) or 0)
    except (TypeError, ValueError):
        strikes = 0
    fingerprint = value.get("fingerprint")
    try:
        rule = int(value.get("rule", 1) or 1)
    except (TypeError, ValueError):
        rule = 1
    record = {
        "strikes": max(strikes, 0),
        "proven": bool(value.get("proven", False)),
        "fingerprint": fingerprint if isinstance(fingerprint, dict) else {},
        "rule": rule,
    }
    cards = [str(card).strip() for card in (value.get("cards") or []) if str(card).strip()]
    if cards:
        record["cards"] = sorted(set(cards))
    per_card: dict[str, dict] = {}
    stored_per_card = value.get("per_card")
    for name, entry in (stored_per_card if isinstance(stored_per_card, dict) else {}).items():
        name = str(name).strip()
        if not name or not isinstance(entry, dict):
            continue
        try:
            entry_strikes = int(entry.get("strikes", 0) or 0)
        except (TypeError, ValueError):
            entry_strikes = 0
        per_card[name] = {
            "strikes": max(entry_strikes, 0),
            "proven": bool(entry.get("proven", False)),
        }
    if per_card:
        record["per_card"] = per_card
    unscoped = value.get("unscoped")
    if isinstance(unscoped, dict):
        record["unscoped"] = _unscoped_evidence({"unscoped": unscoped})
    return record


def _unscoped_evidence(record: Optional[dict]) -> dict:
    """The part of a record no card was named for. It applies to every card, so a later tally for
    one card must add to it rather than replace it."""
    if not isinstance(record, dict):
        return {"strikes": 0, "proven": False}
    stored = record.get("unscoped")
    if isinstance(stored, dict):
        try:
            strikes = int(stored.get("strikes", 0) or 0)
        except (TypeError, ValueError):
            strikes = 0
        return {"strikes": max(strikes, 0), "proven": bool(stored.get("proven", False))}
    per_card = record.get("per_card") if isinstance(record.get("per_card"), dict) else {}
    if not record.get("cards") and not per_card:
        # Never named a card, so all of it is unscoped (the older shapes included).
        return {
            "strikes": int(record.get("strikes", 0) or 0),
            "proven": bool(record.get("proven", False)),
        }
    if not per_card:
        # Cards named without tallies: nothing here can be told apart as unscoped.
        return {"strikes": 0, "proven": False}
    scoped = sum(int((entry or {}).get("strikes", 0) or 0) for entry in per_card.values())
    return {"strikes": max(int(record.get("strikes", 0) or 0) - scoped, 0), "proven": False}


def _stored_accelerator_runtime_failures() -> dict[str, dict]:
    """The persisted map, or an empty one. Never raises: an unreadable store only costs the preference."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Owner-scoped: which sd.cpp build runs here is a fact about the machine, not the user.
        stored = run_as(OWNER, get_app_setting, _ACCELERATOR_RUNTIME_FAILURES_KEY, None)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(stored, str):
        # JSON column: a row saved pre-serialised reads back as the text of a mapping.
        try:
            import json
            stored = json.loads(stored)
        except ValueError:
            return {}
    if isinstance(stored, (list, tuple, set)):
        # The shape an early build wrote: a plain list of accelerator names.
        stored = {str(item).strip().lower(): True for item in stored if str(item).strip()}
    if not isinstance(stored, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in stored.items():
        name = str(key).strip().lower()
        try:
            record = _normalise_failure_record(name, value) if name else None
        except Exception:  # noqa: BLE001 - a malformed entry costs only its own preference
            record = None
        if record is not None:
            out[name] = record
    return out


def _persist_accelerator_runtime_failures(records: dict[str, dict]) -> None:
    try:
        from storage.studio_db import upsert_app_settings
        from utils.account_context import OWNER, run_as
        run_as(OWNER, upsert_app_settings, {_ACCELERATOR_RUNTIME_FAILURES_KEY: records})
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not persist the sd.cpp accelerator failure notes: %s", exc)


def note_accelerator_runtime_failure(
    accelerator: Optional[str],
    *,
    proven: bool = True,
    fingerprint: Optional[dict] = None,
    card: Optional[str] = None,
) -> None:
    """Only for accelerators with a fallback rung (noting cpu would claim sd.cpp cannot run at all); pass fingerprint if the build was already replaced."""
    klass = _accelerator_class_of(accelerator)
    if not klass or klass not in _ACCELERATOR_FALLBACK:
        return
    fingerprint = dict(fingerprint) if isinstance(fingerprint, dict) else _accelerator_fingerprint()
    records = _stored_accelerator_runtime_failures()
    records.update({k: v for k, v in _accelerator_runtime_failures.items() if k not in records})
    previous = _record_under_current_rule(klass, records.get(klass))
    if previous is not None and not _fingerprint_still_applies(
        previous.get("fingerprint"), fingerprint
    ):
        previous = None
    if previous is not None:
        # Keep known fields: an unreadable None must not erase what could invalidate this record.
        fingerprint = _fingerprint_with_known_fields_kept(previous.get("fingerprint"), fingerprint)
    strikes = (previous or {}).get("strikes", 0) + 1
    cards = [c for c in ((previous or {}).get("cards") or []) if c]
    if card and card not in cards:
        cards = sorted([*cards, card])
    # Per-card tallies: one card's proof or strikes must not convict another.
    previous_per_card = (previous or {}).get("per_card")
    per_card = {
        name: dict(entry)
        for name, entry in (
            previous_per_card if isinstance(previous_per_card, dict) else {}
        ).items()
        if isinstance(entry, dict)
    }
    unscoped = _unscoped_evidence(previous)
    if card:
        seen = per_card.get(card) or {}
        per_card[card] = {
            "strikes": int(seen.get("strikes", 0) or 0) + 1,
            "proven": bool(proven) or bool(seen.get("proven", False)),
        }
    else:
        unscoped = {
            "strikes": unscoped["strikes"] + 1,
            "proven": bool(proven) or unscoped["proven"],
        }
    record = {
        "strikes": strikes,
        # The union, for a caller that names no card and for the settings report.
        "proven": bool(proven) or bool((previous or {}).get("proven", False)),
        "fingerprint": fingerprint,
        "rule": _FAILURE_RULE,
    }
    if cards:
        record["cards"] = cards
    if per_card:
        record["per_card"] = per_card
    if unscoped["strikes"] or unscoped["proven"]:
        record["unscoped"] = unscoped
    if previous == record:
        return
    records[klass] = record
    _accelerator_runtime_failures[klass] = record
    _persist_accelerator_runtime_failures(records)


def _fingerprint_with_known_fields_kept(
    previous: Optional[dict], current: Optional[dict]
) -> Optional[dict]:
    """*current*, gaps filled from *previous*. Only valid once the two are known not to contradict."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return current
    merged = dict(current)
    for key, value in previous.items():
        if value is not None and merged.get(key) is None:
            merged[key] = value
    return merged


def _record_diverts(
    record: Optional[dict],
    fingerprint: Optional[dict] = None,
    card: Optional[str] = None,
) -> bool:
    """Whether one record moves this host off its own accelerator. For a named *card*, evidence from
    other named cards says nothing; evidence no card was named for still applies."""
    if not isinstance(record, dict):
        return False
    if not _fingerprint_still_applies(
        record.get("fingerprint"),
        _accelerator_fingerprint() if fingerprint is None else fingerprint,
    ):
        return False
    known_cards = [c for c in (record.get("cards") or []) if c]
    own = (record.get("per_card") or {}).get(card) if card else None
    if card and (isinstance(own, dict) or (known_cards and card not in known_cards)):
        own = own if isinstance(own, dict) else {}
        unscoped = _unscoped_evidence(record)
        if own.get("proven") or unscoped["proven"]:
            return True
        strikes = int(own.get("strikes", 0) or 0) + unscoped["strikes"]
        return strikes >= _AMBIGUOUS_FAILURE_STRIKES
    if record.get("proven"):
        return True
    return int(record.get("strikes", 0) or 0) >= _AMBIGUOUS_FAILURE_STRIKES


def accelerator_runtime_failed(accelerator: Optional[str], card: Optional[str] = None) -> bool:
    """Whether the ``accelerator`` build is already known not to run here, under a fingerprint that still describes this host, and on the *card* named."""
    klass = _accelerator_class_of(accelerator)
    if not klass:
        return False
    fingerprint = _accelerator_fingerprint()
    if _record_diverts(
        _applicable_record(klass, _accelerator_runtime_failures.get(klass)), fingerprint, card
    ):
        return True
    return _record_diverts(
        _applicable_record(klass, _stored_accelerator_runtime_failures().get(klass)),
        fingerprint,
        card,
    )


def _applicable_record(klass: str, record: Optional[dict]) -> Optional[dict]:
    """The evidence in *record* that still counts on this host: old-rule ambiguous Vulkan strikes are
    dropped, and outside the CPU-torch route only decisive Vulkan evidence diverts."""
    record = _record_under_current_rule(klass, record)
    if klass == "vulkan" and not _vulkan_cpu_rung_applies():
        return _decisive_evidence_only(record)
    return record


def off_torch_build_mismatch(off_torch: Any, binary: Optional[str]) -> Optional[str]:
    """Why ``binary`` is not provably the off-torch card's build, else None. Any other build ignores
    CUDA_VISIBLE_DEVICES (Vulkan) or reads it as HIP's mask (ROCm), so it would run on torch's cards
    past the arbiter and the training guard; an unrecorded one (SD_CLI_PATH) could be either."""
    if off_torch is None or not binary:
        return None
    klass = _installed_accelerator_of(binary)
    if klass != _accelerator_class_of(off_torch.accelerator):
        return klass or "unrecorded"
    return None


def _refuse_off_torch_build_mismatch(off_torch: Any, binary: Optional[str]) -> None:
    klass = off_torch_build_mismatch(off_torch, binary)
    if klass:
        raise RuntimeError(
            f"UNSLOTH_DIFFUSION_SD_CPP_DEVICE={off_torch.label} needs the "
            f"{off_torch.accelerator} stable-diffusion.cpp build, but the installed one is "
            f"{klass}. Let the {off_torch.accelerator} build install, or unset the setting, "
            "then load again."
        )


def usable_or_recorded_failure(
    binary,
    requested,
    card = None,
):
    """Checked against the request: offline ensure may hand back another accelerator's build, which passes probes and dies mid-render."""
    if not binary:
        return binary
    try:
        klass = _installed_accelerator_of(binary)
        if not klass:
            return binary
        if requested is not None and _accelerator_class_of(requested) == klass:
            return binary
        if accelerator_runtime_failed(klass, card):
            logger.warning(
                "sd_cpp.recorded_failure_returned: the ensure handed back the %s build in "
                "place of %s, and this host has already recorded it as unrunnable; not "
                "using it",
                klass,
                requested,
            )
            return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not check the sd.cpp accelerator record: %s", exc)
    return binary


def accelerator_runtime_failure_state() -> dict:
    """What the settings route reports: every record, whether it diverts, and its fingerprint."""
    fingerprint = _accelerator_fingerprint()
    records = _stored_accelerator_runtime_failures()
    records.update(_accelerator_runtime_failures)
    # Only the REPORT is gated on the knob, so it cannot say `enabled: false, diverting: true`.
    enabled = sd_cpp_vulkan_fallback_enabled()
    out = []
    for klass in sorted(records):
        record = records[klass]
        out.append(
            {
                "accelerator": klass,
                "fallback": _ACCELERATOR_FALLBACK.get(klass),
                "strikes": int(record.get("strikes", 0) or 0),
                "proven": bool(record.get("proven", False)),
                "diverting": enabled
                and _record_diverts(_applicable_record(klass, record), fingerprint),
                "stale": not _fingerprint_still_applies(record.get("fingerprint"), fingerprint),
            }
        )
    return {
        "records": out,
        "enabled": enabled,
        "diverting": any(r["diverting"] for r in out),
    }


def clear_accelerator_runtime_failures() -> None:
    """Forget every note (``DELETE /settings/diffusion-accelerator-fallback``). BOTH halves: the in-process mirror would otherwise keep diverting."""
    _accelerator_runtime_failures.clear()
    _persist_accelerator_runtime_failures({})


# DECISIVE: the build has no code for this card. First two are #9278 verbatim.
_ACCELERATOR_DECISIVE_FAILURE_MARKERS: tuple[str, ...] = (
    "hipblassetstream",
    "cublas_status_invalid_value",
    "hiperrornobinaryforgpu",
    "no kernel image is available",
    "invalid device function",
    # vulkan-hpp's spellings and the C enum: this driver cannot run the Vulkan build.
    "errorincompatibledriver",
    "vk_error_incompatible_driver",
)

# AMBIGUOUS: GPU faults that do not prove the build is the cause.
_ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS: tuple[str, ...] = (
    "unspecified launch failure",
    "hip error",
    "rocm error",
    "rocblas error",
    "memory access fault by gpu node",
    # Not VK_ERROR_DEVICE_LOST: the Vulkan spec makes it recoverable by recreating the device (TDR, scheduler reset).
    "errorinitializationfailed",
    "vk_error_initialization_failed",
)

# Windows loader deaths print nothing, so only the exit status shows them. DECISIVE.
_WINDOWS_IMAGE_LOAD_FAILURE_STATUSES: tuple[int, ...] = (
    0xC0000135,  # STATUS_DLL_NOT_FOUND: a DLL the image imports is missing
    0xC0000139,  # STATUS_ENTRYPOINT_NOT_FOUND: it is present but the wrong build
    0xC0000142,  # STATUS_DLL_INIT_FAILED: it loaded and its DllMain refused
)

_EXIT_STATUS_RE = re.compile(r"sd-cli exited (-?\d+)")


def output_shows_image_load_failure(text: Optional[str]) -> bool:
    """True when the exit status means the executable never started. The signed reading of the unsigned NTSTATUS is accepted too; both spellings reach python."""
    if not text:
        return False
    for raw in _EXIT_STATUS_RE.findall(str(text)):
        try:
            code = int(raw)
        except ValueError:  # pragma: no cover -- the pattern only matches digits
            continue
        if code < 0:
            code += 1 << 32
        if code in _WINDOWS_IMAGE_LOAD_FAILURE_STATUSES:
            return True
    return False


# Deliberately in neither list: sd.cpp's "Cannot set backend to CK" warning, which builds that render perfectly well also print.
_ACCELERATOR_RUNTIME_FAILURE_MARKERS: tuple[str, ...] = (
    _ACCELERATOR_DECISIVE_FAILURE_MARKERS + _ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS
)

# Checked BEFORE either predicate: "ROCm error: out of memory" contains "rocm error".
_ACCELERATOR_CAPACITY_FAILURE_MARKERS: tuple[str, ...] = (
    "out of memory",
    "outofmemory",
    "out of device memory",
    "out_of_device_memory",
    "out_of_host_memory",
    "hiperroroutofmemory",
    "cudaerrormemoryallocation",
    "failed to allocate",
    "cannot allocate memory",
    "memory allocation failed",
    "alloc_failed",
    "allocation failure",
    "insufficient memory",
    "not enough memory",
    # sd.cpp's graph-cut executor when a segment's device memory cannot be made available (ggml_runner.cpp's
    # fail_segment). Not the model manager's "cannot make enough memory available" warning: it precedes recoveries
    # too, and renders that complete print it.
    "failed during workspace capacity check",
    "failed during workspace allocation",
    "failed during allocated capacity check",
)


def output_shows_capacity_failure(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_CAPACITY_FAILURE_MARKERS)


# A lost Vulkan device: recoverable by creating a new device, so neither a build strike nor a fatal error.
_VULKAN_DEVICE_LOST_MARKERS: tuple[str, ...] = ("errordevicelost", "vk_error_device_lost")


def output_shows_vulkan_device_lost(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in _VULKAN_DEVICE_LOST_MARKERS)


# Appended when a lost Vulkan device repeats and no CPU fallback applies. Must match no failure marker.
_REPEATED_DEVICE_LOSS_HINT = (
    "The GPU lost its Vulkan device twice in a row (usually a driver reset during the render). Try Low VRAM "
    "memory mode, a smaller resolution or a smaller quant, or load the model again to retry on the GPU."
)


def _device_loss_cpu_fallback_applies(state: Any) -> bool:
    """Whether a repeated Vulkan device loss may move this load to the CPU backend: only the CPU-torch Vulkan
    route (the one with a CPU rung, as on main), the Vulkan-to-CPU fallback enabled, and no user-picked GPU."""
    if getattr(state, "device", None) != "vulkan" or getattr(state, "off_torch_device", None):
        return False
    if getattr(state, "gpu_explicit", False):
        return False
    return fallback_accelerator_for("vulkan") == "cpu" and _vulkan_cpu_rung_applies()


def output_shows_accelerator_failure(text: Optional[str]) -> bool:
    """True when sd-cli output names a failure of the GPU BUILD rather than of the request; an unrecognised failure persists nothing."""
    if not text:
        return False
    if output_shows_capacity_failure(text):
        return False
    if output_shows_image_load_failure(text):
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_RUNTIME_FAILURE_MARKERS)


def output_shows_decisive_accelerator_failure(text: Optional[str]) -> bool:
    """True when the output names the BUILD having no code for this card. Only these divert a host on one occurrence."""
    if not text:
        return False
    if output_shows_capacity_failure(text):
        return False
    if output_shows_image_load_failure(text):
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_DECISIVE_FAILURE_MARKERS)


# Imported by the Windows ROCm bundle but not shipped in it; only the AMD HIP SDK provides it.
_WINDOWS_ROCM_HOST_DLLS: tuple[str, ...] = ("hipblas.dll",)
_windows_rocm_skip_logged = False


def _on_windows() -> bool:
    return os.name == "nt"


def windows_rocm_build_mode() -> str:
    """``UNSLOTH_SD_CPP_WINDOWS_ROCM``: ``auto`` (ROCm only if the HIP SDK is on PATH), ``1`` always, ``0`` never."""
    raw = (os.environ.get("UNSLOTH_SD_CPP_WINDOWS_ROCM", "auto") or "").strip().lower()
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    if raw in ("0", "off", "false", "no", "never"):
        return "off"
    return "auto"


def _windows_dll_search_dirs() -> list[str]:
    dirs: list[str] = []
    system_root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    dirs.append(os.path.join(system_root, "System32"))
    dirs.append(system_root)
    dirs.extend(d.strip().strip('"') for d in os.environ.get("PATH", "").split(os.pathsep))
    return [d for d in dirs if d]


def windows_hip_runtime_missing_dlls() -> Optional[list[str]]:
    """Missing HIP SDK DLLs, or None off Windows. File search only: loading hipblas.dll here would clash with torch's rocBLAS."""
    if not _on_windows():
        return None
    dirs = _windows_dll_search_dirs()
    missing = []
    for dll in _WINDOWS_ROCM_HOST_DLLS:
        if not any(os.path.isfile(os.path.join(d, dll)) for d in dirs):
            missing.append(dll)
    return missing


def windows_rocm_build_skipped() -> bool:
    """Whether a Windows ROCm request goes to Vulkan (HIP SDK missing would fail with 0xC0000135)."""
    if not _on_windows() or not sd_cpp_vulkan_fallback_enabled():
        return False
    mode = windows_rocm_build_mode()
    if mode == "force":
        return False
    if mode == "off":
        return True
    return bool(windows_hip_runtime_missing_dlls())


def preferred_accelerator(accelerator: Optional[str], card: Optional[str] = None) -> str:
    """``accelerator``, or its fallback once shown unrunnable here. Applied at the TOP of an ensure ladder, so a crashed build is not installed and probed again.

    On Windows a ROCm request starts at Vulkan when ``windows_rocm_build_skipped``."""
    global _windows_rocm_skip_logged
    chosen = accelerator or "auto"
    klass = _accelerator_class_of(accelerator) or chosen
    if klass == "rocm" and windows_rocm_build_skipped():
        if not _windows_rocm_skip_logged:
            _windows_rocm_skip_logged = True
            logger.info(
                "sd_cpp.windows_rocm_skipped: using the Vulkan stable-diffusion.cpp build; the ROCm "
                "build needs the AMD HIP SDK runtime (%s) on PATH (UNSLOTH_SD_CPP_WINDOWS_ROCM=1 "
                "to use it anyway)",
                ", ".join(windows_hip_runtime_missing_dlls() or _WINDOWS_ROCM_HOST_DLLS),
            )
        chosen = klass = "vulkan"
    seen: set[str] = set()
    # Walk the ladder (rocm -> vulkan -> cpu) past every rung already shown unrunnable.
    while klass not in seen:
        seen.add(klass)
        nxt = fallback_accelerator_for(klass)
        if not (nxt and accelerator_runtime_failed(klass, card)):
            break
        chosen = klass = nxt
    return chosen


def _incomplete_tree_replacement(exc: BaseException) -> bool:
    """A half-replaced tree must not be memoised as a failed upgrade, or the mixed tree gets served."""
    try:
        return isinstance(exc, _installer_module().SupersededBinaryError)
    except Exception:  # noqa: BLE001 -- cannot tell -> an ordinary failure
        return False


def _tree_in_use(backend: Any) -> bool:
    """Covers the resident server, a spawned-but-uncommitted _pending_server, and a cancelling
    generation."""
    if backend is None:
        return False
    state = getattr(backend, "_state", None)
    if state is not None and getattr(state, "server", None) is not None:
        return True  # the resident sd-server is executing its own file
    if getattr(backend, "_pending_server", None) is not None:
        return True  # a server is starting from that same file
    if getattr(backend, "_stopping_servers", 0):
        return True  # unpublished, but stop() has not returned: the process is still alive
    return getattr(backend, "_active_generate_cancel", None) is not None


def _managed_tree_in_use() -> bool:
    """Replacing a running binary fails (ETXTBSY), so upgrades wait; checked here as the router calls ensure before begin_load; unlocked read is deliberate."""
    return _tree_in_use(_sd_cpp_backend) or _external_tree_holder_alive()


# Other backends' processes in the managed tree (H3 sd-server); weak so they cannot pin it.
_external_tree_holders: "weakref.WeakSet[Any]" = weakref.WeakSet()


def register_tree_holder(holder: Any) -> None:
    with _tree_state:
        _external_tree_holders.add(holder)


def unregister_tree_holder(holder: Any) -> None:
    with _tree_state:
        _external_tree_holders.discard(holder)
        _tree_state.notify_all()


def _external_tree_holder_alive() -> bool:
    for holder in list(_external_tree_holders):
        try:
            if holder.is_alive():
                return True
        except Exception:  # noqa: BLE001 -- a broken holder must not wedge installs either way
            continue
    return False


def _accelerator_changed(binary: str, accelerator: str) -> bool:
    """Replaces only an owned copy of another accelerator; an unrecorded install is kept for CPU, since reinstalling legacy installs would redownload."""
    root = owning_managed_root(binary)
    if root is None:
        return False
    if _managed_tree_in_use():
        return False  # an install now would overwrite a running binary; the load retries after teardown
    try:
        mod = _installer_module()
        want = mod.accelerator_class(accelerator)
        if want in _failed_accelerator_upgrades:
            return False
        return _record_mismatch(mod, root, want)
    except Exception:  # noqa: BLE001 -- cannot tell -> keep the existing binary
        return False


def _record_mismatch(mod, root: Path, want: str) -> bool:
    """True when ``root``'s install record names an accelerator other than ``want``. Unrecorded is
    unknown, and on a CPU target unknown is left alone (see ``_accelerator_changed``)."""
    have = mod.installed_accelerator(root)
    if want == "cpu":
        return have is not None and have != "cpu"
    return have != want


def _superseded_legacy_server(binary: Optional[str], accelerator: str) -> bool:
    """Ignores only a mismatched legacy server, and only when the current install records ships_server false; a matching legacy server is still used."""
    root = owning_managed_root(binary)
    if root is None:
        return False
    current = managed_install_root()
    try:
        if root.resolve() == current.resolve():
            return False
    except OSError:
        return False
    try:
        mod = _installer_module()
        want = mod.accelerator_class(accelerator)
        # Superseded = built for another accelerator, or (with the pin upgrade on) for an older pin.
        legacy_stale = _record_mismatch(mod, root, want) or (
            not _pin_upgrade_disabled() and mod.install_is_stale(root)
        )
        if (
            not legacy_stale
            or _record_mismatch(mod, current, want)
            or mod.install_is_stale(current)
        ):
            return False
        return mod.installed_ships_server(current) is False
    except Exception:  # noqa: BLE001 -- cannot tell
        return False


_ARCH_MARKER_CACHE: dict[tuple[str, int, int, str], bool] = {}


def binary_carries_marker(
    binary: Optional[str],
    marker: Optional[str],
    *,
    unreadable: bool = True,
) -> bool:
    """No marker, or an unreadable binary, passes (a failed stat must not refuse a working host) unless the caller passes unreadable=False."""
    if not marker:
        return True
    if not binary:
        return False
    try:
        st = os.stat(binary)
        key = (str(binary), int(st.st_size), int(st.st_mtime_ns), marker)
    except OSError:
        return unreadable
    hit = _ARCH_MARKER_CACHE.get(key)
    if hit is not None:
        return hit
    needle = marker.encode()
    chunk = 8 << 20
    found = False
    try:
        with open(binary, "rb") as fh:
            tail = b""
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                if needle in tail + block:
                    found = True
                    break
                tail = block[-(len(needle) - 1) :] if len(needle) > 1 else b""
    except OSError:
        return unreadable
    _ARCH_MARKER_CACHE[key] = found
    return found


def _native_output_image(fam: Any, im: Any) -> Any:
    """A rendered image as the gallery keeps it: alpha for a family that generates transparency, as
    the diffusers engine returns it, else RGB."""
    if getattr(fam, "condition_image_mode", "RGB") == "RGBA" and im.mode in ("RGBA", "LA", "P"):
        return im.convert("RGBA")
    return im.convert("RGB")


def _layer_count(fam: Any) -> int:
    return int(getattr(fam, "layer_count", 0) or 0)


def _layered_canvas_size(fam: Any, size: tuple[int, int]) -> tuple[int, int]:
    """Copy of diffusion._layered_canvas (the pipeline's calculate_dimensions); this module never imports torch."""
    import math

    iw, ih = size
    area = float(getattr(fam, "layer_resolution", 640) or 640) ** 2
    ratio = float(iw) / float(max(1, ih))
    width = math.sqrt(area * ratio)
    height = width / ratio
    return max(32, int(round(width / 32)) * 32), max(32, int(round(height / 32)) * 32)


def _keep_layers(items: list, layers: int) -> list:
    """Drop sd.cpp's input reconstruction (first of each layers + 1 group), as the diffusers pipeline does."""
    if not layers:
        return list(items)
    per = layers + 1
    return [item for i, item in enumerate(items) if i % per]


def _family_reads_vision(fam: Any) -> bool:
    """Whether ``fam``'s native encoders include a vision projector (``llm_vision``)."""
    return any(kind == "llm_vision" for _r, _f, kind in getattr(fam, "sd_cpp_text_encoders", ()))


# Marker of builds keeping reference alpha (e112ab5) and no centre-crop (78557f8/e012065).
_REFERENCE_FIDELITY_MARKER = "error: allocate memory for channel promotion"


def _native_condition_images(
    fam: Any,
    init_image: str,
    reference_images: Optional[list[str]],
    localized_edit: Any,
    width: Optional[int],
    height: Optional[int],
    *,
    full_fidelity: bool,
    pad_to_output: bool,
    source_sized: bool = False,
) -> tuple[int, int, list[bytes]]:
    """Older builds: flatten references over white (else transparent pixels become noise); with pad_to_output pad, do not crop (white, black for masks)."""
    import io
    import math

    from PIL import Image

    from core.inference.diffusion_conditioning import (
        MIN_OUTPUT_SIDE,
        check_output_size,
        decode_condition_images,
        match_source_size,
    )

    images = decode_condition_images(fam, init_image, reference_images, localized_edit)
    if source_sized and _layer_count(fam):
        # Canvas from the 16 px snapped source, as the diffusers engine picks it.
        sw, sh = images[0].size
        snapped = (max(16, int(round(sw / 16)) * 16), max(16, int(round(sh / 16)) * 16))
        width, height = _layered_canvas_size(fam, snapped)
        if images[0].size != (width, height):
            images[0] = images[0].resize((width, height), Image.LANCZOS)
    elif source_sized:
        multiple = int(getattr(fam, "dimension_multiple", 16) or 16)
        sw, sh = images[0].size
        max_side = int(getattr(fam, "max_output_side", 2048) or 2048)
        max_pixels = int(getattr(fam, "max_output_pixels", 2048 * 2048) or 2048 * 2048)
        # Fit the bounds instead of refusing: the caller has no width / height to change on an edit-only family.
        up = max(1.0, MIN_OUTPUT_SIDE / float(min(sw, sh)))
        fit = min(up, max_side / float(max(sw, sh)), math.sqrt(max_pixels / float(sw * sh)))
        # Same rounding as diffusion._snap_to_multiple.
        floor = -(-MIN_OUTPUT_SIDE // multiple) * multiple if up > 1.0 else multiple
        width = max(floor if sw <= sh else multiple, int(round(sw * fit / multiple)) * multiple)
        height = max(floor if sh <= sw else multiple, int(round(sh * fit / multiple)) * multiple)
        width = min(width, max_side // multiple * multiple)
        height = min(height, max_side // multiple * multiple)
        while width * height > max_pixels:
            if width >= height:
                width -= multiple
            else:
                height -= multiple
        if (width, height) != (sw, sh):
            images[0] = images[0].resize((width, height), Image.LANCZOS)
    elif width is None or height is None:
        width, height = match_source_size(fam, images[0].size, 1024)
    check_output_size(fam, int(width), int(height))
    target = float(width) / float(height)
    mask_index = 1 if getattr(localized_edit, "mode", None) == "mask" else None
    blobs: list[bytes] = []
    for index, img in enumerate(images):
        if full_fidelity:
            img = img if img.mode in ("RGB", "RGBA") else img.convert("RGBA")
        elif img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask = rgba.getchannel("A"))
            img = flat
        else:
            img = img.convert("RGB")
        if pad_to_output and not full_fidelity:
            iw, ih = img.size
            ratio = iw / float(ih)
            if abs(ratio - target) > 0.01 * target:
                pw, ph = (round(ih * target), ih) if ratio < target else (iw, round(iw / target))
                fill = (0, 0, 0) if index == mask_index else (255, 255, 255)
                canvas = Image.new("RGB", (max(pw, iw), max(ph, ih)), fill)
                canvas.paste(img, ((canvas.width - iw) // 2, (canvas.height - ih) // 2))
                img = canvas
        buf = io.BytesIO()
        # Transport only (sd.cpp decodes it back): zlib level 1 is lossless and faster than Pillow's default.
        img.save(buf, format = "PNG", compress_level = 1)
        blobs.append(buf.getvalue())
    return int(width), int(height), blobs


def sd_cpp_binary_runs_family(binary: Optional[str], fam: Any) -> bool:
    """Checks the binary for the family's arch marker; family_sd_cpp_supported only checks assets."""
    return binary_carries_marker(binary, getattr(fam, "sd_cpp_arch_marker", None))


def managed_sd_cpp_accelerator() -> Optional[str]:
    """The accelerator class of the managed sd-server install, else of the managed sd-cli (an
    unowned SD_SERVER_PATH the router drops leaves the CLI to run), or None."""
    try:
        for binary in (find_sd_server_binary(), find_sd_cpp_binary()):
            accelerator = _installed_accelerator_of(binary) if binary else None
            if accelerator:
                return accelerator
    except Exception:  # noqa: BLE001
        pass
    return None


def sd_cpp_binaries_all_cpu() -> bool:
    """Whether every sd.cpp binary a load could run (server, else CLI) is a managed CPU install.
    An unowned SD_SERVER_PATH / SD_CLI_PATH build may render on a GPU, so it never counts."""
    try:
        binaries = [b for b in (find_sd_server_binary(), find_sd_cpp_binary()) if b]
        return bool(binaries) and all(_installed_accelerator_of(b) == "cpu" for b in binaries)
    except Exception:  # noqa: BLE001
        return False


def _installed_accelerator_of(binary: Optional[str]) -> Optional[str]:
    """Not _accelerator_changed: the load only needs to know if the tree was replaced underneath it."""
    # Root the binary is IN: a legacy tree would read "unrecorded" on both sides.
    root = owning_managed_root(binary)
    if root is None:
        return None
    try:
        return _installer_module().installed_accelerator(root)
    except Exception:  # noqa: BLE001 -- cannot tell; the comparison sees None on both sides
        return None


def note_accelerator_failure_from_output(
    binary: Optional[str],
    output: str,
    *,
    source: str = "diffusion",
    card: Optional[str] = None,
) -> None:
    """Record that the sd.cpp build ``binary`` came from cannot run here, when its own output says
    so. Never raises. Here rather than in the video module it grew up in: the image path runs the
    same sd-cli and recorded none of these."""
    if not binary or not output:
        return
    try:
        if not output_shows_accelerator_failure(output):
            return
        accelerator = _installed_accelerator_of(binary)
        if not accelerator or not fallback_accelerator_for(accelerator):
            return
        decisive = output_shows_decisive_accelerator_failure(output)
        logger.warning(
            "%s.sd_cpp_accelerator_runtime_failure: the %s stable-diffusion.cpp build failed "
            "on this host mid-generation (%s); the %s build is the fallback",
            source,
            accelerator,
            "the message names the build, acting on it now"
            if decisive
            else "the message does not establish the build as the cause, counting it",
            fallback_accelerator_for(accelerator),
        )
        note_accelerator_runtime_failure(accelerator, proven = decisive, card = card)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not record the sd.cpp accelerator failure: %s", exc)


def note_unlaunchable_accelerator_build(
    binary: Optional[str],
    *,
    source: str = "diffusion",
    card: Optional[str] = None,
) -> None:
    """Record that the build ``binary`` came from could not be LAUNCHED here; the other recorder
    reads output a loader death never writes. Ambiguous: a missing execute bit looks the same."""
    if not binary:
        return
    try:
        accelerator = _installed_accelerator_of(binary)
        if not accelerator or not fallback_accelerator_for(accelerator):
            return
        logger.warning(
            "%s.sd_cpp_accelerator_launch_failure: the %s stable-diffusion.cpp build could "
            "not be launched on this host; counting it, the %s build is the fallback",
            source,
            accelerator,
            fallback_accelerator_for(accelerator),
        )
        note_accelerator_runtime_failure(accelerator, proven = False, card = card)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not record the sd.cpp launch failure: %s", exc)


def _install_kwargs(accelerator: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"accelerator": accelerator}
    if _accelerator_class_of(accelerator) == "cuda":
        from core.inference.sd_cpp_engine import host_cuda_runtime_major
        major = host_cuda_runtime_major()
        if major:
            kwargs["cuda_major"] = major
    return kwargs


def _unrunnable_build(exc: BaseException) -> bool:
    try:
        return isinstance(exc, _installer_module().UnrunnableBuildError)
    except Exception:  # noqa: BLE001 -- an installer without the probe raises nothing of the kind
        return False


def _install_down_the_ladder(_install: Callable[..., Any], accelerator: str) -> tuple[Any, str]:
    """Install ``accelerator``, stepping down (rocm -> vulkan -> cpu) while the loader refuses the bundle.
    Returns ``(path, accelerator installed)``."""
    current = accelerator
    tried: set[str] = set()
    refused: list[str] = []

    def _note_refused(fingerprint: dict) -> None:
        for klass in refused:
            note_accelerator_runtime_failure(klass, proven = True, fingerprint = fingerprint)

    while True:
        try:
            path = _install(**_install_kwargs(current))
        except Exception as exc:  # noqa: BLE001
            if not _unrunnable_build(exc):
                if refused:
                    _note_refused(_pin_fingerprint())
                raise
            klass = _accelerator_class_of(current)
            tried.add(klass)
            refused.append(klass)
            nxt = fallback_accelerator_for(klass)
            logger.warning(
                "sd_cpp.unrunnable_build: %s; %s",
                exc,
                f"installing the {nxt} build instead" if nxt else "no fallback build to try",
            )
            if not nxt or _accelerator_class_of(nxt) in tried:
                _note_refused(_pin_fingerprint())
                raise
            current = nxt
            continue
        # Keyed on the installed build: a fallback may carry a tag other than the pin.
        if refused:
            _note_refused(_accelerator_fingerprint(path) if path else _pin_fingerprint())
        return path, current


def _pin_fingerprint() -> dict:
    try:
        want = _installer_module()._pinned_tag()
    except Exception:  # noqa: BLE001
        want = None
    return {"bundle": want, **_host_fingerprint()}


def ensure_sd_cpp_binary(*, allow_install: bool = True, accelerator: str = "cpu") -> Optional[str]:
    """Never raises: None means no usable binary and tells the router to fall back to diffusers."""
    found = find_sd_cpp_binary()
    usable = bool(found) and _usable_or_discard_managed(found)
    if usable and not _needs_reinstall(found, accelerator):
        return found
    if not allow_install:
        return found
    with _install_lock:
        found = find_sd_cpp_binary()
        usable = bool(found) and _usable_or_discard_managed(found)
        if usable and not _needs_reinstall(found, accelerator):
            return found
        # A wrong-accelerator binary beats none, so a failed reinstall (no asset, no network) keeps it.
        fallback = found if usable else None
        try:
            _install = _installer_module().install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-cli installer import failed: %s", exc)
            return fallback
        # Claim for the whole install incl. download, or a generation is overwritten mid-run.
        with _tree_claimed_for_install() as claimed:
            if not claimed:
                return fallback  # something is running in there; retry on a later load
            try:
                path, _used = _install_down_the_ladder(_install, accelerator)
                logger.info("sd-cli installed at %s", path)
                return str(path)
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-cli auto-install failed: %s", exc)
                if _incomplete_tree_replacement(exc):
                    # Partial sweep may have removed the fallback; re-find via the usability gate, since a fresh POSIX copy lacks its exec bit until _make_executable runs.
                    refound = find_sd_cpp_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None:
                    _note_failed_upgrade(accelerator)
                    _note_failed_pin_upgrade(accelerator)
                return fallback


def ensure_sd_server_binary(
    *, allow_install: bool = True, accelerator: str = "cpu"
) -> Optional[str]:
    """Installs sd-server even when an older sd-cli exists, since the archive ships both binaries."""
    found = find_sd_server_binary()
    usable = bool(found) and _usable_or_discard_managed(found)
    # Before _accelerator_changed, which says unchanged while the tree is in use.
    if usable and _superseded_legacy_server(found, accelerator):
        return None
    if usable and not _needs_reinstall(found, accelerator):
        return found
    if not allow_install:
        return found
    with _install_lock:
        found = find_sd_server_binary()
        usable = bool(found) and _usable_or_discard_managed(found)
        if usable and _superseded_legacy_server(found, accelerator):
            return None
        if usable and not _needs_reinstall(found, accelerator):
            return found
        # Keep a usable wrong-accelerator server if the matching one cannot be fetched
        fallback = found if usable else None
        try:
            _install = _installer_module().install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-server installer import failed: %s", exc)
            return fallback
        with _tree_claimed_for_install() as claimed:
            if not claimed:
                return fallback  # something is running in there; retry on a later load
            try:
                _path, accelerator = _install_down_the_ladder(_install, accelerator)
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-server auto-install failed: %s", exc)
                # Also when only the CLI survives: the following ensure_sd_cpp_binary probe would re-download.
                if _incomplete_tree_replacement(exc):
                    # The fallback may name a copy the partial sweep removed; a fresh extract skipped _make_executable.
                    refound = find_sd_server_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None or find_sd_cpp_binary() is not None:
                    _note_failed_upgrade(accelerator)
                    _note_failed_pin_upgrade(accelerator)
                return fallback
        installed = find_sd_server_binary()
        # Finder may return a legacy server for another accelerator; return None, not the fallback, so the freshly installed sd-cli runs.
        if installed and _needs_reinstall(installed, accelerator):
            return None
        return installed or fallback


@dataclass(frozen = True)
class _SdState:
    """The loaded native checkpoint: resolved asset paths + run settings. ``server`` is the resident
    ``sd-server`` process (the model is loaded once, inside it) when ``mode == "server"``; in the
    ``"oneshot"`` fallback it is ``None`` and each generation re-runs ``sd-cli``."""

    repo_id: str
    base_repo: str
    family: DiffusionFamily
    device: str
    files: SdCppModelFiles
    display_repo_id: Optional[str] = None
    vae_format: Optional[str] = None
    native_speed: str = "off"
    offload_flags: tuple[str, ...] = ()
    threads: Optional[int] = None
    sampling_method: Optional[str] = None
    flow_shift: Optional[float] = None
    # The base's shipped ``sample_sigmas`` grid, sent as custom sigmas; None keeps sd.cpp's own schedule.
    sample_sigmas: Optional[tuple[float, ...]] = None
    server: Optional[SdCppServer] = None
    mode: str = "server"
    # Token kept so LoRA adapters selected at generate time can be fetched from the Hub.
    hf_token: Optional[str] = None
    resolved: Optional[dict] = None
    # GGUF basename this load committed, since some variants pick their text encoder by filename.
    gguf_filename: Optional[str] = None
    # Kept so the delete guard reconstructs the same encoder pick without re-probing.
    flux2_inner_dim: Optional[int] = None
    # Accelerator at choice time (None on the server path): one-shot re-resolves sd-cli per image and must detect swaps.
    sd_accelerator: Optional[str] = None
    # None unless a single resolved CUDA/ROCm card; others cannot use the VRAM floor.
    physical_gpu_id: Optional[int] = None
    # This load's card, as the failure record names cards.
    selected_card: Optional[str] = None
    # UNSLOTH_DIFFUSION_SD_CPP_DEVICE placement: its label, and the env every spawn gets to open only that card.
    off_torch_device: Optional[str] = None
    child_env: tuple[tuple[str, str], ...] = ()
    # The user picked the GPU (a gpu_ids pool or UNSLOTH_DIFFUSION_SD_CPP_DEVICE): never moved to CPU behind their back.
    gpu_explicit: bool = False

    def spawn_env(self) -> Optional[dict[str, str]]:
        return dict(self.child_env) or None


def _offload_with_device_pin_impl(
    offload: tuple[str, ...] | list[str], binary: Optional[str], ordinal: Optional[int]
) -> list[str]:
    """``offload`` plus the ``--backend`` pin for whichever build is about to run it."""
    flags = list(offload)
    if ordinal is None:
        return flags
    device_name = sd_cpp_device_name_for_ordinal(binary, ordinal)
    if device_name is None:
        # `Vulkan0` is not a physical index; pin by card name instead.
        selected_name, selected_position = physical_card_name(ordinal)
        device_name = sd_cpp_device_named(binary, selected_name, position = selected_position)
    return [*flags, *device_backend_flags(device_name, flags)]


def _resolved_server_physical_gpu_id(
    binary: Optional[str], device: str, ordinal: Optional[int], committed_flags: tuple[str, ...]
) -> Optional[int]:
    """Physical id for a provably single-device resident sd-server."""
    if device != "cuda" or "--offload-to-cpu" in committed_flags:
        return None
    try:
        from utils.hardware import get_parent_visible_gpu_ids
        visible = [int(i) for i in get_parent_visible_gpu_ids()]
    except Exception:  # noqa: BLE001 -- don't block on a flaky probe (timeout etc.)
        return None
    local_ordinal = ordinal
    if local_ordinal is None:
        if len(visible) != 1:
            return None
        local_ordinal = 0
    if local_ordinal < 0 or local_ordinal >= len(visible):
        return None
    device_name = sd_cpp_device_name_for_ordinal(binary, local_ordinal)
    if device_name is None:
        return None
    if ordinal is not None:
        # Attributable only when the argv contains the resolved pin; a failed probe is no evidence.
        specs = [
            committed_flags[index + 1]
            for index, flag in enumerate(committed_flags[:-1])
            if flag == "--backend"
        ]
        if not any(device_name in spec for spec in specs):
            return None
    return visible[local_ordinal]


def _memory_policy(memory_mode: Optional[str], cpu_offload: bool) -> str:
    """Map the diffusers memory knobs onto an sd-cli offload policy. Only meaningful off-CPU (forced
    sd_cpp / MPS); on CPU everything is resident in RAM anyway."""
    mode = (memory_mode or "").strip().lower()
    if mode == "low_vram":
        return OFFLOAD_SEQUENTIAL
    if mode == "balanced":
        return OFFLOAD_GROUP
    if cpu_offload and mode in ("", "auto"):
        return OFFLOAD_MODEL
    return OFFLOAD_NONE


def _native_speed_for(speed_mode: Optional[str]) -> str:
    mode = (speed_mode or "off").strip().lower()
    return mode if mode in ("default", "max") else "off"


@dataclass
class _SdLoading:
    """An in-flight asset download, polled for progress."""

    repo_id: str
    base_repo: str
    # Companion asset repos (VAE / text encoders) so the delete-cached guard protects them
    asset_repos: tuple[str, ...] = ()
    expected_bytes: int = 0
    downloaded_bytes: int = 0
    error: Optional[str] = None
    off_torch_device: Optional[str] = None


@dataclass
class _SdGen:
    """An in-flight generation, updated from parsed sd-cli progress lines."""

    total_steps: int
    step: int = 0
    first_step_at: float = 0.0
    eta_seconds: Optional[float] = None
    # Per sampling pass denominators (Hires fix: base, refine); empty = one pass of ``total_steps``.
    passes: tuple[int, ...] = ()
    pass_index: int = 0
    pass_offset: int = 0


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


def _request_flow_shift(state: Any, width: int, height: int) -> Optional[float]:
    """The family's fixed ``--flow-shift``, except Krea 2 Raw, whose mu follows the output size."""
    if getattr(state.family, "name", None) == "krea-2" and krea2_raw_checkpoint(
        state.gguf_filename,
        getattr(state, "display_repo_id", None),
        state.repo_id,
        state.base_repo,
    ):
        return krea2_sd_cpp_flow_shift(width, height)
    return state.flow_shift


_EMBEDDED_GUIDANCE_FAMILIES = ("flux.1", "flux.1-kontext", "flux.2-dev")
# diffusers computes cond + g * (cond - uncond): g is standard CFG minus 1, and any g > 0 enables guidance.
_CFG_MINUS_ONE_FAMILIES = ("z-image", "krea-2")


def _map_guidance(
    fam: DiffusionFamily, guidance: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(cfg_scale, guidance) for sd-cli. Guidance-distilled FLUX runs cfg 1.0 plus the embedded guidance; the rest
    (FLUX.2-klein included: no guidance embedder) use real CFG, 1.0 when <= 1. Always explicit: sd.cpp defaults to 7.0."""
    if fam.name in _EMBEDDED_GUIDANCE_FAMILIES:
        return 1.0, (float(guidance) if guidance is not None else None)
    if fam.name in _CFG_MINUS_ONE_FAMILIES:
        return (float(guidance) + 1.0 if guidance is not None and guidance > 0.0 else 1.0), None
    cfg = float(guidance) if (guidance is not None and guidance > 1.0) else 1.0
    return cfg, None


def _negative_prompt_guidance_threshold(fam: DiffusionFamily) -> Optional[float]:
    """sd.cpp drops a negative prompt unless _map_guidance gives cfg > 1; embedded-guidance families never use one."""
    if fam.name in _EMBEDDED_GUIDANCE_FAMILIES:
        return None
    return 0.0 if fam.name in _CFG_MINUS_ONE_FAMILIES else 1.0


def _assert_shards_beside_index(kind: str, path: str) -> None:
    """sd-cli resolves a ``*.safetensors.index.json`` encoder's shards relative to the index, so every shard it names
    must sit in the same snapshot directory. The shards are fetched as their own assets, which normally lands them
    there; a cache split across two roots could not, and sd-cli would then fail deep in the load. Say so here."""
    if not str(path).endswith(".safetensors.index.json"):
        return
    try:
        with open(path, encoding = "utf-8") as fh:
            shards = set((json.load(fh).get("weight_map") or {}).values())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read the {kind} shard index '{path}': {exc}") from exc
    missing = sorted(s for s in shards if not (Path(path).parent / s).exists())
    if missing:
        raise RuntimeError(
            f"the {kind} encoder's shards {missing} are not beside its index in '{Path(path).parent}'; "
            "clear that model from the cache and load it again"
        )


def _colocate_shards(index_path: str, fetched: dict[str, str]) -> None:
    """Put every shard an encoder index names beside it, when the shard was fetched into the OTHER cache root.

    ``_fetch_assets`` resolves each file through either root, so after a cache-folder change the index can come from
    the old root while its shards land in the active one. sd-cli only looks beside the index, and an offline user
    cannot repair that by re-downloading, so each missing shard is hard-linked (else copied) into the index's snapshot
    directory, under the name the repo itself gives it. ``fetched`` maps a shard's file name to the path it was
    resolved to. Best-effort: anything left missing is reported by ``_assert_shards_beside_index``."""
    if not str(index_path).endswith(".safetensors.index.json"):
        return
    try:
        with open(index_path, encoding = "utf-8") as fh:
            shards = set((json.load(fh).get("weight_map") or {}).values())
    except (OSError, ValueError):
        return
    parent = Path(index_path).parent
    for shard in sorted(shards):
        target = parent / shard
        source = fetched.get(shard)
        if not source or not Path(source).is_file():
            continue
        real = Path(source).resolve()
        size = real.stat().st_size
        if target.exists():
            # A complete shard is left alone; a short one (an earlier copy cut off) is replaced.
            if target.stat().st_size == size:
                continue
            with contextlib.suppress(OSError):
                target.unlink()
        # A hard link, else a copy, never a symlink: each cache root is deleted and unmounted on its own, so a link
        # from one into the other would dangle, or a cache scan would treat the other root's blob as this one's.
        try:
            os.link(real, target)
            continue
        except OSError:
            pass
        # The copy lands under a temporary name and is renamed into place only once complete, so a full disk or an
        # interrupted copy never leaves a truncated shard that the existence check would accept.
        partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
        try:
            shutil.copyfile(real, partial)
            if partial.stat().st_size != size:
                raise OSError(f"short copy of {real}")
            os.replace(partial, target)
        except OSError:
            with contextlib.suppress(OSError):
                partial.unlink()


def _fetch_repo_map(assets: list[tuple[str, str, str]], hf_token: Optional[str]) -> dict[str, str]:
    """Gated base goes to its mirror; a mirror goes back to a cached repack, sparing an existing install a re-download; decided per repo so staging and load agree."""
    by_repo: dict[str, list[str]] = {}
    for repo, filename, _kind in assets:
        by_repo.setdefault(repo, []).append(filename)
    return {
        # Both cache roots: _fetch_assets resolves each file through either, so an upstream copy in the pre-change
        # root keeps the upstream id instead of re-pulling the mirror.
        repo: prefer_cached_legacy_source(
            prefer_ungated_mirror(repo, hf_token, files = names, other_root = True), names
        )
        for repo, names in by_repo.items()
    }


class _NeverRaised(Exception):
    """Placeholder ``except`` target for a hub layout with no LocalEntryNotFoundError."""


def _cached_text_encoder_alternative(repo: str, filename: str, kind: str) -> Optional[str]:
    """A cached other precision of this encoder when *filename* itself is not loadable from the
    cache, else None. Network-free (``try_to_load_from_cache``)."""
    alternatives = sd_cpp_text_encoder_cached_alternatives(repo, filename, kind)
    if not alternatives:
        return None
    from core.inference.diffusion import DiffusionBackend

    if DiffusionBackend._hub_file_is_loadable(repo, filename):
        return None
    for alternative in alternatives:
        if DiffusionBackend._hub_file_is_loadable(repo, alternative):
            return alternative
    return None


def _local_entry_not_found_error() -> type[BaseException]:
    """Imported lazily so a changed hub layout degrades to the raw error, never breaks the import."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
        return LocalEntryNotFoundError
    except Exception:  # noqa: BLE001 -- an unexpected hub layout keeps the raw error
        return _NeverRaised


def _with_mirrors(repo_ids) -> tuple[str, ...]:
    """``repo_ids`` plus the ungated mirror and the community repack of each, de-duplicated, order
    preserved. The delete-cached guard must protect whichever of the set the bytes landed in, and
    that decision is re-taken per load; naming all of them is cheap and cannot under-protect."""
    out: list[str] = []
    for rid in repo_ids:
        if not rid:
            continue
        out.append(rid)
        mirror = mirror_repo(rid)
        if mirror:
            out.append(mirror)
        legacy = legacy_source_repo(rid)
        if legacy:
            out.append(legacy)
    return tuple(dict.fromkeys(out))


def _assert_pick_is_not_speech(
    repo_id: str,
    gguf_filename: Optional[str],
    hf_token: Optional[str] = None,
    allow_network: bool = True,
) -> None:
    """The shared speech refusal, imported lazily so this module keeps its import cost."""
    from .diffusion_compat import assert_pick_is_not_speech
    assert_pick_is_not_speech(repo_id, gguf_filename, hf_token, allow_network)


# "this thread never ran a load", which is not the same answer as "its load selected no card".
_UNREAD_LOADING_CARD = object()


class SdCppDiffusionBackend:
    """Native sd.cpp backend with the diffusers ``DiffusionBackend`` method surface."""

    # sd.cpp parses A1111 attention syntax ((word:1.2), [word]) in every prompt; it cannot be turned off.
    parses_prompt_attention = True
    # Class-level defaults so an instance built with ``__new__`` answers before any load ran.
    _loading_cards = None
    _committed_loading_card: Optional[str] = None
    # Thread-local like _loading_card: a rejected concurrent load must not move it.
    _loading_families = None
    _loading_family_default: Any = None

    def __init__(self, engine: Optional[SdCppEngine] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._engine = engine  # resolved lazily on first load so import stays cheap
        # An injected engine pins one-shot; a fallback-cached one must not, so a server that becomes available can still be used on the next load.
        self._engine_injected = engine is not None
        self._state: Optional[_SdState] = None
        self._loading: Optional[_SdLoading] = None
        self._load_token = 0
        # Replaced (never cleared) per load, so a cancelled asset pull stays cancelled.
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        self._active_generate_account: Optional[str] = None
        # In-flight load's sd-server before _state commits; unload can stop it mid-startup.
        self._pending_server: Optional[SdCppServer] = None
        # Server a generation is stopping for a replacement: status() must not reap it as an idle crash, or the
        # handover reads an unload and kills the healthy replacement.
        self._replacing_server: Optional[SdCppServer] = None
        self._gen: Optional[_SdGen] = None
        # Set when an accelerator install was skipped for a busy tree; the load retries it once free.
        self._deferred_accelerator_install = False
        # Created here but unset: an explicit None would be this thread's answer forever.
        self._loading_cards = threading.local()
        # unload() stops outside the lock, so fields read idle while the process still runs.
        self._stopping_servers = 0
        # Set once the GPU graph is unrunnable (ggml unsupported-op abort); CPU restarts once per load.
        self._cpu_backend_forced = False

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    @property
    def runs_off_torch_device(self) -> bool:
        """Everything resident or loading sits on a card torch cannot see. A pending load counts, or
        training would cancel it; a torch-placed resident beside it still has to be freed."""
        loading = getattr(self, "_loading", None)
        if loading is not None and loading.error is not None:
            loading = None
        held = [x for x in (self._state, loading) if x is not None]
        return bool(held) and all(getattr(x, "off_torch_device", None) for x in held)

    def _loading_card_store(self) -> threading.local:
        """Lazily, so an instance built with ``__new__`` (the unit-test seam) still answers."""
        store = getattr(self, "_loading_cards", None)
        if store is None:
            store = threading.local()
            self._loading_cards = store
        return store

    @property
    def _loading_card(self) -> Optional[str]:
        """The card THIS worker's load selected (a cancelled load's worker can overlap its
        replacement). Off a load thread, the last committed load's card."""
        own = getattr(self._loading_card_store(), "card", _UNREAD_LOADING_CARD)
        return self._committed_loading_card if own is _UNREAD_LOADING_CARD else own

    @_loading_card.setter
    def _loading_card(self, value: Optional[str]) -> None:
        # Thread-local; _committed_loading_card goes out with _state, since a started load may not commit.
        self._loading_card_store().card = value

    def _clear_loading_card(self) -> None:
        """Back to no own answer (not None): load threads are pooled."""
        try:
            del self._loading_card_store().card
        except AttributeError:
            pass

    def _loading_family_store(self) -> threading.local:
        """Lazily, so an instance built with ``__new__`` (the unit-test seam) still answers."""
        store = getattr(self, "_loading_families", None)
        if store is None:
            store = threading.local()
            self._loading_families = store
        return store

    @property
    def _loading_family(self) -> Any:
        """The family THIS worker is loading. Off a load thread, the last one begun."""
        return getattr(self._loading_family_store(), "family", self._loading_family_default)

    @_loading_family.setter
    def _loading_family(self, value: Any) -> None:
        self._loading_family_store().family = value
        # Per-instance value for off-thread readers; a rejected concurrent load can move it, so the worker must read its thread-local first, never this.
        self._loading_family_default = value

    def _reserve_stop(self, count: int = 1) -> None:
        """Claim ``count`` pending stops. MUST be called under ``_lock`` in the same block that
        unpublishes the servers: incrementing afterwards leaves a gap in which _state,
        _pending_server and the count are all empty while the process is still running."""
        self._stopping_servers += count

    def _stop_reserved(self, server: Any) -> None:
        """Stop a server whose pending stop was already reserved by ``_reserve_stop``. Never raises:
        a teardown may not fail a load or an unload."""
        try:
            server.stop()
        except Exception as exc:  # noqa: BLE001 -- a stop that fails must not fail the caller
            logger.warning("sd-server stop failed: %s", exc)
        finally:
            with self._lock:
                self._stopping_servers -= 1

    def _stop_server(self, server: Any) -> None:
        """Reserve and stop in one go, for a caller that is not already holding ``_lock``."""
        with self._lock:
            self._reserve_stop()
        self._stop_reserved(server)

    @staticmethod
    def _resolved_accelerator(card: Optional[str] = None) -> str:
        """preferred_accelerator applied here so all four call sites agree, or they reinstall each other; the lazy import avoids an import cycle."""
        from core.inference.diffusion_engine_router import image_install_accelerator
        return preferred_accelerator(
            image_install_accelerator(getattr(resolve_diffusion_device_target(), "backend", "cpu")),
            card,
        )

    def _resolve_engine(self) -> SdCppEngine:
        """The SdCppEngine, installing the binary on first use. Raises if unusable."""
        if self._engine is not None and self._engine.is_available():
            # Check the cached engine too: it may predate the family. Only when it names a path, since a no-binary test seam would refuse every marked family.
            cached_binary = getattr(self._engine, "binary", None)
            if cached_binary:
                self._refuse_incapable_build(cached_binary, "sd-cli")
            return self._engine
        # Host accelerator, never cpu: this is also the one-shot fallback, and asking for CPU there reinstalls the plain bundle over a working GPU build.
        binary = ensure_sd_cpp_binary(
            allow_install = _install_allowed() and not _tree_in_use(self),
            accelerator = self._resolved_accelerator(self._loading_card),
        )
        if not binary:
            raise RuntimeError("sd-cli (stable-diffusion.cpp) binary is unavailable.")
        self._refuse_incapable_build(binary, "sd-cli")
        self._engine = SdCppEngine(binary = binary)
        return self._engine

    def _refuse_incapable_build(self, binary: Optional[str], name: str) -> None:
        """Re-checks here: router verdict is local and server and CLI may differ; else a one-shot backend reports ready then dies on first generation."""
        fam = self._loading_family
        if fam is None or sd_cpp_binary_runs_family(binary, fam):
            return
        raise RuntimeError(
            f"the {name} build at {binary} predates {getattr(fam, 'name', 'this model')} support. "
            "Reinstall the native engine (delete the managed stable-diffusion.cpp install, or point "
            "SD_CLI_PATH / SD_SERVER_PATH at a build from after upstream added it)."
        )

    def _resolve_backend(self) -> tuple[str, Optional[str], Optional[SdCppEngine]]:
        """An injected engine forces one-shot for tests; a lazily cached fallback engine does not."""
        if self._engine_injected and self._engine is not None:
            return "oneshot", None, self._resolve_engine()
        accelerator = self._resolved_accelerator(self._loading_card)
        # Upgrades replace running binaries (ETXTBSY / Windows locks): defer until teardown.
        upgrade_pending = _tree_in_use(self) or _managed_tree_in_use()
        self._deferred_accelerator_install = upgrade_pending
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed() and not upgrade_pending, accelerator = accelerator
        )
        if (
            server_binary is not None
            and not sd_cpp_binary_runs_family(server_binary, self._loading_family)
            and self._loading_family is not None
        ):
            # Not fatal alone: the one-shot path resolves its own sd-cli, and _resolve_engine vets that one.
            logger.warning(
                "sd-server at %s predates %s support; trying the one-shot sd-cli instead",
                server_binary,
                getattr(self._loading_family, "name", "this model"),
            )
            server_binary = None
        if server_binary is not None:
            return "server", server_binary, None
        logger.warning(
            "sd-server not found; falling back to one-shot sd-cli (reloads the model per image)."
        )
        return "oneshot", None, self._resolve_engine()

    def _upgrade_server_after_teardown(self, server_binary: Optional[str]) -> Optional[str]:
        """Installs under both locks once nothing runs from the tree, also when server_binary is None, or a CUDA request keeps the old CPU CLI."""
        if not _install_allowed():
            return server_binary
        try:
            accelerator = self._resolved_accelerator(self._loading_card)
            # Judge the tree by its binary (sd-cli on serverless installs) or every deferred load reinstalls.
            probe = server_binary or find_sd_cpp_binary()
            if probe is None or not _accelerator_changed(probe, accelerator):
                return server_binary
            logger.info("installing the %s sd.cpp build now the managed tree is free", accelerator)
            return (
                ensure_sd_server_binary(allow_install = True, accelerator = accelerator)
                or server_binary
            )
        except Exception as exc:  # noqa: BLE001 -- an upgrade may never fail the load
            logger.warning("sd.cpp accelerator upgrade failed: %s", exc)
            return server_binary

    def _upgraded_or_refused(
        self, server_binary: Optional[str], *, mode: str, engine: Optional[Any]
    ) -> Optional[str]:
        """Land the deferred install, then answer the question the router could not: the teardown
        upgrade is never fatal, so the build about to start can still be the condemned one."""
        upgraded = self._upgrade_server_after_teardown(server_binary)
        running = upgraded if mode == "server" else getattr(engine, "binary", None)
        if running and not usable_or_recorded_failure(
            running, self._resolved_accelerator(self._loading_card), self._loading_card
        ):
            raise RuntimeError(
                "the sd.cpp build in the managed tree is recorded as failing on this host "
                "and the replacement for it could not be installed."
            )
        return upgraded

    @pauses_background_embedding
    def begin_load(
        self,
        repo_id: str,
        *,
        # Same name, position and default as DiffusionBackend.begin_load or every load TypeErrors; covers model assets only, not the binary install.
        local_files_only: bool = False,
        display_repo_id: Optional[str] = None,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        # Diffusers-only knobs, accepted for a uniform call and ignored; sd.cpp has no torchao.
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        # Accepted for interface parity; native is GGUF-only (router forces diffusers otherwise).
        model_kind: Optional[str] = None,
        # Diffusers bakes LoRA at load; native applies it per generation, so this selection is ignored.
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ids: Optional[list[int]] = None,
        # The ordinal the ROUTE already ranked, so the preflight and the load agree on one card
        gpu_ordinal: Optional[int] = None,
        # Accepted for parity; native never raises the unified-memory refusal.
        allow_oversized_load: bool = False,
    ) -> dict[str, Any]:
        """Validate, then fetch on a daemon thread, returns at once; an empty token must become None, since a literal empty string breaks the anonymous fallback."""
        hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
        from core.inference.diffusion_engine_router import off_torch_sd_cpp_device

        off_torch = off_torch_sd_cpp_device()
        gpu_explicit = bool(gpu_ids) or off_torch is not None
        if off_torch is not None:
            gpu_ids, gpu_ordinal = None, None
        # Same fallback the diffusers and video backends take: the route ranks the selection and passes the winner,
        # but a direct caller (an MCP client, a test, a plugin) hands over gpu_ids alone, and without this the native
        # engine is the one engine that would drop the pick silently. Re-ranked only when nobody has, so a
        # route-resolved winner is never second-guessed against free VRAM that has moved since.
        auto_ordinal = gpu_ordinal is None and not gpu_ids and off_torch is None
        if gpu_ordinal is None and gpu_ids:
            gpu_ordinal = (
                resolve_selected_cuda_ordinal(gpu_ids)
                if resolve_diffusion_device_target().device == "cuda"
                else None
            )
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        # Family falls back to the filename, as the route validated, so basename-only keywords still load.
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name."
            )
        if not family_sd_cpp_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no native sd.cpp asset mapping.")
        if auto_ordinal:
            # Automatic ranks the torch cards too, after validation, unless sd.cpp runs on one torch cannot see.
            gpu_ordinal = auto_cuda_ordinal_for_load()
        self._loading_family = fam

        base = resolve_base_repo(fam, base_repo)
        # Offline-only here, deliberately: begin_load returns at once by contract
        inner_dim = self._flux2_inner_dim(
            repo_id, gguf_filename, fam, hf_token, allow_network = False
        )
        # Same link the diffusers resolver records, so the delete guard protects a native pick's companions too -- and
        # here that means the repos _asset_specs actually FETCHES. The native engine does not read the diffusers base:
        # FLUX.2 takes its VAE from the FLUX.2 base and its encoder from an unsloth GGUF repo, so recording only the
        # base left every repo the pick really depends on outside the guard, and an unloaded model's encoder could be
        # deleted while its GGUF stayed installed. Best-effort bookkeeping; never fails a load.
        try:
            from hub.utils.companion_assets import record_companion_link
            for asset_repo in dict.fromkeys(
                r
                for r, _f, kind in self._asset_specs(repo_id, gguf_filename, fam, inner_dim)
                if kind != "diffusion_model"
            ):
                record_companion_link(repo_id, asset_repo)
            record_companion_link(repo_id, base)
        except Exception as exc:  # noqa: BLE001
            logger.debug("sd_cpp.companion_link_record_failed: %s", exc)
        with self._lock:
            if self._loading is not None and self._loading.error is None:
                raise RuntimeError("A diffusion load is already in progress.")
            # A superseding load stops in-flight generation, or the old run may save an image after it starts.
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._load_token += 1
            token = self._load_token
            # New event per load: clear() would un-cancel a still-running pull.
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._loading = _SdLoading(
                repo_id = repo_id,
                base_repo = base,
                asset_repos = tuple(
                    dict.fromkeys(
                        r
                        for r, _f, kind in self._asset_specs(repo_id, gguf_filename, fam, inner_dim)
                        if kind != "diffusion_model"
                    )
                ),
                off_torch_device = off_torch.label if off_torch is not None else None,
            )

        account_thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                display_repo_id = display_repo_id,
                local_files_only = local_files_only,
                gguf_filename = gguf_filename,
                base = base,
                explicit_base = bool((base_repo or "").strip()),
                fam = fam,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                gpu_ordinal = gpu_ordinal,
                gpu_explicit = gpu_explicit,
                off_torch = off_torch,
                _load_token = token,
                _cancel_event = cancel_event,
            ),
            daemon = True,
        ).start()
        return self.status()

    @_invalidates_gpu_memory("sd.cpp load")
    def _run_load(
        self,
        *,
        repo_id: str,
        display_repo_id: Optional[str] = None,
        gguf_filename: str,
        base: str,
        explicit_base: bool = False,
        fam: DiffusionFamily,
        family_override: Optional[str] = None,
        hf_token: Optional[str],
        # Cache-only when set: Hub calls are skipped or resolve from disk, so no load pulls bytes.
        local_files_only: bool = False,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        gpu_explicit: bool = False,
        off_torch: Any = None,
        _load_token: int,
        _cancel_event: Optional[threading.Event] = None,
    ) -> None:
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = _cancel_event if _cancel_event is not None else self._cancel_event
        # Outside so the backstop can unpublish it; a leaked one blocks every later install.
        started: Optional[SdCppServer] = None
        # Once, before anything asks: every resolution in this load must give the same answer.
        self._loading_card = selected_card_identity(gpu_ordinal)
        try:
            # Resolve the binary up front so an install failure surfaces before the multi-GB pull.
            mode, server_binary, engine = self._resolve_backend()
            if mode == "server":
                # Probe the server binary before the pull, so an unrunnable build fails before downloads.
                assert server_binary is not None
                if not _server_binary_runnable(server_binary):
                    logger.warning(
                        "sd-server at %s is present but not runnable; trying one-shot sd-cli.",
                        server_binary,
                    )
                    # Resolve ONCE and keep it: two calls can answer with two different binaries if an install lands
                    # between them, and the state below reads the accelerator off whichever object it ends up holding.
                    fallback: Optional[SdCppEngine] = None
                    try:
                        fallback = self._resolve_engine()
                        usable = fallback.version() is not None
                    except Exception:  # noqa: BLE001
                        usable = False
                    if not usable or fallback is None:
                        note_unlaunchable_accelerator_build(server_binary, card = self._loading_card)
                        raise RuntimeError("sd-server binary is present but not runnable.")
                    mode, server_binary, engine = "oneshot", None, fallback
            # Pin the accelerator at choice time; an install during download can replace sd-server.
            server_accelerator = _installed_accelerator_of(server_binary)
            # Same for the one-shot CLI, or the swap check could never fire.
            engine_accelerator = _installed_accelerator_of(getattr(engine, "binary", None))
            if mode == "oneshot":
                # version() is None when a present binary can't run; fail now, not on the first generation
                assert engine is not None
                if engine.version() is None:
                    note_unlaunchable_accelerator_build(
                        getattr(engine, "binary", None),
                        card = self._loading_card,
                    )
                    raise RuntimeError("sd-cli binary is present but not runnable.")

            # Swap once so size probe and download agree; speech refusal runs before any fetch.
            _assert_pick_is_not_speech(
                repo_id, gguf_filename, hf_token, allow_network = not local_files_only
            )
            inner_dim = self._flux2_inner_dim(
                repo_id, gguf_filename, fam, hf_token, allow_network = not local_files_only
            )
            specs = self._asset_specs(
                repo_id, gguf_filename, fam, inner_dim, local_only = local_files_only
            )
            fetch_repo = _fetch_repo_map(specs, hf_token)
            assets = [(fetch_repo[repo], fn, kind) for repo, fn, kind in specs]
            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.asset_repos = tuple(
                        dict.fromkeys(r for r, _f, kind in specs if kind != "diffusion_model")
                    )
            # Record from post-probe specs incl. fetch ids, or a 9B encoder goes unguarded.
            try:
                from hub.utils.companion_assets import record_companion_link
                for asset_repo in dict.fromkeys(
                    rid
                    for repo, _f, kind in specs
                    if kind != "diffusion_model"
                    for rid in (repo, fetch_repo.get(repo, repo))
                ):
                    record_companion_link(repo_id, asset_repo)
            except Exception as exc:  # noqa: BLE001 -- bookkeeping never fails a load
                logger.debug("sd_cpp.companion_link_record_failed: %s", exc)
            # Preflight post-swap repos so a gated companion fails before the prefetch.
            self._preflight_companion_repos(
                self._assets_by_repo(assets),
                fetch_repo.get(repo_id, repo_id),
                hf_token,
                local_files_only = local_files_only,
            )
            # Skipped offline: the size probe is a Hub round trip only for the progress bar.
            if not local_files_only:
                self._set_expected_bytes(assets, hf_token)
            paths = self._fetch_assets(
                assets,
                hf_token,
                cancel_event = cancel_event,
                local_files_only = local_files_only,
                vision_optional = not getattr(fam, "edit", False),
            )
            # A renamed or local FLUX.3 GGUF loaded under the other variant would run the wrong defaults (steps, CFG,
            # negative prompt); the same refusal as the diffusers loader, before any native state is committed.
            from core.inference.diffusion_families import assert_flux3_checkpoint_matches_family

            assert_flux3_checkpoint_matches_family(fam, paths["diffusion_model"])
            sample_sigmas, grid_base = _base_sample_sigmas(
                repo_id,
                base,
                hf_token,
                family = fam.name,
                explicit_base = explicit_base,
                local_files_only = local_files_only,
                named_base = named_variant_base(fam, repo_id, gguf_filename),
            )
            if grid_base != base:
                # The card can name a base begin_load did not know; link it so the delete guard keeps its index.
                try:
                    from hub.utils.companion_assets import record_companion_link
                    record_companion_link(repo_id, grid_base)
                except Exception as exc:  # noqa: BLE001 -- bookkeeping never fails a load
                    logger.debug("sd_cpp.companion_link_record_failed: %s", exc)

            files = SdCppModelFiles(
                diffusion_model = paths["diffusion_model"],
                vae = paths.get("vae"),
                clip_l = paths.get("clip_l"),
                clip_g = paths.get("clip_g"),
                t5xxl = paths.get("t5xxl"),
                llm = paths.get("llm"),
                llm_vision = paths.get("llm_vision"),
                qwen2vl = paths.get("qwen2vl"),
                tokenizer = paths.get("tokenizer"),
            )
            device = (
                off_torch.accelerator
                if off_torch is not None
                else resolve_diffusion_device_target().device
            )
            # A paravirtual Mac GPU runs sd.cpp on the CPU backend (see metal_text_encoder_flags), so record that.
            if sd_cpp_args_mod.forces_cpu_backend(device):
                device = "cpu"
            torch_device = device

            def _running() -> tuple[Optional[str], Optional[str]]:
                if mode == "server":
                    return server_accelerator, server_binary
                return engine_accelerator, getattr(engine, "binary", None)

            def _placement(
                run_accelerator: Optional[str], binary: Optional[str]
            ) -> tuple[str, tuple[str, ...]]:
                # Read off the build this load runs, never the requested one. SD_SERVER_PATH /
                # SD_CLI_PATH builds carry no install record, so their own device listing says.
                if run_accelerator is None and binary:
                    run_accelerator = sd_cpp_listed_device(binary)[0]
                klass = _accelerator_class_of(run_accelerator) if run_accelerator else None
                # A CPU torch running the Vulkan build renders on a card torch never measured, so
                # auto offloads.
                vulkan = torch_device == "cpu" and off_torch is None and klass == "vulkan"
                placed = "vulkan" if vulkan else torch_device
                # The fallback ladder can leave a CUDA / XPU torch running the CPU build. Not
                # MPS: the macOS bundle is recorded as cpu and still renders on Metal.
                if klass == "cpu" and off_torch is None and torch_device in ("cuda", "xpu"):
                    placed = "cpu"
                # Honor speed everywhere; offload only off-CPU (on CPU weights are resident, so the flags are no-ops)
                if placed == "cpu":
                    return placed, ()
                return placed, tuple(
                    offload_flags(_memory_policy(memory_mode, cpu_offload or vulkan))
                )

            device, offload = _placement(*_running())
            spawn_env = off_torch.child_env() if off_torch is not None else None

            @functools.lru_cache(maxsize = 1)
            def _diffusion_weight_types() -> Optional[frozenset[str]]:
                return sd_cpp_args_mod.model_files_quantized_weight_types(files)

            # The device pin is NOT folded in here: the binary can still change below, through the deferred
            # accelerator install, the post-download re-resolve, or a server start that falls back to one-shot, and
            # the ggml device names come from whichever build ends up running. It is added at each point the flags are
            # handed to a binary instead.
            # The route's card, applied only while the placement is CUDA: a deferred upgrade can move
            # a load placed on the CPU bundle back onto the selected GPU.
            selected_ordinal = gpu_ordinal

            def _ordinal() -> Optional[int]:
                return selected_ordinal if device == "cuda" else None

            native_speed = _native_speed_for(speed_mode)

            # Teardown and commit run under _generate_lock, taken only now: abort and wait for a generation begun during the download, so no stale image persists.
            with self._lock:
                if self._load_token != _load_token:
                    return  # superseded / cancelled
                if self._active_generate_cancel is not None:
                    self._active_generate_cancel.set()
            with self._generate_lock:
                with self._lock:
                    if self._load_token != _load_token:
                        return  # superseded / cancelled while waiting
                    old_state = self._state
                    self._state = None  # the old model is being torn down
                    if old_state is not None and old_state.server is not None:
                        self._reserve_stop()
                if old_state is not None and old_state.server is not None:
                    self._stop_reserved(old_state.server)
                # Tree is free now, so the deferred install lands; not gated on resolved mode, since a serverless install resolves one-shot because the deferral skipped it.
                if self._deferred_accelerator_install:
                    self._deferred_accelerator_install = False
                    upgraded = self._upgraded_or_refused(server_binary, mode = mode, engine = engine)
                    if mode == "server":
                        server_binary = upgraded
                    # This load's own install IS the decision; both pins move.
                    server_accelerator = _installed_accelerator_of(server_binary)
                    engine_accelerator = _installed_accelerator_of(getattr(engine, "binary", None))
                    # The upgrade may have landed or kept the old build; place for whichever runs.
                    device, offload = _placement(*_running())
                # A new checkpoint earns a fresh attempt on the GPU backend: the previous abort says nothing about
                # this graph.
                self._cpu_backend_forced = False
                server: Optional[SdCppServer] = None
                if mode == "server":
                    assert server_binary is not None
                    # Re-resolve under the reader claim, held until _pending_server is published; an install may have swept this path during the download.
                    with _tree_reader(server_binary, cancel_event):
                        refreshed = ensure_sd_server_binary(
                            allow_install = False,
                            accelerator = self._resolved_accelerator(self._loading_card),
                        )
                        if refreshed and refreshed != server_binary:
                            logger.info(
                                "sd-server moved during the asset download: %s -> %s",
                                server_binary,
                                refreshed,
                            )
                            server_binary = refreshed
                        if not server_binary or not _server_binary_runnable(server_binary):
                            logger.warning(
                                "sd-server is no longer usable after the asset download; "
                                "falling back to one-shot sd-cli."
                            )
                            mode, server_binary, engine = "oneshot", None, self._resolve_engine()
                            # Pin off THIS engine; the earlier pin was taken against None.
                            engine_accelerator = _installed_accelerator_of(
                                getattr(engine, "binary", None)
                            )
                            device, offload = _placement(*_running())
                        elif _installed_accelerator_of(server_binary) != server_accelerator:
                            raise RuntimeError(
                                "The stable-diffusion.cpp server binary was replaced by an install "
                                "for a different accelerator while this model was loading. Try the "
                                "load again."
                            )
                        else:
                            _refuse_off_torch_build_mismatch(off_torch, server_binary)
                            server = SdCppServer(server_binary)
                            # Publish inside the claim and recheck cancel under the unload lock, since an unload in the revalidation gap would miss the server and hold the device.
                            with self._lock:
                                if self._load_token != _load_token or cancel_event.is_set():
                                    server = None
                                else:
                                    started = server
                                    self._pending_server = server
                            if server is None:
                                raise SdCppCancelled()
                if mode == "server":
                    assert server_binary is not None
                    assert server is not None
                    # Clear started, not server (None on fallback); a started server stays published until _state takes it, or installs overwrite the live binary.
                    started_ok = False
                    try:
                        server_offload = _offload_with_device_pin_impl(
                            offload, server_binary, _ordinal()
                        )
                        spawn_env = _sd_cpp_spawn_env(
                            off_torch,
                            device,
                            _ordinal(),
                            offload,
                            server_offload,
                            _diffusion_weight_types,
                        )
                        server.start(
                            files,
                            vae_format = fam.sd_cpp_vae_format,
                            offload = server_offload,
                            native_speed = native_speed,
                            threads = _default_threads(),
                            env = spawn_env,
                        )
                        started_ok = True
                    except SdCppCancelled:
                        server.stop()
                        raise
                    except Exception as start_exc:  # noqa: BLE001
                        logger.warning(
                            "sd-server failed to start (%s); falling back to one-shot sd-cli.",
                            start_exc,
                        )
                        # The error carries the build's own output.
                        note_accelerator_failure_from_output(
                            server_binary,
                            str(start_exc),
                            card = self._loading_card,
                        )
                        server.stop()
                        # Unpublish first: a stale _pending_server blocks the sd-cli install this fallback needs.
                        with self._lock:
                            if self._pending_server is server:
                                self._pending_server = None
                        server = None
                        # Keep the fallback engine, or sd_accelerator is None and generation rejects it.
                        fallback: Optional[SdCppEngine] = None
                        cached = getattr(self._engine, "binary", None)
                        if cached and _installed_accelerator_of(
                            cached
                        ) != self._resolved_accelerator(self._loading_card):
                            # The failure just recorded can move the preferred build off a cached sd-cli.
                            self._engine = None
                        try:
                            fallback = self._resolve_engine()
                            usable = fallback.version() is not None
                        except Exception:  # noqa: BLE001
                            usable = False
                        if not usable or fallback is None:
                            raise start_exc
                        engine = fallback
                        # Vetted here, so pinned here: this engine was resolved after the download, inside the claim,
                        # and holding it to the pre-download answer would refuse the fallback on an install this load
                        # already lived through.
                        engine_accelerator = _installed_accelerator_of(
                            getattr(fallback, "binary", None)
                        )
                        mode = "oneshot"
                        device, offload = _placement(*_running())
                    finally:
                        if not started_ok:
                            with self._lock:
                                if self._pending_server is started:
                                    self._pending_server = None
                if mode == "oneshot" and (
                    _installed_accelerator_of(getattr(engine, "binary", None)) != engine_accelerator
                ):
                    # Refuse at load: recording the replacement would make later checks agree with it.
                    raise RuntimeError(
                        "The stable-diffusion.cpp binary was replaced by an install for a "
                        "different accelerator while this model was loading. Try the load again."
                    )
                if mode == "oneshot":
                    _refuse_off_torch_build_mismatch(off_torch, getattr(engine, "binary", None))
                committed_offload_flags = tuple(
                    _offload_with_device_pin_impl(
                        offload,
                        server_binary if mode == "server" else getattr(engine, "binary", None),
                        _ordinal(),
                    )
                )
                # Recomputed for what is committed: a deferred install or the one-shot fallback can move placement.
                spawn_env = _sd_cpp_spawn_env(
                    off_torch,
                    device,
                    _ordinal(),
                    offload,
                    committed_offload_flags,
                    _diffusion_weight_types,
                )
                state = _SdState(
                    repo_id = repo_id,
                    display_repo_id = display_repo_id,
                    # The card's base when it shipped the grid, as the diffusers route reports it.
                    base_repo = grid_base if sample_sigmas else base,
                    family = family_for_checkpoint(
                        fam, gguf_filename, repo_id, display_repo_id, base
                    ),
                    device = device,
                    files = files,
                    vae_format = fam.sd_cpp_vae_format,
                    native_speed = native_speed,
                    # Pinned to the binary this load committed to, since a deferred install or fallback may change it.
                    offload_flags = committed_offload_flags,
                    threads = _default_threads(),
                    sampling_method = fam.sd_cpp_sampling_method,
                    flow_shift = fam.sd_cpp_flow_shift,
                    sample_sigmas = sample_sigmas,
                    server = server,
                    mode = mode,
                    hf_token = hf_token,
                    resolved = build_resolved_record(
                        {"family_override": _family_override_resolved(family_override, fam)}
                    ),
                    gguf_filename = gguf_filename,
                    flux2_inner_dim = inner_dim,
                    # Only one-shot needs it: sd-cli is re-resolved per image, with nothing else to verify it.
                    sd_accelerator = engine_accelerator if mode == "oneshot" else None,
                    # Never on an off-torch card: the parent-visible ids are torch's, so CUDA0 would name one of them.
                    physical_gpu_id = (
                        _resolved_server_physical_gpu_id(
                            server_binary,
                            device,
                            _ordinal(),
                            committed_offload_flags,
                        )
                        if mode == "server" and off_torch is None
                        else None
                    ),
                    selected_card = self._loading_card,
                    off_torch_device = off_torch.label if off_torch is not None else None,
                    child_env = tuple(sorted((spawn_env or {}).items())),
                    gpu_explicit = gpu_explicit,
                )
                superseded = False
                orphan: Optional[SdCppServer] = None
                with self._lock:
                    if self._load_token != _load_token:
                        # Reserved in the same block that unpublishes it, so the tree never reads idle.
                        superseded = True
                        if server is not None:
                            self._reserve_stop()
                            orphan = server
                    else:
                        self._state = state
                        # With _state, past the supersession check.
                        self._committed_loading_card = self._loading_card
                        self._loading = None
                    # A started server stays published in _state or the pending stop, so _tree_in_use still sees it.
                    if self._pending_server is started:
                        self._pending_server = None
                if orphan is not None:
                    self._stop_reserved(orphan)
                if superseded:
                    return
                logger.info(
                    "sd_cpp.loaded: repo=%s gguf=%s device=%s mode=%s speed=%s offload_flags=%s",
                    state.repo_id,
                    state.gguf_filename,
                    f"{state.device} ({state.off_torch_device}, outside torch)"
                    if state.off_torch_device
                    else state.device,
                    state.mode,
                    state.native_speed,
                    without_device_backend_flags(state.offload_flags) or "none",
                )
        except SdCppCancelled:
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != _load_token:
                return
            logger.error("sd_cpp.load_failed: %s", exc)
            if self._state is not None:
                # Displaced pipeline: give its account back GPU residency, drop its records.
                from .gpu_arbiter import DIFFUSION, restore_media_need, restore_owner_account
                from hub.services.models.account_access import restore_resident_metadata

                restore_owner_account(DIFFUSION)
                restore_media_need(DIFFUSION)
                restore_resident_metadata("diffusion")
            # Redact filesystem paths before this reaches /images/load-progress (as diffusers does).
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))
        finally:
            # Backstop for an unexpected raise between start() and _state, which would wedge the tree.
            if started is not None:
                with self._lock:
                    if self._pending_server is started:
                        self._pending_server = None
            # The card was this load's, not this worker's.
            self._clear_loading_card()

    def download_plan(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        model_kind: Optional[str] = None,
        hf_token: Optional[str] = None,
        **_load_kwargs: Any,
    ) -> dict[str, Any]:
        """Must plan for the engine the load will pick; the wrong one stages files the load never opens."""
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None or not family_sd_cpp_supported(fam):
            # Unreachable through the route, but a direct caller gets the same message begin_load would raise
            raise ValueError(f"'{repo_id}' has no native sd.cpp asset mapping.")
        # Same reason as the diffusers plan: this is what stages the download.
        _assert_pick_is_not_speech(repo_id, gguf_filename, hf_token)

        specs = self._asset_specs(
            repo_id,
            gguf_filename,
            fam,
            self._flux2_inner_dim(repo_id, gguf_filename, fam, hf_token),
        )
        by_repo = self._assets_by_repo(specs)

        # Stage from the fetch repo: some asset repos are gated and anonymous staging would 401.
        fetch_repo = _fetch_repo_map(specs, hf_token)
        # MERGED, not reassigned: two upstream repos can share one fetch repo (a gated vendor id and its unsloth
        # mirror both stage under the mirror), and a plain comprehension would drop whichever landed first, leaving
        # its files out of both the staged entry and the footprint.
        merged: dict[str, list[str]] = {}
        for repo, names in by_repo.items():
            into = merged.setdefault(fetch_repo[repo], [])
            into.extend(n for n in names if n not in into)
        by_repo = merged
        # A named variant's load reads its model_index.json for the grid; planned so locality cannot skip it.
        if base_repo:
            explicit = resolve_base_repo(fam, base_repo)
            variants = {b.lower() for b in getattr(fam, "named_variant_bases", ()) or ()}
            # As given: that is the cache entry the load reads.
            grid_base = explicit if explicit.lower() in variants else None
        else:
            grid_base = named_variant_base(fam, repo_id, gguf_filename)
        if fam.name in _SAMPLE_SIGMAS_FAMILIES and grid_base:
            names = by_repo.setdefault(_grid_fetch_repo(grid_base, hf_token), [])
            if "model_index.json" not in names:
                names.append("model_index.json")
        fetch_repo_id = fetch_repo.get(repo_id, repo_id)
        # AFTER the swap: preflighting the upstream id would refuse the very picks the ungated mirror exists to rescue
        self._preflight_companion_repos(by_repo, fetch_repo_id, hf_token)
        sizes = self._plan_file_sizes(by_repo, hf_token)
        entries: list[dict[str, Any]] = []
        total = 0
        # Local import, since diffusion.py is the heavier module and routes already load this one.
        from core.inference.diffusion import DiffusionBackend

        for repo, names in by_repo.items():
            total += int(sum(sizes.get((repo, n), 0) for n in names))
            # Same missing-file filter as diffusers, but sized and loadable: a stale live-root copy shadows a good one. required_bytes stays unfiltered.
            missing = [
                n
                for n in names
                if not DiffusionBackend._hub_file_is_loadable(repo, n, None, sizes.get((repo, n)))
            ]
            if not missing:
                continue
            entries.append(
                {
                    "repo_id": repo,
                    # A stable scope lets repeated picks adopt an in-flight download.
                    "files": list(names),
                    "bytes": int(sum(sizes.get((repo, n), 0) for n in missing)),
                    # Only the transformer entry carries the GGUF filename; VAE and encoder entries are plain files.
                    "gguf_filename": gguf_filename if repo == fetch_repo_id else None,
                    # Compared against the post-swap id: a gated pick staged from its mirror differs.
                    "checkpoint": repo == fetch_repo_id and gguf_filename in missing,
                }
            )
        return {
            "entries": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries),
            "required_bytes": total,
            "checkpoint_bytes": int(sizes.get((fetch_repo_id, gguf_filename), 0)),
        }

    @staticmethod
    def _assets_by_repo(specs: list[tuple[str, str, str]]) -> dict[str, list[str]]:
        """repo -> the files this pick needs from it, first-seen order (transformer first), so a
        family whose VAE and text encoder share a repo yields one entry."""
        by_repo: dict[str, list[str]] = {}
        for repo, filename, kind in specs:
            # A local GGUF directory is already on disk; nothing to stage or preflight for it.
            if kind == "diffusion_model":
                try:
                    if Path(repo).expanduser().exists():
                        continue
                except (OSError, RuntimeError, ValueError):
                    pass  # unresolvable home / invalid path characters -> a remote id
            names = by_repo.setdefault(repo, [])
            if filename not in names:
                names.append(filename)
        return by_repo

    @staticmethod
    def _preflight_companion_repos(
        by_repo: dict[str, list[str]],
        repo_id: str,
        hf_token: Optional[str],
        *,
        local_files_only: bool = False,
    ) -> None:
        """Refuses unreadable gated companion repos before any byte is fetched; run from both plan and _run_load, skipped if local_files_only."""
        if local_files_only:
            return
        from core.inference.diffusion import _assert_base_repo_accessible

        for repo, names in by_repo.items():
            # Companions only: the picker only lists repos it could already read.
            if repo != repo_id and names:
                # A VAE-only repo has no pipeline manifest; probe a staged asset.
                _assert_base_repo_accessible(repo, hf_token, names[0])

    def preflight_base_access(
        self,
        repo_id: str,
        fam: Optional[DiffusionFamily],
        *,
        gguf_filename: Optional[str] = None,
        model_kind: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        allow_network: bool = True,  # noqa: ARG002 -- signature parity; no speech probe here
    ) -> None:
        """Runs in the route before the GPU is taken, so a refusal never evicts the resident chat model."""
        if fam is None or not gguf_filename:
            return
        # Post-swap, as the plan and the load are: the swap is pure, so all three decide alike
        specs = self._asset_specs(
            repo_id,
            gguf_filename,
            fam,
            self._flux2_inner_dim(repo_id, gguf_filename, fam, hf_token),
        )
        fetch_repo = _fetch_repo_map(specs, hf_token)
        self._preflight_companion_repos(
            self._assets_by_repo([(fetch_repo[r], fn, kind) for r, fn, kind in specs]),
            fetch_repo.get(repo_id, repo_id),
            hf_token,
        )

    @staticmethod
    def _plan_file_sizes(
        by_repo: dict[str, list[str]], hf_token: Optional[str]
    ) -> dict[tuple[str, str], int]:
        """(repo, filename) -> size in bytes, best-effort (0 for anything the Hub will not answer).
        A missing size only understates the manager's progress total; it must not fail the plan,
        which is the cheap pre-flight for a load that would otherwise download inline."""
        out: dict[tuple[str, str], int] = {}
        try:
            from huggingface_hub import HfApi
            api = HfApi(token = hf_token)
        except Exception:  # noqa: BLE001 -- sizes are best-effort
            return out
        for repo, names in by_repo.items():
            try:
                for info in api.get_paths_info(repo, paths = names, expand = False):
                    out[(repo, getattr(info, "path", ""))] = int(getattr(info, "size", 0) or 0)
            except Exception:  # noqa: BLE001 -- one unreadable repo is non-fatal
                continue
        return out

    @staticmethod
    def _flux2_inner_dim(
        repo_id: str,
        gguf_filename: str,
        fam: DiffusionFamily,
        hf_token: Optional[str],
        *,
        allow_network: bool = True,
    ) -> Optional[int]:
        """The checkpoint's own FLUX.2 size, or None. Header-only and memoised, so the four
        ``_asset_specs`` callers share one probe; skipped outright for every other family, which
        has a single static encoder table and must stay network-free."""
        if fam.name != "flux.2-klein":
            return None
        return flux2_inner_dim_for_pick(
            repo_id, gguf_filename, hf_token, allow_network = allow_network
        )

    def _asset_specs(
        self,
        repo_id: str,
        gguf_filename: str,
        fam: DiffusionFamily,
        inner_dim: Optional[int] = None,
        local_only: bool = False,
    ) -> list[tuple[str, str, str]]:
        """(repo, filename, kind) for every file sd-cli needs. ``kind`` is the SdCppModelFiles
        field; the transformer reuses the diffusers GGUF. Offline / ``local_only`` swaps an uncached
        encoder for a cached other precision so the UI plan and the load agree."""
        specs: list[tuple[str, str, str]] = [(repo_id, gguf_filename, "diffusion_model")]
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        offline = local_only
        if not offline:
            from utils.utils import hf_env_offline
            offline = hf_env_offline()
        # Pick the encoder per variant so a 9B GGUF fetches the right one: from the header when the caller read it,
        # else from the load identity, which a renamed file makes silent.
        for terepo, tefile, kind in sd_cpp_text_encoders_for(
            fam, repo_id, gguf_filename, inner_dim = inner_dim
        ):
            if offline:
                tefile = _cached_text_encoder_alternative(terepo, tefile, kind) or tefile
            specs.append((terepo, tefile, kind))
        tokenizer = getattr(fam, "sd_cpp_tokenizer", None)
        if tokenizer:
            specs.append((tokenizer[0], tokenizer[1], "tokenizer"))
        return specs

    def _set_expected_bytes(
        self, assets: list[tuple[str, str, str]], hf_token: Optional[str]
    ) -> None:
        """Best-effort total download size for the progress bar (0 if unknown)."""
        total = 0
        try:
            from huggingface_hub import HfApi

            api = HfApi(token = hf_token)
            # Sized where _fetch_assets pulls from: a gated vendor id answers an anonymous user with nothing, which
            # left CLIP / T5 at 0 bytes and the bar finalizing while they downloaded.
            fetch_repo = _fetch_repo_map(assets, hf_token)
            for repo, fn, kind in assets:
                # Only the transformer can be a local path; others are always HF ids.
                if kind == "diffusion_model" and Path(repo).expanduser().exists():
                    continue
                try:
                    info = api.get_paths_info(fetch_repo.get(repo, repo), paths = [fn], expand = False)
                    for it in info:
                        total += int(getattr(it, "size", 0) or 0)
                except Exception:  # noqa: BLE001 -- one missing size is non-fatal
                    continue
        except Exception:  # noqa: BLE001 -- estimate is best-effort
            total = 0
        loading = self._loading
        if loading is not None:
            loading.expected_bytes = total

    def _fetch_assets(
        self,
        assets: list[tuple[str, str, str]],
        hf_token: Optional[str],
        cancel_event: Optional[threading.Event] = None,
        local_files_only: bool = False,
        vision_optional: bool = True,
    ) -> dict[str, str]:
        """Offline loads resolve assets from the HF cache only and a miss fails here; cancel_event belongs to this load so a replacement load cannot un-cancel it."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Callers without a per-load event (tests, direct use) fall back to the current one.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        paths: dict[str, str] = {}
        # Shard file name -> where it resolved: the kind repeats per shard, so ``paths`` keeps only the last one.
        shard_paths: dict[str, str] = {}
        # This backend fetches the ASSET repos, never the base, and some are gated, so the swap goes here. Decided per
        # REPO over its whole file list, exactly as download_plan does, so the load pulls from the repo the manager
        # already staged.
        fetch_repo = _fetch_repo_map(assets, hf_token)
        assets = [(fetch_repo[repo], fn, kind) for repo, fn, kind in assets]
        for repo, fn, kind in assets:
            if cancel.is_set():
                raise SdCppCancelled("load cancelled")
            local_root = Path(repo).expanduser()
            if kind == "diffusion_model" and local_root.exists():
                path = str(resolve_local_gguf_child(local_root, fn))
            else:
                # Resolve via the import-time cache root too, or moved assets re-download / 401.
                try:
                    path = hf_hub_download_with_xet_fallback(
                        repo,
                        fn,
                        hf_token,
                        cancel_event = cancel,
                        reuse_other_cache_root = True,
                        local_files_only = local_files_only,
                        gguf_header_delta = True,
                    )
                except _local_entry_not_found_error() as exc:
                    # Not fetchable: an encoder whose default precision changed may be cached as an older quant.
                    alternative = _cached_text_encoder_alternative(repo, fn, kind)
                    if alternative is not None:
                        logger.warning(
                            "sd_cpp.text_encoder_cached_quant: %s/%s is not cached and cannot be "
                            "downloaded; using the cached %s",
                            repo,
                            fn,
                            alternative,
                        )
                        path = hf_hub_download_with_xet_fallback(
                            repo,
                            alternative,
                            hf_token,
                            cancel_event = cancel,
                            reuse_other_cache_root = True,
                            local_files_only = True,
                            gguf_header_delta = True,
                        )
                    else:
                        # Only fires under local_files_only; hub text omits repo/file and this string is the UI toast.
                        if not local_files_only:
                            raise
                        if kind == "llm_vision" and vision_optional:
                            # Only edit reads the projector; text-to-image still loads without it.
                            logger.info(
                                "sd_cpp.llm_vision_not_cached: %s/%s, editing unavailable", repo, fn
                            )
                            continue
                        raise RuntimeError(
                            f"'{fn}' is not in the local cache for '{repo}', and this load may not "
                            f"download (it was not user-initiated). Open the model from the Images "
                            f"page to fetch it."
                        ) from exc
            paths[kind] = path
            if kind.endswith("_shard"):
                shard_paths[fn.rsplit("/", 1)[-1]] = path
            with self._lock:
                if self._loading is not None:
                    try:
                        self._loading.downloaded_bytes += os.path.getsize(path)
                    except OSError:
                        pass
        for kind, path in paths.items():
            _colocate_shards(path, shard_paths)
            _assert_shards_beside_index(kind, path)
        return paths

    def load_progress(self) -> dict[str, Any]:
        loading = self._loading
        if loading is not None and loading.error:
            return _progress("error", error = loading.error)
        if loading is None:
            return _progress("ready" if self._state is not None else None)
        downloaded = loading.downloaded_bytes
        expected = loading.expected_bytes
        if expected > 0 and downloaded >= expected * 0.999:
            return _progress("finalizing", min(downloaded, expected), expected, 1.0)
        fraction = min(downloaded / expected, 1.0) if expected > 0 else 0.0
        return _progress("downloading", downloaded, expected, fraction)

    def loading_repo_ids(self) -> tuple[str, ...]:
        """In-flight load repo ids, companions and mirrors included, so the delete-cached guard
        refuses them."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (loading.repo_id, loading.base_repo, *loading.asset_repos)
            return _with_mirrors(ids)

    def loaded_repo_ids(self) -> tuple[str, ...]:
        """Companion repos a loaded native model reads from disk, so the delete guard must refuse
        them too."""
        with self._lock:
            state = self._state
            if state is None:
                return ()
            fam = state.family
            repos = [state.repo_id, state.base_repo]
            if fam.sd_cpp_vae:
                repos.append(fam.sd_cpp_vae[0])
            if getattr(fam, "sd_cpp_tokenizer", None):
                repos.append(fam.sd_cpp_tokenizer[0])
            # Same per-variant selection as _asset_specs (the header dim this load committed, else repo id AND GGUF
            # filename) so the delete guard protects the encoder repo this load actually downloaded. Read off the
            # state, never re-probed: this runs under _lock.
            repos.extend(
                terepo
                for terepo, _f, _k in sd_cpp_text_encoders_for(
                    fam,
                    state.repo_id,
                    state.gguf_filename,
                    inner_dim = state.flux2_inner_dim,
                )
            )
            return _with_mirrors(repos)

    def _native_binary(self, state: _SdState) -> Optional[str]:
        """The sd.cpp binary this load runs: the resident server's, else the one-shot engine's."""
        binary = getattr(state.server, "binary", None) if state.server is not None else None
        return binary or getattr(getattr(self, "_engine", None), "binary", None)

    def _native_reference_fidelity(self, state: Optional[_SdState]) -> bool:
        """Whether this load's build reads reference images with their alpha and their own size."""
        if state is None:
            return False
        binary = self._native_binary(state)
        return bool(binary) and binary_carries_marker(
            binary, _REFERENCE_FIDELITY_MARKER, unreadable = False
        )

    def _native_edit_ready(self, state: Optional[_SdState]) -> bool:
        """Whether this load can run an edit natively: unified-edit needs its projector and the build's edit marker;
        edit-only needs its projector if it declares one, and the marker only if it declares one."""
        if state is None:
            return False
        fam = state.family
        marker = getattr(fam, "sd_cpp_edit_marker", None)
        if getattr(fam, "edit", False):
            if _family_reads_vision(fam) and not state.files.llm_vision:
                return False
            if not marker:
                return True
        elif not (getattr(fam, "unified_edit", False) and marker and state.files.llm_vision):
            return False
        binary = self._native_binary(state)
        if not binary:
            return False
        return binary_carries_marker(binary, marker, unreadable = False)

    def _native_reference_ready(self, state: Optional[_SdState], workflow: Optional[str]) -> bool:
        """Projector-free references (FLUX.3): sd.cpp feeds them through the VAE, so the arch gate that admitted the
        load is all a reference call needs. A reference-edit family's Edit is that same call."""
        if state is None or not getattr(state.family, "sd_cpp_reference", False):
            return False
        if workflow == "edit":
            return bool(getattr(state.family, "reference_edit", False))
        return workflow in (None, "reference")

    @holds_inference
    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = 1024,
        height: Optional[int] = 1024,
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        # Batched prompt/seed lists are rejected here: sd-cli would render them serially.
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        # Accepted for interface parity; native is text-to-image only, so image-conditioned requests are rejected below.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        upscale: Optional[float] = None,  # needs an init image; rejected by the guard below
        reference_images: Optional[list[str]] = None,  # GPU/diffusers-only (FLUX.2)
        workflow: Optional[str] = None,
        reference_resolution: Optional[int] = None,
        localized_edit: Any = None,
        # LoRA (id, weight) pairs, resolved up front: prompt tags for sd-cli, structured lora for sd-server.
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet is diffusers-only; rejected by the guard below (accepted for parity).
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
        # load_identity() of the caller's status() read; refuse rather than run a different load (#9448)
        expected_load: Optional[LoadIdentity] = None,
        # Interface parity only: the activation guard is diffusers-only.
        allow_oversized: bool = False,
        hires: Any = None,
    ) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        # Edit-only families: an input image with no workflow is an edit (re-checked under the lock below).
        loaded = self._state
        if (
            workflow is None
            and init_image is not None
            and loaded is not None
            and getattr(loaded.family, "edit", False)
        ):
            workflow = "edit"
        conditioned = workflow in ("edit", "reference")
        if conditioned:
            if init_image is None:
                raise ValueError(f"The {workflow} workflow requires a source image (init_image).")
            if reference_resolution is not None:
                # No such control natively: sd.cpp sizes every input image to the output area itself.
                raise ValueError(
                    "Reference detail is not adjustable on the native sd.cpp engine: it resizes "
                    "each input image to the output area. Leave reference_resolution unset."
                )
        elif (
            init_image is not None
            or mask_image is not None
            or reference_images
            or workflow is not None
            or reference_resolution is not None
            or localized_edit is not None
            or (upscale is not None and upscale > 1)
        ):
            raise ValueError(
                "img2img / inpaint / reference / upscale are not yet supported on the native "
                "sd.cpp engine; run on a GPU (diffusers) for image-conditioned workflows."
            )
        if prompts is not None or seeds is not None:
            raise ValueError(
                "Batched prompt/seed lists are not supported on the native sd.cpp engine "
                "(it renders serially); run on a GPU (diffusers) for batched generation, "
                "or use batch_size for a serial native batch."
            )
        # strength 0/None disables ControlNet (matches diffusers), so no-op it rather than 400
        if controlnet is not None and controlnet[3] in (None, 0, 0.0):
            controlnet = None
        if controlnet is not None:
            raise ValueError(
                "ControlNet is not yet supported on the native sd.cpp engine; run on a GPU "
                "(diffusers) for ControlNet conditioning."
            )

        cancel = threading.Event()
        from core.inference.generation_ticket import bind_generation_cancel

        bind_generation_cancel(cancel)
        from hub.services.models.account_access import media_generation_slot

        with self._generate_lock, media_generation_slot("diffusion"):
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # A resident server can exit while idle: drop stale state so clients reload instead.
                if (
                    state.mode == "server"
                    and state.server is not None
                    and not state.server.is_alive()
                ):
                    self._state = None
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
                # Same window as the diffusers engine: a replacement can commit while this waits (#9448)
                loaded_id = load_identity(state.repo_id, state.base_repo, state.family.name)
                if expected_load is not None and expected_load != loaded_id:
                    raise DiffusionModelReplacedError(expected_load, loaded_id)
                self._active_generate_cancel = cancel
                self._active_generate_account = current_account_id()
                # Publish step 0 before slow setup so a reload probe does not read idle.
                self._gen = _SdGen(total_steps = int(steps))
            try:
                ref_pngs: list[bytes] = []
                edit_only = bool(getattr(state.family, "edit", False))
                if edit_only and not conditioned:
                    raise ValueError(
                        f"{state.family.name} is an image-editing model: provide an input image."
                    )
                if edit_only and workflow == "reference":
                    raise ValueError(
                        f"The reference workflow is not supported for the '{state.family.name}' "
                        "model family."
                    )
                from core.inference.diffusion_conditioning import edit_takes_extra_images

                # sd.cpp's Qwen-Image-Edit conditioner reads every ref image as "Picture N", so a multi-image edit runs
                # natively too.
                if (
                    reference_images
                    and not getattr(state.family, "reference", False)
                    and not edit_takes_extra_images(state.family)
                ):
                    raise ValueError(
                        f"Reference images are not supported for the '{state.family.name}' "
                        "model family."
                    )
                if conditioned:
                    from core.inference.diffusion_conditioning import check_conditioned_fields

                    check_conditioned_fields(
                        workflow,
                        state.family,
                        mask_image = mask_image,
                        strength = strength,
                        upscale = upscale,
                        controlnet = controlnet,
                        localized_edit = localized_edit,
                    )
                    if not (
                        self._native_edit_ready(state)
                        or self._native_reference_ready(state, workflow)
                    ):
                        raise ValueError(
                            f"Image editing is not available for '{state.family.name}' on the native "
                            "sd.cpp engine with this build and its assets."
                        )
                    width, height, ref_pngs = _native_condition_images(
                        state.family,
                        init_image,
                        reference_images,
                        localized_edit,
                        width,
                        height,
                        full_fidelity = self._native_reference_fidelity(state),
                        pad_to_output = state.mode == "server" and state.server is not None,
                        source_sized = edit_only,
                    )
                elif width is None or height is None:
                    raise ValueError("width and height are required for this workflow.")
                else:
                    from core.inference.diffusion_conditioning import check_output_size
                    check_output_size(state.family, int(width), int(height))
                hires_plan = None
                if hires is not None:
                    from core.inference import diffusion_hires

                    if conditioned:
                        raise ValueError("Hires fix runs on text to image only.")
                    hires_plan = diffusion_hires.plan_sd_cpp(
                        state.family,
                        hires,
                        mode = state.mode,
                        binary = getattr(state.server, "binary", None),
                        width = int(width),
                        height = int(height),
                        steps = int(steps),
                    )
                    # sd.cpp samples the base and Hires schedules separately, each reporting its own denominator.
                    passes = (int(steps), int(hires_plan.refine_steps))
                    self._gen = _SdGen(total_steps = sum(passes), passes = passes)
                if seed is None:
                    seed = int.from_bytes(os.urandom(6), "big") & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                cfg_scale, flux_guidance = _map_guidance(state.family, guidance)
                if cfg_scale <= 1.0:
                    negative_prompt = None
                # Resolve selected LoRAs up front (a bad id gives a clear 400). Drop weight-0 rows BEFORE the support
                # gate so an only-disabled request stays a no-op.
                lora_resolved: list = []
                active_loras = [(i, w) for (i, w) in (loras or []) if w != 0]
                if active_loras:
                    if not diffusion_lora.supports_lora(
                        engine = "sd_cpp",
                        family = state.family.name,
                        model_kind = "gguf",
                        transformer_quant = None,
                    ):
                        raise ValueError(
                            f"LoRA is not supported for {state.family.name} on the native "
                            "sd.cpp engine."
                        )
                    lora_resolved = diffusion_lora.resolve_specs(
                        active_loras,
                        family = state.family.name,
                        hf_token = state.hf_token,
                        cancel_event = cancel,
                    )
                try:
                    if state.mode == "server" and state.server is not None:
                        images, seeds = self._generate_server(
                            state,
                            prompt = prompt,
                            negative_prompt = negative_prompt,
                            width = width,
                            height = height,
                            steps = steps,
                            seed = seed,
                            batch_size = batch_size,
                            cfg_scale = cfg_scale,
                            flux_guidance = flux_guidance,
                            lora_resolved = lora_resolved,
                            cancel = cancel,
                            ref_pngs = ref_pngs,
                            layers = _layer_count(state.family) or None,
                            hires_plan = hires_plan,
                        )
                    else:
                        images, seeds = self._generate_oneshot(
                            state,
                            prompt = prompt,
                            negative_prompt = negative_prompt,
                            width = width,
                            height = height,
                            steps = steps,
                            seed = seed,
                            batch_size = batch_size,
                            cfg_scale = cfg_scale,
                            flux_guidance = flux_guidance,
                            lora_resolved = lora_resolved,
                            cancel = cancel,
                            ref_pngs = ref_pngs,
                            layers = _layer_count(state.family) or None,
                        )
                except RuntimeError as exc:
                    # The mid-render hipBLAS death the video path records too; not a cancellation.
                    if not cancel.is_set() and DIFFUSION_CANCELLED_MSG not in str(exc):
                        # The binary that ran it: on the server path `self._engine` is None.
                        _failed_binary = getattr(
                            getattr(state, "server", None), "binary", None
                        ) or getattr(getattr(self, "_engine", None), "binary", None)
                        note_accelerator_failure_from_output(
                            _failed_binary,
                            str(exc),
                            source = "diffusion",
                            card = getattr(state, "selected_card", None),
                        )
                    raise
                # Under _lock like cancel_generate so check and cancel cannot interleave.
                with self._lock:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None
                result = {
                    "images": images,
                    "seed": int(seed),
                    "seeds": seeds,
                    "negative_prompt": negative_prompt or None,
                    "repo_id": state.display_repo_id or state.repo_id,
                    # The repo id does not say which GGUF quant ran.
                    "model_kind": "gguf",
                    "gguf_filename": state.gguf_filename,
                    # Offload is real (sd-cli flags), so record it; omitting it persisted null for every native image.
                    "transformer_quant": None,
                    "text_encoder_quant": None,
                    "memory_mode": None,
                    # Policy flags only: a --backend pin is a card choice, not offload.
                    "offload_policy": (
                        "active" if without_device_backend_flags(state.offload_flags) else "none"
                    ),
                    "speed_mode": state.native_speed,
                    "cpu_offload": bool(without_device_backend_flags(state.offload_flags)),
                    "workflow": workflow if conditioned else "txt2img",
                    "reference_resolution": None,
                    "localized_edit": getattr(localized_edit, "mode", None)
                    if conditioned
                    else None,
                }
                if hires_plan is not None:
                    result["hires"] = diffusion_hires.result_block(hires_plan)
                logger.info(
                    "diffusion.generated: %s",
                    format_generation_for_log(
                        result,
                        engine = "sd_cpp",
                        steps = steps,
                        strength = strength,
                        loras = active_loras,
                    ),
                )
                return result
            except SdCppCancelled as exc:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG) from exc
            finally:
                self._gen = None
                with self._lock:
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None

    def _generate_server(
        self,
        state: _SdState,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        seed: int,
        batch_size: int,
        cfg_scale: Optional[float],
        flux_guidance: Optional[float],
        lora_resolved: list,
        cancel: threading.Event,
        ref_pngs: Optional[list[bytes]] = None,
        layers: Optional[int] = None,
        hires_plan: Any = None,
    ) -> tuple[list, list[int]]:
        """Chunked at base+offset so per-image seeds reproduce; LoRA goes in the structured lora field, since the sdcpp API ignores lora tags in the prompt."""
        import io
        import os
        import shutil

        from PIL import Image

        from core.inference import diffusion_lora

        assert state.server is not None
        total = max(1, int(batch_size))
        # sd.cpp's image seed is signed int64; mask base and derived seeds to that range.
        base_seed = int(seed) & ((1 << 63) - 1)
        images: list = []
        seeds: list[int] = []

        # Stage LoRAs into a per-request subdir of the server lora-model-dir (so a prior request's adapters cannot
        # leak in); removed after.
        lora_payload: Optional[list[dict]] = None
        lora_stages: list[Path] = []

        def _stage_loras(server: Any) -> Optional[list[dict]]:
            """The ``lora`` block, with the adapters staged into THIS server's lora-model-dir."""
            if not lora_resolved or not server.lora_dir:
                return None
            stage = Path(server.lora_dir) / f"gen_{os.urandom(6).hex()}"
            lora_stages.append(stage)
            materialized = diffusion_lora.materialize_native_dir(
                lora_resolved, stage, real_files = True
            )
            return [
                {"path": f"{stage.name}/{Path(m.path).name}", "multiplier": float(m.weight)}
                for m in materialized
            ]

        # One deadline for the whole request, shared by its chunks: a batch is chunked only because the server caps
        # images per job, so each chunk gets whatever is left rather than its own full budget and a long batch still
        # ends on time.
        deadline = time.monotonic() + NATIVE_GENERATION_TIMEOUT_S
        # One same-build respawn per request; a second loss moves this load to CPU where the rung applies. Persists nothing.
        device_respawned = False
        device_losses = 0
        hires_payload: Optional[dict] = None
        hires_staged: list[Path] = []

        def _stage_hires(server: Any) -> Optional[dict]:
            """The ``hires`` block, with a model upscaler staged into THIS server's upscaler dir."""
            if hires_plan is None:
                return None
            from core.inference import diffusion_hires, diffusion_upscalers

            upscaler_id = hires_plan.params.upscaler
            staged_name = None
            if diffusion_upscalers.is_model_upscaler(upscaler_id):
                if not server.lora_dir:
                    raise ValueError(
                        "The native server has no upscaler directory for a model upscaler."
                    )
                staged_name, staged = diffusion_hires.stage_sd_cpp_upscaler(
                    upscaler_id, server.lora_dir, cancel = cancel
                )
                hires_staged.append(staged)
            return diffusion_hires.sd_cpp_payload(
                hires_plan.params,
                int(steps),
                hires_plan.target,
                diffusion_hires.sd_cpp_upscaler_name(upscaler_id, staged_name),
                request_steps = hires_plan.request_steps,
            )

        try:
            lora_payload = _stage_loras(state.server)
            hires_payload = _stage_hires(state.server)
            for offset in range(0, total, _MAX_SERVER_BATCH):
                if cancel.is_set():
                    raise SdCppCancelled("sd-server generation was cancelled.")
                count = min(_MAX_SERVER_BATCH, total - offset)
                chunk_seed = (base_seed + offset) & ((1 << 63) - 1)
                payload = build_img_gen_request(
                    prompt = prompt,
                    negative_prompt = negative_prompt or None,
                    width = int(width),
                    height = int(height),
                    steps = int(steps),
                    seed = chunk_seed,
                    batch_count = count,
                    sample_method = state.sampling_method,
                    flow_shift = _request_flow_shift(state, int(width), int(height)),
                    custom_sigmas = (
                        sd_cpp_sample_sigmas(state.sample_sigmas, int(steps))
                        if state.sample_sigmas
                        else None
                    ),
                    cfg_scale = cfg_scale,
                    distilled_guidance = flux_guidance,
                    lora = lora_payload,
                    ref_images = [
                        "data:image/png;base64," + base64.b64encode(b).decode("ascii")
                        for b in ref_pngs or []
                    ],
                    qwen_image_layers = layers,
                    hires = hires_payload,
                )
                try:
                    blobs = state.server.img_gen(
                        payload,
                        on_step = self._on_log,
                        cancel_event = cancel,
                        total_timeout = max(deadline - time.monotonic(), 1.0),
                    )
                except RuntimeError as first_exc:
                    # A ggml unsupported-op abort killed the server: this graph cannot run on the GPU backend at all,
                    # so restart the model on the CPU backend once and retry this chunk. A lost Vulkan device needs a
                    # fresh device: respawn the same build once. Any other death propagates.
                    exc = first_exc
                    while True:
                        moved_to_cpu = False
                        server = self._restart_server_on_cpu_backend(state, str(exc), cancel)
                        if server is None and output_shows_vulkan_device_lost(str(exc)):
                            device_losses += 1
                            if device_losses >= 2:
                                server = self._restart_server_on_cpu_backend(
                                    state, str(exc), cancel, after_device_loss = True
                                )
                                if server is None:
                                    raise RuntimeError(
                                        f"{exc} {_REPEATED_DEVICE_LOSS_HINT}"
                                    ) from exc
                                moved_to_cpu = True
                            elif not device_respawned:
                                server = self._respawn_server_after_device_loss(
                                    state, str(exc), cancel
                                )
                                device_respawned = server is not None
                        if server is None:
                            raise exc
                        old_server = state.server
                        state = replace(state, server = server)
                        if moved_to_cpu:
                            # Placement follows the server: the flags the CPU server got, no Vulkan pin.
                            state = replace(
                                state,
                                device = "cpu",
                                offload_flags = tuple(
                                    without_device_backend_flags(state.offload_flags)
                                ),
                                physical_gpu_id = None,
                            )
                        self._hand_over_replacement(state, server, old_server)
                        # The old server's stop removed its scratch dir, and the staged adapters with it.
                        if lora_payload is not None:
                            lora_payload = _stage_loras(server)
                            if lora_payload is None:
                                raise RuntimeError(
                                    "The replacement native server has no LoRA directory for the selected LoRAs."
                                )
                            payload["lora"] = list(lora_payload)
                        if hires_plan is not None:
                            # The replacement has its own scratch dir: stage the upscaler there again.
                            hires_payload = _stage_hires(server)
                            payload["hires"] = dict(hires_payload)
                        try:
                            blobs = server.img_gen(
                                payload,
                                on_step = self._on_log,
                                cancel_event = cancel,
                                total_timeout = max(deadline - time.monotonic(), 1.0),
                            )
                            break
                        except RuntimeError as retry_exc:
                            exc = retry_exc
                device_losses = 0
                # All-or-nothing per chunk; a layered generation decodes layers + 1 images.
                expected = count * ((layers + 1) if layers else 1)
                if not cancel.is_set() and len(blobs) != expected:
                    raise RuntimeError(
                        f"sd-server returned {len(blobs)} of {expected} requested images in the batch."
                    )
                kept = _keep_layers(blobs, layers or 0)
                images.extend(
                    _native_output_image(state.family, Image.open(io.BytesIO(b))) for b in kept
                )
                # sd.cpp advances the seed per generation; every layer of one carries that seed.
                per = len(kept) // count if count else 1
                seeds.extend(
                    (chunk_seed + i // max(1, per)) & ((1 << 63) - 1) for i in range(len(kept))
                )
        finally:
            for stage in lora_stages:
                shutil.rmtree(stage, ignore_errors = True)
            for staged in hires_staged:
                try:
                    staged.unlink()
                except OSError:
                    pass
        return images, seeds

    def _begin_replacement(self, state: _SdState) -> bool:
        """Mark ``state.server`` as being replaced, under ``_lock``, before the caller stops it, so status() reports
        the load as still loaded instead of reaping it. False when an unload already took the state: no restart."""
        with self._lock:
            if self._state is None or self._state.server is not state.server:
                return False
            self._replacing_server = state.server
            return True

    def _end_replacement(self, old_server: Any) -> None:
        with self._lock:
            if self._replacing_server is old_server:
                self._replacing_server = None

    def _hand_over_replacement(self, state: _SdState, server: SdCppServer, old_server: Any) -> None:
        """Publish a restarted sd-server into ``_state`` in the same locked step that drops it from
        ``_pending_server``, so an unload always sees it in one of the two. Only while ``_state`` still holds the
        server being replaced: when an unload cleared it or a superseding load replaced it, stop the replacement
        and cancel instead of generating on an untracked process or overwriting the newer load."""
        with self._lock:
            if self._pending_server is server:
                self._pending_server = None
            if self._replacing_server is old_server:
                self._replacing_server = None
            published = self._state is not None and self._state.server is old_server
            if published:
                self._state = state
            else:
                self._reserve_stop()
        if not published:
            self._stop_reserved(server)
            raise SdCppCancelled("sd-server generation was cancelled.")

    def _start_replacement(
        self, state: _SdState, server: SdCppServer, cancel: threading.Event, **start_kwargs: Any
    ) -> Optional[SdCppServer]:
        """Start a published-as-pending replacement sd-server for this generation; None when it fails.

        The start blocks for the model load (up to its startup timeout) while the caller holds
        ``_generate_lock``. ``cancel_generate()`` and a superseding load only set ``cancel``, so a watcher stops
        the pending server when that happens: the readiness wait bails at once and the generation raises
        ``SdCppCancelled`` instead of loading a replacement nobody wants."""
        done = threading.Event()

        def _watch() -> None:
            while not done.is_set():
                if cancel.wait(0.25):
                    server.stop()
                    return

        watcher = threading.Thread(target = _watch, name = "sd-server-replacement-cancel", daemon = True)
        watcher.start()
        failed = False
        try:
            server.start(state.files, **start_kwargs)
        except Exception:  # noqa: BLE001 -- the original server error is the more useful one
            failed = True
        finally:
            done.set()
            watcher.join()
        if failed or cancel.is_set():
            with self._lock:
                if self._pending_server is server:
                    self._pending_server = None
            server.stop()
            self._end_replacement(state.server)
            if cancel.is_set():
                raise SdCppCancelled("sd-server generation was cancelled.")
            return None
        # Still pending on success: _hand_over_replacement publishes it, so an unload never misses it.
        return server

    def _respawn_server_after_device_loss(
        self, state: _SdState, error_text: str, cancel: threading.Event
    ) -> Optional[SdCppServer]:
        """Restart this checkpoint's sd-server on the same build after a lost Vulkan device; None if that
        does not apply. The Vulkan spec makes VK_ERROR_DEVICE_LOST recoverable only by creating a new device,
        which the running server never does, so every later job on it would fail. No build strike: a lost
        device (a TDR or a scheduler reset) says nothing about whether this build runs here."""
        if not output_shows_vulkan_device_lost(error_text):
            return None
        if state.server is None or cancel.is_set():
            return None
        if not self._begin_replacement(state):
            return None
        logger.warning(
            "sd-server lost its Vulkan device; restarting it on the same build. Details: %s",
            error_text[:300],
        )
        state.server.stop()
        server = SdCppServer(state.server.binary)
        with self._lock:
            self._pending_server = server
        return self._start_replacement(
            state,
            server,
            cancel,
            vae_format = state.vae_format,
            offload = state.offload_flags,
            native_speed = state.native_speed,
            threads = state.threads,
            env = state.spawn_env(),
        )

    def _restart_server_on_cpu_backend(
        self,
        state: _SdState,
        error_text: str,
        cancel: threading.Event,
        *,
        after_device_loss: bool = False,
    ) -> Optional[SdCppServer]:
        """Relaunch this checkpoint's sd-server with ``--backend cpu``; None if that does not apply.

        ggml's Metal backend checks every node against ``ggml_metal_device_supports_op`` and calls
        ``GGML_ABORT`` when one is not implemented for that device, because a single-backend graph
        has nowhere else to put the node -- there is no per-op CPU fallback. The whole sd-server
        dies with SIGABRT mid-generation, so the user sees "the native image renderer stopped
        unexpectedly" with no way forward. Observed on macos-14 arm64 with FLUX.2-klein-4B Q2_K: the
        encoder is already pinned to CPU and the abort then moves into the denoise loop
        (`unsupported op 'MUL_MAT' -> ggml_abort` under `sample_k_diffusion`).

        ``--backend cpu`` is the only flag that changes which backend EXECUTES the graph
        (``--offload-to-cpu`` moves parameters, not compute), so the restart runs the same
        checkpoint slower rather than not at all. Done once per load: the second abort, or any other
        cause of death, is surfaced to the caller.

        ``after_device_loss``: the respawned server lost its Vulkan device too (a host whose driver resets the
        card on every render). Only on the CPU-torch Vulkan route, the one route with a CPU rung, and never for a
        GPU the user picked. For this load only and nothing is persisted: the next load tries Vulkan again.
        """
        if after_device_loss:
            if not _device_loss_cpu_fallback_applies(state):
                return None
        elif not is_ggml_unsupported_op_abort(error_text):
            return None
        if self._cpu_backend_forced or state.device == "cpu":
            return None  # already on CPU: the abort is not a backend-placement problem
        if state.server is None or cancel.is_set():
            return None
        server_binary = find_sd_server_binary()
        if not server_binary:
            return None
        if not self._begin_replacement(state):
            return None
        if after_device_loss:
            logger.warning(
                "sd-server lost its Vulkan device again after a restart; this model now renders on the CPU "
                "backend (slower) until it is loaded again, which tries Vulkan first. Details: %s",
                error_text[:300],
            )
        else:
            logger.warning(
                "sd-server aborted on an op the '%s' backend cannot run; restarting on the CPU "
                "backend (slower, but it completes). Details: %s",
                state.device,
                error_text[:300],
            )
        self._cpu_backend_forced = True
        state.server.stop()
        server = SdCppServer(server_binary)
        with self._lock:
            self._pending_server = server
        return self._start_replacement(
            state,
            server,
            cancel,
            vae_format = state.vae_format,
            # WITHOUT the device pin. sd.cpp joins repeated --backend values into one spec instead of replacing,
            # and an explicit diffusion=CUDA0 outranks the bare `cpu` default, so leaving the pin on would restart
            # the server onto the very backend that just aborted.
            offload = without_device_backend_flags(state.offload_flags),
            native_speed = state.native_speed,
            threads = state.threads,
            env = state.spawn_env(),
            extra_args = list(CPU_BACKEND_FLAGS),
        )

    def _generate_oneshot(
        self,
        state: _SdState,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        steps: int,
        seed: int,
        batch_size: int,
        cfg_scale: Optional[float],
        flux_guidance: Optional[float],
        lora_resolved: list,
        cancel: threading.Event,
        ref_pngs: Optional[list[bytes]] = None,
        layers: Optional[int] = None,
    ) -> tuple[list, list[int]]:
        """One-shot sd-cli LoRA uses --lora-model-dir plus <lora:ALIAS:w> prompt tags that sd-cli
        parses out."""
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        engine = self._resolve_engine()
        extra_args: list[str] = []
        if state.vae_format:
            extra_args += ["--vae-format", state.vae_format]
        flow_shift = _request_flow_shift(state, int(width), int(height))
        if flow_shift is not None:
            extra_args += ["--flow-shift", repr(float(flow_shift))]
        if state.sample_sigmas:
            sigmas = sd_cpp_sample_sigmas(state.sample_sigmas, int(steps))
            extra_args += ["--sigmas", ",".join(repr(float(x)) for x in sigmas)]

        images = []
        seeds: list[int] = []
        with tempfile.TemporaryDirectory(prefix = "sdcpp_gen_") as tmpdir:
            # Materialize LoRAs into a scan dir and inject <lora:ALIAS:w> tags (deduped).
            eff_prompt = prompt
            lora_dir: Optional[str] = None
            if lora_resolved:
                materialized = diffusion_lora.materialize_native_dir(
                    lora_resolved, Path(tmpdir) / "loras"
                )
                eff_prompt = diffusion_lora.inject_prompt_tags(prompt, materialized)
                lora_dir = str(Path(tmpdir) / "loras")
            # Inside the run's temporary directory, so every exit path removes them.
            ref_paths: list[str] = []
            for i, blob in enumerate(ref_pngs or []):
                ref_path = Path(tmpdir) / f"ref_{i + 1:02d}.png"
                ref_path.write_bytes(blob)
                ref_paths.append(str(ref_path))
            for index in range(max(1, int(batch_size))):
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # Mask to int64; 53 bits would collide large explicit seeds.
                seed_i = (seed + index) & ((1 << 63) - 1)
                out_path = str(Path(tmpdir) / f"img_{index}.png")
                params = SdCppGenParams(
                    prompt = eff_prompt,
                    negative_prompt = negative_prompt or None,
                    width = int(width),
                    height = int(height),
                    steps = int(steps),
                    cfg_scale = cfg_scale,
                    guidance = flux_guidance,
                    seed = seed_i,
                    sampling_method = state.sampling_method,
                    batch_count = 1,
                    lora_dir = lora_dir,
                    lora_apply_mode = "auto" if lora_dir else None,
                    ref_images = tuple(ref_paths),
                    qwen_image_layers = layers,
                )
                # Hold installs off while sd-cli runs; injected engines may have no binary.
                with _tree_reader(getattr(engine, "binary", None), cancel):
                    # Re-resolve inside the claim, per image: an install may have moved sd-cli, and a batch releases the claim between images.
                    engine = self._resolve_engine()
                    # Refuse a different-accelerator CLI: device/offload policy was chosen for the old one.
                    if (
                        _installed_accelerator_of(getattr(engine, "binary", None))
                        != state.sd_accelerator
                    ):
                        raise RuntimeError(
                            "The stable-diffusion.cpp binary was replaced by an install for a "
                            "different accelerator while this model was loaded. Load the model "
                            "again."
                        )
                    engine.generate(
                        state.files,
                        params,
                        output_path = out_path,
                        offload = list(state.offload_flags) or None,
                        native_speed = state.native_speed,
                        threads = state.threads,
                        extra_args = extra_args or None,
                        env = state.spawn_env(),
                        on_log = self._on_log,
                        cancel_event = cancel,
                    )
                outputs = sd_cli_output_paths(out_path, (layers + 1) if layers else 1)
                for path in _keep_layers(outputs, layers or 0):
                    with Image.open(path) as im:
                        images.append(_native_output_image(state.family, im.copy()))
                    seeds.append(seed_i)
        return images, seeds

    def _on_log(self, line: str) -> None:
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return
        for a, b in _STEP_RE.findall(line):
            a, b = int(a), int(b)
            if gen.passes:
                i = gen.pass_index
                nxt = (i + 1) % len(gen.passes)
                # The next pass (the next image's base pass after the last one) starts once this one completed; a
                # repeat of this pass's last line is not a start.
                if (
                    nxt != i
                    and b == gen.passes[nxt]
                    and gen.step >= gen.pass_offset + gen.passes[i]
                    and (b != gen.passes[i] or a < b)
                ):
                    gen.pass_offset = gen.pass_offset + gen.passes[i] if nxt else 0
                    gen.pass_index = i = nxt
                if b != gen.passes[i]:
                    continue
                a += gen.pass_offset
            elif b != gen.total_steps:
                continue
            now = time.time()
            gen.step = min(a, gen.total_steps)
            if gen.first_step_at == 0.0:
                gen.first_step_at = now
            gen.eta_seconds = _estimate_eta(gen.total_steps, gen.step, gen.first_step_at, now)

    def generate_progress(self) -> dict[str, Any]:
        gen = self._gen
        if gen is None or gen.total_steps <= 0:
            return {
                "active": False,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
            }
        return {
            "active": True,
            "step": gen.step,
            "total_steps": gen.total_steps,
            "fraction": min(gen.step / gen.total_steps, 1.0),
            "eta_seconds": gen.eta_seconds,
        }

    def cancel_generate(self, expected_account: Optional[str] = None) -> bool:
        """Native runner polls this event and kills the sd-cli process tree, so stop lands within
        one poll."""
        with self._lock:
            cancel = self._active_generate_cancel
            if cancel is None:
                return False
            # Rechecked under the lock that bound it: the slot may have changed hands.
            if expected_account is not None and self._active_generate_account not in (
                None,
                expected_account,
            ):
                return False
            cancel.set()
            return True

    # ── Unload / status ──────────────────────────────────────────────────────
    @_invalidates_gpu_memory("sd.cpp unload")
    def unload(
        self,
        *,
        expected_account: Optional[str] = None,
        only_if: Optional[Callable[[], bool]] = None,
    ) -> Optional[dict[str, Any]]:
        with self._lock:
            # Rechecked under the load worker's lock, so a load that ended since costs nothing.
            if only_if is not None and not only_if():
                return None
            if expected_account is not None:
                from .gpu_arbiter import DIFFUSION, GpuBusyForAnotherAccountError
                from hub.services.models.account_access import require_resident_control

                if (
                    self._active_generate_cancel is not None
                    and self._active_generate_account != expected_account
                ):
                    raise GpuBusyForAnotherAccountError(DIFFUSION, 1)
                require_resident_control(
                    DIFFUSION, self._state.repo_id if self._state is not None else None
                )
            # Under the lock: begin_load rebinds this attribute, so an unlocked read could set an event the current load
            # no longer watches.
            self._cancel_event.set()
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            state = self._state
            self._state = None
            self._replacing_server = None
            self._load_token += 1
            self._loading = None
            # Grab a mid-start() uncommitted server too so we can stop it (startup is abortable).
            pending = self._pending_server
            self._pending_server = None
            # Reserve before fields clear, or a probe reads idle and reinstalls over a live process.
            to_stop = [
                srv
                for srv in (state.server if state is not None else None, pending)
                if srv is not None
            ]
            if pending is not None and state is not None and pending is state.server:
                to_stop = to_stop[:1]
            self._reserve_stop(len(to_stop))
        # Stop the resident server outside the lock (terminate can take seconds); a mid-flight generation unwinds as
        # the process goes away.
        for srv in to_stop:
            self._stop_reserved(srv)
        # Wait for a signalled generation to exit: callers treat return as "device is free".
        with self._generate_lock:
            pass
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        # A resident sd-server can exit after load; drop stale state so clients reload, not 500 each time.
        if (
            state is not None
            and state.mode == "server"
            and state.server is not None
            and state.server is not self._replacing_server
            and not state.server.is_alive()
        ):
            with self._lock:
                # Re-checked under the lock: a generation stopping it for a replacement marks it first.
                reap = self._state is state and state.server is not self._replacing_server
                if reap:
                    logger.warning("sd-server exited after load; clearing loaded state")
                    self._state = None
            if reap:
                state = None
        if state is None:
            return {
                "loaded": False,
                "repo_id": None,
                "display_repo_id": None,
                "family": None,
                "base_repo": None,
                "device": None,
                "dtype": None,
                "gguf_variant": None,
                "cpu_offload": False,
                "offload_policy": None,
                "vae_tiling": False,
                "memory_mode": None,
                "speed_mode": None,
                "speed_optims": [],
                "text_encoder_quant": None,
                "transformer_quant": None,
                "attention_backend": None,
                "transformer_cache": None,
                "resolved": None,
                "engine": "sd_cpp",
                "native_mode": None,
                "supports_lora": False,
                "supports_controlnet": False,
                "workflows": [],
                "conditioning": None,
            }
        from core.inference import diffusion_lora
        from core.inference.diffusion_conditioning import conditioning_capabilities
        from hub.utils.gguf import extract_quant_token

        if getattr(state.family, "edit", False):
            workflows = ["edit"] if self._native_edit_ready(state) else []
        else:
            workflows = ["txt2img"]
            if self._native_edit_ready(state):
                workflows += ["reference", "edit"]
            elif self._native_reference_ready(state, "reference"):
                workflows.append("reference")
                if self._native_reference_ready(state, "edit"):
                    workflows.append("edit")
        conditioning = conditioning_capabilities(state.family, workflows)
        # No reference detail natively: sd.cpp sizes inputs to the output area.
        conditioning["reference_resolutions"] = []
        full_fidelity = self._native_reference_fidelity(state)
        conditioning["alpha"] = full_fidelity
        notes: list[str] = []
        # The alpha / padding notes are about the unified family's RGBA path; edit-only families are RGB, source-sized.
        if "edit" in workflows and getattr(state.family, "unified_edit", False):
            if not full_fidelity:
                notes.append(
                    "Transparent parts of input images are filled with white on this native build."
                )
                if state.mode == "server":
                    notes.append(
                        "Input images with a different shape from the output are padded to its "
                        "aspect ratio on this native build."
                    )
            # Measured with upstream c92d73c: 3.7% of pixels stayed transparent at guidance 1, 47.1% at 6.
            notes.append(
                "For transparent output on the native engine, raise Guidance above 1 (upstream uses 6)."
            )
        conditioning["notes"] = notes

        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "display_repo_id": state.display_repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": "gguf",
            "model_kind": "gguf",
            "gguf_filename": state.gguf_filename,
            "gguf_variant": extract_quant_token(state.gguf_filename)
            if state.gguf_filename
            else None,
            # Offload flags passed to sd-cli, minus the --backend pin (a card choice, not an offload).
            "cpu_offload": bool(without_device_backend_flags(state.offload_flags)),
            "offload_policy": (
                "active" if without_device_backend_flags(state.offload_flags) else "none"
            ),
            "vae_tiling": False,
            "memory_mode": None,
            "speed_mode": state.native_speed,
            "speed_optims": [],
            "text_encoder_quant": None,
            "transformer_quant": None,
            "attention_backend": None,
            "transformer_cache": None,
            "resolved": state.resolved,
            "engine": "sd_cpp",
            "supports_lora": diffusion_lora.supports_lora(
                engine = "sd_cpp",
                family = state.family.name,
                model_kind = "gguf",
                transformer_quant = None,
            ),
            "supports_controlnet": False,
            "supports_negative_prompt": state.family.name not in _EMBEDDED_GUIDANCE_FAMILIES,
            "negative_prompt_min_guidance": _negative_prompt_guidance_threshold(state.family),
            # Every family maps guidance to sd.cpp's cfg or embedded guidance (_map_guidance). No denoise previews.
            "supports_guidance": True,
            "supports_live_preview": False,
            "prompt_hint": state.family.prompt_hint,
            # "server" = resident sd-server (load once); "oneshot" = legacy per-image sd-cli.
            "native_mode": state.mode,
            # txt2img always; reference and edit only where this build and its loaded assets run them.
            "workflows": workflows,
            "conditioning": conditioning,
            **_native_hires_status(state),
        }


def _native_hires_status(state: Any) -> dict[str, Any]:
    try:
        from core.inference import diffusion_hires
        hires, reason = diffusion_hires.sd_cpp_capability(
            state.family,
            mode = state.mode,
            binary = getattr(getattr(state, "server", None), "binary", None),
        )
    except Exception as exc:  # noqa: BLE001 - a broken probe hides the control, never the status
        logger.debug("sd_cpp.hires: capability probe failed: %s", exc)
        hires, reason = None, None
    # No img2img on this engine, so the Upscale workflow (and its model upscalers) is not offered.
    return {"hires": hires, "hires_unavailable_reason": reason, "upscale_models": []}


def _install_allowed() -> bool:
    """Whether lazy binary install is permitted (UNSLOTH_DIFFUSION_SD_CPP_INSTALL)."""
    val = os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "auto").strip().lower()
    return val not in ("0", "off", "false", "no")


def _progress(
    phase: Optional[str],
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
    fraction: float = 0.0,
    *,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
        "fraction": fraction,
        "error": error,
    }


_sd_cpp_backend: Optional[SdCppDiffusionBackend] = None


def get_sd_cpp_backend() -> SdCppDiffusionBackend:
    global _sd_cpp_backend
    if _sd_cpp_backend is None:
        _sd_cpp_backend = SdCppDiffusionBackend()
    return _sd_cpp_backend


def generation_in_flight() -> bool:
    """Read the active-generation marker without constructing or locking the backend."""
    backend = _sd_cpp_backend
    return backend is not None and backend._gen is not None
