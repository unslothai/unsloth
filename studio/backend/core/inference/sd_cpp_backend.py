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
import logging
import os
import re
import sys
import threading
import time
import weakref
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from core.inference.diffusion_auto_policy import build_resolved_record, format_generation_for_log
from core.inference.diffusion_compat import flux2_inner_dim_for_pick
from core.inference.diffusion_device import (
    resolve_diffusion_device_target,
    resolve_selected_cuda_ordinal,
)
from core.inference.diffusion_families import (
    DIFFUSION_CANCELLED_MSG,
    DIFFUSION_NOT_LOADED_MSG,
    DiffusionFamily,
    DiffusionModelReplacedError,
    LoadIdentity,
    detect_family_for_pick,
    load_identity,
    family_sd_cpp_supported,
    mirror_repo,
    legacy_source_repo,
    prefer_cached_legacy_source,
    prefer_ungated_mirror,
    resolve_base_repo,
    resolve_local_gguf_child,
    sd_cpp_text_encoders_for,
    supported_family_names,
    _family_override_resolved,
)
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)
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

_install_lock = threading.Lock()

# Installs are writers, one-shot sd-cli runs readers: admission must be one decision.
# Held only across the state change; readers never take _install_lock, so no lock cycle.
_tree_state = threading.Condition()
_tree_readers = 0
_tree_installing = False
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
    """Run ``binary`` out of the managed tree, holding off any install for the duration.

    Only a MANAGED copy needs this. An sd-cli from ``SD_CLI_PATH`` / ``UNSLOTH_SD_CPP_PATH``, an
    in-tree build or ``PATH`` is one the installer never touches, so claiming for it would block
    that generation behind an unrelated bundle download for nothing (and, on a timeout, fail it). A
    timeout is NOT admission: the install still holds the tree, and starting the binary it is
    replacing is the exact race this exists to prevent.

    The wait is cancellable. The caller already holds the generate lock here, so an unload or a
    cancel that could not get out of this would read as a hung Unsloth for up to the whole timeout
    while nothing has even started. Nothing notifies the condition on cancel, so the wait is
    re-checked on a short tick rather than once.
    """
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
                    # The video path recognises only its own cancel sentinel.
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


def _default_threads() -> int:
    """Physical-core thread count for the sd.cpp CPU backend. ``threads = None`` lets sd.cpp pick
    its own default, which is the logical-core count (all hyperthreads). For the compute-bound
    GGML matmuls the diffusion CPU path runs, oversubscribing the hyperthreads adds scheduling
    contention without extra throughput, so pin to physical cores instead. Falls back to 8 when
    the count is unknown, and clamps to at least 1."""
    cpu = os.cpu_count()
    return max(1, cpu // 2 if cpu else 8)


def _server_binary_runnable(binary: str) -> bool:
    """Best-effort probe that ``binary`` can actually execute (not just exist). Runs ``<binary>
    --help`` with the same runtime env the server will use, so a present but unrunnable build
    (wrong arch, missing shared libs, no execute bit) is caught before a multi-GB asset download.
    Conservative: only a clear "cannot launch" signal (OSError, the dynamic-loader exit codes
    126/127, or a Windows image-load status such as 0xC0000135) returns False; anything else is
    treated as runnable so a quirky ``--help`` exit code never blocks a working binary."""
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
        return False
    except Exception:  # noqa: BLE001 -- don't block on a flaky probe
        return True
    # Negative = signal death (e.g. SIGILL); Windows loader failures are large positive codes.
    return (
        proc.returncode >= 0
        and proc.returncode not in (126, 127)
        and proc.returncode not in _WINDOWS_IMAGE_LOAD_FAILURE_STATUSES
    )


def _usable_or_discard_managed(binary: str) -> bool:
    """True if ``binary`` can be kept; False if it is an unusable copy WE own (now removed).

    ``find_sd_*_binary`` only checks that the path is a file, so an interrupted extraction (or a
    prebuilt for the wrong CPU) left a present-but-unrunnable binary that the installer then never
    retried: every load probed it, rejected it, and fell back to diffusers, so native inference
    stayed off until the user deleted the directory by hand.

    Only a copy the installer may replace is removed, i.e. one under the installer-owned root that
    carries its ownership marker. SD_CLI_PATH, UNSLOTH_SD_CPP_PATH, an in-tree build, anything on
    PATH and an unmarked directory at the default path (a user's own stable-diffusion.cpp checkout
    looks exactly like that) are the user's, so an unrunnable one of those is reported as-is rather
    than deleted or reinstalled over. Deleting one would also be unrepairable: install() refuses an
    unmarked non-empty target, so the binary would be gone AND the reinstall refused.
    """
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
    """Combined stdout+stderr of ``binary <args>``, or None when it could not be read. ``sd-cli``
    prints ``--help`` on stdout and exits 0, and logs errors on stderr, so both streams are
    folded together. None means "could not tell" -- cannot exec, timed out, or a non-zero exit
    (which is how an older build rejects a flag it does not know) -- and is never evidence that a
    feature is absent, so every caller has to stay conservative on it."""
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
    """True unless ``binary``'s ``--help`` demonstrably predates MiniMax-H3 support. Conservative by
    design: an unreadable ``--help`` returns True, because the load's existing
    ``SdCppEngine.version()`` gate already refuses a binary that cannot run, and guessing "no H3"
    from a probe failure would take native video away from a working build."""
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
    """True unless ``binary`` demonstrably enumerates the CPU ggml device and nothing else.

    ``sd-cli --list-devices`` prints one ``name<TAB>description`` line per available ggml backend
    device and exits 0, so a CPU-only prebuilt answers ``CPU\t<cpu model>`` while a CUDA / ROCm /
    Vulkan / Metal build adds its own device. That is the only way to tell the two apart after the
    fact: ``find_sd_cpp_binary`` returns whatever is installed regardless of which accelerator it
    was asked for.

    Conservative everywhere else -- unreadable output, or an older build that rejects the flag --
    because neither is evidence that the accelerator is missing. A missing binary is False: there is
    nothing to run on the GPU at all.
    """
    if not binary:
        return False
    return accelerator_verdict_keeps_gpu(sd_cpp_accelerator_device_verdict(binary))


def accelerator_verdict_keeps_gpu(verdict: Optional[bool]) -> bool:
    """ "Could not tell" keeps the GPU: an unreadable probe is not evidence of no accelerator."""
    return True if verdict is None else verdict


def sd_cpp_accelerator_device_verdict(binary: str) -> Optional[bool]:
    """``sd_cpp_lists_accelerator_device`` without the conservative default: None means the probe
    said nothing usable, rather than being folded into "assume it has one". A caller COMPARING
    two readings needs that apart: against a recorded decision, the collapsed True is
    indistinguishable from a real accelerator, so an unreadable re-probe would read as a build
    that changed underneath the load and refuse it."""
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
    """The ``--list-devices`` name for CUDA/ROCm physical index ``ordinal``, or None. None whenever
    the answer is not certain -- no selection, an unreadable probe, a build whose devices are in
    another namespace, an index it does not list -- since the fallback is sd.cpp's own device
    choice, i.e. today's behaviour."""
    if not binary or ordinal is None:
        return None
    text = _sd_cpp_probe_output(binary, "--list-devices")
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


# Last --list-devices answer per binary + (size, mtime_ns); only the capability read reuses it, never the verdict.
_LAST_DEVICE_LISTING: dict = {}


def _binary_stat_identity(binary: str) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(binary)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _remember_device_listing(binary: Optional[str], text: Optional[str]) -> None:
    if not binary:
        return
    if text is None:
        _LAST_DEVICE_LISTING.pop(binary, None)
        return
    _LAST_DEVICE_LISTING[binary] = (_binary_stat_identity(binary), text)


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
    text = None
    seen = _LAST_DEVICE_LISTING.get(binary)
    if seen is not None and seen[0] == _binary_stat_identity(binary):
        text = seen[1]
    elif probe:
        text = _sd_cpp_probe_output(binary, "--list-devices")
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


_GGML_DEVICE_PREFIXES: tuple[str, ...] = ("CUDA", "ROCM", "VULKAN", "SYCL", "METAL", "OPENCL")


def _normalized_card_name(text: str) -> str:
    """Letters and digits only, lowercased: `AMD Radeon RX 7900 XTX (RADV NAVI31)` then contains `AMD Radeon RX 7900 XTX`."""
    return "".join(character for character in text.lower() if character.isalnum())


_DRIVER_TAG_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _without_driver_tag(text: str) -> str:
    """The ICD's trailing driver tag removed, so the rest can be compared for EQUALITY: containment cannot, `RX 7600` is inside `RX 7600 XT`."""
    return _DRIVER_TAG_RE.sub("", text).strip()


# ROCm composes: ROCR filters agents, then HIP (CUDA_VISIBLE_DEVICES) indexes the rest.
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
    hip = next((source.get(var) for var in _HIP_MASK_VARS if source.get(var) is not None), None)
    opaque = source.get(_OPAQUE_MASK_VAR)
    if rocr is None and hip is None and opaque is None:
        return ordinal, False
    if opaque is not None:
        return None, True
    visible: "Optional[list[int]]" = None
    for raw in (rocr, hip):
        if raw is None:
            continue
        entries = _mask_entries(raw)
        if entries is None:
            return None, True
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
    # Count by NAME (what sd_cpp_device_named indexes by), not _card_identity.
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
) -> Optional[str]:
    """The ggml device that IS *card_name*, when exactly one of them is. The Vulkan build names its
    devices in its own namespace, so a physical ordinal names nothing in it. ``position`` breaks a
    tie on the ASSUMPTION that RADV and HIP walk one vendor's GPUs in the same order -- unreadable
    here, so only among devices already agreed to be the right MODEL."""
    if not binary or not card_name:
        return None
    wanted = _normalized_card_name(card_name)
    if not wanted:
        return None
    text = _sd_cpp_probe_output(binary, "--list-devices")
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
    """The trailing "or delete it" clause of the H3 refusal, or "" when there is nothing to delete.

    Only a binary in a layout the installer writes to can be recovered by clearing that layout:
    ``install()`` refuses a non-empty unmarked target, so an empty one is what lets the next load
    put the pinned prebuilt there. Anything PATH or an env var named is elsewhere entirely; the
    refusal used to end with "or remove that directory" whatever the binary was, which for a
    ``/usr/bin/sd`` PATH discovery read as "remove /usr/bin".

    MOVE, never remove. Only the caller's unowned branch reaches this, so a root that matches here
    necessarily carries no ownership marker -- it is the user's own build sitting at the path the
    installer would use, which a ``git clone`` of leejet's repo produces verbatim. Moving it aside
    frees the path without destroying anything. ``in_tree_install_root`` is not consulted at all:
    the installer never writes there.
    """
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
    """``ensure_sd_cpp_binary`` for the MiniMax-H3 path, which additionally requires the binary to
    ADVERTISE H3 support.

    ``ensure_sd_cpp_binary`` hands back whatever ``find_sd_cpp_binary`` locates and only probes
    runnability, so an install that predates H3 is returned unchanged, the H3 load reports ready on
    it, and the first generation fails. Only this path is stricter: image generation must keep
    working on any user-supplied build. Its caller runs it BEFORE resolving the H3 assets, so a
    refusal costs no download.

    A stale copy we own is deleted so the installer puts the pinned prebuilt back; a user's own
    build is left alone and the load fails with a message naming it, the same ownership split
    ``_usable_or_discard_managed`` makes. Returns None when no H3-capable binary can be produced. A
    user-supplied binary that is not stable-diffusion.cpp AT ALL gets its own message: "no H3
    options" is true of every unrelated program, and reporting it as an outdated build is what sent
    #8507 looking for a newer stable-diffusion.cpp that was never installed.
    """
    binary = ensure_sd_cpp_binary(allow_install = allow_install, accelerator = accelerator)
    if not binary:
        return binary
    # One --help answers identity and H3; None is "could not tell" and keeps the binary.
    help_text = _sd_cpp_probe_output(binary, "--help")
    if help_text is None:
        return binary
    # Identity before capability: unrelated tools also expose --ref-video (#8507).
    identified = help_text_identifies_sd_cpp(help_text)
    if identified and help_text_supports_minimax_h3(help_text):
        return binary
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
        logger.warning("managed sd.cpp binary %s %s", binary, fault)
        return None
    # Deleting is a write to the managed tree; take the install claim (not reentrant).
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

    studio_dir = Path(__file__).resolve().parents[3]
    if str(studio_dir) not in sys.path:
        sys.path.insert(0, str(studio_dir))
    import install_sd_cpp_prebuilt

    return install_sd_cpp_prebuilt


# Without this, a host with no asset for its GPU would re-download on every load.
_failed_accelerator_upgrades: set[str] = set()


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


# Upstream ROCm archive needs host HIP/BLAS runtime; fall back to Vulkan, one-way, one-deep.
_ACCELERATOR_FALLBACK: dict[str, str] = {"rocm": "vulkan"}


# The sonames the upstream ROCm archive imports and does not ship.
_ROCM_RUNTIME_SONAMES: tuple[str, ...] = (
    "libamdhip64.so.7",
    "libhipblas.so.3",
    "librocblas.so.5",
)


def rocm_runtime_resolvable() -> Optional[bool]:
    """Whether this host can load the ROCm runtime the prebuilt needs. None when it cannot be asked.

    Uses the real loader (``ctypes``), NOT ``ldconfig -p``: on a gfx1151 runner the cache omitted
    hipblas and rocblas that the loader resolved from /opt/rocm. Not Windows, where the exit status
    (0xC0000135) already decides."""
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


def accelerator_probe_failure_is_decisive(accelerator: Optional[str]) -> bool:
    """Whether a NEGATIVE device probe for ``accelerator`` is explained, so one occurrence is enough.

    A CPU-only answer alone is ambiguous (a busy or masked GPU looks the same); it is decisive only
    when the loader proves the ROCm runtime absent. Otherwise the two-strike rule stands."""
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
    record = {
        "strikes": max(strikes, 0),
        "proven": bool(value.get("proven", False)),
        "fingerprint": fingerprint if isinstance(fingerprint, dict) else {},
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
        return {
            "strikes": int(record.get("strikes", 0) or 0),
            "proven": bool(record.get("proven", False)),
        }
    if not per_card:
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
    """Record that the ``accelerator`` sd.cpp build could not be run on this host. Only for an
    accelerator with a rung below it: noting "cpu" would claim the host cannot run sd.cpp at all.
    Pass ``fingerprint`` when the caller already CHANGED what is fingerprinted -- the load installs
    the fallback first, so a reading taken here would describe the build that REPLACED it."""
    klass = _accelerator_class_of(accelerator)
    if not klass or klass not in _ACCELERATOR_FALLBACK:
        return
    fingerprint = dict(fingerprint) if isinstance(fingerprint, dict) else _accelerator_fingerprint()
    records = _stored_accelerator_runtime_failures()
    records.update({k: v for k, v in _accelerator_runtime_failures.items() if k not in records})
    previous = records.get(klass)
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
        "proven": bool(proven) or bool((previous or {}).get("proven", False)),
        "fingerprint": fingerprint,
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
    if _record_diverts(_accelerator_runtime_failures.get(klass), fingerprint, card):
        return True
    return _record_diverts(_stored_accelerator_runtime_failures().get(klass), fingerprint, card)


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
    """``binary``, unless it is a recorded-unrunnable build that is not the one asked for. An ensure
    does not promise the accelerator it was given: offline it hands back whatever is in the tree,
    which passes every runnability probe and dies mid-render. Compared against the REQUEST, since
    with the fallback off ROCm is asked for on purpose."""
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
                "diverting": enabled and _record_diverts(record, fingerprint),
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
)

# AMBIGUOUS: GPU faults that do not prove the build is the cause.
_ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS: tuple[str, ...] = (
    "unspecified launch failure",
    "hip error",
    "rocm error",
    "rocblas error",
    "memory access fault by gpu node",
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
)


def output_shows_capacity_failure(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_CAPACITY_FAILURE_MARKERS)


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


def preferred_accelerator(accelerator: Optional[str], card: Optional[str] = None) -> str:
    """``accelerator``, or its fallback once shown unrunnable here. Applied at the TOP of an ensure ladder, so a crashed build is not installed and probed again.

    A function of the stored record only, never of the live host: the loader preflight is applied
    where the runtime verdict is interpreted, so this stays reproducible."""
    klass = _accelerator_class_of(accelerator) or (accelerator or "auto")
    nxt = fallback_accelerator_for(klass)
    if nxt and accelerator_runtime_failed(klass, card):
        return nxt
    return accelerator or "auto"


def _incomplete_tree_replacement(exc: BaseException) -> bool:
    """True when an install failed PART WAY through replacing the managed tree. That leaves a
    mixture of two bundles, and the installer withholds the record precisely so the next load
    retries the sweep. Memoising it as a failed upgrade would do the opposite: the mismatch is
    then suppressed for the rest of the process and the mixed tree is served as if it were the
    accelerator that was asked for."""
    try:
        return isinstance(exc, _installer_module().SupersededBinaryError)
    except Exception:  # noqa: BLE001 -- cannot tell -> an ordinary failure
        return False


def _tree_in_use(backend: Any) -> bool:
    """True while ``backend`` may still have a native process executing out of the managed tree.
    Three windows, and all three are live processes running the files an install would replace:
    the resident sd-server; a server that has been spawned but has not committed to ``_state``
    yet (``_pending_server``, exactly the ``SdCppServer.start()`` span, minutes on a large
    checkpoint); and a generation that has been signalled to cancel but has not finished."""
    if backend is None:
        return False
    state = getattr(backend, "_state", None)
    if state is not None and getattr(state, "server", None) is not None:
        return True
    if getattr(backend, "_pending_server", None) is not None:
        return True
    if getattr(backend, "_stopping_servers", 0):
        return True
    return getattr(backend, "_active_generate_cancel", None) is not None


def _managed_tree_in_use() -> bool:
    """True while a native process may still be executing out of the managed install tree.

    An accelerator upgrade REPLACES the binaries in that tree, and Linux refuses to open a running
    executable for writing (ETXTBSY) while Windows locks it, so an install attempted now fails and
    can leave the tree half-written. The load path knows when it is safe and retries after its own
    teardown, but it is not the only entry point: the engine router calls
    ``ensure_sd_server_binary`` / ``ensure_sd_cpp_binary`` directly, BEFORE ``begin_load`` stops
    anything. Answering here covers every caller instead of one.

    Reads the singleton without a lock on purpose: a stale answer either defers an upgrade to the
    next load (harmless) or lets one through in a window the load path guards anyway.
    """
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
    """True when ``binary`` is a managed install built for a DIFFERENT accelerator than the one now
    asked for, so reusing it would silently run the wrong build.

    The case that matters: a host that installed the CPU bundle later forces the native engine on a
    CUDA/ROCm/Vulkan GPU. Both ``ensure_*`` return any runnable binary they find, so without this
    the CPU sd-server is reused forever and generation stays on the CPU even though a matching GPU
    build now exists. The reverse matters too: a host with a recorded GPU install whose device
    target later resolves to CPU keeps running on the GPU, because nothing in the command line asks
    for a CPU backend -- the build itself is the choice.

    Deliberately conservative: only a copy the installer owns is ever replaced, and an install with
    NO record is left alone when the CPU build is wanted, since unrecorded is unknown (GPU assets
    shipped before the record did) and reinstalling every legacy install on a CPU target would
    redownload the bundle for the common case, where the install almost certainly is the CPU one
    already.
    """
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
    """True when ``binary`` is a MISMATCHED sd-server out of the tree an older build left beside the
    Unsloth home, while the CURRENT managed root holds a completed install for ``accelerator`` whose
    bundle shipped no sd-server.

    That install is the authoritative one, and the recorded fact that its bundle is serverless makes
    "no server" the answer rather than "install again": otherwise the finder keeps handing the
    legacy server back, ``_accelerator_changed`` keeps rejecting it as the wrong build, and every
    single load reinstalls the bundle that is already on disk.

    Both halves are required. A legacy server that MATCHES the wanted accelerator is a working
    server and is still preferred over the one-shot CLI. And an install whose record does not say
    ``ships_server: false`` -- an older record without the field, or a bundle that did ship one
    whose binary was later deleted -- is NOT evidence of a serverless bundle, so it must keep
    reinstalling, which is what repairs the missing server.
    """
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
    """Whether the sd.cpp executable at ``binary`` contains the literal ``marker``.

    An architecture upstream has not implemented leaves no trace in the build, so this answers
    "can this particular sd-cli/sd-server run that model" for a build whose version string cannot:
    our mirror releases all name the same upstream base in their tag whatever tree was built, and a
    binary reached through SD_CLI_PATH has no install record to consult at all.

    No marker means no claim, so an unmarked family is True and nothing changes for it. An
    unreadable binary is True as well: refusing a load because a stat failed would take the native
    route away on a host where it works, and the old behaviour (attempt, fail, surface the error)
    is no worse than what this replaces. An optional capability passes ``unreadable = False`` to fail
    closed instead.

    Scanned in chunks with an overlap, since the literal can straddle a boundary, and memoised on
    (path, size, mtime) so a reinstall is noticed while repeated loads read 100+ MB once.
    """
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
    """(width, height, ordered PNG bytes) for one native reference / edit call, decoded through
    the diffusers engine's helper. ``source_sized`` (edit-only families): the output is the source's size on
    the family grid, whatever width / height asked for. A ``full_fidelity`` build gets every image as decoded. An older
    build reads references as RGB, so each is flattened over white first (else transparent pixels
    become noise), and its sd-server centre-crops references to the output aspect, so with
    ``pad_to_output`` each is padded to it instead: white for images, black for a separate mask.
    """
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
        img.save(buf, format = "PNG")
        blobs.append(buf.getvalue())
    return int(width), int(height), blobs


def sd_cpp_binary_runs_family(binary: Optional[str], fam: Any) -> bool:
    """Whether the sd.cpp build at ``binary`` implements ``fam``'s architecture.

    The companion to ``family_sd_cpp_supported``, which only says whether the family has assets to
    hand sd-cli. This is the half that looks at the disk.
    """
    return binary_carries_marker(binary, getattr(fam, "sd_cpp_arch_marker", None))


def _installed_accelerator_of(binary: Optional[str]) -> Optional[str]:
    """The accelerator class recorded for the managed install ``binary`` belongs to.

    None for a binary the installer does not own and for a record that cannot be read: neither is an
    answer, and the only caller uses this to notice that the answer CHANGED, never to decide what to
    install.

    Deliberately not ``_accelerator_changed``: that one answers "should an install run", so it
    stands down while the tree is in use and while an upgrade for this accelerator has already
    failed, and a load that keeps a usable wrong-accelerator build on purpose would be refused by it
    on every load. What the load needs is narrower -- did the tree it resolved this binary out of
    get replaced underneath it.
    """
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


def ensure_sd_cpp_binary(*, allow_install: bool = True, accelerator: str = "cpu") -> Optional[str]:
    """Path to a usable ``sd-cli`` binary, installing the prebuilt once if needed. Returns the
    binary path, or None when it is absent and cannot be installed (install disabled, no network,
    unsupported platform). Never raises -- a None return is the router's signal to fall back to
    diffusers."""
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
        fallback = found if usable else None
        try:
            _install = _installer_module().install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-cli installer import failed: %s", exc)
            return fallback
        # Claim for the whole install incl. download, or a generation is overwritten mid-run.
        with _tree_claimed_for_install() as claimed:
            if not claimed:
                return fallback
            try:
                path = _install(accelerator = accelerator)
                logger.info("sd-cli installed at %s", path)
                return str(path)
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-cli auto-install failed: %s", exc)
                if _incomplete_tree_replacement(exc):
                    # Partial sweep may have removed the fallback; re-find through the usability gate.
                    refound = find_sd_cpp_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None:
                    _note_failed_upgrade(accelerator)
                    _note_failed_pin_upgrade(accelerator)
                return fallback


def ensure_sd_server_binary(
    *, allow_install: bool = True, accelerator: str = "cpu"
) -> Optional[str]:
    """Path to a usable ``sd-server`` binary, installing the prebuilt once if needed. Unlike
    ``ensure_sd_cpp_binary``, this installs when *sd-server specifically* is missing -- even if
    an ``sd-cli`` from an older install is already present -- so an existing one-shot install is
    upgraded to the persistent server (the prebuilt archive ships both). Returns None when it is
    absent and cannot be installed; the backend then uses the one-shot fallback. Never raises."""
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
        fallback = found if usable else None
        try:
            _install = _installer_module().install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-server installer import failed: %s", exc)
            return fallback
        with _tree_claimed_for_install() as claimed:
            if not claimed:
                return fallback
            try:
                _install(accelerator = accelerator)
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-server auto-install failed: %s", exc)
                if _incomplete_tree_replacement(exc):
                    refound = find_sd_server_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None or find_sd_cpp_binary() is not None:
                    _note_failed_upgrade(accelerator)
                    _note_failed_pin_upgrade(accelerator)
                return fallback
        installed = find_sd_server_binary()
        # Finder may hit a legacy server for another accelerator; prefer the fresh sd-cli.
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
    server: Optional[SdCppServer] = None
    mode: str = "server"
    hf_token: Optional[str] = None
    resolved: Optional[dict] = None
    gguf_filename: Optional[str] = None
    # Kept so the delete guard reconstructs the same encoder pick without re-probing.
    flux2_inner_dim: Optional[int] = None
    # Accelerator at choice time: one-shot re-resolves sd-cli per image and must detect swaps.
    sd_accelerator: Optional[str] = None
    # None unless a single resolved CUDA/ROCm card; others cannot use the VRAM floor.
    physical_gpu_id: Optional[int] = None
    selected_card: Optional[str] = None
    off_torch_device: Optional[str] = None
    child_env: tuple[tuple[str, str], ...] = ()

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


def _estimate_eta(total_steps: int, step: int, first_step_at: float, now: float) -> Optional[float]:
    steps_since_first = step - 1
    if not first_step_at or steps_since_first <= 0:
        return None
    per_step = (now - first_step_at) / steps_since_first
    return max(0.0, (total_steps - step) * per_step)


_EMBEDDED_GUIDANCE_FAMILIES = ("flux.1", "flux.1-kontext", "flux.2-dev")


def _map_guidance(
    fam: DiffusionFamily, guidance: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(cfg_scale, guidance) for sd-cli. Guidance-distilled FLUX runs cfg 1.0 plus the embedded guidance; the rest
    (FLUX.2-klein included: no guidance embedder) use real CFG, 1.0 when <= 1. Always explicit: sd.cpp defaults to 7.0."""
    if fam.name in _EMBEDDED_GUIDANCE_FAMILIES:
        return 1.0, (float(guidance) if guidance is not None else None)
    if fam.name == "z-image":
        # diffusers Z-Image computes pos + g * (pos - neg), so its g is standard CFG minus 1 (sd.cpp's cfg 4 == g 3).
        return (float(guidance) + 1.0 if guidance is not None and guidance > 0.0 else 1.0), None
    cfg = float(guidance) if (guidance is not None and guidance > 1.0) else 1.0
    return cfg, None


def _fetch_repo_map(assets: list[tuple[str, str, str]], hf_token: Optional[str]) -> dict[str, str]:
    """upstream asset repo -> the repo to actually fetch from (its ungated mirror, or itself).
    Decided per REPO over its whole file list, the same input ``download_plan`` uses, so staging
    and the load agree even when a repo carries several assets. Two swaps, in order: a GATED
    vendor base goes to its ungated mirror, then a mirror whose community repack is already
    cached goes back to the repack. The second only ever spares an existing install a re-download
    of bytes it already holds under the old repo key; a fresh one still pulls the mirror."""
    by_repo: dict[str, list[str]] = {}
    for repo, filename, _kind in assets:
        by_repo.setdefault(repo, []).append(filename)
    return {
        repo: prefer_cached_legacy_source(prefer_ungated_mirror(repo, hf_token, files = names), names)
        for repo, names in by_repo.items()
    }


class _NeverRaised(Exception):
    """Placeholder ``except`` target for a hub layout with no LocalEntryNotFoundError."""


def _local_entry_not_found_error() -> type[BaseException]:
    """huggingface_hub's "not cached and downloads are disabled" error, or an unraisable stand-in.
    Resolved lazily and defensively for the same reason the rest of this module imports
    ``huggingface_hub`` inside functions: an unexpected hub layout must degrade to today's error,
    never break the import or swallow an unrelated exception. The stand-in matches nothing, so a
    missing class simply leaves the raw hub error on load-progress."""
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


_UNREAD_LOADING_CARD = object()


class SdCppDiffusionBackend:
    """Native sd.cpp backend with the diffusers ``DiffusionBackend`` method surface."""

    _loading_cards = None
    _committed_loading_card: Optional[str] = None
    # Thread-local like _loading_card: a rejected concurrent load must not move it.
    _loading_families = None
    _loading_family_default: Any = None

    def __init__(self, engine: Optional[SdCppEngine] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._engine = engine
        # An injected engine pins one-shot; a fallback-cached one must not.
        self._engine_injected = engine is not None
        self._state: Optional[_SdState] = None
        self._loading: Optional[_SdLoading] = None
        self._load_token = 0
        # Replaced (never cleared) per load, so a cancelled asset pull stays cancelled.
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        self._active_generate_account: Optional[str] = None
        self._pending_server: Optional[SdCppServer] = None
        self._gen: Optional[_SdGen] = None
        self._deferred_accelerator_install = False
        # Created here but unset: an explicit None would be this thread's answer forever.
        self._loading_cards = threading.local()
        # unload() stops outside the lock, so fields read idle while the process still runs.
        self._stopping_servers = 0
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
        # Only for off-thread readers; the worker reads its thread-local first.
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
        """The installer accelerator this host's device target resolves to (cpu / cuda / rocm /
        vulkan). Lazy import avoids an import cycle with the engine router.

        ``preferred_accelerator`` is applied here so all four call sites agree, or
        ``_accelerator_changed`` would reinstall over what the others chose."""
        from core.inference.diffusion_engine_router import image_install_accelerator
        return preferred_accelerator(
            image_install_accelerator(getattr(resolve_diffusion_device_target(), "backend", "cpu")),
            card,
        )

    def _resolve_engine(self) -> SdCppEngine:
        """The SdCppEngine, installing the binary on first use. Raises if unusable."""
        if self._engine is not None and self._engine.is_available():
            # Check the cached engine too: it outlives the load and may predate the family.
            cached_binary = getattr(self._engine, "binary", None)
            if cached_binary:
                self._refuse_incapable_build(cached_binary, "sd-cli")
            return self._engine
        # Host's accelerator, never cpu: this is also the server-start fallback path.
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
        """Raise when ``binary`` does not implement the family being loaded.

        The router drops an incapable build before choosing native, but its verdict is local to that
        call: this class resolves both executables again, and the two can be DIFFERENT builds
        (separate SD_SERVER_PATH / SD_CLI_PATH, or a partially upgraded tree). A current server that
        fails to start then falls through to a pre-Qwen-2.1 sd-cli, which the router never approved.
        Failing the load here is the point: the alternative is a one-shot backend that reports ready
        and dies on the first generation.
        """
        fam = self._loading_family
        if fam is None or sd_cpp_binary_runs_family(binary, fam):
            return
        raise RuntimeError(
            f"the {name} build at {binary} predates {getattr(fam, 'name', 'this model')} support. "
            "Reinstall the native engine (delete the managed stable-diffusion.cpp install, or point "
            "SD_CLI_PATH / SD_SERVER_PATH at a build from after upstream added it)."
        )

    def _resolve_backend(self) -> tuple[str, Optional[str], Optional[SdCppEngine]]:
        """Pick the native execution mode: ("server", binary, None) or ("oneshot", None, engine).
        The persistent ``sd-server`` is preferred (load once, serve many); the one-shot
        ``sd-cli`` is the fallback for older / custom builds that lack the server target. An
        explicitly injected engine forces one-shot (the unit-test seam and an escape hatch), so a
        test never spawns a real server or triggers an install. A lazily cached fallback engine
        does NOT force one-shot: once a resident server becomes available, the next load can use
        it instead of being pinned to one-shot for the whole session."""
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
        """Land the install this load deferred, now the managed tree is free. Called under both
        locks with the old server stopped and the previous generation finished, which is the only
        moment nothing is executing out of the tree. ``server_binary`` is None on a serverless
        install (one-shot sd-cli only): the install still has to run, since the same archive
        carries the sd-cli this load is about to generate with -- skipping it there is what left
        a CUDA request committing the old CPU CLI. Returns the upgraded path, the one passed in
        when nothing changed or the install could not deliver, or None when there was no server
        and the archive has none: never worse than what the load already had."""
        if not _install_allowed():
            return server_binary
        try:
            accelerator = self._resolved_accelerator(self._loading_card)
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

    def begin_load(
        self,
        repo_id: str,
        *,
        # Covers model assets only, not the sd-cli/sd-server binary install.
        local_files_only: bool = False,
        display_repo_id: Optional[str] = None,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        model_kind: Optional[str] = None,
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ids: Optional[list[int]] = None,
        gpu_ordinal: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate, then fetch assets on a daemon thread. Returns at once."""
        hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
        from core.inference.diffusion_engine_router import off_torch_sd_cpp_device

        off_torch = off_torch_sd_cpp_device()
        if off_torch is not None:
            gpu_ids, gpu_ordinal = None, None
        # Direct callers pass gpu_ids alone; re-rank only when the route did not.
        if gpu_ordinal is None:
            gpu_ordinal = (
                resolve_selected_cuda_ordinal(gpu_ids)
                if gpu_ids and resolve_diffusion_device_target().device == "cuda"
                else None
            )
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name."
            )
        if not family_sd_cpp_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no native sd.cpp asset mapping.")
        self._loading_family = fam

        base = resolve_base_repo(fam, base_repo)
        inner_dim = self._flux2_inner_dim(
            repo_id, gguf_filename, fam, hf_token, allow_network = False
        )
        # Record the repos _asset_specs actually fetches so the delete guard protects them.
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
                fam = fam,
                family_override = family_override,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                gpu_ordinal = gpu_ordinal,
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
        fam: DiffusionFamily,
        family_override: Optional[str] = None,
        hf_token: Optional[str],
        local_files_only: bool = False,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        off_torch: Any = None,
        _load_token: int,
        _cancel_event: Optional[threading.Event] = None,
    ) -> None:
        cancel_event = _cancel_event if _cancel_event is not None else self._cancel_event
        # Outside so the backstop can unpublish it; a leaked one blocks every later install.
        started: Optional[SdCppServer] = None
        self._loading_card = selected_card_identity(gpu_ordinal)
        try:
            mode, server_binary, engine = self._resolve_backend()
            if mode == "server":
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
                assert engine is not None
                if engine.version() is None:
                    note_unlaunchable_accelerator_build(
                        getattr(engine, "binary", None),
                        card = self._loading_card,
                    )
                    raise RuntimeError("sd-cli binary is present but not runnable.")

            # Swap once so size probe and download agree; offline asks memo/local header only.
            _assert_pick_is_not_speech(
                repo_id, gguf_filename, hf_token, allow_network = not local_files_only
            )
            inner_dim = self._flux2_inner_dim(
                repo_id, gguf_filename, fam, hf_token, allow_network = not local_files_only
            )
            specs = self._asset_specs(repo_id, gguf_filename, fam, inner_dim)
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

            files = SdCppModelFiles(
                diffusion_model = paths["diffusion_model"],
                vae = paths.get("vae"),
                clip_l = paths.get("clip_l"),
                clip_g = paths.get("clip_g"),
                t5xxl = paths.get("t5xxl"),
                llm = paths.get("llm"),
                llm_vision = paths.get("llm_vision"),
                qwen2vl = paths.get("qwen2vl"),
            )
            device = (
                off_torch.accelerator
                if off_torch is not None
                else resolve_diffusion_device_target().device
            )
            spawn_env = off_torch.child_env() if off_torch is not None else None
            offload: tuple[str, ...] = ()
            if device != "cpu":
                offload = tuple(offload_flags(_memory_policy(memory_mode, cpu_offload)))
            # Device pin added where flags reach a binary: the binary can still change below.
            gpu_ordinal = gpu_ordinal if device == "cuda" else None
            native_speed = _native_speed_for(speed_mode)

            # Lock taken only now so the download never serialises generation.
            with self._lock:
                if self._load_token != _load_token:
                    return
                if self._active_generate_cancel is not None:
                    self._active_generate_cancel.set()
            with self._generate_lock:
                with self._lock:
                    if self._load_token != _load_token:
                        return
                    old_state = self._state
                    self._state = None
                    if old_state is not None and old_state.server is not None:
                        self._reserve_stop()
                if old_state is not None and old_state.server is not None:
                    self._stop_reserved(old_state.server)
                # Tree is free now; the deferred install can land.
                if self._deferred_accelerator_install:
                    self._deferred_accelerator_install = False
                    upgraded = self._upgraded_or_refused(server_binary, mode = mode, engine = engine)
                    if mode == "server":
                        server_binary = upgraded
                    # This load's own install IS the decision; both pins move.
                    server_accelerator = _installed_accelerator_of(server_binary)
                    engine_accelerator = _installed_accelerator_of(getattr(engine, "binary", None))
                # A new checkpoint earns a fresh attempt on the GPU backend: the previous abort says nothing about
                # this graph.
                self._cpu_backend_forced = False
                server: Optional[SdCppServer] = None
                if mode == "server":
                    assert server_binary is not None
                    # Re-resolve under the reader claim: an install may have swept this path during the download.
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
                        elif _installed_accelerator_of(server_binary) != server_accelerator:
                            raise RuntimeError(
                                "The stable-diffusion.cpp server binary was replaced by an install "
                                "for a different accelerator while this model was loading. Try the "
                                "load again."
                            )
                        else:
                            _refuse_off_torch_build_mismatch(off_torch, server_binary)
                            server = SdCppServer(server_binary)
                            # Publish inside the claim and recheck cancel under the same lock as unload.
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
                    # Clear `started`, not `server` (None on fallback); a started server stays published.
                    started_ok = False
                    try:
                        server.start(
                            files,
                            vae_format = fam.sd_cpp_vae_format,
                            offload = _offload_with_device_pin_impl(
                                offload, server_binary, gpu_ordinal
                            ),
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
                        gpu_ordinal,
                    )
                )
                state = _SdState(
                    repo_id = repo_id,
                    display_repo_id = display_repo_id,
                    base_repo = base,
                    family = fam,
                    device = device,
                    files = files,
                    vae_format = fam.sd_cpp_vae_format,
                    native_speed = native_speed,
                    offload_flags = committed_offload_flags,
                    threads = _default_threads(),
                    sampling_method = fam.sd_cpp_sampling_method,
                    flow_shift = fam.sd_cpp_flow_shift,
                    server = server,
                    mode = mode,
                    hf_token = hf_token,
                    resolved = build_resolved_record(
                        {"family_override": _family_override_resolved(family_override, fam)}
                    ),
                    gguf_filename = gguf_filename,
                    flux2_inner_dim = inner_dim,
                    sd_accelerator = engine_accelerator if mode == "oneshot" else None,
                    # Never on an off-torch card: the parent-visible ids are torch's, so CUDA0 would name one of them.
                    physical_gpu_id = (
                        _resolved_server_physical_gpu_id(
                            server_binary,
                            device,
                            gpu_ordinal,
                            committed_offload_flags,
                        )
                        if mode == "server" and off_torch is None
                        else None
                    ),
                    selected_card = self._loading_card,
                    off_torch_device = off_torch.label if off_torch is not None else None,
                    child_env = tuple(sorted((spawn_env or {}).items())),
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
                        self._committed_loading_card = self._loading_card
                        self._loading = None
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
                from .gpu_arbiter import DIFFUSION, restore_owner_account
                from hub.services.models.account_access import restore_resident_metadata

                restore_owner_account(DIFFUSION)
                restore_resident_metadata("diffusion")
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
        """The repos + exact files a NATIVE load of this pick needs, in the same envelope the diffusers
        backend returns, so the Hub download manager stages what sd-cli will actually open.

        The two engines want different files: diffusers builds a pipeline around the base repo's
        sharded components, while sd-cli reads the single-file VAE + text encoders declared in
        ``diffusion_families``. Planning with the wrong engine stages tens of GB the load never
        opens and then pulls the native assets inline, outside the manager's progress and disk
        preflight -- so the route asks whichever engine it predicts the load will select. The
        diffusers-only kwargs (quant / memory / LoRA) are accepted and ignored, exactly as
        ``begin_load`` accepts them.
        """
        if not gguf_filename:
            raise ValueError(
                "gguf_filename is required: the native engine loads single-file GGUF checkpoints only."
            )
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None or not family_sd_cpp_supported(fam):
            raise ValueError(f"'{repo_id}' has no native sd.cpp asset mapping.")
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
        # Merged: two upstream repos can share one fetch repo.
        merged: dict[str, list[str]] = {}
        for repo, names in by_repo.items():
            into = merged.setdefault(fetch_repo[repo], [])
            into.extend(n for n in names if n not in into)
        by_repo = merged
        fetch_repo_id = fetch_repo.get(repo_id, repo_id)
        # AFTER the swap: preflighting the upstream id would refuse the very picks the ungated mirror exists to rescue
        self._preflight_companion_repos(by_repo, fetch_repo_id, hf_token)
        sizes = self._plan_file_sizes(by_repo, hf_token)
        entries: list[dict[str, Any]] = []
        total = 0
        from core.inference.diffusion import DiffusionBackend

        for repo, names in by_repo.items():
            total += int(sum(sizes.get((repo, n), 0) for n in names))
            # Same missing-file filter as diffusers; required_bytes keeps the unfiltered footprint.
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
            if kind == "diffusion_model":
                try:
                    if Path(repo).expanduser().exists():
                        continue
                except (OSError, RuntimeError, ValueError):
                    pass
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
        """Refuse a companion repo this pick cannot read, before any byte is fetched.

        The native asset list carries its own companion repos (flux.1's VAE is the gated
        black-forest-labs/FLUX.1-schnell), and neither ``_plan_file_sizes`` nor the size probe
        surfaces the 401: the entry is planned at 0 bytes and the fetch dies on the bare Hub token
        error this replaces. Run from BOTH the plan and ``_run_load``, as the diffusers backend
        does, because the UI falls back to /images/load when the plan call fails.

        ``local_files_only`` skips it entirely. The probe is a ``model_info`` call plus, for a gated
        repo, a metadata HEAD -- pure network, whose whole purpose is to turn a 401 that would
        otherwise arrive mid-download into a licence URL up front. A cache-only load never starts
        that download: it either resolves the companion from disk or fails on the local miss, which
        is the clearer error of the two.
        """
        if local_files_only:
            return
        from core.inference.diffusion import _assert_base_repo_accessible

        for repo, names in by_repo.items():
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
        """The companion refusal ``_run_load`` makes, run by the route BEFORE it takes the GPU. Same
        signature and reason as the diffusers backend's: ``_run_load`` runs on the load thread,
        after a forced-native load on a GPU host already evicted chat, so a pick refused only
        there unloads the resident model first. Nothing to check without a family or checkpoint
        name."""
        if fam is None or not gguf_filename:
            return
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
    ) -> list[tuple[str, str, str]]:
        """(repo, filename, kind) for every file sd-cli needs. ``kind`` is the SdCppModelFiles
        field; the transformer reuses the diffusers GGUF."""
        specs: list[tuple[str, str, str]] = [(repo_id, gguf_filename, "diffusion_model")]
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        # Encoder per variant: header dim when read, else load identity.
        for terepo, tefile, kind in sd_cpp_text_encoders_for(
            fam, repo_id, gguf_filename, inner_dim = inner_dim
        ):
            specs.append((terepo, tefile, kind))
        return specs

    def _set_expected_bytes(
        self, assets: list[tuple[str, str, str]], hf_token: Optional[str]
    ) -> None:
        """Best-effort total download size for the progress bar (0 if unknown)."""
        total = 0
        try:
            from huggingface_hub import HfApi
            api = HfApi(token = hf_token)
            for repo, fn, kind in assets:
                if kind == "diffusion_model" and Path(repo).expanduser().exists():
                    continue
                try:
                    info = api.get_paths_info(repo, paths = [fn], expand = False)
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
        """Download every asset (cancellable via this load's own ``cancel_event``, so a replacement
        load cannot un-cancel this pull), returning kind -> local path. ``local_files_only``
        resolves each asset from the HF cache and never from the network; an asset that is not
        there fails HERE, with the repo and filename named, rather than being quietly pulled.
        This is the last and only network call left on an offline load's path, so it is the one
        that has to honour the flag rather than merely accept it."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        cancel = cancel_event if cancel_event is not None else self._cancel_event
        paths: dict[str, str] = {}
        # Assets may be gated: swap per repo exactly as download_plan does.
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
                    )
                except _local_entry_not_found_error() as exc:
                    # Only fires under local_files_only; restate with repo and file for the toast.
                    if not local_files_only:
                        raise
                    if kind == "llm_vision" and vision_optional:
                        # Only editing reads the projector; older caches still load for text-to-image.
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
            with self._lock:
                if self._loading is not None:
                    try:
                        self._loading.downloaded_bytes += os.path.getsize(path)
                    except OSError:
                        pass
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
        """Repo ids an in-flight background load is downloading (empty when idle). Mirrors the
        diffusers backend so the delete-cached guard can query whichever engine is active.
        Includes the companion VAE / text-encoder repos, since deleting one mid-load would remove
        files the committed SdCppModelFiles paths need, and the mirror of each, where those bytes
        land once a gated asset repo is swapped out."""
        with self._lock:
            loading = self._loading
            if loading is None or loading.error is not None:
                return ()
            ids = (loading.repo_id, loading.base_repo, *loading.asset_repos)
            return _with_mirrors(ids)

    def loaded_repo_ids(self) -> tuple[str, ...]:
        """Repo ids the COMMITTED native model reads from disk (empty when unloaded). The one-shot
        sd-cli re-reads the companion VAE / text-encoder files from the HF cache on every
        generation (server mode keeps them in the resident process, but the extra ids are
        harmless there), so the delete-cached guard must refuse those companion repos while the
        model is loaded -- status().repo_id covers only the main GGUF. Reconstructed from the
        committed family, mirroring loading_repo_ids(), and carrying the mirrors too."""
        with self._lock:
            state = self._state
            if state is None:
                return ()
            fam = state.family
            repos = [state.repo_id, state.base_repo]
            if fam.sd_cpp_vae:
                repos.append(fam.sd_cpp_vae[0])
            # Same per-variant selection as _asset_specs; never re-probed under _lock.
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
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        upscale: Optional[float] = None,
        reference_images: Optional[list[str]] = None,
        workflow: Optional[str] = None,
        reference_resolution: Optional[int] = None,
        localized_edit: Any = None,
        loras: Optional[list[tuple[str, float]]] = None,
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
        # load_identity() of the caller's status() read; refuse rather than run a different load (#9448)
        expected_load: Optional[LoadIdentity] = None,
        allow_oversized: bool = False,
    ) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

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
        from hub.services.models.account_access import media_generation_slot

        with self._generate_lock, media_generation_slot("diffusion"):
            with self._lock:
                state = self._state
                if state is None:
                    raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
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
                if reference_images and not getattr(state.family, "reference", False):
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
                    if not self._native_edit_ready(state):
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
                    if not cancel.is_set() and DIFFUSION_CANCELLED_MSG not in str(exc):
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
    ) -> tuple[list, list[int]]:
        """Generate via the resident sd-server (no model reload).

        A batch larger than the server's per-job limit is split into chunks: the server rejects a
        batch_count above _MAX_SERVER_BATCH, and the one-shot path served large batches
        image-by-image. The base seed is masked to sd.cpp's signed-int64 range (the request model /
        diffusers accept larger seeds), and each chunk is submitted at base+offset so the per-image
        seeds stay reproducible. Each chunk gets a timeout proportional to its image count so a slow
        CPU batch is not cancelled partway through on one fixed deadline.

        LoRA on the server goes through the structured ``lora`` request field, NOT prompt tags (the
        sdcpp API intentionally ignores ``<lora:>`` in the prompt). Selected adapters are staged
        into the server's ``--lora-model-dir`` scratch dir, which the server rescans per request,
        and referenced by their staged filename.
        """
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
        lora_payload: Optional[list[dict]] = None
        lora_stage: Optional[Path] = None
        if lora_resolved:
            server_lora_dir = state.server.lora_dir
            if server_lora_dir:
                lora_stage = Path(server_lora_dir) / f"gen_{os.urandom(6).hex()}"
                materialized = diffusion_lora.materialize_native_dir(lora_resolved, lora_stage)
                lora_payload = [
                    {
                        "path": f"{lora_stage.name}/{Path(m.path).name}",
                        "multiplier": float(m.weight),
                    }
                    for m in materialized
                ]
        # One deadline shared by all chunks so a long batch still ends on time.
        deadline = time.monotonic() + NATIVE_GENERATION_TIMEOUT_S
        try:
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
                    flow_shift = state.flow_shift,
                    cfg_scale = cfg_scale,
                    distilled_guidance = flux_guidance,
                    lora = lora_payload,
                    ref_images = [
                        "data:image/png;base64," + base64.b64encode(b).decode("ascii")
                        for b in ref_pngs or []
                    ],
                    qwen_image_layers = layers,
                )
                try:
                    blobs = state.server.img_gen(
                        payload,
                        on_step = self._on_log,
                        cancel_event = cancel,
                        total_timeout = max(deadline - time.monotonic(), 1.0),
                    )
                except RuntimeError as exc:
                    # A ggml unsupported-op abort killed the server: this graph cannot run on the GPU backend at all,
                    # so restart the model on the CPU backend once and retry this chunk. Any other death propagates.
                    server = self._restart_server_on_cpu_backend(state, str(exc), cancel)
                    if server is None:
                        raise
                    state = replace(state, server = server)
                    with self._lock:
                        if self._state is not None and self._state.server is not None:
                            self._state = state
                    blobs = server.img_gen(
                        payload,
                        on_step = self._on_log,
                        cancel_event = cancel,
                        total_timeout = max(deadline - time.monotonic(), 1.0),
                    )
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
            if lora_stage is not None:
                shutil.rmtree(lora_stage, ignore_errors = True)
        return images, seeds

    def _restart_server_on_cpu_backend(
        self, state: _SdState, error_text: str, cancel: threading.Event
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
        """
        if not is_ggml_unsupported_op_abort(error_text):
            return None
        if self._cpu_backend_forced or state.device == "cpu":
            return None  # already on CPU: the abort is not a backend-placement problem
        if state.server is None or cancel.is_set():
            return None
        server_binary = find_sd_server_binary()
        if not server_binary:
            return None
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
        try:
            server.start(
                state.files,
                vae_format = state.vae_format,
                # Drop the device pin: sd.cpp joins repeated --backend values, so it would win again.
                offload = without_device_backend_flags(state.offload_flags),
                native_speed = state.native_speed,
                threads = state.threads,
                env = state.spawn_env(),
                extra_args = list(CPU_BACKEND_FLAGS),
            )
        except Exception:  # noqa: BLE001 -- the original abort is the more useful error
            server.stop()
            return None
        finally:
            with self._lock:
                if self._pending_server is server:
                    self._pending_server = None
        return server

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
        """Fallback path: re-run one-shot sd-cli per image (reloads the model each time). LoRA on
        the one-shot path uses sd-cli's own mechanism: materialize the selected adapters into a
        ``--lora-model-dir`` and inject matching ``<lora:ALIAS:w>`` tags into the prompt (sd-cli
        parses and strips them). supports_lora already gated the family upstream, so a non-empty
        ``lora_resolved`` is safe to apply here."""
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        engine = self._resolve_engine()
        extra_args: list[str] = []
        if state.vae_format:
            extra_args += ["--vae-format", state.vae_format]
        if state.flow_shift is not None:
            extra_args += ["--flow-shift", repr(float(state.flow_shift))]

        images = []
        seeds: list[int] = []
        with tempfile.TemporaryDirectory(prefix = "sdcpp_gen_") as tmpdir:
            eff_prompt = prompt
            lora_dir: Optional[str] = None
            if lora_resolved:
                materialized = diffusion_lora.materialize_native_dir(
                    lora_resolved, Path(tmpdir) / "loras"
                )
                eff_prompt = diffusion_lora.inject_prompt_tags(prompt, materialized)
                lora_dir = str(Path(tmpdir) / "loras")
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
                    # Re-resolve inside the claim: an install may have moved sd-cli.
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
            if int(b) == gen.total_steps:
                now = time.time()
                gen.step = min(int(a), gen.total_steps)
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
        """Signal the in-flight generation to stop, matching DiffusionBackend.cancel_generate.

        The native engine is stricter than best-effort: the runner polls this event and kills
        the sd-cli process tree, so the stop lands within the poll interval rather than at the
        next step boundary. Returns False when nothing is running."""
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

    @_invalidates_gpu_memory("sd.cpp unload")
    def unload(self, *, expected_account: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
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
            self._load_token += 1
            self._loading = None
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
        if (
            state is not None
            and state.mode == "server"
            and state.server is not None
            and not state.server.is_alive()
        ):
            logger.warning("sd-server exited after load; clearing loaded state")
            with self._lock:
                if self._state is state:
                    self._state = None
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
        conditioning = conditioning_capabilities(state.family, workflows)
        conditioning["reference_resolutions"] = []
        full_fidelity = self._native_reference_fidelity(state)
        conditioning["alpha"] = full_fidelity
        notes: list[str] = []
        if "edit" in workflows and not getattr(state.family, "edit", False):
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
            # "server" = resident sd-server (load once); "oneshot" = legacy per-image sd-cli.
            "native_mode": state.mode,
            "workflows": workflows,
            "conditioning": conditioning,
        }


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
