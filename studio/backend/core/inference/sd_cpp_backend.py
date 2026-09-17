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

import contextlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

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
from utils.account_context import account_thread, current_account_id
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = get_logger(__name__)

# A sampling-progress line ("4/4", "[ 12/ 28]", "sampling: 50%|...| 14/28"). Only a denominator matching the requested
# step count is trusted, so a stray "1/100" cannot move the bar.
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Serialises the one-time binary install so concurrent first-loads don't race.
_install_lock = threading.Lock()

# Admission control over the managed tree, because "is anything running in there?" and "start replacing it" have to be
# ONE decision. _managed_tree_in_use() alone is a point-in-time sample, and an install spends seconds to minutes
# downloading before it extracts: a one-shot generation admitted inside that window launches the very sd-cli the
# extraction then overwrites. Installs are the writers, one-shot sd-cli runs are the readers. Held only across the
# state change, never across a download or a generation, and the readers never take _install_lock, so there is no
# cycle.
_tree_state = threading.Condition()
_tree_readers = 0
_tree_installing = False
# A download can legitimately take minutes; wait rather than run a binary that is being replaced
_TREE_WAIT_TIMEOUT_S = 900.0
# How often the wait re-checks for cancellation. Nothing notifies the condition when a request is cancelled, so a
# single long wait would hold the generate lock past an unload.
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
                    # The caller's own sentinel: the video path recognises only its own, and an image message reaching
                    # it reads as "Video generation failed" for what is an ordinary cancellation.
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
    Conservative: only a clear "cannot launch" signal (OSError, or the dynamic-loader exit codes
    126/127) returns False; anything else is treated as runnable so a quirky ``--help`` exit code
    never blocks a working binary."""
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
    # Negative return code = signal death (e.g. -4 SIGILL from an incompatible prebuilt): launches then crashes, so
    # treat as unavailable.
    return proc.returncode >= 0 and proc.returncode not in (126, 127)


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


# The ``--help`` token that marks a build carrying MiniMax-H3 support. Upstream added the mode's own options
# (``--ref-video`` / ``--ref-audio``, and the "MiniMax-H3 Ref2VA" wording on ``--ref-image``) in the very commit that
# added H3, so their presence is exactly the capability signal. ``--version`` cannot stand in for it: the release
# prebuilts are built without a .git dir and answer "version unknown, commit unknown".
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
    """Both of ``ensure_h3_sd_cpp_binary``'s questions against a live binary, on ONE ``--help``. The
    capability marker cannot stand alone here: ``--ref-video`` is a plain option name that
    unrelated reference-video tools expose too, so a caller re-checking only capability would
    accept a program the gate itself would have refused on identity -- the difference between "an
    sd.cpp build too old for H3" and "not sd.cpp at all" (#8507). Same conservative default as
    ``sd_cpp_supports_minimax_h3``: an unreadable ``--help`` is "could not tell", and the
    caller's own ``version()`` gate already refuses a binary that will not run."""
    text = _sd_cpp_probe_output(binary, "--help")
    if text is None:
        return True
    return help_text_identifies_sd_cpp(text) and help_text_supports_minimax_h3(text)


# The ``--help`` tokens marking a build with the graph-cut executor; both are required, since --stream-layers does
# nothing without --max-vram.
_GRAPH_CUT_HELP_MARKERS: tuple[str, ...] = ("--max-vram", "--stream-layers")


def sd_cpp_supports_graph_cut(binary: Optional[str]) -> bool:
    """True only when ``binary``'s ``--help`` advertises the graph-cut executor. The opposite
    default to ``sd_cpp_supports_minimax_h3``, and for the same reason each is safe: that gate
    refuses a build, so "cannot tell" has to keep it, while this one ADDS flags, and sd-cli exits
    non-zero on an option it does not know. Guessing yes from an unreadable ``--help`` would
    break every generation on an older build instead of merely leaving it as slow as it is today."""
    if not binary:
        return False
    text = _sd_cpp_probe_output(binary, "--help")
    if text is None:
        return False
    return all(marker in text for marker in _GRAPH_CUT_HELP_MARKERS)


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
    """The conservative collapse ``sd_cpp_lists_accelerator_device`` applies, as a function of a
    verdict already in hand: "could not tell" keeps the GPU, because an unreadable probe is not
    evidence that the accelerator is missing.

    Split out so the rule lives in ONE place. A caller that needs the raw verdict for a second
    question (the H3 load needs it to tell "enumerates the CPU only" from "could not be asked at
    all") would otherwise re-implement the collapse beside it and the two could drift."""
    return True if verdict is None else verdict


def sd_cpp_accelerator_device_verdict(binary: str) -> Optional[bool]:
    """``sd_cpp_lists_accelerator_device`` without the conservative default: None means the probe
    said nothing usable, rather than being folded into "assume it has one". A caller COMPARING
    two readings needs that apart: against a recorded decision, the collapsed True is
    indistinguishable from a real accelerator, so an unreadable re-probe would read as a build
    that changed underneath the load and refuse it."""
    text = _sd_cpp_probe_output(binary, "--list-devices")
    if text is None:
        return None
    names = [line.split("\t", 1)[0].strip() for line in text.splitlines() if "\t" in line]
    if not names:
        return None
    return any(name.upper() != "CPU" for name in names)


# ggml device-name prefixes indexed by CUDA physical ordinal (ggml names its HIP backend either way); Vulkan is
# excluded, its ordinals are another namespace.
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
    # Said out loud rather than dropped in silence: the load still runs, on whichever device this build picks for
    # itself, which is what happens today for every native load. Refusing instead would take the GPU selection from
    # "not honoured here" to "cannot load at all" on any build older than the one that added --list-devices, including
    # a user's own SD_CLI_PATH copy, since sd.cpp treats an unknown argument as fatal.
    logger.warning(
        "sd_cpp.device_pin_unresolved: this build does not report a CUDA/ROCm device %s "
        "(--list-devices %s), so the graph runs on its own default device",
        ordinal,
        "was unreadable" if text is None else "does not list it",
    )
    return None


# Every namespace ggml names a device in. The physical-index list above is deliberately
# narrower: these are the prefixes that identify a device LINE, whatever indexing scheme it
# belongs to, which is what a match by card name needs.
_GGML_DEVICE_PREFIXES: tuple[str, ...] = ("CUDA", "ROCM", "VULKAN", "SYCL", "METAL", "OPENCL")


def _normalized_card_name(text: str) -> str:
    """A card description reduced to the letters and digits in it, lowercased.

    Two spellings of the same card differ in punctuation and in the parenthesised driver
    tag a Vulkan ICD adds: `AMD Radeon RX 7900 XTX` against
    `AMD Radeon RX 7900 XTX (RADV NAVI31)`. Comparing the reduced forms by containment is
    what lets the second be recognised as the first.
    """
    return "".join(character for character in text.lower() if character.isalnum())


_DRIVER_TAG_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _without_driver_tag(text: str) -> str:
    """A card description with the trailing parenthesised driver tag removed.

    A Vulkan ICD appends its own: `AMD Radeon RX 7600 (RADV NAVI33)`. Dropping it is what
    lets the rest be compared for EQUALITY, which containment cannot do -- `AMD Radeon RX
    7600` is contained in `AMD Radeon RX 7600 XT`, and those are two different cards.
    """
    return _DRIVER_TAG_RE.sub("", text).strip()


# The masks that make torch's device list something other than the physical one. ROCm applies
# them in this order, and they COMPOSE: ROCR_VISIBLE_DEVICES filters the agents the ROC runtime
# reports, and HIP_VISIBLE_DEVICES (for which CUDA_VISIBLE_DEVICES is the alias, so setting both
# is documented as unsafe) then indexes into what is left rather than into the physical order.
# Each is "the devices listed, in the order listed", so a mask both filters AND reorders.
# GPU_DEVICE_ORDINAL is a third spelling at the HIP level; it is not composed here, it only makes
# the translation decline, which is the safe direction.
_ROCR_MASK_VAR = "ROCR_VISIBLE_DEVICES"
_HIP_MASK_VARS = ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
_OPAQUE_MASK_VAR = "GPU_DEVICE_ORDINAL"


def _mask_entries(value: Optional[str]) -> "Optional[list[int]]":
    """A visibility mask as physical indices, or ``None`` if it is not written as indices.

    A mask may name UUIDs (`GPU-...`), which this cannot translate into an enumeration order,
    so an unreadable mask declines rather than guessing.
    """
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
    """``(HIP device id, a mask was set)`` for a torch-visible *ordinal*.

    With no mask set the two namespaces are the same list, so the ordinal is returned as it
    arrived. With a mask this composes the levels; when any of them cannot be read as indices
    the index is ``None`` and the caller declines the tie-break rather than pinning a guess.
    """
    source = os.environ if env is None else env
    rocr = source.get(_ROCR_MASK_VAR)
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


def _physical_position_of(hip_index: int) -> "tuple[Optional[str], Optional[int]]":
    """``(name, position)`` for a HIP device id, read from the host's own card enumeration.

    torch cannot answer this once a mask is set: it can only see the cards it was left. The
    hardware inventory enumerates every card, so it is what says how many cards of the same
    kind come before this one in the order the Vulkan build will walk.

    The two numbers are NOT the same number. What a visibility mask names, and what torch
    reports as `cuda:N`, is the HIP device id, derived from the KFD node id. The inventory's
    `index` is its own probe row, amd-smi's enumeration order over its KFD/sysfs view, which
    its docstring says is not a pin. They coincide on most hosts and not on all, which is why
    `utils.hardware.amd.get_hip_id_by_gpu_index` exists -- amd-smi publishes the mapping for
    exactly this. Without that mapping there is no way to say which row the mask selected, so
    this declines and `sd_cpp_device_named` falls back to a name-only pin rather than
    counting positions in the wrong space.
    """
    try:
        from utils.hardware.amd import get_hip_id_by_gpu_index
        from utils.hardware.hardware import get_physical_gpu_inventory

        inventory = get_physical_gpu_inventory(block = False)
        if (inventory or {}).get("unknown"):
            return None, None
        devices = [
            device
            for device in ((inventory or {}).get("devices") or [])
            if isinstance(device, dict) and device.get("index") is not None
        ]
        hip_by_row = get_hip_id_by_gpu_index()
    except Exception:  # noqa: BLE001 -- no reader, no position; the name alone still pins
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
    # Counted on the NAME, because the name is what the matcher counts: `sd_cpp_device_named`
    # indexes the Vulkan devices whose description matches, and it has nothing else to index
    # by. Counting on `_card_identity` instead -- which carries the gfx target, for the
    # fingerprint's purposes -- made a gfx1103 APU and a gfx1151 card that both call
    # themselves `AMD Radeon(TM) Graphics` two different groups, so selecting the second gave
    # position 0 and the pin named the first Vulkan device of that name.
    name = (selected.get("name") or "").strip() or None
    if name is None:
        # The grouping the matcher will use cannot be established from a row the OS did not
        # name, so the tie-break is withheld rather than guessed. A card that is alone of its
        # name is still pinned, by name.
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
    """The card at torch-visible index *ordinal* as the driver names it, and its place among
    the physical cards of that same name. ``(None, None)`` when unreadable.

    For the accelerator fallback, where the build's devices are in a namespace the physical
    index means nothing in. The name is what the two namespaces agree on, so it is what the
    pin is matched by. `torch.cuda` is the reader on ROCm as well: HIP is exposed through the
    same API and reports the same marketing name the Vulkan ICD prints.

    The POSITION is what breaks a tie between identical cards, which a name cannot: two
    7900 XTXs describe themselves identically in both namespaces, so what is carried instead
    is "this is the second one". See `sd_cpp_device_named` for the assumption that rests on.

    That count has to be taken over the PHYSICAL cards. `HIP_VISIBLE_DEVICES` and its
    relatives filter and reorder what torch enumerates, and the Vulkan child gets no
    equivalent mask -- Vulkan does not read those variables -- so it walks every card. Counted
    inside the visible list, a mask of `1` on two identical cards made the selected card "the
    first one of its name", and the pin then named physical card 0 while Studio reserved and
    accounted for card 1. So the ordinal is translated back to a physical index and the
    position is taken from the host's own enumeration; where that translation or that
    enumeration cannot be had, the position is ``None`` and `sd_cpp_device_named` declines the
    tie-break instead of pinning the wrong card.
    """
    if ordinal is None:
        return None, None
    try:
        import torch
        if not torch.cuda.is_available() or ordinal >= torch.cuda.device_count():
            return None, None
        names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 -- no reader, no pin; the load runs as it does today
        return None, None
    name = (names[ordinal] or "").strip()
    physical_index, masked = _physical_index_of(ordinal)
    if not masked:
        # The visible list IS the physical list, so torch answers both halves.
        if not name:
            return None, None
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
    """The ggml device that IS *card_name*, when exactly one of them is.

    For the accelerator fallback. The Vulkan build's devices are in their own namespace, so
    the physical ordinal the user picked names nothing in it and `--backend` went unwritten:
    sd.cpp then chose its own default device, normally the first, while Studio recorded and
    reserved the card that was selected. On a mixed box the card's own name is the one thing
    both namespaces agree on, so that is what this matches.

    Two identical cards produce two identical descriptions, and the name alone cannot tell
    them apart. ``position`` is how many cards of that same name come BEFORE the selected one
    in the physical enumeration, and the match at that position is taken. Both namespaces
    enumerate the GPUs of one vendor in the order the driver reports them -- PCI bus order for
    amdgpu, which is what RADV's `vkEnumeratePhysicalDevices` and HIP's device list both walk
    -- so the n-th of a run of identical cards is the same card in either. That is an
    assumption about the driver, not a fact this can read, which is why it is only used to
    break a tie between devices already agreed to be the right MODEL, and why the counts have
    to agree: a ``position`` outside the matches pins nothing.

    None unless the match is unambiguous. No name, no position and several candidates means a
    guess dressed as a pin, which is what this exists to stop.
    """
    if not binary or not card_name:
        return None
    wanted = _normalized_card_name(card_name)
    if not wanted:
        return None
    text = _sd_cpp_probe_output(binary, "--list-devices")
    if text is None:
        return None
    # Two readings of every line, because containment alone cannot tell two models apart:
    # `AMD Radeon RX 7600` is contained in `AMD Radeon RX 7600 XT`. The exact one drops the
    # driver tag and compares the rest for equality, which is the model; the loose one is the
    # containment that recognises a tagged spelling of the same card.
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
        # Every candidate is the selected MODEL, so a run of them is a run of identical cards
        # and the position means what it says.
        matches, same_model = exact, True
    else:
        matches = [name for name, _described in loose]
        # Only when the candidates describe one model. `position` counts the cards of the
        # selected name in the PHYSICAL enumeration, so applying it to a list that mixes a
        # 7600 with a 7600 XT pins by a count that never included the other model: position 0
        # can land on the XT when the non-XT was selected, and the load then runs on, and
        # accounts for, a different GPU. Mixed models are an ambiguity, not a tie.
        same_model = len({described for _name, described in loose}) == 1
    if matches and position is not None and not (same_model and 0 <= position < len(matches)):
        # A position the enumeration cannot honour is UNRESOLVED, including when only one
        # device answers: nothing says that singleton is the card at that position, and the
        # selection was made against the physical list. Returning it anyway meant the graph
        # could run on one card while the arbiter reserved and accounted for the other.
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
        # A run of identical cards, and the selection's place in that run. Same vendor, same
        # driver, same enumeration order on both sides; see the note above for why that is
        # the assumption and not a reading.
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
    # ONE --help, two questions: is this stable-diffusion.cpp, and does this build carry H3. A second spawn would
    # double the cost of the refusal path and could read a different build than the one just judged. None is "could
    # not tell", which stays conservative on both counts. Conservative HERE means keeping the binary, the opposite of
    # the engine's identity probe, which rejects on an unreadable one: this decides whether to refuse a binary the
    # user chose, where a probe failure must not take native video away from a working build, while the engine decides
    # whether to ADOPT an ambiguously named PATH candidate on no evidence at all.
    help_text = _sd_cpp_probe_output(binary, "--help")
    if help_text is None:
        return binary
    # Identity BEFORE capability, never the marker alone. --ref-video is a plain option name that unrelated
    # reference-video tools also expose, so returning early on it would readmit exactly the class of program #8507 was
    # about, through SD_CLI_PATH instead of PATH. Upstream added H3 eight months after print_usage started with the
    # project banner, so a genuine H3 build always answers both.
    identified = help_text_identifies_sd_cpp(help_text)
    if identified and help_text_supports_minimax_h3(help_text):
        return binary
    # What is wrong with it, for the log lines on the managed path below: a managed copy that is not sd.cpp at all is
    # still deleted and reinstalled, but calling it an old build would be false.
    fault = "does not advertise MiniMax-H3 support" if identified else "is not stable-diffusion.cpp"
    if not is_managed_binary(binary):
        # Not an old sd.cpp -- not sd.cpp at all. Worth its own message: the H3 marker is missing from EVERY program
        # that is not stable-diffusion.cpp, so reporting the capability verdict here sent users hunting for a newer
        # build of something they never installed (#8507, where the binary was Debian/Ubuntu's `sd` find-and-replace
        # tool). Discovery already skips an unrelated PATH `sd`, so what reaches this line came from somewhere the
        # identity gate does not cover -- an SD_CLI_PATH / UNSLOTH_SD_CPP_PATH override, an in-tree developer build,
        # or a PATH `sd-cli`. None of them is ours to overwrite, so all four say so and stop.
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
    # Deleting it is a WRITE to the managed tree, so it takes the same admission an install does. An image one-shot
    # may be executing this very file: on Linux the running child survives the unlink but the next image in the batch
    # can no longer resolve it, and on Windows the unlink fails outright and the H3 load is refused. Held only across
    # the unlink -- ensure_sd_cpp_binary below claims the tree itself, and the claim is not reentrant.
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


# Accelerators whose upgrade install already failed this process. Without this, a host that asks for a GPU build it
# has no asset for would re-resolve (and re-download) on every single load, because the wrong-accelerator binary it
# keeps still does not match the request.
_failed_accelerator_upgrades: set[str] = set()


def _note_failed_upgrade(accelerator: str) -> None:
    """Stop retrying an accelerator upgrade that just failed while a usable binary is kept."""
    try:
        _failed_accelerator_upgrades.add(_installer_module().accelerator_class(accelerator))
    except Exception:  # noqa: BLE001
        pass


# The accelerator to try when the one a host asks for cannot be made to run there, and nothing else.
#
# A ROCm sd.cpp prebuilt is published for one generic ROCm target, not per gfx arch, so a card whose
# hipBLAS that build carries no kernels for cannot start it at all: sd-cli dies in hipblasSetStream
# with CUBLAS_STATUS_INVALID_VALUE on gfx1201 (#9278) and never gets past tensor loading on gfx1100
# (#8814). Vulkan runs on both of those cards, the installer already resolves a "vulkan" asset for
# Linux and Windows alike, and the alternative on this path is the CPU build. So Vulkan is the rung
# between "the ROCm build does not work here" and "give up on the GPU".
#
# One-way and one-deep on purpose: nothing falls back FROM Vulkan (the CPU rung below it already
# exists), and no other accelerator has a broken-per-arch story to fall back from. CUDA either
# works or the host has no CUDA asset at all, which the CPU rung has always handled.
_ACCELERATOR_FALLBACK: dict[str, str] = {"rocm": "vulkan"}


def sd_cpp_vulkan_fallback_enabled() -> bool:
    """Whether the ROCm -> Vulkan rung may be taken. ``UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0``
    turns it off for a host that would rather see the ROCm failure than be moved to Vulkan."""
    raw = (os.environ.get("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "auto") or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def fallback_accelerator_for(accelerator: Optional[str]) -> Optional[str]:
    """The accelerator to try after ``accelerator`` could not be made to run, or None.

    None whenever there is no next rung, the knob is off, or the answer would be the accelerator
    that was already asked for, so a caller can ladder on this without special-casing any of them.
    """
    if not sd_cpp_vulkan_fallback_enabled():
        return None
    try:
        want = _installer_module().accelerator_class(accelerator)
    except Exception:  # noqa: BLE001 -- cannot classify it, so there is no rung to take
        want = (accelerator or "").strip().lower()
    nxt = _ACCELERATOR_FALLBACK.get(want)
    return nxt if nxt and nxt != want else None


# Accelerators whose BUILD could not be run on this host, as distinct from _failed_accelerator_upgrades
# above, which is about an install that could not be fetched. Persisted as well as memoised: a ROCm
# build that crashes mid-render (#9278 crashes after tensor loading, minutes in) would otherwise be
# selected again by the very next load, and again after a restart, because nothing about the install
# is wrong -- the asset is present, correct for "rocm", and simply does not run on this card.
#
# The record is NOT a bare set of accelerator names. A name alone outlives the thing it is a fact
# about: the claim being stored is "THIS sd.cpp build cannot drive THESE cards through THIS runtime",
# and a driver upgrade, a new card or a new bundle all make it stale while the name stays the same.
# So each entry carries the fingerprint it was observed under and is dropped once that provably
# changes; see _accelerator_fingerprint and _fingerprint_still_applies.
_ACCELERATOR_RUNTIME_FAILURES_KEY = "sd_cpp_accelerator_runtime_failures"
# accelerator class -> record dict, the in-process mirror of the persisted map.
_accelerator_runtime_failures: dict[str, dict] = {}

# How many AMBIGUOUS failures it takes to divert a host. A decisive one is enough on its own; see
# _ACCELERATOR_RUNTIME_FAILURE_MARKERS below for which is which. Two rather than one because the
# ambiguous markers name a GPU fault without proving the BUILD is the cause, and the cost of being
# wrong is a working ROCm installation bypassed until someone finds the reset.
_AMBIGUOUS_FAILURE_STRIKES = 2


def _accelerator_class_of(accelerator: Optional[str]) -> str:
    try:
        return _installer_module().accelerator_class(accelerator)
    except Exception:  # noqa: BLE001 -- the raw string is close enough for a preference note
        return (accelerator or "").strip().lower()


def _discovered_managed_root() -> Optional[str]:
    """The managed root of the build this host would actually use, or None when there is none.

    Cheap and never raises: `find_sd_cpp_binary` is a path lookup, and every caller here has a
    default to fall back on.
    """
    try:
        found = find_sd_cpp_binary()
        return owning_managed_root(found) if found else None
    except Exception:  # noqa: BLE001 -- a root that cannot be resolved is simply not one
        return None


def _accelerator_fingerprint(binary: Optional[str] = None) -> dict:
    """What the note is a fact ABOUT: the sd.cpp bundle, the GPU runtime and the cards themselves.

    Every component is optional and every one is gathered from something already resolved or
    already cached, so this adds no probe to the load path. A component that cannot be read is
    recorded as None rather than omitted, so "we could not tell" is distinguishable from "it was
    not part of the fingerprint when this was written".

    Never raises. A fingerprint that cannot be built at all is ``{}``, which
    ``_fingerprint_still_applies`` treats as "cannot tell", i.e. the record keeps applying.
    """
    fp: dict = {"bundle": None}
    try:
        # The root the BINARY is in when one is named, not the current default. The finder also
        # serves a tree an older build left beside the Unsloth home, and a failure recorded for
        # a binary out of that one would otherwise carry the default root's tag -- None, or a
        # tag belonging to an unrelated install -- so replacing or upgrading the legacy bundle
        # could never invalidate the record and the host stayed diverted until it was cleared
        # by hand.
        # With no binary named -- every consultation -- the root is resolved from the build
        # that is actually DISCOVERED rather than from the current default. The finder also
        # serves a tree an older build left beside the Unsloth home, and a failure recorded
        # against that tree's tag was then compared with the default root's: stale at once if
        # that root carries any other tag, and never retired at all if it carries none, so
        # replacing the legacy bundle could not clear the diversion.
        root = owning_managed_root(binary) if binary else _discovered_managed_root()
        record = _installer_module().read_install_record(root or managed_install_root())
        if isinstance(record, dict):
            # The release the managed tree came from, read LIVE rather than memoised: a new bundle
            # is a new build, an install can land inside the life of this process, and a new build
            # is exactly the thing that can start working on a card the old one could not serve.
            tag = record.get("tag")
            fp["bundle"] = str(tag) if tag else None
    except Exception:  # noqa: BLE001 -- no bundle component, not a failure
        pass
    fp.update(_host_fingerprint())
    return fp


# The GPU runtime this interpreter is bound to. Memoised because it genuinely cannot change inside
# one process: it is a string baked into the torch wheel at build time, and the fingerprint is
# consulted on EVERY load through accelerator_runtime_failed. Cleared by _reset_host_fingerprint.
#
# The CARDS are deliberately NOT memoised beside it; see _host_fingerprint.
_RUNTIME_FINGERPRINT_MEMO: Optional[dict] = None
# Retained under its old name so a caller (or a test) that pinned the whole host half still works:
# when it is set, it wins, exactly as it used to.
_HOST_FINGERPRINT_MEMO: Optional[dict] = None


def _reset_host_fingerprint() -> None:
    """Drop the memoised host half. For tests, and for anything that knows the host changed."""
    global _HOST_FINGERPRINT_MEMO, _RUNTIME_FINGERPRINT_MEMO
    _HOST_FINGERPRINT_MEMO = None
    _RUNTIME_FINGERPRINT_MEMO = None


def _runtime_fingerprint() -> dict:
    """``{"runtime": ...}``, computed once per process. Never raises."""
    global _RUNTIME_FINGERPRINT_MEMO
    if _RUNTIME_FINGERPRINT_MEMO is not None:
        return dict(_RUNTIME_FINGERPRINT_MEMO)
    fp: dict = {"runtime": None}
    try:
        import torch  # noqa: PLC0415 -- already imported by the backend; local to keep this cheap

        # torch.version.hip is the ROCm version the torch WHEEL was built against, written into
        # torch/version.py at build time. Read for what it is: it moves when the user reinstalls
        # torch from a different ROCm index, and it does NOT move when only the driver or /opt/rocm
        # is upgraded underneath the same wheel. So it retires a record on a torch swap and it is
        # not the component that answers a driver-only fix; the cards below, and failing those the
        # DELETE route, are. It needs no subprocess and no driver call to read.
        runtime = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        fp["runtime"] = str(runtime) if runtime else None
    except Exception:  # noqa: BLE001
        pass
    _RUNTIME_FINGERPRINT_MEMO = dict(fp)
    return fp


def _card_identity(device: dict) -> Optional[str]:
    """How one enumerated card is named in the fingerprint, falling back when the OS gives no name.

    The marketing name is preferred because it is the most legible and the most specific. But it
    is frequently absent on exactly the hosts this feature is about: measured on the gfx1151 Linux
    runner, `get_physical_gpu_inventory` answers from `sysfs-drm` with
    ``{'vendor': 'amd', 'index': 0, 'name': None, 'memory_total_gb': 64.0,
    'gfx_candidates': ['gfx11', 'gfx1151']}``. Filtering on ``name`` dropped that device, the list
    came out empty, and the cards component was None on a machine with a card in it -- so a record
    written there could never be retired by a card change, which is the one component that answers
    a driver-side fix (``torch.version.hip`` is baked into the wheel and does not move).

    The gfx target is carried ALONGSIDE the name rather than only as its fallback, because the
    claim being stored is "this sd.cpp build has no code objects for this card", which is a
    statement about the gfx target rather than about the retail name -- and AMD ships a great many
    distinct targets under one generic name. `AMD Radeon(TM) Graphics` is the name both a gfx1103
    laptop APU and a gfx1151 Strix Halo report, so a name-only identity would let a card swap that
    genuinely changes the answer leave the fingerprint untouched and keep diverting the new card to
    Vulkan. Vendor, index and pool size are the last resort so a card that answers with none of the
    above still contributes something that changes when the hardware does.

    Returns None only for a device that reports nothing at all, which stays filtered out.
    """
    name = device.get("name")
    gfx = device.get("gfx_candidates") or device.get("gfx") or device.get("arch")
    if isinstance(gfx, (list, tuple)):
        # Most specific last in the inventory's own ordering (['gfx11', 'gfx1151']), and all of it
        # is kept: a shorter family string alone would make gfx1100 and gfx1151 the same card.
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
    """The fingerprint identity of the card at torch-visible index *ordinal*, or ``None``.

    The same ``_card_identity`` the fingerprint is built from -- name AND gfx target -- so a
    record written about one card can be compared with the card a later load selected. Read
    off the host's own enumeration through the mask translation `physical_card_name` uses,
    because a visibility mask filters and reorders what torch sees and the inventory does not.

    ``None`` means "cannot tell", which every consumer treats as "the record still applies":
    nothing here may narrow a note on a guess.
    """
    if ordinal is None:
        return None
    try:
        from utils.hardware.amd import get_hip_id_by_gpu_index
        from utils.hardware.hardware import get_physical_gpu_inventory

        physical_index, masked = _physical_index_of(ordinal)
        if physical_index is None:
            return None
        inventory = get_physical_gpu_inventory(block = False)
        if (inventory or {}).get("unknown"):
            return None
        devices = [
            device
            for device in ((inventory or {}).get("devices") or [])
            if isinstance(device, dict) and device.get("index") is not None
        ]
        if masked:
            # A mask names HIP ids; the inventory rows are its own probe order, and amd-smi
            # publishes the mapping between them. Without it there is no saying which row was
            # selected, so this declines rather than counting in the wrong space.
            hip_by_row = get_hip_id_by_gpu_index()
            if not hip_by_row:
                return None
            physical_index = next(
                (row for row, hip in hip_by_row.items() if hip == physical_index), None
            )
            if physical_index is None:
                return None
        selected = next((d for d in devices if d.get("index") == physical_index), None)
        if selected is None:
            return None
        return _card_identity(selected)
    except Exception:  # noqa: BLE001 -- no reader, no narrowing
        return None


def _host_fingerprint() -> dict:
    """``{"runtime": ..., "gpus": ...}``. Never raises.

    The cards are re-read on every call rather than memoised with the runtime. Memoising them was
    measured to freeze the wrong answer: ``get_physical_gpu_inventory(block = False)`` on a COLD
    cache returns the unknown sentinel and only schedules the probe, so the first call in a process
    yields ``gpus = None`` even on a host with cards, and a memo then carried that None for the
    life of the process -- past the refresh landing a second later, and past any card added or
    removed while Studio runs. Since the note is written by a load, the record that a card change
    is supposed to retire is exactly the one most likely to have been written with no cards in it.
    Re-reading costs a monotonic compare and a dict return once the cache is warm, and the call is
    still non-blocking, so a wedged driver still cannot stall the load path.
    """
    if _HOST_FINGERPRINT_MEMO is not None:
        return dict(_HOST_FINGERPRINT_MEMO)
    fp: dict = {"gpus": None}
    fp.update(_runtime_fingerprint())
    try:
        from utils.hardware.hardware import get_physical_gpu_inventory

        # Non-blocking, the same call the request path uses: on a load path a wedged driver must
        # never stall this, and a cold cache answering "unknown" simply leaves this component None.
        # The cards the OS enumerates are the right grain, since swapping the card is the other way
        # a note stops being true.
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
    """Whether a record written under ``stored`` still speaks for a host fingerprinted ``current``.

    A component invalidates the record only when BOTH sides are known and they DIFFER. Unknown on
    either side counts as matching, deliberately: the components are best-effort reads, and a probe
    that intermittently answers "unknown" would otherwise make the host flip between ROCm and Vulkan
    from load to load, which is worse than either answer. The direction that matters is that a real,
    observed change in driver, bundle or cards does retire the record.
    """
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
    """One persisted entry in the shape the readers expect, or None when it is unusable.

    Tolerates the bare-list shape an early build of this feature wrote, reading it as one decisive
    strike with no fingerprint, which is what it meant. Never raises.
    """
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
    # The cards this was seen on, when the writer could tell. Dropping them here would read a
    # card-scoped note back as a host-wide one on the next process, which is the whole of what
    # the scoping is for; an entry that names none keeps applying to every card, as it did.
    cards = [str(card).strip() for card in (value.get("cards") or []) if str(card).strip()]
    if cards:
        record["cards"] = sorted(set(cards))
    return record


def _stored_accelerator_runtime_failures() -> dict[str, dict]:
    """The persisted map, or an empty one. Never raises: a store this cannot read only costs the
    preference, and refusing the load over it would be worse than the failure it is avoiding."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Owner-scoped, like every other host-level preference: which sd.cpp build runs on this
        # machine is a fact about the machine, not about whoever happens to be generating.
        stored = run_as(OWNER, get_app_setting, _ACCELERATOR_RUNTIME_FAILURES_KEY, None)
    except Exception:  # noqa: BLE001
        return {}
    if isinstance(stored, str):
        # A string is tolerated because the value is written through a JSON column: a row saved by
        # a caller that pre-serialised it reads back as the text of a mapping, not a mapping.
        try:
            import json
            stored = json.loads(stored)
        except ValueError:
            return {}
    if isinstance(stored, (list, tuple, set)):
        # The shape an early build of this feature wrote: a plain list of accelerator names.
        stored = {str(item).strip().lower(): True for item in stored if str(item).strip()}
    if not isinstance(stored, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in stored.items():
        name = str(key).strip().lower()
        record = _normalise_failure_record(name, value) if name else None
        if record is not None:
            out[name] = record
    return out


def _persist_accelerator_runtime_failures(records: dict[str, dict]) -> None:
    """Write the map back. Never raises: the in-process mirror still holds for this run."""
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
    """Record that the ``accelerator`` sd.cpp build could not be run on this host.

    Only ever records for an accelerator that has a rung below it: noting one with no fallback
    would persist a row nothing reads, and noting "cpu" would be a claim that the host cannot run
    stable-diffusion.cpp at all, which no failure here establishes.

    ``proven`` is the difference between evidence and suspicion. The load path passes True because
    it has ENUMERATED the answer: the host's own build could not be asked for its devices and the
    fallback build listed one. A decisive sd-cli message passes True for the same reason, the
    message names the build having no code for the card. An ambiguous message passes False, and
    those only divert the host once ``_AMBIGUOUS_FAILURE_STRIKES`` of them have been seen under one
    fingerprint, so a single transient GPU fault never moves a working installation.

    A record observed under a different fingerprint is REPLACED rather than added to: the strikes
    counted against the old driver or the old bundle are not evidence about the new one.

    ``fingerprint`` is for a caller that has already CHANGED the thing being fingerprinted. The
    load path installs the fallback bundle before it gets here, and the bundle tag is read live
    out of the managed install record, so a fingerprint taken now describes the build that
    replaced the failed one. That is the wrong fact: the next release's ROCm asset would then
    match the stored tag and the record would suppress the retry that might have worked. Taking
    the reading BEFORE the install and passing it in is what keeps the note about the build that
    actually failed. Omitted, it is read now, which is right for every caller that has installed
    nothing.
    """
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
        # Keep what the earlier strike KNEW. A component that cannot be read right now is
        # recorded as None, and `_fingerprint_still_applies` reads an unknown as compatible,
        # so overwriting a known GPU or bundle value with None loses the only thing that could
        # ever invalidate this record: once the record is diverting the host, a later card or
        # bundle change is compared against an unknown and skipped, and the note becomes
        # permanent. Merging is safe in the other direction too -- the check above has already
        # established that nothing known about this host contradicts the stored fingerprint.
        fingerprint = _fingerprint_with_known_fields_kept(previous.get("fingerprint"), fingerprint)
    strikes = (previous or {}).get("strikes", 0) + 1
    # WHICH card this was seen on, when the caller could tell. A host can carry cards the same
    # build serves and cards it does not -- one ROCm bundle, several gfx targets -- and a note
    # keyed on the accelerator alone moved every later load off ROCm because one card could not
    # run it. Accumulated rather than overwritten: a second card failing the same build is one
    # more card it does not serve, not a correction of the first. An unknown card adds nothing
    # and leaves the record unnarrowed, which is what keeps it applying to every selection this
    # cannot identify.
    cards = [c for c in ((previous or {}).get("cards") or []) if c]
    if card and card not in cards:
        cards = sorted([*cards, card])
    record = {
        "strikes": strikes,
        "proven": bool(proven) or bool((previous or {}).get("proven", False)),
        "fingerprint": fingerprint,
    }
    if cards:
        record["cards"] = cards
    if previous == record:
        return
    records[klass] = record
    _accelerator_runtime_failures[klass] = record
    _persist_accelerator_runtime_failures(records)


def _fingerprint_with_known_fields_kept(
    previous: Optional[dict], current: Optional[dict]
) -> Optional[dict]:
    """*current*, with any component it could not read taken from *previous* instead.

    Only called once the two have been shown not to contradict each other, so the carried
    value is a fact about this same host that this reading merely failed to re-take.
    """
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
    """Whether one record is enough to move this host off its own accelerator.

    *card* is the card this selection will actually run on, when the caller knows it. A record
    that names the cards it was seen on says nothing about a different one, so it does not
    divert it. Either side unknown, or a record from before the cards were recorded, keeps the
    record applying: narrowing a note on an unknown is how a card that really cannot run the
    build gets handed it again.
    """
    if not isinstance(record, dict):
        return False
    known_cards = [c for c in (record.get("cards") or []) if c]
    if card and known_cards and card not in known_cards:
        return False
    if not _fingerprint_still_applies(
        record.get("fingerprint"),
        _accelerator_fingerprint() if fingerprint is None else fingerprint,
    ):
        return False
    if record.get("proven"):
        return True
    return int(record.get("strikes", 0) or 0) >= _AMBIGUOUS_FAILURE_STRIKES


def accelerator_runtime_failed(accelerator: Optional[str], card: Optional[str] = None) -> bool:
    """Whether the ``accelerator`` build is already known not to run on this host, under a
    fingerprint that still describes this host -- and, when the caller names the card this
    selection will run on, on THAT card rather than on any card of this host."""
    klass = _accelerator_class_of(accelerator)
    if not klass:
        return False
    fingerprint = _accelerator_fingerprint()
    if _record_diverts(_accelerator_runtime_failures.get(klass), fingerprint, card):
        return True
    return _record_diverts(_stored_accelerator_runtime_failures().get(klass), fingerprint, card)


def usable_or_recorded_failure(
    binary,
    requested,
    card = None,
):
    """``binary``, unless it is a build this host has already recorded as unrunnable AND it is
    not the one that was asked for.

    An ensure does not promise the accelerator it was given: with installing switched off, on
    an offline host, or after a failed download it returns whatever usable build is already in
    the managed tree. On a host that recorded a ROCm crash and cannot fetch the Vulkan rung
    that is the ROCm build, which still answers ``--list-devices``, so the runnability probes
    pass and the load commits the very build the record exists to avoid -- and the failure the
    record describes is a crash MID-RENDER, minutes and a full download later.

    The comparison is against what was REQUESTED, not against the record alone. When the
    fallback is switched off (``UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0``)
    ``preferred_accelerator`` deliberately asks for ROCm again despite the record, and that
    opt-out means "run it anyway": rejecting the build that was asked for would send the load
    to the CPU rung instead of retrying ROCm, which is the opposite of what the switch
    promises. So only a SUBSTITUTE that is recorded bad is refused.

    A user-supplied binary has no recorded class and is never refused here: unrecorded is
    unknown, and the identity and capability tests already cover it.
    """
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
    except Exception as exc:  # noqa: BLE001 -- cannot tell -> unchanged behaviour
        logger.debug("could not check the sd.cpp accelerator record: %s", exc)
    return binary


def accelerator_runtime_failure_state() -> dict:
    """What the settings route reports: every record, whether it is currently diverting the host,
    and the fingerprint it was taken under. Read-only and never raises."""
    fingerprint = _accelerator_fingerprint()
    records = _stored_accelerator_runtime_failures()
    records.update(_accelerator_runtime_failures)
    # "Diverting" is a claim about what loads are doing right now, and with the knob off they
    # are not: `fallback_accelerator_for` returns None, so `preferred_accelerator` leaves the
    # host's own accelerator selected however many strikes the record holds. Reporting the
    # record's own verdict here produced `enabled: false, diverting: true`, which tells an
    # operator their loads are being redirected on the one host where they have explicitly
    # asked that they not be.
    #
    # Only the report is gated. `_record_diverts` still answers for the record alone, which is
    # what `accelerator_runtime_failed` wants: whether this build is known not to run here is a
    # fact about the host, not about the switch.
    #
    # Nothing is hidden by this: `strikes`, `proven` and `stale` are unchanged, so a record
    # that WOULD divert is still fully visible next to `enabled: false`.
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
    """Forget every note, so the next load tries the host's own accelerator again. This is what a
    driver update or a new card is: the note is about a build on a host, and both of those change
    the host.

    Reached from the settings route (``DELETE /settings/diffusion-accelerator-fallback``), which is
    the point: a preference that outlives the condition that set it has to have a way back that is
    not "edit the application database". Clearing BOTH halves matters, since the in-process mirror
    would otherwise keep diverting this process after the persisted record is gone."""
    _accelerator_runtime_failures.clear()
    _persist_accelerator_runtime_failures({})


# What sd-cli prints when the GPU BACKEND it was built against cannot serve this card, as opposed to
# the ordinary run-time failures (a bad prompt, a missing file, an out-of-memory) that say nothing
# about the build. Lower-cased substrings, matched against the tail of the output sd-cli exited on,
# which ``SdCppEngine._run`` puts in the RuntimeError it raises.
#
# DECISIVE: the message names the build having no code for this card, which is the defect itself.
# The first two are #9278 verbatim ("CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream"); the
# kernel-image ones are what a generic ROCm build does on a gfx target it carries no code objects
# for. One of these is enough to divert the host.
_ACCELERATOR_DECISIVE_FAILURE_MARKERS: tuple[str, ...] = (
    "hipblassetstream",
    "cublas_status_invalid_value",
    "hiperrornobinaryforgpu",
    "no kernel image is available",
    "invalid device function",
)

# AMBIGUOUS: the message names a GPU fault without establishing that the BUILD is the cause. #9278
# with offload enabled reports "unspecified launch failure" out of ggml_cuda_mul_mat_q, so these do
# belong here, but the same strings come out of a wedged queue, a driver reset, a VRAM exhaustion
# mid-render and a card another process is mistreating. They are counted, not acted on, and it takes
# _AMBIGUOUS_FAILURE_STRIKES of them under ONE fingerprint before a host is moved.
#
# The last two are the same defect class as the decisive list -- ROCm libraries that ship no kernels
# for this card's gfx target -- printed by a layer that does not say so. "rocBLAS error: Could not
# initialize Tensile host: No devices found" is what rocBLAS prints when the installed Tensile
# library carries nothing for the target (ROCm/hipBLASLt#831 for gfx1100; the same line is the
# standing report for gfx803 and for MI300X inside a container), and the HSA runtime's "Memory
# access fault by GPU node-N ... Reason: Page not present or supervisor privilege" is how the same
# situation ends when it reaches a kernel launch. Ambiguous rather than decisive because both also
# have mundane causes: the rocBLAS line appears when /dev/kfd is simply not passed into a container,
# and the HSA fault appears from flash attention, from multi-GPU P2P and from a kernel/firmware
# mismatch on hosts whose ROCm is otherwise fine. Two strikes is the right bar for both.
_ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS: tuple[str, ...] = (
    "unspecified launch failure",
    "hip error",
    "rocm error",
    "rocblas error",
    "memory access fault by gpu node",
)

# The failure that prints NOTHING, and the reason a text-only reading is not enough on Windows.
# Measured on a Strix Halo (gfx1151) Windows 11 runner: the generic Windows ROCm sd.cpp asset
# (`sd-master-...-bin-win-rocm-7.14.0-x64.zip`) ships `stable-diffusion.dll` and no hipBLAS, and a
# box with only the driver's `amdhip64_6.dll` -- no HIP SDK -- cannot resolve `hipblas.dll` or
# `rocblas.dll`. Every invocation, `--list-devices` included, then exits 0xC0000135
# (STATUS_DLL_NOT_FOUND) in 0.02s having written zero bytes to stdout and stderr, so not one marker
# above can fire however the tiers are arranged. The Vulkan asset on the same card ships its 17
# DLLs, loads, and renders.
#
# The runner surfaces that exit status as `sd-cli exited <code>` (sd_cpp_engine's generate), which
# is the only evidence there is. It is DECISIVE rather than ambiguous because an image that never
# starts is a fact about the BUILD on this host and can never be a fact about the request: a
# prompt, a resolution or a busy card cannot make the loader fail to find a DLL. Capacity is still
# checked first, as it is for every other marker.
_WINDOWS_IMAGE_LOAD_FAILURE_STATUSES: tuple[int, ...] = (
    0xC0000135,  # STATUS_DLL_NOT_FOUND: a DLL the image imports is missing
    0xC0000139,  # STATUS_ENTRYPOINT_NOT_FOUND: it is present but the wrong build
    0xC0000142,  # STATUS_DLL_INIT_FAILED: it loaded and its DllMain refused
)

_EXIT_STATUS_RE = re.compile(r"sd-cli exited (-?\d+)")


def output_shows_image_load_failure(text: Optional[str]) -> bool:
    """True when the message reports an exit status meaning the executable never started.

    Windows reports these as unsigned NTSTATUS values; a signed interpretation is accepted too,
    since the same number reaches python differently depending on who read it.
    """
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


# Deliberately in neither list: sd.cpp's "Cannot set backend to CK" warning, which is printed by
# builds that then go on to render perfectly well.
_ACCELERATOR_RUNTIME_FAILURE_MARKERS: tuple[str, ...] = (
    _ACCELERATOR_DECISIVE_FAILURE_MARKERS + _ACCELERATOR_AMBIGUOUS_FAILURE_MARKERS
)

# CAPACITY, and the reason the markers above cannot be read on their own. ROCm reports an
# exhausted card as "ROCm error: out of memory", which contains "rocm error" and so counted as an
# ambiguous strike; two of those under one fingerprint moved the host to Vulkan permanently. That
# is the opposite of the right answer, because an OOM is a statement about the REQUEST and not
# about the build: the same build renders the same model at a smaller size, and the fingerprint
# cannot expire the note because nothing about the host changed.
#
# Checked BEFORE either predicate, decisive included, so an allocation failure whose tail also
# carries a build-shaped string cannot divert the host on one occurrence either. That is the
# conservative direction this whole path is written in: withholding a diversion leaves the user
# with the real error, while a wrong diversion is silent and, once persisted, sticky.
# The forms are the ones this repository's own OOM classifier already recognises, in
# `utils.utils` -- "cannot allocate memory", "memory allocation failed" and
# "cublas_status_alloc_failed" among them -- because "out of memory" is not the only way a
# card says it is full. `ROCm error: CUBLAS_STATUS_ALLOC_FAILED` and `rocBLAS error: memory
# allocation failed` both carry a build-shaped string and neither said anything about
# capacity here, so two large requests were enough to divert a host whose ROCm build works.
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
    """True when sd-cli output names an exhausted card rather than an unusable build."""
    if not text:
        return False
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_CAPACITY_FAILURE_MARKERS)


def output_shows_accelerator_failure(text: Optional[str]) -> bool:
    """True when sd-cli output names a failure of the GPU BUILD rather than of the request.

    Conservative: an unrecognised failure returns False, so a transient error, a bad argument or an
    out-of-memory never persists a preference away from the host's own accelerator."""
    if not text:
        return False
    if output_shows_capacity_failure(text):
        return False
    if output_shows_image_load_failure(text):
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_RUNTIME_FAILURE_MARKERS)


def output_shows_decisive_accelerator_failure(text: Optional[str]) -> bool:
    """True when the output names the BUILD having no code for this card, as opposed to naming a
    GPU fault whose cause it does not establish. Only these divert a host on one occurrence."""
    if not text:
        return False
    if output_shows_capacity_failure(text):
        return False
    if output_shows_image_load_failure(text):
        return True
    lowered = str(text).lower()
    return any(marker in lowered for marker in _ACCELERATOR_DECISIVE_FAILURE_MARKERS)


def preferred_accelerator(accelerator: Optional[str], card: Optional[str] = None) -> str:
    """``accelerator``, or its fallback when this host has already been shown it cannot run it.

    Applied at the TOP of an ensure ladder, so a host that has seen the ROCm build crash does not
    pay for installing and probing it again on every later load. Returns the input unchanged
    whenever there is no note, no rung, or the knob is off."""
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
        return True  # the resident sd-server is executing its own file
    if getattr(backend, "_pending_server", None) is not None:
        return True  # a server is starting from that same file
    if getattr(backend, "_stopping_servers", 0):
        return True  # unpublished, but stop() has not returned: the process is still alive
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
    return _tree_in_use(_sd_cpp_backend)


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
        if not _record_mismatch(mod, root, want) or _record_mismatch(mod, current, want):
            return False
        return mod.installed_ships_server(current) is False
    except Exception:  # noqa: BLE001 -- cannot tell
        return False


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
    # From the root the binary is actually IN, not the current default. The finder also serves a tree an older build
    # left beside the Unsloth home, and reading the current root for a binary out of that one reports "unrecorded" on
    # both sides of the comparison, so a swap underneath this load reads as no change at all.
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
    """Record that the sd.cpp build ``binary`` came from cannot run on this host, when its own
    output says so.

    Two gates, because a render can fail for reasons that have nothing to do with the build: the
    output has to name a GPU-backend failure (``output_shows_accelerator_failure``), and the build
    has to be one with a rung below it to fall back to. Anything else is left alone, so an ordinary
    failure never moves a working host off its own accelerator. Never raises: this is a preference,
    and the caller is on its way to re-raising the real error.

    A third distinction decides whether this ONE failure is allowed to divert the host. A decisive
    message names the build having no code for this card and is acted on immediately; an ambiguous
    one names a GPU fault whose cause it does not establish (a wedged queue, a driver reset, a card
    another process is mistreating all print the same strings) and is only counted. A host has to
    produce several of those under one fingerprint before it is moved, so a single transient error
    on a machine whose ROCm works cannot bypass it.

    Here rather than in the video module it grew up in, because the same sd-cli, with the same
    hipBLAS failure, is run by the image path too: an image-only host was reporting exactly these
    errors and having none of them recorded, so every retry reinstalled and ran ROCm again and the
    Vulkan rung below it was never reached.
    """
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
    except Exception as exc:  # noqa: BLE001 -- a preference, never a reason to mask the real error
        logger.debug("could not record the sd.cpp accelerator failure: %s", exc)


def note_unlaunchable_accelerator_build(
    binary: Optional[str],
    *,
    source: str = "diffusion",
    card: Optional[str] = None,
) -> None:
    """Record that the build ``binary`` came from could not be LAUNCHED on this host.

    The load path's other recorder reads the child's own output, and a build that dies in the
    dynamic loader produces none: on Windows an unsigned loader status like 0xC0000135 (a
    dependent DLL missing, which is what a ROCm build on a host without the HIP runtime looks
    like) is a large POSITIVE exit code, so `_server_binary_runnable` accepts it and the
    failure surfaces later as a start that never comes up. With nothing recorded, every retry
    chose the same build and the rung below it was never reached.

    Always an ambiguous strike, never proven: a missing execute bit, an interrupted extraction
    or a prebuilt for the wrong CPU fail to launch in exactly the same way and say nothing
    about the accelerator. Several of them under one fingerprint move the host; one does not.
    """
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
    except Exception as exc:  # noqa: BLE001 -- a preference, never a reason to mask the error
        logger.debug("could not record the sd.cpp launch failure: %s", exc)


def ensure_sd_cpp_binary(*, allow_install: bool = True, accelerator: str = "cpu") -> Optional[str]:
    """Path to a usable ``sd-cli`` binary, installing the prebuilt once if needed. Returns the
    binary path, or None when it is absent and cannot be installed (install disabled, no network,
    unsupported platform). Never raises -- a None return is the router's signal to fall back to
    diffusers."""
    found = find_sd_cpp_binary()
    usable = bool(found) and _usable_or_discard_managed(found)
    if usable and not _accelerator_changed(found, accelerator):
        return found
    if not allow_install:
        return found
    with _install_lock:
        found = find_sd_cpp_binary()
        usable = bool(found) and _usable_or_discard_managed(found)
        if usable and not _accelerator_changed(found, accelerator):
            return found
        # A usable binary of the wrong accelerator is still better than none, so an install that cannot deliver the
        # right one (no such asset for this host, no network) keeps it.
        fallback = found if usable else None
        try:
            _install = _installer_module().install
        except Exception as exc:  # noqa: BLE001 -- import path / module issues are non-fatal
            logger.warning("sd-cli installer import failed: %s", exc)
            return fallback
        # Claim the tree for the whole install, download included: a point-in-time check would let a generation start
        # during the download and be overwritten by the extraction.
        with _tree_claimed_for_install() as claimed:
            if not claimed:
                return fallback  # something is running in there; retry on a later load
            try:
                path = _install(accelerator = accelerator)
                logger.info("sd-cli installed at %s", path)
                return str(path)
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-cli auto-install failed: %s", exc)
                if _incomplete_tree_replacement(exc):
                    # The sweep got part way, and it takes sd-cli before sd-server, so the fallback resolved before
                    # the install may be one of the copies it already removed. Re-find, so this returns a file that
                    # exists -- often the one the new bundle just extracted -- or None, never a path that is gone.
                    # Through the usability gate, not raw: the raise came BEFORE install()'s _make_executable, so on
                    # POSIX a freshly extracted copy has no execute bit and find_sd_cpp_binary only checks that the
                    # path is a file. The gate probes it and, being ours, removes it if it cannot run, so the next
                    # load reinstalls rather than being handed a binary that fails to launch.
                    refound = find_sd_cpp_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None:
                    _note_failed_upgrade(accelerator)
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
    # Ahead of _accelerator_changed, which reports "unchanged" while the managed tree is in use (an install would
    # overwrite a running binary) and would hand the mismatched legacy server to a load that has the matching
    # serverless build right here. None IS the answer: the one-shot sd-cli of the right build runs, and the bundle is
    # not downloaded again on every later load.
    if usable and _superseded_legacy_server(found, accelerator):
        return None
    if usable and not _accelerator_changed(found, accelerator):
        return found
    if not allow_install:
        return found
    with _install_lock:
        found = find_sd_server_binary()
        usable = bool(found) and _usable_or_discard_managed(found)
        if usable and _superseded_legacy_server(found, accelerator):
            return None
        if usable and not _accelerator_changed(found, accelerator):
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
                _install(accelerator = accelerator)  # extracts sd-cli AND sd-server
            except Exception as exc:  # noqa: BLE001 -- download/extract failure -> fall back
                logger.warning("sd-server auto-install failed: %s", exc)
                # Also when only the CLI survives (a legacy server-less tree): the router probes ensure_sd_cpp_binary
                # immediately after this, and without the record that probe resolves and downloads the same bundle a
                # second time inside one selection.
                if _incomplete_tree_replacement(exc):
                    # As above: the fallback may name a copy the partial sweep removed, and a freshly extracted one
                    # never reached _make_executable.
                    refound = find_sd_server_binary()
                    return refound if refound and _usable_or_discard_managed(refound) else None
                if fallback is not None or find_sd_cpp_binary() is not None:
                    _note_failed_upgrade(accelerator)
                return fallback
        installed = find_sd_server_binary()
        # The finder also probes the tree an older build left beside the Unsloth home, so when the bundle just
        # installed ships no sd-server the hit here can be that legacy server, built for a different accelerator.
        # None, not the fallback: an install just completed, so the router's next step resolves the sd-cli it landed,
        # and a one-shot run on the right build beats a resident server on the wrong one. The fallback stays for the
        # failure path above, where no matching binary was fetched at all.
        if installed and _accelerator_changed(installed, accelerator):
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
    vae_format: Optional[str] = None
    native_speed: str = "off"
    offload_flags: tuple[str, ...] = ()
    threads: Optional[int] = None
    sampling_method: Optional[str] = None
    flow_shift: Optional[float] = None
    server: Optional[SdCppServer] = None
    mode: str = "server"
    # Token kept so LoRA adapters selected at generate time can be fetched from the Hub.
    hf_token: Optional[str] = None
    # The GGUF basename this load committed: some variants pick their encoder by filename, and a local *klein-9B*.gguf
    # carries that keyword only in the basename.
    gguf_filename: Optional[str] = None
    # The FLUX.2 inner_dim this load read out of the checkpoint's own header, when it could. Kept so the delete guard
    # reconstructs the SAME encoder pick without re-probing under the lock.
    flux2_inner_dim: Optional[int] = None
    # The managed tree's recorded accelerator when this load chose its binary. The one-shot path re-resolves sd-cli
    # per image, so without this it would silently adopt an install that landed between images even when that install
    # is for a DIFFERENT accelerator, while ``device`` and ``offload_flags`` still describe the build the load
    # committed to. None on the server path, which asks the same question at start time against its own local copy.
    sd_accelerator: Optional[str] = None
    # Physical card behind the server's single resolved CUDA/ROCm backend. None for CPU/Metal/Vulkan, an unresolved
    # pin, automatic multi-GPU, and one-shot mode; those shapes cannot safely consume the startup VRAM floor.
    physical_gpu_id: Optional[int] = None
    # The card this load selected, as the failure record names cards. A generation that fails mid
    # render is a fact about THAT card, and the generate path is far from the ordinal that chose
    # it, so the answer is resolved once here and carried.
    selected_card: Optional[str] = None


def _offload_with_device_pin_impl(
    offload: tuple[str, ...] | list[str], binary: Optional[str], ordinal: Optional[int]
) -> list[str]:
    """``offload`` plus the ``--backend`` pin for whichever build is about to run it."""
    flags = list(offload)
    if ordinal is None:
        return flags
    device_name = sd_cpp_device_name_for_ordinal(binary, ordinal)
    if device_name is None:
        # The fallback build's devices are in their own namespace. `Vulkan0` is not the
        # physical index the user picked, so the lookup above answers None, no `--backend` is
        # written, and sd.cpp takes its own default device -- normally the first card -- while
        # this load's reservation and accounting name the card that WAS selected. On a
        # multi-GPU host that is a reservation against hardware nothing is running on, and an
        # overcommit of whatever Vulkan chose. The card's own name is the one thing the two
        # namespaces agree on, which is what the video path already pins by.
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
        # An explicit request is attributable only when the committed argv actually contains the resolved per-module
        # pin. If the probe failed, sd.cpp falls back to its own default and the requested ordinal is not evidence of
        # placement.
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


def _map_guidance(
    fam: DiffusionFamily, guidance: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(cfg_scale, guidance) for sd-cli from the single diffusers ``guidance`` value. FLUX families
    take a distilled embedded ``--guidance``; everyone else uses real classifier-free
    ``--cfg-scale``. A distilled 0/1 means CFG off (sd-cli's 1.0); a value > 1 is real CFG."""
    if fam.name in ("flux.1", "flux.2-klein", "flux.2-dev"):
        return None, (float(guidance) if guidance is not None else None)
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


class SdCppDiffusionBackend:
    """Native sd.cpp backend with the diffusers ``DiffusionBackend`` method surface."""

    def __init__(self, engine: Optional[SdCppEngine] = None) -> None:
        self._lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._engine = engine  # resolved lazily on first load so import stays cheap
        # An injected engine (test seam) pins one-shot mode; a fallback-cached engine must NOT, so a now-available
        # server can still be used next load.
        self._engine_injected = engine is not None
        self._state: Optional[_SdState] = None
        self._loading: Optional[_SdLoading] = None
        self._load_token = 0
        # Replaced (never cleared) per load, so a cancelled asset pull stays cancelled.
        self._cancel_event = threading.Event()
        self._active_generate_cancel: Optional[threading.Event] = None
        self._active_generate_account: Optional[str] = None
        # sd-server started for an in-flight load, before it commits to _state; tracked so an unload can stop it
        # mid-startup.
        self._pending_server: Optional[SdCppServer] = None
        self._gen: Optional[_SdGen] = None
        # Set by _resolve_backend when it had to skip an accelerator install because the managed tree was still in
        # use; the load retries it once the tree is free.
        self._deferred_accelerator_install = False
        # The card THIS load selected, resolved once from its ordinal. Every accelerator
        # resolution inside the load asks about it, so a failure recorded against another card
        # on a heterogeneous host does not divert this one: the router keeps ROCm for the
        # supported card and the backend would otherwise switch the install back to Vulkan.
        # None outside a load and whenever the selection cannot be identified, which is the
        # answer that leaves every record applying, as before.
        self._loading_card: Optional[str] = None
        # Servers taken out of _state/_pending_server whose stop() has not returned yet. unload() deliberately stops
        # outside the lock (terminate can take seconds), so between the clear and the stop the fields say idle while
        # the process is still running its own executable.
        self._stopping_servers = 0
        # Set once this load's graph proved unrunnable on the GPU backend (a ggml unsupported-op abort), so the CPU
        # restart happens once per load. Cleared by each load.
        self._cpu_backend_forced = False

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

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

        Through ``preferred_accelerator``, so a host already shown it cannot run the build for its
        own accelerator is not handed it again. HERE rather than at the four call sites below
        (``_resolve_engine``, ``_resolve_backend``, ``_upgrade_server_after_teardown`` and the
        mid-load re-resolve), because those four have to agree: ``_accelerator_changed`` in the
        deferred upgrade compares the
        installed tree against this answer, so a call site that skipped the preference would read
        the Vulkan tree the other two installed as the wrong accelerator and reinstall ROCm over it
        on every load."""
        from core.inference.diffusion_engine_router import _install_accelerator_for
        return preferred_accelerator(
            _install_accelerator_for(getattr(resolve_diffusion_device_target(), "backend", "cpu")),
            card,
        )

    def _resolve_engine(self) -> SdCppEngine:
        """The SdCppEngine, installing the binary on first use. Raises if unusable."""
        if self._engine is not None and self._engine.is_available():
            return self._engine
        # The accelerator this host resolves to, never the "cpu" default: this is also the one-shot FALLBACK path (a
        # GPU sd-server that would not start lands here), and asking for the CPU build there would reinstall the plain
        # bundle over the working GPU one and run the whole generation on the CPU.
        binary = ensure_sd_cpp_binary(
            allow_install = _install_allowed() and not _tree_in_use(self),
            accelerator = self._resolved_accelerator(self._loading_card),
        )
        if not binary:
            raise RuntimeError("sd-cli (stable-diffusion.cpp) binary is unavailable.")
        self._engine = SdCppEngine(binary = binary)
        return self._engine

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
        # An accelerator upgrade REPLACES the binaries in the managed tree, and this runs before the load stops the
        # resident server or waits out an in-flight one-shot sd-cli. Linux refuses to open a running executable for
        # writing (ETXTBSY) and Windows locks it, so an install here fails and can leave the tree half-written.
        # _accelerator_changed refuses the upgrade while the tree is in use, whoever asks; record that it did, so this
        # load retries it after its own teardown. _managed_tree_in_use covers the singleton for callers that never see
        # this instance (the engine router); this load's own state is the authority for this load.
        upgrade_pending = _tree_in_use(self) or _managed_tree_in_use()
        self._deferred_accelerator_install = upgrade_pending
        server_binary = ensure_sd_server_binary(
            allow_install = _install_allowed() and not upgrade_pending, accelerator = accelerator
        )
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
            # Judge the tree by whatever binary it holds: on a serverless install that is the sd-cli, and without this
            # the retry would reinstall on every deferred load, matching accelerator or not.
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
        """Land the deferred install, then answer the question the router could not.

        The router keeps a native selection whose only defence was an install that had not run
        yet: a resident server executing out of the managed tree cannot be replaced while it
        runs, so both ensures hand back the build the record condemns and refusing it there
        would send the first reload after a mid-render failure to diffusers for nothing.

        It has run now. ``_upgrade_server_after_teardown`` is deliberately never fatal and
        returns the path it was given when the install could not deliver -- an offline host, a
        failed download, an archive that carries no server -- so the build about to be started
        can still be the condemned one. That is the point where the answer is real, and the
        same record check the router skipped decides it: this load fails instead of committing
        a build that dies mid-render, minutes and a full download later, and by then the server
        is stopped, so the next selection finds the tree free and routes to diffusers.

        A build that IS the requested accelerator is not refused, here or in the router: with
        the Vulkan fallback switched off the request is ROCm again on purpose, and that opt-out
        means run it anyway.
        """
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
        # Same name, position and default as DiffusionBackend.begin_load: the route calls whichever engine was
        # activated through ONE call site and passes this unconditionally, so an engine that does not declare it
        # TypeErrors every load on the hosts that select it. Covers the MODEL ASSETS only: the GGUF, the VAE and the
        # text encoders this pick fetches from the Hub. It deliberately says nothing about the sd-cli/sd-server
        # BINARY, which is a separate managed tree with its own install policy; a background load may still install
        # one, exactly as it does today.
        local_files_only: bool = False,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        family_override: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        # diffusers-only knobs accepted for a uniform call and ignored (sd.cpp has no torchao quant / SDPA dispatcher
        # / fbcache)
        text_encoder_quant: Optional[str] = None,
        transformer_quant: Optional[str] = None,
        transformer_quant_fast_accum: Optional[bool] = None,
        transformer_prequant_path: Optional[str] = None,
        attention_backend: Optional[str] = None,
        transformer_cache: Optional[str] = None,
        transformer_cache_threshold: Optional[float] = None,
        # Accepted for interface parity; native is GGUF-only (router forces diffusers otherwise).
        model_kind: Optional[str] = None,
        # Parity with the diffusers load-time LoRA bake; native applies LoRA per generation, so a load-time selection
        # is ignored.
        loras: Optional[list[tuple[str, float]]] = None,
        gpu_ids: Optional[list[int]] = None,
        # The ordinal the ROUTE already ranked, so the preflight and the load agree on one card
        gpu_ordinal: Optional[int] = None,
    ) -> dict[str, Any]:
        """Validate, then fetch assets on a daemon thread. Returns at once."""
        # Empty/whitespace token = "no token"; "" verbatim breaks the anonymous fallback.
        hf_token = hf_token.strip() if hf_token and hf_token.strip() else None
        # Same fallback the diffusers and video backends take: the route ranks the selection and passes the winner,
        # but a direct caller (an MCP client, a test, a plugin) hands over gpu_ids alone, and without this the native
        # engine is the one engine that would drop the pick silently. Re-ranked only when nobody has, so a
        # route-resolved winner is never second-guessed against free VRAM that has moved since.
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
        # Filename-fallback detector (as the route validated) so a local .gguf whose family keyword lives only in the
        # basename still loads
        fam = detect_family_for_pick(repo_id, gguf_filename, family_override)
        if fam is None:
            raise ValueError(
                f"'{repo_id}' is not a supported diffusion image model. Supported families: "
                f"{', '.join(supported_family_names())}. If this is a variant of one of them, "
                f"pass family_override with that family name."
            )
        if not family_sd_cpp_supported(fam):
            raise ValueError(f"Family '{fam.name}' has no native sd.cpp asset mapping.")

        base = resolve_base_repo(fam, base_repo)
        # Offline-only here, deliberately: begin_load returns at once by contract
        inner_dim = self._flux2_inner_dim(
            repo_id, gguf_filename, fam, hf_token, allow_network = False
        )
        # Same link the diffusers resolver records, so the delete guard protects a native pick's companions too -- and
        # here that means the repos _asset_specs actually FETCHES. The native engine does not read the diffusers base:
        # FLUX.2 takes its VAE from unsloth/FLUX.2-VAE and its encoders from another repo again, so recording only the
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
            # A superseding load must stop any in-flight generation, else the old run can still persist an image after
            # the new load starts
            if self._active_generate_cancel is not None:
                self._active_generate_cancel.set()
            self._load_token += 1
            token = self._load_token
            # A NEW event per load, never a clear() of the shared one: unload() sets the event the running worker
            # holds but also drops _loading, so a clear() here would un-cancel its still-running multi-gigabyte pull.
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
            )

        account_thread(
            target = self._run_load,
            kwargs = dict(
                repo_id = repo_id,
                local_files_only = local_files_only,
                gguf_filename = gguf_filename,
                base = base,
                fam = fam,
                hf_token = hf_token,
                cpu_offload = cpu_offload,
                memory_mode = memory_mode,
                speed_mode = speed_mode,
                gpu_ordinal = gpu_ordinal,
                _load_token = token,
                _cancel_event = cancel_event,
            ),
            daemon = True,
        ).start()
        return self.status()

    def _run_load(
        self,
        *,
        repo_id: str,
        gguf_filename: str,
        base: str,
        fam: DiffusionFamily,
        hf_token: Optional[str],
        # Cache-only when set: every Hub call below is either skipped or told to resolve from disk, so a load nobody
        # asked for cannot pull bytes. See begin_load for what it does not cover.
        local_files_only: bool = False,
        cpu_offload: bool = False,
        memory_mode: Optional[str] = None,
        speed_mode: Optional[str] = None,
        gpu_ordinal: Optional[int] = None,
        _load_token: int,
        _cancel_event: Optional[threading.Event] = None,
    ) -> None:
        # This load's own event: a later load replaces self._cancel_event rather than clearing it.
        cancel_event = _cancel_event if _cancel_event is not None else self._cancel_event
        # The server this load publishes to _pending_server, out here so the backstop below can always unpublish it. A
        # leaked _pending_server reads as "the managed tree is busy" for the rest of the process and blocks every
        # later install.
        started: Optional[SdCppServer] = None
        # Resolved once, before anything asks: the translation reads the host's enumeration and
        # every resolution in this load has to give the same answer.
        self._loading_card = selected_card_identity(gpu_ordinal)
        try:
            # Resolve mode (server preferred, one-shot fallback) + binary up front so an install failure surfaces
            # before the multi-GB pull.
            mode, server_binary, engine = self._resolve_backend()
            if mode == "server":
                # Probe the server binary before the pull: a present-but-unrunnable build would download everything
                # then fail
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
                        # Neither this accelerator's server nor its one-shot CLI will launch.
                        # There is no output to classify -- a build that dies in the loader
                        # prints nothing -- so it is counted rather than acted on.
                        note_unlaunchable_accelerator_build(server_binary, card = self._loading_card)
                        raise RuntimeError("sd-server binary is present but not runnable.")
                    mode, server_binary, engine = "oneshot", None, fallback
            # The accelerator the managed tree held when THIS binary was chosen, taken where the choice is made rather
            # than sampled again later. The asset download below runs for minutes with no claim on the tree, and an
            # install that lands in that window replaces sd-server in place: same path, still runnable, a different
            # build. "It exists and it runs" is therefore not evidence that it is the build this load resolved its
            # device and offload policy for, so the answer is re-asked under the reader claim.
            server_accelerator = _installed_accelerator_of(server_binary)
            # The same pin for the one-shot CLI, and for the same reason. Sampling it only at state construction,
            # after the download, would record whatever an install left in the tree in that window, and the first
            # generation -- which re-reads the tree and compares -- would then agree with the replacement, so the
            # check that exists to notice a swap could never fire for one that landed during the download.
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

            # Swap ONCE so the size probe and the download agree: sizes come from paths-info, which -- unlike
            # model_info -- 401s anonymously on a gated repo, so probing the upstream drops the VAE from the progress
            # total the mirror then pulls. The probe is a RANGE READ off the Hub when the checkpoint is not on disk,
            # so an offline load asks it the way begin_load does: memo or local header or nothing. A None here only
            # falls back to the filename heuristic for the encoder pick, and a cache-only load can fetch nothing the
            # heuristic did not already have. The speech verdict lands here rather than in begin_load, which is
            # offline-only by contract; before _asset_specs, so the refusal precedes any fetch.
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
            # And record them, from the SAME post-probe specs. begin_load records what it can, but it resolves the
            # header offline, so a remote or renamed FLUX.2-klein 9B checkpoint with no cached probe records the
            # default 4B encoder while this load fetches the 9B one. Whatever that leaves unrecorded is a companion
            # the delete guard would let go while its GGUF is still installed. Recorded on the FETCH ids too, since a
            # gated mirror or a cached community repack is where the bytes land.
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
            # Same preflight the plan runs, on POST-swap repos: catch a gated companion here, not 15 GiB into the
            # prefetch, without refusing one an ungated mirror stands in for. The plan alone is not enough: the images
            # page falls back to this load when it fails.
            self._preflight_companion_repos(
                self._assets_by_repo(assets),
                fetch_repo.get(repo_id, repo_id),
                hf_token,
                local_files_only = local_files_only,
            )
            # Skipped outright offline: the size probe is get_paths_info, a Hub round trip, and its only product is
            # the progress bar's denominator. A cache-only load resolves every asset from disk in milliseconds, so 0
            # (the value this method already reports for any size the Hub will not answer) costs nothing and asking
            # would be the one network call left on the path.
            if not local_files_only:
                self._set_expected_bytes(assets, hf_token)
            paths = self._fetch_assets(
                assets,
                hf_token,
                cancel_event = cancel_event,
                local_files_only = local_files_only,
            )

            files = SdCppModelFiles(
                diffusion_model = paths["diffusion_model"],
                vae = paths.get("vae"),
                clip_l = paths.get("clip_l"),
                clip_g = paths.get("clip_g"),
                t5xxl = paths.get("t5xxl"),
                llm = paths.get("llm"),
                qwen2vl = paths.get("qwen2vl"),
            )
            device = resolve_diffusion_device_target().device
            # Honor speed everywhere; offload only off-CPU (on CPU weights are resident, so the flags are no-ops)
            offload: tuple[str, ...] = ()
            if device != "cpu":
                offload = tuple(offload_flags(_memory_policy(memory_mode, cpu_offload)))
            # The device pin is NOT folded in here: the binary can still change below, through the deferred
            # accelerator install, the post-download re-resolve, or a server start that falls back to one-shot, and
            # the ggml device names come from whichever build ends up running. It is added at each point the flags are
            # handed to a binary instead.
            gpu_ordinal = gpu_ordinal if device == "cuda" else None
            native_speed = _native_speed_for(speed_mode)

            # Tear down the old model then commit the new one under _generate_lock: abort and WAIT for a generation
            # started during the download, so no stale run persists an image. Taken only now, so the download never
            # serialises generation.
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
                # The tree is free now, so an install deferred in _resolve_backend can land: this runs under both
                # locks, the old server is stopped, the previous generation has finished and no new one can start. Not
                # gated on the resolved mode: a serverless install resolves to one-shot precisely BECAUSE the deferral
                # suppressed the install, and its sd-cli comes out of the same archive.
                if self._deferred_accelerator_install:
                    self._deferred_accelerator_install = False
                    upgraded = self._upgraded_or_refused(server_binary, mode = mode, engine = engine)
                    if mode == "server":
                        server_binary = upgraded
                    # This load's own install just rewrote the tree, under the install claim, so what it left behind
                    # IS the decision here -- comparing against the answer from before it would fail the load on the
                    # upgrade it asked for. Both pins move: a serverless upgrade lands the sd-cli out of the same
                    # archive.
                    server_accelerator = _installed_accelerator_of(server_binary)
                    engine_accelerator = _installed_accelerator_of(getattr(engine, "binary", None))
                # A new checkpoint earns a fresh attempt on the GPU backend: the previous abort says nothing about
                # this graph.
                self._cpu_backend_forced = False
                server: Optional[SdCppServer] = None
                if mode == "server":
                    assert server_binary is not None
                    # _fetch_assets above runs for minutes with no claim on the tree (there is nothing executing in it
                    # yet to claim for), so an install can have swept this path between layouts since _resolve_backend
                    # picked it. Re-resolve before starting: the stale path would drop the load to one-shot for
                    # nothing, or start a build this load did not select. Under the READER, and held until
                    # _pending_server is published. Re-resolving alone does not close the race: allow_install=False
                    # only declines to install, it does not wait for or claim the tree, so an installer that has
                    # already passed its in-use check can sweep this executable between the re-read and the start.
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
                            # And pin off THIS engine. The one-shot pin above was taken while the mode was still
                            # "server", i.e. off an engine of None, so leaving it would compare the sd-cli just
                            # resolved against None and refuse the documented fallback on every load that reaches it.
                            # Resolved here, inside the claim, so it is vetted at the moment it is pinned.
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
                            server = SdCppServer(server_binary)
                            # Published INSIDE the claim: _tree_in_use reads _pending_server, so this is the handover
                            # from "a reader holds the tree" to "a starting server does", with no gap between them.
                            # Cancellation is re-read in the SAME block: the revalidation above can sit for 20s in
                            # _server_binary_runnable, and an unload arriving in that window finds no _pending_server
                            # to stop, so without this the load would spawn the process anyway and hold the device for
                            # the whole start() timeout. Asked under the lock that publishes, so an unload either
                            # stops this server or is seen here.
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
                    # ``started`` (set with _pending_server above) is the object to clear below. ``server`` itself is
                    # set to None when start() fails and the load falls back to one-shot, and comparing THAT against
                    # _pending_server left the stopped server published forever, which reads as "the managed tree is
                    # busy" for the rest of the process. A server that DID start stays published until _state takes it
                    # over, under the same lock: clearing it here would leave a window in which the tree reads as idle
                    # while the process is up and running out of it, and an ensure_* landing in that window admits an
                    # install that later overwrites the executable underneath the live server.
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
                        # The load half of the same recording the generate paths do. A server
                        # that starts and dies carries the build's own output in this error --
                        # "sd-server exited N. Last output: ..." -- and nothing else on this
                        # path was reading it, so a ROCm build that cannot come up at all left
                        # no note and every retry chose it again.
                        note_accelerator_failure_from_output(
                            server_binary,
                            str(start_exc),
                            card = self._loading_card,
                        )
                        server.stop()
                        # Unpublish BEFORE resolving the one-shot engine: _pending_server means "a process is running
                        # out of the tree", and leaving this stopped one there would block the very sd-cli install
                        # this fallback needs.
                        with self._lock:
                            if self._pending_server is server:
                                self._pending_server = None
                        server = None
                        # KEEP the engine this fallback resolved. Discarding it left the local `engine` at the server
                        # path's None, so state.sd_accelerator was recorded as None and the first one-shot generation,
                        # which re-resolves sd-cli and reads its real accelerator, rejected it as a
                        # different-accelerator replacement: the load reports success and then cannot generate.
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
                    # Runnable, at the same path, and still not the build this load vetted -- the one-shot half of the
                    # check the server path makes just above. Refused at load rather than recorded, because recording
                    # the replacement is what makes the per-generation comparison agree with it forever after.
                    raise RuntimeError(
                        "The stable-diffusion.cpp binary was replaced by an install for a "
                        "different accelerator while this model was loading. Try the load again."
                    )
                committed_offload_flags = tuple(
                    _offload_with_device_pin_impl(
                        offload,
                        server_binary if mode == "server" else getattr(engine, "binary", None),
                        gpu_ordinal,
                    )
                )
                state = _SdState(
                    repo_id = repo_id,
                    base_repo = base,
                    family = fam,
                    device = device,
                    files = files,
                    vae_format = fam.sd_cpp_vae_format,
                    native_speed = native_speed,
                    # Pinned against the binary this load COMMITTED to, which a deferred install or a one-shot
                    # fallback may have changed since the policy was built.
                    offload_flags = committed_offload_flags,
                    threads = _default_threads(),
                    sampling_method = fam.sd_cpp_sampling_method,
                    flow_shift = fam.sd_cpp_flow_shift,
                    server = server,
                    mode = mode,
                    hf_token = hf_token,
                    gguf_filename = gguf_filename,
                    flux2_inner_dim = inner_dim,
                    # Only the one-shot path needs to carry it: it re-resolves sd-cli per image, long after this
                    # decision, and has nothing else to check the answer against.
                    sd_accelerator = engine_accelerator if mode == "oneshot" else None,
                    physical_gpu_id = (
                        _resolved_server_physical_gpu_id(
                            server_binary,
                            device,
                            gpu_ordinal,
                            committed_offload_flags,
                        )
                        if mode == "server"
                        else None
                    ),
                    selected_card = self._loading_card,
                )
                superseded = False
                orphan: Optional[SdCppServer] = None
                with self._lock:
                    if self._load_token != _load_token:
                        # Superseded / unloaded while loading: discard the started server so it does not leak.
                        # Reserved in the SAME block that unpublishes it, so the tree never reads as idle while the
                        # process is still coming down.
                        superseded = True
                        if server is not None:
                            self._reserve_stop()
                            orphan = server
                    else:
                        self._state = state
                        self._loading = None
                    # The exchange the started server stayed published for: it is _state's now, or reserved for the
                    # stop below, and either way _tree_in_use still sees it.
                    if self._pending_server is started:
                        self._pending_server = None
                if orphan is not None:
                    self._stop_reserved(orphan)
                if superseded:
                    return
        except SdCppCancelled:
            return
        except Exception as exc:  # noqa: BLE001 -- surfaced via load_progress
            if self._load_token != _load_token:
                return
            logger.error("sd_cpp.load_failed: %s", exc)
            if self._state is not None:
                # Displaced pipeline: give its account back GPU residency, drop its records.
                from .gpu_arbiter import DIFFUSION, restore_owner_account
                from hub.services.models.account_access import restore_resident_metadata

                restore_owner_account(DIFFUSION)
                restore_resident_metadata("diffusion")
            # Redact filesystem paths before this reaches /images/load-progress (as diffusers does).
            from utils.native_path_leases import redact_native_paths

            with self._lock:
                if self._load_token == _load_token and self._loading is not None:
                    self._loading.error = redact_native_paths(str(exc))
        finally:
            # Backstop for the window the started server is deliberately left published across (start() -> _state).
            # Every path through that window unpublishes it itself; this only catches an unexpected raise in between,
            # which would otherwise wedge the tree as busy.
            if started is not None:
                with self._lock:
                    if self._pending_server is started:
                        self._pending_server = None

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

        # STAGED before the load runs, so each entry must name the repo _fetch_assets will pull from: some asset repos
        # are gated (the FLUX.1 VAE lives in black-forest-labs/FLUX.1-schnell) and an anonymous user would 401 at
        # staging, never reaching the swap. Same per-repo file list on both sides, so both take the same decision.
        fetch_repo = _fetch_repo_map(specs, hf_token)
        # MERGED, not reassigned: two upstream repos can share one fetch repo (the FLUX.2 VAE and the dev encoders
        # both come from Comfy-Org/flux2-dev once that repack is cached), and a plain comprehension would drop
        # whichever landed first, leaving its files out of both the staged entry and the footprint.
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
        # Imported here, not at module scope: diffusion.py is the heavier module and the routes already load this one
        # on its own
        from core.inference.diffusion import DiffusionBackend

        for repo, names in by_repo.items():
            total += int(sum(sizes.get((repo, n), 0) for n in names))
            # Same missing-file filter the diffusers planner applies: _fetch_assets already reads both cache roots, so
            # staging an asset it can resolve re-downloads it for nothing and fails offline. required_bytes keeps the
            # UNFILTERED sum -- it is the disk footprint. Sized, so a republished asset under the same name is a miss
            # rather than a silent inline fetch during the load. Loadable, not merely cached: a stale live-root copy
            # shadows a good one in the other root, because the fetch only switches roots when the live lookup finds
            # nothing.
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
                    # Only the transformer entry carries the GGUF filename; the VAE / encoder entries are plain single
                    # files.
                    "gguf_filename": gguf_filename if repo == fetch_repo_id else None,
                    # Said plainly for the panel's label: the transformer IS the pick, the VAE / encoders are required
                    # assets. Compared against the POST-swap id, because a gated pick staged from its ungated mirror
                    # no longer matches the id the caller asked for. Native picks are always single-file, so there is
                    # no pipeline case.
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
            # Companions only: the picker only lists repos it could already read.
            if repo != repo_id and names:
                # Probe an asset THIS pick stages: a VAE-only repo has no pipeline manifest, so the default name would
                # neither verify access nor see the cache.
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
    ) -> list[tuple[str, str, str]]:
        """(repo, filename, kind) for every file sd-cli needs. ``kind`` is the SdCppModelFiles
        field; the transformer reuses the diffusers GGUF."""
        specs: list[tuple[str, str, str]] = [(repo_id, gguf_filename, "diffusion_model")]
        if fam.sd_cpp_vae:
            specs.append((fam.sd_cpp_vae[0], fam.sd_cpp_vae[1], "vae"))
        # Pick the encoder per variant so a 9B GGUF fetches the right one: from the header when the caller read it,
        # else from the load identity, which a renamed file makes silent.
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
                # Only the transformer can be a local path; others are always HF ids.
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
    ) -> dict[str, str]:
        """Download every asset (cancellable via this load's own ``cancel_event``, so a replacement
        load cannot un-cancel this pull), returning kind -> local path. ``local_files_only``
        resolves each asset from the HF cache and never from the network; an asset that is not
        there fails HERE, with the repo and filename named, rather than being quietly pulled.
        This is the last and only network call left on an offline load's path, so it is the one
        that has to honour the flag rather than merely accept it."""
        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # Callers without a per-load event (tests, direct use) fall back to the current one.
        cancel = cancel_event if cancel_event is not None else self._cancel_event
        paths: dict[str, str] = {}
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
                # Resolve an asset cached only under huggingface_hub's import-time root through that root, as the
                # preflight does. Pinned to the live root, a cache-folder change re-downloads every moved asset and
                # 401s on an already-downloaded gated base.
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
                    # Raised by huggingface_hub for exactly "not cached and outgoing traffic is disabled", so it can
                    # only fire under local_files_only. Its own text names neither the repo nor the file, and this
                    # string is what /images/load-progress toasts, so restate it with both. Re-raised untouched in the
                    # (unreachable) online case rather than relabelled, so nothing changes when the flag is off.
                    if not local_files_only:
                        raise
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

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        guidance: float = 0.0,
        seed: Optional[int] = None,
        batch_size: int = 1,
        # Batched prompt/seed lists are diffusers-engine features; accepted for parity and rejected below, since
        # sd-cli would render them serially.
        prompts: Optional[list[str]] = None,
        seeds: Optional[list[int]] = None,
        # Accepted for interface parity; native is text-to-image only, so image-conditioned requests are rejected below.
        init_image: Optional[str] = None,
        mask_image: Optional[str] = None,
        strength: Optional[float] = None,
        upscale: Optional[float] = None,  # needs an init image; rejected by the guard below
        reference_images: Optional[list[str]] = None,  # GPU/diffusers-only (FLUX.2)
        # LoRA (id, weight) pairs; resolved up front then applied per path: prompt tags for one-shot sd-cli,
        # structured `lora` for sd-server.
        loras: Optional[list[tuple[str, float]]] = None,
        # ControlNet is diffusers-only; rejected by the guard below (accepted for parity).
        controlnet: Optional[tuple[str, str, str, float, float, float]] = None,
        # load_identity() of the caller's status() read; refuse rather than run a different load (#9448)
        expected_load: Optional[LoadIdentity] = None,
    ) -> dict[str, Any]:
        import tempfile

        from PIL import Image

        from core.inference import diffusion_lora

        if (
            init_image is not None
            or mask_image is not None
            or reference_images
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
                # A resident server can exit while idle; drop stale state and report not-loaded so the client gets the
                # reload path
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
                # Publish an active (step 0) state before the slow pre-generate setup so a reload probe does not read
                # idle while this holds _generate_lock.
                self._gen = _SdGen(total_steps = int(steps))
            try:
                if seed is None:
                    seed = int.from_bytes(os.urandom(6), "big") & ((1 << 53) - 1)
                else:
                    seed = int(seed)
                cfg_scale, flux_guidance = _map_guidance(state.family, guidance)
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
                        )
                except RuntimeError as exc:
                    # The same failure the video path records, from the same sd-cli: the ROCm
                    # build starts, loads the tensors and dies in hipBLAS mid-render. Nothing
                    # about the install is wrong, so without this the next load picks the same
                    # build again -- and on an image-only host nothing else was recording it,
                    # so the Vulkan rung below it was never reached however many times it
                    # failed. A cancellation is not a failure of the build.
                    if not cancel.is_set() and DIFFUSION_CANCELLED_MSG not in str(exc):
                        # The binary that RAN it. On the server path the model is loaded
                        # inside the resident sd-server and `_resolve_backend` returns no
                        # engine at all, so reading `self._engine` there passes None and the
                        # recorder returns immediately -- which is the whole case this is for,
                        # since a ROCm server that starts and then dies in hipBLAS is exactly
                        # what the fallback rung exists for. The engine as it stands for the
                        # one-shot path, never a fresh resolve: resolving here can itself raise
                        # ("binary is unavailable"), and an exception from inside an except
                        # block replaces the failure the caller is waiting for.
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
                # Check and deregister under _lock, the lock cancel_generate takes, so the two cannot interleave: a
                # cancel that saw this event registered ran strictly before the check and the run unwinds as
                # cancelled, and one arriving after finds nothing to set and answers false. Same critical section as
                # DiffusionBackend.generate; /images/generate/cancel resolves through the engine router, so a native
                # host has to give the same answer. The finally repeats the clear for every other exit.
                with self._lock:
                    if cancel.is_set():
                        raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                    if self._active_generate_cancel is cancel:
                        self._active_generate_cancel = None
                        self._active_generate_account = None
                return {
                    "images": images,
                    "seed": int(seed),
                    "seeds": seeds,
                    "repo_id": state.repo_id,
                    # The BUILD, for the recipe: the repo id alone does not say WHICH GGUF quant ran, and two quants
                    # make different pixels.
                    "model_kind": "gguf",
                    "gguf_filename": state.gguf_filename,
                    # The rest of the build the recipe records. The native engine has no dense quant path and no
                    # memory-mode planner, so those two are honestly null -- but the offload it ran under is real
                    # (sd-cli flags) and status() already derives it the same way. Omitting them here persisted null
                    # for every native image and left the recipe unable to say how the picture was produced.
                    "transformer_quant": None,
                    "text_encoder_quant": None,
                    "memory_mode": None,
                    # The POLICY flags only: a --backend pin says which card ran the graph, not that anything was
                    # offloaded, and a `fast` load carries no policy flags at all.
                    "offload_policy": (
                        "active" if without_device_backend_flags(state.offload_flags) else "none"
                    ),
                }
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
        # Stage LoRAs into a per-request subdir of the server lora-model-dir (so a prior request's adapters cannot
        # leak in); removed after.
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
        # One deadline for the whole request, shared by its chunks: a batch is chunked only because the server caps
        # images per job, so each chunk gets whatever is left rather than its own full budget and a long batch still
        # ends on time.
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
                # All-or-nothing per chunk: fail rather than silently drop images from the batch.
                if not cancel.is_set() and len(blobs) != count:
                    raise RuntimeError(
                        f"sd-server returned {len(blobs)} of {count} requested images in the batch."
                    )
                images.extend(Image.open(io.BytesIO(b)).convert("RGB") for b in blobs)
                # sd.cpp advances the seed per image within a job, so report chunk_seed+i.
                seeds.extend((chunk_seed + i) & ((1 << 63) - 1) for i in range(len(blobs)))
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
                # WITHOUT the device pin. sd.cpp joins repeated --backend values into one spec instead of replacing,
                # and an explicit diffusion=CUDA0 outranks the bare `cpu` default, so leaving the pin on would restart
                # the server onto the very backend that just aborted.
                offload = without_device_backend_flags(state.offload_flags),
                native_speed = state.native_speed,
                threads = state.threads,
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
            # Materialize LoRAs into a scan dir and inject <lora:ALIAS:w> tags (deduped).
            eff_prompt = prompt
            lora_dir: Optional[str] = None
            if lora_resolved:
                materialized = diffusion_lora.materialize_native_dir(
                    lora_resolved, Path(tmpdir) / "loras"
                )
                eff_prompt = diffusion_lora.inject_prompt_tags(prompt, materialized)
                lora_dir = str(Path(tmpdir) / "loras")
            for index in range(max(1, int(batch_size))):
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                # Distinct reproducible seed per image; mask to int64 (53 bits would truncate large explicit seeds and
                # collide them)
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
                )
                # Each sd-cli run executes out of the managed tree, so hold installs off for its duration (and wait
                # here if one is already extracting). getattr: an INJECTED engine is the unit-test seam / escape hatch
                # and need not name a file at all, and nothing without a path is a binary an install replaces.
                with _tree_reader(getattr(engine, "binary", None), cancel):
                    # Re-resolve INSIDE the claim. An install that finished while this image was waiting can have put
                    # its sd-cli somewhere else and swept the copy resolved above, so the cached path would launch a
                    # file that is no longer there. Also covers a batch, which releases the claim between images.
                    # Cheap when nothing moved.
                    engine = self._resolve_engine()
                    # Existence is not identity here either. The install that moved the CLI may have been for a
                    # different accelerator (an H3 load putting the CPU fallback in, say), and this state's device and
                    # offload policy were chosen for the other one, so running it would either spend unaccounted VRAM
                    # or drop the whole generation onto the CPU while the arbiter's accounting says otherwise. The
                    # server path refuses exactly this mismatch before it starts; refusing here costs a reload, which
                    # re-resolves device, accelerator and install together.
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
                        on_log = self._on_log,
                        cancel_event = cancel,
                    )
                with Image.open(out_path) as im:
                    images.append(im.copy())
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

    # ── Unload / status ──────────────────────────────────────────────────────

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
            # Grab a mid-start() uncommitted server too so we can stop it (startup is abortable).
            pending = self._pending_server
            self._pending_server = None
            # Reserved HERE, not at the stop below: the fields are already empty by then, and a router probe landing
            # in that gap would read an idle tree and reinstall over a process that is still running.
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
        # Barrier: wait for a signalled one-shot generation to exit before reporting unloaded, since callers treat
        # this return as "device is free".
        with self._generate_lock:
            pass
        return self.status()

    def status(self) -> dict[str, Any]:
        state = self._state
        # A resident sd-server can exit after load (OOM/crash while idle); drop stale state so clients reload instead
        # of 500ing per generation.
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
                "engine": "sd_cpp",
                "native_mode": None,
                "supports_lora": False,
                "supports_controlnet": False,
                "workflows": [],
            }
        from core.inference import diffusion_lora
        from hub.utils.gguf import extract_quant_token

        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family.name,
            "base_repo": state.base_repo,
            "device": state.device,
            "dtype": "gguf",
            "gguf_variant": extract_quant_token(state.gguf_filename)
            if state.gguf_filename
            else None,
            # Reflect the offload flags actually passed to sd-cli (empty on CPU -> "none"), minus the --backend device
            # pin, which is a card choice rather than an offload decision.
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
            "engine": "sd_cpp",
            "supports_lora": diffusion_lora.supports_lora(
                engine = "sd_cpp",
                family = state.family.name,
                model_kind = "gguf",
                transformer_quant = None,
            ),
            "supports_controlnet": False,
            # "server" = resident sd-server (load once); "oneshot" = legacy per-image sd-cli.
            "native_mode": state.mode,
            # Native supports txt2img only; advertise it so the UI doesn't disable the Create tab.
            "workflows": ["txt2img"],
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
