# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several CUDA/ROCm cards in ONE computer per pipeline (single process, single host, no NCCL and no multi-node),
split only at component or denoiser-call boundaries, never inside a forward. Component placement and the guidance
(CFG) split keep outputs bit-identical to one card; splitting a batch across cards changes the GEMM shapes each card
runs, so a batch split can differ slightly from one card. UNSLOTH_DIFFUSION_MULTIGPU=0 disables."""

from __future__ import annotations

import contextlib
import contextvars
import copy
import dataclasses
import os
import queue
import threading
import time
import types
import weakref
from collections import OrderedDict
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

MULTIGPU_ENV = "UNSLOTH_DIFFUSION_MULTIGPU"
CFG_PARALLEL_ENV = "UNSLOTH_DIFFUSION_CFG_PARALLEL"

DENOISER_ATTRS: tuple[str, ...] = (
    "transformer",
    "transformer_2",
    "unconditional_transformer",
    "unet",
)
_KEEP_WITH_DENOISER_PREFIXES: tuple[str, ...] = ("controlnet",)
_ENTRY_METHODS: tuple[str, ...] = (
    "forward",
    "encode",
    "decode",
    "get_text_features",
    "get_image_features",
)
# Sentinel for ``MemoryPlan.secondary_components``: only the text encoders move (the VAE decodes beside the denoiser).
TEXT_ENCODERS_ONLY: tuple[str, ...] = ("<text_encoders>",)
TEXT_ENCODER_HEADROOM_MIB = 1024
_RESIDENT_FRACTION = 0.85


def multigpu_mode() -> str:
    """``off`` / ``auto`` (default) / ``force``."""
    raw = (os.environ.get(MULTIGPU_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if raw == "force":
        return "force"
    return "auto"


MULTIGPU_ALL_ENV = "UNSLOTH_DIFFUSION_MULTIGPU_ALL"


def multigpu_all_enabled() -> bool:
    """Multi-GPU paths for the remaining families and formats; ``0`` keeps the one-card choices."""
    if multigpu_mode() == "off":
        return False
    return (os.environ.get(MULTIGPU_ALL_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
        "disable",
        "disabled",
    )


UNET_CFG_PARALLEL_ENV = "UNSLOTH_DIFFUSION_CFG_PARALLEL_UNET"
# Automatic (no gpu_ids) arms CFG-parallel on an idle card of the same model; 0 keeps the replica to explicit picks.
AUTO_CFG_PARALLEL_ENV = "UNSLOTH_DIFFUSION_CFG_PARALLEL_AUTO"
# A card counts as idle when no more than this is in use on it (driver + another process's context).
AUTO_CFG_PARALLEL_IDLE_USED_MIB = 1024


def automatic_cfg_parallel_off() -> bool:
    return (os.environ.get(AUTO_CFG_PARALLEL_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


# ... and computes at most this much (nvidia-smi utilization.gpu, percent): a card running another job is not lent.
AUTO_CFG_PARALLEL_IDLE_UTIL_PCT = 5.0


def _idle_card(probe: Any, own_mib: int = 0) -> bool:
    """``own_mib``: memory this load's own components (a companion split's text encoders) hold on the card. It is not
    another process's, so it is not counted as in use. Utilisation is always checked: NVML cannot say whose kernels
    ran, so a busy reading on a card holding this load's encoder is re-read once they are done
    (:func:`_settled_util_pct`), never waived."""
    free, total = getattr(probe, "free_mib", None), getattr(probe, "total_mib", None)
    if free is None or total is None:
        return False
    own = max(0, int(own_mib or 0))
    util = getattr(probe, "util_pct", None)
    if util is not None and float(util) > AUTO_CFG_PARALLEL_IDLE_UTIL_PCT:
        return False
    return int(total) - int(free) - own <= AUTO_CFG_PARALLEL_IDLE_USED_MIB


# nvidia-smi's utilization.gpu covers the last sample period (1/6 s to 1 s by card): wait past it before a re-read
_UTIL_SETTLE_S = 1.2


def _settled_util_pct(ordinal: int, util: Optional[float], own_mib: int) -> Optional[float]:
    """``util`` of a card, re-read once after :data:`_UTIL_SETTLE_S` when it reads busy while this load's encoder is on
    the card: the encoder may just have run there (it runs before the first step), and by the re-read it has
    finished, so what the re-read still sees is another process's work. Otherwise ``util`` unchanged."""
    if not own_mib or util is None or float(util) <= AUTO_CFG_PARALLEL_IDLE_UTIL_PCT:
        return util
    time.sleep(_UTIL_SETTLE_S)
    again = _smi_util_pct(int(ordinal))
    return util if again is None else again


def automatic_partner(
    primary: Any,
    cards: Sequence[Any],
    need_mib: int,
    *,
    p2p: Optional[dict[int, bool]] = None,
    own_mib: Optional[Mapping[int, int]] = None,
    pending_mib: Optional[Mapping[int, int]] = None,
) -> tuple[Optional[int], str]:
    """The card Automatic lends a load's replica to, from ``cards`` (DeviceProbe-like: ordinal, name, total_mib,
    free_mib, util_pct), or None and the reason. Pure: no device is touched. ``own_mib`` (ordinal -> MiB): this load's
    own components already on a card (a companion split's text encoders), never counted as another process's use;
    ``pending_mib``: what this load is about to place there, so the replica must fit beside it.

    One partner per load, on any number of cards: CFG-parallel and the batch split each need exactly one second
    copy, and a third copy would have no second call to run (a step has at most two guidance branches, and a batch is
    cut in two). The cards left over stay free for another load (keep multiple models loaded) or another app.

    A card is a candidate only when it is (in this order of rejection):
      - not the primary;
      - the same card model as the primary (same kernels, so the branches agree and neither card waits on a slower
        one; a mixed pair such as a 4090 with a 3060 is never paired automatically, ``force`` overrides);
      - idle: at most ``AUTO_CFG_PARALLEL_IDLE_USED_MIB`` in use and at most ``AUTO_CFG_PARALLEL_IDLE_UTIL_PCT``
        utilisation (another process's work is never slowed and its memory never taken); this load's own encoders
        do not count (:func:`_idle_card`);
      - able to hold the replica plus its reserve beside what this load places there.
    Among candidates: the most usable free memory, then a peer-to-peer link to the primary, then the lowest ordinal.
    A static rule, not a trial: on identical idle cards the measured gains are 1.46x to 1.79x (well beyond A/A)."""
    p2p = p2p or {}
    own_mib = own_mib or {}
    pending_mib = pending_mib or {}
    pidx = getattr(primary, "ordinal", None)
    pname = getattr(primary, "name", None)
    rejected: dict[str, list[int]] = {}
    fits: list[tuple[int, int, int]] = []
    for card in cards:
        o = int(card.ordinal)
        if o == pidx:
            continue
        if not pname or card.name != pname:
            rejected.setdefault("a different card model", []).append(o)
            continue
        if not _idle_card(card, int(own_mib.get(o, 0))):
            rejected.setdefault("busy", []).append(o)
            continue
        free = int(card.free_mib)
        pending = max(0, int(pending_mib.get(o, 0)))
        usable = free - _replica_reserve_mib(free, card.total_mib) - pending
        if usable < int(need_mib):
            beside = f" beside this load's {pending} MiB" if pending else ""
            rejected.setdefault(f"short of the {int(need_mib)} MiB replica{beside}", []).append(o)
            continue
        # a card holding nothing of this load first: one beside its encoders is the fallback
        clean = 0 if o in own_mib or pending else 1
        fits.append((clean, usable, 1 if p2p.get(o) else 0, -o))
    if fits:
        best = max(fits)
        return -best[3], (
            "idle card of the same model"
            if best[0]
            else "same-model card beside this load's text encoders"
        )
    if not rejected:
        return None, "no other card"
    return None, "; ".join(f"cuda:{','.join(map(str, v))} {k}" for k, v in rejected.items())


def cfg_parallel_mode() -> str:
    """``off`` / ``auto`` (default) / ``force`` (also across different card models)."""
    if multigpu_mode() == "off":
        return "off"
    raw = (os.environ.get(CFG_PARALLEL_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if raw == "force":
        return "force"
    return "auto"


@dataclasses.dataclass(frozen = True)
class MultiGpuSelection:
    primary: int
    secondaries: tuple[int, ...]
    # No gpu_ids (the UI's Automatic): only the companion split runs, and only when one card would offload.
    automatic: bool = False
    # (ordinal, why) of every picked card the selection left out; reported with the placement decision, never compared
    excluded: tuple[tuple[int, str], ...] = dataclasses.field(default = (), compare = False)


_SELECTION: contextvars.ContextVar[Optional[MultiGpuSelection]] = contextvars.ContextVar(
    "unsloth_diffusion_multigpu_selection", default = None
)


AUTOMATIC_ORDINALS = "automatic"

# bf16 compute (the dtype a load resolves on an Ampere or newer primary) needs sm_80 on the card that runs it.
BF16_MIN_CAPABILITY = (8, 0)


def _sm(cap: Optional[tuple[int, int]]) -> str:
    return "sm unknown" if cap is None else f"sm {int(cap[0])}.{int(cap[1])}"


def _card_label(ordinal: int) -> str:
    try:
        import torch
        name = str(torch.cuda.get_device_name(int(ordinal)))
    except Exception:  # noqa: BLE001
        name = ""
    return (
        f"cuda:{int(ordinal)} ({name}, {_sm(_capability(int(ordinal)))})"
        if name
        else (f"cuda:{int(ordinal)} ({_sm(_capability(int(ordinal)))})")
    )


def _torch_arch_floor() -> Optional[tuple[int, int]]:
    """The lowest ``sm_XY`` this torch build ships kernels for (``torch.cuda.get_arch_list``), or None if unreadable."""
    try:
        import torch

        caps = []
        for arch in torch.cuda.get_arch_list():
            digits = str(arch).partition("_")[2]
            if str(arch).startswith("sm_") and digits.isdigit() and len(digits) >= 2:
                caps.append((int(digits[:-1]), int(digits[-1])))
        return min(caps) if caps else None
    except Exception:  # noqa: BLE001
        return None


def _capability_exclusion(primary: int, ordinal: int) -> Optional[str]:
    """Why a CUDA card cannot be a secondary of ``primary``'s load at all, or None. Only a card below the primary that
    this torch build has no kernels for is left out; what the planner may place on any other card (bf16 sm_80, FP8
    sm_89, INT8, packed weights, attention backends) is decided per component (:func:`placement_refusal`), so a mixed
    pair (an Ada beside a 3090, an A100 beside a T4) keeps both cards. Unknown reads as kept."""
    need = _capability(int(primary))
    have = _capability(int(ordinal))
    if need is None or have is None or have >= need:
        return None
    floor = _torch_arch_floor()
    if floor is not None and have < floor:
        return (
            f"{_card_label(ordinal)} left out: this torch build has no kernels below {_sm(floor)}"
        )
    return None


def selection_and_reason(
    primary: Optional[int],
    ordinals: Optional[Sequence[int]],
    *,
    automatic: bool = False,
) -> tuple[Optional[MultiGpuSelection], Optional[str]]:
    """:func:`make_selection` plus, when at least one other card was picked but none is left, why the load runs on one
    card (a mask, an unreadable count, a card the load's dtype cannot run on). The reason is None otherwise."""
    if primary is None or multigpu_mode() == "off":
        return None, None
    picked = tuple(dict.fromkeys(int(o) for o in (ordinals or ()) if int(o) != int(primary)))
    others = picked
    # cards the load's dtype cannot run on (reported with the decision); masked ones only explain a one-card load
    excluded: list[tuple[int, str]] = []
    masked: list[str] = []
    try:
        import torch

        # nvidia-smi can list a card this process's torch cannot address (a mask applied after the listing).
        if torch.cuda.is_available():
            visible = int(torch.cuda.device_count())
            masked = [
                f"cuda:{o} left out: this process sees {visible} CUDA card(s) (CUDA_VISIBLE_DEVICES)"
                for o in others
                if not 0 <= o < visible
            ]
            others = tuple(o for o in others if 0 <= o < visible)
    except Exception:  # noqa: BLE001 - unknown count: the capability filter below still applies
        pass
    if not others:
        return None, ("; ".join(masked) or None) if picked else None
    if _rocm_torch():
        # ROCm's capability is the gfx target (gfx942 (9, 4)), which does not order; match the primary's load dtype
        # instead (arch-pinned attention kernels are gated per card in diffusion_layer_split_plan).
        for o in others:
            if not _rocm_dtype_parity(int(primary), o):
                excluded.append(
                    (
                        o,
                        f"cuda:{o} left out: no native bf16 (this load runs bf16 on cuda:{int(primary)})",
                    )
                )
        others = tuple(o for o in others if _rocm_dtype_parity(int(primary), o))
    else:
        # Only a card torch has no kernels for is left out; per-component dtype and kernels are gated at placement.
        for o in others:
            why = _capability_exclusion(int(primary), o)
            if why is not None:
                excluded.append((o, why))
        dropped = {o for o, _why in excluded}
        others = tuple(o for o in others if o not in dropped)
    if not others:
        return None, "; ".join(why for _o, why in excluded)
    return (
        MultiGpuSelection(
            primary = int(primary),
            secondaries = others,
            automatic = automatic,
            excluded = tuple(excluded),
        ),
        None,
    )


def make_selection(
    primary: Optional[int],
    ordinals: Optional[Sequence[int]],
    *,
    automatic: bool = False,
) -> Optional[MultiGpuSelection]:
    """A selection when at least two distinct cards were picked and the lever is on, else None (one card)."""
    return selection_and_reason(primary, ordinals, automatic = automatic)[0]


def models_kept_loaded() -> bool:
    """Keep multiple models loaded is on (``fit`` / ``always``). Unreadable reads as off, like the arbiter."""
    try:
        from .gpu_residency import current_policy
        return current_policy() != "off"
    except Exception:  # noqa: BLE001
        return False


def automatic_selection(primary: Optional[int]) -> Optional[MultiGpuSelection]:
    """The selection of an automatic load: the ranked card (or the current one) plus every other parent-visible CUDA
    card. None on one GPU, ROCm, XPU, MPS and CPU (the same cheap gate automatic ranking uses)."""
    return automatic_selection_and_reason(primary)[0]


def automatic_selection_and_reason(
    primary: Optional[int],
) -> tuple[Optional[MultiGpuSelection], Optional[str]]:
    """:func:`automatic_selection` plus why a host past the ranking gate (several CUDA cards) ends on one card."""
    if multigpu_mode() == "off":
        return None, None
    try:
        from .diffusion_device import auto_cuda_ranking_possible, resolve_diffusion_device_target
        if not auto_cuda_ranking_possible() or resolve_diffusion_device_target().device != "cuda":
            return None, None
    except Exception:  # noqa: BLE001 -- an unreadable host keeps one card
        return None, None
    try:
        import torch
        if primary is None:
            primary = int(torch.cuda.current_device())
        count = int(torch.cuda.device_count())
    except Exception as exc:  # noqa: BLE001 -- an unreadable host keeps one card
        return None, f"the CUDA card count could not be read ({type(exc).__name__})"
    return selection_and_reason(primary, range(count), automatic = True)


def _capability(ordinal: int) -> Optional[tuple[int, int]]:
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(int(ordinal))
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


def _rocm_torch() -> bool:
    """A ROCm torch build (AMD SDK / Radeon wheels included, which leave ``torch.version.hip`` unset)."""
    try:
        import torch

        from .rocm_bf16 import is_rocm_torch
        return is_rocm_torch(torch)
    except Exception:  # noqa: BLE001
        return False


def _rocm_bf16(ordinal: int) -> Optional[bool]:
    try:
        import torch

        from .rocm_bf16 import rocm_bf16_supported
        return bool(rocm_bf16_supported(torch, int(ordinal)))
    except Exception:  # noqa: BLE001
        return None


def _rocm_dtype_parity(primary: int, ordinal: int) -> bool:
    """A ROCm card runs the dtype the load resolves for the primary: bf16 when the primary has native bf16 (RDNA1/2,
    GCN and Vega emulate it), anything otherwise. An unreadable card is kept, like an unknown CUDA capability."""
    if _rocm_bf16(primary) is not True:
        return True
    return _rocm_bf16(ordinal) is not False


@contextlib.contextmanager
def selection_scope(
    selection: Optional[MultiGpuSelection],
) -> Iterator[Optional[MultiGpuSelection]]:
    """Make ``selection`` visible to the planner and placement for the duration of one load (thread / task local)."""
    if selection is None:
        yield None
        return
    # Only a CFG replica reads the recorded compile arguments; automatic placement builds one unless its lever is off.
    if (
        not selection.automatic
        or cfg_parallel_mode() == "force"
        or not automatic_cfg_parallel_off()
    ):
        install_compile_recorder()
    token = _SELECTION.set(selection)
    try:
        yield selection
    finally:
        _SELECTION.reset(token)


def _kept_loaded_policy() -> Optional[str]:
    """``fit`` / ``always`` while Keep multiple models loaded is on, else None (unreadable reads as off)."""
    try:
        from .gpu_residency import current_policy
        policy = str(current_policy())
    except Exception:  # noqa: BLE001
        return None
    return None if policy == "off" else policy


# Keep multiple models loaded: the residency arbiter admits and claims an image or video load on the ONE card it priced
# before the load (gpu_residency.media_fit_kwargs passes gpu_ids=[gpu_ordinal]); the multi-GPU planner picks its
# secondary only after the weights load, so a second card would hold memory the arbiter never priced, claimed or
# evicts for. Such a load stays on one card and says so (one_card_note).
KEPT_LOADED_REASON = (
    "Keep multiple models loaded is {policy}: the residency arbiter prices and claims one card per image or video "
    "load, so a second card would hold memory it never accounted for"
)


@dataclasses.dataclass(frozen = True)
class OneCardNote:
    """A multi-card request (explicit gpu_ids, or Automatic on a host with several CUDA cards) that runs on one
    card, and why. Recorded as the load's placement decision (``chosen`` one_card) and logged once."""

    primary: Optional[int]
    automatic: bool
    reason: str

    def summary(self) -> str:
        card = f"cuda:{self.primary}" if self.primary is not None else "the selected card"
        return f"one card: everything on {card} ({self.reason})"


def one_card_note(
    primary: Optional[int], ordinals: Any
) -> tuple[Optional[MultiGpuSelection], Optional[OneCardNote]]:
    """The selection of a load given begin_load's ``_gpu_ordinals``, and the note when a multi-card request ends on
    one card. Selections are those of :func:`automatic_selection` / :func:`make_selection`."""
    automatic = ordinals == AUTOMATIC_ORDINALS
    # the reason is asked for only when the selection is None, so a load keeps calling the two selection functions
    if automatic:
        selection = automatic_selection(primary)
        why = automatic_selection_and_reason(primary)[1] if selection is None else None
    else:
        selection = make_selection(primary, ordinals)
        why = selection_and_reason(primary, ordinals)[1] if selection is None and ordinals else None
    if selection is not None and models_kept_loaded():
        policy = (_kept_loaded_policy() or "on").title()
        return None, OneCardNote(
            selection.primary, automatic, KEPT_LOADED_REASON.format(policy = policy)
        )
    if selection is None and why is not None:
        if automatic and primary is None:
            with contextlib.suppress(Exception):
                import torch
                primary = int(torch.cuda.current_device())
        return None, OneCardNote(None if primary is None else int(primary), automatic, why)
    return selection, None


def _log_info(message: str, *args: Any) -> None:
    try:
        from loggers import get_logger
        get_logger(__name__).info(message, *args)
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).info(message, *args)


def record_one_card_note(pipe: Any, note: Optional[OneCardNote]) -> bool:
    """Report ``note`` as ``pipe``'s placement decision unless the planner recorded one. True when recorded."""
    if pipe is None or note is None or placement_decision(pipe) is not None:
        return False
    with contextlib.suppress(TypeError):
        _DECISIONS[pipe] = PlacementDecision(
            chosen = "one_card",
            summary = f"placement: {note.summary()}",
            primary = note.primary,
            automatic = bool(note.automatic),
        )
        return True
    return False


def with_load_selection(run_load: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a backend's ``_run_load(self, **kwargs)``: pop ``_gpu_ordinals`` (every selected card, from
    begin_load, or :data:`AUTOMATIC_ORDINALS`) and run the load inside :func:`selection_scope` with ``gpu_ordinal``
    as the primary. Without a multi-card selection the load runs exactly as before; a multi-card request that ends on
    one card logs why and reports it as the loaded pipeline's placement decision."""
    import functools

    @functools.wraps(run_load)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ordinals = kwargs.pop("_gpu_ordinals", None)
        selection, note = one_card_note(kwargs.get("gpu_ordinal"), ordinals)
        if note is not None:
            _log_info("diffusion.multigpu: %s", note.summary())
        before = getattr(getattr(self, "_state", None), "pipe", None)
        with selection_scope(selection):
            result = run_load(self, *args, **kwargs)
        if note is not None:
            pipe = getattr(getattr(self, "_state", None), "pipe", None)
            if pipe is not None and pipe is not before:
                record_one_card_note(pipe, note)
        return result

    return wrapper


def peer_selection(selection: Optional[MultiGpuSelection]) -> Optional[MultiGpuSelection]:
    """``selection`` without the secondaries below the primary's capability (the rule every selection used before mixed
    pairs were kept), or None when none is left. For the sd.cpp servers: a separate binary whose kernels Studio does not
    gate per card. ROCm and unknown capabilities are kept, as before."""
    if selection is None or _rocm_torch():
        return selection
    need = _capability(int(selection.primary))
    if need is None:
        return selection
    kept = tuple(o for o in selection.secondaries if (_capability(o) or need) >= need)
    if kept == tuple(selection.secondaries):
        return selection
    dropped = [o for o in selection.secondaries if o not in kept]
    _log_info(
        "diffusion.multigpu: sd.cpp keeps %s off this load (below cuda:%s's %s)",
        ", ".join(_card_label(o) for o in dropped),
        selection.primary,
        _sm(need),
    )
    if not kept:
        return None
    return dataclasses.replace(selection, secondaries = kept)


def current_selection() -> Optional[MultiGpuSelection]:
    if multigpu_mode() == "off":
        return None
    return _SELECTION.get()


@dataclasses.dataclass(frozen = True)
class DeviceProbe:
    ordinal: int
    name: str
    total_mib: Optional[int]
    free_mib: Optional[int]
    util_pct: Optional[float] = None


_PROBE_LOCK = threading.Lock()
_NAMES: dict[int, str] = {}


def _free_total_mib(ordinal: int) -> tuple[Optional[int], Optional[int]]:
    selection = _SELECTION.get()
    if selection is not None and selection.automatic and int(ordinal) != selection.primary:
        # Sized from nvidia-smi: an automatic load must not open a CUDA context on a card it may never use.
        from .diffusion_device import smi_free_total_mib
        return smi_free_total_mib(int(ordinal))
    try:
        import torch

        from utils.hardware import trusted_mem_get_info

        free, total = trusted_mem_get_info(int(ordinal), module = torch.cuda)
        return int(free) // (1 << 20), int(total) // (1 << 20)
    except Exception:  # noqa: BLE001
        return None, None


def probe_devices(ordinals: Sequence[int]) -> dict[int, DeviceProbe]:
    """Name (cached per process) and free / total memory of each card."""
    import torch

    ords = [int(o) for o in ordinals]
    with _PROBE_LOCK:
        for o in ords:
            if o not in _NAMES:
                try:
                    _NAMES[o] = str(torch.cuda.get_device_name(o))
                except Exception:  # noqa: BLE001
                    _NAMES[o] = ""
    from .diffusion_device import smi_rows_snapshot

    out: dict[int, DeviceProbe] = {}
    with smi_rows_snapshot():
        for o in ords:
            free, total = _free_total_mib(o)
            out[o] = DeviceProbe(
                ordinal = o,
                name = _NAMES.get(o, ""),
                total_mib = total,
                free_mib = free,
                util_pct = _util_pct(o),
            )
    return out


def _util_pct(ordinal: int) -> Optional[float]:
    """nvidia-smi utilisation of a card an automatic load may borrow; None elsewhere (unknown counts as idle)."""
    selection = _SELECTION.get()
    if selection is None or not selection.automatic or int(ordinal) == selection.primary:
        return None
    return _smi_util_pct(ordinal)


def _smi_util_pct(ordinal: int) -> Optional[float]:
    """nvidia-smi utilisation of ``ordinal`` (no CUDA context), read outside any load scope; None when unknown."""
    try:
        from .diffusion_device import smi_util_pct
        return smi_util_pct(int(ordinal))
    except Exception:  # noqa: BLE001
        return None


def best_secondary(
    selection: MultiGpuSelection,
    probes: Optional[dict[int, DeviceProbe]] = None,
    *,
    mode: Optional[str] = None,
) -> Optional[int]:
    if not selection.secondaries:
        return None
    from .diffusion_memory import (
        DeviceMemory,
        _fast_device_budget_mib,
        _safe_device_budget_mib,
        _user_budget_reserve_mib,
    )

    budgeted = _user_budget_reserve_mib(1 << 20) > 0
    limit_of = _fast_device_budget_mib if mode == "fast" else _safe_device_budget_mib

    def usable(free: Optional[int], total: Optional[int]) -> int:
        if free is None:
            return -1
        if not budgeted:
            return int(free)
        memory = DeviceMemory("cuda", "cuda", "discrete_vram", int(free), total)
        return int(limit_of(memory) or 0)

    peer = _peer_rank(selection)
    if probes is None and selection.automatic:
        free = {o: _free_total_mib(o) for o in selection.secondaries}
        return max(selection.secondaries, key = lambda o: (peer(o), usable(*free[o]), -o))
    if probes is None:
        probes = probe_devices((selection.primary, *selection.secondaries))

    def key(o: int) -> tuple[int, int, int]:
        p = probes.get(o)
        return (peer(o), usable(p.free_mib, p.total_mib) if p is not None else -1, -o)

    return max(selection.secondaries, key = key)


def _peer_rank(selection: MultiGpuSelection) -> Callable[[int], int]:
    """1 for a secondary at or above the primary's capability (or unknown), 0 for a lower one: a card the old rule kept
    is tried first, so a mixed selection decides as before whenever such a card is in it. Constant on identical cards."""
    if _rocm_torch():
        return lambda o: 1
    need = _capability(int(selection.primary))
    if need is None:
        return lambda o: 1
    return lambda o: 1 if (_capability(int(o)) or need) >= need else 0


def same_model_secondaries(
    selection: Optional[MultiGpuSelection], logger: Any = None
) -> tuple[int, ...]:
    """The secondaries of the primary's card model (by device name), in selection order: the cards a replica-type
    split (the same work on both cards) may use without pricing a slower card. Unreadable names keep every card."""
    if selection is None or not selection.secondaries:
        return ()
    try:
        import torch
        name = str(torch.cuda.get_device_name(int(selection.primary)))
        names = {o: str(torch.cuda.get_device_name(int(o))) for o in selection.secondaries}
    except Exception:  # noqa: BLE001
        return tuple(selection.secondaries)
    kept = tuple(o for o in selection.secondaries if names[o] == name)
    if logger is not None and len(kept) < len(selection.secondaries):
        logger.info(
            "diffusion.multigpu: %s not used for a split of the same work (a different card model than cuda:%s)",
            ", ".join(f"cuda:{o}" for o in selection.secondaries if o not in kept),
            selection.primary,
        )
    return kept


def secondary_below_primary(ordinal: int) -> bool:
    """``ordinal`` is a CUDA card below the current selection's primary capability (a mixed pair). False without a
    selection, on ROCm or when either capability is unknown."""
    selection = current_selection()
    if selection is None or _rocm_torch():
        return False
    need, have = _capability(int(selection.primary)), _capability(int(ordinal))
    return need is not None and have is not None and have < need


@dataclasses.dataclass(frozen = True)
class CompanionSplit:
    secondary_ordinal: int
    primary_required_mib: int
    primary_limit_mib: int
    secondary_required_mib: int
    secondary_limit_mib: int
    reason: str
    # None = every companion moves; TEXT_ENCODERS_ONLY = only the encoders (the VAE stays with the denoiser)
    components: Optional[tuple[str, ...]] = None
    # working memory the secondary is priced with above its weights (an encoders-only split: the encode peak)
    secondary_headroom_mib: int = 0


def plan_companion_split(
    *,
    single_card_policy: str,
    model_dense_mib: Optional[int],
    companion_dense_mib: Optional[int],
    runtime_headroom_mib: int,
    base_overhead_mib: int,
    primary_limit_mib: Optional[int],
    secondary_limit_mib: Callable[[int], Optional[int]],
    selection: Optional[MultiGpuSelection],
    text_encoder_dense_mib: Optional[int] = None,
    phase_headroom_mib: Optional[tuple[int, int, int]] = None,
) -> Optional[CompanionSplit]:
    """Pure decision: return a split when the single-card plan offloads (or ``force``), the denoiser fits the primary
    under the resident rule, and the moved companions fit a secondary. Tried in order: every companion (text encoders
    + VAE + the rest, the secondary priced with the full runtime headroom since it runs the decode), then the text
    encoders alone (the VAE decodes on the primary; the secondary needs only an encoder forward's room).
    ``primary_limit_mib`` / ``secondary_limit_mib(ordinal)`` are each card's resident limit under the caller's memory
    mode rule (auto: 85% of the safe budget; fast: the fast budget). ``phase_headroom_mib`` = measured (encode,
    denoise, decode) MiB above the weights for families Studio has calibrated: each card is then priced only for the
    phases it runs instead of the flat runtime headroom."""
    mode = multigpu_mode()
    if selection is None or mode == "off":
        return None
    if single_card_policy == "none" and mode != "force":
        return None
    if model_dense_mib is None or not companion_dense_mib or primary_limit_mib is None:
        return None
    limits: dict[int, int] = {}
    for ordinal in selection.secondaries:
        limit = secondary_limit_mib(ordinal)
        if limit is not None:
            limits[ordinal] = int(limit)
    if not limits:
        return None
    runtime = int(runtime_headroom_mib)
    if phase_headroom_mib is not None:
        encode, denoise, decode = (int(v) for v in phase_headroom_mib)
        # (moved MiB, primary headroom, secondary headroom, components, label)
        companions_split = (denoise, max(encode, decode))
        encoders_split = (max(denoise, decode), encode)
    else:
        companions_split = (runtime, runtime)
        encoders_split = (runtime, TEXT_ENCODER_HEADROOM_MIB)
    candidates: list[tuple[int, int, int, Optional[tuple[str, ...]], str]] = [
        (int(companion_dense_mib), *companions_split, None, "text encoders + VAE"),
    ]
    if text_encoder_dense_mib and 0 < int(text_encoder_dense_mib) <= int(companion_dense_mib):
        candidates.append(
            (int(text_encoder_dense_mib), *encoders_split, TEXT_ENCODERS_ONLY, "text encoders")
        )
    for moved_mib, pri_headroom, sec_headroom, components, label in candidates:
        primary_required = (
            max(0, int(model_dense_mib) - moved_mib) + int(pri_headroom) + int(base_overhead_mib)
        )
        if primary_required > int(primary_limit_mib):
            continue
        secondary_required = moved_mib + sec_headroom + int(base_overhead_mib)
        fits = [o for o, limit in limits.items() if secondary_required <= limit]
        if not fits:
            continue
        peer = _peer_rank(selection)
        best = max(fits, key = lambda o: (peer(o), limits[o], -o))
        return CompanionSplit(
            secondary_ordinal = best,
            primary_required_mib = primary_required,
            primary_limit_mib = int(primary_limit_mib),
            secondary_required_mib = secondary_required,
            secondary_headroom_mib = int(sec_headroom),
            secondary_limit_mib = limits[best],
            components = components,
            reason = (
                f"multi-GPU: denoiser resident on cuda:{selection.primary} ({primary_required} of "
                f"{int(primary_limit_mib)} MiB), {label} on cuda:{best} ({secondary_required} of {limits[best]} MiB); "
                f"single card would use '{single_card_policy}'"
            ),
        )
    return None


@dataclasses.dataclass(frozen = True)
class AssignmentTrial:
    """One companion assignment the planner tried: what moves, and each card's need against its resident limit."""

    moved: tuple[str, ...]
    moved_mib: int
    primary_need_mib: int
    primary_limit_mib: int
    secondary_need_mib: int
    secondary_limit_mib: int

    @property
    def fits(self) -> bool:
        return (
            self.primary_need_mib <= self.primary_limit_mib
            and self.secondary_need_mib <= self.secondary_limit_mib
        )

    def why_not(self, primary: int, secondary: int) -> str:
        moved = "+".join(self.moved)
        if self.primary_need_mib > self.primary_limit_mib:
            return (
                f"{moved} on cuda:{secondary}: cuda:{primary} still needs {_gib(self.primary_need_mib)} of "
                f"{_gib(self.primary_limit_mib)}"
            )
        return (
            f"{moved} on cuda:{secondary}: needs {_gib(self.secondary_need_mib)} there, "
            f"{_gib(self.secondary_limit_mib)} allowed"
        )


def _gib(mib: Optional[int]) -> str:
    return "unknown" if mib is None else f"{int(mib) / 1024:.1f} GiB"


def component_assignment_trials(
    sizes: dict[str, tuple[int, str]],
    *,
    headroom_mib: int,
    overhead_mib: int,
    primary_limit_mib: int,
    secondary_limit_mib: int,
    move_denoisers: bool = True,
    encoders_only_limit_mib: Optional[int] = None,
) -> list[AssignmentTrial]:
    """Every assignment :func:`choose_component_assignment` tries, in its order, priced (the first that fits wins).
    ``encoders_only_limit_mib``: what a secondary holding nothing but text encoders may take (its free memory minus
    ComfyUI's per-card floor); such a card runs one encoder forward per prompt, so it is priced with the weights and
    an encoder forward's room instead of the base overhead and the 85% resident margin of a card that denoises."""
    dits = [n for n, (_m, r) in sizes.items() if r == "dit"]
    if not dits:
        return []
    first = "transformer" if "transformer" in dits else dits[0]
    seconds = [n for n in dits if n != first]
    companions = [
        n
        for n, (_m, r) in sizes.items()
        if r != "dit" and not n.startswith(_KEEP_WITH_DENOISER_PREFIXES)
    ]
    encoders = [n for n in companions if sizes[n][1] == "text_encoder"]
    candidates: list[tuple[str, ...]] = []
    if companions:
        candidates.append(tuple(companions))
    if encoders and len(encoders) < len(companions):
        candidates.append(tuple(encoders))
    if seconds and move_denoisers:
        candidates.append(tuple(seconds))
        if companions:
            candidates.append(tuple(seconds + companions))

    def need(names: list) -> int:
        weights = sum(sizes[n][0] for n in names)
        busy = any(sizes[n][1] != "text_encoder" for n in names)
        return weights + (headroom_mib if busy else TEXT_ENCODER_HEADROOM_MIB) + overhead_mib

    def secondary(moved: tuple[str, ...]) -> tuple[int, int]:
        standard = (need(list(moved)), int(secondary_limit_mib))
        if encoders_only_limit_mib is None or any(sizes[n][1] != "text_encoder" for n in moved):
            return standard
        encoders = (
            sum(sizes[n][0] for n in moved) + TEXT_ENCODER_HEADROOM_MIB,
            int(encoders_only_limit_mib),
        )
        return encoders if encoders[1] - encoders[0] > standard[1] - standard[0] else standard

    trials = []
    for moved in candidates:
        sec_need, sec_limit = secondary(moved)
        trials.append(
            AssignmentTrial(
                moved = moved,
                moved_mib = sum(sizes[n][0] for n in moved),
                primary_need_mib = need([n for n in sizes if n not in moved]),
                primary_limit_mib = int(primary_limit_mib),
                secondary_need_mib = sec_need,
                secondary_limit_mib = sec_limit,
            )
        )
    return trials


def choose_component_assignment(
    sizes: dict[str, tuple[int, str]],
    *,
    headroom_mib: int,
    overhead_mib: int,
    primary_limit_mib: int,
    secondary_limit_mib: int,
    move_denoisers: bool = True,
    encoders_only_limit_mib: Optional[int] = None,
) -> Optional[tuple[str, ...]]:
    """Names to place on the secondary so each card holds its share resident, or None. ``sizes`` maps component name
    -> (MiB, role) with role 'dit' / 'text_encoder' / 'other'. A card running a denoiser or the VAE (any 'other'
    component) is priced with the runtime headroom; a card holding only text encoders with an encoder forward's room.
    Tried in order: every companion; the text encoders alone; the second denoiser; the second denoiser plus the
    companions. The first denoiser always stays on the primary; ``move_denoisers=False`` (the secondary cannot run the
    denoiser's attention backend) keeps every denoiser there."""
    for trial in component_assignment_trials(
        sizes,
        headroom_mib = headroom_mib,
        overhead_mib = overhead_mib,
        primary_limit_mib = primary_limit_mib,
        secondary_limit_mib = secondary_limit_mib,
        move_denoisers = move_denoisers,
        encoders_only_limit_mib = encoders_only_limit_mib,
    ):
        if trial.fits:
            return trial.moved
    return None


_FP8_DTYPE_NAMES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
)
_PLAIN_TENSOR_TYPES: tuple[str, ...] = ("Tensor", "Parameter", "Buffer")


def _weight_kernels(module: Any) -> set[str]:
    """What the weights of ``module`` need a card to run: ``bf16`` (bf16 compute), ``fp8`` (FP8 weights: scaled_mm /
    fp8-stored Linears), ``int8`` (plain int8 tensors: torch._int_mm / the INT8 GEMMs) or ``packed`` (GGUF, NVFP4 /
    MXFP8 blocks, bitsandbytes / torchao or any other tensor subclass: kernels chosen for the primary's card). Empty for
    fp16 / fp32 dense weights."""
    kinds: set[str] = set()
    if module is None:
        return {"unknown"}
    with contextlib.suppress(Exception):
        from .diffusion_comfy_quant import fp8_stored_layers
        if fp8_stored_layers(module) > 0:
            kinds.add("fp8")
    with contextlib.suppress(Exception):
        from .diffusion_grouped_int8 import grouped_packed_layers
        from .diffusion_packed_block import packed_block_layers
        if packed_block_layers(module) > 0 or grouped_packed_layers(module) > 0:
            kinds.add("packed")
    try:
        tensors = [*module.parameters(), *module.buffers()]
    except Exception:  # noqa: BLE001 - unreadable: only the layer counters above
        tensors = []
    if not tensors and not kinds:
        # nothing to read (a stub, a module not built yet): the caller assumes the load's dtype
        return {"unknown"}
    for t in tensors:
        dtype = str(getattr(t, "dtype", "")).rpartition(".")[2]
        cls = type(t).__name__
        low = cls.lower()
        if dtype in _FP8_DTYPE_NAMES or "float8" in low or "fp8" in low:
            kinds.add("fp8")
        elif dtype == "int8" and cls in _PLAIN_TENSOR_TYPES:
            kinds.add("int8")
        elif dtype in ("uint8", "int4", "uint4") or cls not in _PLAIN_TENSOR_TYPES:
            kinds.add("packed")
        elif dtype == "bfloat16":
            kinds.add("bf16")
    return kinds


def _kernel_floors() -> dict[str, tuple[int, int]]:
    """Minimum capability per weight kind: bf16 sm_80; int8 and fp8 from the dense-quant floors the hosted-checkpoint
    ladder uses (diffusion_transformer_quant._HOSTED_AUTO_FLOORS: torch._int_mm sm_75, FP8 scaled_mm sm_89)."""
    floors = {"bf16": BF16_MIN_CAPABILITY, "int8": (7, 5), "fp8": (8, 9)}
    with contextlib.suppress(Exception):
        from .diffusion_transformer_quant import _HOSTED_AUTO_FLOORS, TQ_FP8, TQ_INT8

        table = dict(_HOSTED_AUTO_FLOORS)
        floors["int8"] = tuple(table.get(TQ_INT8, floors["int8"]))
        floors["fp8"] = tuple(table.get(TQ_FP8, floors["fp8"]))
    return floors


_KIND_LABELS = {
    "bf16": "bf16 compute needs",
    "fp8": "FP8 weights (scaled_mm) need",
    "int8": "INT8 GEMMs need",
}


def placement_refusal(pipe: Any, names: Sequence[str], primary: int, ordinal: int) -> Optional[str]:
    """Why the components ``names`` of ``pipe`` cannot run on ``ordinal`` beside ``primary``, or None. Only a CUDA
    card of lower capability than the primary is checked (an equal or higher one, an unknown capability and ROCm
    keep today's rule), so identical cards decide exactly as before. Each component needs what its weights run
    (:func:`_weight_kernels`): bf16 sm_80, FP8 sm_89, INT8 sm_75, packed / unrecognised formats the primary's
    capability. A denoiser must also run the engaged attention backend there (diffusion_layer_split_plan._backend_cards:
    an arch-pinned backend such as flash3 / Sage / FA2 only on the primary's capability)."""
    if _rocm_torch():
        return None
    need = _capability(int(primary))
    have = _capability(int(ordinal))
    if need is None or have is None or have >= need:
        return None
    if any(str(n) in DENOISER_ATTRS for n in names):
        with contextlib.suppress(Exception):
            from .diffusion_layer_split_plan import _backend_cards
            kept, backend = _backend_cards(pipe, int(primary), (int(ordinal),), None, guard = False)
            if int(ordinal) not in kept:
                return (
                    f"denoiser cannot run on cuda:{int(ordinal)} ({_sm(have)}): attention backend {backend} was "
                    f"verified for cuda:{int(primary)}'s architecture only"
                )
    floors = _kernel_floors()
    for name in names:
        kinds = _weight_kernels(getattr(pipe, str(name), None))
        if "unknown" in kinds:
            # unreadable weights run the load's dtype: bf16 on an Ampere or newer primary
            kinds = {"bf16"} if need >= BF16_MIN_CAPABILITY else set()
        for kind in ("packed", "fp8", "int8", "bf16"):
            if kind not in kinds:
                continue
            floor = need if kind == "packed" else floors[kind]
            if have >= floor:
                continue
            what = (
                f"packed / quantized weights whose kernels were chosen for cuda:{int(primary)} ({_sm(need)})"
                if kind == "packed"
                else f"{_KIND_LABELS[kind]} {_sm(floor)}"
            )
            return f"{name} cannot run on cuda:{int(ordinal)} ({_sm(have)}): {what}"
    return None


def _secondary_runs_denoiser(
    pipe: Any, selection: MultiGpuSelection, secondary: int, sizes: dict, logger: Any
) -> bool:
    """Whether a second denoiser may move wholesale to ``secondary``. ROCm admits mixed gfx targets (make_selection),
    so the card must pass the layer split's per-card attention gate: an arch-pinned kernel (flash_attn, AITER) only on
    the primary's gfx target, native SDPA not math-only beside a fused primary. On CUDA only a card below the primary
    (a mixed pair, see make_selection) is checked, by the layer split's backend gate."""
    if sum(1 for _m, role in sizes.values() if role == "dit") < 2:
        return True
    if not _rocm_torch():
        # A card below the primary (a mixed pair) must run the engaged attention backend: an arch-pinned one
        # (flash3 / Sage / FA2) only on the primary's capability. Equal or higher cards keep today's rule.
        need, have = _capability(int(selection.primary)), _capability(int(secondary))
        if need is None or have is None or have >= need:
            return True
        try:
            from .diffusion_layer_split_plan import _backend_cards
            kept, backend = _backend_cards(pipe, int(selection.primary), (int(secondary),), None)
        except Exception:  # noqa: BLE001 - unverifiable: keep the denoisers on the primary
            return False
        if int(secondary) in kept:
            return True
        if logger is not None:
            logger.info(
                "diffusion.multigpu: second denoiser stays on cuda:%s (attention backend %s was verified for "
                "cuda:%s's architecture only)",
                selection.primary,
                backend,
                selection.primary,
            )
        return False
    try:
        from .diffusion_layer_split_plan import _rocm_backend_cards
        kept, _backend = _rocm_backend_cards(
            pipe, int(selection.primary), (int(secondary),), logger
        )
    except Exception:  # noqa: BLE001 - unverifiable: keep the denoisers on the primary
        return False
    if int(secondary) in kept:
        return True
    if logger is not None:
        logger.info(
            "diffusion.multigpu: second denoiser stays on cuda:%s (attention backend not verified on cuda:%s)",
            selection.primary,
            secondary,
        )
    return False


@dataclasses.dataclass(frozen = True)
class PlacementCandidate:
    """One placement the automatic planner considered: ``kind`` is one of ``one_card`` (everything resident on the
    primary), ``companion_split``, ``layer_split``, ``offload`` (one card, streamed), ``cfg_replica`` (the one-card
    plan plus a CFG replica on a second card) or ``cfg_parallel``."""

    kind: str
    reason: str


@dataclasses.dataclass(frozen = True)
class PlacementDecision:
    """What the multi-GPU refinement chose for a load under a multi-card selection, and why every other candidate
    was rejected. Recorded per pipeline; ``status()["multi_gpu"]["decision"]`` shows it (``"placement"`` there stays
    the companion split's component-to-device map). ``fallback_resident``: the one-card plan a split would fall back
    to keeps everything resident (a forced split of a model that fits one card)."""

    chosen: str
    summary: str
    primary: Optional[int]
    automatic: bool
    rejected: tuple[PlacementCandidate, ...] = ()
    fallback_resident: bool = False
    # a cfg_replica beside a companion split's encoders: what runs when the replica declines (the split stays placed)
    fallback_kind: Optional[str] = None
    fallback_summary: Optional[str] = None

    def describe(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "summary": self.summary,
            "primary": None if self.primary is None else f"cuda:{self.primary}",
            "automatic": self.automatic,
            "rejected": [{"kind": c.kind, "reason": c.reason} for c in self.rejected],
        }


_DECISIONS: "weakref.WeakKeyDictionary[Any, PlacementDecision]" = weakref.WeakKeyDictionary()


def placement_decision(pipe: Any) -> Optional[PlacementDecision]:
    try:
        return _DECISIONS.get(pipe) if pipe is not None else None
    except TypeError:
        return None


def _record_decision(
    pipe: Any,
    selection: MultiGpuSelection,
    chosen: str,
    summary: str,
    rejected: Sequence[PlacementCandidate] = (),
    *,
    fallback_resident: bool = False,
    fallback_kind: Optional[str] = None,
    fallback_summary: Optional[str] = None,
) -> None:
    # cards the selection left out (a card the load's dtype cannot run on, a masked one): named with the decision
    left_out = tuple(PlacementCandidate("card", why) for _o, why in selection.excluded)
    with contextlib.suppress(TypeError):
        _DECISIONS[pipe] = PlacementDecision(
            chosen = chosen,
            summary = f"placement: {summary}",
            primary = int(selection.primary),
            automatic = bool(selection.automatic),
            rejected = (*tuple(rejected), *left_out),
            fallback_resident = bool(fallback_resident),
            fallback_kind = fallback_kind,
            fallback_summary = fallback_summary,
        )


def forget_placement_decision(pipe: Any) -> None:
    with contextlib.suppress(TypeError):
        _DECISIONS.pop(pipe, None)


def _denoiser_mib(sizes: dict[str, tuple[int, str]]) -> int:
    return sum(m for m, role in sizes.values() if role == "dit")


def _one_card_rejection(plan: Any, primary: int) -> PlacementCandidate:
    required = plan.estimates.get("resident_required_mib")
    budget = plan.estimates.get("safe_device_budget_mib")
    return PlacementCandidate(
        "one_card",
        f"everything resident on cuda:{primary} needs {_gib(required)}, {_gib(budget)} free after the reserve; "
        f"one card would use '{plan.offload_policy}'",
    )


# One card plus a copy before a layer split when a copy fits and streaming is hidden; below
# STREAM_HIDDEN_FRACTION exposed, card speed spread decides instead; splits add no parallelism.
STREAM_HIDDEN_FRACTION = 0.05
STREAM_GATE_ENV = "UNSLOTH_DIFFUSION_SPLIT_STREAM_GATE"
_GEMM_PROBES: dict = {}


def _gemm_tflops(device_index: int, dtype: Any) -> Optional[float]:
    """Dense GEMM TFLOPS of ``device_index`` in ``dtype`` (the denoiser's compute dtype: fp16 on a T4, where bf16 runs
    emulated), best of a few runs, cached."""
    key = (int(device_index), str(dtype))
    if key in _GEMM_PROBES:
        return _GEMM_PROBES[key]
    result = None
    try:
        torch = _torch()
        with torch.cuda.device(int(device_index)):
            a = torch.randn(4096, 3072, dtype = dtype, device = "cuda")
            b = torch.randn(3072, 3072, dtype = dtype, device = "cuda")
            start, end = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            torch.mm(a, b)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(3):
                start.record()
                for _ in range(10):
                    torch.mm(a, b)
                end.record()
                end.synchronize()
                best = min(best, start.elapsed_time(end) / 10)
            del a, b
        if best > 0:
            result = 2.0 * 4096 * 3072 * 3072 / (best / 1000.0) / 1e12
    except Exception:  # noqa: BLE001 - unmeasurable: no gate
        result = None
    _GEMM_PROBES[key] = result
    return result


def stream_exposure(
    streamed_mib: int,
    dense_params: float,
    *,
    h2d_gib_s: float,
    tflops: float,
    side_px: int = 1024,
) -> tuple[float, float, float]:
    """(transfer s, compute s, exposed share) of one denoise step streaming ``streamed_mib`` while computing a DiT of
    ``dense_params`` parameters at ``side_px``. Exposed share = time the step waits on the stream / step time. Pure."""
    from .diffusion_memory import _DIT_COMPUTE_EFFICIENCY, _DIT_PIXELS_PER_TOKEN

    transfer = max(0, int(streamed_mib)) / 1024.0 / max(float(h2d_gib_s), 1e-6)
    tokens = (int(side_px) // _DIT_PIXELS_PER_TOKEN) ** 2
    compute = (
        _DIT_COMPUTE_EFFICIENCY
        * 2.0
        * float(dense_params)
        * tokens
        / (max(float(tflops), 1e-6) * 1e12)
    )
    step = max(transfer, compute)
    exposed = max(0.0, transfer - compute) / step if step > 0 else 0.0
    return transfer, compute, exposed


# Code buffers of fused quantized modules; float buffers are scales or tables.
_QUANT_CODE_DTYPES = frozenset(
    ("int8", "uint8", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")
)


def logical_weight_elements(module: Any) -> int:
    """Logical weight count for the ``2 x params x tokens`` price; ``parameters()`` misses quantized/packed layouts.

    Leaf linears count ``in x out`` + bias; other modules count own params and integer / fp8 code buffers."""
    seen: set[int] = set()
    total = 0
    for sub in module.modules():
        in_f, out_f = getattr(sub, "in_features", None), getattr(sub, "out_features", None)
        linear_like = (
            isinstance(in_f, int)
            and isinstance(out_f, int)
            and in_f > 0
            and out_f > 0
            and next(sub.children(), None) is None
        )
        if linear_like:
            own = [*sub.parameters(recurse = False), *sub.buffers(recurse = False)]
            if any(id(t) in seen for t in own):
                continue  # tied to a layer already counted
            seen.update(id(t) for t in own)
            total += int(in_f) * int(out_f)
            bias = getattr(sub, "bias", None)
            if bias is not None and hasattr(bias, "numel"):
                total += int(bias.numel())
            continue
        for t in sub.parameters(recurse = False):
            if id(t) not in seen:
                seen.add(id(t))
                total += int(t.numel())
        for t in sub.buffers(recurse = False):
            if id(t) not in seen and str(t.dtype).replace("torch.", "") in _QUANT_CODE_DTYPES:
                seen.add(id(t))
                total += int(t.numel())
    return total


def _hidden_stream_reason(pipe: Any, plan: Any, sizes: dict, selection: Any) -> Optional[str]:
    """Why Automatic keeps the one-card streamed plan instead of a layer split, or None (try the split as before).

    Only for an automatic selection whose one-card plan streams the denoiser in groups (a capacity plan such as
    sequential / whole-module offload keeps the split), priced at Studio's cost canvas from this card's measured
    pinned H2D bandwidth and its GEMM throughput in the denoiser's dtype. ``UNSLOTH_DIFFUSION_SPLIT_STREAM_GATE=0``
    restores the old choice."""
    if not getattr(selection, "automatic", False):
        return None
    if (os.environ.get(STREAM_GATE_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    if getattr(plan, "offload_policy", None) != "group":
        return None
    denoiser_cls = type(getattr(pipe, "transformer", None)).__name__
    if denoiser_cls not in _DIT_BATCH_SPLIT_KWARGS:
        # Only where a second-card copy can serve every kind of request this family runs (an even batch splits, a
        # guided step pairs): elsewhere (Z-Image, measured 2.3-3.1x faster split) the split stays as on main.
        return None
    partner = _copy_partner(pipe, selection)
    if partner is None:
        # the copy does not fit an idle same-model card: the split stays (FLUX.2-klein-9B: 0-14% faster split)
        return None
    try:
        from .diffusion_memory import _COST_CANVAS_PX, _offload_cost_measurements

        measured = _offload_cost_measurements()
        denoiser = getattr(pipe, "transformer", None)
        if measured is None or denoiser is None:
            return None
        dtype = next((p.dtype for p in denoiser.parameters() if p.is_floating_point()), None)
        dtype = getattr(denoiser, "dtype", None) or dtype
        if dtype is None or not getattr(dtype, "is_floating_point", False) or dtype.itemsize < 2:
            dtype = _torch().float16
        if dtype == _torch().float32:
            dtype = _torch().float16
        tflops = _gemm_tflops(int(selection.primary), dtype)
        if tflops is None:
            return None
        dit = _denoiser_mib(sizes)
        resident = int(getattr(plan, "resident_transformer_mib", None) or 0)
        streamed = max(0, dit - resident)
        params = logical_weight_elements(denoiser)
        transfer, compute, exposed = stream_exposure(
            streamed, params, h2d_gib_s = measured[0], tflops = tflops, side_px = _COST_CANVAS_PX
        )
    except Exception:  # noqa: BLE001 - unpriced: the split is tried as before
        return None
    try:
        _STREAM_ESTIMATES[pipe] = {
            "streamed_mib": streamed,
            "h2d_gib_s": round(measured[0], 2),
            "tflops": round(tflops, 1),
            "transfer_s": round(transfer, 3),
            "compute_s": round(compute, 3),
            "exposed": round(exposed, 3),
        }
    except TypeError:
        pass
    if exposed >= STREAM_HIDDEN_FRACTION:
        return None
    if streamed > 0:
        # Split vs whole depends on which card runs cooler, so main's split stays (no image slower
        # than main); applies when primary holds the whole denoiser.
        return None
    return (
        f"one card plus a copy on cuda:{partner}: streaming is hidden behind compute ({_gib(streamed)} per step in "
        f"{transfer:.2f} s against {compute:.2f} s, {exposed:.0%} exposed at {_COST_CANVAS_PX}px), so a split would "
        f"not be faster, while the copy splits even batches and guided steps; {STREAM_GATE_ENV}=0 tries the split"
    )


def _copy_partner(pipe: Any, selection: Any) -> Optional[int]:
    """The idle same-model card a whole copy of the denoiser fits on (:func:`automatic_partner`), or None. Sized from
    nvidia-smi for the secondaries, so no context is opened on them."""
    try:
        denoiser = getattr(pipe, "transformer", None)
        need = replica_need_mib(denoiser)
        probes = probe_devices((selection.primary, *selection.secondaries))
        primary = probes.get(int(selection.primary))
        cards = [
            probes[o]
            for o in selection.secondaries
            if probes.get(o) is not None and probes[o].free_mib is not None
        ]
        if primary is None:
            return None
        pick, _why = automatic_partner(primary, cards, need)
        return pick
    except Exception:  # noqa: BLE001 - unsized: the split is tried as before
        return None


_STREAM_ESTIMATES: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


# A concurrent CFG replica beats a serial split; 0 = split first, force = unpriced replica.
CFG_REPLICA_PLACEMENT_ENV = "UNSLOTH_DIFFUSION_CFG_REPLICA_PLACEMENT"
# The FLOP price under-reads a real forward (T4 Wan 5B: 1.13 s priced, 2.5 s measured), so the real margin is wider.
REPLICA_MIN_GAIN = 1.05
# The 1 GiB reserve already holds a block working set up to that size.
_REPLICA_BASE_RESERVE_MIB = 1024
# A step split by tokens over two cards, as a share of the one-card step (two swaps per attention included). Kaggle
# 2x T4 PCIe: Wan 2.2 5B 1.70-1.79x (0.56-0.59).
SEQUENCE_SPLIT_STEP_SHARE = 0.6


def cfg_replica_placement_mode() -> str:
    """``off`` / ``auto`` (default) / ``force``."""
    raw = (os.environ.get(CFG_REPLICA_PLACEMENT_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    if raw == "force":
        return "force"
    return "auto"


@dataclasses.dataclass(frozen = True)
class DefaultRequest:
    """``guided`` None = unknown; ``tokens`` per branch, None = the 1024px image canvas."""

    guided: Optional[bool]
    steps: Optional[int] = None
    tokens: Optional[int] = None
    why: str = ""
    # joint-sequence denoisers (FLUX.2): the prompt's tokens beside ``tokens`` image tokens
    text_tokens: Optional[int] = None


_DEFAULT_REQUESTS: "weakref.WeakKeyDictionary[Any, DefaultRequest]" = weakref.WeakKeyDictionary()


def note_default_request(
    pipe: Any,
    *,
    guided: Optional[bool],
    steps: Optional[int] = None,
    tokens: Optional[int] = None,
    why: str = "",
    text_tokens: Optional[int] = None,
) -> None:
    if pipe is None:
        return
    with contextlib.suppress(TypeError):
        _DEFAULT_REQUESTS[pipe] = DefaultRequest(
            guided = None if guided is None else bool(guided),
            steps = int(steps) if steps else None,
            tokens = int(tokens) if tokens else None,
            why = str(why or ""),
            text_tokens = int(text_tokens) if text_tokens else None,
        )


def default_request(pipe: Any) -> Optional[DefaultRequest]:
    try:
        return _DEFAULT_REQUESTS.get(pipe) if pipe is not None else None
    except TypeError:
        return None


def video_default_tokens(pipe: Any, width: int, height: int, frames: int) -> Optional[int]:
    try:
        spatial = int(getattr(pipe, "vae_scale_factor_spatial"))
        temporal = int(getattr(pipe, "vae_scale_factor_temporal"))
        config = getattr(getattr(pipe, "transformer", None), "config", None)
        patch = getattr(config, "patch_size", None)
        if isinstance(patch, (tuple, list)):
            sides = [int(p) for p in patch]
            pt, ph, pw = (1, sides[0], sides[-1]) if len(sides) < 3 else tuple(sides[-3:])
        else:
            pt, ph, pw = 1, int(patch), int(patch)
        pt = max(1, int(getattr(config, "patch_size_t", None) or pt))
        if min(spatial, temporal, ph, pw) <= 0:
            return None
        latent_frames = (max(1, int(frames)) - 1) // temporal + 1
        return (
            -(-latent_frames // pt)
            * (int(height) // (spatial * ph))
            * (int(width) // (spatial * pw))
        ) or None
    except Exception:  # noqa: BLE001 - unknown shape: the image canvas prices the step
        return None


REPLICA_LAYOUT_UNSPLITTABLE: dict[str, str] = {
    "HiDreamImageTransformer2DModel": "the pipeline batches both branches inside its llama states (dim 1), which no "
    "cut separates",
    "Ideogram4Transformer2DModel": "the negative branch runs a separate unconditional transformer",
    "MiniMaxH3Transformer3DModel": "one denoiser call per step (no guidance pair)",
    "MiniMaxH3ImageTransformer2DModel": "guidance-distilled: one denoiser call per step",
    "CosmosTransformer3DModel": "Anima's modular guider prepares the models before each branch, so the branches "
    "never overlap",
}


# Guidance split only runs while no denoiser block streams.
STREAMED_REPLICA_REFUSED = {
    "ZImageTransformer2DModel": "Z-Image splits guidance only while no group-offload block streams, and this one-card "
    "plan streams the denoiser: the replica would sit idle",
}


def _plan_streams_denoiser(plan: Any, sizes: Mapping[str, tuple[int, str]]) -> bool:
    if getattr(plan, "offload_policy", None) != "group" or not bool(
        getattr(plan, "stream_transformer", True)
    ):
        return False
    dit = int(sizes.get("transformer", (0, "dit"))[0] or _denoiser_mib(dict(sizes)))
    return dit - int(getattr(plan, "resident_transformer_mib", None) or 0) > 0


def replica_layout_refusal(denoiser: Any) -> Optional[str]:
    name = type(denoiser).__name__
    if name in REPLICA_LAYOUT_UNSPLITTABLE:
        return REPLICA_LAYOUT_UNSPLITTABLE[name]
    if name == "LTX2VideoTransformer3DModel" and not multigpu_all_enabled():
        return f"LTX-2 batches both branches in one call, and its batch split is off ({MULTIGPU_ALL_ENV}=0)"
    if name == "LTX2VideoTransformer3DModel" and _batch_split_disabled():
        # one call carries both branches: only the batch split hands half of it to the replica
        return f"LTX-2 batches both branches in one call, and its batch split is off ({BATCH_SPLIT_ENV}=0)"
    if name == "ZImageTransformer2DModel":
        from .diffusion_zimage_cfg_split import ZIMAGE_CFG_SPLIT_ENV, zimage_cfg_split_disabled
        if zimage_cfg_split_disabled():
            return f"Z-Image batches both branches in one call, and its guidance split is off ({ZIMAGE_CFG_SPLIT_ENV}=0)"
    if name == "Cosmos3OmniTransformer" and not multigpu_all_enabled():
        return f"Cosmos 3's branches pair only with {MULTIGPU_ALL_ENV} on"
    try:
        from .diffusion_cfg_batch import NOT_BATCHABLE, PIPELINE_BATCHED, STATELESS_CLASSES
    except Exception:  # noqa: BLE001
        return "the CFG policy table is unavailable"
    if name not in STATELESS_CLASSES and name not in PIPELINE_BATCHED and name not in NOT_BATCHABLE:
        return f"{name} is not a denoiser class Studio has mapped"
    return None


def _module_mib(module: Any) -> int:
    from .diffusion_memory import _storage_nbytes

    seen: set[int] = set()
    nbytes = 0
    for t in (*module.parameters(), *module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        nbytes += sum(_storage_nbytes(t))
    return int(nbytes >> 20)


def replica_working_mib(module: Any) -> int:
    """Twice the largest block, at least 1 GiB (measured 0.6 GiB klein-4B, 1.2 GiB FLUX.2-dev at 1024px)."""
    from .diffusion_layer_split_plan import _WORKING_SET_BLOCKS, _WORKING_SET_MIN_MIB

    try:
        # not the regional compile's discovery, which patches some classes
        names = set(
            getattr(module, "_repeated_blocks", None)
            or getattr(module, "_no_split_modules", None)
            or ()
        )
        largest = max(
            (_module_mib(m) for m in module.modules() if type(m).__name__ in names), default = 0
        )
    except Exception:  # noqa: BLE001 - unreadable layout: the floor
        largest = 0
    return max(int(_WORKING_SET_MIN_MIB), int(_WORKING_SET_BLOCKS) * int(largest))


def _config_int(config: Any, *names: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is None and isinstance(config, Mapping):
            value = config.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
    return 0


def _replica_tokens(module: Any) -> int:
    """Tokens per branch of ``module``'s default request, else the 1024px canvas (as :func:`replica_layout`)."""
    with contextlib.suppress(Exception):
        for pipe, request in list(_DEFAULT_REQUESTS.items()):
            if request.tokens and any(
                getattr(pipe, n, None) is module for n in ("transformer", "unet")
            ):
                return int(request.tokens)
    from .diffusion_memory import _COST_CANVAS_PX, _DIT_PIXELS_PER_TOKEN

    return (_COST_CANVAS_PX // _DIT_PIXELS_PER_TOKEN) ** 2


def replica_activation_mib(module: Any, tokens: Optional[int] = None) -> int:
    """MiB one batch-1 guidance branch holds in the widest block over the weights; 0 without a DiT width (U-Nets)."""
    try:
        config = getattr(module, "config", None)
        heads = _config_int(config, "num_attention_heads", "num_heads")
        head_dim = _config_int(config, "attention_head_dim", "head_dim")
        inner = heads * head_dim
        hidden = _config_int(config, "hidden_size", "dim", "inner_dim") or inner
        if hidden <= 0:
            return 0
        dtype = getattr(module, "dtype", None)
        if not getattr(dtype, "is_floating_point", False):
            dtype = next((p.dtype for p in module.parameters() if p.is_floating_point()), None)
        # quantized weights still compute in >= 16-bit activations
        elem = max(2, int(getattr(dtype, "itemsize", 2) or 2))
        from .diffusion_attention import block_activation_bytes

        per_query, per_key = block_activation_bytes(hidden, inner or hidden, hidden, elem)
        n = int(tokens) if tokens else _replica_tokens(module)
        return int(-(-n * (per_query + per_key) // (1 << 20)))
    except Exception:  # noqa: BLE001 - unreadable config: the weights-only need, as before
        return 0


def latent_tokens(module: Any, latent: Any) -> Optional[int]:
    """Tokens per sample for a (B, C, [F,] H, W) latent after patching, or a packed (B, N, C) one; None if unreadable."""
    try:
        shape = tuple(int(s) for s in latent.shape)
        if len(shape) == 3:
            return shape[1] or None
        if len(shape) not in (4, 5):
            return None
        config = getattr(module, "config", None)
        patch = getattr(config, "patch_size", None)
        if patch is None and isinstance(config, Mapping):
            patch = config.get("patch_size")
        if isinstance(patch, (tuple, list)):
            sides = [max(1, int(p)) for p in patch]
            pt, ph, pw = (1, sides[0], sides[-1]) if len(sides) < 3 else tuple(sides[-3:])
        else:
            pt = 1
            ph = pw = max(1, int(patch or 1))
        if len(shape) == 5:
            pt = max(1, int(getattr(config, "patch_size_t", None) or pt))
            return (-(-shape[2] // pt) * (shape[3] // ph) * (shape[4] // pw)) or None
        return ((shape[2] // ph) * (shape[3] // pw)) or None
    except Exception:  # noqa: BLE001 - unknown shape: the default request prices the replica
        return None


def replica_need_mib(
    module: Any,
    tokens: Optional[int] = None,
    *,
    activations: bool = True,
) -> int:
    """MiB beyond :func:`_replica_reserve_mib`. ``activations=False`` for user selections, priced later on the real shape."""
    weights = sum(
        t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers())
    ) // (1 << 20)
    return (
        int(weights)
        + (replica_activation_mib(module, tokens) if activations else 0)
        + max(0, replica_working_mib(module) - _REPLICA_BASE_RESERVE_MIB)
    )


def _prices_default_request(selection: Any) -> bool:
    """Only Automatic declines on the default request; explicit/forced selections are priced at replica build."""
    return bool(getattr(selection, "automatic", False)) and cfg_parallel_mode() != "force"


def replica_fits(free_mib: Optional[int], total_mib: Optional[int], need_mib: int) -> bool:
    if free_mib is None:
        return False
    return int(need_mib) + _replica_reserve_mib(free_mib, total_mib) <= int(free_mib)


def price_replica_layout(
    *,
    guided: bool,
    compute_s: float,
    primary_stream_s: float,
    steps: int,
    encoder_stream_s: float = 0.0,
    sequence_split: bool = False,
) -> tuple[float, float]:
    """(replica layout, alternative) seconds per render: steps x ``compute + primary stream`` vs steps x
    ``branches x compute``; ``encoder_stream_s`` = encoders the alternative keeps resident, streamed once per render."""
    branches = 2 if guided else 1
    # ``sequence_split``: an unguided step runs half its tokens on the replica (diffusion_sequence_parallel)
    share = SEQUENCE_SPLIT_STEP_SHARE if sequence_split and not guided else 1.0
    replica = int(steps) * (share * float(compute_s) + float(primary_stream_s)) + float(
        encoder_stream_s
    )
    alternative = int(steps) * branches * float(compute_s)
    return replica, alternative


def choose_replica_layout(
    *,
    guided: bool,
    compute_s: float,
    primary_stream_s: float,
    steps: int,
    encoder_stream_s: float = 0.0,
    min_gain: float = REPLICA_MIN_GAIN,
    sequence_split: bool = False,
) -> tuple[bool, float, float]:
    replica, alternative = price_replica_layout(
        guided = guided,
        compute_s = compute_s,
        primary_stream_s = primary_stream_s,
        steps = steps,
        encoder_stream_s = encoder_stream_s,
        sequence_split = sequence_split,
    )
    return replica > 0 and alternative >= float(min_gain) * replica, replica, alternative


def denoiser_compute_s(params: float, tokens: int, tflops: float) -> float:
    """Priced seconds of one denoiser forward: ``2 x params x tokens`` FLOPs at the card's measured GEMM rate."""
    from .diffusion_memory import _DIT_COMPUTE_EFFICIENCY
    return (
        _DIT_COMPUTE_EFFICIENCY
        * 2.0
        * float(params)
        * int(tokens)
        / (max(float(tflops), 1e-6) * 1e12)
    )


def _denoiser_compute_dtype(denoiser: Any) -> Any:
    torch = _torch()
    dtype = next((p.dtype for p in denoiser.parameters() if p.is_floating_point()), None)
    dtype = getattr(denoiser, "dtype", None) or dtype
    if dtype is None or not getattr(dtype, "is_floating_point", False) or dtype.itemsize < 2:
        dtype = torch.float16
    if dtype == torch.float32:
        dtype = torch.float16
    return dtype


def _replica_card(
    selection: Any,
    need_mib: int,
    *,
    exclude: Sequence[int] = (),
    own_mib: Optional[Mapping[int, int]] = None,
    pending_mib: Optional[Mapping[int, int]] = None,
) -> tuple[Optional[int], str]:
    """(ordinal or None, why): the secondary :func:`install_cfg_parallel` would pick, by the same rule. ``own_mib`` /
    ``pending_mib`` (ordinal -> MiB): this load's components already on / about to be placed on a card
    (:func:`automatic_partner`)."""
    pending_mib = pending_mib or {}
    probes = probe_devices((selection.primary, *selection.secondaries))
    primary = probes.get(int(selection.primary))
    cards = [
        probes[o]
        for o in selection.secondaries
        if probes.get(o) is not None and probes[o].free_mib is not None and o not in exclude
    ]
    if primary is None or not cards:
        return None, "no other card"
    if selection.automatic and cfg_parallel_mode() != "force":
        return automatic_partner(primary, cards, need_mib, own_mib = own_mib, pending_mib = pending_mib)
    if cfg_parallel_mode() != "force":
        same = [c for c in cards if c.name and c.name == primary.name]
        if not same:
            return None, "secondary card is a different model (bit-identity needs the same kernels)"
        cards = same

    def pending(c: Any) -> int:
        return max(0, int(pending_mib.get(int(c.ordinal), 0)))

    fits = [c for c in cards if replica_fits(c.free_mib, c.total_mib, need_mib + pending(c))]
    if not fits:
        best = max(cards, key = lambda c: int(c.free_mib) - pending(c))
        beside = f" beside this load's {pending(best)} MiB" if pending(best) else ""
        return (
            None,
            f"replica needs {need_mib} MiB{beside}, {best.free_mib} MiB free on cuda:{best.ordinal}",
        )
    own_mib = own_mib or {}
    best = max(
        fits,
        key = lambda c: (
            0 if pending(c) or int(c.ordinal) in own_mib else 1,
            int(c.free_mib) - _replica_reserve_mib(c.free_mib, c.total_mib) - pending(c),
            -c.ordinal,
        ),
    )
    return int(best.ordinal), "same-model card"


_MODE_ESTIMATES: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()

_ALTERNATIVE_LABELS = {
    "layer_split": "layer split",
    "companion_split": "companion split",
    "one_card": "one card",
}


def _sequence_split_layout_refusal(denoiser: Any, request: Any, selection: Any) -> Optional[str]:
    """Why a replica would not split the unguided default request by tokens (diffusion_sequence_parallel), or None.
    Only joint-sequence classes (FLUX.2) take a replica for it, and only when the default request's own size (image
    plus text tokens, recorded at load) clears the threshold: an unknown size keeps the other layouts."""
    from . import diffusion_sequence_parallel as sp

    name = type(denoiser).__name__
    if sp.mode() == "off":
        return f"{sp.SP_ENV}=0"
    if sp.read_plan(type(denoiser)) is None:
        return f"{name} has no checked sequence-parallel plan"
    if name not in sp.SP_JOINT_PREFIX:
        return f"{name} takes a replica for guided steps only"
    # Speed optims run before placement: a compiled denoiser is already visible here, and every call refuses the split.
    from .diffusion_cfg_batch import compiled_denoiser, per_call_scaled

    compiled = compiled_denoiser(denoiser)
    if compiled is not None:
        return f"compiled submodules ({compiled}) never split by tokens"
    # each token half would take its own activation amax (dynamic NVFP4): the router refuses every such call
    scaled = per_call_scaled(denoiser)
    if scaled is not None:
        return f"per-call activation scaling ({scaled}) never splits by tokens"
    if not request.tokens or not request.text_tokens:
        return "the default request's token count is unknown"
    tokens = int(request.tokens) + int(request.text_tokens)
    if sp.mode() != "force" and tokens < sp.MIN_TOKENS:
        return f"the default request has {tokens} tokens, under {sp.MIN_TOKENS}"
    return None


def replica_layout(
    pipe: Any,
    plan: Any,
    sizes: Mapping[str, tuple[int, str]],
    selection: Any,
    *,
    alternative: str,
    moved: Sequence[str] = (),
    own_mib: Optional[Mapping[int, int]] = None,
    pending_mib: Optional[Mapping[int, int]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """(why the replica wins, None), (None, why it lost), or (None, None) when it does not apply. ``own_mib`` /
    ``pending_mib``: this load's components on / about to land on each card (:func:`_replica_card`), for a replica
    beside a companion split's text encoders. Unpriced (no host-to-device or GEMM measurement yet, a fresh install):
    a guided default request whose denoiser stays resident on the primary takes the replica, since each step then
    runs one forward per card instead of two on one; a primary that streams its denoiser keeps the old order."""
    mode = cfg_replica_placement_mode()
    if mode == "off" or cfg_parallel_mode() == "off" or selection is None:
        return None, None
    if multigpu_mode() == "force" and mode != "force":
        return None, None
    if selection.automatic and cfg_parallel_mode() != "force" and automatic_cfg_parallel_off():
        return None, None
    torch = _torch()
    denoiser = getattr(pipe, "transformer", None)
    if not isinstance(denoiser, torch.nn.Module):
        return None, None
    request = default_request(pipe)
    if request is None or request.guided is None:
        return None, None
    if not request.guided and mode != "force":
        why_not = _sequence_split_layout_refusal(denoiser, request, selection)
        if why_not is not None:
            return None, (
                f"the default request runs one denoiser branch per step ({request.why or 'no CFG'}): a copy would "
                f"only split even batches ({why_not})"
            )
    if getattr(pipe, "transformer_2", None) is not None:
        return None, "two-expert denoiser"
    refusal = replica_layout_refusal(denoiser)
    if refusal is not None:
        return None, refusal
    policy = getattr(plan, "offload_policy", None)
    if policy not in ("none", "group"):
        return None, f"the one-card plan pages the denoiser per call ('{policy}')"
    if type(denoiser).__name__ in STREAMED_REPLICA_REFUSED and _plan_streams_denoiser(plan, sizes):
        return None, STREAMED_REPLICA_REFUSED[type(denoiser).__name__]
    from .diffusion_bg_compile import load_time_enabled as bg_compile_enabled

    if bg_compile_enabled():
        return (
            None,
            "background compile replays through the denoiser (UNSLOTH_DIFFUSION_BG_COMPILE)",
        )
    try:
        need = replica_need_mib(denoiser, activations = _prices_default_request(selection))
        partner, why = _replica_card(selection, need, own_mib = own_mib, pending_mib = pending_mib)
    except Exception as exc:  # noqa: BLE001 - unsized: the old order
        return None, f"replica card unknown ({type(exc).__name__})"
    if partner is None:
        return None, f"no card for a replica ({why})"
    if not request.guided:
        # the router refuses the same pairs at every call
        from . import diffusion_sequence_parallel as sp

        probes = probe_devices((int(selection.primary), int(partner)))

        def probed(device: Any) -> Optional[tuple]:
            probe = probes.get(device.index)
            return (probe.name,) if probe is not None and probe.name else None

        unpaired = sp.pair_refusal(
            f"cuda:{int(selection.primary)}",
            f"cuda:{int(partner)}",
            type(denoiser).__name__,
            identify = probed,
        )
        if unpaired is not None:
            return None, f"no token split on this pair ({unpaired})"
    label = _ALTERNATIVE_LABELS.get(alternative, alternative)
    if alternative == "one_card" or mode == "force":
        runs = (
            "the default request is guided, so each step runs its second branch on the replica at the same time"
            if request.guided
            else "the default request has one branch per step, so each step runs half its tokens on the replica"
        )
        return (
            f"one card plus a CFG replica on cuda:{partner} ({need} MiB fits there; {runs})"
            + (
                ""
                if alternative == "one_card"
                else f"; {CFG_REPLICA_PLACEMENT_ENV}=force over the {label}"
            )
        ), None
    try:
        from .diffusion_memory import (
            _COST_CANVAS_PX,
            _DIT_PIXELS_PER_TOKEN,
            _offload_cost_measurements,
        )
        measured = _offload_cost_measurements()
        tflops = _gemm_tflops(int(selection.primary), _denoiser_compute_dtype(denoiser))
    except Exception:  # noqa: BLE001
        measured, tflops = None, None
    dit = int(sizes.get("transformer", (0, "dit"))[0] or _denoiser_mib(dict(sizes)))
    streamed = 0
    if policy == "group" and bool(getattr(plan, "stream_transformer", True)):
        streamed = max(0, dit - int(getattr(plan, "resident_transformer_mib", None) or 0))
    if measured is None or tflops is None:
        if streamed == 0 and (dit > 0 or policy == "none"):
            with contextlib.suppress(TypeError):
                _MODE_ESTIMATES[pipe] = {
                    "alternative": alternative,
                    "replica_card": f"cuda:{partner}",
                    "replica_need_mib": int(need),
                    "priced": False,
                    "chosen": "cfg_replica",
                }
            return (
                f"one card plus a CFG replica on cuda:{partner} ({need} MiB fits there): unpriced (no host-to-device "
                f"or GEMM measurement on this card yet), but the default request is guided and the denoiser stays "
                f"resident on cuda:{selection.primary}, so each step runs one forward per card instead of two on one; "
                f"{CFG_REPLICA_PLACEMENT_ENV}=0 keeps the {label}"
            ), None
        return (
            None,
            "unpriced (no host-to-device or GEMM measurement on this card) and the one-card plan streams the "
            "denoiser: the split order stays",
        )
    h2d = float(measured[0])
    tokens = int(request.tokens or (_COST_CANVAS_PX // _DIT_PIXELS_PER_TOKEN) ** 2)
    params = float(logical_weight_elements(denoiser))
    compute = denoiser_compute_s(params, tokens, tflops)
    stream_s = streamed / 1024.0 / max(h2d, 1e-6)
    encoders_s = 0.0
    if alternative == "companion_split" and bool(getattr(plan, "stream_text_encoders", False)):
        te = sum(int(sizes[n][0]) for n in moved if n in sizes and sizes[n][1] == "text_encoder")
        encoders_s = te / 1024.0 / max(h2d, 1e-6)
    steps = int(request.steps or 20)
    take, replica_s, alt_s = choose_replica_layout(
        guided = bool(request.guided),
        compute_s = compute,
        primary_stream_s = stream_s,
        steps = steps,
        encoder_stream_s = encoders_s,
        sequence_split = not request.guided,
    )
    with contextlib.suppress(TypeError):
        _MODE_ESTIMATES[pipe] = {
            "alternative": alternative,
            "replica_card": f"cuda:{partner}",
            "replica_need_mib": int(need),
            "tokens": tokens,
            "steps": steps,
            "compute_s": round(compute, 3),
            "primary_streamed_mib": int(streamed),
            "primary_stream_s": round(stream_s, 3),
            "encoder_stream_s": round(encoders_s, 3),
            "h2d_gib_s": round(h2d, 2),
            "tflops": round(float(tflops), 1),
            "replica_render_s": round(replica_s, 2),
            "alternative_render_s": round(alt_s, 2),
            "chosen": "cfg_replica" if take else alternative,
        }
    priced = (
        f"{replica_s / steps:.2f} s/step with the replica ({_gib(streamed)} streamed per step on cuda:"
        f"{selection.primary}) against {alt_s / steps:.2f} s/step for the {label} over {steps} steps at {tokens} "
        f"tokens ({h2d:.1f} GiB/s, {float(tflops):.0f} TFLOPS measured)"
    )
    if take:
        runs = (
            "the default request is guided, so the replica runs the second branch beside the first"
            if request.guided
            else "the default request has one branch per step, so the replica runs half of each step's tokens"
        )
        return (
            f"one card plus a CFG replica on cuda:{partner}: {runs}; predicted {priced}; "
            f"{CFG_REPLICA_PLACEMENT_ENV}=0 keeps the {label}"
        ), None
    return None, f"predicted slower or within {REPLICA_MIN_GAIN:.2f}x: {priced}"


def _weakrefable(obj: Any) -> bool:
    try:
        weakref.ref(obj)
        return True
    except TypeError:
        return False


def _split_moved_names(
    pipe: Any,
    plan: Any,
    sizes: Optional[Mapping[str, tuple[int, str]]] = None,
) -> tuple[str, ...]:
    """The components a planned companion split moves (``place_split``'s rule for None / ``TEXT_ENCODERS_ONLY``);
    ``sizes`` (the loaded component sizes) names them when the pipeline lists no components."""
    names = getattr(plan, "secondary_components", None)
    companions = companion_names(pipe) or tuple(
        n for n, (_mib, role) in (sizes or {}).items() if role != "dit" and n not in DENOISER_ATTRS
    )
    if names and tuple(names) == TEXT_ENCODERS_ONLY:
        return tuple(n for n in companions if n.startswith("text_encoder"))
    return tuple(names) if names else companions


def _replica_shares_component(name: str) -> bool:
    """A replica may sit beside this component of its own load: a text encoder runs before the first denoiser step,
    never beside it (a VAE decodes on its card after the last step and needs that card's headroom)."""
    return str(name).startswith("text_encoder")


def _own_mib_on_cards(pipe: Any, names: Sequence[str]) -> dict[int, int]:
    """ordinal -> MiB of ``names`` (this load's components) resident on each CUDA card now, counted per tensor, so a
    component spread over cards (or partly on the host) is credited only where its bytes are."""
    torch = _torch()
    from .diffusion_memory import _storage_nbytes

    nbytes: dict[int, int] = {}
    seen: set[int] = set()
    components = getattr(pipe, "components", {}) or {}
    for name in names:
        module = components.get(name) if isinstance(components, Mapping) else None
        if not isinstance(module, torch.nn.Module):
            continue
        try:
            tensors = (*module.parameters(), *module.buffers())
        except Exception:  # noqa: BLE001 - unreadable: not counted
            continue
        for t in tensors:
            if id(t) in seen:
                continue
            seen.add(id(t))
            device = getattr(t, "device", None)
            if getattr(device, "type", None) != "cuda" or device.index is None:
                continue
            try:
                size = sum(_storage_nbytes(t))
            except Exception:  # noqa: BLE001
                continue
            nbytes[int(device.index)] = nbytes.get(int(device.index), 0) + int(size)
    return {o: b >> 20 for o, b in nbytes.items()}


_ENCODER_WORKING: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()


def note_encoder_working_mib(pipe: Any, mib: Optional[int]) -> None:
    """The encode peak (MiB above the weights) the split priced this load's text encoder card with."""
    if pipe is None or not mib:
        return
    with contextlib.suppress(TypeError):
        _ENCODER_WORKING[pipe] = int(mib)


def encoder_working_mib(pipe: Any) -> int:
    """This load's text encoder working memory: what its split was priced with, at least the encoder headroom."""
    try:
        noted = _ENCODER_WORKING.get(pipe) if pipe is not None else None
    except TypeError:
        noted = None
    return max(TEXT_ENCODER_HEADROOM_MIB, int(noted or 0))


def beside_encoder_extra_mib(
    pipe: Any,
    denoiser: Any,
    need_mib: int,
    reserve_mib: int = _REPLICA_BASE_RESERVE_MIB,
) -> int:
    """MiB a replica on the card of this load's text encoders must leave free beyond its own need and reserve. The
    replica's weights stay between renders, so the next uncached prompt encodes beside them: that encode may reuse the
    replica's working set and the reserve (both idle while it runs), and needs only what its peak exceeds them by."""
    try:
        weights = _module_mib(denoiser)
    except Exception:  # noqa: BLE001 - unsized: the whole need counts as weights
        weights = int(need_mib)
    working = max(0, int(need_mib) - int(weights))
    return max(0, encoder_working_mib(pipe) - working - int(reserve_mib))


def _replica_over_planned_split(
    pipe: Any,
    plan: Any,
    selection: Any,
    split_summary: str,
    logger: Any = None,
) -> tuple[Any, list[PlacementCandidate]]:
    """(plan to run, []) when a CFG replica replaces or joins a companion split planned before the load, else
    (None, the rejected replica candidates). Two layouts, the first that applies wins:

    - beside the encoders: the split moves only text encoders, and the replica fits their card beside them (they run
      before the first step, the replica during the steps). The split plan is kept; the primary stays resident.
    - instead of the split: the one-card plan the split replaced (``plan.fallback_plan``) plus a replica on an idle
      card, priced like the post-load companion split (:func:`replica_layout`, which charges the encoders that plan
      streams once per render).

    Without either, the split stays (the order before), and the reasons go to the status."""
    rejected: list[PlacementCandidate] = []
    try:
        from .diffusion_memory import _loaded_component_mib
        sizes = dict(_loaded_component_mib(pipe) or {})
    except Exception:  # noqa: BLE001 - unsized: the split stays
        return None, rejected
    moved = _split_moved_names(pipe, plan, sizes)
    secondary = int(plan.secondary_ordinal)
    label = _ALTERNATIVE_LABELS["companion_split"]
    if moved and all(_replica_shares_component(n) for n in moved):
        own = _own_mib_on_cards(pipe, moved)
        moved_mib = sum(int(sizes[n][0]) for n in moved if n in sizes)
        note_encoder_working_mib(
            pipe, (getattr(plan, "estimates", None) or {}).get("multigpu_secondary_headroom_mib")
        )
        extra = 0
        with contextlib.suppress(Exception):  # unsized here: replica_layout reports it
            denoiser = getattr(pipe, "transformer", None)
            extra = beside_encoder_extra_mib(
                pipe,
                denoiser,
                replica_need_mib(denoiser, activations = _prices_default_request(selection)),
            )
        pending = {secondary: max(0, moved_mib - int(own.get(secondary, 0))) + extra}
        replica, why = replica_layout(
            pipe,
            plan,
            sizes,
            selection,
            alternative = "companion_split",
            moved = moved,
            own_mib = own,
            pending_mib = pending,
        )
        if replica is not None:
            prefix = f"one card plus a CFG replica on cuda:{secondary}"
            on_split_card = replica.startswith(prefix) and not replica[len(prefix) :][:1].isdigit()
            beside = (
                f"the replica sits beside {', '.join(moved)} on cuda:{secondary}; "
                if on_split_card
                else ""
            )
            rejected.append(
                PlacementCandidate(
                    "companion_split",
                    f"the {label} runs both guidance branches on cuda:{selection.primary}; kept for the encoders, "
                    "with a CFG replica for the second branch",
                )
            )
            _record_decision(
                pipe,
                selection,
                "cfg_replica",
                f"{replica}; {beside}{split_summary}",
                rejected,
                fallback_resident = False,
                fallback_kind = "companion_split",
                fallback_summary = split_summary,
            )
            if logger is not None:
                logger.info("diffusion.multigpu: %s; %s(the split stays placed)", replica, beside)
            return plan, []
        if why:
            rejected.append(
                PlacementCandidate("cfg_replica", f"beside the encoders on cuda:{secondary}: {why}")
            )
    fallback = getattr(plan, "fallback_plan", None)
    if (
        fallback is None
        or getattr(fallback, "secondary_ordinal", None) is not None
        or getattr(fallback, "layer_split", None) is not None
    ):
        return None, rejected
    replica, why = replica_layout(
        pipe, fallback, sizes, selection, alternative = "companion_split", moved = moved
    )
    if replica is None:
        if why:
            rejected.append(PlacementCandidate("cfg_replica", f"instead of the {label}: {why}"))
        return None, rejected
    rejected.append(
        PlacementCandidate(
            "companion_split",
            f"the {label} would fit, but the CFG replica layout is predicted faster (one forward per guided step "
            "instead of two)",
        )
    )
    _record_decision(
        pipe,
        selection,
        "cfg_replica",
        f"{replica}; cuda:{selection.primary} runs its one-card '{fallback.offload_policy}' plan",
        rejected,
        fallback_resident = getattr(fallback, "offload_policy", None) == "none",
    )
    if logger is not None:
        logger.info("diffusion.multigpu: %s (instead of the %s)", replica, label)
    return fallback, []


def refine_plan_for_multigpu(
    pipe: Any,
    plan: Any,
    logger: Any = None,
    *,
    card_reserve_mib: Optional[Mapping[int, int]] = None,
) -> Any:
    """Post-load: an offloading plan becomes a resident split when the LOADED components can be shared by the cards.
    ``card_reserve_mib`` (ordinal -> MiB): memory a sidecar process holds on a card, uncounted by the loaded sizes.

    No-op without a selection, for explicit offload modes, a plan already resident (unless forced) or split, unified memory
    or an unknown budget. When no component assignment fits, a layer split (diffusion_layer_split_plan) is tried before
    keeping the offload plan. Returns the plan (a ``dataclasses.replace`` copy when it changed). Under a selection the
    decision and every rejected candidate are recorded for :func:`placement_status`."""
    selection = current_selection()
    if selection is None or plan is None:
        return plan
    primary = int(selection.primary)
    with contextlib.suppress(TypeError):
        _MODE_ESTIMATES.pop(pipe, None)

    def keep(chosen: str, summary: str, *rejected: PlacementCandidate) -> Any:
        fallback = getattr(plan, "fallback_plan", None)
        # without its replica, a cfg_replica load is this plan itself
        resident = fallback if chosen != "cfg_replica" else plan
        _record_decision(
            pipe,
            selection,
            chosen,
            summary,
            (*planned_rejected, *rejected),
            fallback_resident = getattr(resident, "offload_policy", None) == "none",
        )
        return plan

    def one_card_plan(why: str) -> Any:
        if plan.offload_policy == "none":
            return keep("one_card", f"one card: everything resident on cuda:{primary} ({why})")
        return keep(
            "offload", f"one card with '{plan.offload_policy}' offload on cuda:{primary} ({why})"
        )

    planned_rejected: list[PlacementCandidate] = []
    if getattr(plan, "secondary_ordinal", None) is not None:
        refused = placement_refusal(
            pipe, _split_moved_names(pipe, plan), primary, int(plan.secondary_ordinal)
        )
        fallback = getattr(plan, "fallback_plan", None)
        if refused is not None and fallback is not None:
            # planned before the load from sizes; the loaded weights cannot run there: plan again from the one-card plan
            planned_rejected.append(PlacementCandidate("companion_split", refused))
            if logger is not None:
                logger.info("diffusion.multigpu: planned companion split dropped (%s)", refused)
            plan = fallback
    if getattr(plan, "secondary_ordinal", None) is not None:
        split_summary = (
            f"companion split: {', '.join(plan.secondary_components or ('companions',))} on "
            f"cuda:{plan.secondary_ordinal}, the denoiser resident on cuda:{primary} (planned before the load)"
        )
        replaced, replica_rejected = _replica_over_planned_split(
            pipe, plan, selection, split_summary, logger
        )
        if replaced is not None:
            return replaced
        return keep("companion_split", split_summary, *replica_rejected)
    if getattr(plan, "layer_split", None) is not None:
        return keep("layer_split", f"layer split across {_cards(plan.layer_split.cards)}")
    from .diffusion_layer_split_plan import layer_split_mode, refine_plan_for_layer_split

    layer_force = layer_split_mode() == "force"
    if plan.offload_policy == "none" and multigpu_mode() != "force" and not layer_force:
        required = plan.estimates.get("resident_required_mib")
        budget = plan.estimates.get("safe_device_budget_mib")
        not_needed = f"not needed: the model fits cuda:{primary}"
        resident = f"everything resident on cuda:{primary} (needs {_gib(required)}, {_gib(budget)} free after the reserve)"
        unneeded = (
            PlacementCandidate("companion_split", not_needed),
            PlacementCandidate("layer_split", not_needed),
        )
        replica, replica_why = replica_layout(pipe, plan, {}, selection, alternative = "one_card")
        if replica is not None:
            return keep("cfg_replica", f"{replica}; {resident}", *unneeded)
        extra = (PlacementCandidate("cfg_replica", replica_why),) if replica_why else ()
        return keep("one_card", f"one card: {resident}", *unneeded, *extra)
    if getattr(plan, "requested_mode", None) not in ("auto", "fast"):
        return one_card_plan(
            f"memory mode '{getattr(plan, 'requested_mode', None)}' was chosen explicitly"
        )
    if plan.estimates.get("explicit_cpu_offload"):
        return one_card_plan("CPU offload was requested explicitly")
    memory = plan.device_memory
    if memory.is_unified or getattr(memory, "device", None) != "cuda":
        if memory.is_unified:
            from .diffusion_layer_split_plan import note_unified_primary
            note_unified_primary(pipe, selection, logger)
        return one_card_plan("unified memory" if memory.is_unified else "not a CUDA card")
    budget = plan.estimates.get("safe_device_budget_mib")
    if budget is None:
        return one_card_plan(f"free memory of cuda:{primary} is unknown")
    from .diffusion_memory import _loaded_component_mib

    sizes = _loaded_component_mib(pipe)
    if not sizes:
        return one_card_plan("component sizes are unreadable")
    mode = getattr(plan, "requested_mode", None)
    secondary = best_secondary(selection, mode = mode)
    if secondary is None:
        return one_card_plan("no second card")
    reserve = {int(o): max(0, int(m)) for o, m in (card_reserve_mib or {}).items()}

    def card_limit(ordinal: int) -> Optional[int]:
        limit = secondary_resident_limit_mib(ordinal, mode)
        return None if limit is None else int(limit) - reserve.get(int(ordinal), 0)

    sec_limit = card_limit(secondary)
    if sec_limit is None:
        return one_card_plan(f"free memory of cuda:{secondary} is unknown")
    from .diffusion_memory import _fast_device_budget_mib

    primary_limit = (
        _fast_device_budget_mib(memory) if mode == "fast" else int(int(budget) * _RESIDENT_FRACTION)
    )
    if primary_limit is None:
        return one_card_plan(f"free memory of cuda:{primary} is unknown")
    from .diffusion_memory import plan_keeps_transformer_resident

    if mode != "fast" and plan.offload_policy != "none" and plan_keeps_transformer_resident(plan):
        # Denoiser stays resident on one card, so the 85% margin must not refuse a lighter split.
        primary_limit = max(int(primary_limit), int(budget))
    primary_limit = int(primary_limit) - reserve.get(primary, 0)
    measured_headroom = plan.estimates.get("measured_runtime_headroom_mib")
    flat_headroom = (
        plan.estimates.get("calibrated_headroom_mib")
        or plan.estimates.get("calibrated_split_headroom_mib")
        or plan.estimates.get("runtime_headroom_mib")
    )
    headroom = int(measured_headroom or flat_headroom or 0)
    if measured_headroom:
        headroom += int(plan.estimates.get("lora_activation_mib") or 0)
    if plan.estimates.get("measured_omits_references"):
        # A split has no resident groups to stream back, so a reference request needs the flat headroom.
        headroom = max(headroom, int(flat_headroom or 0))
    overhead = int(plan.estimates.get("base_overhead_mib") or 0)
    rejected: list[PlacementCandidate] = list(planned_rejected)
    if plan.offload_policy != "none":
        rejected.append(_one_card_rejection(plan, primary))

    def replica_instead(alternative: str, moved: Sequence[str] = ()) -> bool:
        replica, why = replica_layout(
            pipe, plan, sizes, selection, alternative = alternative, moved = moved
        )
        if replica is None:
            if why:
                rejected.append(PlacementCandidate("cfg_replica", why))
            return False
        label = _ALTERNATIVE_LABELS.get(alternative, alternative)
        request = default_request(pipe)
        per_step = (
            "half of each step's tokens on each card"
            if request is not None and request.guided is False
            else "one forward per guided step instead of two"
        )
        rejected.append(
            PlacementCandidate(
                alternative,
                f"the {label} would fit, but the CFG replica layout is predicted faster ({per_step})",
            )
        )
        _record_decision(
            pipe,
            selection,
            "cfg_replica",
            f"{replica}; cuda:{primary} runs its one-card '{plan.offload_policy}' plan",
            rejected,
            fallback_resident = plan.offload_policy == "none",
        )
        if logger is not None:
            logger.info("diffusion.multigpu: %s", replica)
        return True

    def layer_split_or_plan() -> Any:
        if not layer_force and replica_instead("layer_split"):
            return plan
        hidden = None if layer_force else _hidden_stream_reason(pipe, plan, sizes, selection)
        if hidden is not None:
            rejected.append(PlacementCandidate("layer_split", hidden))
            _record_decision(
                pipe,
                selection,
                "offload",
                f"one card with '{plan.offload_policy}' offload on cuda:{primary} ({hidden})",
                rejected,
            )
            if logger is not None:
                logger.info("diffusion.multigpu: layer split not used (%s)", hidden)
            return plan

        def split_with(primary_mib: int, card_mib: Callable[[int], Optional[int]], log: Any) -> Any:
            return refine_plan_for_layer_split(
                pipe,
                plan,
                selection = selection,
                sizes = sizes,
                primary_limit_mib = int(primary_mib),
                secondary_limit_mib = card_mib,
                headroom_mib = headroom,
                overhead_mib = overhead,
                logger = log,
            )

        # Retry at the primary's full safe budget if the 85% margin fails; secondaries keep 85%.
        safe_primary = max(int(primary_limit), int(budget) - reserve.get(primary, 0))
        roomier = safe_primary > int(primary_limit)
        from .diffusion_layer_split_plan import _DECLINES, FIT_DECLINE, _record_decline

        with contextlib.suppress(TypeError):
            _DECLINES.pop(pipe, None)  # what the first try records is this attempt's own decline
        refined = split_with(int(primary_limit), card_limit, None if roomier else logger)
        if getattr(refined, "layer_split", None) is None and roomier:
            try:
                declined = str(_DECLINES.get(pipe) or "")
            except TypeError:
                declined = ""
            if declined.startswith(FIT_DECLINE):
                refined = split_with(safe_primary, card_limit, logger)
            elif declined:
                _record_decline(pipe, declined, logger)  # the first try's decline, logged once
        split = getattr(refined, "layer_split", None)
        if split is not None:
            _record_decision(
                pipe,
                selection,
                "layer_split",
                f"layer split: denoiser blocks across {_cards(split.cards)} (denoiser "
                f"{_gib(_denoiser_mib(sizes))} does not fit one card with its working memory)",
                rejected,
                fallback_resident = plan.offload_policy == "none",
            )
            return refined
        from .diffusion_layer_split_plan import (
            LAYER_SPLIT_ENV,
            engine_available,
            layer_split_status,
            plan_offloads_denoiser,
        )

        if layer_split_mode() == "off":
            declined = f"{LAYER_SPLIT_ENV}=0"
        elif not engine_available():
            declined = "the layer split engine is unavailable"
        elif not plan_offloads_denoiser(plan) and not layer_force:
            declined = f"not needed: the denoiser stays resident on cuda:{primary}"
        else:
            declined = (layer_split_status(pipe) or {}).get("declined") or "no split fits"
        rejected.append(PlacementCandidate("layer_split", str(declined)))
        if plan.offload_policy == "none":
            # a forced split of a model one card already holds: the plan returned is that resident one-card plan
            _record_decision(
                pipe,
                selection,
                "one_card",
                f"one card: everything resident on cuda:{primary} (no forced split fits)",
                rejected,
            )
        else:
            _record_decision(
                pipe,
                selection,
                "offload",
                f"one card with '{plan.offload_policy}' offload on cuda:{primary} (no split fits)",
                rejected,
            )
        return refined

    if layer_force:
        return layer_split_or_plan()
    move_denoisers = _secondary_runs_denoiser(pipe, selection, secondary, sizes, logger)
    encoders_limit = encoder_card_limit_mib(secondary)
    trials = component_assignment_trials(
        sizes,
        headroom_mib = headroom,
        overhead_mib = overhead,
        primary_limit_mib = int(primary_limit),
        secondary_limit_mib = int(sec_limit),
        move_denoisers = move_denoisers,
        encoders_only_limit_mib = (
            None if encoders_limit is None else int(encoders_limit) - reserve.get(int(secondary), 0)
        ),
    )
    # the weights each trial moves must run on the secondary (a lower-capability card of a mixed pair)
    refusals = {
        trial.moved: placement_refusal(pipe, trial.moved, primary, secondary) for trial in trials
    }
    winner = next((t for t in trials if t.fits and refusals.get(t.moved) is None), None)
    moved = winner.moved if winner is not None else None
    for trial in trials:
        if trial is winner:
            break
        why = refusals.get(trial.moved) if trial.fits else None
        rejected.append(
            PlacementCandidate("companion_split", why or trial.why_not(primary, secondary))
        )
    if moved is None:
        if not trials:
            rejected.append(
                PlacementCandidate("companion_split", "no denoiser among the loaded components")
            )
        return layer_split_or_plan()
    if replica_instead("companion_split", moved):
        return plan
    reason = (
        f"multi-GPU (loaded sizes): {', '.join(moved)} on cuda:{secondary}, the rest resident on "
        f"cuda:{selection.primary}; single card would use '{plan.offload_policy}'"
    )
    if logger is not None:
        logger.info("diffusion.multigpu: %s", reason)
    on_primary = winner.primary_need_mib
    _record_decision(
        pipe,
        selection,
        "companion_split",
        f"companion split: {', '.join(moved)} on cuda:{secondary} (denoiser {_gib(_denoiser_mib(sizes))} fits "
        f"cuda:{primary}, {_gib(on_primary)} of {_gib(int(primary_limit))} with its working memory; "
        f"{', '.join(moved)} {_gib(winner.moved_mib)} does not fit beside it)",
        rejected,
        fallback_resident = plan.offload_policy == "none",
    )
    return dataclasses.replace(
        plan,
        offload_policy = "none",
        # Kept: the fit above may be priced with the tiled decode peak (calibrated_headroom_mib), so the savers the
        # single-card plan chose must stay on.
        vae_tiling = plan.vae_tiling,
        vae_slicing = plan.vae_slicing,
        stream_text_encoders = False,
        stream_transformer = True,
        resident_transformer_mib = None,
        resident_text_encoder_mib = None,
        reasons = tuple(plan.reasons) + (reason,),
        secondary_ordinal = secondary,
        secondary_components = tuple(moved),
        fallback_plan = plan,
    )


def _cards(ordinals: Sequence[int]) -> str:
    return ", ".join(f"cuda:{int(o)}" for o in ordinals)


def encoder_card_limit_mib(ordinal: int) -> Optional[int]:
    """What a secondary that only runs text encoders may hold: free memory minus ComfyUI's per-card floor (the layer
    split prices its block-only cards the same way). None when the card cannot be read."""
    from .diffusion_layer_split_plan import secondary_reserve_mib
    from .diffusion_memory import _user_budget_reserve_mib

    free, total = _free_total_mib(int(ordinal))
    if free is None:
        return None
    return max(0, int(free) - max(secondary_reserve_mib(), _user_budget_reserve_mib(total or free)))


def _replica_reserve_mib(free: Optional[int], total: Optional[int]) -> int:
    """Room a CFG replica leaves free on its card: 1 GiB, or what the user's VRAM Budget keeps free when larger."""
    from .diffusion_memory import _user_budget_reserve_mib
    return max(1024, _user_budget_reserve_mib(total or free))


def secondary_device_memory(ordinal: int) -> Any:
    from .diffusion_memory import DeviceMemory

    free, total = _free_total_mib(int(ordinal))
    if free is None:
        return None
    return DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "discrete_vram", free_mib = free, total_mib = total
    )


def secondary_safe_budget_mib(ordinal: int) -> Optional[int]:
    from .diffusion_memory import _safe_device_budget_mib
    memory = secondary_device_memory(ordinal)
    return None if memory is None else _safe_device_budget_mib(memory)


def secondary_resident_limit_mib(ordinal: int, mode: str) -> Optional[int]:
    from .diffusion_memory import _fast_device_budget_mib

    if mode == "fast":
        memory = secondary_device_memory(ordinal)
        return None if memory is None else _fast_device_budget_mib(memory)
    budget = secondary_safe_budget_mib(ordinal)
    return None if budget is None else int(int(budget) * _RESIDENT_FRACTION)


def _torch():
    import torch
    return torch


def move_to_device(
    obj: Any,
    device: Any,
    _depth: int = 0,
    *,
    tensor_fn: Optional[Callable[[Any], Any]] = None,
    objects: bool = True,
) -> Any:
    """Tensors in ``obj`` (containers, dataclasses, ModelOutputs, plain attribute holders) on ``device``.
    ``tensor_fn(t)`` replaces ``t.to(device)`` for a tensor that is elsewhere. ``objects=False`` leaves dataclasses
    and attribute holders as they are (a per-layer KV cache the callee writes into must keep its identity)."""
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        return obj.to(device) if tensor_fn is None else tensor_fn(obj)
    if (
        _depth > 6
        or obj is None
        or isinstance(obj, (str, bytes, int, float, bool, torch.device, torch.dtype))
    ):
        return obj
    if isinstance(obj, dict):
        # ModelOutput is an OrderedDict that also mirrors its keys as attributes; __setitem__ keeps both in sync.
        try:
            out = copy.copy(obj)
            for key, value in obj.items():
                moved = move_to_device(
                    value, device, _depth + 1, tensor_fn = tensor_fn, objects = objects
                )
                if moved is not value:
                    out[key] = moved
            return out
        except Exception:  # noqa: BLE001
            return type(obj)(
                (k, move_to_device(v, device, _depth + 1, tensor_fn = tensor_fn, objects = objects))
                for k, v in obj.items()
            )
    if isinstance(obj, tuple):
        moved = [
            move_to_device(v, device, _depth + 1, tensor_fn = tensor_fn, objects = objects) for v in obj
        ]
        if all(a is b for a, b in zip(moved, obj)):
            return obj
        if hasattr(obj, "_fields"):
            return type(obj)(*moved)
        return type(obj)(moved) if type(obj) is not tuple else tuple(moved)
    if isinstance(obj, list):
        return [
            move_to_device(v, device, _depth + 1, tensor_fn = tensor_fn, objects = objects) for v in obj
        ]
    if type(obj).__name__ == "BlockMask" and callable(getattr(obj, "to", None)):
        return obj.to(device)
    if isinstance(obj, torch.nn.Module) or not objects:
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name, None)
            moved = move_to_device(value, device, _depth + 1, tensor_fn = tensor_fn, objects = objects)
            if moved is not value:
                changes[f.name] = moved
        if not changes:
            return obj
        try:
            return dataclasses.replace(obj, **changes)
        except Exception:  # noqa: BLE001 - init=False fields
            out = copy.copy(obj)
            for k, v in changes.items():
                object.__setattr__(out, k, v)
            return out
    state = getattr(obj, "__dict__", None)
    if (
        isinstance(state, dict)
        and state
        and any(isinstance(v, torch.Tensor) for v in state.values())
    ):
        if all(
            isinstance(
                v, (torch.Tensor, bool, int, float, str, type(None), torch.dtype, torch.device)
            )
            for v in state.values()
        ):
            out = copy.copy(obj)
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    setattr(out, k, v.to(device) if tensor_fn is None else tensor_fn(v))
            return out
    return obj


def _module_device(module: Any) -> Any:
    torch = _torch()
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return None


def _execution_device(module: Any) -> Any:
    """The card ``module`` computes on: its first tensor's device, or, for a group-offloaded denoiser whose first
    tensors sit in host memory between calls, the CUDA card its offload groups load onto."""
    device = _module_device(module)
    if device is not None and getattr(device, "type", None) == "cuda":
        return device
    try:
        from .diffusion_memory import _offload_groups
        for group in _offload_groups(module):
            onload = getattr(group, "onload_device", None)
            if onload is not None and getattr(onload, "type", None) == "cuda":
                onload = _torch().device(onload)
                if onload.index is None:
                    # Bare "cuda" onloads: name the card so replica search compares ordinals.
                    onload = _torch().device("cuda", _torch().cuda.current_device())
                return onload
    except Exception:  # noqa: BLE001 - unknown: the parameter device stands
        pass
    return device


def _device_context(device: Any):
    torch = _torch()
    if getattr(device, "type", None) == "cuda" and device.index is not None:
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _crosses(device: Any, home: Any) -> bool:
    """A call must move tensors: the module sits on a different accelerator card than the pipeline's home. A module on
    the CPU (offloaded, or between loads) is left to its own hooks."""
    return device is not None and device != home and device.type == "cuda"


def install_boundary(module: Any, home: Any) -> list[str]:
    """Wrap ``module``'s entry points: arguments go to the module's CURRENT card, results come back to ``home``."""
    torch = _torch()
    home = torch.device(home)
    wrapped_names: list[str] = []
    for name in _ENTRY_METHODS:
        bound = getattr(module, name, None)
        if not callable(bound) or getattr(bound, "_unsloth_boundary", None) is not None:
            continue

        def _make(orig: Callable[..., Any]) -> Callable[..., Any]:
            def boundary(*args: Any, **kwargs: Any) -> Any:
                device = _module_device(module)
                if not _crosses(device, home):
                    return orig(*args, **kwargs)
                args, kwargs = move_to_device((args, kwargs), device)
                with _device_context(device):
                    out = orig(*args, **kwargs)
                return move_to_device(out, home)

            boundary._unsloth_boundary = orig  # type: ignore[attr-defined]
            boundary.__wrapped__ = orig  # type: ignore[attr-defined]
            boundary._unsloth_had_override = name in module.__dict__  # type: ignore[attr-defined]
            return boundary

        setattr(module, name, _make(bound))
        wrapped_names.append(name)
    return wrapped_names


def remove_boundary(module: Any) -> None:
    for name in _ENTRY_METHODS:
        current = module.__dict__.get(name) if hasattr(module, "__dict__") else None
        if current is not None and getattr(current, "_unsloth_boundary", None) is not None:
            if getattr(current, "_unsloth_had_override", False):
                module.__dict__[name] = current._unsloth_boundary
                continue
            try:
                delattr(module, name)
            except AttributeError:
                pass


_EXEC_PATCH_LOCK = threading.Lock()


# Set once this process places a split, so a one-card load never walks its components looking for one.
_SPLIT_EVER_PLACED = False


def _split_home(pipe: Any) -> Any:
    if not _SPLIT_EVER_PLACED:
        return None
    try:
        components = pipe.components
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(components, dict):
        return None
    from .diffusion_layer_split import layer_split_of

    for module in components.values():
        layered = layer_split_of(module)
        if layered is not None:
            return layered.current_home()
    for module in components.values():
        home = getattr(module, "_unsloth_split_home", None)
        if home is not None:
            return home
    return None


def _pipeline_bases() -> list[type]:
    """``DiffusionPipeline`` plus ``ModularPipeline`` (FLUX.3 / MiniMax-H3; not a subclass) when this diffusers has it."""
    bases: list[type] = []
    try:
        from diffusers import DiffusionPipeline
        bases.append(DiffusionPipeline)
    except Exception:  # noqa: BLE001
        pass
    try:
        from diffusers import ModularPipeline
        bases.append(ModularPipeline)
    except Exception:  # noqa: BLE001 - diffusers before modular pipelines
        pass
    return bases


def _patch_execution_device(cls: type) -> bool:
    if getattr(cls, "_unsloth_split_exec_patch", False):
        return True
    orig_exec = cls.__dict__.get("_execution_device")
    orig_dev = cls.__dict__.get("device")
    if not isinstance(orig_exec, property) or not isinstance(orig_dev, property):
        return False

    def _exec(self: Any) -> Any:
        home = _split_home(self)
        return home if home is not None else orig_exec.fget(self)

    def _dev(self: Any) -> Any:
        home = _split_home(self)
        return home if home is not None else orig_dev.fget(self)

    cls._execution_device = property(_exec, doc = orig_exec.__doc__)
    cls.device = property(_dev, doc = orig_dev.__doc__)
    cls._unsloth_split_exec_patch = True
    return True


def install_execution_device_patch() -> bool:
    """``_execution_device`` / ``.device`` report the primary card of a split pipeline. diffusers returns the first
    non-CPU component (the VAE) or, for a modular pipeline, the first component (the text encoder), so latents landed
    on the secondary. Inert without a split denoiser; covers ``from_pipe`` siblings. True when ``DiffusionPipeline``
    is patched."""
    bases = _pipeline_bases()
    if not bases:
        return False
    with _EXEC_PATCH_LOCK:
        patched = [_patch_execution_device(cls) for cls in bases]
    return patched[0]


def companion_names(pipe: Any) -> tuple[str, ...]:
    torch = _torch()
    names = []
    for name, module in (getattr(pipe, "components", {}) or {}).items():
        if not isinstance(module, torch.nn.Module):
            continue
        if name in DENOISER_ATTRS or str(name).startswith(_KEEP_WITH_DENOISER_PREFIXES):
            continue
        names.append(str(name))
    return tuple(names)


_PLACEMENTS: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def place_split(
    pipe: Any,
    *,
    primary: Any,
    secondary_ordinal: int,
    names: Optional[Sequence[str]] = None,
    logger: Any = None,
) -> dict[str, str]:
    """Place ``names`` (default: the companions) on ``cuda:<secondary_ordinal>`` and everything else on ``primary``;
    returns component -> device. Raises on failure (the caller falls back to single-card offload)."""
    torch = _torch()
    from .diffusion_fast_load import fast_upload

    primary_dev = torch.device(primary)
    if primary_dev.type == "cuda" and primary_dev.index is None:
        primary_dev = torch.device("cuda", torch.cuda.current_device())
    secondary_dev = torch.device("cuda", int(secondary_ordinal))
    if names and tuple(names) == TEXT_ENCODERS_ONLY:
        moved = tuple(n for n in companion_names(pipe) if n.startswith("text_encoder"))
    else:
        moved = tuple(names) if names else companion_names(pipe)
    components = {
        str(n): m
        for n, m in (getattr(pipe, "components", {}) or {}).items()
        if isinstance(m, torch.nn.Module)
    }
    placement: dict[str, str] = {}
    order = sorted(components, key = lambda n: 0 if n in moved else 1)
    for name in order:
        module = components[name]
        device = secondary_dev if name in moved else primary_dev
        with fast_upload([module], device, logger = logger):
            module.to(device)
        placement[name] = str(device)
    for name in moved:
        if name in components:
            install_boundary(components[name], primary_dev)
    global _SPLIT_EVER_PLACED
    _SPLIT_EVER_PLACED = True
    for name, module in components.items():
        if name not in moved:
            module._unsloth_split_home = primary_dev
    install_execution_device_patch()
    info = {
        "placement": placement,
        "primary": str(primary_dev),
        "secondary": str(secondary_dev),
        "moved": list(moved),
    }
    try:
        _PLACEMENTS[pipe] = info
    except TypeError:
        pass
    if logger is not None:
        logger.info(
            "diffusion.multigpu: placed %s on %s, the rest on %s",
            ", ".join(moved) or "nothing",
            secondary_dev,
            primary_dev,
        )
    return placement


def undo_split(pipe: Any) -> None:
    """Remove every split placed on ``pipe``: a layer split of any denoiser (its block boundary and the planner's
    record) and the component boundaries. Weights stay where they are."""
    torch = _torch()
    from .diffusion_layer_split import undo_layer_split as undo_layers
    from .diffusion_layer_split_plan import forget_applied_layer_split

    for name in DENOISER_ATTRS:
        try:
            undo_layers(pipe, attr = name)
        except Exception:  # noqa: BLE001 - best effort; the component boundaries below still come off
            pass
    forget_applied_layer_split(pipe)
    for module in (getattr(pipe, "components", {}) or {}).values():
        if isinstance(module, torch.nn.Module):
            remove_boundary(module)
            if "_unsloth_split_home" in module.__dict__:
                del module.__dict__["_unsloth_split_home"]
    try:
        _PLACEMENTS.pop(pipe, None)
    except TypeError:
        pass


def apply_split_plan(
    pipe: Any,
    plan: Any,
    *,
    placement: str,
    logger: Any = None,
) -> bool:
    """Place a split plan."""
    secondary = getattr(plan, "secondary_ordinal", None)
    if secondary is None:
        return False
    try:
        place_split(
            pipe,
            primary = placement,
            secondary_ordinal = int(secondary),
            names = getattr(plan, "secondary_components", None),
            logger = logger,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - fall back to one card
        if logger is not None:
            logger.warning(
                "diffusion.multigpu: split placement failed (%s); falling back to one card", exc
            )
        undo_split(pipe)
        try:
            pipe.to("cpu")
            _torch().cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return False


def is_split(pipe: Any) -> bool:
    """A companion split or a layer split is placed on ``pipe``."""
    if pipe is None:
        return False
    try:
        if pipe in _PLACEMENTS:
            return True
    except TypeError:
        return False
    from .diffusion_layer_split_plan import is_layer_split

    return is_layer_split(pipe)


def placement_status(pipe: Any) -> Optional[dict[str, Any]]:
    if pipe is None:
        return None
    try:
        info = _PLACEMENTS.get(pipe)
        cfg = _CFG_ROUTERS.get(pipe)
        declined = _CFG_DECLINES.get(pipe)
    except TypeError:
        return None
    from .diffusion_layer_split_plan import layer_split_status

    layer = layer_split_status(pipe)
    decision = placement_decision(pipe)
    if info is None and cfg is None and declined is None and layer is None and decision is None:
        return None
    out: dict[str, Any] = dict(info or {})
    try:
        stream = _STREAM_ESTIMATES.get(pipe)
    except TypeError:
        stream = None
    if stream is not None:
        out["stream_estimate"] = dict(stream)
    try:
        priced = _MODE_ESTIMATES.get(pipe)
    except TypeError:
        priced = None
    if priced is not None:
        out["mode_estimate"] = dict(priced)
    from .diffusion_layer_split import layer_split_stats

    live = {
        name: stats
        for name in DENOISER_ATTRS
        if (stats := layer_split_stats(getattr(pipe, name, None))) is not None
    }
    if layer is not None:
        out["layer_split"] = dict(layer)
        if live:
            out["layer_split"]["live"] = live
    elif (
        "layer_split" in out
    ):  # placed by the engine directly (no planner record): its own summary, live
        engine = out["layer_split"]
        out["layer_split"] = dict(live.get(engine.get("denoiser", "transformer")) or engine)
    if cfg is not None:
        out["cfg_parallel"] = cfg.describe()
    elif declined is not None:
        out["cfg_parallel"] = {"armed": False, "declined": declined}
    if decision is not None:
        if cfg is None:
            cfg_rejected = declined
        elif cfg.disabled_reason not in (None, "uninstalled"):
            # armed, then fell back to one card at run time: say so where the placement is reported
            cfg_rejected = f"disabled at run time: {cfg.disabled_reason}"
        else:
            cfg_rejected = None
        out["decision"] = _live_decision(pipe, decision, info, layer, cfg_rejected)
        choice = _replica_choice(out["decision"])
        if choice is not None:
            out.setdefault("cfg_parallel", {"armed": False})["placement"] = choice
    return out


def _replica_choice(live: Mapping[str, Any]) -> Optional[dict[str, str]]:
    """What the placement decided about a CFG replica and why: ``chosen`` is ``cfg_replica`` or the layout that ran
    instead. None when the planner did not consider one (one card selected, an unguided or unknown request)."""
    if live.get("chosen") == "cfg_replica":
        return {"chosen": "cfg_replica", "reason": str(live.get("summary") or "")}
    for candidate in live.get("rejected") or ():
        if candidate.get("kind") == "cfg_replica":
            return {"chosen": str(live.get("chosen")), "reason": str(candidate.get("reason") or "")}
    return None


def _live_decision(
    pipe: Any,
    decision: PlacementDecision,
    info: Optional[dict],
    layer: Optional[dict],
    cfg_declined: Optional[str],
) -> dict[str, Any]:
    """The recorded decision, corrected by what engaged: a split whose placement failed reports its one-card fallback,
    and a declined CFG-parallel replica joins the rejected candidates."""
    out = decision.describe()
    failed = None
    if decision.chosen == "companion_split" and info is None:
        failed = "the companion split did not engage at placement"
    elif decision.chosen == "layer_split" and not (layer or {}).get("cards"):
        failed = f"the layer split did not engage ({(layer or {}).get('declined') or 'placement failed'})"
    elif decision.chosen == "cfg_replica" and cfg_declined is not None:
        failed = f"the CFG replica fell back to one card: {cfg_declined}"
        cfg_declined = None
    if failed is not None:
        out["rejected"] = [{"kind": decision.chosen, "reason": failed}, *out["rejected"]]
        if decision.fallback_kind is not None and (
            decision.fallback_kind != "companion_split" or info is not None
        ):
            out["chosen"] = decision.fallback_kind
            out["summary"] = (
                f"placement: {decision.fallback_summary or decision.fallback_kind} ({failed})"
            )
        elif decision.fallback_resident:
            out["chosen"] = "one_card"
            out["summary"] = (
                f"placement: one card: everything resident on cuda:{decision.primary} ({failed})"
            )
        else:
            out["chosen"] = "offload"
            out["summary"] = (
                f"placement: one card with offload on cuda:{decision.primary} ({failed})"
            )
    if cfg_declined is not None:
        out["rejected"] = [*out["rejected"], {"kind": "cfg_parallel", "reason": cfg_declined}]
    return out


_COMPILE_RECORDER_LOCK = threading.Lock()


def install_compile_recorder() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    with _COMPILE_RECORDER_LOCK:
        current = torch.nn.Module.compile
        if getattr(current, "_unsloth_recorder", False):
            return True

        def compile(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: A001 - mirrors nn.Module.compile
            self.__dict__["_unsloth_compile_args"] = (args, dict(kwargs))
            return current(self, *args, **kwargs)

        compile._unsloth_recorder = True  # type: ignore[attr-defined]
        compile.__wrapped__ = current  # type: ignore[attr-defined]
        torch.nn.Module.compile = compile
        return True


_ROOT_HOOK_DICTS = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


# Offload hooks move weights without changing the math; a resident replica drops them.
_OFFLOAD_REGISTRY_HOOKS = frozenset(
    {"group_offloading", "layer_execution_tracker", "lazy_prefetch_group_offloading"}
)
_OFFLOAD_MODULE_MARKERS = (
    "diffusers.hooks.group_offloading",
    "diffusers.hooks.hooks",
    "accelerate.hooks",
    "diffusion_offload_prefetch",
    "diffusion_small_host",
    "diffusion_pinned_arena",
)


def _origin_modules(fn: Any) -> list[str]:
    out: list[str] = []
    seen = 0
    while fn is not None and seen < 6:
        seen += 1
        out.append(str(getattr(fn, "__module__", "") or ""))
        out.append(str(getattr(type(fn), "__module__", "") or ""))
        out.append(str(getattr(getattr(fn, "func", None), "__module__", "") or ""))
        fn = (
            getattr(fn, "__wrapped__", None)
            or getattr(fn, "__func__", None)
            or getattr(fn, "func", None)
        )
    return out


def _is_offload_object(obj: Any) -> bool:
    return any(marker in mod for mod in _origin_modules(obj) for marker in _OFFLOAD_MODULE_MARKERS)


def _registry_kind(sub: Any) -> str:
    registry = sub.__dict__.get("_diffusers_hook")
    hooks = getattr(registry, "hooks", None) if registry is not None else None
    if not hooks:
        return "none"
    return "offload" if set(hooks) <= _OFFLOAD_REGISTRY_HOOKS else "other"


def _immutable_capture(value: Any, depth: int) -> bool:
    """A closure cell a replica may share with its source. deepcopy keeps a function (and so its closure) as the same
    object, and the two branches call it concurrently, so only values nobody can mutate qualify: scalars, strings,
    dtypes / devices, tuples / frozensets of those, and plain functions whose own captures qualify. A list, dict, set,
    tensor, bound method or arbitrary object (a counter, a cache) would be shared state."""
    torch = _torch()
    if depth > 3:
        return False
    if isinstance(
        value, (str, bytes, int, float, complex, bool, type(None), torch.dtype, torch.device, type)
    ):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable_capture(v, depth + 1) for v in value)
    if isinstance(value, types.FunctionType):
        return _portable(value, depth)
    return False


def _portable(fn: Any, depth: int = 0) -> bool:
    """True when a hook / forward override can be shared by a copy of its module: nothing it captures (closure cells,
    bound ``self``, partial arguments, the attributes of a captured object) is a module or a CUDA tensor, which would
    keep pointing at the source card."""
    torch = _torch()
    if depth > 3:
        return False
    if isinstance(fn, torch.nn.Module) or (
        isinstance(fn, torch.Tensor) and fn.device.type != "cpu"
    ):
        return False
    if isinstance(fn, (list, tuple, set)):
        return all(_portable(v, depth + 1) for v in fn)
    if isinstance(fn, dict):
        return all(_portable(v, depth + 1) for v in fn.values())
    if isinstance(
        fn,
        (str, bytes, int, float, bool, type(None), torch.Tensor, torch.dtype, torch.device, type),
    ):
        return True
    if isinstance(fn, (types.FunctionType, types.MethodType)):
        owner = getattr(fn, "__self__", None)
        if owner is not None and not _portable(owner, depth + 1):
            return False
        func = getattr(fn, "__func__", fn)
        for cell in getattr(func, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if not _immutable_capture(value, depth + 1):
                return False
        return True
    partial_func = getattr(fn, "func", None)
    if partial_func is not None and hasattr(fn, "args"):
        return _portable(partial_func, depth + 1) and all(
            _portable(a, depth + 1)
            for a in (*fn.args, *(getattr(fn, "keywords", None) or {}).values())
        )
    state = getattr(fn, "__dict__", None)
    if isinstance(state, dict):
        return all(_portable(v, depth + 1) for v in state.values())
    return True


# Foreign hooks may keep state in a module global that replica threads race on, so they decline.
_TRUSTED_HOOK_MODULES = ("core.inference.", "studio.backend.core.inference.")


def _hook_origin(fn: Any) -> str:
    target = getattr(fn, "func", None) if hasattr(fn, "args") and hasattr(fn, "func") else fn
    target = getattr(target, "__func__", target)
    if isinstance(target, (types.FunctionType, types.BuiltinFunctionType)):
        return str(getattr(target, "__module__", None) or "")
    return str(getattr(type(target), "__module__", None) or "")


def _trusted_hook(fn: Any) -> bool:
    return _hook_origin(fn).startswith(_TRUSTED_HOOK_MODULES) and _portable(fn)


def _named_modules(module: Any) -> list:
    """``list(module.named_modules())`` without nested generators: same order and de-duplication."""
    base = _torch().nn.Module.named_modules
    if type(module).named_modules is not base or "named_modules" in module.__dict__:
        return list(module.named_modules())
    out: list = []
    memo: set = set()
    stock: set = set()
    add = out.append

    def visit(prefix: str, sub: Any) -> None:
        cls = type(sub)
        if cls not in stock:
            if cls.named_modules is not base:
                out.extend(sub.named_modules(memo, prefix, True))
                return
            stock.add(cls)
        if "named_modules" in sub.__dict__:
            out.extend(sub.named_modules(memo, prefix, True))
            return
        if sub in memo:
            return
        memo.add(sub)
        add((prefix, sub))
        children = sub._modules
        if children:
            dot = prefix + "." if prefix else ""
            for key, child in children.items():
                if child is not None:
                    visit(dot + key, child)

    visit("", module)
    return out


def _plain_lookup(cls: Any) -> bool:
    torch = _torch()
    return (
        cls.__getattr__ is torch.nn.Module.__getattr__
        and cls.__getattribute__ is object.__getattribute__
        and not any(hasattr(cls, n) for n in _HOOK_SLOTS)
    )


_HOOK_SLOTS = ("_hf_hook", "_forward_hooks", "_forward_pre_hooks")


def _nothing_to_report(state: dict) -> bool:
    """True only when :func:`_hooks_present` provably finds nothing in this ``__dict__``."""
    if (
        state.get("_diffusers_hook") is not None
        or "_hf_hook" in state
        or state.get("forward") is not None
        or "_forward_hooks" not in state
        or "_forward_pre_hooks" not in state
        or state["_forward_hooks"]
        or state["_forward_pre_hooks"]
    ):
        return False
    for slot in ("_parameters", "_buffers", "_modules"):
        members = state.get(slot)
        if members is not None and "_hf_hook" in members:
            return False
    return True


def _hooks_present(module: Any) -> Optional[str]:
    """Name of the first submodule carrying state a replica cannot inherit faithfully, else None: a non-offload hook
    registry (step caches), an accelerate hook, or a forward hook / instance forward override below the root whose
    captured state points at the source (deepcopy treats functions as atoms). Offload machinery (diffusers group
    offloading, Studio's prefetchers) is dropped by :func:`build_replica`, which holds every weight resident. The
    ROOT's torch forward hooks run on the source around the router for every branch and the copy does not carry them,
    but whatever per-call state they write must reach the replica: only Studio's own root hooks are accepted, whose
    state is the ``_unsloth_*`` scalars :func:`_mirror_hook_state` copies."""
    plain: dict = {}
    for name, sub in _named_modules(module):
        cls = type(sub)
        fast = plain.get(cls)
        if fast is None:
            fast = plain[cls] = _plain_lookup(cls)
        if fast and _nothing_to_report(sub.__dict__):
            continue
        kind = _registry_kind(sub)
        if kind == "other":
            return f"{name or '<root>'}._diffusers_hook ({', '.join(sorted(sub._diffusers_hook.hooks))})"
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'}._hf_hook"
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            for hook in (getattr(sub, attr, None) or {}).values():
                if _is_offload_object(hook):
                    continue
                if name and not _trusted_hook(hook):
                    return f"{name}.{attr}"
                if not name and not _hook_origin(hook).startswith(_TRUSTED_HOOK_MODULES):
                    return f"<root>.{attr} (a hook outside Unsloth: its per-call state cannot be mirrored)"
        override = sub.__dict__.get("forward")
        if override is None or isinstance(override, (types.MethodType, CfgParallelRouter)):
            continue
        if not name and _resident_graph(sub) is not None:
            # install_cfg_parallel moves the root's graph under the router
            continue
        if kind == "offload" or _is_offload_object(override):
            continue
        if callable(override) and not _trusted_hook(override):
            return f"{name or '<root>'}.forward (instance override)"
    return None


def _offload_attr_keys(sub: Any) -> list[str]:
    torch = _torch()
    keys = []
    kind = _registry_kind(sub)
    for key, value in sub.__dict__.items():
        if key in ("_parameters", "_buffers", "_modules") or key.startswith("_forward_"):
            continue
        if key == "_diffusers_hook" and kind == "offload":
            keys.append(key)
        elif key == "forward" and (kind == "offload" or _is_offload_object(value)):
            keys.append(key)
        elif not isinstance(value, (torch.Tensor, torch.nn.Module)) and _is_offload_object(value):
            keys.append(key)
    return keys


_DROP = object()


def _rebind_root_forward(original: Any, source: Any, replica: Any) -> None:
    """Give ``replica`` the root ``forward`` the source ran before the router took its slot: offload wrappers (math
    neutral, dropped with the offload state) are peeled off, and a method bound to the source is rebound to the copy.
    Anything else cannot be reproduced and raises (CFG-parallel then declines)."""
    seen = 0
    while original is not None and _is_offload_object(original) and seen < 8:
        seen += 1
        original = getattr(original, "__wrapped__", None)
    if isinstance(original, types.MethodType) and original.__self__ is source:
        if original.__func__ is getattr(type(source), "forward", None):
            return
        replica.__dict__["forward"] = types.MethodType(original.__func__, replica)
        return
    raise RuntimeError(
        f"the denoiser's root forward override ({original!r}) cannot be reproduced on a replica"
    )


def build_replica(module: Any, device: Any) -> Any:
    """A copy of ``module`` whose parameters and buffers are created directly on ``device`` (never staged on the
    source card), with every recorded ``Module.compile`` re-applied to the copy. Raises when it cannot be built.

    Built outside inference mode: inside it, torchao weights made outside raise "Cannot set version_counter for
    inference tensor" and ones made inside silently stay on the source card."""
    if getattr(module, "_unsloth_tp2", None) is not None:
        raise RuntimeError(
            "a tensor-parallel denoiser already spans both cards; it has no single-card replica"
        )
    torch = _torch()
    device = torch.device(device)
    with torch.inference_mode(False), torch.no_grad():
        replica = _copy_module(module, device)
    stray = sorted(
        {
            str(t.device)
            for t in (*replica.parameters(), *replica.buffers())
            if t.device.type == "cuda" and t.device != device
        }
    )
    if stray:
        raise RuntimeError(f"replica weights stayed on {', '.join(stray)}, not {device}")
    # The ROCm flash gate is thread-local state its root hooks set; the copy runs on its own worker thread.
    from .diffusion_attention import track_rocm_flash_replica

    track_rocm_flash_replica(module, replica, device)
    return replica


def _copy_module(module: Any, device: Any) -> Any:
    torch = _torch()
    memo: dict[int, Any] = {}
    keep: list[Any] = []
    for _name, param in module.named_parameters(remove_duplicate = True):
        if id(param) in memo:
            continue
        copied = param.detach().to(device)
        new = copied
        if isinstance(param, torch.nn.Parameter):
            try:
                new = torch.nn.Parameter(copied, requires_grad = param.requires_grad)
            except Exception:  # noqa: BLE001 - a tensor subclass that refuses the wrapper stays a plain tensor
                new = copied
        memo[id(param)] = new
        keep.append(param)
    for _name, buffer in module.named_buffers(remove_duplicate = True):
        if id(buffer) in memo:
            continue
        memo[id(buffer)] = buffer.detach().to(device)
        keep.append(buffer)
    from .diffusion_flux_convrot_fused import is_replica_record as f3_qkv_record
    from .diffusion_flux_convrot_fused import rebuild_replica as f3_qkv_rebuild
    from .diffusion_moss_fused import is_replica_record as moss_qkv_record
    from .diffusion_moss_fused import rebuild_replica as moss_qkv_rebuild
    from .diffusion_qwenimage21_dense_fused import is_replica_record, rebuild_replica
    from .diffusion_zimage_fused import is_replica_record as zimage_qkv_record
    from .diffusion_zimage_fused import rebuild_replica as zimage_qkv_rebuild

    for sub in module.modules():
        offload_keys = set(_offload_attr_keys(sub))
        for key, value in sub.__dict__.items():
            if (
                isinstance(value, CfgParallelRouter)
                or key in offload_keys
                or key in _SOURCE_ONLY_ATTRS
                or is_replica_record(key, value)
                or f3_qkv_record(key, value)
                or moss_qkv_record(key, value)
                or zimage_qkv_record(key, value)
            ):
                # Pinned offload copies and source-card caches are dropped, not deep-copied.
                memo[id(value)] = _DROP
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            for hook in (getattr(sub, attr, None) or {}).values():
                if _is_offload_object(hook):
                    memo[id(hook)] = _DROP
        # shared, or the two CFG branches of one step run different GEMMs
        for value in _shared_state_values(sub):
            memo[id(value)] = value
    guard = module.__dict__.get("_unsloth_compile_guard")
    if guard is not None:
        memo[id(guard)] = _DROP
    if "_unsloth_adaln" in module.__dict__:
        from .diffusion_adaln_precompute import replica_memo
        replica_memo(module, device, memo)
    # A baked LoRA's scale pinned as a 0-d device tensor (diffusion._pin_baked_lora_scales) lives in a plain dict,
    # not a buffer: the copy needs its own on ``device``.
    for sub in module.modules():
        scaling = sub.__dict__.get("scaling")
        if isinstance(scaling, dict):
            for value in scaling.values():
                if isinstance(value, torch.Tensor) and id(value) not in memo:
                    memo[id(value)] = value.detach().to(device)
                    keep.append(value)
    memo[id(memo)] = keep
    replica = copy.deepcopy(module, memo)
    router = module.__dict__.get("forward")
    for sub in replica.modules():
        for key in [k for k, v in sub.__dict__.items() if v is _DROP]:
            del sub.__dict__[key]
        for attr in _ROOT_HOOK_DICTS:
            hooks = sub.__dict__.get(attr)
            if hooks:
                for hook_id in [k for k, v in hooks.items() if v is _DROP]:
                    del hooks[hook_id]
                    for other in _ROOT_HOOK_DICTS:
                        sub.__dict__.get(other, {}).pop(hook_id, None)
    # Release the copies before re-fusing on the replica's card, else each replaced allocation stays alive.
    memo.clear()
    del keep
    param = copied = new = buffer = (
        None  # the copy loop's last parameter, still referenced by these names
    )
    try:
        rebuild_replica(module, replica)
    except Exception:  # noqa: BLE001 - the replica keeps the stock projections
        pass
    try:
        f3_qkv_rebuild(module, replica)
    except Exception:  # noqa: BLE001 - the replica runs three GEMMs
        pass
    for rebuild in (moss_qkv_rebuild, zimage_qkv_rebuild):
        try:
            rebuild(module, replica)
        except Exception:  # noqa: BLE001 - the replica runs three GEMMs
            pass
    if isinstance(router, CfgParallelRouter) and router.had_override:
        _rebind_root_forward(router.orig_forward, module, replica)
    # The source's own __call__ runs its root hooks around every branch; the copy must not run them a second time.
    for attr in _ROOT_HOOK_DICTS:
        hooks = getattr(replica, attr, None)
        if hooks is not None:
            hooks.clear()
    compiled_subs = 0
    for (name, src), (_rname, dst) in zip(module.named_modules(), replica.named_modules()):
        for key, value in list(dst.__dict__.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.device != device
                and value.device.type == "cuda"
            ):
                dst.__dict__[key] = value.to(device)
        if getattr(src, "_compiled_call_impl", None) is not None:
            recorded = src.__dict__.get("_unsloth_compile_args")
            dst._compiled_call_impl = None
            if recorded is None:
                raise RuntimeError(f"compiled submodule '{name}' has no recorded compile arguments")
            args, kwargs = recorded
            dst.compile(*args, **kwargs)
            compiled_subs += 1
        elif "_compiled_call_impl" in dst.__dict__:
            dst._compiled_call_impl = None
    if compiled_subs:
        _rewrap_compiled(module, replica)
    replica._unsloth_replica_compiled = compiled_subs
    replica.eval()
    return replica


# Per-step precision state only the source's step loop advances (the replica never enters ``cache_context``).
_SHARED_STATE_TYPES = frozenset({"NVFP4StepController", "StepPolicy"})
# Source-only caches are not copied (copies would pin source-card tensors no render clears); FLUX.3
# conditioning reaches the replica via diffusion_cond_blocks; AdaLN's lock and tables stay on the
# source, so the replica gets real modulation modules (diffusion_adaln_precompute.replica_memo).
_SOURCE_ONLY_ATTRS = frozenset(
    {"_unsloth_cond_cache", "_unsloth_rocm_flash_replicas", "_unsloth_adaln"}
)


def _shared_state_values(sub: Any) -> list:
    return [value for value in sub.__dict__.values() if type(value).__name__ in _SHARED_STATE_TYPES]


def _rewrap_compiled(source: Any, replica: Any) -> None:
    """Re-apply the wrappers Studio's compile path puts around the source's compiled blocks (``dst.compile`` replaced
    them): the compile guard (eager fallback on a compile-time failure, the background-compile eager gate, the compile
    config on this thread) and the block-0 restride, in the order the source got them."""
    if "_unsloth_compile_guard" in source.__dict__:
        from .diffusion_speed import guard_compiled_blocks
        guard_compiled_blocks(replica)
    from . import diffusion_block_restride

    if any(
        diffusion_block_restride.is_wrapped(getattr(m, "_compiled_call_impl", None))
        for m in source.modules()
    ):
        diffusion_block_restride.install(replica)


_ADAPTER_ATTRS = (
    "scaling",
    # PEFT stores the selection here; ``active_adapters`` is a property over it (never in ``__dict__``)
    "_active_adapter",
    "active_adapters",
    "merged_adapters",
    "disable_adapters",
    "_disable_adapters",
)
_ADAPTER_ATTR_SET = frozenset(_ADAPTER_ATTRS)
_TRAVERSAL_METHODS = ("named_modules", "named_parameters", "named_buffers", "_named_members")


def _stock_traversal(module: Any) -> bool:
    """No traversal override on the root, so :func:`_named_modules` is equivalent."""
    base = _torch().nn.Module
    cls = type(module)
    return all(
        getattr(cls, n, None) is getattr(base, n) and n not in module.__dict__
        for n in _TRAVERSAL_METHODS
    )


def _stock_fingerprint_parts(named: list) -> list:
    """:func:`weights_fingerprint`'s parts in one pass, with torch's ``_named_members`` order and de-duplication."""
    params: list = []
    buffers: list = []
    adapters: list = []
    seen_params: set = set()
    seen_buffers: set = set()
    for prefix, sub in named:
        own_params = sub._parameters
        own_buffers = sub._buffers
        if own_params or own_buffers:
            dot = prefix + "." if prefix else ""
            if own_params:
                _member_parts(own_params, dot, seen_params, params)
            if own_buffers:
                _member_parts(own_buffers, dot, seen_buffers, buffers)
        if not _ADAPTER_ATTR_SET.isdisjoint(sub.__dict__):
            _adapter_parts(prefix, sub, adapters)
    return params + buffers + adapters


def _member_parts(members: dict, dot: str, seen: set, out: list) -> None:
    for key, value in members.items():
        if value is None or value in seen:
            continue
        seen.add(value)
        try:
            version = value._version
        except Exception:  # noqa: BLE001 - inference tensors carry no version
            version = -1
        out.append((dot + key, id(value), version))


def _adapter_parts(name: str, sub: Any, parts: list) -> None:
    for attr in _ADAPTER_ATTRS:
        value = sub.__dict__.get(attr)
        if isinstance(value, dict):
            value = {
                k: (id(v), v._version) if hasattr(v, "_version") else v for k, v in value.items()
            }
        if value is not None and not callable(value):
            parts.append((name, attr, repr(value)))


def weights_fingerprint(module: Any) -> int:
    """Changes whenever the denoiser's weights or adapter state do: a parameter or buffer (NVFP4 / quantised layers
    keep their packed weights and scales as buffers) replaced, edited in place (fused LoRA bumps the version counter),
    added or removed (PEFT injection), or an adapter's scale / activation changed. A ``.data`` swap to different values
    is not seen; diffusers' LoRA paths do not do that."""
    # Identity + version, not data pointer: offload round trips must not force a rebuild.
    if _stock_traversal(module):
        return hash(tuple(_stock_fingerprint_parts(_named_modules(module))))
    parts: list = []
    for name, param in (*module.named_parameters(), *module.named_buffers()):
        try:
            version = param._version
        except Exception:  # noqa: BLE001 - inference tensors carry no version
            version = -1
        parts.append((name, id(param), version))
    for name, sub in module.named_modules():
        _adapter_parts(name, sub, parts)
    return hash(tuple(parts))


_PROBE_SAMPLES = 64


def _dense_leaves(t: Any, depth: int = 0) -> Iterator[Any]:
    """The plain tensors holding ``t``'s values: ``t`` itself, or a torchao subclass's inner tensors (qdata, scales)."""
    flatten = getattr(type(t), "__tensor_flatten__", None)
    if flatten is None or depth > 4:
        yield t
        return
    try:
        names, _ctx = t.__tensor_flatten__()
    except Exception:  # noqa: BLE001 - an opaque subclass: probe the outer tensor
        yield t
        return
    for name in names:
        inner = getattr(t, name, None)
        if inner is not None:
            yield from _dense_leaves(inner, depth + 1)


def weights_probe(module: Any, index_cache: Optional[dict] = None) -> Optional[bytes]:
    """Sampled bytes of every inference tensor (no version counter, so invisible to :func:`weights_fingerprint`),
    read back in one host sync; None when there are none. Edits missing every sampled byte go unseen."""
    torch = _torch()
    cache = {} if index_cache is None else index_cache
    by_device: dict = {}
    with torch.no_grad():
        for _name, t in (*module.named_parameters(), *module.named_buffers()):
            if not t.is_inference():
                continue
            for leaf in _dense_leaves(t):
                if not isinstance(leaf, torch.Tensor) or leaf.numel() == 0 or leaf.is_meta:
                    continue
                # bytes, not elements: a gather is not implemented for every weight dtype (fp8, packed fp4)
                flat = _as_bytes(leaf)
                n = flat.numel()
                key = (n, flat.device)
                idx = cache.get(key)
                if idx is None:
                    # integer spacing: a float32 linspace rounds n - 1 up to n past 2**24 bytes and reads past the end
                    k = min(n, _PROBE_SAMPLES)
                    idx = torch.arange(k, device = flat.device) * (n - 1) // max(k - 1, 1)
                    cache[key] = idx
                by_device.setdefault(flat.device, []).append(flat.index_select(0, idx))
    if not by_device:
        return None
    return b"".join(
        torch.cat(chunks).cpu().numpy().tobytes()
        for _dev, chunks in sorted(by_device.items(), key = lambda kv: str(kv[0]))
    )


_LATENT_KEYS = ("hidden_states", "sample", "x", "latents", "img", "latent")


def _latent_arg(args: tuple, kwargs: dict) -> Any:
    torch = _torch()
    for key in _LATENT_KEYS:
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return value
    # Cosmos 3: ``vision_tokens=[t]``, a new list each call around the same tensor
    vision = kwargs.get("vision_tokens")
    if (
        isinstance(vision, (list, tuple))
        and len(vision) == 1
        and isinstance(vision[0], torch.Tensor)
        and multigpu_all_enabled()
    ):
        return vision[0]
    for value in args:
        if isinstance(value, torch.Tensor):
            return value
    return None


_PLAIN_TYPES = (type(None), bool, int, float, str, bytes)


def _holds_opaque_tensors(obj: Any, depth: int = 0) -> bool:
    """True when ``obj`` carries tensors inside something other than a plain tuple / list / dict (a prefix-KV cache
    object, a namedtuple, a value holder). The replica walker cannot move those to its card, and a mutable cache must
    not be shared across the two threads, so such a branch stays on the primary."""
    torch = _torch()
    if isinstance(obj, (torch.Tensor, *_PLAIN_TYPES, torch.device, torch.dtype, torch.nn.Module)):
        return False
    if depth < 4 and isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return any(_holds_opaque_tensors(v, depth + 1) for v in obj)
    if depth < 4 and isinstance(obj, list):
        return any(_holds_opaque_tensors(v, depth + 1) for v in obj)
    if depth < 4 and isinstance(obj, dict):
        return any(_holds_opaque_tensors(v, depth + 1) for v in obj.values())
    if isinstance(obj, tuple):
        values: Any = obj
    else:
        try:
            values = vars(obj).values()
        except TypeError:
            return False
    for value in values:
        if isinstance(value, torch.Tensor) or (
            isinstance(value, (list, tuple, dict)) and next(_iter_tensors(value), None) is not None
        ):
            return True
        if (
            depth < 4
            and not isinstance(value, _PLAIN_TYPES)
            and _holds_opaque_tensors(value, depth + 1)
        ):
            return True
    return False


# Prefix-KV cache (Qwen-Image-2.1): step-0 prefill ("extract") runs on the primary; "cached" odd branches use a replica copy.
_KV_KWARG = "kv_cache"
_KV_MODE_KWARG = "kv_cache_mode"


class _Unmirrorable(Exception):
    pass


def _walk_cache(
    obj: Any,
    move: Optional[Callable[[Any], Any]],
    found: list,
    depth: int = 0,
) -> Any:
    """Structural copy of a cache object (``move`` applied to every tensor), or with ``move`` None only the tensors
    collected into ``found``. Raises :class:`_Unmirrorable` on anything but plain values, containers and ``__dict__``
    value holders (a module, a slotted or foreign object), which then stays on the primary."""
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        found.append(obj)
        return move(obj) if move is not None else obj
    if isinstance(obj, (*_PLAIN_TYPES, torch.device, torch.dtype)):
        return obj
    if depth >= 4 or isinstance(obj, torch.nn.Module):
        raise _Unmirrorable(type(obj).__name__)
    if isinstance(obj, list):
        out = [_walk_cache(v, move, found, depth + 1) for v in obj]
        return out if move is not None else obj
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        out_t = tuple(_walk_cache(v, move, found, depth + 1) for v in obj)
        return out_t if move is not None else obj
    if isinstance(obj, dict) and type(obj) is dict:
        out_d = {k: _walk_cache(v, move, found, depth + 1) for k, v in obj.items()}
        return out_d if move is not None else obj
    attrs = getattr(obj, "__dict__", None)
    if type(attrs) is not dict or hasattr(type(obj), "__slots__"):
        raise _Unmirrorable(type(obj).__name__)
    copied = {k: _walk_cache(v, move, found, depth + 1) for k, v in attrs.items()}
    if move is None:
        return obj
    cls = type(obj)
    clone = cls.__new__(cls)  # not object.__new__: a C base (SimpleNamespace) refuses it
    clone.__dict__.update(copied)
    return clone


def _cache_tensors(cache: Any) -> Optional[list]:
    """Tensors held by a mirrorable cache, in fixed order; None if not mirrorable."""
    torch = _torch()
    if cache is None or isinstance(cache, (torch.Tensor, list, tuple, dict, *_PLAIN_TYPES)):
        return None
    found: list = []
    try:
        _walk_cache(cache, None, found)
    except _Unmirrorable:
        return None
    return found


def _prefix_kv_mode(kwargs: Any) -> Optional[str]:
    if not isinstance(kwargs, dict) or kwargs.get(_KV_KWARG) is None:
        return None
    mode = kwargs.get(_KV_MODE_KWARG)
    if mode not in ("extract", "cached"):
        return None
    return mode if _cache_tensors(kwargs[_KV_KWARG]) is not None else None


def _is_prefill(kwargs: Any) -> bool:
    """A prefill writes (or returns, FLUX.2) the cache the pipeline reads back: always on the primary."""
    return isinstance(kwargs, dict) and kwargs.get(_KV_MODE_KWARG) == "extract"


def _without_prefix_kv(args: tuple, kwargs: dict) -> tuple:
    if _prefix_kv_mode(kwargs) != "cached":
        return args, kwargs
    return args, {k: v for k, v in kwargs.items() if k != _KV_KWARG}


def _iter_tensors(obj: Any, depth: int = 0) -> Iterator[Any]:
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        yield obj
    elif depth < 4 and isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v, depth + 1)
    elif depth < 4 and isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v, depth + 1)


def _to_primary(out: Any, primary: Any) -> Any:
    """Copy a replica result home with captures held off: the job's own hold dropped when the worker finished."""
    from .diffusion_cuda_graph import outside_capture
    with outside_capture():
        return move_to_device(out, primary)


class _ReplicaWorker:
    """One daemon thread bound to the replica's card: set_device, grad / inference / autocast mode and the caller's
    context variables (inductor overrides live in one on torch >= 2.12) are applied per job."""

    def __init__(self, device: Any):
        self.device = device
        self._jobs: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(
            target = self._loop, name = f"unsloth-cfg-replica-{device}", daemon = True
        )
        # Global capture mode forbids this thread's CUDA allocations while any recording runs, so hold them out.
        from .diffusion_cuda_graph import capture_hold, capture_release

        capture_hold()
        try:
            self._thread.start()
        except BaseException:
            capture_release()
            raise

    def _loop(self) -> None:
        from .diffusion_cuda_graph import capture_release, held_work

        torch = _torch()
        try:
            with held_work():
                if getattr(self.device, "type", None) == "cuda":
                    try:
                        torch.cuda.set_device(self.device)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001 - refused while a capture records: each job binds its device anyway
            pass
        finally:
            capture_release()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, ctx, modes, done = job
            job = None
            try:
                try:
                    with held_work():
                        try:
                            done["result"] = ctx.run(self._run, fn, modes)
                        except BaseException as exc:  # noqa: BLE001 - surfaced on the caller's thread
                            done["error"] = exc
                        fn = ctx = None
                except Exception as exc:  # noqa: BLE001 - refused while a capture records: the job never ran
                    done["error"] = exc
                    fn = ctx = None
            finally:
                capture_release()
                done["event"].set()

    def _run(self, fn: Callable[[], Any], modes: dict) -> Any:
        torch = _torch()
        with contextlib.ExitStack() as stack:
            if modes.get("inference"):
                stack.enter_context(torch.inference_mode())
            stack.enter_context(torch.set_grad_enabled(modes.get("grad", False)))
            autocast_dtype = modes.get("autocast")
            if autocast_dtype is not None:
                stack.enter_context(torch.autocast(device_type = "cuda", dtype = autocast_dtype))
            stack.enter_context(_device_context(self.device))
            return fn()

    def submit(self, fn: Callable[[], Any]) -> dict:
        torch = _torch()
        autocast = None
        try:
            if torch.is_autocast_enabled("cuda"):
                autocast = torch.get_autocast_dtype("cuda")
        except Exception:  # noqa: BLE001 - older torch
            pass
        modes = {
            "inference": torch.is_inference_mode_enabled(),
            "grad": torch.is_grad_enabled(),
            "autocast": autocast,
        }
        done: dict[str, Any] = {"event": threading.Event()}
        from .diffusion_cuda_graph import capture_hold

        capture_hold()
        self._jobs.put((fn, contextvars.copy_context(), modes, done))
        return done

    @staticmethod
    def wait(done: dict) -> Any:
        done["event"].wait()
        if "error" in done:
            raise done["error"]
        return done.get("result")

    def call(self, fn: Callable[[], Any]) -> Any:
        return self.wait(self.submit(fn))

    def close(self) -> None:
        self._jobs.put(None)


def _leaves(
    obj: Any,
    path: tuple = (),
    depth: int = 0,
) -> list[tuple[tuple, Any]]:
    if depth < 4 and isinstance(obj, (tuple, list)) and not hasattr(obj, "_fields"):
        out: list = []
        for i, v in enumerate(obj):
            out.extend(_leaves(v, path + (i,), depth + 1))
        return out or [(path, obj)]
    if depth < 4 and isinstance(obj, dict):
        out = []
        for k, v in obj.items():
            out.extend(_leaves(v, path + (k,), depth + 1))
        return out or [(path, obj)]
    return [(path, obj)]


def _rebuild(
    template: Any,
    values: dict,
    path: tuple = (),
    depth: int = 0,
) -> Any:
    if (
        depth < 4
        and isinstance(template, (tuple, list))
        and not hasattr(template, "_fields")
        and len(template)
    ):
        items = [_rebuild(v, values, path + (i,), depth + 1) for i, v in enumerate(template)]
        return type(template)(items) if isinstance(template, tuple) else items
    if depth < 4 and isinstance(template, dict) and template:
        return type(template)(
            (k, _rebuild(v, values, path + (k,), depth + 1)) for k, v in template.items()
        )
    return values[path]


_SMALL_TENSOR_NUMEL = 4096


def _as_bytes(t: Any) -> Any:
    """Flat uint8 view; a one-element ``t.expand(n)`` keeps stride 0 through ``contiguous()``, so copy it."""
    torch = _torch()
    flat = t.reshape(-1).contiguous()
    if flat.numel() and flat.stride(0) != 1:
        flat = flat.clone(memory_format = torch.contiguous_format)
    return flat.view(torch.uint8)


def _bitwise_equal(a: Any, b: Any) -> bool:
    torch = _torch()
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if a is b:
        return True
    try:
        return bool(torch.equal(_as_bytes(a), _as_bytes(b)))
    except Exception:  # noqa: BLE001 - a layout that cannot be viewed as bytes is never assumed equal
        return False


def _bitwise_equal_all(pairs: Any) -> bool:
    """``all(_bitwise_equal(a, b) for a, b in pairs)`` with one host read per device instead of one per pair: each
    pair's comparison stays on its device and only the combined flag is read back. Same verdict as the per-pair form
    (a byte view compared elementwise is what ``torch.equal`` does on it); metadata mismatches still answer on the host
    without touching the device."""
    torch = _torch()
    by_device: dict = {}
    for a, b in pairs:
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return False
        if a is b or a.numel() == 0:
            continue
        try:
            if a.device.type == "cpu":
                if not torch.equal(_as_bytes(a), _as_bytes(b)):
                    return False
                continue
            by_device.setdefault(a.device, []).append((_as_bytes(a) == _as_bytes(b)).all())
        except Exception:  # noqa: BLE001 - a layout that cannot be viewed as bytes is never assumed equal
            return False
    try:
        return all(bool(torch.stack(flags).all().item()) for flags in by_device.values())
    except Exception:  # noqa: BLE001 - an unreadable verdict is never assumed equal
        return False


def _unchanged_since_launch(snapshots: list) -> bool:
    """True when every (source, launch-time copy) pair still holds the same bytes. One host sync for all pairs, on
    the replica's card; it waits for the primary branch (the copy is ordered after it), which the guidance combine
    waits for anyway, while the replica keeps running."""
    torch = _torch()
    checks = []
    for src, snap in snapshots:
        if src.shape != snap.shape or src.dtype != snap.dtype:
            return False
        if src.numel() == 0:
            continue
        try:
            now = _as_bytes(src.to(snap.device))
            checks.append(
                torch.equal(now, _as_bytes(snap))
                if snap.device.type == "cpu"
                else (now == _as_bytes(snap)).all()
            )
        except Exception:  # noqa: BLE001 - an unverifiable tensor never accepts a prediction
            return False
    if not checks:
        return True
    host = [c for c in checks if isinstance(c, bool)]
    device = [c for c in checks if not isinstance(c, bool)]
    return all(host) and (not device or bool(torch.stack(device).all().item()))


def _same_value(a: Any, b: Any) -> bool:
    if a is b:
        return True
    torch = _torch()
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return False
    try:
        return bool(a == b) and type(a) is type(b)
    except Exception:  # noqa: BLE001 - incomparable objects must be the same object
        return False


# Per-call _unsloth_* state (e.g. HunyuanVideo-1.5 null-mask flag) is written by source-only root
# hooks; the replica mirrors it before each call and verifies the prediction against it.
_STATE_PREFIX = "_unsloth_"
_STATE_TYPES = (bool, int, float, str, type(None))
_STATE_KEEP = frozenset({"_unsloth_replica_compiled"})
_STATE_MARK = "\0" + _STATE_PREFIX


def _may_hold_state(state: dict) -> bool:
    """False only when no key starts with ``_unsloth_``."""
    try:
        joined = "\0".join(state)
    except TypeError:
        return True
    return joined.startswith(_STATE_PREFIX) or _STATE_MARK in joined


def _hook_state(pairs: list) -> tuple:
    out = []
    for i, (src, _dst) in enumerate(pairs):
        state = src.__dict__
        if not _may_hold_state(state):
            continue
        for key, value in state.items():
            if key.startswith(_STATE_PREFIX) and isinstance(value, _STATE_TYPES):
                out.append((i, key, value))
    return tuple(out)


def _mirror_hook_state(pairs: list) -> tuple:
    state = _hook_state(pairs)
    wanted: dict = {}
    for i, key, value in state:
        wanted.setdefault(i, {})[key] = value
    for i, (_src, dst) in enumerate(pairs):
        mine = wanted.get(i, {})
        if not mine and not _may_hold_state(dst.__dict__):
            continue
        for key in [
            k
            for k, v in dst.__dict__.items()
            if k.startswith(_STATE_PREFIX)
            and isinstance(v, _STATE_TYPES)
            and k not in mine
            and k not in _STATE_KEEP
        ]:
            del dst.__dict__[key]
        for key, value in mine.items():
            if dst.__dict__.get(key, _DROP) is not value and dst.__dict__.get(key, _DROP) != value:
                dst.__dict__[key] = value
    return state


def _autocast_state(torch: Any) -> tuple:
    try:
        return (
            torch.is_autocast_cache_enabled(),
            *(
                (dev, torch.get_autocast_dtype(dev))
                if torch.is_autocast_enabled(dev)
                else (dev, None)
                for dev in ("cuda", "cpu")
            ),
        )
    except Exception:  # noqa: BLE001 - torch without per-device autocast queries
        return ()


def _call_signature(args: tuple, kwargs: dict) -> tuple:
    """What a compiled forward's guards key on, numbers by value: 0 / 1 and ``dynamic=False`` compiles stay specialised."""
    torch = _torch()
    sig: list = []
    for path, leaf in _leaves((args, kwargs)):
        if isinstance(leaf, torch.Tensor):
            try:
                stride = tuple(leaf.stride())
                offset = leaf.storage_offset()
            except Exception:  # noqa: BLE001 - layouts without strides
                stride = offset = None
            sig.append(
                (
                    path,
                    tuple(leaf.shape),
                    stride,
                    offset,
                    leaf.dtype,
                    leaf.requires_grad,
                    leaf.is_inference(),
                )
            )
        elif leaf is None or isinstance(leaf, (bool, str)):
            sig.append((path, leaf))
        elif isinstance(leaf, (int, float)):
            sig.append((path, type(leaf), leaf))
        elif path and path[-1] == _KV_KWARG and (held := _cache_tensors(leaf)) is not None:
            # a new prefix length (another prompt) may compile: it must not count as warm
            sig.append((path, type(leaf), tuple((tuple(t.shape), t.dtype) for t in held)))
        else:
            sig.append((path, type(leaf)))
    return (
        tuple(sig),
        torch.is_inference_mode_enabled(),
        torch.is_grad_enabled(),
        _autocast_state(torch),
    )


def _shared_states(pairs: list) -> tuple:
    found: dict = {}
    for pair in pairs:
        for sub in pair:
            for value in _shared_state_values(sub):
                found.setdefault(id(value), value)
    return tuple(found.values())


def _shared_state_key(states: tuple) -> tuple:
    key = []
    for state in states:
        try:
            key.append(state.compile_key())
        except Exception:  # noqa: BLE001 - an unreadable state is never warm
            key.append(object())
    return tuple(key)


def _note_warm(seen: set, sig: tuple) -> None:
    if len(seen) >= 256:
        seen.clear()
    seen.add(sig)


def _background_compile_call() -> bool:
    try:
        from .diffusion_bg_compile import capture_suppressed, eager_forced
    except Exception:  # noqa: BLE001
        return False
    return bool(eager_forced() or capture_suppressed())


class CfgParallelRouter:
    """Installed as the denoiser's ``forward``: routes odd branches of a step to the replica.

    Once a step has shown two branches, the next step's first odd branch is predicted from the previous one (same
    structure; the latent, and every tensor the previous odd call shared with its primary call, replaced by this
    step's; the rest -- negative embeds, masks -- kept) and launched on the replica BEFORE the primary branch runs, so
    the two cards overlap even when the primary forward blocks the host. The real call must then pass exactly the
    predicted tensors (identity) and equal plain values, else the prediction is discarded and the branch runs
    unpredicted.

    A whole-step CUDA graph installs under the router (``inner``); split callers (Z-Image) use :meth:`run_pair`."""

    _unsloth_cfg_router_layer = True

    def __init__(
        self,
        module: Any,
        *,
        primary: Any,
        secondary: Any,
        replica_factory: Optional[Callable[[], Any]] = None,
        logger: Any = None,
    ):
        torch = _torch()
        self.module_ref = weakref.ref(module)
        self.orig_forward = module.forward
        # FLUX.3's modular denoise step passes only kwargs named in forward's signature: expose the wrapped one.
        try:
            import inspect
            self.__signature__ = inspect.signature(self.orig_forward)
        except (TypeError, ValueError):
            pass
        self.inner: Any = None
        self._pairing = False
        self.had_override = "forward" in module.__dict__
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        self.logger = logger
        self.replica: Any = None
        self.replica_factory = replica_factory or (lambda: build_replica(module, self.secondary))
        self.worker: Optional[_ReplicaWorker] = None
        self.disabled_reason: Optional[str] = None
        self.prev_latent: Any = None
        self.branch = 0
        self.cfg_seen = False
        self.singles = 0
        self.prefetched: dict[int, tuple[Any, Any]] = {}
        # Tensors the odd branches of the previous step read (negative embeds, masks): copied again before the next
        # step's primary branch is queued, so they are always fresh and never wait behind it.
        self.carry: list[Any] = []
        self.next_carry: list[Any] = []
        self.stats = {
            "replica_calls": 0,
            "primary_calls": 0,
            "steps_parallel": 0,
            "replica_build_s": None,
            "replica_rebuilds": 0,
            "predicted_hits": 0,
            "predicted_misses": 0,
            "opaque_primary_calls": 0,
            "pairs_parallel": 0,
            "exclusion_skips": 0,
            "kv_cache_copies": 0,
            "kv_cache_declined": 0,
            "kv_prefill_primary_calls": 0,
            "replica_failures": 0,
            "batch_splits": 0,
            "batch_split_skip": None,
            "sequence_splits": 0,
            "sequence_split_skip": None,
            "sequence_split_released": None,
        }
        # why this load's calls are never split by tokens (set once a split failed), else None
        self.sequence_split_off: Optional[str] = None
        # the last render ran one call per step (no guidance branches): a batch can split from its first step
        self.last_render_single = False
        # source prefix-KV cache id -> (source cache, the tensors it held when copied, the replica's copy)
        self.kv_mirrors: dict[int, tuple[Any, list, Any]] = {}
        self.kv_declined: dict[int, tuple[Any, list]] = {}
        self._opaque_logged = False
        # Set when replica work was refused by a long recording: until that capture ends, later replica attempts go
        # straight to the source instead of each waiting OUTSIDE_WAIT_S again.
        self.exclusion_refused = False
        self.template: Optional[tuple[Any, Any, dict]] = None
        self.pending: Optional[dict] = None
        self.b0_leaves: dict = {}
        self.fingerprint: Optional[int] = None
        self.probe: Optional[bytes] = None
        self.probe_index: dict = {}
        self.probed_this_render = False
        self.pairs: list = []
        # Shared per-step states' compile guards join every warm-call signature.
        self.shared_states: tuple = ()
        # A call new to either side may compile (FX tracing is process-wide up to torch 2.13): it never overlaps the other side.
        self.warm_primary: set = set()
        self.warm_replica: set = set()
        self._pair_job: Optional[dict] = None
        # why the denoiser's layers scale activations per call (a split batch would change them), read once
        self._split_scaled: Optional[tuple] = None

    def describe(self) -> dict[str, Any]:
        return {
            "secondary": str(self.secondary),
            "armed": self.disabled_reason is None,
            "replica_built": self.replica is not None,
            "replica_compiled_submodules": getattr(self.replica, "_unsloth_replica_compiled", None),
            "disabled_reason": self.disabled_reason,
            **self.stats,
        }

    def disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            if (
                reason.startswith("replica ")
                or "no longer idle" in reason
                or "carries hooks" in reason
            ):
                self.stats["replica_failures"] += 1
                # the load's CFG batching lever takes over after this render (end_cfg_render), not unbatched branches
                self.fallback_pending = getattr(self, "fallback", None) is not None
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.multigpu: CFG-parallel disabled (%s); one card from here", reason
                )
        self.drop_replica()
        if self.stats["replica_failures"] and self.secondary.type == "cuda":
            # a replica that ran out of memory leaves its blocks cached on the card it borrowed: give them back
            with contextlib.suppress(Exception):
                from .diffusion_cuda_graph import outside_capture
                with outside_capture(), _device_context(self.secondary):
                    _torch().cuda.empty_cache()

    def drop_replica(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._drop_pending()
        self.replica = None
        self.kv_mirrors.clear()
        self.kv_declined.clear()
        self.pairs = []
        self.shared_states = ()
        self.warm_primary.clear()
        self.warm_replica.clear()
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}

    def end_render(self) -> None:
        self._drop_pending()
        self.prefetched.clear()
        self.kv_mirrors.clear()
        self.kv_declined.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}
        self.last_render_single = self.singles >= 1 and not self.cfg_seen
        self.branch = 0
        # the next render shows its own guidance shape: no speculative branch before its first odd call
        self.cfg_seen = False
        self.singles = 0
        self.probed_this_render = False

    def _note_refusal(self, exc: BaseException) -> None:
        from .diffusion_cuda_graph import OutsideWorkRefused
        if isinstance(exc, OutsideWorkRefused):
            self.exclusion_refused = True

    def _refused_this_capture(self) -> bool:
        """Whether replica work was refused and the capture that refused it is still recording."""
        if not self.exclusion_refused:
            return False
        from .diffusion_cuda_graph import capture_in_progress

        if capture_in_progress():
            self.stats["exclusion_skips"] += 1
            return True
        self.exclusion_refused = False
        return False

    def _signature(self, args: tuple, kwargs: dict) -> tuple:
        """``_call_signature`` plus guarded model state: an NVFP4 protect flip recompiles under seen args."""
        return (*_call_signature(args, kwargs), _shared_state_key(self.shared_states))

    def _primary(self, *args: Any, **kwargs: Any) -> Any:
        inner = self.inner
        return (inner if inner is not None else self.orig_forward)(*args, **kwargs)

    def _arm(self, args: tuple, kwargs: dict) -> None:
        """AdaLN precompute rows for a source call on inputs other than the outer call's (a batch half): the pre-hook
        armed the whole batch."""
        module = self.module_ref()
        if module is not None and "_unsloth_adaln" in module.__dict__:
            from .diffusion_adaln_precompute import arm_call
            arm_call(module, args, kwargs)

    def _primary_noting(self, sig: Optional[tuple], args: tuple, kwargs: dict) -> Any:
        sig = self._signature(args, kwargs) if sig is None else sig
        job = self._pair_job
        if sig not in self.warm_primary and job is not None:
            job["event"].wait()  # may compile: not while the replica runs
        out = self._primary(*args, **kwargs)
        _note_warm(self.warm_primary, sig)
        return out

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._pairing:
            self.stats["primary_calls"] += 1
            return self._primary_noting(None, args, kwargs)
        if self.disabled_reason is not None or _background_compile_call():
            # Disabled: no branch state to keep. A render that must run eager while its compile is in flight, or the
            # background compile thread itself replaying recorded inputs: both branches stay on the source, which
            # keeps that render bit-identical to an eager one and leaves the router's per-render state untouched.
            self.stats["primary_calls"] += 1
            return self._primary(*args, **kwargs)
        latent = _latent_arg(args, kwargs)
        sig = None
        if latent is not None and self.prev_latent is not None and latent is self.prev_latent:
            self.branch += 1
        else:
            if self.branch == 0 and self.prev_latent is not None:
                self.singles += 1
                if self.singles >= 3:
                    self.cfg_seen = False
            self.branch = 0
            self.prev_latent = latent
            self.prefetched.clear()
            self.carry, self.next_carry = self.next_carry, []
            if self.replica is not None and self.disabled_reason is None:
                self._drop_if_stale()
            self._drop_pending()
            self.b0_leaves = dict(_leaves((args, kwargs)))
            if self.cfg_seen and self.replica is not None and self.disabled_reason is None:
                # Copy inputs before queuing the primary; a later copy serialises the branches.
                self._prefetch(args, kwargs)
                sig = self._signature(args, kwargs)
                self._launch_prediction(latent, sig)
        if (
            self.branch == 0
            and not self.cfg_seen
            and (self.singles >= 1 or self.last_render_single)
            and self.disabled_reason is None
        ):
            out = self._batch_split(latent, args, kwargs)
            if out is None:
                out = self._sequence_split(latent, args, kwargs)
            if out is not None:
                return out
        prefill = self.branch % 2 == 1 and _is_prefill(kwargs)
        if (
            self.branch % 2 == 1
            and self.disabled_reason is None
            and not prefill
            and _holds_opaque_tensors(_without_prefix_kv(args, kwargs))
        ):
            self._drop_pending()
            self.template = None
            self.stats["opaque_primary_calls"] += 1
            if not self._opaque_logged and self.logger is not None:
                self._opaque_logged = True
                self.logger.info(
                    "diffusion.multigpu: a guidance branch carries tensors inside an object the replica cannot copy;"
                    " that branch stays on %s",
                    self.primary,
                )
        elif self.branch % 2 == 1 and self.disabled_reason is None:
            self.singles = 0
            if self.replica is None:
                self.cfg_seen = True
                self._build_replica(latent)
                if self.branch == 1 and not prefill:
                    self._record_template(args, kwargs, latent)
            elif prefill:
                self._drop_pending()
                self.template = None
            elif self.cfg_seen and self._refused_this_capture():
                # this step's prefetch or prediction was refused: the branch runs on the source, no further wait
                self._drop_pending()
            elif self.cfg_seen:
                if self.branch == 1:
                    self._record_template(args, kwargs, latent)
                try:
                    out = self._take_prediction(args, kwargs) if self.branch == 1 else None
                    if out is None:
                        out = self._run_replica(args, kwargs)
                    self.stats["replica_calls"] += 1
                    if self.branch == 1:
                        self.stats["steps_parallel"] += 1
                    return out
                except _KVMirrorDeclined:
                    self._drop_pending()
                except Exception as exc:  # noqa: BLE001 - this branch is recomputed on the primary below
                    self._note_refusal(exc)
                    self.disable(f"replica call failed: {type(exc).__name__}: {exc}")
            self.cfg_seen = True
        if prefill:
            self.stats["kv_prefill_primary_calls"] += 1
        self.stats["primary_calls"] += 1
        return self._primary_noting(sig, args, kwargs)

    def _batch_split(self, latent: Any, args: tuple, kwargs: dict) -> Any:
        """A step without guidance branches (a distilled model, or both branches batched into one call) whose batch is
        even: the first half runs on the source, the second on the replica at the same time. Only for denoiser classes
        whose every batched input is known (``_DIT_BATCH_SPLIT_KWARGS`` / ``_ALL``); anything else, an odd
        batch, or the kill switch keeps the call whole. None = not split (the caller runs it whole)."""
        torch = _torch()
        module = self.module_ref()
        allowed, joins_all = _batch_split_kwargs(module)
        if allowed is None or _batch_split_disabled() or torch.compiler.is_compiling():
            return None
        if getattr(module, "_compiled_call_impl", None) is not None:
            return self._skip_split("denoiser is compiled whole")
        n = latent.shape[0] if isinstance(latent, torch.Tensor) and latent.dim() >= 1 else 0
        if n < 2 or n % 2:
            return None
        if args:
            return self._skip_split("positional inputs")
        extra = sorted(
            k for k, v in kwargs.items() if k not in allowed and v is not None and v is not False
        )
        if extra:
            return self._skip_split(f"unknown inputs {extra}")
        if kwargs.get("joint_attention_kwargs") or (joins_all and kwargs.get("attention_kwargs")):
            return self._skip_split("attention kwargs")
        if self._split_scaled is None:
            from .diffusion_cfg_batch import per_call_scaled
            self._split_scaled = (per_call_scaled(module),)
        if self._split_scaled[0] is not None:
            # each half would take its own activation amax (dynamic NVFP4): not the one-card computation
            return self._skip_split(f"per-call activation scaling ({self._split_scaled[0]})")
        try:
            (a0, k0), (a1, k1) = [
                (_split_batch(args, n, h), _split_batch(kwargs, n, h)) for h in (0, 1)
            ]
        except TypeError as exc:
            return self._skip_split(f"unsplittable input ({exc})")

        # inside the denoiser's own call: its root hooks already wrap the whole batch, so run_pair need not refuse them
        def first_half() -> Any:
            self._arm(a0, k0)
            return self._primary_noting(None, a0, k0)

        res = self.run_pair(first_half, a1, k1, inside_call = True)
        if res is None:
            self._skip_split(f"pair not run ({self.disabled_reason or 'replica unavailable'})")
            self._arm(args, kwargs)  # the caller runs the whole batch: its rows, not a half's
        # run_pair forgets the step: keep this one's latent so the next call is read as a new step
        self.prev_latent = latent
        self.branch = 0
        if res is None:
            return None
        first, second = res
        if second is None:
            self._arm(a1, k1)
            second = self._primary(*a1, **k1)
        if joins_all:
            try:
                joined = _join_outputs(first, second)
            except TypeError as exc:
                return self._skip_split(f"unjoinable output ({exc})")
        else:
            joined = None
        self.stats["primary_calls"] += 1
        self.stats["batch_splits"] += 1
        return joined if joins_all else _join_batch(first, second)

    def _skip_split(self, reason: str) -> None:
        self.stats["batch_split_skip"] = reason
        return None

    def _sequence_split(self, latent: Any, args: tuple, kwargs: dict) -> Any:
        """A step without guidance branches that the batch split could not take (a batch of one): the source runs the
        first half of the tokens and the replica the second, swapping heads at each self-attention
        (``diffusion_sequence_parallel``). Only for denoiser classes with a checked plan; None = not split (the caller
        runs it whole on the source, which is also what a failed split falls back to for the rest of the load)."""
        from . import diffusion_sequence_parallel as sp

        module = self.module_ref()
        if module is None:
            return None
        if self.sequence_split_off is not None:
            self.stats["sequence_split_skip"] = self.sequence_split_off
            return None
        why = sp.refusal(module, sp.call_tokens(module, latent, kwargs))
        if why is None and (args or _holds_opaque_tensors((args, kwargs))):
            why = "positional or opaque inputs"
        if why is None:
            why = sp.call_refusal(module, kwargs)
        if why is None:
            why = sp.pair_refusal(self.primary, self.secondary, type(module).__name__)
        if why is None:
            if self._split_scaled is None:
                from .diffusion_cfg_batch import per_call_scaled
                self._split_scaled = (per_call_scaled(module),)
            if self._split_scaled[0] is not None:
                # each token half would take its own activation amax (dynamic NVFP4): not the one-card computation
                why = f"per-call activation scaling ({self._split_scaled[0]})"
        if why is not None:
            self.stats["sequence_split_skip"] = why
            return None
        self._drop_pending()
        self.template = None
        if self.replica is None and self.disabled_reason is None:
            self._build_replica(latent)
        if self.replica is None or self.worker is None or self.disabled_reason is not None:
            self.stats["sequence_split_skip"] = f"replica unavailable ({self.disabled_reason})"
            self.prev_latent = latent
            self.branch = 0
            return None
        self.prefetched.clear()
        _mirror_hook_state(self.pairs)
        try:
            r_kwargs = self._to_replica(kwargs)
        except Exception as exc:  # noqa: BLE001 - e.g. no room on the replica card: this step runs whole
            self.disable(f"replica input copy failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self.next_carry = []
            self.prefetched.clear()
        worker = self.worker
        try:
            out = sp.run(
                module,
                self.replica,
                worker.submit,
                _ReplicaWorker.wait,
                primary_forward = self.orig_forward,
                args = (),
                kwargs = kwargs,
                replica_args = (),
                replica_kwargs = r_kwargs,
                devices = (self.primary, self.secondary),
            )
        except Exception as exc:  # noqa: BLE001 - the step reruns whole on the source; no later step splits
            from .diffusion_batched import is_oom_error

            cause = exc.__cause__ if isinstance(exc, sp.SequenceParallelError) else None
            shown = cause if cause is not None else exc
            self.sequence_split_off = f"a split call failed ({type(shown).__name__}: {shown})"[:300]
            self.stats["sequence_split_skip"] = self.sequence_split_off
            if is_oom_error(exc):
                # out of memory on either card: free the replica's copy too, so the rerun (and anything after it) runs
                # with the memory main would have; an OOM in the rerun reaches the caller as on main
                self.stats["replica_failures"] += 1
                self.drop_replica()
                with contextlib.suppress(Exception):
                    from .diffusion_cuda_graph import outside_capture
                    for card in (self.secondary, self.primary):
                        if card.type == "cuda":
                            with outside_capture(), _device_context(card):
                                _torch().cuda.empty_cache()
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.multigpu: sequence-parallel step failed (%s); this load runs its single calls on %s",
                    exc,
                    self.primary,
                )
            self.prev_latent = latent
            self.branch = 0
            return None
        # a new step's latent is told apart by identity: keep this one so the next call reads as a new step
        self.prev_latent = latent
        self.branch = 0
        self.stats["primary_calls"] += 1
        self.stats["replica_calls"] += 1
        self.stats["steps_parallel"] += 1
        self.stats["sequence_splits"] += 1
        return out

    def run_pair(
        self,
        primary_call: Callable[[], Any],
        args: tuple,
        kwargs: dict,
        *,
        inside_call: bool = False,
    ) -> Optional[tuple]:
        """Second half on the replica, ``primary_call`` on the source. ``(first, second)``; None = run both on source;
        ``second`` None = replica failed. ``inside_call``: called from within the denoiser's own ``__call__`` (batch
        split), whose root hooks already ran around the whole call."""
        if self.disabled_reason is not None or _background_compile_call():
            return None
        module = self.module_ref()
        if module is None or _holds_opaque_tensors((args, kwargs)):
            return None
        if not inside_call and any(getattr(module, attr, None) for attr in _ROOT_HOOK_DICTS):
            return None
        self._drop_pending()
        self.template = None
        self.prev_latent = None
        self.branch = 0
        if self.replica is not None:
            self._drop_if_stale()
        if self.replica is None and self.disabled_reason is None:
            self._build_replica(_latent_arg(args, kwargs))
        if self.replica is None or self.worker is None or self.disabled_reason is not None:
            return None
        if self._refused_this_capture():
            return None
        self.prefetched.clear()
        hook_state = _mirror_hook_state(self.pairs)
        sig = self._signature(args, kwargs)
        try:
            r_args = self._to_replica(args)
            r_kwargs = self._to_replica(kwargs)
        except _KVMirrorDeclined:
            return None  # the prefix cache does not fit on the replica card: both halves run on the source
        except Exception as exc:  # noqa: BLE001 - e.g. no room left on the replica card: both halves run on the source
            self.disable(f"replica input copy failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self.next_carry = []
            self.prefetched.clear()
        job = self.worker.submit(_replica_call(self.replica, r_args, r_kwargs))
        if sig not in self.warm_replica:
            job["event"].wait()  # may compile: the primary half waits
        self._pairing = True
        self._pair_job = job
        try:
            first = primary_call()
        except BaseException:
            with contextlib.suppress(BaseException):
                _ReplicaWorker.wait(job)
            raise
        finally:
            self._pairing = False
            self._pair_job = None
        try:
            second = _ReplicaWorker.wait(job)
            if _hook_state(self.pairs) != hook_state:
                # source hook state moved during the first half: rerun
                _mirror_hook_state(self.pairs)
                second = self.worker.call(_replica_call(self.replica, r_args, r_kwargs))
            second = _to_primary(second, self.primary)
        except Exception as exc:  # noqa: BLE001 - the caller recomputes the second half on the source
            self.disable(f"replica call failed: {type(exc).__name__}: {exc}")
            return first, None
        _note_warm(self.warm_replica, sig)
        self.stats["replica_calls"] += 1
        self.stats["steps_parallel"] += 1
        self.stats["pairs_parallel"] += 1
        return first, second

    def _record_template(self, args: tuple, kwargs: dict, latent: Any) -> None:
        torch = _torch()
        b0_by_id = {id(v): p for p, v in self.b0_leaves.items() if isinstance(v, torch.Tensor)}
        kinds: dict = {}
        for path, leaf in _leaves((args, kwargs)):
            if isinstance(leaf, torch.Tensor):
                if leaf is latent:
                    kinds[path] = ("latent",)
                elif id(leaf) in b0_by_id and self.b0_leaves.get(b0_by_id[id(leaf)]) is leaf:
                    kinds[path] = ("b0", b0_by_id[id(leaf)])
                elif (
                    isinstance(self.b0_leaves.get(path), torch.Tensor)
                    and leaf.numel() <= _SMALL_TENSOR_NUMEL
                    and _bitwise_equal(leaf, self.b0_leaves[path])
                ):
                    kinds[path] = ("b0eq", path)
                else:
                    kinds[path] = ("const", leaf)
            else:
                kinds[path] = ("value", leaf)
        self.template = ((args, kwargs), kinds)

    def _predict(self, latent: Any) -> Optional[tuple]:
        if self.template is None or latent is None:
            return None
        structure, kinds = self.template
        values: dict = {}
        for path, kind in kinds.items():
            if kind[0] == "latent":
                values[path] = latent
            elif kind[0] in ("b0", "b0eq"):
                if kind[1] not in self.b0_leaves:
                    return None
                values[path] = self.b0_leaves[kind[1]]
            else:
                values[path] = kind[1]
        try:
            return _rebuild(structure, values)
        except KeyError:
            return None

    def _launch_prediction(self, latent: Any, sig: tuple) -> None:
        if self._refused_this_capture():
            return
        predicted = self._predict(latent)
        if predicted is None or self.worker is None:
            return
        p_args, p_kwargs = predicted
        if _is_prefill(p_kwargs):
            return
        if (
            sig not in self.warm_primary
            or self._signature(p_args, p_kwargs) not in self.warm_replica
        ):
            return  # a side may compile: the odd branch runs unpredicted, after the primary one
        try:
            r_args = self._to_replica(p_args)
            r_kwargs = self._to_replica(p_kwargs)
        except Exception as exc:  # noqa: BLE001 - no prediction this step; the branch runs unpredicted
            self._note_refusal(exc)
            return
        torch = _torch()
        # Bitwise-compared snapshots catch in-place edits; version counters miss inference tensors.
        snapshots = []
        for (_path, src), (_rpath, used) in zip(
            _leaves((p_args, p_kwargs)), _leaves((r_args, r_kwargs))
        ):
            if isinstance(src, torch.Tensor):
                snapshots.append((src, used.clone() if used is src else used))
        replica = self.replica
        _structure, kinds = self.template
        hook_state = _mirror_hook_state(self.pairs)
        self.pending = {
            "hook_state": hook_state,
            "leaves": _leaves((p_args, p_kwargs)),
            "by_value": {path for path, kind in kinds.items() if kind[0] == "b0eq"},
            "snapshots": snapshots,
            "job": self.worker.submit(_replica_call(replica, r_args, r_kwargs)),
        }

    def _take_prediction(self, args: tuple, kwargs: dict) -> Any:
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        actual = _leaves((args, kwargs))
        predicted = pending["leaves"]
        by_value = pending["by_value"]
        # Host-side checks first (paths, plain values), then every by-value tensor pair under one device read.
        ok = (
            len(actual) == len(predicted)
            and all(
                pa == pp and (pa in by_value or _same_value(va, vp))
                for (pa, va), (pp, vp) in zip(actual, predicted)
            )
            and _bitwise_equal_all(
                (va, vp) for (pa, va), (_pp, vp) in zip(actual, predicted) if pa in by_value
            )
        )
        # the source's root hooks have now run for THIS call: the replica must have seen the same per-call state
        ok = ok and _hook_state(self.pairs) == pending["hook_state"]
        ok = ok and _unchanged_since_launch(pending["snapshots"])
        try:
            out = _ReplicaWorker.wait(pending["job"])
        except Exception:
            if ok:
                raise
            out = None
        if not ok:
            self.stats["predicted_misses"] += 1
            self.prefetched.clear()
            return None
        self.stats["predicted_hits"] += 1
        return _to_primary(out, self.primary)

    def _drop_pending(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            with contextlib.suppress(Exception):
                _ReplicaWorker.wait(pending["job"])
            self.stats["predicted_misses"] += 1

    def _drop_if_stale(self) -> None:
        module = self.module_ref()
        if module is None:
            return
        try:
            current = weights_fingerprint(module)
        except Exception as exc:  # noqa: BLE001 - unverifiable weights: never risk a stale replica
            self.disable(f"cannot fingerprint the denoiser: {exc}")
            return
        if current == self.fingerprint and not self.probed_this_render:
            from .diffusion_cuda_graph import OutsideWorkRefused, outside_capture
            try:
                with outside_capture():
                    probe = weights_probe(module, self.probe_index)
            except OutsideWorkRefused:
                probe = self.probe  # another pipeline is recording: re-read on the next step
            except Exception as exc:  # noqa: BLE001 - unverifiable weights: never risk a stale replica
                self.disable(f"cannot probe the denoiser weights: {exc}")
                return
            else:
                self.probed_this_render = True
            if probe != self.probe:
                current = None
        if current != self.fingerprint:
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: denoiser weights changed; rebuilding the CFG-parallel replica"
                )
            self.stats["replica_rebuilds"] += 1
            self.drop_replica()
            return
        # A step cache (transformer_cache=auto) may be installed on the source at generate time, after the replica
        # was built: the copy would then run uncached while the source's cache saw only the even branches.
        hooked = _hooks_present(module)
        if hooked is not None:
            self.disable(f"denoiser gained hooks after the replica was built ({hooked})")

    def _build_replica(self, latent: Any = None) -> None:
        module = self.module_ref()
        if module is None:
            return
        from .diffusion_cuda_graph import OutsideWorkRefused, outside_capture

        t0 = time.perf_counter()
        try:
            hooked = _hooks_present(module)
            if hooked is not None:
                raise RuntimeError(f"denoiser carries hooks ({hooked})")
            need_mib = replica_need_mib(module, latent_tokens(module, latent))
            automatic = getattr(self, "automatic_idle_only", False)
            # Outside the load scope: read the card directly, before the memory read opens our context (3-7% busy).
            own_mib = int(getattr(self, "own_mib", 0) or 0)
            util = (
                _settled_util_pct(
                    self.secondary.index, _smi_util_pct(self.secondary.index), own_mib
                )
                if automatic
                else None
            )
            free, total = self._secondary_free_total_mib()
            if automatic:
                if not _idle_card(
                    DeviceProbe(
                        ordinal = self.secondary.index,
                        name = "",
                        total_mib = total,
                        free_mib = free,
                        util_pct = util,
                    ),
                    own_mib,
                ):
                    # Automatic borrowed this card because it was idle at load; something else took it since
                    raise RuntimeError(
                        f"{self.secondary} is no longer idle ({free} of {total} MiB free, {util}% busy)"
                    )
            reserve = _replica_reserve_mib(free, total)
            encode = 0
            if getattr(self, "encoder_card", False):
                ref = getattr(self, "pipe_ref", None)
                encode = beside_encoder_extra_mib(ref() if ref else None, module, need_mib, reserve)
            if free is not None and need_mib + encode + reserve > free:
                beside = f" and {encode} MiB more for the next encode beside it" if encode else ""
                raise RuntimeError(
                    f"replica needs {need_mib} MiB{beside}, {free} MiB free on {self.secondary}"
                )
            self.fingerprint = weights_fingerprint(module)
            # Copying the weights to the replica card is CUDA work another pipeline's capture prohibits.
            with outside_capture(), _device_context(self.secondary):
                self.probe = weights_probe(module, self.probe_index)
                self.replica = self.replica_factory()
            self.probed_this_render = True
            self.pairs = list(zip(module.modules(), self.replica.modules()))
            self.shared_states = _shared_states(self.pairs)
            self.worker = _ReplicaWorker(self.secondary)
            weakref.finalize(self, self.worker.close)
            self.stats["replica_build_s"] = round(time.perf_counter() - t0, 3)
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: CFG-parallel replica on %s built in %.2fs",
                    self.secondary,
                    time.perf_counter() - t0,
                )
        except Exception as exc:  # noqa: BLE001
            self.replica = None
            # A refused build allocated nothing, and its capture is still recording: no allocator call may run now.
            # Any other failure frees what it left behind, itself with captures held off.
            if not isinstance(exc, OutsideWorkRefused):
                with contextlib.suppress(Exception):
                    with outside_capture():
                        _torch().cuda.empty_cache()
            self.disable(f"replica unavailable: {exc}")

    def _prefetch(self, args: tuple, kwargs: dict) -> None:
        from .diffusion_cuda_graph import outside_capture
        if self._refused_this_capture():
            return
        try:
            with outside_capture():
                for tensor in (*_iter_tensors((args, kwargs)), *self.carry):
                    if tensor.device == self.secondary or id(tensor) in self.prefetched:
                        continue
                    self.prefetched[id(tensor)] = (tensor, tensor.to(self.secondary))
        except Exception as exc:  # noqa: BLE001 - only an optimisation: no prefetch, the call copies its inputs itself
            self.prefetched.clear()
            self._note_refusal(exc)

    def _to_replica(self, obj: Any) -> Any:
        torch = _torch()

        def one(tensor: Any) -> Any:
            if tensor is not self.prev_latent and not any(tensor is t for t in self.next_carry):
                self.next_carry.append(tensor)
            if tensor.device == self.secondary:
                return tensor
            hit = self.prefetched.get(id(tensor))
            if hit is not None and hit[0] is tensor:
                return hit[1]
            copied = tensor.to(self.secondary)
            self.prefetched[id(tensor)] = (tensor, copied)
            return copied

        def walk(value: Any, depth: int = 0) -> Any:
            if isinstance(value, torch.Tensor):
                return one(value)
            if depth == 0 and isinstance(value, dict) and _prefix_kv_mode(value) == "cached":
                return type(value)(
                    (k, self._kv_mirror(v) if k == _KV_KWARG else walk(v, depth + 1))
                    for k, v in value.items()
                )
            if depth < 4 and isinstance(value, tuple) and not hasattr(value, "_fields"):
                return tuple(walk(v, depth + 1) for v in value)
            if depth < 4 and isinstance(value, list):
                return [walk(v, depth + 1) for v in value]
            if depth < 4 and isinstance(value, dict):
                return type(value)((k, walk(v, depth + 1)) for k, v in value.items())
            return value

        from .diffusion_cuda_graph import outside_capture

        with outside_capture():
            return walk(obj)

    def _kv_mirror(self, cache: Any) -> Any:
        """Replica copy of a filled prefix-KV cache, reused while the source holds the same K/V tensors."""
        held = _cache_tensors(cache) or []
        entry = self.kv_mirrors.get(id(cache))
        if (
            entry is not None
            and entry[0] is cache
            and len(entry[1]) == len(held)
            and all(a is b for a, b in zip(entry[1], held))
        ):
            return entry[2]
        declined = self.kv_declined.get(id(cache))
        if (
            declined is not None
            and declined[0] is cache
            and len(declined[1]) == len(held)
            and all(a is b for a, b in zip(declined[1], held))
        ):
            raise _KVMirrorDeclined("prefix cache does not fit on the replica card")
        need = sum(t.numel() * t.element_size() for t in held)
        free, total = self._secondary_free_total_mib()
        if free is not None and need + (_replica_reserve_mib(free, total) << 20) > free * (1 << 20):
            self.kv_declined[id(cache)] = (cache, held)
            self.stats["kv_cache_declined"] += 1
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: prefix KV cache (%d MiB) does not fit on %s (%d MiB free);"
                    " its guidance branch stays on %s this render",
                    need >> 20,
                    self.secondary,
                    free,
                    self.primary,
                )
            raise _KVMirrorDeclined("prefix cache does not fit on the replica card")
        secondary = self.secondary
        mirror = _walk_cache(cache, lambda t: t if t.device == secondary else t.to(secondary), [])
        while len(self.kv_mirrors) >= 4:
            self.kv_mirrors.pop(next(iter(self.kv_mirrors)))
        self.kv_mirrors[id(cache)] = (cache, held, mirror)
        self.stats["kv_cache_copies"] += 1
        return mirror

    def _secondary_free_total_mib(self) -> tuple[Optional[int], Optional[int]]:
        """(free, total) MiB on the replica card; free counts the allocator's cached-but-unused blocks, minus CUDA graph pools."""
        free, total = _free_total_mib(self.secondary.index)
        if free is not None:
            with contextlib.suppress(Exception):
                from .diffusion_memory import cuda_reusable_cache_bytes
                free += cuda_reusable_cache_bytes(self.secondary) // (1 << 20)
        return free, total

    def _run_replica(self, args: tuple, kwargs: dict) -> Any:
        # Unpredicted: inputs copied after the primary; in-place edits make the copy stale.
        self.prefetched.clear()
        _mirror_hook_state(self.pairs)
        sig = self._signature(args, kwargs)
        r_args = self._to_replica(args)
        r_kwargs = self._to_replica(kwargs)
        replica = self.replica
        out = self.worker.call(_replica_call(replica, r_args, r_kwargs))
        _note_warm(self.warm_replica, sig)
        return _to_primary(out, self.primary)


class _KVMirrorDeclined(RuntimeError):
    """A prefix-KV cache too large for the replica card's free memory: the branch runs on the primary."""


def _replica_call(replica: Any, args: tuple, kwargs: dict) -> Callable[[], Any]:
    return lambda: replica(*args, **kwargs)


# U-Net keyword arguments a batch split may carry; any other tensor-bearing keyword keeps the call whole on the source.
_UNET_SPLIT_KWARGS = frozenset(
    {
        "encoder_hidden_states",
        "timestep_cond",
        "added_cond_kwargs",
        "cross_attention_kwargs",
        "down_block_additional_residuals",
        "mid_block_additional_residual",
        "return_dict",
    }
)


def _unet_cfg_parallel_disabled() -> bool:
    return (os.environ.get(UNET_CFG_PARALLEL_ENV) or "").strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    )


def _split_batch(
    value: Any,
    n: int,
    half: int,
    depth: int = 0,
) -> Any:
    """``value`` with every tensor of leading size ``n`` cut to its ``half`` (0 = first, 1 = second) along dim 0;
    other tensors and plain values are shared. Raises TypeError on an object it cannot see into."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and value.shape[0] == n:
            b = n // 2
            return value[:b] if half == 0 else value[b:]
        return value
    if value is None or isinstance(value, _PLAIN_TYPES):
        return value
    if depth > 4:
        raise TypeError("nested too deep")
    if isinstance(value, dict):
        return {k: _split_batch(v, n, half, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_split_batch(v, n, half, depth + 1) for v in value)
    raise TypeError(f"cannot split {type(value).__name__}")


def _join_batch(a: Any, b: Any) -> Any:
    torch = _torch()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.cat([a, b.to(a.device)], dim = 0)
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and a:
        return (_join_batch(a[0], b[0]),) + tuple(a[1:])
    sample_a, sample_b = getattr(a, "sample", None), getattr(b, "sample", None)
    if isinstance(sample_a, torch.Tensor) and isinstance(sample_b, torch.Tensor):
        return type(a)(sample = torch.cat([sample_a, sample_b.to(sample_a.device)], dim = 0))
    raise TypeError(f"unexpected U-Net output {type(a).__name__}")


class UnetCfgParallelRouter(CfgParallelRouter):
    """CFG-parallel for a U-Net (SDXL / SD 1.x / 2.x): diffusers sends both guidance branches through ONE call at batch
    ``2 * B``, so there is no second call to route. Each such call is cut in half along the batch: the first half runs
    on the source card, the second on the replica at the same time (:meth:`CfgParallelRouter.run_pair`), and the two
    outputs are concatenated. A U-Net treats every sample on its own, so the math is the same; the bits differ only
    where the batch-B kernels round differently from batch 2B. Any even batch splits (two images without guidance
    too, which then builds the replica). Odd batches, unknown inputs, a compiled-whole module and an unavailable
    replica run the whole call on the source. ``UNSLOTH_DIFFUSION_CFG_PARALLEL_UNET=0`` keeps U-Nets on one card."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        torch = _torch()
        module = self.module_ref()
        if (
            self._pairing
            or self.disabled_reason is not None
            or _background_compile_call()
            or torch.compiler.is_compiling()
            # compiled whole after the router was installed (a deferred speed profile): its graph owns the call
            or getattr(module, "_compiled_call_impl", None) is not None
        ):
            self.stats["primary_calls"] += 1
            return self._primary(*args, **kwargs)
        sample = kwargs.get("sample", args[0] if args else None)
        n = sample.shape[0] if isinstance(sample, torch.Tensor) and sample.dim() >= 1 else 0
        halves = None
        if (
            n >= 2
            and n % 2 == 0
            and set(k for k in kwargs if k not in ("sample", "timestep")) <= _UNET_SPLIT_KWARGS
        ):
            try:
                halves = [(_split_batch(args, n, h), _split_batch(kwargs, n, h)) for h in (0, 1)]
            except TypeError:
                halves = None
        if halves is None:
            self.stats["primary_calls"] += 1
            return self._primary(*args, **kwargs)
        (a0, k0), (a1, k1) = halves
        self.stats["primary_calls"] += 1
        # the primary half waits for a cold replica (a first call may compile) and marks its signature warm
        res = self.run_pair(lambda: self._primary_noting(None, a0, k0), a1, k1)
        if res is None:
            return self._primary(*args, **kwargs)
        first, second = res
        if second is None:
            second = self._primary(*a1, **k1)
        return _join_batch(first, second)


# Batch-cut DiT classes need per-sample dim-0 tensors (unbatched ids lead with the token count);
# Z-Image has its own split; Lumina 2 pads captions per half, masked (fp order only).
_DIT_BATCH_SPLIT_KWARGS = {
    "FluxTransformer2DModel": frozenset(
        {
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
            "joint_attention_kwargs",
            "return_dict",
        }
    ),
    "Flux2Transformer2DModel": frozenset(
        {
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
            "joint_attention_kwargs",
            "return_dict",
        }
    ),
    "Lumina2Transformer2DModel": frozenset(
        {
            "hidden_states",
            "timestep",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "return_dict",
        }
    ),
}
# Every tensor input is per sample on dim 0 or unbatched (ids / masks); LTX-2 returns (video, audio), both joined.
# Left out: HiDream (its llama states are layers x batch, batched on dim 1), MOSS-Image (positional inputs), Anima
# (Studio's wrapper renders one image per call, so no batch reaches the denoiser), FLUX.3 /
# Qwen-Image 2.1 (a per-branch prefix KV cache object), and an attention_kwargs that is not empty.
_DIT_BATCH_SPLIT_KWARGS_ALL = {
    "LTX2VideoTransformer3DModel": frozenset(
        {
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "audio_encoder_hidden_states",
            "timestep",
            "audio_timestep",
            "sigma",
            "audio_sigma",
            "encoder_attention_mask",
            "audio_encoder_attention_mask",
            "num_frames",
            "height",
            "width",
            "fps",
            "audio_num_frames",
            "video_coords",
            "audio_coords",
            "isolate_modalities",
            "use_cross_timestep",
            "attention_kwargs",
            "return_dict",
        }
    ),
    "ChromaTransformer2DModel": frozenset(
        {
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "img_ids",
            "txt_ids",
            "attention_mask",
            "joint_attention_kwargs",
            "return_dict",
        }
    ),
    "QwenImageTransformer2DModel": frozenset(
        {
            "hidden_states",
            "encoder_hidden_states",
            "encoder_hidden_states_mask",
            "timestep",
            "img_shapes",
            "guidance",
            "attention_kwargs",
            "return_dict",
        }
    ),
    "HunyuanImageTransformer2DModel": frozenset(
        {
            "hidden_states",
            "timestep",
            "timestep_r",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "encoder_hidden_states_2",
            "encoder_attention_mask_2",
            "guidance",
            "attention_kwargs",
            "return_dict",
        }
    ),
    "Krea2Transformer2DModel": frozenset(
        {
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "position_ids",
            "encoder_attention_mask",
            "attention_kwargs",
            "return_dict",
        }
    ),
}
BATCH_SPLIT_ENV = "UNSLOTH_DIFFUSION_BATCH_SPLIT"


def _batch_split_kwargs(module: Any) -> tuple[Optional[frozenset], bool]:
    """(the inputs a batch split of ``module`` may carry, or None; whether every output tensor is joined)."""
    name = type(module).__name__ if module is not None else None
    allowed = _DIT_BATCH_SPLIT_KWARGS.get(name)
    if allowed is not None:
        return allowed, False
    if name in _DIT_BATCH_SPLIT_KWARGS_ALL and multigpu_all_enabled():
        return _DIT_BATCH_SPLIT_KWARGS_ALL[name], True
    return None, False


def _join_outputs(
    a: Any,
    b: Any,
    depth: int = 0,
) -> Any:
    """Join two halves' outputs on dim 0: every tensor of a tuple / list / output dict."""
    torch = _torch()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.cat([a, b.to(a.device)], dim = 0)
    if (
        depth < 4
        and isinstance(a, (tuple, list))
        and isinstance(b, (tuple, list))
        and len(a) == len(b)
    ):
        joined = [_join_outputs(x, y, depth + 1) for x, y in zip(a, b)]
        return tuple(joined) if isinstance(a, tuple) else joined
    if depth < 4 and isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        return type(a)(**{k: _join_outputs(a[k], b[k], depth + 1) for k in a})
    if a is None and b is None:
        return None
    sample_a, sample_b = getattr(a, "sample", None), getattr(b, "sample", None)
    if isinstance(sample_a, torch.Tensor) and isinstance(sample_b, torch.Tensor):
        return type(a)(sample = torch.cat([sample_a, sample_b.to(sample_a.device)], dim = 0))
    raise TypeError(f"unexpected denoiser output {type(a).__name__}")


def _batch_split_disabled() -> bool:
    return (os.environ.get(BATCH_SPLIT_ENV) or "").strip().lower() in ("0", "off", "false", "no")


_CFG_ROUTERS: "weakref.WeakKeyDictionary[Any, CfgParallelRouter]" = weakref.WeakKeyDictionary()
_CFG_DECLINES: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def _cfg_denoiser(pipe: Any) -> Any:
    """The module CFG-parallel routes: ``pipe.transformer``, else a U-Net (batch split, :class:`UnetCfgParallelRouter`)
    unless ``UNSLOTH_DIFFUSION_CFG_PARALLEL_UNET=0``."""
    denoiser = getattr(pipe, "transformer", None)
    if denoiser is not None:
        return denoiser
    unet = getattr(pipe, "unet", None)
    if (
        unet is not None
        and type(unet).__name__ == "UNet2DConditionModel"
        and not _unet_cfg_parallel_disabled()
    ):
        if _batch_coupled_processors(unet):
            return None  # PAG-style processors chunk the batch themselves: a dim-0 split would mix their thirds
        return unet
    return None


def _batch_coupled_processors(unet: Any) -> bool:
    try:
        processors = getattr(unet, "attn_processors", None) or {}
        return any("PAG" in type(p).__name__ for p in processors.values())
    except Exception:  # noqa: BLE001 - unreadable: treat as coupled
        return True


def cfg_parallel_decline(
    pipe: Any, *, offload_policy: str, selection: Optional[MultiGpuSelection]
) -> Optional[str]:
    if selection is None:
        return "one card selected"
    mode = cfg_parallel_mode()
    if mode == "off":
        return f"{CFG_PARALLEL_ENV}=0"
    from .diffusion_layer_split_plan import is_layer_split

    # A replica is a whole denoiser on one card, which a denoiser cut across cards cannot be.
    if is_layer_split(pipe):
        return "denoiser is split across cards (layer split)"
    # A replica takes a whole denoiser's worth of another card: automatic placement borrows one only when it is the
    # same card model and idle at load (install_cfg_parallel keeps only those), or always under force.
    if selection.automatic and mode != "force" and automatic_cfg_parallel_off():
        return f"automatic GPU placement (select several cards or set {CFG_PARALLEL_ENV}=force)"
    from .diffusion_bg_compile import load_time_enabled as bg_compile_enabled

    if bg_compile_enabled():
        # its replays call the denoiser, which would enter the router and race a render's branch state
        return "background compile replays through the denoiser (UNSLOTH_DIFFUSION_BG_COMPILE)"
    denoiser = _cfg_denoiser(pipe)
    torch = _torch()
    if not isinstance(denoiser, torch.nn.Module):
        if (
            isinstance(getattr(pipe, "unet", None), torch.nn.Module)
            and _unet_cfg_parallel_disabled()
        ):
            return f"U-Net CFG-parallel off ({UNET_CFG_PARALLEL_ENV}=0)"
        return "no DiT denoiser (a U-Net batches its guidance branches)"
    from .diffusion_layer_split import is_layer_split

    if is_layer_split(denoiser):
        return "denoiser is split across cards (layer split)"
    if getattr(denoiser, "_unsloth_tp2", None) is not None:
        # its blocks already span both cards (diffusion_tp2): a replica would copy them a second time
        return "denoiser is tensor parallel across cards"
    # Whole-module / sequential offload page the denoiser per call; a group / streaming plan is fine only when it left
    # the denoiser itself resident (encoders streamed), which shows as a denoiser without offload hooks.
    if offload_policy in ("model", "sequential"):
        return f"denoiser is offloaded ({offload_policy})"
    hooked = _hooks_present(denoiser)
    if hooked is not None:
        return f"denoiser is offloaded or hooked ({offload_policy}: {hooked})"
    if getattr(pipe, "transformer_2", None) is not None:
        return "two-expert denoiser"
    if getattr(denoiser, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    device = _execution_device(denoiser)
    if device is None or device.type != "cuda":
        return "denoiser is not on a CUDA card"
    return None


def _resident_graph(module: Any) -> Any:
    slot = module.__dict__.get("forward")
    try:
        from .diffusion_cuda_graph import GraphedCompiledCall, GraphedForward
    except Exception:  # noqa: BLE001
        return None
    if (
        type(slot) is GraphedForward
        and slot.placement is None
        and not isinstance(slot, GraphedCompiledCall)
    ):
        return slot
    return None


def install_cfg_parallel(
    pipe: Any,
    *,
    offload_policy: str,
    logger: Any = None,
) -> Optional[str]:
    """Arm CFG-parallel on ``pipe.transformer`` (or a U-Net) when a secondary card can hold a replica. The replica is
    built on the first step that shows two branches, or, for a FLUX.1 / FLUX.2 denoiser, the first step of a render
    without branches whose batch is even (batch split): a distilled model rendering one image never pays for it.
    Automatic placement lends one card per load (:func:`automatic_partner`). Returns the decline reason or None."""
    selection = current_selection()
    reason = cfg_parallel_decline(pipe, offload_policy = offload_policy, selection = selection)
    if reason is None:
        denoiser = _cfg_denoiser(pipe)
        primary = _execution_device(denoiser)
        probes = probe_devices((selection.primary, *selection.secondaries))
        same = [
            o
            for o in selection.secondaries
            if probes.get(o) is not None
            and probes.get(primary.index) is not None
            and probes[o].name == probes[primary.index].name
        ]
        candidates = same if cfg_parallel_mode() != "force" else list(selection.secondaries)
        if cfg_parallel_mode() == "force":
            # force pairs different card models; a card below the primary must still run the replica's weights
            names = tuple(n for n in DENOISER_ATTRS if getattr(pipe, n, None) is denoiser) or (
                "transformer",
            )
            candidates = [
                o
                for o in candidates
                if placement_refusal(pipe, names, int(selection.primary), int(o)) is None
            ]
        automatic = selection.automatic and cfg_parallel_mode() != "force"
        auto_reason = None
        try:
            placed = dict(((_PLACEMENTS.get(pipe) or {}).get("placement") or {}).items())
        except TypeError:
            placed = {}
        # A card holding only this load's text encoders can take the replica when it fits beside them (they run
        # before the first step); a card holding the denoiser's other companions (the VAE decodes there) cannot.
        occupied = {str(d) for n, d in placed.items() if not _replica_shares_component(n)} - {
            str(primary)
        }
        shared = [
            n for n, d in placed.items() if _replica_shares_component(n) and str(d) != str(primary)
        ]
        own = {
            int(str(placed[n]).rpartition(":")[2]): 0
            for n in shared
            if str(placed[n]).startswith("cuda:") and str(placed[n]).rpartition(":")[2].isdigit()
        }
        own.update(_own_mib_on_cards(pipe, shared))
        if automatic:
            for o, mib in own.items():
                probe = probes.get(o)
                if probe is not None:
                    settled = _settled_util_pct(o, probe.util_pct, max(1, int(mib)))
                    if settled != probe.util_pct:
                        probes[o] = dataclasses.replace(probe, util_pct = settled)
        if automatic:
            # nothing else may live on a card Automatic borrows for a replica (busy: memory in use or computing)
            need = replica_need_mib(denoiser)
            beside = {o: beside_encoder_extra_mib(pipe, denoiser, need) for o in own}
            pick, auto_reason = automatic_partner(
                probes.get(primary.index) or DeviceProbe(primary.index, "", None, None),
                # a card holding the split's decode components cannot take the replica: never the one Automatic lends
                [
                    probes[o]
                    for o in selection.secondaries
                    if probes.get(o) is not None
                    and probes[o].free_mib is not None
                    and f"cuda:{o}" not in occupied
                ],
                need,
                own_mib = own,
                pending_mib = {o: mib for o, mib in beside.items() if mib},
            )
            candidates = [o for o in candidates if _idle_card(probes.get(o), own.get(o, 0))]
            if pick is not None:
                candidates = [pick]
        if occupied:
            candidates = [o for o in candidates if f"cuda:{o}" not in occupied]
        if not candidates and occupied:
            reason = f"the only replica card ({', '.join(sorted(occupied))}) holds the split's components"
        elif not candidates and automatic and same:
            reason = (
                f"automatic GPU placement: no idle card for a replica ({auto_reason}; idle = at most "
                f"{AUTO_CFG_PARALLEL_IDLE_USED_MIB} MiB in use and {AUTO_CFG_PARALLEL_IDLE_UTIL_PCT:g}% busy; "
                f"select several cards or set {CFG_PARALLEL_ENV}=force)"
            )
        elif not candidates:
            reason = "secondary card is a different model (bit-identity needs the same kernels)"
        else:
            need_mib = replica_need_mib(denoiser, activations = automatic)

            def extra(o: int) -> int:
                # beside this load's text encoders: room for their next encode too
                if o not in own or probes.get(o) is None:
                    return 0
                return beside_encoder_extra_mib(
                    pipe,
                    denoiser,
                    need_mib,
                    _replica_reserve_mib(probes[o].free_mib, probes[o].total_mib),
                )

            def usable(o: int) -> int:
                if probes.get(o) is None:
                    return -1
                free_o = probes[o].free_mib
                if free_o is None:
                    return -1
                return int(free_o) - _replica_reserve_mib(free_o, probes[o].total_mib) - extra(o)

            # a card holding nothing of this load first; one beside its encoders only when no other fits
            fitting = [
                o
                for o in candidates
                if probes.get(o) is not None
                and probes[o].free_mib is not None
                and need_mib
                + extra(o)
                + _replica_reserve_mib(probes[o].free_mib, probes[o].total_mib)
                <= probes[o].free_mib
            ]
            best = max(fitting or candidates, key = lambda o: (0 if o in own else 1, usable(o)))
            free = probes[best].free_mib
            if (
                free is not None
                and need_mib + extra(best) + _replica_reserve_mib(free, probes[best].total_mib)
                > free
            ):
                encode = (
                    f" and {extra(best)} MiB more for the next encode of the text encoder beside it"
                    if extra(best)
                    else ""
                )
                reason = f"replica needs {need_mib} MiB{encode}, {free} MiB free on cuda:{best}"
            else:
                graph = _resident_graph(denoiser)
                if graph is not None:
                    del denoiser.__dict__["forward"]
                router_cls = (
                    UnetCfgParallelRouter
                    if denoiser is not getattr(pipe, "transformer", None)
                    else CfgParallelRouter
                )
                router = router_cls(
                    denoiser, primary = primary, secondary = f"cuda:{best}", logger = logger
                )
                router.inner = graph
                router.automatic_idle_only = automatic
                # this load's own encoders on the replica card: not another process's use at the build's idle check
                router.own_mib = int(own.get(best, 0))
                router.encoder_card = best in own
                router.pipe_ref = weakref.ref(pipe) if _weakrefable(pipe) else None
                try:
                    _CFG_ROUTERS[pipe] = router
                except TypeError:
                    if graph is not None:
                        denoiser.__dict__["forward"] = graph
                    return "pipeline cannot be tracked (not weak-referenceable)"
                denoiser.forward = router
                denoiser._unsloth_cfg_router = router
                if logger is not None:
                    logger.info(
                        "diffusion.multigpu: CFG-parallel armed (replica on cuda:%s when guidance runs two branches or"
                        " an even batch splits%s)",
                        best,
                        "; U-Net batch split" if router_cls is UnetCfgParallelRouter else "",
                    )
                return None
    if selection is not None:
        with contextlib.suppress(TypeError):
            _CFG_DECLINES[pipe] = reason
        if logger is not None:
            logger.info("diffusion.multigpu: CFG-parallel not armed (%s)", reason)
    return reason


def release_sequence_split_replica(
    pipe: Any,
    reason: str,
    logger: Any = None,
) -> bool:
    """A load whose default request is unguided got its replica for token splits only. Once the denoiser is compiled
    (Auto's deferred profile), no call splits, so free the second card's copy and stop splitting; an even batch still
    builds one again when it needs it, as without the split. True when a replica was released."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    request = default_request(pipe)
    if router is None or router.replica is None or request is None or request.guided is not False:
        return False
    router.sequence_split_off = reason
    router.stats["sequence_split_skip"] = reason
    router.stats["sequence_split_released"] = reason
    router.drop_replica()
    if router.secondary.type == "cuda":
        with contextlib.suppress(Exception):
            from .diffusion_cuda_graph import outside_capture
            with outside_capture(), _device_context(router.secondary):
                _torch().cuda.empty_cache()
    if logger is not None:
        logger.info(
            "diffusion.multigpu: token-split replica on %s released (%s); one card from here",
            router.secondary,
            reason,
        )
    return True


def invalidate_cfg_replica(
    pipe: Any,
    reason: str,
    logger: Any = None,
) -> bool:
    """Drop the CFG-parallel replica of ``pipe`` after its denoiser was mutated in ways the per-step staleness check
    does not see (attention backend, eager patches, compile); the next guided step rebuilds it from the source. True
    when a replica was dropped."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    if router is None or router.replica is None:
        return False
    if logger is not None:
        logger.info(
            "diffusion.multigpu: CFG-parallel replica dropped (%s); rebuilt on the next guided step",
            reason,
        )
    router.stats["replica_rebuilds"] += 1
    router.drop_replica()
    return True


_ADAPTER_SCALE_ATTRS = ("scaling", "_active_adapter", "_disable_adapters")


def mirror_cfg_replica_adapter_scales(pipe: Any, logger: Any = None) -> bool:
    """After a LoRA weight-only change (``set_adapters`` on the same adapter set), copy each adapter layer's scales
    and active-adapter state onto the CFG-parallel replica instead of dropping it. True when mirrored (the replica now
    computes what the source does); False when there was nothing to mirror or the layers do not line up, in which case
    the caller drops the replica as before."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    if router is None or router.replica is None:
        return False
    module = router.module_ref()
    if module is None:
        return False
    src = [
        (n, m) for n, m in module.named_modules() if isinstance(getattr(m, "scaling", None), dict)
    ]
    dst = dict(
        (n, m)
        for n, m in router.replica.named_modules()
        if isinstance(getattr(m, "scaling", None), dict)
    )
    if not src or len(src) != len(dst):
        return False
    for name, layer in src:
        copy_layer = dst.get(name)
        if copy_layer is None or set(copy_layer.scaling) != set(layer.scaling):
            return False  # a different adapter set: only a rebuild is right
    for name, layer in src:
        copy_layer = dst[name]
        for attr in _ADAPTER_SCALE_ATTRS:
            if attr in layer.__dict__:
                value = layer.__dict__[attr]
                copy_layer.__dict__[attr] = (
                    dict(value) if isinstance(value, dict) else copy.copy(value)
                )
    try:
        # the fingerprint covers adapter scales: record the new ones as the replica's, else the next step rebuilds it
        router.fingerprint = weights_fingerprint(module)
    except Exception:  # noqa: BLE001 - unverifiable: drop the replica as before
        return False
    # Python-valued scales are guards of a compiled region: both sides may recompile, so they warm one at a time again
    router.warm_primary.clear()
    router.warm_replica.clear()
    router.stats["adapter_scale_mirrors"] = router.stats.get("adapter_scale_mirrors", 0) + 1
    if logger is not None:
        logger.info(
            "diffusion.multigpu: CFG-parallel replica kept; LoRA scales mirrored onto it (%d layers)",
            len(src),
        )
    return True


def _router_slot(module: Any, router: Any) -> Any:
    """``module`` when ``router`` is its forward, the outer layer (static step skip) whose inner it is, else None."""
    slot = module.__dict__.get("forward")
    if slot is router:
        return module
    if (
        getattr(slot, "_unsloth_outer_forward", False) is True
        and getattr(slot, "inner", None) is router
    ):
        return slot
    return None


def drop_displaced_cfg_router(
    pipe: Any,
    reason: str,
    logger: Any = None,
) -> bool:
    """Uninstall ``pipe``'s CFG-parallel router when a later layer (the deferred speed profile's graphs) replaced the
    denoiser's forward and bypasses it, so status reports CFG-parallel as declined rather than armed. True if dropped."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    if router is None:
        return False
    module = router.module_ref()
    holder = _router_slot(module, router) if module is not None else None
    if holder is module and module is not None:
        return False
    if holder is not None:
        # The replica cannot carry a step skip (it holds the source module), so the router could never build one.
        reason = "a static step skip wraps the denoiser"
    uninstall_cfg_parallel(pipe)
    with contextlib.suppress(TypeError):
        _CFG_DECLINES[pipe] = reason
    if logger is not None:
        logger.info("diffusion.multigpu: CFG-parallel dropped (%s)", reason)
    return True


def end_cfg_render(pipe: Any) -> None:
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return
    if router is not None:
        router.end_render()
        if getattr(router, "fallback_pending", False):
            # The replica failed at render time: hand the denoiser to the fallback (CFG batching, #16261) between
            # renders, never inside one. The render that failed finished on the source.
            fallback, router.fallback, router.fallback_pending = router.fallback, None, False
            reason = router.disabled_reason
            uninstall_cfg_parallel(pipe)
            # else status()["multi_gpu"] still reads "cfg_replica" with nothing saying why one card runs
            with contextlib.suppress(TypeError):
                _CFG_DECLINES[pipe] = f"disabled at run time: {reason}"
            try:
                result = fallback(pipe)
            except Exception as exc:  # noqa: BLE001 - an optional lever: one card stands
                result = f"{type(exc).__name__}: {exc}"
            if router.logger is not None:
                router.logger.info(
                    "diffusion.multigpu: CFG-parallel off after a render-time replica failure (%s); CFG batching %s",
                    reason,
                    "armed" if result is None else f"not armed ({result})",
                )


def set_cfg_parallel_fallback(pipe: Any, fallback: Callable[[Any], Optional[str]]) -> bool:
    """Register the lever that replaces CFG-parallel when its replica fails at render time (CFG batching): called once
    with the pipeline, after ``uninstall_cfg_parallel``, between renders; returns its decline reason or None when armed.
    It must not hold the pipeline itself (the router is the value of a weak-keyed map on it, so that would keep a
    rolled-back load alive). True when a router was armed to carry it."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    if router is None:
        return False
    router.fallback = fallback
    router.fallback_pending = False
    return True


def release_cfg_prefix_kv(pipe: Any) -> None:
    """Drop the replica's prefix-KV copies (and their hold on the primary's tensors) after the last denoise step."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
    except TypeError:
        return
    if router is not None:
        router.kv_mirrors.clear()
        router.kv_declined.clear()


def uninstall_cfg_parallel(pipe: Any) -> None:
    try:
        router = _CFG_ROUTERS.pop(pipe, None) if pipe is not None else None
    except TypeError:
        return
    if router is None:
        return
    if router.disabled_reason is None:
        router.disabled_reason = "uninstalled"
    router.drop_replica()
    module = router.module_ref()
    if module is not None:
        holder = _router_slot(module, router)
        if holder is module:
            if router.inner is not None:
                module.__dict__["forward"] = router.inner
                router.inner = None
            elif router.had_override:
                module.__dict__["forward"] = router.orig_forward
            else:
                del module.__dict__["forward"]
        elif holder is not None:
            # under a static step skip: hand the skip what the router wrapped
            holder.inner = (
                router.inner
                if router.inner is not None
                else (router.orig_forward if router.had_override else None)
            )
            router.inner = None
        if module.__dict__.get("_unsloth_cfg_router") is router:
            del module.__dict__["_unsloth_cfg_router"]
