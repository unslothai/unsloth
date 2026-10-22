# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several selected CUDA / ROCm cards for one image or video pipeline, without NCCL.

Three levers, all at a component or denoiser-call boundary, never inside a model forward:

* Component split (planner): the DiT stays whole and resident on the primary card while the text encoders, VAE and
  other companions sit on a secondary card, when that removes offload on the primary. Inputs move to a module's own
  card at its entry points and outputs come back to the primary, so the pipeline still sees one device. Copies are
  exact, so the output is bit-identical to a single-card resident run.
* Layer split (``diffusion_layer_split``): a DiT that fits no single selected card is split by contiguous ranges of
  its repeated blocks, the rest of it on the primary. Activations cross at the range boundaries only; outputs are
  bit-identical to a one-card run.
* Second denoiser (after load): a two-DiT pipeline (an MoE ``transformer_2``) keeps one expert per card when both
  experts do not fit one card.
* CFG-parallel (generate): when guidance runs separate cond / uncond denoiser calls on the same latents, every odd
  branch of a step runs on a replica of the denoiser on the secondary card, launched from a worker thread bound to
  that card, while the primary computes the even branch. One output is copied per step; guidance and the scheduler
  step stay on the primary. Branches are recognised from the call boundary alone (same latent tensor object in
  consecutive calls), so true CFG, rescale and multi-branch guidance work unchanged and a guidance-distilled model
  (one call per step) never builds a replica.
* Layer split (after load, diffusion_layer_split_plan): when one card would offload the denoiser itself and no
  component assignment fits, the denoiser's blocks are cut across the cards. Excludes CFG-parallel.

Kill switches: ``UNSLOTH_DIFFUSION_MULTIGPU=0`` (everything here; one card as before), ``UNSLOTH_DIFFUSION_CFG_PARALLEL=0``
(only the replica). ``UNSLOTH_DIFFUSION_MULTIGPU=force`` splits the companions even when one card would hold
everything (for proving bit-identity). With one selected card nothing here engages.
"""

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
from typing import Any, Callable, Iterator, Optional, Sequence

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


_SELECTION: contextvars.ContextVar[Optional[MultiGpuSelection]] = contextvars.ContextVar(
    "unsloth_diffusion_multigpu_selection", default = None
)


AUTOMATIC_ORDINALS = "automatic"


def make_selection(
    primary: Optional[int],
    ordinals: Optional[Sequence[int]],
    *,
    automatic: bool = False,
) -> Optional[MultiGpuSelection]:
    """A selection when at least two distinct cards were picked and the lever is on, else None (one card)."""
    if primary is None or multigpu_mode() == "off":
        return None
    others = tuple(dict.fromkeys(int(o) for o in (ordinals or ()) if int(o) != int(primary)))
    try:
        import torch

        # nvidia-smi can list a card this process's torch cannot address (a mask applied after the listing).
        if torch.cuda.is_available():
            visible = int(torch.cuda.device_count())
            others = tuple(o for o in others if 0 <= o < visible)
    except Exception:  # noqa: BLE001 - unknown count: the capability filter below still applies
        pass
    if not others:
        return None
    if _rocm_torch():
        # ROCm's capability is the gfx target (gfx942 (9, 4)), which does not order; match the primary's load dtype
        # instead (arch-pinned attention kernels are gated per card in diffusion_layer_split_plan).
        others = tuple(o for o in others if _rocm_dtype_parity(int(primary), o))
        return (
            MultiGpuSelection(primary = int(primary), secondaries = others, automatic = automatic)
            if others
            else None
        )
    # The load dtype and the device-specific patches are resolved for the primary (bf16 only from Ampere, FP8 kernels
    # from Ada): a card of a lower compute capability may not run them, so it takes no component.
    need = _capability(int(primary))
    if need is not None:
        others = tuple(o for o in others if (_capability(o) or need) >= need)
    if not others:
        return None
    return MultiGpuSelection(primary = int(primary), secondaries = others, automatic = automatic)


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
    if multigpu_mode() == "off":
        return None
    try:
        from .diffusion_device import auto_cuda_ranking_possible, resolve_diffusion_device_target

        if not auto_cuda_ranking_possible() or resolve_diffusion_device_target().device != "cuda":
            return None
        import torch

        if primary is None:
            primary = int(torch.cuda.current_device())
        count = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 -- an unreadable host keeps one card
        return None
    return make_selection(primary, range(count), automatic = True)


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
    # Only a CFG replica reads the recorded compile arguments; automatic placement never builds one.
    if not selection.automatic or cfg_parallel_mode() == "force":
        install_compile_recorder()
    token = _SELECTION.set(selection)
    try:
        yield selection
    finally:
        _SELECTION.reset(token)


def with_load_selection(run_load: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a backend's ``_run_load(self, **kwargs)``: pop ``_gpu_ordinals`` (every selected card, from
    begin_load, or :data:`AUTOMATIC_ORDINALS`) and run the load inside :func:`selection_scope` with ``gpu_ordinal``
    as the primary. Without a multi-card selection the load runs exactly as before."""
    import functools

    @functools.wraps(run_load)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ordinals = kwargs.pop("_gpu_ordinals", None)
        if ordinals and models_kept_loaded():
            selection = None
        elif ordinals == AUTOMATIC_ORDINALS:
            selection = automatic_selection(kwargs.get("gpu_ordinal"))
        else:
            selection = make_selection(kwargs.get("gpu_ordinal"), ordinals)
        with selection_scope(selection):
            return run_load(self, *args, **kwargs)

    return wrapper


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
    out: dict[int, DeviceProbe] = {}
    for o in ords:
        free, total = _free_total_mib(o)
        out[o] = DeviceProbe(ordinal = o, name = _NAMES.get(o, ""), total_mib = total, free_mib = free)
    return out


def best_secondary(
    selection: MultiGpuSelection, probes: Optional[dict[int, DeviceProbe]] = None
) -> Optional[int]:
    if not selection.secondaries:
        return None
    if probes is None and selection.automatic:
        free = {o: _free_total_mib(o)[0] for o in selection.secondaries}
        return max(
            selection.secondaries, key = lambda o: (free[o] if free[o] is not None else -1, -o)
        )
    if probes is None:
        probes = probe_devices((selection.primary, *selection.secondaries))

    def key(o: int) -> tuple[int, int]:
        p = probes.get(o)
        return (p.free_mib if p is not None and p.free_mib is not None else -1, -o)

    return max(selection.secondaries, key = key)


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
        best = max(fits, key = lambda o: (limits[o], -o))
        return CompanionSplit(
            secondary_ordinal = best,
            primary_required_mib = primary_required,
            primary_limit_mib = int(primary_limit_mib),
            secondary_required_mib = secondary_required,
            secondary_limit_mib = limits[best],
            components = components,
            reason = (
                f"multi-GPU: denoiser resident on cuda:{selection.primary} ({primary_required} of "
                f"{int(primary_limit_mib)} MiB), {label} on cuda:{best} ({secondary_required} of {limits[best]} MiB); "
                f"single card would use '{single_card_policy}'"
            ),
        )
    return None


def choose_component_assignment(
    sizes: dict[str, tuple[int, str]],
    *,
    headroom_mib: int,
    overhead_mib: int,
    primary_limit_mib: int,
    secondary_limit_mib: int,
    move_denoisers: bool = True,
) -> Optional[tuple[str, ...]]:
    """Names to place on the secondary so each card holds its share resident, or None. ``sizes`` maps component name
    -> (MiB, role) with role 'dit' / 'text_encoder' / 'other'. A card running a denoiser or the VAE (any 'other'
    component) is priced with the runtime headroom; a card holding only text encoders with an encoder forward's room.
    Tried in order: every companion; the text encoders alone; the second denoiser; the second denoiser plus the
    companions. The first denoiser always stays on the primary; ``move_denoisers=False`` (the secondary cannot run the
    denoiser's attention backend) keeps every denoiser there."""
    dits = [n for n, (_m, r) in sizes.items() if r == "dit"]
    if not dits:
        return None
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

    for moved in candidates:
        on_primary = [n for n in sizes if n not in moved]
        if need(on_primary) <= primary_limit_mib and need(list(moved)) <= secondary_limit_mib:
            return moved
    return None


def _secondary_runs_denoiser(
    pipe: Any, selection: MultiGpuSelection, secondary: int, sizes: dict, logger: Any
) -> bool:
    """Whether a second denoiser may move wholesale to ``secondary``. ROCm admits mixed gfx targets (make_selection),
    so the card must pass the layer split's per-card attention gate: an arch-pinned kernel (flash_attn, AITER) only on
    the primary's gfx target, native SDPA not math-only beside a fused primary. CUDA's capability order already holds."""
    if not _rocm_torch() or sum(1 for _m, role in sizes.values() if role == "dit") < 2:
        return True
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


def refine_plan_for_multigpu(
    pipe: Any,
    plan: Any,
    logger: Any = None,
) -> Any:
    """Post-load: an offloading plan becomes a resident split when the LOADED components can be shared by the cards.

    No-op without a selection, for explicit offload modes, a plan already resident (unless forced) or split, unified memory
    or an unknown budget. When no component assignment fits, a layer split (diffusion_layer_split_plan) is tried before
    keeping the offload plan. Returns the plan (a ``dataclasses.replace`` copy when it changed)."""
    selection = current_selection()
    if selection is None or plan is None:
        return plan
    if (
        getattr(plan, "secondary_ordinal", None) is not None
        or getattr(plan, "layer_split", None) is not None
    ):
        return plan
    from .diffusion_layer_split_plan import layer_split_mode, refine_plan_for_layer_split

    layer_force = layer_split_mode() == "force"
    if plan.offload_policy == "none" and multigpu_mode() != "force" and not layer_force:
        return plan
    if getattr(plan, "requested_mode", None) not in ("auto", "fast"):
        return plan
    if plan.estimates.get("explicit_cpu_offload"):
        return plan
    memory = plan.device_memory
    if memory.is_unified or getattr(memory, "device", None) != "cuda":
        if memory.is_unified:
            from .diffusion_layer_split_plan import note_unified_primary
            note_unified_primary(pipe, selection, logger)
        return plan
    budget = plan.estimates.get("safe_device_budget_mib")
    if budget is None:
        return plan
    from .diffusion_memory import _loaded_component_mib

    sizes = _loaded_component_mib(pipe)
    if not sizes:
        return plan
    secondary = best_secondary(selection)
    if secondary is None:
        return plan
    mode = getattr(plan, "requested_mode", None)
    sec_limit = secondary_resident_limit_mib(secondary, mode)
    if sec_limit is None:
        return plan
    from .diffusion_memory import _fast_device_budget_mib

    primary_limit = (
        _fast_device_budget_mib(memory) if mode == "fast" else int(int(budget) * _RESIDENT_FRACTION)
    )
    if primary_limit is None:
        return plan
    headroom = int(
        plan.estimates.get("measured_runtime_headroom_mib")
        or plan.estimates.get("calibrated_headroom_mib")
        or plan.estimates.get("runtime_headroom_mib")
        or 0
    )
    overhead = int(plan.estimates.get("base_overhead_mib") or 0)

    def layer_split_or_plan() -> Any:
        return refine_plan_for_layer_split(
            pipe,
            plan,
            selection = selection,
            sizes = sizes,
            primary_limit_mib = int(primary_limit),
            secondary_limit_mib = lambda o: secondary_resident_limit_mib(o, mode),
            headroom_mib = headroom,
            overhead_mib = overhead,
            logger = logger,
        )

    if layer_force:
        return layer_split_or_plan()
    moved = choose_component_assignment(
        sizes,
        headroom_mib = headroom,
        overhead_mib = overhead,
        primary_limit_mib = int(primary_limit),
        secondary_limit_mib = int(sec_limit),
        move_denoisers = _secondary_runs_denoiser(pipe, selection, secondary, sizes, logger),
    )
    if moved is None:
        return layer_split_or_plan()
    reason = (
        f"multi-GPU (loaded sizes): {', '.join(moved)} on cuda:{secondary}, the rest resident on "
        f"cuda:{selection.primary}; single card would use '{plan.offload_policy}'"
    )
    if logger is not None:
        logger.info("diffusion.multigpu: %s", reason)
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


def install_execution_device_patch() -> bool:
    """``DiffusionPipeline._execution_device`` / ``.device`` report the primary card of a split pipeline. diffusers
    returns the first non-CPU component in signature order, which is the VAE (on the secondary) for most families.
    Inert for any pipeline without a split denoiser; also covers ``from_pipe`` siblings, which share the modules."""
    try:
        from diffusers import DiffusionPipeline
    except Exception:  # noqa: BLE001
        return False
    with _EXEC_PATCH_LOCK:
        if getattr(DiffusionPipeline, "_unsloth_split_exec_patch", False):
            return True
        orig_exec = DiffusionPipeline.__dict__.get("_execution_device")
        orig_dev = DiffusionPipeline.__dict__.get("device")
        if not isinstance(orig_exec, property) or not isinstance(orig_dev, property):
            return False

        def _exec(self: Any) -> Any:
            home = _split_home(self)
            return home if home is not None else orig_exec.fget(self)

        def _dev(self: Any) -> Any:
            home = _split_home(self)
            return home if home is not None else orig_dev.fget(self)

        DiffusionPipeline._execution_device = property(_exec, doc = orig_exec.__doc__)
        DiffusionPipeline.device = property(_dev, doc = orig_dev.__doc__)
        DiffusionPipeline._unsloth_split_exec_patch = True
        return True


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
    if info is None and cfg is None and declined is None and layer is None:
        return None
    out: dict[str, Any] = dict(info or {})
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


# Offload machinery moves weights between host and card without changing the math. A replica holds its weights
# resident, so it drops these instead of inheriting them.
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


# Where a hook shared by a replica may come from. Studio's own hooks keep their per-call state on the module (mirrored
# by _mirror_hook_state); a foreign hook may keep it in a module global that both threads would then race on, and that
# cannot be seen from the function, so it declines the replica.
_TRUSTED_HOOK_MODULES = ("core.inference.", "studio.backend.core.inference.")


def _hook_origin(fn: Any) -> str:
    target = getattr(fn, "func", None) if hasattr(fn, "args") and hasattr(fn, "func") else fn
    target = getattr(target, "__func__", target)
    if isinstance(target, (types.FunctionType, types.BuiltinFunctionType)):
        return str(getattr(target, "__module__", None) or "")
    return str(getattr(type(target), "__module__", None) or "")


def _trusted_hook(fn: Any) -> bool:
    return _hook_origin(fn).startswith(_TRUSTED_HOOK_MODULES) and _portable(fn)


def _hooks_present(module: Any) -> Optional[str]:
    """Name of the first submodule carrying state a replica cannot inherit faithfully, else None: a non-offload hook
    registry (step caches), an accelerate hook, or a forward hook / instance forward override below the root whose
    captured state points at the source (deepcopy treats functions as atoms). Offload machinery (diffusers group
    offloading, Studio's prefetchers) is dropped by :func:`build_replica`, which holds every weight resident. The
    ROOT's torch forward hooks run on the source around the router for every branch and the copy does not carry them,
    but whatever per-call state they write must reach the replica: only Studio's own root hooks are accepted, whose
    state is the ``_unsloth_*`` scalars :func:`_mirror_hook_state` copies."""
    for name, sub in module.named_modules():
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
    source card), with every recorded ``Module.compile`` re-applied to the copy. Raises when it cannot be built."""
    torch = _torch()
    device = torch.device(device)
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
    for sub in module.modules():
        offload_keys = set(_offload_attr_keys(sub))
        for key, value in sub.__dict__.items():
            if isinstance(value, CfgParallelRouter) or key in offload_keys:
                # never traversed: offload state holds pinned host copies a deepcopy would duplicate
                memo[id(value)] = _DROP
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            for hook in (getattr(sub, attr, None) or {}).values():
                if _is_offload_object(hook):
                    memo[id(hook)] = _DROP
        # Per-step precision state (the NVFP4 W4A16 protect schedule) is advanced by the source's scheduler: the copy
        # must read the same controller, or the two CFG branches of one step run different GEMMs.
        for value in sub.__dict__.values():
            if type(value).__name__ in _SHARED_STATE_TYPES:
                memo[id(value)] = value
    guard = module.__dict__.get("_unsloth_compile_guard")
    if guard is not None:
        memo[id(guard)] = _DROP
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


_SHARED_STATE_TYPES = frozenset({"NVFP4StepController"})


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


def weights_fingerprint(module: Any) -> int:
    """Changes whenever the denoiser's weights or adapter state do: a parameter or buffer (NVFP4 / quantised layers
    keep their packed weights and scales as buffers) replaced, edited in place (fused LoRA bumps the version counter),
    added or removed (PEFT injection), or an adapter's scale / activation changed. A ``.data`` swap to different values
    is not seen; diffusers' LoRA paths do not do that."""
    parts: list = []
    for name, param in (*module.named_parameters(), *module.named_buffers()):
        try:
            version = param._version
        except Exception:  # noqa: BLE001 - inference tensors carry no version
            version = -1
        # Identity, not the data pointer: an encode-time offload round trip moves the same weights and must not force
        # a rebuild, while a replaced parameter (new object) or an in-place edit (version) must.
        parts.append((name, id(param), version))
    for name, sub in module.named_modules():
        for attr in _ADAPTER_ATTRS:
            value = sub.__dict__.get(attr)
            if value is not None and not callable(value):
                parts.append((name, attr, repr(value)))
    return hash(tuple(parts))


_LATENT_KEYS = ("hidden_states", "sample", "x", "latents", "img", "latent")


def _latent_arg(args: tuple, kwargs: dict) -> Any:
    torch = _torch()
    for key in _LATENT_KEYS:
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            return value
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
        if not isinstance(value, (*_PLAIN_TYPES, list, tuple, dict)) and depth < 4:
            if _holds_opaque_tensors(value, depth + 1):
                return True
    return False


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


class _ReplicaWorker:
    """One daemon thread bound to the replica's card: set_device, grad / inference / autocast mode and the caller's
    context variables (inductor overrides live in one on torch >= 2.12) are applied per job."""

    def __init__(self, device: Any):
        self.device = device
        self._jobs: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(
            target = self._loop, name = f"unsloth-cfg-replica-{device}", daemon = True
        )
        self._thread.start()

    def _loop(self) -> None:
        torch = _torch()
        if getattr(self.device, "type", None) == "cuda":
            try:
                torch.cuda.set_device(self.device)
            except Exception:  # noqa: BLE001
                pass
        while True:
            job = self._jobs.get()
            if job is None:
                return
            fn, ctx, modes, done = job
            try:
                done["result"] = ctx.run(self._run, fn, modes)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the caller's thread
                done["error"] = exc
            finally:
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


# Per-call state Studio's root hooks write onto denoiser submodules (``_unsloth_*`` scalars, e.g. the HunyuanVideo-1.5
# trim hook's null-mask flag). The root hooks run only on the source, around the router, so the replica's copy of that
# state is mirrored from the source before every replica call and the prediction is verified against it.
_STATE_PREFIX = "_unsloth_"
_STATE_TYPES = (bool, int, float, str, type(None))
_STATE_KEEP = frozenset({"_unsloth_replica_compiled"})


def _hook_state(pairs: list) -> tuple:
    out = []
    for i, (src, _dst) in enumerate(pairs):
        for key, value in src.__dict__.items():
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
    unpredicted."""

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
        }
        self.template: Optional[tuple[Any, Any, dict]] = None
        self.pending: Optional[dict] = None
        self.b0_leaves: dict = {}
        self.fingerprint: Optional[int] = None
        self.pairs: list = []

    def describe(self) -> dict[str, Any]:
        return {
            "secondary": str(self.secondary),
            "replica_built": self.replica is not None,
            "replica_compiled_submodules": getattr(self.replica, "_unsloth_replica_compiled", None),
            "disabled_reason": self.disabled_reason,
            **self.stats,
        }

    def disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.multigpu: CFG-parallel disabled (%s); one card from here", reason
                )
        self.drop_replica()

    def drop_replica(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._drop_pending()
        self.replica = None
        self.pairs = []
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}

    def end_render(self) -> None:
        self._drop_pending()
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}
        self.branch = 0
        # the next render shows its own guidance shape: no speculative branch before its first odd call
        self.cfg_seen = False
        self.singles = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.disabled_reason is not None or _background_compile_call():
            # Disabled: no branch state to keep. A render that must run eager while its compile is in flight, or the
            # background compile thread itself replaying recorded inputs: both branches stay on the source, which
            # keeps that render bit-identical to an eager one and leaves the router's per-render state untouched.
            self.stats["primary_calls"] += 1
            return self.orig_forward(*args, **kwargs)
        latent = _latent_arg(args, kwargs)
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
                # Copy this step's inputs to the replica card BEFORE the primary branch is queued: a copy queued after
                # it would wait for it on the source stream and serialise the two branches.
                self._prefetch(args, kwargs)
                self._launch_prediction(latent)
        if (
            self.branch % 2 == 1
            and self.disabled_reason is None
            and _holds_opaque_tensors((args, kwargs))
        ):
            self._drop_pending()
            self.template = None
            self.stats["opaque_primary_calls"] += 1
        elif self.branch % 2 == 1 and self.disabled_reason is None:
            self.singles = 0
            if self.replica is None:
                self.cfg_seen = True
                self._build_replica()
                if self.branch == 1:
                    self._record_template(args, kwargs, latent)
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
                except Exception as exc:  # noqa: BLE001 - this branch is recomputed on the primary below
                    self.disable(f"replica call failed: {type(exc).__name__}: {exc}")
            self.cfg_seen = True
        self.stats["primary_calls"] += 1
        return self.orig_forward(*args, **kwargs)

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

    def _launch_prediction(self, latent: Any) -> None:
        predicted = self._predict(latent)
        if predicted is None or self.worker is None:
            return
        p_args, p_kwargs = predicted
        try:
            r_args = self._to_replica(p_args)
            r_kwargs = self._to_replica(p_kwargs)
        except Exception:  # noqa: BLE001 - no prediction this step; the branch runs unpredicted
            return
        torch = _torch()
        # What the replica consumes, kept per tensor: at the real call the sources are compared bitwise against it, so
        # an in-place edit between launch and the odd branch (same object, new values) rejects the prediction.
        # Version counters cannot do this: inference tensors do not track one.
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
            "job": self.worker.submit(lambda: replica(*r_args, **r_kwargs)),
        }

    def _take_prediction(self, args: tuple, kwargs: dict) -> Any:
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        actual = _leaves((args, kwargs))
        predicted = pending["leaves"]
        by_value = pending["by_value"]
        ok = len(actual) == len(predicted) and all(
            pa == pp and (_bitwise_equal(va, vp) if pa in by_value else _same_value(va, vp))
            for (pa, va), (pp, vp) in zip(actual, predicted)
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
        return move_to_device(out, self.primary)

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

    def _build_replica(self) -> None:
        module = self.module_ref()
        if module is None:
            return
        t0 = time.perf_counter()
        try:
            hooked = _hooks_present(module)
            if hooked is not None:
                raise RuntimeError(f"denoiser carries hooks ({hooked})")
            need_mib = sum(
                t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers())
            ) // (1 << 20)
            free, _total = _free_total_mib(self.secondary.index)
            if free is not None:
                with contextlib.suppress(Exception):
                    torch = _torch()
                    free += (
                        torch.cuda.memory_reserved(self.secondary)
                        - torch.cuda.memory_allocated(self.secondary)
                    ) // (1 << 20)
            if free is not None and need_mib + 1024 > free:
                raise RuntimeError(
                    f"replica needs {need_mib} MiB, {free} MiB free on {self.secondary}"
                )
            self.fingerprint = weights_fingerprint(module)
            with _device_context(self.secondary):
                self.replica = self.replica_factory()
            self.pairs = list(zip(module.modules(), self.replica.modules()))
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
            with contextlib.suppress(Exception):
                _torch().cuda.empty_cache()
            self.disable(f"replica unavailable: {exc}")

    def _prefetch(self, args: tuple, kwargs: dict) -> None:
        for tensor in (*_iter_tensors((args, kwargs)), *self.carry):
            if tensor.device == self.secondary or id(tensor) in self.prefetched:
                continue
            self.prefetched[id(tensor)] = (tensor, tensor.to(self.secondary))

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
            if depth < 4 and isinstance(value, tuple) and not hasattr(value, "_fields"):
                return tuple(walk(v, depth + 1) for v in value)
            if depth < 4 and isinstance(value, list):
                return [walk(v, depth + 1) for v in value]
            if depth < 4 and isinstance(value, dict):
                return type(value)((k, walk(v, depth + 1)) for k, v in value.items())
            return value

        return walk(obj)

    def _run_replica(self, args: tuple, kwargs: dict) -> Any:
        # Unpredicted: nothing verified the copies made before the primary branch ran, and an input edited in place
        # since then would reach the replica stale. Copy this call's inputs now (ordered after the primary branch).
        self.prefetched.clear()
        _mirror_hook_state(self.pairs)
        r_args = self._to_replica(args)
        r_kwargs = self._to_replica(kwargs)
        replica = self.replica
        out = self.worker.call(lambda: replica(*r_args, **r_kwargs))
        return move_to_device(out, self.primary)


_CFG_ROUTERS: "weakref.WeakKeyDictionary[Any, CfgParallelRouter]" = weakref.WeakKeyDictionary()
_CFG_DECLINES: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


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
    # A replica takes a whole denoiser's worth of another card; automatic placement only borrows one to avoid offload.
    if selection.automatic and mode != "force":
        return f"automatic GPU placement (select several cards or set {CFG_PARALLEL_ENV}=force)"
    from .diffusion_bg_compile import load_time_enabled as bg_compile_enabled

    if bg_compile_enabled():
        # its replays call the denoiser, which would enter the router and race a render's branch state
        return "background compile replays through the denoiser (UNSLOTH_DIFFUSION_BG_COMPILE)"
    denoiser = getattr(pipe, "transformer", None)
    torch = _torch()
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser (a U-Net batches its guidance branches)"
    from .diffusion_layer_split import is_layer_split

    if is_layer_split(denoiser):
        return "denoiser is split across cards (layer split)"
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
    if _module_device(denoiser) is None or _module_device(denoiser).type != "cuda":
        return "denoiser is not on a CUDA card"
    return None


def install_cfg_parallel(
    pipe: Any,
    *,
    offload_policy: str,
    logger: Any = None,
) -> Optional[str]:
    """Arm CFG-parallel on ``pipe.transformer`` when a secondary card can hold a replica. The replica is built on the
    first step that shows two branches, so a guidance-distilled model never pays for it. Returns the decline reason or
    None when armed."""
    selection = current_selection()
    reason = cfg_parallel_decline(pipe, offload_policy = offload_policy, selection = selection)
    if reason is None:
        denoiser = pipe.transformer
        primary = _module_device(denoiser)
        probes = probe_devices((selection.primary, *selection.secondaries))
        same = [
            o
            for o in selection.secondaries
            if probes.get(o) is not None
            and probes.get(primary.index) is not None
            and probes[o].name == probes[primary.index].name
        ]
        candidates = same if cfg_parallel_mode() != "force" else list(selection.secondaries)
        try:
            placed = ((_PLACEMENTS.get(pipe) or {}).get("placement") or {}).values()
        except TypeError:
            placed = ()
        occupied = {str(d) for d in placed} - {str(primary)}
        if occupied:
            candidates = [o for o in candidates if f"cuda:{o}" not in occupied]
        if not candidates and occupied:
            reason = f"the only replica card ({', '.join(sorted(occupied))}) holds the split's components"
        elif not candidates:
            reason = "secondary card is a different model (bit-identity needs the same kernels)"
        else:
            need_mib = sum(
                t.numel() * t.element_size() for t in (*denoiser.parameters(), *denoiser.buffers())
            ) // (1 << 20)
            best = max(candidates, key = lambda o: probes[o].free_mib or -1)
            free = probes[best].free_mib
            if free is not None and need_mib + 1024 > free:
                reason = f"replica needs {need_mib} MiB, {free} MiB free on cuda:{best}"
            else:
                router = CfgParallelRouter(
                    denoiser, primary = primary, secondary = f"cuda:{best}", logger = logger
                )
                try:
                    _CFG_ROUTERS[pipe] = router
                except TypeError:
                    return "pipeline cannot be tracked (not weak-referenceable)"
                denoiser.forward = router
                denoiser._unsloth_cfg_router = router
                if logger is not None:
                    logger.info(
                        "diffusion.multigpu: CFG-parallel armed (replica on cuda:%s when guidance runs two branches)",
                        best,
                    )
                return None
    if selection is not None:
        with contextlib.suppress(TypeError):
            _CFG_DECLINES[pipe] = reason
        if logger is not None:
            logger.info("diffusion.multigpu: CFG-parallel not armed (%s)", reason)
    return reason


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
    if module is not None and module.__dict__.get("forward") is router:
        return False
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
        if module.__dict__.get("forward") is router:
            if router.had_override:
                module.__dict__["forward"] = router.orig_forward
            else:
                del module.__dict__["forward"]
        if module.__dict__.get("_unsloth_cfg_router") is router:
            del module.__dict__["_unsloth_cfg_router"]
