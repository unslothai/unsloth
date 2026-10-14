# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several selected CUDA / ROCm cards for one image or video pipeline, without NCCL.

Three levers, all at a component or denoiser-call boundary, never inside a model forward:

* Component split (planner): the DiT stays whole and resident on the primary card while the text encoders, VAE and
  other companions sit on a secondary card, when that removes offload on the primary. Inputs move to a module's own
  card at its entry points and outputs come back to the primary, so the pipeline still sees one device. Copies are
  exact, so the output is bit-identical to a single-card resident run.
* Second denoiser (after load): a two-DiT pipeline (an MoE ``transformer_2``) keeps one expert per card when both
  experts do not fit one card.
* CFG-parallel (generate): when guidance runs separate cond / uncond denoiser calls on the same latents, every odd
  branch of a step runs on a replica of the denoiser on the secondary card, launched from a worker thread bound to
  that card, while the primary computes the even branch. One output is copied per step; guidance and the scheduler
  step stay on the primary. Branches are recognised from the call boundary alone (same latent tensor object in
  consecutive calls), so true CFG, rescale and multi-branch guidance work unchanged and a guidance-distilled model
  (one call per step) never builds a replica.

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

# Denoiser attributes: they stay on the primary card under the companion split (the same contract as the speed and
# offload layers, diffusion_autoopt/targets.py).
DENOISER_ATTRS: tuple[str, ...] = ("transformer", "transformer_2", "unconditional_transformer", "unet")
# Run inside every step beside the denoiser, so moving them to another card buys nothing and costs a copy per step.
_KEEP_WITH_DENOISER_PREFIXES: tuple[str, ...] = ("controlnet",)
# Public entry points a pipeline calls on a companion. Internal helpers are not wrapped: they run on the module's own
# card once the entry point moved the inputs there.
_ENTRY_METHODS: tuple[str, ...] = (
    "forward",
    "encode",
    "decode",
    "get_text_features",
    "get_image_features",
)
# Sentinel for ``MemoryPlan.secondary_components``: only the text encoders move (the VAE decodes beside the denoiser).
TEXT_ENCODERS_ONLY: tuple[str, ...] = ("<text_encoders>",)
# Activation room for a card that only runs text encoders: one forward over a prompt, far below a denoise step or a
# VAE decode, which is what the runtime headroom prices.
TEXT_ENCODER_HEADROOM_MIB = 1024
# The resident-fit rule of the single-card planner (diffusion_memory.plan_diffusion_memory): weights plus headroom
# within 85% of the safe budget.
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


# --------------------------------------------------------------------------------------------------------------------
# Selection: the cards one load may use, carried from begin_load to the planner of the same load.
# --------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen = True)
class MultiGpuSelection:
    primary: int
    secondaries: tuple[int, ...]

    def as_public_dict(self) -> dict[str, Any]:
        return {"primary": self.primary, "secondaries": list(self.secondaries)}


_SELECTION: contextvars.ContextVar[Optional[MultiGpuSelection]] = contextvars.ContextVar(
    "unsloth_diffusion_multigpu_selection", default = None
)


def make_selection(primary: Optional[int], ordinals: Optional[Sequence[int]]) -> Optional[MultiGpuSelection]:
    """A selection when at least two distinct cards were picked and the lever is on, else None (one card)."""
    if primary is None or multigpu_mode() == "off":
        return None
    others = tuple(dict.fromkeys(int(o) for o in (ordinals or ()) if int(o) != int(primary)))
    if not others:
        return None
    return MultiGpuSelection(primary = int(primary), secondaries = others)


@contextlib.contextmanager
def selection_scope(selection: Optional[MultiGpuSelection]) -> Iterator[Optional[MultiGpuSelection]]:
    """Make ``selection`` visible to the planner and placement for the duration of one load (thread / task local)."""
    if selection is None:
        yield None
        return
    install_compile_recorder()
    token = _SELECTION.set(selection)
    try:
        yield selection
    finally:
        _SELECTION.reset(token)


def with_load_selection(run_load: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a backend's ``_run_load(self, **kwargs)``: pop ``_gpu_ordinals`` (every selected card, from
    begin_load) and run the load inside :func:`selection_scope` with ``gpu_ordinal`` as the primary. Without a
    multi-card selection the load runs exactly as before."""
    import functools

    @functools.wraps(run_load)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        selection = make_selection(kwargs.get("gpu_ordinal"), kwargs.pop("_gpu_ordinals", None))
        with selection_scope(selection):
            return run_load(self, *args, **kwargs)

    return wrapper


def current_selection() -> Optional[MultiGpuSelection]:
    if multigpu_mode() == "off":
        return None
    return _SELECTION.get()


# --------------------------------------------------------------------------------------------------------------------
# Device probe: once per process for the static facts (name, peer access, bandwidth); free memory read fresh.
# --------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen = True)
class DeviceProbe:
    ordinal: int
    name: str
    total_mib: Optional[int]
    free_mib: Optional[int]
    h2d_gib_s: Optional[float] = None
    # ordinal -> (can access peer, measured GiB/s from this card to that one)
    peers: tuple[tuple[int, bool, Optional[float]], ...] = ()

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "total_mib": self.total_mib,
            "free_mib": self.free_mib,
            "h2d_gib_s": self.h2d_gib_s,
            "peers": [
                {"ordinal": o, "peer_access": access, "copy_gib_s": gbps} for o, access, gbps in self.peers
            ],
        }


_PROBE_LOCK = threading.Lock()
_STATIC_PROBE: dict[int, dict[str, Any]] = {}
_PAIR_PROBE: dict[tuple[int, int], tuple[bool, Optional[float]]] = {}
_BANDWIDTH_MIB = 64


def _measure_copy_gib_s(src: Any, dst_device: Any) -> Optional[float]:
    import torch

    try:
        torch.cuda.synchronize(dst_device)
        if src.is_cuda:
            torch.cuda.synchronize(src.device)
        dst = torch.empty_like(src, device = dst_device)
        dst.copy_(src)  # warm the path (peer mapping, pinned staging)
        torch.cuda.synchronize(dst_device)
        if src.is_cuda:
            torch.cuda.synchronize(src.device)
        t0 = time.perf_counter()
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize(dst_device)
        if src.is_cuda:
            torch.cuda.synchronize(src.device)
        dt = time.perf_counter() - t0
        del dst
        return round(3 * src.numel() * src.element_size() / dt / (1 << 30), 2) if dt > 0 else None
    except Exception:  # noqa: BLE001 -- a probe never fails a load
        return None


def _free_total_mib(ordinal: int) -> tuple[Optional[int], Optional[int]]:
    try:
        import torch

        free, total = torch.cuda.mem_get_info(ordinal)
        return int(free) // (1 << 20), int(total) // (1 << 20)
    except Exception:  # noqa: BLE001
        return None, None


def probe_devices(ordinals: Sequence[int], *, measure: bool = True) -> dict[int, DeviceProbe]:
    """Name, memory, peer access and copy bandwidth for ``ordinals``. Static facts are cached per process; free
    memory is read on every call because it moves."""
    import torch

    out: dict[int, DeviceProbe] = {}
    ords = [int(o) for o in ordinals]
    with _PROBE_LOCK:
        for o in ords:
            static = _STATIC_PROBE.get(o)
            if static is None:
                static = {"name": "", "h2d": None}
                try:
                    static["name"] = str(torch.cuda.get_device_name(o))
                except Exception:  # noqa: BLE001
                    pass
                if measure:
                    try:
                        host = torch.empty(_BANDWIDTH_MIB << 20, dtype = torch.uint8).pin_memory()
                        static["h2d"] = _measure_copy_gib_s(host, torch.device("cuda", o))
                        del host
                    except Exception:  # noqa: BLE001
                        pass
                _STATIC_PROBE[o] = static
        for a in ords:
            for b in ords:
                if a == b or (a, b) in _PAIR_PROBE:
                    continue
                try:
                    access = bool(torch.cuda.can_device_access_peer(a, b))
                except Exception:  # noqa: BLE001
                    access = False
                gbps = None
                if measure:
                    try:
                        src = torch.empty(_BANDWIDTH_MIB << 20, dtype = torch.uint8, device = torch.device("cuda", a))
                        gbps = _measure_copy_gib_s(src, torch.device("cuda", b))
                        del src
                    except Exception:  # noqa: BLE001
                        pass
                _PAIR_PROBE[(a, b)] = (access, gbps)
    for o in ords:
        free, total = _free_total_mib(o)
        static = _STATIC_PROBE.get(o, {})
        out[o] = DeviceProbe(
            ordinal = o,
            name = static.get("name", ""),
            total_mib = total,
            free_mib = free,
            h2d_gib_s = static.get("h2d"),
            peers = tuple((b, *_PAIR_PROBE.get((o, b), (False, None))) for b in ords if b != o),
        )
    return out


def _reset_probe_cache_for_tests() -> None:
    with _PROBE_LOCK:
        _STATIC_PROBE.clear()
        _PAIR_PROBE.clear()


def best_secondary(selection: MultiGpuSelection, probes: Optional[dict[int, DeviceProbe]] = None) -> Optional[int]:
    """The secondary with the most free memory (ties to the lowest ordinal)."""
    if not selection.secondaries:
        return None
    if probes is None:
        probes = probe_devices((selection.primary, *selection.secondaries), measure = False)

    def key(o: int) -> tuple[int, int]:
        p = probes.get(o)
        return (p.free_mib if p is not None and p.free_mib is not None else -1, -o)

    return max(selection.secondaries, key = key)


# --------------------------------------------------------------------------------------------------------------------
# Planner (pre-load): companions on a secondary card when that keeps the denoiser resident on the primary.
# --------------------------------------------------------------------------------------------------------------------


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
        candidates.append((int(text_encoder_dense_mib), *encoders_split, TEXT_ENCODERS_ONLY, "text encoders"))
    for moved_mib, pri_headroom, sec_headroom, components, label in candidates:
        primary_required = max(0, int(model_dense_mib) - moved_mib) + int(pri_headroom) + int(base_overhead_mib)
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


# --------------------------------------------------------------------------------------------------------------------
# Post-load refinement: assign whole components to cards from their LOADED sizes (covers the MoE second expert).
# --------------------------------------------------------------------------------------------------------------------


def choose_component_assignment(
    sizes: dict[str, tuple[int, str]],
    *,
    headroom_mib: int,
    overhead_mib: int,
    primary_limit_mib: int,
    secondary_limit_mib: int,
) -> Optional[tuple[str, ...]]:
    """Names to place on the secondary so each card holds its share resident, or None. ``sizes`` maps component name
    -> (MiB, role) with role 'dit' / 'text_encoder' / 'other'. A card running a denoiser or the VAE (any 'other'
    component) is priced with the runtime headroom; a card holding only text encoders with an encoder forward's room.
    Tried in order: every companion; the text encoders alone; the second denoiser; the second denoiser plus the
    companions. The first denoiser always stays on the primary."""
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
    if seconds:
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


def refine_plan_for_multigpu(pipe: Any, plan: Any, logger: Any = None) -> Any:
    """Post-load: an offloading plan becomes a resident split when the LOADED components can be shared by the cards.

    No-op without a selection, for explicit offload modes, a plan already resident (or already split), unified memory
    or an unknown budget. Returns the plan (a ``dataclasses.replace`` copy when it changed)."""
    selection = current_selection()
    if selection is None or plan is None:
        return plan
    if getattr(plan, "secondary_ordinal", None) is not None or plan.offload_policy == "none":
        return plan
    if getattr(plan, "requested_mode", None) not in ("auto", "fast"):
        return plan
    memory = plan.device_memory
    if memory.is_unified or getattr(memory, "device", None) != "cuda":
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
    # the measured figure when the loaded-weights refinement or the calibration table priced this family
    headroom = int(
        plan.estimates.get("measured_runtime_headroom_mib")
        or plan.estimates.get("calibrated_headroom_mib")
        or plan.estimates.get("runtime_headroom_mib")
        or 0
    )
    overhead = int(plan.estimates.get("base_overhead_mib") or 0)
    moved = choose_component_assignment(
        sizes,
        headroom_mib = headroom,
        overhead_mib = overhead,
        primary_limit_mib = int(primary_limit),
        secondary_limit_mib = int(sec_limit),
    )
    if moved is None:
        return plan
    reason = (
        f"multi-GPU (loaded sizes): {', '.join(moved)} on cuda:{secondary}, the rest resident on "
        f"cuda:{selection.primary}; single card would use '{plan.offload_policy}'"
    )
    if logger is not None:
        logger.info("diffusion.multigpu: %s", reason)
    return dataclasses.replace(
        plan,
        offload_policy = "none",
        vae_tiling = False,
        vae_slicing = False,
        stream_text_encoders = False,
        stream_transformer = True,
        resident_transformer_mib = None,
        resident_text_encoder_mib = None,
        reasons = tuple(plan.reasons) + (reason,),
        secondary_ordinal = secondary,
        secondary_components = tuple(moved),
    )


def secondary_device_memory(ordinal: int) -> Any:
    """A ``DeviceMemory`` reading for a secondary card, or None when unreadable."""
    from .diffusion_memory import DeviceMemory

    free, total = _free_total_mib(int(ordinal))
    if free is None:
        return None
    return DeviceMemory(backend = "cuda", device = "cuda", memory_kind = "discrete_vram", free_mib = free, total_mib = total)


def secondary_safe_budget_mib(ordinal: int) -> Optional[int]:
    """The single-card planner's safe budget for ``ordinal`` (same reserve rule as the primary)."""
    from .diffusion_memory import _safe_device_budget_mib

    memory = secondary_device_memory(ordinal)
    return None if memory is None else _safe_device_budget_mib(memory)


def secondary_resident_limit_mib(ordinal: int, mode: str) -> Optional[int]:
    """The resident limit of ``ordinal`` under ``mode``'s rule: ``fast`` = the fast budget, otherwise 85% of the safe
    budget (the single-card planner's two rules)."""
    from .diffusion_memory import _fast_device_budget_mib

    if mode == "fast":
        memory = secondary_device_memory(ordinal)
        return None if memory is None else _fast_device_budget_mib(memory)
    budget = secondary_safe_budget_mib(ordinal)
    return None if budget is None else int(int(budget) * _RESIDENT_FRACTION)


# --------------------------------------------------------------------------------------------------------------------
# Placement: whole components on their card, entry points moving tensors across the boundary.
# --------------------------------------------------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def move_to_device(obj: Any, device: Any, _depth: int = 0) -> Any:
    """``obj`` with every tensor inside on ``device`` (exact copies). Containers keep their type; a tensor already
    there is returned as is."""
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)
    if _depth > 6 or obj is None or isinstance(obj, (str, bytes, int, float, bool, torch.device, torch.dtype)):
        return obj
    if isinstance(obj, dict):
        # ModelOutput is an OrderedDict that also mirrors its keys as attributes; __setitem__ keeps both in sync.
        try:
            out = copy.copy(obj)
            for key, value in obj.items():
                moved = move_to_device(value, device, _depth + 1)
                if moved is not value:
                    out[key] = moved
            return out
        except Exception:  # noqa: BLE001
            return type(obj)((k, move_to_device(v, device, _depth + 1)) for k, v in obj.items())
    if isinstance(obj, tuple):
        moved = [move_to_device(v, device, _depth + 1) for v in obj]
        if all(a is b for a, b in zip(moved, obj)):
            return obj
        if hasattr(obj, "_fields"):  # namedtuple
            return type(obj)(*moved)
        return type(obj)(moved) if type(obj) is not tuple else tuple(moved)
    if isinstance(obj, list):
        return [move_to_device(v, device, _depth + 1) for v in obj]
    if isinstance(obj, torch.nn.Module):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        changes = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name, None)
            moved = move_to_device(value, device, _depth + 1)
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
    # Plain value holders only (diffusers' DiagonalGaussianDistribution): every attribute a tensor or a scalar.
    if isinstance(state, dict) and state and any(isinstance(v, torch.Tensor) for v in state.values()):
        if all(
            isinstance(v, (torch.Tensor, bool, int, float, str, type(None), torch.dtype, torch.device))
            for v in state.values()
        ):
            out = copy.copy(obj)
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device != device:
                    setattr(out, k, v.to(device))
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
    """Wrap ``module``'s entry points: arguments go to the module's CURRENT card, results come back to ``home``. The
    card is read per call, so a later ``.to()`` of the whole pipeline keeps working. Returns the wrapped names."""
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
            return boundary

        setattr(module, name, _make(bound))
        wrapped_names.append(name)
    return wrapped_names


def remove_boundary(module: Any) -> None:
    for name in _ENTRY_METHODS:
        current = module.__dict__.get(name) if hasattr(module, "__dict__") else None
        if current is not None and getattr(current, "_unsloth_boundary", None) is not None:
            try:
                delattr(module, name)
            except AttributeError:
                pass


_EXEC_PATCH_LOCK = threading.Lock()


def _split_home(pipe: Any) -> Any:
    try:
        components = pipe.components
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(components, dict):
        return None
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
    """Every module component the denoise loop does not run per step."""
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
        str(n): m for n, m in (getattr(pipe, "components", {}) or {}).items() if isinstance(m, torch.nn.Module)
    }
    placement: dict[str, str] = {}
    # Secondary first: its bytes never touch the primary.
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
            "diffusion.multigpu: placed %s on %s, the rest on %s", ", ".join(moved) or "nothing", secondary_dev, primary_dev
        )
    return placement


def undo_split(pipe: Any) -> None:
    """Remove the boundary wrappers and home markers (the caller then places the pipeline on one device)."""
    torch = _torch()
    for module in (getattr(pipe, "components", {}) or {}).values():
        if isinstance(module, torch.nn.Module):
            remove_boundary(module)
            if "_unsloth_split_home" in module.__dict__:
                del module.__dict__["_unsloth_split_home"]
    try:
        _PLACEMENTS.pop(pipe, None)
    except TypeError:
        pass


def apply_split_plan(pipe: Any, plan: Any, *, placement: str, logger: Any = None) -> bool:
    """Place a split plan. True when placed; on any failure the split is undone and False is returned (the caller
    then applies its single-card fallback)."""
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
            logger.warning("diffusion.multigpu: split placement failed (%s); falling back to one card", exc)
        undo_split(pipe)
        try:
            pipe.to("cpu")
            _torch().cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return False


def placement_status(pipe: Any) -> Optional[dict[str, Any]]:
    if pipe is None:
        return None
    try:
        info = _PLACEMENTS.get(pipe)
        cfg = _CFG_ROUTERS.get(pipe)
        declined = _CFG_DECLINES.get(pipe)
    except TypeError:  # not weak-referenceable: nothing can have been recorded for it
        return None
    if info is None and cfg is None and declined is None:
        return None
    out: dict[str, Any] = dict(info or {})
    if cfg is not None:
        out["cfg_parallel"] = cfg.describe()
    elif declined is not None:
        out["cfg_parallel"] = {"armed": False, "declined": declined}
    return out


# --------------------------------------------------------------------------------------------------------------------
# CFG-parallel: odd guidance branches on a replica of the denoiser on the secondary card.
# --------------------------------------------------------------------------------------------------------------------

_COMPILE_RECORDER_LOCK = threading.Lock()


def install_compile_recorder() -> bool:
    """Record ``nn.Module.compile`` arguments on the module, so a replica can be compiled exactly like its source.
    Inert otherwise (the original method runs unchanged)."""
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


def _hooks_present(module: Any) -> Optional[str]:
    """Name of the first submodule carrying a forward hook or an offload / cache hook registry, else None. A replica
    cannot inherit such state faithfully (closures point at the source), so CFG-parallel declines."""
    for name, sub in module.named_modules():
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            if getattr(sub, attr, None):
                return f"{name or '<root>'}.{attr}"
        if getattr(sub, "_diffusers_hook", None) is not None:
            registry = sub._diffusers_hook
            hooks = getattr(registry, "hooks", None)
            if hooks:
                return f"{name or '<root>'}._diffusers_hook"
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'}._hf_hook"
        # An instance-level forward that is a plain function or closure would keep pointing at the source module once
        # copied (deepcopy treats functions as atoms). Bound methods are rebound to the copy, the router is dropped.
        override = sub.__dict__.get("forward")
        if (
            override is not None
            and not isinstance(override, (types.MethodType, CfgParallelRouter))
            and callable(override)
        ):
            return f"{name or '<root>'}.forward (instance override)"
    return None


_DROP = object()


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
    dropped = []
    for sub in module.modules():
        for key, value in sub.__dict__.items():
            if isinstance(value, CfgParallelRouter):
                memo[id(value)] = _DROP
                dropped.append(key)
    memo[id(memo)] = keep  # deepcopy's own keep-alive convention
    replica = copy.deepcopy(module, memo)
    for sub in replica.modules():
        for key in [k for k, v in sub.__dict__.items() if v is _DROP]:
            del sub.__dict__[key]
    compiled_subs = 0
    for (name, src), (_rname, dst) in zip(module.named_modules(), replica.named_modules()):
        # Plain tensor attributes (cached RoPE tables and the like) were deep-copied on the source card.
        for key, value in list(dst.__dict__.items()):
            if isinstance(value, torch.Tensor) and value.device != device and value.device.type == "cuda":
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
    replica._unsloth_replica_compiled = compiled_subs
    replica.eval()
    return replica


_ADAPTER_ATTRS = ("scaling", "active_adapters", "merged_adapters", "disable_adapters", "_disable_adapters")


def weights_fingerprint(module: Any) -> int:
    """Changes whenever the denoiser's weights or adapter state do: a parameter replaced, edited in place (fused
    LoRA bumps the version counter), added or removed (PEFT injection), or an adapter's scale / activation changed."""
    parts: list = []
    for name, param in module.named_parameters():
        try:
            version = param._version
        except Exception:  # noqa: BLE001 - inference tensors carry no version
            version = -1
        try:
            pointer = param.data_ptr()
        except Exception:  # noqa: BLE001 - wrapper subclasses (torchao) have no storage of their own
            pointer = id(param)
        parts.append((name, pointer, version))
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
        self._thread = threading.Thread(target = self._loop, name = f"unsloth-cfg-replica-{device}", daemon = True)
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

    def call(self, fn: Callable[[], Any]) -> Any:
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
        done["event"].wait()
        if "error" in done:
            raise done["error"]
        return done.get("result")

    def close(self) -> None:
        self._jobs.put(None)


class CfgParallelRouter:
    """Installed as the denoiser's ``forward``: routes odd branches of a step to the replica."""

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
        }
        self.fingerprint: Optional[int] = None

    # -- status
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
                self.logger.warning("diffusion.multigpu: CFG-parallel disabled (%s); one card from here", reason)
        self.drop_replica()

    def drop_replica(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self.replica = None
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None

    # -- the forward
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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
            if self.cfg_seen and self.replica is not None and self.disabled_reason is None:
                # Copy this step's inputs to the replica card BEFORE the primary branch is queued: a copy queued after
                # it would wait for it on the source stream and serialise the two branches.
                self._prefetch(args, kwargs)
        if self.branch % 2 == 1 and self.disabled_reason is None:
            self.singles = 0
            if self.replica is None:
                self.cfg_seen = True
                self._build_replica()
            elif self.cfg_seen:
                try:
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
                self.logger.info("diffusion.multigpu: denoiser weights changed; rebuilding the CFG-parallel replica")
            self.stats["replica_rebuilds"] += 1
            self.drop_replica()

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
            if free is not None and need_mib + 1024 > free:
                raise RuntimeError(f"replica needs {need_mib} MiB, {free} MiB free on {self.secondary}")
            self.fingerprint = weights_fingerprint(module)
            with _device_context(self.secondary):
                self.replica = self.replica_factory()
            self.worker = _ReplicaWorker(self.secondary)
            # the worker thread ends with the router (the module drops it on unload)
            weakref.finalize(self, self.worker.close)
            self.stats["replica_build_s"] = round(time.perf_counter() - t0, 3)
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: CFG-parallel replica on %s built in %.2fs", self.secondary, time.perf_counter() - t0
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
        r_args = self._to_replica(args)
        r_kwargs = self._to_replica(kwargs)
        replica = self.replica
        out = self.worker.call(lambda: replica(*r_args, **r_kwargs))
        return move_to_device(out, self.primary)


_CFG_ROUTERS: "weakref.WeakKeyDictionary[Any, CfgParallelRouter]" = weakref.WeakKeyDictionary()
# Why CFG-parallel was not armed on a multi-card load, for status.
_CFG_DECLINES: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def cfg_parallel_decline(pipe: Any, *, offload_policy: str, selection: Optional[MultiGpuSelection]) -> Optional[str]:
    """Why CFG-parallel cannot be armed on this load, or None."""
    if selection is None:
        return "one card selected"
    mode = cfg_parallel_mode()
    if mode == "off":
        return f"{CFG_PARALLEL_ENV}=0"
    if offload_policy != "none":
        return f"denoiser is offloaded ({offload_policy})"
    denoiser = getattr(pipe, "transformer", None)
    torch = _torch()
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser (a U-Net batches its guidance branches)"
    if getattr(pipe, "transformer_2", None) is not None:
        return "two-expert denoiser"
    if getattr(denoiser, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    if _module_device(denoiser) is None or _module_device(denoiser).type != "cuda":
        return "denoiser is not on a CUDA card"
    return None


def install_cfg_parallel(pipe: Any, *, offload_policy: str, logger: Any = None) -> Optional[str]:
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
        if not candidates:
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


def uninstall_cfg_parallel(pipe: Any) -> None:
    try:
        router = _CFG_ROUTERS.pop(pipe, None) if pipe is not None else None
    except TypeError:
        return
    if router is None:
        return
    router.drop_replica()
    module = router.module_ref()
    if module is not None and module.__dict__.get("forward") is router:
        del module.__dict__["forward"]
        module.__dict__.pop("_unsloth_cfg_router", None)
