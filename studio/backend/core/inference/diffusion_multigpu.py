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
CFG_BATCH_ENV = "UNSLOTH_DIFFUSION_CFG_BATCH"
# Overrides every measured card-to-card copy rate (GiB/s) used by strategy pricing.
LINK_ENV = "UNSLOTH_DIFFUSION_LINK_GIB_S"

# Denoiser attributes: they stay on the primary card under the companion split (the same contract as the speed and
# offload layers, diffusion_autoopt/targets.py).
DENOISER_ATTRS: tuple[str, ...] = (
    "transformer",
    "transformer_2",
    "unconditional_transformer",
    "unet",
)
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


def cfg_batch_mode() -> str:
    """``off`` / ``auto`` (default: only an offloaded denoiser) / ``force`` (a resident one too, as ComfyUI does)."""
    raw = (os.environ.get(CFG_BATCH_ENV) or "").strip().lower()
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


def make_selection(
    primary: Optional[int], ordinals: Optional[Sequence[int]]
) -> Optional[MultiGpuSelection]:
    """A selection when at least two distinct cards were picked and the lever is on, else None (one card)."""
    if primary is None or multigpu_mode() == "off":
        return None
    others = tuple(dict.fromkeys(int(o) for o in (ordinals or ()) if int(o) != int(primary)))
    # The load dtype and the device-specific patches are resolved for the primary (bf16 only from Ampere, FP8 kernels
    # from Ada): a card of a lower compute capability may not run them, so it takes no component.
    need = _capability(int(primary))
    if need is not None:
        others = tuple(o for o in others if (_capability(o) or need) >= need)
    if not others:
        return None
    return MultiGpuSelection(primary = int(primary), secondaries = others)


def _capability(ordinal: int) -> Optional[tuple[int, int]]:
    """``(major, minor)`` CUDA compute capability of ``ordinal``, None when it cannot be read."""
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(int(ordinal))
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def selection_scope(
    selection: Optional[MultiGpuSelection],
) -> Iterator[Optional[MultiGpuSelection]]:
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
                {"ordinal": o, "peer_access": access, "copy_gib_s": gbps}
                for o, access, gbps in self.peers
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

        from utils.hardware import trusted_mem_get_info

        # not torch.cuda.mem_get_info: on Windows ROCm its free figure ignores what this process already holds
        free, total = trusted_mem_get_info(int(ordinal), module = torch.cuda)
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
                # a failed measurement is not cached: the next probe tries again
                if a == b or (
                    (a, b) in _PAIR_PROBE and (_PAIR_PROBE[(a, b)][1] is not None or not measure)
                ):
                    continue
                try:
                    access = bool(torch.cuda.can_device_access_peer(a, b))
                except Exception:  # noqa: BLE001
                    access = False
                gbps = None
                if measure:
                    try:
                        src = torch.empty(
                            _BANDWIDTH_MIB << 20, dtype = torch.uint8, device = torch.device("cuda", a)
                        )
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


def link_gib_s(
    src: int,
    dst: int,
    probes: Optional[dict[int, DeviceProbe]] = None,
) -> Optional[float]:
    """Measured ``src`` -> ``dst`` copy rate in GiB/s (peer or via host), None when unmeasured; ``LINK_ENV`` overrides."""
    raw = (os.environ.get(LINK_ENV) or "").strip()
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    if probes is None or src not in probes:
        probes = probe_devices((src, dst))
    probe = probes.get(int(src))
    if probe is None:
        return None
    for other, _access, gbps in probe.peers:
        if other == int(dst):
            return gbps
    return None


# Links within this fraction of the fastest count as equally fast (probe noise).
_LINK_TIER = 0.75


def rank_secondaries(
    primary: int,
    candidates: Sequence[int],
    free_of: Callable[[int], Optional[int]],
    probes: Optional[dict[int, DeviceProbe]] = None,
) -> list[int]:
    """``candidates`` best first: measured fast-link tier (within ``_LINK_TIER``), then measured slower links, then
    unmeasured links (CFG-parallel pricing refuses those); free memory, then ordinal, within a tier. With no link
    measured at all, every card is one tier (free memory decides)."""
    cands = [int(o) for o in candidates]
    if not cands:
        return []
    if probes is None:
        probes = probe_devices((int(primary), *cands))

    def round_trip(o: int) -> Optional[float]:
        there, back = link_gib_s(primary, o, probes), link_gib_s(o, primary, probes)
        if there is None or back is None:
            return None
        return min(there, back)

    links = {o: round_trip(o) for o in cands}
    known = [v for v in links.values() if v]
    top = max(known) if known else None

    def key(o: int) -> tuple[int, int, int]:
        if top is None:
            tier = 2
        elif links[o] is None:
            tier = 0
        else:
            tier = 2 if links[o] >= _LINK_TIER * top else 1
        free = free_of(o)
        return (tier, free if free is not None else -1, -o)

    return sorted(cands, key = key, reverse = True)


def best_secondary(
    selection: MultiGpuSelection, probes: Optional[dict[int, DeviceProbe]] = None
) -> Optional[int]:
    """Top of :func:`rank_secondaries`."""
    if not selection.secondaries:
        return None
    if probes is None:
        probes = probe_devices((selection.primary, *selection.secondaries))

    def free_of(o: int) -> Optional[int]:
        p = probes.get(o)
        return p.free_mib if p is not None else None

    ranked = rank_secondaries(selection.primary, selection.secondaries, free_of, probes)
    return ranked[0] if ranked else None


def price_cfg_parallel(
    *,
    branch_s: float,
    in_bytes: int,
    out_bytes: int,
    link_in_gib_s: Optional[float],
    link_out_gib_s: Optional[float],
    replica_bytes: int = 0,
    primary_s: Optional[float] = None,
) -> dict[str, Any]:
    """One card (``primary + branch``) vs CFG-parallel (``max(primary, branch + copy)``: the primary keeps its branch,
    the replica runs ``branch`` after its copies). Chosen when it is at most ``_PRICE_PARALLEL_FRACTION`` of one card.
    ``branch_s`` = the measured odd branch, ``primary_s`` = the measured even branch of the same step (``branch_s``
    when unmeasured). With equal branches this is the old rule: copy <= 0.8 of a branch."""
    measured = bool(link_in_gib_s and link_out_gib_s)
    if measured:
        copy_s = in_bytes / (link_in_gib_s * (1 << 30)) + out_bytes / (link_out_gib_s * (1 << 30))
        build_s: Optional[float] = replica_bytes / (link_in_gib_s * (1 << 30))
    else:
        copy_s, build_s = 0.0, None
    primary = branch_s if primary_s is None else primary_s
    single = primary + branch_s
    parallel = max(primary, branch_s + copy_s)
    # an unmeasured link is unknown, not free: one card until a probe measures it (retried on the next request)
    chosen = (
        "cfg_parallel" if measured and parallel <= _PRICE_PARALLEL_FRACTION * single else "single"
    )
    return {
        "strategy": chosen,
        "link_unmeasured": not measured,
        "primary_branch_s": round(primary, 4),
        "branch_s": round(branch_s, 4),
        "copy_s": round(copy_s, 4),
        "single_step_s": round(single, 4),
        "cfg_parallel_step_s": round(parallel, 4),
        "replica_build_s_est": None if build_s is None else round(build_s, 3),
        "link_in_gib_s": link_in_gib_s,
        "link_out_gib_s": link_out_gib_s,
    }


_PRICE_PARALLEL_FRACTION = 0.9


_ROOM_REASON = "replica card lacks room for this shape"
_SHORT_RENDER_REASON = "short render"


def _bg_eager_forced() -> bool:
    try:
        from .diffusion_bg_compile import eager_forced
    except Exception:  # noqa: BLE001
        return False
    return eager_forced()


class ReplicaError(RuntimeError):
    """A CFG-parallel replica failure, typed by what it means; the router's one handler dispatches on the type."""


class ReplicaCapacity(ReplicaError):
    """The replica card lacks room for this request shape: declined for the shape only, re-read every request (other
    shapes, and this one once memory frees, try again)."""


class ReplicaTransient(ReplicaError):
    """Out of memory while copying or running the replica for this shape: declined for the shape, its in-flight
    copies torn down; the replica stays for the shapes that fit."""


class ReplicaStructural(ReplicaError):
    """The denoiser cannot be reproduced on a copy (wrapper, root override, compile record, unverifiable weights, a
    replica call that fails for any other reason): CFG-parallel off for the load."""


def replica_failure_kind(exc: BaseException) -> type:
    """The outcome class of a failure in a replica path: its own type when typed, out of memory as transient, and
    anything else as structural (an unknown failure is never retried per shape)."""
    if isinstance(exc, ReplicaError):
        return type(exc) if type(exc) is not ReplicaError else ReplicaStructural
    if is_out_of_memory(exc):
        return ReplicaTransient
    return ReplicaStructural


def is_out_of_memory(exc: BaseException) -> bool:
    """Studio's one out-of-memory predicate (diffusion_batched.is_oom_error): class names across torch versions and
    backends (``OutOfMemoryError_``), backend messages, and the ``__cause__`` / ``__context__`` chain a torch.compile
    or accelerate wrapper hides an allocation failure behind."""
    from .diffusion_batched import is_oom_error

    return is_oom_error(exc)


# Pricing and verification are per request shape: (canvas, every input tensor's shape: latent batch, tokens,
# conditioning length). The few most recent shapes keep their verdicts; a new shape is priced on its own first
# guided step.
_SHAPE_BOOK_SIZE = 4


def _request_shape(request_px: Optional[int], args: tuple, kwargs: dict) -> tuple:
    return (request_px, tuple(tuple(t.shape) for t in _iter_tensors((args, kwargs))))


_MISS_LIMIT = 3


class _PredictionPolicy:
    """Per request shape: does the speculative doubled call pay? A hit (the odd branch matched what was batched)
    resets the run of misses; a miss (the odd branch changed, or never came) extends it. After ``_MISS_LIMIT``
    consecutive misses batching stops for that shape (each miss costs a doubled call plus a plain rerun: three
    branches of compute for two). Reset rules: a new request shape starts with a fresh policy; a shape seen before
    resumes its own (from the shape book); ``reprice()`` (the denoiser changed) clears every shape's."""

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.consecutive = 0

    def hit(self) -> None:
        self.hits += 1
        self.consecutive = 0

    def miss(self) -> bool:
        """Record a miss; True when batching should stop for the shape."""
        self.misses += 1
        self.consecutive += 1
        return self.consecutive >= _MISS_LIMIT


def _fresh(default: Any) -> Any:
    """A shape-book default for a new shape: a copy of a list, a new instance of a class, else the value."""
    if isinstance(default, list):
        return list(default)
    if isinstance(default, type):
        return default()
    return default


class _ShapeBook:
    """Per-shape copies of a router's pricing fields, most recent last, bounded."""

    def __init__(self, fields: dict):
        self.defaults = fields
        self.key: Any = None
        self.saved: "OrderedDict[Any, dict]" = OrderedDict()

    def switch(self, router: Any, key: Any) -> bool:
        """Make ``key`` the router's current shape; True when it changed."""
        if key == self.key:
            return False
        if self.key is None and not self.saved:
            # the first shape owns whatever the router holds already
            self.key = key
            return True
        # take the requested shape out before saving the outgoing one, so a full book never evicts it
        state = self.saved.pop(key, None)
        if self.key is not None:
            self.saved[self.key] = {name: getattr(router, name) for name in self.defaults}
            self.saved.move_to_end(self.key)
            while len(self.saved) > _SHAPE_BOOK_SIZE:
                self.saved.popitem(last = False)
        for name, default in self.defaults.items():
            value = state[name] if state is not None else _fresh(default)
            setattr(router, name, list(value) if isinstance(value, list) else value)
        self.key = key
        return True

    def clear(self, router: Any) -> None:
        self.saved.clear()
        self.key = None
        for name, default in self.defaults.items():
            setattr(router, name, _fresh(default))

    def forget(self, router: Any, predicate: Callable[[dict], bool]) -> None:
        """Drop saved shapes (and reset the current one) whose state matches ``predicate``."""
        for key in [k for k, v in self.saved.items() if predicate(v)]:
            del self.saved[key]
        current = {name: getattr(router, name) for name in self.defaults}
        if predicate(current):
            for name, default in self.defaults.items():
                setattr(router, name, _fresh(default))


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
        best = (
            rank_secondaries(selection.primary, fits, limits.get)[0] if len(fits) > 1 else fits[0]
        )
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
    encode_headroom_mib: Optional[int] = None,
) -> Optional[tuple[str, ...]]:
    """Names to place on the secondary so each card holds its share resident, or None. ``sizes`` maps component name
    -> (MiB, role) with role 'dit' / 'text_encoder' / 'other'. A card running a denoiser or the VAE (any 'other'
    component) is priced with the runtime headroom; a card holding only text encoders with an encoder forward's room:
    ``encode_headroom_mib`` (the plan's calibrated encode phase) when known, else the static allowance.
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

    encode_room = int(encode_headroom_mib) if encode_headroom_mib else TEXT_ENCODER_HEADROOM_MIB

    def need(names: list) -> int:
        weights = sum(sizes[n][0] for n in names)
        busy = any(sizes[n][1] != "text_encoder" for n in names)
        return weights + (headroom_mib if busy else encode_room) + overhead_mib

    for moved in candidates:
        on_primary = [n for n in sizes if n not in moved]
        if need(on_primary) <= primary_limit_mib and need(list(moved)) <= secondary_limit_mib:
            return moved
    return None


def refine_plan_for_multigpu(
    pipe: Any,
    plan: Any,
    logger: Any = None,
) -> Any:
    """Post-load: an offloading plan becomes a resident split when the LOADED components can be shared by the cards.

    No-op without a selection, for explicit offload modes, a plan already resident (unless forced) or split, unified memory
    or an unknown budget. Returns the plan (a ``dataclasses.replace`` copy when it changed)."""
    selection = current_selection()
    if selection is None or plan is None:
        return plan
    if getattr(plan, "secondary_ordinal", None) is not None:
        return plan
    # force: a resident plan too (the pre-load planner keeps one card resident when sizes were unknown)
    if plan.offload_policy == "none" and multigpu_mode() != "force":
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
    first = best_secondary(selection)
    if first is None:
        return plan
    mode = getattr(plan, "requested_mode", None)
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
    # the best card first, then the rest in link-rank order: a fast-link card too small for any share must not hide
    # a slower one that holds it
    rest = [o for o in selection.secondaries if o != first]
    if len(rest) > 1:
        probes = probe_devices((selection.primary, *selection.secondaries))
        rest = rank_secondaries(
            selection.primary,
            rest,
            lambda o: probes[o].free_mib if o in probes else None,
            probes,
        )
    moved, secondary = None, None
    for ordinal in (first, *rest):
        sec_limit = secondary_resident_limit_mib(ordinal, mode)
        if sec_limit is None:
            continue
        moved = choose_component_assignment(
            sizes,
            headroom_mib = headroom,
            overhead_mib = overhead,
            primary_limit_mib = int(primary_limit),
            secondary_limit_mib = int(sec_limit),
            # the same per-phase source as the pre-load split (plan_companion_split's phase_headroom_mib)
            encode_headroom_mib = plan_encode_headroom_mib(plan),
        )
        if moved is not None:
            secondary = ordinal
            break
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


def plan_denoise_headroom_mib(plan: Any) -> Optional[int]:
    """The plan's activation room above the weights (measured, calibrated or estimated), None when unknown."""
    estimates = getattr(plan, "estimates", None) or {}
    # The calibrated denoise phase when the family has one; else the whole-request figure (it includes the VAE decode
    # peak, so it is conservative: video plans and uncalibrated image plans have no per-phase number).
    value = (
        estimates.get("calibrated_denoise_mib")
        or estimates.get("measured_runtime_headroom_mib")
        or estimates.get("calibrated_headroom_mib")
        or estimates.get("runtime_headroom_mib")
    )
    return int(value) if value else None


# The canvas the per-family activation calibration was measured at (CalibratedImageActivation: 1024x1024).
_CALIBRATION_PX = 1024 * 1024


def scaled_headroom_mib(
    headroom_mib: Optional[int],
    reference_px: Optional[int],
    request_px: Optional[int],
    batch: Optional[int] = None,
) -> int:
    """A denoise headroom calibrated at one sample of ``reference_px`` pixels, scaled linearly by the request's canvas
    (when both are known) and by the latent batch (the calibration batch is 1)."""
    need = int(headroom_mib or 0)
    if reference_px and request_px:
        need = int(need * request_px / reference_px)
    if batch and batch > 1:
        need *= int(batch)
    return need


def plan_encode_headroom_mib(plan: Any) -> Optional[int]:
    """The calibrated text-encode phase's room above the weights, None without a per-phase figure (the whole-request
    figure then covers it)."""
    estimates = getattr(plan, "estimates", None) or {}
    if not estimates.get("calibrated_denoise_mib"):
        return None
    value = estimates.get("calibrated_encode_mib")
    return int(value) if value else None


_TEXT_ENCODER_ATTRS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")


@dataclasses.dataclass(frozen = True)
class PhaseNeeds:
    """One request's activation room above the resident weights per phase, in MiB (batch, canvas, frames and
    reference tokens included), with the cards the encode and decode run on. The denoise runs on every card a CFG
    router uses. The phases run one after another, never together: a card's need is the max over the phases it runs."""

    encode_mib: Optional[int] = None
    denoise_mib: Optional[int] = None
    decode_mib: Optional[int] = None
    encode_ordinals: tuple[int, ...] = ()
    decode_ordinal: Optional[int] = None

    def room_on(self, ordinal: Optional[int]) -> int:
        room = int(self.denoise_mib or 0)
        if ordinal is not None and ordinal in self.encode_ordinals and self.encode_mib:
            room = max(room, int(self.encode_mib))
        if ordinal is not None and ordinal == self.decode_ordinal and self.decode_mib:
            room = max(room, int(self.decode_mib))
        return room


def _cuda_ordinal(module: Any) -> Optional[int]:
    try:
        device = _module_device(module) if module is not None else None
    except Exception:  # noqa: BLE001
        return None
    return device.index if getattr(device, "type", None) == "cuda" else None


def text_encoder_ordinals(pipe: Any) -> tuple[int, ...]:
    """CUDA cards holding one of ``pipe``'s text encoders (resident there: an offloaded one encodes on the primary)."""
    found = []
    for attr in _TEXT_ENCODER_ATTRS:
        ordinal = _cuda_ordinal(getattr(pipe, attr, None))
        if ordinal is not None and ordinal not in found:
            found.append(ordinal)
    return tuple(found)


def request_phase_needs(
    pipe: Any,
    *,
    encode_mib: Optional[int],
    denoise_mib: Optional[int],
    decode_mib: Optional[int],
) -> PhaseNeeds:
    """A request's per-phase needs (from the generation guard's own estimator) placed on ``pipe``'s cards."""
    return PhaseNeeds(
        encode_mib = None if encode_mib is None else int(encode_mib),
        denoise_mib = None if denoise_mib is None else int(denoise_mib),
        decode_mib = None if decode_mib is None else int(decode_mib),
        encode_ordinals = text_encoder_ordinals(pipe),
        decode_ordinal = _cuda_ordinal(getattr(pipe, "vae", None)),
    )


def plan_decode_headroom_mib(plan: Any) -> Optional[int]:
    """The calibrated VAE decode phase's room above the weights, None when the plan has no per-phase figure (then
    :func:`plan_denoise_headroom_mib` is the whole-request figure, which already includes the decode peak)."""
    estimates = getattr(plan, "estimates", None) or {}
    if not estimates.get("calibrated_denoise_mib"):
        return None
    value = estimates.get("calibrated_decode_mib")
    return int(value) if value else None


def plan_denoise_reference_px(plan: Any) -> Optional[int]:
    """The canvas (pixels) :func:`plan_denoise_headroom_mib`'s figure holds for when it is the calibrated denoise
    phase, else None (a whole-request figure is not scaled)."""
    estimates = getattr(plan, "estimates", None) or {}
    return _CALIBRATION_PX if estimates.get("calibrated_denoise_mib") else None


def secondary_device_memory(ordinal: int) -> Any:
    """A ``DeviceMemory`` reading for a secondary card, or None when unreadable."""
    from .diffusion_memory import DeviceMemory

    free, total = _free_total_mib(int(ordinal))
    if free is None:
        return None
    return DeviceMemory(
        backend = "cuda", device = "cuda", memory_kind = "discrete_vram", free_mib = free, total_mib = total
    )


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


def move_to_device(
    obj: Any,
    device: Any,
    _depth: int = 0,
) -> Any:
    """``obj`` with every tensor inside on ``device`` (exact copies). Containers keep their type; a tensor already
    there is returned as is."""
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        return obj if obj.device == device else obj.to(device)
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
        str(n): m
        for n, m in (getattr(pipe, "components", {}) or {}).items()
        if isinstance(m, torch.nn.Module)
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
            "diffusion.multigpu: placed %s on %s, the rest on %s",
            ", ".join(moved) or "nothing",
            secondary_dev,
            primary_dev,
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


def apply_split_plan(
    pipe: Any,
    plan: Any,
    *,
    placement: str,
    logger: Any = None,
) -> bool:
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
    """True when ``pipe``'s components were placed across cards by :func:`place_split`."""
    if pipe is None:
        return False
    try:
        return pipe in _PLACEMENTS
    except TypeError:
        return False


def placement_status(pipe: Any) -> Optional[dict[str, Any]]:
    if pipe is None:
        return None
    try:
        info = _PLACEMENTS.get(pipe)
        cfg = _CFG_ROUTERS.get(pipe)
        declined = _CFG_DECLINES.get(pipe)
        batch = _CFG_BATCH.get(pipe)
    except TypeError:  # not weak-referenceable: nothing can have been recorded for it
        return None
    if info is None and cfg is None and declined is None and batch is None:
        return None
    out: dict[str, Any] = dict(info or {})
    if batch is not None:
        out["cfg_batch"] = batch.describe()
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
        # a functools.partial carries the wrapped forward's __module__; its .func names the real origin
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
    """'none', 'offload' (only group-offload hooks) or 'other' for a module's diffusers hook registry."""
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
            except ValueError:  # empty cell
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
    # any other object (a hook object, a captured state holder): portable when its attributes are
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
    """The one hook predicate both CFG routers screen with (submodule and root hooks alike): a Studio hook that captures
    nothing a copy or a second branch could diverge on, or a Studio hook marked ``_unsloth_shared_ok`` (it touches only
    process-global state or its call's own arguments, never module state the forward reads)."""
    return _hook_origin(fn).startswith(_TRUSTED_HOOK_MODULES) and (
        _portable(fn) or getattr(fn, "_unsloth_shared_ok", False) is True
    )


def _studio_outer_layers(forward: Any) -> tuple[list[Any], Any]:
    """Peel Studio's own outer forward layers (``_unsloth_outer_forward``: the static step skip, defined in Studio's
    modules) off an instance forward: (the layers, what they wrap; None = the class forward). A layer whose class is
    not Studio's, or one that hides what it wraps, stops the peel and stays in place for the caller to judge."""
    layers: list[Any] = []
    fn = forward
    for _ in range(8):
        if (
            fn is None
            or not getattr(fn, "_unsloth_outer_forward", False)
            or not str(type(fn).__module__ or "").startswith(_TRUSTED_HOOK_MODULES)
            or not hasattr(fn, "inner")
        ):
            break
        layers.append(fn)
        fn = fn.inner
    return layers, fn


def _hooks_present(
    module: Any,
    *,
    offload_hooks_ok: bool = False,
    strict_bound: bool = False,
) -> Optional[str]:
    """Name of the first submodule carrying state a replica cannot inherit faithfully, else None: a non-offload hook
    registry (step caches), an accelerate hook, or a forward hook / instance forward override below the root whose
    captured state points at the source (deepcopy treats functions as atoms). Offload machinery (diffusers group
    offloading, Studio's prefetchers) is dropped by :func:`build_replica`, which holds every weight resident. The
    ROOT's hooks are screened too: for CFG-parallel only their origin matters (the copy never carries them; a foreign
    one writing module state would update the source only), for CFG-batch the full :func:`_trusted_hook` check. ``offload_hooks_ok``: accelerate's offload hooks pass too
    (CFG-batch runs the source module itself, so they keep working)."""
    for name, sub in module.named_modules():
        kind = _registry_kind(sub)
        if kind == "other":
            return f"{name or '<root>'}._diffusers_hook ({', '.join(sorted(sub._diffusers_hook.hooks))})"
        if getattr(sub, "_hf_hook", None) is not None and not offload_hooks_ok:
            return f"{name or '<root>'}._hf_hook"
        # Root hooks are screened for both routers. CFG-batch computes the odd branch inside the first branch's root
        # hook run, so they pass the full trusted-hook check there. A CFG-parallel replica is built WITHOUT the root
        # hooks (the source runs them around every branch): a foreign root hook writing module state the forward reads
        # would update the source only, so it blocks; Studio's own root hooks and offload hooks pass.
        for attr in ("_forward_hooks", "_forward_pre_hooks"):
            for hook in (getattr(sub, attr, None) or {}).values():
                if _is_offload_object(hook):
                    continue
                if not name and not offload_hooks_ok:
                    # CFG-parallel root: the copy never carries it (the source runs it around every branch), so
                    # portability is moot; Studio's root hooks pass (their per-call module state is mirrored), a foreign
                    # one blocks (it may write module state only the source would see)
                    if not _hook_origin(hook).startswith(_TRUSTED_HOOK_MODULES):
                        return f"<root>.{attr}"
                    continue
                if not _trusted_hook(hook):
                    return f"{name or '<root>'}.{attr}"
        override = sub.__dict__.get("forward")
        if not name and not offload_hooks_ok:
            # CFG-parallel: Studio's outer layers above the router run on the source around every branch and are
            # never copied into the replica (build_replica drops them); judge what they wrap
            _layers, override = _studio_outer_layers(override)
        if (
            strict_bound
            and isinstance(override, types.MethodType)
            and not (
                override.__self__ is sub
                and override.__func__ is getattr(type(sub).forward, "__func__", type(sub).forward)
            )
            and not _hook_origin(override).startswith(_TRUSTED_HOOK_MODULES)
        ):
            # a foreign bound forward may keep per-call state on its owner: one doubled call is not two
            return (
                f"{name or '<root>'}.forward (bound override from {_hook_origin(override) or '?'}; "
                f"{CFG_BATCH_ENV}=force opts in)"
            )
        if override is None or isinstance(
            override, (types.MethodType, CfgParallelRouter, CfgBatchRouter)
        ):
            # bound methods are rebound to the copy, the router is dropped
            continue
        if kind == "offload" or _is_offload_object(override):
            continue  # the registry's forward wrapper, dropped with it
        if callable(override) and not _trusted_hook(override):
            return f"{name or '<root>'}.forward (instance override)"
    return None


def _offload_attr_keys(sub: Any) -> list[str]:
    """Instance attributes of ``sub`` that belong to offload machinery (never copied into a replica)."""
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
            return  # the class forward: the copy already has it
        replica.__dict__["forward"] = types.MethodType(original.__func__, replica)
        return
    raise ReplicaStructural(
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
    # Studio's outer layers above the root forward (the static step skip) stay on the source, which runs them around
    # every branch: the copy gets what they wrap
    outer_layers, router = _studio_outer_layers(module.__dict__.get("forward"))
    for layer in outer_layers:
        memo[id(layer)] = _DROP
    memo[id(memo)] = keep  # deepcopy's own keep-alive convention
    replica = copy.deepcopy(module, memo)
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
    ledger: list = []
    replica.__dict__[_REPLICA_LEDGER] = ledger
    try:
        compiled_subs = _compile_replica_blocks(module, replica, device, ledger)
    except BaseException:
        release_replica(replica)  # a half-built copy leaves nothing registered behind
        raise
    replica._unsloth_replica_compiled = compiled_subs
    replica.eval()
    return replica


def _compile_replica_blocks(module: Any, replica: Any, device: Any, ledger: list) -> int:
    torch = _torch()
    compiled_subs = 0
    for (name, src), (_rname, dst) in zip(module.named_modules(), replica.named_modules()):
        # Plain tensor attributes (cached RoPE tables and the like) were deep-copied on the source card.
        for key, value in list(dst.__dict__.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.device != device
                and value.device.type == "cuda"
            ):
                dst.__dict__[key] = value.to(device)
        if getattr(src, "_compiled_call_impl", None) is not None:
            compile_like_source(src, dst, owner = replica, name = name, ledger = ledger)
            compiled_subs += 1
        elif "_compiled_call_impl" in dst.__dict__:
            dst._compiled_call_impl = None
    return compiled_subs


# Studio wrappers a source block's ``_compiled_call_impl`` can carry, by marker. Each is rebuilt on the replica's copy
# by compile_like_source (the guard and the background-compile gate keep the copy eager while the source is; the
# restride keeps one compiled graph). CUDA-graph wrappers (GraphedForward, BlockGraph) hold graphs captured on the
# source card: the copy runs the compiled call without them (same math, no replay).
_GRAPH_WRAPPERS = ("GraphedForward", "BlockGraph")
_LAYER_MARKERS = ("_unsloth_bg_gate", "_unsloth_block_restride", "_unsloth_compile_guard", "_unsloth_stg_adapter")


def _is_torch_compiled(fn: Any) -> bool:
    """torch's own compiled call (Module.compile / torch.compile): the bottom of a wrapper chain."""
    origin = _hook_origin(fn)
    return origin.startswith("torch") or str(getattr(fn, "__module__", "") or "").startswith("torch")


def _wrapper_inner(fn: Any) -> Any:
    """The one link below a wrapper layer, or None: ``__wrapped__`` (Studio's closures and functools.wraps), the
    background gate's ``_unsloth_bg_inner``, a ``functools.partial``'s ``func``, a bound method's owner object, a
    graph wrapper's ``compiled`` / ``compute``."""
    for attr in ("__wrapped__", "_unsloth_bg_inner"):
        inner = getattr(fn, attr, None)
        if inner is not None and inner is not fn:
            return inner
    if hasattr(fn, "func") and hasattr(fn, "args") and hasattr(fn, "keywords"):
        return fn.func
    if isinstance(fn, types.MethodType):
        owner = fn.__self__
        return owner if owner is not fn else None
    for attr in ("compiled", "compute"):
        inner = getattr(fn, attr, None)
        if callable(inner) and inner is not fn:
            return inner
    return None


def _source_wrapper_markers(outer: Any, name: str) -> dict:
    """Walk ``outer`` (a source block's ``_compiled_call_impl``) layer by layer down to torch's compiled call, through
    one inner link per layer (:func:`_wrapper_inner`), recording the Studio layers ({marker: value}). Raises on a layer
    it cannot rebuild exactly or cannot see below: a wrapper is never dropped silently."""
    found: dict = {}
    fn = outer
    for _ in range(16):
        if fn is None:
            raise ReplicaStructural(f"compiled submodule '{name}': the wrapper chain ends without torch's compiled call")
        if _is_torch_compiled(fn) and not any(getattr(fn, m, None) for m in _LAYER_MARKERS):
            return found
        if getattr(fn, "_unsloth_stg_adapter", False):
            raise ReplicaStructural(f"compiled submodule '{name}' carries the LTX-2 STG adapter (not rebuilt on a copy)")
        markers = [m for m in _LAYER_MARKERS if getattr(fn, m, None) not in (None, False)]
        graph = type(fn).__name__ in _GRAPH_WRAPPERS and _hook_origin(fn).startswith(_TRUSTED_HOOK_MODULES)
        if not markers and not graph and not isinstance(fn, types.MethodType):
            raise ReplicaStructural(
                f"compiled submodule '{name}' carries an unknown wrapper {type(fn).__name__} (not rebuilt on a copy)"
            )
        guard = getattr(fn, "_unsloth_compile_guard", None)
        if guard is not None:
            found.setdefault("guard", guard)
        for marker in markers:
            found[marker] = True
        below = _wrapper_inner(fn)
        if below is None:
            raise ReplicaStructural(
                f"compiled submodule '{name}': cannot see below wrapper {type(fn).__name__} (no inner link)"
            )
        fn = below
    raise ReplicaStructural(f"compiled submodule '{name}': wrapper chain too deep to inspect")


def compile_like_source(
    src: Any, dst: Any, *, owner: Any = None, name: str = "", ledger: Optional[list] = None
) -> None:
    """Compile a replica block ``dst`` exactly the way its source ``src`` was: the recorded ``Module.compile``
    arguments, then the same Studio wrappers rebuilt around the copy's own compiled call (canonical order: restride,
    the shared compile guard, the background-compile gate). The one builder every replica block goes through."""
    recorded = src.__dict__.get("_unsloth_compile_args")
    if recorded is None:
        raise ReplicaStructural(f"compiled submodule '{name}' has no recorded compile arguments")
    markers = _source_wrapper_markers(src._compiled_call_impl, name)
    args, kwargs = recorded
    dst._compiled_call_impl = None
    dst.compile(*args, **kwargs)
    call = dst._compiled_call_impl
    eager = dst._call_impl
    if markers.get("_unsloth_block_restride"):
        from .diffusion_block_restride import wrap as _restride

        call = _restride(call)
    guard = markers.get("guard")
    if guard is not None:
        # the source's guard object: one compile failure drops the source and the copy to eager together
        call = guard.wrap(call, eager, owner if owner is not None else dst)
        dst_ref = weakref.ref(dst)  # the long-lived source guard must never keep the copy alive

        def restore(ref: Any = dst_ref) -> None:
            m = ref()
            if m is not None:
                m._compiled_call_impl = None

        guard.restores.append(restore)
        if ledger is not None:
            ledger.append(lambda g = guard, f = restore: _remove_identity(g.restores, f))
    if markers.get("_unsloth_bg_gate"):
        from .diffusion_bg_compile import eager_gate

        call = eager_gate(call, eager)
    dst._compiled_call_impl = call


def _remove_identity(items: list, value: Any) -> None:
    for i in range(len(items) - 1, -1, -1):
        if items[i] is value:
            del items[i]


_REPLICA_LEDGER = "_unsloth_replica_ledger"


def release_replica(replica: Any) -> None:
    """The one teardown for every replica drop path (eviction, invalidation, stale weights, disable, uninstall, unload):
    every registration the replica made on source-side objects (compile-guard callbacks, ...) is undone, so nothing
    long-lived keeps the copy, and its card memory, alive. Idempotent."""
    if replica is None:
        return
    ledger = replica.__dict__.get(_REPLICA_LEDGER)
    while ledger:
        undo = ledger.pop()
        with contextlib.suppress(Exception):
            undo()


_ADAPTER_ATTRS = (
    "scaling",
    "active_adapters",
    "merged_adapters",
    "disable_adapters",
    "_disable_adapters",
)


def weights_fingerprint(module: Any) -> int:
    """Changes whenever the denoiser's weights or adapter state do: a parameter replaced, edited in place (fused
    LoRA bumps the version counter), added or removed (PEFT injection), or an adapter's scale / activation changed.
    A ``param.data`` swap to different values is not seen; diffusers' LoRA paths do not do that."""
    parts: list = []
    for name, param in module.named_parameters():
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
        """Queue ``fn`` with the caller's grad / inference / autocast mode and context variables; returns a handle
        for :meth:`wait`."""
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
    """(path, leaf) pairs of a call's arguments; tuples / lists / dicts are walked, anything else is a leaf."""
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


# A per-call scalar the pipeline recomputes inline (``timestep / 1000``) is a new object with the primary call's value.
_SMALL_TENSOR_NUMEL = 4096


def _bitwise_equal(a: Any, b: Any) -> bool:
    torch = _torch()
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return False
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    if a is b:
        return True
    try:
        return bool(
            torch.equal(
                a.reshape(-1).contiguous().view(torch.uint8),
                b.reshape(-1).contiguous().view(torch.uint8),
            )
        )
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
            now = src.to(snap.device).reshape(-1).contiguous().view(torch.uint8)
            checks.append(
                torch.equal(now, snap.reshape(-1).contiguous().view(torch.uint8))
                if snap.device.type == "cpu"
                else (now == snap.reshape(-1).contiguous().view(torch.uint8)).all()
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
    """Copy the source's ``_unsloth_*`` scalar state onto the replica's matching submodules (and drop what the source
    no longer has); returns the mirrored state."""
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
        headroom_mib: Optional[int] = None,
        headroom_reference_px: Optional[int] = None,
        candidates: Optional[Sequence[int]] = None,
        decode_headroom_mib: Optional[int] = None,
        vae_ordinal: Optional[int] = None,
        encode_headroom_mib: Optional[int] = None,
        te_ordinals: Sequence[int] = (),
    ):
        torch = _torch()
        # Load-time (calibrated) phase figures, the fallback when a request supplies none: the VAE decode's room is
        # required on the replica card only when it hosts the VAE, the text encode's only when it hosts an encoder
        self.decode_headroom_mib = decode_headroom_mib
        self.vae_ordinal = vae_ordinal
        self.encode_headroom_mib = encode_headroom_mib
        self.te_ordinals = tuple(int(o) for o in te_ordinals)
        # this request's per-phase needs (Studio's generation guard estimator), when supplied
        self.request_needs: Optional[PhaseNeeds] = None
        # the odd branch's activations on the replica card, priced like CFG-batch's (canvas and batch scaled)
        self.headroom_mib = headroom_mib
        self.headroom_reference_px = headroom_reference_px
        self.request_px: Optional[int] = None
        self.module_ref = weakref.ref(module)
        self.orig_forward = module.forward
        # an instance-level forward (bound override, offload wrapper) the replica must reproduce or drop, and that
        # uninstall puts back
        self.had_override = "forward" in module.__dict__
        self.prev_override = module.__dict__.get("forward")
        self.primary = torch.device(primary)
        self.secondary = torch.device(secondary)
        # every card that fit the replica at install: a later request may move to a better-measured one
        self.candidates: list[int] = [int(o) for o in (candidates or ())]
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
            "replica_blocked": 0,
        }
        # why the replica could not be built on the last attempt, when that is retryable (hooks present)
        self.blocked_by: Optional[str] = None
        # the previous step's odd call, classified against its primary call; the launched prediction
        self.template: Optional[tuple[Any, Any, dict]] = None
        self.pending: Optional[dict] = None
        self.b0_leaves: dict = {}
        self.fingerprint: Optional[int] = None
        self.pairs: list = []
        self.pricing: Optional[dict[str, Any]] = None
        self.b0_s: Optional[float] = None
        # why the CURRENT request shape keeps one card (priced out, link unmeasured); per shape, not per load
        self.shape_disabled: Optional[str] = None
        self.shapes = _ShapeBook({"pricing": None, "shape_disabled": None})
        # the replica card's room for this request's activations was checked (per request and per shape change)
        self.capacity_checked = False
        # this request's denoise steps (counted at each new step) and its step count when Studio / the scheduler
        # knows it: the first replica build is amortised over what is left
        self.request_id = 0
        self.request_step = 0
        self.request_total_steps: Any = None
        self.steps_fn: Optional[Callable[[], Any]] = None
        self.priced_at: Optional[tuple[int, int]] = None

    # -- status
    def describe(self) -> dict[str, Any]:
        off = self.disabled_reason or self.shape_disabled
        return {
            "strategy": "cfg_parallel" if off is None else "single",
            "pricing": self.pricing,
            "shapes_priced": len(self.shapes.saved) + (self.shapes.key is not None),
            "secondary": str(self.secondary),
            "replica_built": self.replica is not None,
            "replica_compiled_submodules": getattr(self.replica, "_unsloth_replica_compiled", None),
            "disabled_reason": off,
            "disabled_for_shape_only": self.disabled_reason is None
            and self.shape_disabled is not None,
            "blocked_by": self.blocked_by,
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
        """Every drop path ends here: the worker, pending work, prefetched copies and the replica's registrations on
        source-side objects (release_replica), so the copy and its card memory are freed."""
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._drop_pending()
        release_replica(self.replica)
        self.replica = None
        self.pairs = []
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}

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
            self._drop_pending()
            self.b0_leaves = dict(_leaves((args, kwargs)))
            self.b0_s = None
            self.request_step += 1
            if self.shapes.switch(self, _request_shape(self.request_px, args, kwargs)):
                self.template = (
                    None  # the previous shape's odd-branch prediction does not carry over
                )
                self.capacity_checked = False
            if (
                not self.capacity_checked
                and self.replica is not None
                and self.disabled_reason is None
            ):
                self.capacity_checked = True
                self._check_capacity(latent)
            if (
                self.cfg_seen
                and self.replica is not None
                and self.disabled_reason is None
                and self.shape_disabled is None
                and (self.pricing is not None or not self._should_price())
                and not _bg_eager_forced()
            ):
                # Copy this step's inputs to the replica card BEFORE the primary branch is queued: a copy queued after
                # it would wait for it on the source stream and serialise the two branches.
                try:
                    self._prefetch(args, kwargs)
                    self._launch_prediction(latent)
                except Exception as exc:  # noqa: BLE001 - dispatched on its type by the one handler
                    self._replica_failure("prefetch", exc)
        if (
            self.branch % 2 == 1
            and self.disabled_reason is None
            and _holds_opaque_tensors((args, kwargs))
        ):
            # A cache object the replica cannot be handed: this branch runs on the primary, in turn.
            self._drop_pending()
            self.template = None
            self.stats["opaque_primary_calls"] += 1
        elif (
            self.branch % 2 == 1
            and self.disabled_reason is None
            and self.shape_disabled is None
            # the source's compile in flight in the background: no replica build or call this render (it would
            # compile the copy inline on the secondary); the branch runs on the primary, eager like the source
            and not _bg_eager_forced()
        ):
            self.singles = 0
            if self.pricing is None and self.replica is not None and self._should_price():
                # a new request shape on a load that already has a replica: price this shape too
                self._drop_pending()
                out, branch_s = self._timed_primary(args, kwargs)
                self.pricing = self._price(args, kwargs, out, branch_s)
                self._shape_verdict()
                if self.branch == 1:
                    self._record_template(args, kwargs, latent)
                return out
            if self.replica is None:
                self.cfg_seen = True
                if self.pricing is None and self._should_price():
                    out, branch_s = self._timed_primary(args, kwargs)
                    self.pricing = self._price(args, kwargs, out, branch_s)
                    self._shape_verdict()
                    # the replica is built on a LATER guided step of this request, if the rest of it repays the build
                    self.priced_at = (self.request_id, self.request_step)
                    if self.branch == 1:
                        self._record_template(args, kwargs, latent)
                    return out
                if not self._should_build():
                    if self.branch == 1:
                        self._record_template(args, kwargs, latent)
                    self.stats["primary_calls"] += 1
                    return self.orig_forward(*args, **kwargs)
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
                except Exception as exc:  # noqa: BLE001 - recomputed on the primary below; typed by the handler
                    self._replica_failure("call", exc)
            self.cfg_seen = True
        if (
            self.branch == 0
            and self.pricing is None
            and self.disabled_reason is None
            and self.shape_disabled is None
            and self.singles == 0
            and self._should_price()
        ):
            # until the first guided step is priced (and only before any single-branch step): time the even branch
            out, self.b0_s = self._timed_primary(args, kwargs)
            return out
        self.stats["primary_calls"] += 1
        return self.orig_forward(*args, **kwargs)

    def _check_capacity(self, latent: Any) -> None:
        """An existing replica, a new request or shape: decline this shape when the replica card lacks room for its
        (canvas and batch scaled) activations. A room verdict is re-read every request: free memory moves."""
        if self.shape_disabled is not None and self.shape_disabled.startswith(_ROOM_REASON):
            self.shape_disabled = None
        if self.shape_disabled is not None:
            return
        free = self._secondary_free_mib()
        if free is None:
            return
        batch = (
            int(latent.shape[0])
            if getattr(latent, "dim", None) is not None and latent.dim() >= 1
            else None
        )
        room = self._room_mib(batch)
        if room is not None and room > free:
            self._shape_off(
                f"{_ROOM_REASON}: needs {int(room)} MiB, {free} MiB free on {self.secondary}"
            )

    def _phase_need_mib(self, phase: str) -> Optional[int]:
        """What ``phase`` (encode / decode) allocates on the replica card, None when that card does not run it: the
        request's own figure, else the load-time one (decode scaled by canvas)."""
        index = self.secondary.index
        needs = self.request_needs
        if phase == "encode":
            if needs is not None and needs.denoise_mib:
                return int(needs.encode_mib or 0) if index in needs.encode_ordinals else None
            return int(self.encode_headroom_mib or 0) if index in self.te_ordinals else None
        if phase == "decode":
            if needs is not None and needs.denoise_mib:
                return int(needs.decode_mib or 0) if index == needs.decode_ordinal else None
            if self.vae_ordinal is None or index != self.vae_ordinal:
                return None
            return scaled_headroom_mib(
                self.decode_headroom_mib, self.headroom_reference_px, self.request_px, None
            )
        return None

    def ensure_phase_room(self, phase: str) -> bool:
        """At a phase boundary that allocates on the replica card (encode at request entry, decode at decoder entry):
        evict the retained replica when the phase would not fit beside it, so a wrong estimate can never cause an OOM
        the one-card path would not have had. True when the replica stays (or there is none)."""
        if self.replica is None:
            return True
        need = self._phase_need_mib(phase)
        if not need:
            return True
        free = self._secondary_free_mib()
        if free is None or need <= free:
            return True
        if self.logger is not None:
            self.logger.info(
                "diffusion.multigpu: evicting the CFG-parallel replica from %s before %s (%d MiB needed, %d free)",
                self.secondary,
                phase,
                need,
                free,
            )
        self.drop_replica()
        self.stats["replica_evictions"] = self.stats.get("replica_evictions", 0) + 1
        with contextlib.suppress(Exception):
            _torch().cuda.empty_cache()
        return False

    def _secondary_free_mib(self) -> Optional[int]:
        """Free MiB on the replica card: the driver's figure plus this process's idle cached allocator blocks there
        (the replica's last activations, reusable without a driver allocation), read on the secondary itself."""
        free, _total = _free_total_mib(self.secondary.index)
        if free is None:
            return None
        if self.secondary.type == "cuda":
            try:
                torch = _torch()
                idle = (
                    torch.cuda.memory_reserved(self.secondary)
                    - torch.cuda.memory_allocated(self.secondary)
                ) // (1 << 20)
                free += max(0, int(idle))
            except Exception:  # noqa: BLE001 - the driver figure alone (conservative)
                pass
        return int(free)

    def _room_mib(self, batch: Optional[int]) -> int:
        """Activation room the replica card needs beside the (retained) replica: the max over the phases that card
        runs. The odd branch's denoise always; the VAE decode when it hosts the VAE (the routers' inputs are released at
        decode entry); the text encode when it hosts an encoder (the replica stays across requests, so it sits beside
        the next request's encode). Phases are sequential, so the max, not the sum. The request's own figures when the
        generate path supplied them; else the load-time figures, denoise and decode scaled by canvas and batch."""
        needs = self.request_needs
        if needs is not None and needs.denoise_mib:
            return needs.room_on(self.secondary.index)
        room = scaled_headroom_mib(self.headroom_mib, self.headroom_reference_px, self.request_px, batch)
        if (
            self.encode_headroom_mib
            and self.secondary.index is not None
            and self.secondary.index in self.te_ordinals
        ):
            room = max(room, int(self.encode_headroom_mib))
        if (
            self.decode_headroom_mib
            and self.vae_ordinal is not None
            and self.secondary.index == self.vae_ordinal
        ):
            room = max(
                room,
                scaled_headroom_mib(
                    self.decode_headroom_mib, self.headroom_reference_px, self.request_px, batch
                ),
            )
        return room

    def _replica_failure(self, where: str, exc: BaseException) -> None:
        """The one handler for every failure in a replica path (build, prefetch, call, staleness check): dispatched on
        :func:`replica_failure_kind` only. Capacity and transient failures decline this request shape (a capacity
        verdict is re-read every request; a transient one tears the in-flight copies down); a structural one turns
        CFG-parallel off for the load."""
        kind = replica_failure_kind(exc)
        if kind is ReplicaCapacity:
            self._shape_off(f"{_ROOM_REASON} ({where}): {exc}")
        elif kind is ReplicaTransient:
            self._shape_oom(where, exc)
        else:
            self.disable(f"replica {where} failed: {type(exc).__name__}: {exc}")

    def _shape_oom(self, where: str, exc: BaseException) -> None:
        self._drop_pending()
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        with contextlib.suppress(Exception):
            _torch().cuda.empty_cache()
        self._shape_off(f"out of memory on the replica card ({where}): {type(exc).__name__}")

    def _shape_off(self, reason: str) -> None:
        self.shape_disabled = reason
        if self.logger is not None:
            self.logger.info(
                "diffusion.multigpu: CFG-parallel off for this request shape (%s)", reason
            )

    def _remaining_steps(self) -> Optional[int]:
        """Denoise steps left in this request, this one included: Studio's step count, else the scheduler's."""
        total = self.request_total_steps
        if total is None and self.steps_fn is not None:
            with contextlib.suppress(Exception):
                total = self.steps_fn()
        try:
            total = int(total) if total else None
        except (TypeError, ValueError):
            total = None
        if total is None:
            return None
        return max(0, total - self.request_step + 1)

    def _should_build(self) -> bool:
        """The one decision for every replica build (the first one and each rebuild after an eviction, a stale-weights
        drop or an invalidation): never inside the background-compile window; an unpriced router (force, non-CUDA)
        builds at once; a priced one never on the step it was priced on, never on the first guided step of a request
        whose length is unknown, and otherwise only when the rest of the request repays the build."""
        if _bg_eager_forced():
            # the source's compile is still in flight in the background (this render runs eager): building, and then
            # first-calling, a compiled copy now would compile it inline on the secondary
            return False
        if self.pricing is None:
            return True
        if self.priced_at == (self.request_id, self.request_step):
            return False
        if self._remaining_steps() is None:
            return self.request_step >= 2
        return self._build_pays_off()

    def _build_pays_off(self) -> bool:
        """Amortise the first replica build over the request's remaining guided steps (this step runs on the primary
        while the replica is built, so it gains nothing). Unknown step count: build (this is already the second guided
        step). Declines this shape for this request only."""
        remaining = self._remaining_steps()
        if remaining is None:
            return True
        gain_s = float(self.pricing["single_step_s"]) - float(self.pricing["cfg_parallel_step_s"])
        build_s = self.pricing.get("replica_build_s_est") or 0.0
        parallel_steps = remaining - 1
        if parallel_steps > 0 and gain_s * parallel_steps > build_s:
            return True
        self._shape_off(
            f"{_SHORT_RENDER_REASON}: building the replica (~{build_s:.2f}s) does not repay over "
            f"{max(0, parallel_steps)} remaining steps at {gain_s:.4f}s saved each"
        )
        return False

    def _shape_verdict(self) -> bool:
        """Apply the current shape's price: True when CFG-parallel runs for it."""
        if self.pricing["strategy"] == "cfg_parallel":
            return True
        if self.pricing.get("link_unmeasured"):
            self.shape_disabled = (
                "link unmeasured: one card until a probe measures it (retried next request)"
            )
        else:
            self.shape_disabled = (
                "priced out: one card takes "
                f"{self.pricing['single_step_s']:.4f}s per step, CFG-parallel "
                f"{self.pricing['cfg_parallel_step_s']:.4f}s (link copies {self.pricing['copy_s']:.4f}s)"
            )
        if self.logger is not None:
            self.logger.info(
                "diffusion.multigpu: CFG-parallel off for this request shape (%s)",
                self.shape_disabled,
            )
        return False

    # -- request lifecycle (Studio's generate path; without it the call pattern decides)
    def begin_request(
        self,
        width: Any = None,
        height: Any = None,
        steps: Any = None,
        steps_fn: Optional[Callable[[], Any]] = None,
        needs: Optional[PhaseNeeds] = None,
    ) -> None:
        self.release_inputs()
        self.request_needs = needs
        self.branch = 0
        self.singles = 0
        # before the encode: it allocates on the replica card when that card hosts a text encoder
        self.ensure_phase_room("encode")
        self.b0_s = None
        self.capacity_checked = False
        self.request_id += 1
        self.request_step = 0
        self.request_total_steps = steps
        self.steps_fn = steps_fn
        # short-render and room verdicts hold for one request: a longer one may repay the build, and free memory moves
        per_request = (_SHORT_RENDER_REASON, _ROOM_REASON)
        for state in self.shapes.saved.values():
            if str(state.get("shape_disabled") or "").startswith(per_request):
                state["shape_disabled"] = None
        if str(self.shape_disabled or "").startswith(per_request):
            self.shape_disabled = None
        if self.replica is None:
            self._reselect_secondary()
        # an unmeasured link is retried on every request
        self.shapes.forget(self, lambda st: bool((st.get("pricing") or {}).get("link_unmeasured")))
        try:
            self.request_px = int(width) * int(height) if width and height else None
        except (TypeError, ValueError):
            self.request_px = None

    def _reselect_secondary(self) -> None:
        """No replica yet: re-rank the install-time cards on a fresh probe (failed link measurements are retried), so
        a card whose link measures now wins over one still unmeasured."""
        if len(self.candidates) < 2 or self.primary.type != "cuda":
            return
        try:
            probes = probe_devices((self.primary.index, *self.candidates))

            def free_of(o: int) -> Optional[int]:
                p = probes.get(o)
                return p.free_mib if p is not None else None

            best = rank_secondaries(self.primary.index, self.candidates, free_of, probes)[0]
        except Exception:  # noqa: BLE001 - keep the current card
            return
        if best != self.secondary.index:
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: CFG-parallel replica card cuda:%s -> cuda:%s (link measured)",
                    self.secondary.index,
                    best,
                )
            self.secondary = _torch().device("cuda", best)
            # every verdict was priced against the old card's link
            self.shapes.clear(self)

    def end_request(self) -> None:
        self.release_inputs()
        self.request_needs = None

    def release_inputs(self) -> None:
        """Drop every tensor kept between calls (the replica and per-load pricing stay)."""
        self._drop_pending()
        self.prefetched.clear()
        self.carry = []
        self.next_carry = []
        self.prev_latent = None
        self.template = None
        self.b0_leaves = {}

    def reprice(self) -> None:
        """Forget every shape's price; a priced-out router is armed again and prices the next guided step."""
        self.shapes.clear(self)
        self.b0_s = None
        if self.disabled_reason is not None and self.disabled_reason.startswith("priced out"):
            self.disabled_reason = None

    # -- pricing (once per router, on the first guided step)
    def _should_price(self) -> bool:
        return (
            cfg_parallel_mode() != "force"
            and self.primary.type == "cuda"
            and self.secondary.type == "cuda"
        )

    def _timed_primary(self, args: tuple, kwargs: dict) -> tuple[Any, float]:
        torch = _torch()
        torch.cuda.synchronize(self.primary)
        t0 = time.perf_counter()
        out = self.orig_forward(*args, **kwargs)
        torch.cuda.synchronize(self.primary)
        self.stats["primary_calls"] += 1
        return out, time.perf_counter() - t0

    def _price(self, args: tuple, kwargs: dict, out: Any, branch_s: float) -> dict[str, Any]:
        in_bytes = sum(t.numel() * t.element_size() for t in _iter_tensors((args, kwargs)))
        out_bytes = sum(t.numel() * t.element_size() for t in _iter_tensors(_as_walkable(out)))
        module = self.module_ref()
        replica_bytes = (
            sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))
            if module is not None
            else 0
        )
        pricing = price_cfg_parallel(
            branch_s = branch_s,
            in_bytes = in_bytes,
            out_bytes = out_bytes,
            link_in_gib_s = link_gib_s(self.primary.index, self.secondary.index),
            link_out_gib_s = link_gib_s(self.secondary.index, self.primary.index),
            replica_bytes = replica_bytes,
            primary_s = self.b0_s,
        )
        if self.logger is not None:
            self.logger.info("diffusion.multigpu: CFG strategy priced %s", pricing)
        return pricing

    # -- prediction of the next odd branch
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
                    # Recomputed per call with the primary call's value (``timestep / 1000``): predicted from the
                    # primary call, verified bitwise at the branch. The check reads a few bytes; the branches are
                    # waited on right after anyway.
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
            return None
        self.stats["predicted_hits"] += 1
        return move_to_device(out, self.primary)

    def _drop_pending(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            # let the replica finish before anything else is queued on it; the result is discarded
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
            self._replica_failure(
                "staleness check", ReplicaStructural(f"cannot fingerprint the denoiser: {exc}")
            )
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
            self._replica_failure(
                "staleness check",
                ReplicaStructural(f"denoiser gained hooks after the replica was built ({hooked})"),
            )

    def _build_replica(self) -> None:
        module = self.module_ref()
        if module is None:
            return
        t0 = time.perf_counter()
        try:
            hooked = _hooks_present(module)
            if hooked is not None:
                # Transient (a recording pre-hook of the background compile, a step cache toggled per request): the
                # odd branch runs on the primary and the next guided step tries again.
                self.blocked_by = f"denoiser carries hooks ({hooked})"
                self.stats["replica_blocked"] += 1
                return
            self.blocked_by = None
            need_mib = sum(
                t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers())
            ) // (1 << 20)
            free = self._secondary_free_mib()
            batch = (
                int(self.prev_latent.shape[0])
                if getattr(self.prev_latent, "dim", None) is not None
                and self.prev_latent.dim() >= 1
                else None
            )
            room = max(1024, self._room_mib(batch))
            if free is not None and need_mib + room > free:
                raise ReplicaCapacity(
                    f"replica needs {need_mib} MiB + {room} MiB room, {free} MiB free on {self.secondary}"
                )
            try:
                self.fingerprint = weights_fingerprint(module)
            except Exception as exc:  # noqa: BLE001 - typed below
                raise ReplicaStructural(f"cannot fingerprint the denoiser: {exc}") from exc
            with _device_context(self.secondary):
                self.replica = self.replica_factory()
            self.pairs = list(zip(module.modules(), self.replica.modules()))
            self.worker = _ReplicaWorker(self.secondary)
            # the worker thread ends with the router (the module drops it on unload)
            weakref.finalize(self, self.worker.close)
            self.stats["replica_build_s"] = round(time.perf_counter() - t0, 3)
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: CFG-parallel replica on %s built in %.2fs",
                    self.secondary,
                    time.perf_counter() - t0,
                )
        except Exception as exc:  # noqa: BLE001 - dispatched on its type by the one handler
            release_replica(self.replica)
            self.replica = None
            self._replica_failure("build", exc)

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
        _mirror_hook_state(self.pairs)
        r_args = self._to_replica(args)
        r_kwargs = self._to_replica(kwargs)
        replica = self.replica
        out = self.worker.call(lambda: replica(*r_args, **r_kwargs))
        return move_to_device(out, self.primary)


_CFG_ROUTERS: "weakref.WeakKeyDictionary[Any, CfgParallelRouter]" = weakref.WeakKeyDictionary()
# Why CFG-parallel was not armed on a multi-card load, for status.
_CFG_DECLINES: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()


def cfg_parallel_decline(
    pipe: Any, *, offload_policy: str, selection: Optional[MultiGpuSelection]
) -> Optional[str]:
    """Why CFG-parallel cannot be armed on this load, or None."""
    if selection is None:
        return "one card selected"
    mode = cfg_parallel_mode()
    if mode == "off":
        return f"{CFG_PARALLEL_ENV}=0"
    denoiser = getattr(pipe, "transformer", None)
    torch = _torch()
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser (a U-Net batches its guidance branches)"
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
    headroom_mib: Optional[int] = None,
    headroom_reference_px: Optional[int] = None,
    decode_headroom_mib: Optional[int] = None,
    encode_headroom_mib: Optional[int] = None,
) -> Optional[str]:
    """Arm CFG-parallel when a secondary card can hold a replica, else CFG-batch for an offloaded denoiser.
    Returns the CFG-parallel decline reason or None. ``headroom_mib``: the batched call's extra activation need.
    ``decode_headroom_mib`` / ``encode_headroom_mib``: the plan's calibrated VAE decode / text encode room, required
    on a replica card that also hosts the VAE / a text encoder."""
    reason = _install_cfg_parallel_only(
        pipe,
        offload_policy = offload_policy,
        logger = logger,
        headroom_mib = headroom_mib,
        headroom_reference_px = headroom_reference_px,
        decode_headroom_mib = decode_headroom_mib,
        encode_headroom_mib = encode_headroom_mib,
    )
    if (
        reason is not None
        and multigpu_mode() != "off"
        and (offload_policy not in (None, "none") or cfg_batch_mode() == "force")
    ):
        try:
            batch_reason = install_cfg_batch(
                pipe,
                offload_policy = offload_policy,
                headroom_mib = headroom_mib,
                headroom_reference_px = headroom_reference_px,
                logger = logger,
            )
        except Exception as exc:  # noqa: BLE001 - an optional lever never fails a load
            batch_reason = f"{type(exc).__name__}: {exc}"
        if batch_reason is not None and logger is not None:
            logger.info("diffusion.multigpu: CFG-batch not armed (%s)", batch_reason)
    return reason


def _install_cfg_parallel_only(
    pipe: Any,
    *,
    offload_policy: str,
    logger: Any = None,
    headroom_mib: Optional[int] = None,
    headroom_reference_px: Optional[int] = None,
    decode_headroom_mib: Optional[int] = None,
    encode_headroom_mib: Optional[int] = None,
) -> Optional[str]:
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
            # only cards that hold the replica AND its denoise activations (unknown free memory kept), then the
            # fast-link tier; the canvas is not known yet, so the calibrated canvas prices it here
            room = max(1024, int(headroom_mib or 0))
            vae = getattr(pipe, "vae", None)
            vae_device = _module_device(vae) if vae is not None else None
            vae_ordinal = (
                vae_device.index if getattr(vae_device, "type", None) == "cuda" else None
            )
            te_ordinals = text_encoder_ordinals(pipe)
            load_needs = PhaseNeeds(
                encode_mib = encode_headroom_mib,
                denoise_mib = room,
                decode_mib = decode_headroom_mib,
                encode_ordinals = te_ordinals,
                decode_ordinal = vae_ordinal,
            )

            def room_on(o: int) -> int:
                # the max over the phases that card runs (they never overlap): denoise, plus the VAE decode / the text
                # encode where it hosts the VAE / an encoder (the replica is kept across requests)
                return load_needs.room_on(o)

            fits = [
                o
                for o in candidates
                if probes[o].free_mib is None or need_mib + room_on(o) <= probes[o].free_mib
            ]
            if not fits:
                most = max(candidates, key = lambda o: probes[o].free_mib or -1)
                reason = (
                    f"replica needs {need_mib} MiB, {probes[most].free_mib} MiB free on cuda:{most}"
                )
            else:
                best = rank_secondaries(primary.index, fits, lambda o: probes[o].free_mib, probes)[
                    0
                ]
                router = CfgParallelRouter(
                    denoiser,
                    primary = primary,
                    secondary = f"cuda:{best}",
                    logger = logger,
                    headroom_mib = headroom_mib,
                    headroom_reference_px = headroom_reference_px,
                    candidates = fits,
                    decode_headroom_mib = decode_headroom_mib,
                    vae_ordinal = vae_ordinal,
                    encode_headroom_mib = encode_headroom_mib,
                    te_ordinals = te_ordinals,
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
    """The denoiser of ``pipe`` was mutated in ways the per-step checks do not see (attention backend, eager patches,
    compile): drop the CFG-parallel replica (rebuilt on the next guided step) and forget every per-load price and
    priced or measured verdict of both CFG routers, so the changed denoiser is priced again. True when a router was
    reset."""
    try:
        router = _CFG_ROUTERS.get(pipe) if pipe is not None else None
        batch = _CFG_BATCH.get(pipe) if pipe is not None else None
    except TypeError:
        return False
    reset = False
    if batch is not None:
        batch.reprice()
        reset = True
    if router is None:
        return reset
    if logger is not None:
        logger.info(
            "diffusion.multigpu: CFG-parallel replica dropped and repriced (%s); rebuilt on the next guided step",
            reason,
        )
    if router.replica is not None:
        router.stats["replica_rebuilds"] += 1
    router.drop_replica()
    router.reprice()
    return True


def _routers_of(pipe: Any) -> list:
    """The CFG routers of ``pipe.transformer``, from the registry kept on the denoiser at install time, so a layer
    installed above the router later (the deferred static step skip) does not hide it (a from_pipe workflow pipe
    shares the denoiser)."""
    denoiser = getattr(pipe, "transformer", None)
    if denoiser is None:
        return []
    router = getattr(denoiser, "__dict__", {}).get("_unsloth_cfg_router")
    if isinstance(router, (CfgParallelRouter, CfgBatchRouter)):
        return [router]
    # not installed by install_cfg_*: walk the forward slot through known outer layers
    fn = getattr(denoiser, "__dict__", {}).get("forward")
    for _ in range(8):
        if isinstance(fn, (CfgParallelRouter, CfgBatchRouter)):
            return [fn]
        fn = getattr(fn, "inner", None) or getattr(fn, "__wrapped__", None)
        if fn is None:
            break
    return []


@contextlib.contextmanager
def cfg_request_scope(
    pipe: Any,
    width: Any = None,
    height: Any = None,
    steps: Any = None,
    needs: Optional[PhaseNeeds] = None,
) -> Iterator[None]:
    """One Studio generate request around ``pipe()``: the routers forget every per-request field on entry and drop
    every retained tensor on exit. Outside this scope the routers infer request boundaries from the call pattern.
    ``steps``: the request's denoise steps when Studio knows them; else the scheduler's timesteps, read when the
    first replica build is decided (the pipeline has set them by then). ``needs``: the request's per-phase memory from
    the generation guard's estimator (:func:`request_phase_needs`); without it the routers scale the load-time figures."""
    routers = _routers_of(pipe)
    scheduler = getattr(pipe, "scheduler", None)
    steps_fn = None
    if scheduler is not None:

        def steps_fn() -> Optional[int]:
            timesteps = getattr(scheduler, "timesteps", None)
            return len(timesteps) if timesteps is not None else None

    for router in routers:
        router.begin_request(width, height, steps, steps_fn, needs)
    try:
        yield
    finally:
        for router in routers:
            router.end_request()


def release_cfg_inputs(pipe: Any) -> None:
    """The denoise is over (the decoder was entered): drop the tensors the routers keep between branches, and evict a
    retained replica whose card runs the decode when the decode would not fit beside it."""
    for router in _routers_of(pipe):
        router.release_inputs()
        ensure = getattr(router, "ensure_phase_room", None)
        if ensure is not None:
            ensure("decode")


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
        # back to the forward the router replaced: an offload wrapper / bound override stays in force
        if router.had_override:
            module.__dict__["forward"] = router.prev_override
        else:
            del module.__dict__["forward"]
        module.__dict__.pop("_unsloth_cfg_router", None)


# CFG-batch: both guidance branches through one forward of an offloaded / streamed denoiser.


def _as_walkable(out: Any) -> Any:
    if isinstance(out, (tuple, list, dict)) or not dataclasses.is_dataclass(out):
        return out
    return {f.name: getattr(out, f.name, None) for f in dataclasses.fields(out)}


# Below this upload share of a plain call the prefetch stream hides the upload and batching gains nothing.
_BATCH_UPLOAD_FRACTION = 0.6
# The first batched calls must beat the two plain calls they replace by this factor, else batching stops.
_BATCH_MEASURED_FRACTION = 0.92
_BATCH_VERIFY_CALLS = 2


def price_cfg_batch(
    *,
    call_s: Optional[float],
    upload_s: Optional[float],
    streamed_bytes: int = 0,
    h2d_gib_s: Optional[float] = None,
) -> dict[str, Any]:
    """Two plain calls vs one batched call: batch when upload-bound or the upload is unmeasured."""
    if call_s is None or upload_s is None:
        chosen = "cfg_batch"
    else:
        chosen = "cfg_batch" if upload_s >= _BATCH_UPLOAD_FRACTION * call_s else "single"
    share = round(upload_s / call_s, 3) if call_s and upload_s is not None else None
    return {
        "strategy": chosen,
        "plain_call_s": None if call_s is None else round(call_s, 4),
        "upload_s": None if upload_s is None else round(upload_s, 4),
        "streamed_mib": streamed_bytes >> 20,
        "h2d_gib_s": h2d_gib_s,
        "upload_share_of_call": share,
        "single_step_s": None if call_s is None else round(2 * call_s, 4),
    }


# Non-tensor leaves both branches may share in one batched call: immutable plain values, dtypes / devices and the
# empty containers _leaves returns as leaves.
def _mutable_container(args: tuple, kwargs: dict) -> tuple[Optional[str], list]:
    """(path of the first list / dict / set inside the call's arguments or None, the empty dicts passed). A batched
    call rebuilds the containers it flattens, so a cache dict or counter list the model mutates would never see the
    per-branch calls; only tuples of tensors and immutable values pass. An EMPTY dict is let through (FLUX.1 passes
    ``joint_attention_kwargs = {}``): the batched call gets a fresh empty stand-in, checked still empty afterwards."""
    empties: list = []

    def walk(value: Any, path: tuple, depth: int) -> Optional[str]:
        if type(value) is dict and not value:
            empties.append(value)
            return None
        if isinstance(value, (list, dict, set, bytearray)):
            return "/".join(str(p) for p in path) or "<arg>"
        if isinstance(value, tuple) and depth < 6:
            for i, v in enumerate(value):
                found = walk(v, path + (i,), depth + 1)
                if found is not None:
                    return found
        return None

    for i, v in enumerate(args):
        found = walk(v, (i,), 0)
        if found is not None:
            return found, empties
    for k, v in kwargs.items():
        found = walk(v, (k,), 0)
        if found is not None:
            return found, empties
    return None, empties


_STATE_SCALARS = (bool, int, float, str)


# Buffers up to this many bytes are copied before every batched call so a change can be undone (FLUX.2 klein-base:
# no buffers at all); larger ones are only version-checked.
_STATE_BUFFER_COPY_BYTES = 64 << 20
# Plain guided calls whose writes are observed before CFG-batch may speculate (both branches of the first step).
_OBSERVE_CALLS = 2


class _StateWatch:
    """What a stateful forward could advance, decided by observing writes rather than guessing from sizes.

    Every parameter / buffer version counter (bumped by every in-place op: ``copy_``, ``index_put_``, ``add_``) is
    read around the first plain guided calls (``_OBSERVE_CALLS``), before any speculation: a buffer whose version
    never moves is read-only and needs no snapshot (Studio's native int8 / fp8 / NVFP4 / ComfyUI quant weights are
    buffers of GBs). Only buffers seen to change are snapshotted (within ``_STATE_BUFFER_COPY_BYTES``); speculation
    is refused when one seen to change does not fit. A tensor without a version counter (an inference tensor) counts
    as read-only when a Studio-defined module owns it (its quant weight buffers), else as written. Around each batched
    call every version is compared again: a move outside the snapshot set is a write this watch did not see.
    Plain scalar attributes of the submodules are watched too (Studio's ``_unsloth_*`` hook state excluded)."""

    def __init__(self, module: Any):
        self.tensors: list = []
        for name, t in module.named_parameters():
            self.tensors.append((name, t, False))
        owners = {}
        for sub_name, sub in module.named_modules():
            for leaf in sub._buffers:
                owners[f"{sub_name}.{leaf}" if sub_name else leaf] = sub
        for name, t in module.named_buffers():
            self.tensors.append((name, t, True))
        self.owners = owners
        self.observed_calls = 0
        self.written: set = set()
        self.copied: set = set()
        self.unrestorable: list = []
        self.slots = [
            (sub, key)
            for sub in module.modules()
            for key, value in sub.__dict__.items()
            if isinstance(value, _STATE_SCALARS) and not key.startswith(_STATE_PREFIX)
        ]
        for name, t, is_buffer in self.tensors:
            if is_buffer and _tensor_version(t) is None and not self._studio_owned(name):
                # no version counter and not a known Studio quant weight: possibly written
                self.written.add(name)
        self._settle()

    def _studio_owned(self, name: str) -> bool:
        owner = self.owners.get(name)
        return str(getattr(type(owner), "__module__", "") or "").startswith(_TRUSTED_HOOK_MODULES)

    def _settle(self) -> None:
        """Snapshot set: the buffers seen (or presumed) written first, smallest first within the budget; what budget is
        left covers the smallest of the others too, so a write the observation missed in a small buffer (a counter
        that starts moving late) is still undone. Large read-only buffers (quant weights) are never copied."""
        sizes = {
            name: t.numel() * t.element_size() for name, t, is_buffer in self.tensors if is_buffer
        }
        self.copied = set()
        spent = 0
        for name in sorted(self.written, key = lambda n: sizes.get(n, 0)):
            if name not in sizes or spent + sizes[name] > _STATE_BUFFER_COPY_BYTES:
                continue
            self.copied.add(name)
            spent += sizes[name]
        for name in sorted((n for n in sizes if n not in self.written), key = lambda n: sizes[n]):
            if spent + sizes[name] > _STATE_BUFFER_COPY_BYTES:
                break
            self.copied.add(name)
            spent += sizes[name]
        self.unrestorable = sorted(n for n in self.written if n in sizes and n not in self.copied)

    @property
    def ready(self) -> bool:
        return self.observed_calls >= _OBSERVE_CALLS

    def versions(self) -> list:
        return [_tensor_version(t) for _name, t, _buf in self.tensors]

    def observe(self, before: list) -> None:
        """After a plain call: record every buffer whose version moved."""
        for (name, t, is_buffer), version in zip(self.tensors, before):
            if not is_buffer:
                continue
            now = _tensor_version(t)
            if version is not None and now is not None and now != version:
                self.written.add(name)
        self.observed_calls += 1
        self._settle()

    def snapshot(self) -> dict:
        versions = self.versions()
        copies = [
            t.detach().clone() if is_buffer and name in self.copied else None
            for name, t, is_buffer in self.tensors
        ]
        values = [sub.__dict__.get(key, _DROP) for sub, key in self.slots]
        return {"versions": versions, "copies": copies, "values": values}

    def restore_changed(self, snap: dict) -> Optional[str]:
        """Name of the first state the call changed (snapshotted buffers and scalars restored), else None."""
        changed, _unseen = self.restore(snap)
        return changed

    def restore(self, snap: dict) -> tuple[Optional[str], Optional[str]]:
        """(first state changed, first write outside the snapshot set): snapshotted buffers and scalars restored; a
        parameter or an unsnapshotted buffer written in place is reported (and remembered as written) but cannot be."""
        torch = _torch()
        changed = unseen = None
        for (name, t, is_buffer), version, copy in zip(
            self.tensors, snap["versions"], snap["copies"]
        ):
            now = _tensor_version(t)
            moved = version is not None and now is not None and now != version
            if copy is not None and version is None:
                moved = not _bitwise_equal(t.detach(), copy)  # an inference tensor: compare the value
            if moved:
                changed = changed or name
                if copy is not None:
                    with torch.no_grad():
                        t.copy_(copy)
                else:
                    unseen = unseen or name
                    if is_buffer:
                        self.written.add(name)
        for (sub, key), value in zip(self.slots, snap["values"]):
            now = sub.__dict__.get(key, _DROP)
            if now is not value and now != value:
                changed = changed or key
                if value is _DROP:
                    sub.__dict__.pop(key, None)
                else:
                    sub.__dict__[key] = value
        if unseen is not None:
            self._settle()
        return changed, unseen


def _tensor_version(tensor: Any) -> Optional[int]:
    try:
        return int(tensor._version)
    except Exception:  # noqa: BLE001 - inference tensors track no version
        return None


def _inputs_untouched(fresh_inputs: list, versions: list) -> bool:
    """True when no fresh input of the batched call was written in place: its version counter unchanged, or, for an
    inference tensor without one, its value still the concatenation (or copy) of the caller's untouched sources."""
    torch = _torch()
    for (tensor, sources), version in zip(fresh_inputs, versions):
        if version is not None:
            if _tensor_version(tensor) != version:
                return False
            continue
        expected = sources[0] if len(sources) == 1 else torch.cat(sources, dim = 0)
        if not _bitwise_equal(tensor, expected):
            return False
    return True


def _batch_shareable(value: Any) -> bool:
    torch = _torch()
    if isinstance(value, (*_PLAIN_TYPES, torch.dtype, torch.device)):
        return True
    return isinstance(value, (tuple, list, dict, frozenset)) and not value


class _NotBatchable(Exception):
    pass


def _batch_leaves(first: list, second: list, batch: int) -> dict:
    """One call carrying both branches: per-sample tensors concatenated on dim 0, equal shared values passed once."""
    torch = _torch()
    if len(first) != len(second):
        raise _NotBatchable("the branches pass different arguments")
    values: dict = {}
    for (path_a, a), (path_b, b) in zip(first, second):
        if path_a != path_b:
            raise _NotBatchable("the branches pass different arguments")
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                raise _NotBatchable(f"{path_a}: tensor in one branch only")
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise _NotBatchable(f"{path_a}: {tuple(a.shape)} vs {tuple(b.shape)}")
            if a.dim() >= 1 and a.shape[0] == batch:
                values[path_a] = torch.cat((a, b), dim = 0)
            elif a is b or _bitwise_equal(a, b):
                values[path_a] = a
            else:
                raise _NotBatchable(f"{path_a}: differs per branch without a batch dim")
        elif not (_batch_shareable(a) and _batch_shareable(b)):
            # a cache / holder object: one batched call would share (and mutate) it for both branches, and the
            # odd branch's check would compare the same object with itself
            raise _NotBatchable(f"{path_a}: opaque {type(a).__name__} argument")
        elif _same_value(a, b):
            values[path_a] = a
        else:
            raise _NotBatchable(f"{path_a}: differs per branch")
    return values


def _split_output(
    out: Any,
    batch: int,
    depth: int = 0,
) -> tuple[Any, Any]:
    """(first, second) branch of a batched output, split on leading dim ``2 * batch``; else :class:`_NotBatchable`."""
    torch = _torch()
    if isinstance(out, torch.Tensor):
        if out.dim() >= 1 and out.shape[0] == 2 * batch:
            return out[:batch], out[batch:]
        raise _NotBatchable(f"output of shape {tuple(out.shape)} has no doubled batch")
    if out is None or isinstance(out, _PLAIN_TYPES):
        return out, out
    if depth > 4:
        raise _NotBatchable("output nests too deep")
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        pairs = {
            f.name: _split_output(getattr(out, f.name), batch, depth + 1)
            for f in dataclasses.fields(out)
        }
        return (
            dataclasses.replace(out, **{k: v[0] for k, v in pairs.items()}),
            dataclasses.replace(out, **{k: v[1] for k, v in pairs.items()}),
        )
    if isinstance(out, tuple) and not hasattr(out, "_fields"):
        pairs = [_split_output(v, batch, depth + 1) for v in out]
        return tuple(p[0] for p in pairs), tuple(p[1] for p in pairs)
    if isinstance(out, list):
        pairs = [_split_output(v, batch, depth + 1) for v in out]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if isinstance(out, dict):
        pairs = {k: _split_output(v, batch, depth + 1) for k, v in out.items()}
        return type(out)((k, v[0]) for k, v in pairs.items()), type(out)(
            (k, v[1]) for k, v in pairs.items()
        )
    raise _NotBatchable(f"output type {type(out).__name__}")


def _registry_hooks_other(module: Any) -> Optional[str]:
    """A diffusers hook registry that is not offload machinery (a step cache) anywhere in ``module``, else None."""
    for name, sub in module.named_modules():
        if _registry_kind(sub) == "other":
            return f"{name or '<root>'}._diffusers_hook ({', '.join(sorted(sub._diffusers_hook.hooks))})"
    return None


class CfgBatchRouter:
    """Offloaded denoiser ``forward`` running both guidance branches of a step through ONE forward (one upload).

    Branches are recognised as in :class:`CfgParallelRouter`; the predicted odd branch is batched with the first and
    its half kept. The real odd call must match the prediction bitwise (snapshots, so in-place edits are seen) with
    unchanged hook state, else it runs plain. Not bit-identical to two calls, so only offloaded denoisers by default.
    Unbatchable steps run as before; OOM on the batched call disables batching and recomputes the branch alone."""

    def __init__(
        self,
        module: Any,
        *,
        headroom_mib: Optional[int] = None,
        headroom_reference_px: Optional[int] = None,
        reason: str = "offloaded denoiser",
        logger: Any = None,
    ):
        self.module_ref = weakref.ref(module)
        self.orig_forward = module.forward
        # the instance forward the router replaced (an offload wrapper), put back on uninstall
        self.had_override = "forward" in module.__dict__
        self.prev_override = module.__dict__.get("forward")
        self.headroom_mib = headroom_mib
        # the canvas (pixels) ``headroom_mib`` was calibrated at; scaled linearly to the request's canvas
        self.request_needs: Optional[PhaseNeeds] = None
        self.headroom_reference_px = headroom_reference_px
        self.request_px: Optional[int] = None
        # every batched call checks the denoiser for state its forward advances (slots collected once per load)
        self.state_watch: Optional[_StateWatch] = None
        self.reason = reason
        self.logger = logger
        self.disabled_reason: Optional[str] = None
        self.prev_latent: Any = None
        self.branch = 0
        self.b0_leaves: dict = {}
        self.template: Optional[tuple[Any, dict]] = None
        self.kept: Optional[dict] = None
        self.last_skip: Optional[str] = None
        self.pricing: Optional[dict[str, Any]] = None
        self.plain_times: list[float] = []
        self.timing_off = False
        self.verify_batched = _BATCH_VERIFY_CALLS
        self.batched_times: list[float] = []
        self.disabled_pending: Optional[str] = None
        # why the CURRENT request shape runs one branch per call (priced out, measured no gain); per shape
        self.shape_disabled: Optional[str] = None
        self.shapes = _ShapeBook(
            {
                "pricing": None,
                "plain_times": [],
                "batched_times": [],
                "verify_batched": _BATCH_VERIFY_CALLS,
                "disabled_pending": None,
                "shape_disabled": None,
                "prediction": _PredictionPolicy,
            }
        )
        self.prediction = _PredictionPolicy()
        self.stats = {
            "batched_steps": 0,
            "batched_hits": 0,
            "batched_misses": 0,
            "unbatched_steps": 0,
            "plain_calls": 0,
        }

    def describe(self) -> dict[str, Any]:
        off = self.disabled_reason or self.shape_disabled
        return {
            "strategy": "cfg_batch" if off is None else "single",
            "pricing": self.pricing,
            "shapes_priced": len(self.shapes.saved) + (self.shapes.key is not None),
            "engaged_for": self.reason,
            "disabled_reason": off,
            "disabled_for_shape_only": self.disabled_reason is None
            and self.shape_disabled is not None,
            "last_unbatched_reason": self.last_skip,
            **self.stats,
        }

    # -- request lifecycle (Studio's generate path; without it the call pattern decides)
    def begin_request(
        self,
        width: Any = None,
        height: Any = None,
        steps: Any = None,
        steps_fn: Optional[Callable[[], Any]] = None,
        needs: Optional[PhaseNeeds] = None,
    ) -> None:
        self.release_inputs()
        self.request_needs = needs
        self.branch = 0
        self.timing_off = False
        if self.pricing is None:
            self.plain_times = []
        try:
            self.request_px = int(width) * int(height) if width and height else None
        except (TypeError, ValueError):
            self.request_px = None

    def end_request(self) -> None:
        self.release_inputs()
        self.request_px = None
        self.request_needs = None

    def release_inputs(self) -> None:
        """Drop every tensor kept between calls; a pending verdict is applied first."""
        self._apply_pending()
        self.kept = None
        self.template = None
        self.b0_leaves = {}
        self.prev_latent = None

    def reprice(self) -> None:
        """The denoiser changed: forget every shape's price and measured verdicts, so each is priced again."""
        self.shapes.clear(self)
        self.timing_off = False
        self.state_watch = None
        if self.disabled_reason is not None and self.disabled_reason.startswith(
            ("priced out", "measured no gain")
        ):
            self.disabled_reason = None

    def disable(self, reason: str) -> None:
        if self.disabled_reason is None:
            self.disabled_reason = reason
            if self.logger is not None:
                self.logger.warning("diffusion.multigpu: CFG-batch disabled (%s)", reason)
        self.kept = None
        self.template = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        latent = _latent_arg(args, kwargs)
        if latent is not None and self.prev_latent is not None and latent is self.prev_latent:
            self.branch += 1
        else:
            if self.branch == 0 and self.prev_latent is not None:
                # The previous step made one call: a single-branch request. Forget the guided state (no speculation,
                # no stale timing); a later guided step rebuilds it from scratch.
                self.template = None
                self.plain_times = []
                if self.pricing is None:
                    self.timing_off = True  # nothing to price: stop syncing
            self.branch = 0
            self.prev_latent = latent
            if self.kept is not None:  # the previous step never asked for its odd branch
                self.kept = None
                self._record_miss("the odd branch never came")
            self._apply_pending()
            self.b0_leaves = dict(_leaves((args, kwargs)))
            if self.shapes.switch(self, _request_shape(self.request_px, args, kwargs)):
                self.template = (
                    None  # the previous shape's odd-branch prediction does not carry over
                )
                if self.pricing is None:
                    self.timing_off = False  # a new shape: its first guided step is timed plain
        if self.disabled_reason is not None or self.shape_disabled is not None or latent is None:
            return self._plain(args, kwargs)
        if self.branch == 1 and self.timing_off:
            # guidance resumed: retake both branch timings on the next step
            self.timing_off = False
            self.plain_times = []
        timing = self.pricing is None and not self.timing_off and self._can_time(latent)
        if self.branch == 0:
            if (
                self.template is not None and not timing
            ):  # unpriced: this step times both plain calls first
                try:
                    out = self._batched(args, kwargs, latent)
                except Exception as exc:  # noqa: BLE001 - anything in the speculative path: plain from here
                    self.disable(f"speculative batch failed ({type(exc).__name__}: {exc})")
                    out = None
                if out is not None:
                    return out
            if timing:
                self.plain_times = []
                return self._timed_plain(args, kwargs)
            return self._plain(args, kwargs)
        if self.branch == 1:
            if self.kept is not None:
                out = self._take(args, kwargs)
                if out is not None:
                    self._record_template(args, kwargs, latent)
                    return out
            self._record_template(args, kwargs, latent)
            if timing and len(self.plain_times) == 1:
                out = self._timed_plain(args, kwargs)
                self._price()
                return out
        return self._plain(args, kwargs)

    def _can_time(self, latent: Any) -> bool:
        return cfg_batch_mode() != "force" and getattr(latent.device, "type", None) == "cuda"

    def _timed_plain(self, args: tuple, kwargs: dict) -> Any:
        torch = _torch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self._plain(args, kwargs)
        torch.cuda.synchronize()
        self.plain_times.append(time.perf_counter() - t0)
        return out

    def _streamed_bytes(self) -> int:
        """Bytes of denoiser weights held off the card between calls: what one call uploads."""
        module = self.module_ref()
        if module is None:
            return 0
        total = 0
        for t in (*module.parameters(), *module.buffers()):
            if t.device.type in ("cpu", "meta"):
                total += t.numel() * t.element_size()
        return total

    def _price(self) -> None:
        call_s = min(self.plain_times) if self.plain_times else None
        streamed = self._streamed_bytes()
        h2d = None
        try:
            torch = _torch()
            ordinal = torch.cuda.current_device()
            h2d = probe_devices((ordinal,)).get(ordinal).h2d_gib_s
        except Exception:  # noqa: BLE001 - unmeasured link: the batched call's own timing decides
            h2d = None
        upload_s = streamed / (h2d * (1 << 30)) if h2d else None
        self.pricing = price_cfg_batch(
            call_s = call_s, upload_s = upload_s, streamed_bytes = streamed, h2d_gib_s = h2d
        )
        if self.logger is not None:
            self.logger.info("diffusion.multigpu: CFG-batch priced %s", self.pricing)
        if self.pricing["strategy"] != "cfg_batch":
            self._disable_shape(
                f"priced out: a call takes {call_s:.4f}s, its upload {upload_s:.4f}s (compute-bound; the streamed "
                "blocks already hide behind compute, so one batched call costs as much as two)"
            )

    def _plain(self, args: tuple, kwargs: dict) -> Any:
        self.stats["plain_calls"] += 1
        watch = self._watch()
        if watch is None or watch.ready:
            return self.orig_forward(*args, **kwargs)
        # before any speculation: which buffers does a plain call write?
        before = watch.versions()
        out = self.orig_forward(*args, **kwargs)
        watch.observe(before)
        return out

    def _watch(self) -> Optional["_StateWatch"]:
        if self.state_watch is None:
            module = self.module_ref()
            if module is None:
                return None
            self.state_watch = _StateWatch(module)
        return self.state_watch

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
                    kinds[path] = ("b0", path)  # recomputed per call with the first branch's value
                else:
                    kinds[path] = ("const", leaf)
            else:
                kinds[path] = ("value", leaf)
        self.template = ((args, kwargs), kinds)

    def _skip(self, reason: str) -> None:
        self.last_skip = reason
        self.stats["unbatched_steps"] += 1

    def _batched(self, args: tuple, kwargs: dict, latent: Any) -> Any:
        torch = _torch()
        module = self.module_ref()
        if module is None:
            return None
        # Only as the outermost forward with no step cache: later layers key per-branch state on separate calls.
        if module.__dict__.get("forward") is not self:
            self._skip("an outer forward layer was installed after this router")
            return None
        outer = _outer_forward_layer(module, self.orig_forward)
        if outer is not None:
            self._skip(f"{outer} under this router")
            return None
        # a step cache, or a foreign hook whose per-call state would see one doubled call: the CFG-parallel screen
        hooked = _hooks_present(
            module, offload_hooks_ok = True, strict_bound = cfg_batch_mode() != "force"
        )
        if hooked is not None:
            self._skip(f"stateful hook present ({hooked})")
            return None
        mutable, _empty = _mutable_container(args, kwargs)
        if mutable is not None:
            self._skip(
                f"{mutable}: mutable container argument (the doubled call would not reach the caller's)"
            )
            return None
        structure, kinds = self.template
        values: dict = {}
        for path, kind in kinds.items():
            if kind[0] == "latent":
                values[path] = latent
            elif kind[0] == "b0":
                if kind[1] not in self.b0_leaves:
                    self._skip("the first branch changed structure")
                    return None
                values[path] = self.b0_leaves[kind[1]]
            elif kind[0] == "const":
                values[path] = kind[1]
            else:
                values[path] = kind[1]
        try:
            predicted = _rebuild(structure, values)
        except KeyError:
            self._skip("the first branch changed structure")
            return None
        first = _leaves((args, kwargs))
        second = _leaves(predicted)
        batch = int(latent.shape[0]) if latent.dim() >= 1 else 0
        if batch < 1:
            self._skip("latent without a batch dim")
            return None
        if not self._memory_ok(latent.device, batch):
            self._skip("not enough free memory for a second branch's activations")
            return None
        try:
            merged = _batch_leaves(first, second, batch)
            # The caller's empty dicts never reach the speculative call: each gets a fresh empty stand-in, so a
            # write during the doubled call cannot dirty the object the plain rerun and the caller will use.
            stand_ins = []
            # Tensors likewise: a concatenation is already fresh storage; a tensor passed through by identity is
            # cloned, so an in-place write by the model never reaches the caller's tensors. Each fresh input is
            # checked after the call (version counter, else its value against the caller's untouched source).
            first_by_path = dict(first)
            fresh_inputs = []
            for path, value in list(merged.items()):
                if type(value) is dict and not value:
                    merged[path] = {}
                    stand_ins.append(merged[path])
                elif isinstance(value, torch.Tensor):
                    source = first_by_path.get(path)
                    if value is source:
                        merged[path] = value.clone()
                        fresh_inputs.append((merged[path], (source,)))
                    else:
                        fresh_inputs.append((value, (source, dict(second)[path])))
            fresh_versions = [_tensor_version(t) for t, _src in fresh_inputs]
            b_args, b_kwargs = _rebuild((args, kwargs), merged)
            snapshots = [
                (path, v.clone() if isinstance(v, torch.Tensor) else v) for path, v in second
            ]
        except _NotBatchable as exc:
            self._skip(str(exc))
            return None
        except Exception as exc:  # noqa: BLE001 - out of memory: this shape; anything else: structural
            merged = b_args = b_kwargs = snapshots = None
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
            if is_out_of_memory(exc):
                # the concatenation of a larger canvas or batch ran out: this shape only, the branch runs plain
                self._disable_shape(
                    "out of memory merging the branches; one branch per call for this shape"
                )
                return None
            self.disable(
                f"merging the branches failed ({type(exc).__name__}: {exc}); one branch per call from here"
            )
            return None
        state = _hook_state([(m, m) for m in module.modules()])
        verify = (
            self.verify_batched > 0 and len(self.plain_times) == 2 and latent.device.type == "cuda"
        )
        watch = self._watch()
        if not watch.ready:
            self._skip("observing which buffers a plain call writes")
            return None
        if watch.unrestorable and cfg_batch_mode() != "force":
            # a buffer the forward writes, too large to snapshot: a speculative call could not be undone
            self._disable_shape(
                f"buffer {watch.unrestorable[0]} is written by the forward and too large to snapshot around a "
                "speculative call; one branch per call for this shape"
            )
            return None
        state_snapshot = watch.snapshot()
        try:
            if verify:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            out = self.orig_forward(*b_args, **b_kwargs)
            if state_snapshot is not None:
                changed, unseen = self._undo_attempt(state_snapshot, report_unseen = True)
                if unseen is not None:
                    # A write the observation did not see, to a buffer with no snapshot: it cannot be undone, and no
                    # rerun is exactly the one-card path (rerunning both branches plain applies it once more; keeping
                    # the doubled outputs applies it once instead of twice). The allowlisted stateless classes write
                    # no buffers in forward, so this is only reachable through a foreign patch or force mode: fail
                    # safe, CFG-batch off for the whole load from this call on, loudly.
                    self.disable(
                        f"the batched call wrote {unseen}, a buffer no plain call was seen writing and too large to "
                        "snapshot; this call's state may differ from one card by that write; one branch per call "
                        "for this load"
                    )
                    return None
                if changed is not None:
                    # its forward advances state once per call: one doubled call is not two (state restored)
                    self.disable(
                        f"the denoiser's forward changes its own state ({changed}); one branch per call"
                    )
                    return None
            if verify:
                torch.cuda.synchronize()
                self._check_batched(time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - the doubled batch was refused: the branch reruns plain
            # a call that raised may have advanced watched state first: undo it before the plain rerun
            self._undo_attempt(state_snapshot)
            if is_out_of_memory(exc):
                del b_args, b_kwargs, merged
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                # a larger canvas or batch can run out where smaller ones fit: this shape only
                self._disable_shape(
                    "out of memory on the batched call; one branch per call for this shape"
                )
                return None
            self.disable(
                f"the batched call failed ({type(exc).__name__}: {exc}); one branch per call from here"
            )
            return None
        if not _inputs_untouched(fresh_inputs, fresh_versions):
            # the model writes its inputs in place: one doubled call is not two calls (the caller's are untouched)
            self._undo_attempt(state_snapshot)
            self.disable(
                "the batched call wrote into an input tensor in place; one branch per call from here"
            )
            return None
        if any(stand_ins):
            # the model writes into that dict: one doubled call is not two calls (the caller's dict is untouched)
            self._undo_attempt(state_snapshot)
            self.disable(
                "the batched call wrote into an argument dict; one branch per call from here"
            )
            return None
        try:
            mine, other = _split_output(out, batch)
        except Exception as exc:  # noqa: BLE001 - incl. a container that cannot be rebuilt: the branch reruns plain
            self._undo_attempt(state_snapshot)
            self.disable(f"output cannot be split per branch ({type(exc).__name__}: {exc})")
            return None
        self.kept = {"out": other, "snapshots": snapshots, "state": state}
        self.stats["batched_steps"] += 1
        return mine

    def _undo_attempt(self, snapshot: Optional[dict], report_unseen: bool = False) -> Any:
        """The one restore every exit out of a batched call goes through (state change, OOM, exception, an input or
        dict written, an unsplittable output): snapshotted buffers and scalar slots back to the snapshot, before the
        branch reruns plain. Returns what had changed (and, with ``report_unseen``, the first write outside the
        snapshot set), else None."""
        if snapshot is None or self.state_watch is None:
            return (None, None) if report_unseen else None
        try:
            changed, unseen = self.state_watch.restore(snapshot)
        except Exception:  # noqa: BLE001 - restoring is best effort on an already failing path
            changed = unseen = None
        return (changed, unseen) if report_unseen else changed

    def _check_batched(self, batched_s: float) -> None:
        """No gain from the fastest early batched call vs two plain calls -> stop batching after this step."""
        self.verify_batched -= 1
        self.batched_times.append(batched_s)
        if self.verify_batched > 0:
            return
        batched_s = min(self.batched_times)
        plain = sum(self.plain_times)
        if self.pricing is not None:
            self.pricing["batched_call_s"] = round(batched_s, 4)
            self.pricing["two_plain_calls_s"] = round(plain, 4)
        if batched_s > _BATCH_MEASURED_FRACTION * plain:
            self.disabled_pending = f"measured no gain: one batched call {batched_s:.4f}s vs two plain calls {plain:.4f}s"

    def _memory_ok(
        self,
        device: Any,
        batch: Optional[int] = None,
    ) -> bool:
        if self.headroom_mib is None or getattr(device, "type", None) != "cuda":
            return True
        torch = _torch()
        try:
            free, _total = _free_total_mib(
                device.index if device.index is not None else torch.cuda.current_device()
            )
            reserved_idle = (
                torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            ) // (1 << 20)
        except Exception:  # noqa: BLE001 - unreadable: let the OOM backstop decide
            return True
        if free is None:
            return True
        return free + int(reserved_idle) >= self.needed_mib(batch)

    def needed_mib(self, batch: Optional[int] = None) -> int:
        """The headroom the batched call needs for the second branch of ``batch`` samples (the latent's leading dim):
        the request's denoise figure from the generation guard's estimator (canvas, batch, frames and reference tokens
        included) when the generate path supplied it; else the load-time calibrated figure scaled by the request's
        canvas over the calibration canvas and by the batch, linearly."""
        needs = self.request_needs
        if needs is not None and needs.denoise_mib:
            return int(needs.denoise_mib)
        return scaled_headroom_mib(
            self.headroom_mib, self.headroom_reference_px, self.request_px, batch
        )

    def _take(self, args: tuple, kwargs: dict) -> Any:
        kept, self.kept = self.kept, None
        actual = _leaves((args, kwargs))
        snaps = kept["snapshots"]
        ok = len(actual) == len(snaps) and all(
            pa == ps
            and (_bitwise_equal(va, vs) if isinstance(vs, _torch().Tensor) else _same_value(va, vs))
            for (pa, va), (ps, vs) in zip(actual, snaps)
        )
        module = self.module_ref()
        if ok and module is not None:
            ok = _hook_state([(m, m) for m in module.modules()]) == kept["state"]
        if not ok:
            self._record_miss("the odd branch differed from the batched prediction")
            self._apply_pending()
            return None
        self.stats["batched_hits"] += 1
        self.prediction.hit()
        self._apply_pending()
        return kept["out"]

    def _record_miss(self, why: str) -> None:
        self.stats["batched_misses"] += 1
        if self.prediction.miss():
            self._disable_shape(
                f"the speculative doubled call missed {self.prediction.consecutive} consecutive steps ({why}); one "
                "branch per call for this shape"
            )

    def _apply_pending(self) -> None:
        if self.disabled_pending is not None:
            reason, self.disabled_pending = self.disabled_pending, None
            self._disable_shape(reason)

    def _disable_shape(self, reason: str) -> None:
        """A priced / measured verdict: one branch per call for THIS request shape only."""
        if self.shape_disabled is None:
            self.shape_disabled = reason
            if self.logger is not None:
                self.logger.info(
                    "diffusion.multigpu: CFG-batch off for this request shape (%s)", reason
                )
        self.kept = None
        self.template = None


_CFG_BATCH: "weakref.WeakKeyDictionary[Any, CfgBatchRouter]" = weakref.WeakKeyDictionary()


def _outer_forward_layer(module: Any, forward: Any = _DROP) -> Optional[str]:
    """Default-deny for the forward chain CFG-batch calls, peeled layer by layer through offload wrappers (accelerate
    and diffusers wrap with ``functools.update_wrapper``, so ``__wrapped__`` names the layer below; accelerate also
    keeps ``_old_forward``). Every layer must be offload machinery or a Studio forward patch, and the chain must end at
    the class forward bound to ``module`` or a Studio patch. Studio's outer layers (``_unsloth_outer_forward``: the
    static step skip) anywhere in the chain, an offload wrapper that hides what it wraps, and anything foreign are
    refused: its name, else None."""
    if forward is _DROP:
        forward = module.__dict__.get("forward")
    if forward is None:
        return None
    class_forward = getattr(type(module).forward, "__func__", type(module).forward)
    fn = forward
    for _ in range(12):
        if isinstance(fn, CfgBatchRouter):
            fn = fn.orig_forward
            continue
        if getattr(fn, "_unsloth_outer_forward", False):
            return f"outer forward layer {type(fn).__name__}"
        if (
            isinstance(fn, types.MethodType)
            and fn.__self__ is module
            and fn.__func__ is class_forward
        ):
            return None
        if fn is class_forward:
            return None
        if _is_offload_object(fn):
            below = getattr(fn, "__wrapped__", None)
            if below is None and getattr(module, "_hf_hook", None) is not None:
                below = getattr(module, "_old_forward", None)
            if below is None:
                return f"offload wrapper {type(fn).__name__} hides what it wraps"
            fn = below
            continue
        if _hook_origin(fn).startswith(_TRUSTED_HOOK_MODULES):
            below = getattr(fn, "__wrapped__", None)
            if below is None:
                return None  # a Studio forward patch at the bottom of the chain
            fn = below
            continue
        return f"outer forward layer {type(fn).__name__}"
    return "forward chain too deep to inspect"


# Denoiser classes whose forward keeps no state across calls, so one doubled call equals two per-branch calls. Read
# from the diffusers source (every method outside __init__ scanned for attribute writes and in-place tensor ops):
# FLUX.2 / klein writes only into a Flux2KVCache passed as an argument (refused as an opaque argument); Z-Image's
# RopeEmbedder fills its frequency table once, a pure function of the config; Qwen-Image's RoPE tables are pure
# lru caches; FLUX.1 and Wan write nothing (Wan's fuse_projections is never called by forward). A fully general
# detector is not possible (a forward may keep arbitrary Python state), so other classes run one branch per call
# unless UNSLOTH_DIFFUSION_CFG_BATCH=force opts them in.
_CFG_BATCH_STATELESS_CLASSES = frozenset(
    {
        "Flux2Transformer2DModel",
        "FluxTransformer2DModel",
        "ZImageTransformer2DModel",
        "QwenImageTransformer2DModel",
        "WanTransformer3DModel",
    }
)


def _denoiser_class(module: Any) -> type:
    """The real transformer class behind compile / PEFT wrappers (``_orig_mod``, ``base_model.model``)."""
    for _ in range(4):
        inner = (
            module.__dict__.get("_modules", {}).get("_orig_mod")
            if hasattr(module, "__dict__")
            else None
        )
        if inner is None:
            base = getattr(module, "base_model", None)
            inner = (
                getattr(base, "model", None) if base is not None and base is not module else None
            )
        if inner is None or inner is module:
            break
        module = inner
    return type(module)


def cfg_batch_class_listed(module: Any) -> tuple[bool, str]:
    """(whether the denoiser's class is on the stateless-forward list, its qualified name)."""
    cls = _denoiser_class(module)
    name = f"{cls.__module__}.{cls.__qualname__}"
    return cls.__module__.startswith(
        "diffusers."
    ) and cls.__qualname__ in _CFG_BATCH_STATELESS_CLASSES, name


def cfg_batch_decline(pipe: Any, *, offload_policy: str) -> Optional[str]:
    """Why both guidance branches cannot share one forward on this load, or None."""
    mode = cfg_batch_mode()
    if mode == "off":
        return f"{CFG_BATCH_ENV}=0"
    torch = _torch()
    denoiser = getattr(pipe, "transformer", None)
    if not isinstance(denoiser, torch.nn.Module):
        return "no DiT denoiser (a U-Net batches its guidance branches)"
    if getattr(pipe, "transformer_2", None) is not None:
        return "two-expert denoiser"
    if getattr(denoiser, "_compiled_call_impl", None) is not None:
        return "denoiser is compiled whole"
    if isinstance(denoiser.__dict__.get("forward"), (CfgParallelRouter, CfgBatchRouter)):
        return "a CFG router is already installed"
    outer = _outer_forward_layer(denoiser)
    if outer is not None:
        return f"{outer} (it would see one doubled call per step)"
    hooked = _registry_hooks_other(denoiser)
    if hooked is not None:
        return f"step cache present ({hooked})"
    if mode == "force":
        return None
    listed, cls_name = cfg_batch_class_listed(denoiser)
    if not listed:
        return (
            f"denoiser class {cls_name} is not on the stateless-forward list "
            f"({CFG_BATCH_ENV}=force opts it in)"
        )
    streamed = offload_policy in ("group", "streaming", "sequential") and (
        any(_registry_kind(sub) == "offload" for sub in denoiser.modules())
        or any(getattr(sub, "_hf_hook", None) is not None for sub in denoiser.modules())
        or any(
            _is_offload_object(sub.__dict__.get("forward"))
            for sub in denoiser.modules()
            if "forward" in sub.__dict__
        )
    )
    if not streamed:
        return "denoiser is resident (two calls are exact and the weights are not re-uploaded)"
    return None


def install_cfg_batch(
    pipe: Any,
    *,
    offload_policy: str,
    headroom_mib: Optional[int] = None,
    headroom_reference_px: Optional[int] = None,
    logger: Any = None,
) -> Optional[str]:
    """Arm :class:`CfgBatchRouter` on ``pipe.transformer``; returns the decline reason or None when armed."""
    reason = cfg_batch_decline(pipe, offload_policy = offload_policy)
    if reason is not None:
        return reason
    denoiser = pipe.transformer
    router = CfgBatchRouter(
        denoiser,
        headroom_mib = headroom_mib,
        headroom_reference_px = headroom_reference_px,
        reason = "forced"
        if cfg_batch_mode() == "force"
        else f"offloaded denoiser ({offload_policy})",
        logger = logger,
    )
    try:
        _CFG_BATCH[pipe] = router
    except TypeError:
        return "pipeline cannot be tracked (not weak-referenceable)"
    denoiser.forward = router
    denoiser.__dict__["_unsloth_cfg_router"] = router
    if logger is not None:
        logger.info("diffusion.multigpu: CFG-batch armed (%s)", router.reason)
    return None


def uninstall_cfg_batch(pipe: Any) -> None:
    try:
        router = _CFG_BATCH.pop(pipe, None) if pipe is not None else None
    except TypeError:
        return
    if router is None:
        return
    module = router.module_ref()
    if module is not None and module.__dict__.get("_unsloth_cfg_router") is router:
        del module.__dict__["_unsloth_cfg_router"]
    if module is not None and module.__dict__.get("forward") is router:
        # back to the forward the router replaced: an offload wrapper / bound override stays in force
        if router.had_override:
            module.__dict__["forward"] = router.prev_override
        else:
            del module.__dict__["forward"]
