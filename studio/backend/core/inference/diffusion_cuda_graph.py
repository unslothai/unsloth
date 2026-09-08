# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static-buffer CUDA-graph capture of ONE denoiser forward, for the diffusion backends.

WHAT IT WRAPS
-------------
Studio never calls the DiT itself: a diffusers pipeline calls ``transformer(...)`` from inside
``pipe(**kwargs)``, once (or twice, under CFG or a dual-expert family) per denoising step, and
every kernel of that step is launched from Python. For a host-bound render that dispatch cost is
the render. A CUDA graph records the whole forward once per input shape and replays it with a
single launch, so the Python cost of a step collapses to one ``copy_`` per input plus a replay.

The install point is therefore the module boundary: each module from
``diffusion_speed._denoiser_dits(pipe)`` gets a ``GraphedForward`` written into its INSTANCE
``__dict__`` under the name ``forward``.

WHY THE FORWARD-SLOT SWAP AND NOT A HOOK OR A PROXY
---------------------------------------------------
* ``nn.Module.__call__`` resolves ``self.forward`` through the instance ``__dict__`` before the
  class attribute, so writing the slot intercepts the pipeline's call without touching the class
  (which every other load of the same family shares).
* Not a proxy object: the pipelines read ``.dtype``, ``.config`` and ``.cache_context`` straight
  off the transformer, so anything that is not the module itself breaks them.
* Not a forward hook: hooks move ``nn.Module._call_impl`` off its fast path, an uncontrolled
  difference between a graphed and an ungraphed load, and offload already owns the hook list.
* ``uninstall()`` pops the slot, so an ungraphed load's call path is byte-identical to a process
  in which this module was never imported, rather than the original reached through one extra
  Python frame.
* The one thing a slot swap DOES bypass is ``nn.Module.compile``, which installs
  ``_compiled_call_impl`` and is what a U-Net denoiser (SDXL) gets from the speed layer: replacing
  ``forward`` there would silently throw the compile away. So a pipe with a whole-compiled U-Net
  is refused outright (``graph_eligible``). Regional compile (``compile_repeated_blocks``) mutates
  the BLOCKS, not the module forward, so ``type(m).forward.__get__(m)`` stays the correct eager
  callable for a DiT and the two levers compose.

WHAT IT REFUSES TO CAPTURE, AND WHY
-----------------------------------
* Any input tensor that is not on CUDA. A host tensor read inside a captured region is baked into
  the graph at its RECORDED VALUE, which is the single most likely way to turn this into a fast
  wrong answer, so it is a hard refusal that poisons the wrapper rather than a silent constant.
* A Python float anywhere in the call tree (SDXL hands the U-Net a float timestep). A float is
  part of the cache key, so a float that changes per step means one graph per step: the cap would
  be hit on step 5 and the capture cost paid for nothing.
* ``return_dict`` anything but an explicit ``False``: the output would be a dataclass built at
  capture time and handed back on every replay.
* Anything in the call tree that is not a tensor / list / tuple / dict / scalar / None, keyed as
  ``("o", typename, id)`` so it can never collide with a capturable key.
* A live step cache (FBCache / MagCache). The cond and uncond forwards of one step share a key but
  carry different residual state, so a replay would serve a stale residual. The caller bypasses
  per chunk and the wrapper also checks the ``_unsloth_step_cache`` marker on the module itself.

MEASURED
--------
Prototype (``scripts/g833``, shared pool from ``scripts/g840``) on z-image at 512px: bf16 1.29x,
torchao fp8 1.72x, int8 1.49x, all bit-identical against the ungraphed run in every repeat. At
1024px the render is device-bound and the same capture is 1.02-1.04x, which is the expected shape
of the result: this removes host dispatch, not GPU work. The cost is the pool, roughly one step's
activations, reserved and not allocated.

KILL SWITCH
-----------
``UNSLOTH_DISABLE_CUDA_GRAPH=1`` (also ``true`` / ``yes`` / ``on``, case-insensitive) refuses
eligibility before anything else is inspected, which is also how an A/B against the ungraphed path
is run. Torch is imported lazily inside functions so this module imports on a torch-free host,
matching ``diffusion_speed``.
"""

from __future__ import annotations

import gc
import inspect
import os
import traceback
import weakref
from functools import update_wrapper
from typing import Any, Optional

CUDA_GRAPH_DISABLE_ENV = "UNSLOTH_DISABLE_CUDA_GRAPH"

# Per module, not per process. Legitimate second keys exist (a CFG batch flip mid-loop, an
# OOM-halved chunk, H3's step-0 timestep being one element longer than every later step), and each
# costs a pool-sized slice of VRAM, so the cap is small and blowing through it degrades to eager
# rather than to a capture storm.
MAX_GRAPHS_PER_MODULE = 4

# Three is what the PyTorch docs prescribe. It also gives everything lazily created inside the
# region (cuBLAS handles, autotune buffers, flashinfer workspaces, allocator blocks) a chance to be
# created OUTSIDE the capture, which is what makes the capture legal.
WARMUP_ITERS = 3

_TRUE_TOKENS = ("1", "true", "yes", "on")


def _torch():
    """The torch module, imported lazily so this file imports on a torch-free host.

    Everything below goes through this rather than a module-level import: ``diffusion_speed``
    imports on a machine with no torch at all, this module is imported beside it, and the test
    suite stubs ``sys.modules["torch"]`` to exercise the tree walk without a GPU."""
    import torch
    return torch


def cuda_graph_disabled() -> bool:
    """Whether the env kill switch is set. Read first, before any torch or pipe inspection."""
    return os.environ.get(CUDA_GRAPH_DISABLE_ENV, "").strip().lower() in _TRUE_TOKENS


# -- the call tree --
# Small, explicit, and deliberately NOT ``torch.utils._pytree``: pytree treats an unregistered
# object as a LEAF, which is precisely the case that must stay visible here (an unknown object in
# the tree cannot be given a static buffer and must not be keyed as if it could).


def _flatten(obj: Any, out: list) -> tuple:
    """Append every tensor in ``obj`` to ``out``; return a spec that rebuilds ``obj``.

    Raises TypeError on anything that is neither a tensor nor one of the four handled kinds, so a
    structure this layer does not understand can never be turned into a static buffer."""
    if _torch().is_tensor(obj):
        out.append(obj)
        return ("t", len(out) - 1)
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), [_flatten(o, out) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, _flatten(v, out)) for k, v in obj.items()])
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    raise TypeError(f"cannot make a static buffer for a {type(obj).__name__} in the call tree")


def _rebuild(spec: tuple, tensors: list) -> Any:
    """Inverse of ``_flatten``: the same structure with ``tensors`` in the tensor positions."""
    kind = spec[0]
    if kind == "t":
        return tensors[spec[1]]
    if kind == "l":
        values = [_rebuild(s, tensors) for s in spec[2]]
        return tuple(values) if spec[1] else values
    if kind == "d":
        return {k: _rebuild(s, tensors) for k, s in spec[1]}
    return spec[1]


def graph_key(obj: Any) -> tuple:
    """Hashable identity of everything that must match for a recorded graph to be replayable.

    Shape AND stride AND dtype AND device, because a graph replays into the exact buffers it was
    recorded against: a differently strided tensor of the same shape is a different graph, not the
    same one. Scalars are keyed by VALUE (they were constants at capture time). Anything else keys
    as ``("o", typename, id)``, which ``_uncapturable`` then refuses; the id keeps it from
    colliding with a capturable key rather than pretending the object is stable."""
    if _torch().is_tensor(obj):
        return (
            "t",
            tuple(obj.shape),
            tuple(obj.stride()),
            obj.dtype,
            obj.device.type,
            obj.device.index,
        )
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), tuple(graph_key(o) for o in obj))
    if isinstance(obj, dict):
        return ("d", tuple((k, graph_key(v)) for k, v in obj.items()))
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    return ("o", type(obj).__name__, id(obj))


def _uncapturable(key: tuple) -> bool:
    """True when the key holds anything this layer cannot give a static buffer to."""
    kind = key[0]
    if kind == "o":
        return True
    if kind == "l":
        return any(_uncapturable(k) for k in key[2])
    if kind == "d":
        return any(_uncapturable(k) for _, k in key[1])
    return False


def _has_float(key: tuple) -> bool:
    """True when the key holds a Python float leaf.

    A float is part of the key, so a per-step float (SDXL's timestep) means one graph per step:
    refusing is cheaper than capturing four graphs and then hitting the cap forever. ``bool`` is a
    subclass of ``int``, not of ``float``, so a ``return_dict`` style flag is not caught here."""
    kind = key[0]
    if kind == "v":
        return isinstance(key[1], float)
    if kind == "l":
        return any(_has_float(k) for k in key[2])
    if kind == "d":
        return any(_has_float(k) for _, k in key[1])
    return False


# -- the shared memory pool --
# ``torch.cuda.graph(g)`` with no pool gives every graph a PRIVATE pool holding the whole captured
# region's intermediates. One dual-DiT family (two 14B experts) already fails to capture its second
# expert that way. The graphs never replay concurrently, so one pool for every graph in the process
# is safe and is what torch's own ``make_graphed_callables`` does. A one-element box, so the first
# capture's pool id is visible to every later capture including other modules'.
_POOL_BOX: list = [None]

# Every wrapper ever built, weakly. The pool id is only meaningful while some graph recorded into
# it is alive; once the last wrapper has dropped its graphs the id must not be handed to a capture
# in a later load, whose pool would then be a dangling reference.
_LIVE_WRAPPERS: "weakref.WeakSet" = weakref.WeakSet()


def _drop_pool_if_unused() -> None:
    """Forget the shared pool id once no live wrapper holds a captured graph. Never raises."""
    try:
        for wrapper in tuple(_LIVE_WRAPPERS):
            if wrapper.cache:
                return
        _POOL_BOX[0] = None
    except Exception:  # noqa: BLE001 - teardown helper, never the reason a load fails
        pass


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.cuda_graph: %s failed: %s", what, exc)


class _Entry:
    """One captured graph plus the static buffers it replays into."""

    __slots__ = ("graph", "static", "in_spec", "out_spec", "out_tensors")


class GraphedForward:
    """A callable that replaces one denoiser module's ``forward`` with a CUDA-graph replay.

    Family- and arm-agnostic by construction: it walks the ``(args, kwargs)`` tree it is handed
    rather than naming positions, so a family that calls its DiT with eight keyword tensors and a
    dict, one that passes a list of latents, and one whose timestep changes LENGTH between step 0
    and step 1 all go through the identical object. One graph per key; a key that has never been
    seen falls through to a capture; a capture that raises poisons the wrapper so the load runs
    eager for the rest of its life AND says so, rather than half-graphing itself."""

    def __init__(
        self,
        module: Any,
        *,
        warmup: int = WARMUP_ITERS,
        max_graphs: int = MAX_GRAPHS_PER_MODULE,
        logger: Any = None,
    ) -> None:
        self.module = module
        # The CLASS forward bound to this instance: the eager callable, unaffected by whatever is
        # sitting in the instance slot (including a previous wrapper).
        self.orig = type(module).forward.__get__(module)
        try:
            # ``__name__`` / ``__doc__`` / ``__wrapped__``, so anything that introspects the
            # denoiser's forward sees the forward and not this class.
            update_wrapper(self, self.orig)
        except Exception:  # noqa: BLE001 - cosmetic only
            pass
        try:
            # LOAD-BEARING, not cosmetic: MiniMax-H3's modular pipeline filters the kwargs it
            # passes by ``inspect.signature(transformer.forward).parameters``, so a wrapper with a
            # bare ``(*args, **kwargs)`` signature loses token_tags / position_ids / the index
            # tensors and the denoise step crashes.
            self.__signature__ = inspect.signature(self.orig)
        except Exception:  # noqa: BLE001 - a signature-less forward still graphs fine
            pass
        self.warmup = int(warmup)
        self.max_graphs = int(max_graphs)
        self.logger = logger
        self.enabled = False
        self.bypassed = False
        self.poisoned = False
        self.capture_error: Optional[dict] = None
        self.cache: dict = {}
        self.cap_hit = False
        self.stats = {
            "captures": 0,
            "replays": 0,
            "eager_calls": 0,
            "fallbacks": 0,
            "refused_float": 0,
            "refused_host_tensor": 0,
            "refused_object": 0,
            "cap_skips": 0,
        }
        _LIVE_WRAPPERS.add(self)

    # -- lifecycle --
    def install(self) -> "GraphedForward":
        """Write the instance ``forward`` slot. Straight into ``__dict__``: ``nn.Module``'s
        ``__setattr__`` inspects what it is given, and this is not a parameter, buffer or
        submodule."""
        self.module.__dict__["forward"] = self
        return self

    def uninstall(self) -> "GraphedForward":
        self.module.__dict__.pop("forward", None)
        return self

    def enable(self) -> "GraphedForward":
        self.install()
        self.enabled = True
        return self

    def disable(self) -> "GraphedForward":
        """Disarm AND uninstall, so a disabled load's call path is byte-identical to one that
        never built a wrapper."""
        self.enabled = False
        self.uninstall()
        return self

    def set_bypass(self, on: bool) -> "GraphedForward":
        """Run eager without dropping the captured graphs (a chunk under a step cache)."""
        self.bypassed = bool(on)
        return self

    def reset(self) -> "GraphedForward":
        """Drop every captured graph. Used when the weights change under us (LoRA load / unload /
        adapter switch), where a replay would keep serving the old weights forever."""
        self.cache.clear()
        try:
            gc.collect()
            cuda = getattr(_torch(), "cuda", None)
            if cuda is not None and hasattr(cuda, "empty_cache"):
                cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass
        return self

    def poison(self, exc: BaseException) -> "GraphedForward":
        """Record why capture failed and run eager for the rest of this wrapper's life."""
        self.capture_error = {
            "type": type(exc).__name__,
            "msg": str(exc)[:4000],
            "traceback": traceback.format_exc()[-6000:],
        }
        self.poisoned = True
        return self

    def free(self) -> "GraphedForward":
        """Drop the graphs and the slot. After the last wrapper is freed the shared pool id is
        forgotten by ``_drop_pool_if_unused``."""
        self.reset()
        self.uninstall()
        self.enabled = False
        _LIVE_WRAPPERS.discard(self)
        return self

    def describe(self) -> dict:
        """A JSON-safe summary for the status payload and the debug log. The capture traceback is
        deliberately not in here; it stays on ``capture_error``."""
        error = self.capture_error
        return {
            "module": type(self.module).__name__,
            "enabled": bool(self.enabled),
            "bypassed": bool(self.bypassed),
            "poisoned": bool(self.poisoned),
            "graphs": len(self.cache),
            "warmup": int(self.warmup),
            "max_graphs": int(self.max_graphs),
            "cap_hit": bool(self.cap_hit),
            "stats": dict(self.stats),
            "capture_error": None
            if not error
            else {
                "type": str(error.get("type")),
                "msg": str(error.get("msg")),
            },
        }

    # -- the call --
    def _eager(self, args: tuple, kwargs: dict) -> Any:
        self.stats["eager_calls"] += 1
        return self.orig(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if (
            not self.enabled
            or self.bypassed
            or self.poisoned
            # Belt to the caller's per-chunk bypass: FBCache / MagCache keep residual state that a
            # replay cannot see, and the cond and uncond forwards of one step share a key.
            or getattr(self.module, "_unsloth_step_cache", None)
        ):
            return self._eager(args, kwargs)

        # An explicit False is the only capturable value: True (or absent) means the forward builds
        # an output dataclass, which a replay would hand back frozen at its capture-time contents.
        if kwargs.get("return_dict", True) is not False:
            return self._eager(args, kwargs)

        try:
            key = graph_key((args, kwargs))
            entry = self.cache.get(key)
        except Exception:  # noqa: BLE001 - an unhashable tree is simply not capturable
            self.stats["refused_object"] += 1
            return self._eager(args, kwargs)

        if entry is None:
            if _uncapturable(key):
                self.stats["refused_object"] += 1
                return self._eager(args, kwargs)
            if _has_float(key):
                self.stats["refused_float"] += 1
                return self._eager(args, kwargs)
            if len(self.cache) >= self.max_graphs:
                # The graphs already recorded keep replaying: a new key here is a legitimate but
                # rare event (an OOM-halved chunk, a CFG batch flip), not a reason to give up the
                # ones that are paying for themselves.
                self.stats["cap_skips"] += 1
                if not self.cap_hit:
                    self.cap_hit = True
                    if self.logger is not None:
                        self.logger.warning(
                            "diffusion.cuda_graph: graph cap of %d reached on %s; further input "
                            "shapes run eager",
                            self.max_graphs,
                            type(self.module).__name__,
                        )
                return self._eager(args, kwargs)
            try:
                entry = self._capture(args, kwargs)
            except Exception as exc:  # noqa: BLE001 - a failed capture must never fail the render
                self.poison(exc)
                self.stats["fallbacks"] += 1
                if self.logger is not None:
                    # Type and message only; the traceback stays on ``capture_error``.
                    self.logger.warning(
                        "diffusion.cuda_graph: capture failed on %s (%s: %s); this load runs eager",
                        type(self.module).__name__,
                        type(exc).__name__,
                        exc,
                    )
                return self._eager(args, kwargs)
            self.cache[key] = entry
            if self.logger is not None:
                self.logger.debug(
                    "diffusion.cuda_graph: captured %s graph %d/%d over %d input tensor(s)",
                    type(self.module).__name__,
                    len(self.cache),
                    self.max_graphs,
                    len(entry.static),
                )

        live: list = []
        _flatten((args, kwargs), live)
        for dst, src in zip(entry.static, live):
            dst.copy_(src)
        entry.graph.replay()
        self.stats["replays"] += 1
        # Cloned out of the graph's pool: the replay writes the SAME buffers every time and the
        # pipeline keeps the previous step's tensor alive across the step boundary.
        return _rebuild(entry.out_spec, [t.clone() for t in entry.out_tensors])

    # -- capture --
    def _capture(self, args: tuple, kwargs: dict) -> _Entry:
        torch = _torch()
        entry = _Entry()
        live: list = []
        entry.in_spec = _flatten((args, kwargs), live)

        bad = [(i, str(t.device)) for i, t in enumerate(live) if t.device.type != "cuda"]
        if bad:
            # Hard refusal, and it poisons: a host tensor read inside a captured region is baked
            # into the graph at its recorded value, so replaying would be fast and wrong.
            self.stats["refused_host_tensor"] += 1
            raise RuntimeError(
                f"{len(bad)} input tensor(s) are not on cuda ({bad[:4]}); a host tensor read "
                f"inside a captured region is baked in at its recorded value"
            )

        # Static buffers must outlive the capture and be writable. Renders run inside
        # ``torch.inference_mode()`` and a tensor CREATED there is an inference tensor, which
        # ``copy_`` from normal mode refuses. Build them with inference mode explicitly off.
        with torch.inference_mode(False):
            entry.static = [torch.empty_like(t) for t in live]
        for dst, src in zip(entry.static, live):
            dst.copy_(src)
        static_args, static_kwargs = _rebuild(entry.in_spec, entry.static)

        # Warm up on a SIDE stream, with a wait in both directions and a full synchronize after.
        # Anything lazily allocated in here lands in the ordinary allocator rather than in the
        # graph's pool, which is what makes the capture legal: a workspace first created DURING
        # capture is only valid while recording, and the first replay reads garbage from it.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.orig(*static_args, **static_kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # ``pool = None`` is identical to omitting the argument on every torch this ships against,
        # so the first capture and every later one take one code path.
        with torch.cuda.graph(graph, pool = _POOL_BOX[0]):
            out = self.orig(*static_args, **static_kwargs)
        if _POOL_BOX[0] is None:
            try:
                _POOL_BOX[0] = graph.pool()
            except Exception:  # noqa: BLE001 - a private pool per graph still works
                pass

        entry.graph = graph
        out_tensors: list = []
        entry.out_spec = _flatten(out, out_tensors)
        entry.out_tensors = out_tensors
        self.stats["captures"] += 1
        return entry


# -- module API --
def graph_eligible(
    target: Any,
    *,
    family: Any,
    pipe: Any,
    offload_active: bool,
    cache_active: bool,
    speed_mode: str,
    family_default: bool = True,
    logger: Any = None,
) -> tuple[bool, str]:
    """Whether this load may be graphed, and a short reason when it may not.

    The reason is stashed on the pipe by the caller and surfaced in the resolved record, so a load
    that quietly did not get graphs says which rule refused it. Ordered cheapest first, and the
    kill switch is first of all so an A/B never has to reason about the rest."""
    if cuda_graph_disabled():
        return False, f"disabled by {CUDA_GRAPH_DISABLE_ENV}"

    device = getattr(target, "device", None)
    if device != "cuda":
        return False, f"device is {device or 'unknown'}"

    # ROCm reports device "cuda" and has no usable graph capture here; XPU / MPS / CPU never do.
    backend = getattr(target, "backend", "cuda")
    if backend != "cuda":
        return False, f"backend is {backend}"

    if offload_active:
        # Onload hooks move weights between the host and the card between steps, so the pointers
        # baked into a recorded graph stop being the weights the next step wants.
        return False, "offload active"

    if cache_active:
        # FBCache / MagCache residual state is invisible to a replay; see the marker check in
        # ``__call__``.
        return False, "step cache active"

    mode = str(speed_mode or "").strip().lower()
    if mode not in ("default", "max"):
        return False, f"speed tier {mode or 'off'}"

    # Lazy, inside the function: ``diffusion_speed`` reaches back into this module for the install
    # arm, so a module-level import here would be a cycle.
    from .diffusion_speed import _denoiser_dits, _denoiser_unet

    if _denoiser_unet(pipe) is not None:
        # A whole-compiled U-Net serves its compiled artifact from ``_compiled_call_impl``, which a
        # forward-slot swap bypasses. Refusing keeps the (larger) compile win.
        return False, "denoiser is a U-Net"

    if not _denoiser_dits(pipe):
        return False, "no denoiser transformer"

    try:
        cuda = getattr(_torch(), "cuda", None)
        if cuda is None or not hasattr(cuda, "CUDAGraph") or not cuda.is_available():
            return False, "torch.cuda unavailable"
    except Exception as exc:  # noqa: BLE001 - a torch-free host is simply not eligible
        _warn(logger, "availability probe", exc)
        return False, "torch.cuda unavailable"

    # Opt-in per family on the video backend (only MiniMax-H3 measured a win there and only H3
    # keeps its input tree stable per step), opt-out per family on the image backend.
    if not bool(getattr(family, "supports_cuda_graph", family_default)):
        return False, "family opts out"

    return True, "eligible"


def install_cuda_graphs(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm one ``GraphedForward`` per denoiser module and stash the handles on the pipe.

    Per module try/except: a dual-DiT family whose second expert cannot be wrapped still gets the
    first one graphed, and neither case may fail the load. Nothing is captured here; the first
    denoising step of the first generation pays the capture."""
    from .diffusion_speed import _denoiser_dits

    handles: list = []
    for module in _denoiser_dits(pipe):
        try:
            handles.append(GraphedForward(module, max_graphs = max_graphs, logger = logger).enable())
        except Exception as exc:  # noqa: BLE001 - optimisation only
            _warn(logger, f"install on {type(module).__name__}", exc)

    installed = tuple(handles)
    try:
        pipe._unsloth_cuda_graphs = installed
    except Exception as exc:  # noqa: BLE001 - a pipe that refuses attributes still runs
        _warn(logger, "handle stash", exc)
    if installed and logger is not None:
        logger.info("diffusion.cuda_graph: armed on %d denoiser module(s)", len(installed))
    return installed


def set_bypass(handles: Any, on: bool) -> None:
    """Run eager without dropping graphs, for every handle. Never raises."""
    for handle in handles or ():
        try:
            handle.set_bypass(on)
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def reset_all(handles: Any) -> None:
    """Drop every captured graph, for every handle (weights changed). Never raises."""
    for handle in handles or ():
        try:
            handle.reset()
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def uninstall_all(handles: Any, *, logger: Any = None) -> None:
    """Free every handle and forget the shared pool if nothing else holds a graph.

    Idempotent and never raises: it runs from unload and from the pre-commit rollback paths, where
    a second call on already-freed handles must be a no-op rather than the exception that hides the
    original failure."""
    for handle in handles or ():
        try:
            if logger is not None:
                logger.debug("diffusion.cuda_graph: %s", handle.describe())
            handle.free()
        except Exception:  # noqa: BLE001 - teardown, never the reason an unload fails
            pass
    _drop_pool_if_unused()


def stats(handles: Any) -> dict:
    """JSON-safe aggregate over the handles, for the status payload and the benchmarks."""
    out = {
        "graphs": 0,
        "captures": 0,
        "replays": 0,
        "eager_calls": 0,
        "fallbacks": 0,
        "cap_skips": 0,
        "poisoned": False,
        "capture_error": None,
    }
    for handle in handles or ():
        try:
            out["graphs"] += len(handle.cache)
            for field in ("captures", "replays", "eager_calls", "fallbacks", "cap_skips"):
                out[field] += int(handle.stats.get(field, 0))
            if handle.poisoned:
                out["poisoned"] = True
            error = handle.capture_error
            if error and out["capture_error"] is None:
                out["capture_error"] = {
                    "type": str(error.get("type")),
                    "msg": str(error.get("msg")),
                }
        except Exception:  # noqa: BLE001 - a stats call must never fail a status request
            pass
    return out
