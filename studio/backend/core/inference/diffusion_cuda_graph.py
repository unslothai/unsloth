# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static-buffer CUDA-graph capture of ONE denoiser forward, for the diffusion backends.

Each denoiser module gets a ``GraphedForward`` in its INSTANCE ``__dict__`` under ``forward``.
Not a proxy: the pipelines read ``.dtype`` / ``.config`` / ``.cache_context`` off the transformer.
Not a hook: hooks move ``nn.Module._call_impl`` off its fast path and offload owns the hook list.
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

# Per module. Legitimate second keys exist, but each costs a pool-sized slice of VRAM.
MAX_GRAPHS_PER_MODULE = 4

# Three per the PyTorch docs. The warmup forces lazy allocations OUTSIDE the capture.
WARMUP_ITERS = 3

_TRUE_TOKENS = ("1", "true", "yes", "on")


def _torch():
    """The torch module, imported lazily so this file imports on a torch-free host."""
    import torch
    return torch


def cuda_graph_disabled() -> bool:
    """Whether the env kill switch is set. Read before any torch or pipe inspection."""
    return os.environ.get(CUDA_GRAPH_DISABLE_ENV, "").strip().lower() in _TRUE_TOKENS


# Not ``torch.utils._pytree``: it makes an unregistered object a LEAF, which must stay visible.


def _flatten(obj: Any, out: list) -> tuple:
    """Append every tensor in ``obj`` to ``out``; return a spec that rebuilds ``obj``."""
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
    """Hashable identity of a call: shape AND stride AND dtype AND device, because a graph replays
    into the exact buffers it recorded against. Scalars key by VALUE, constants at capture time."""
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
    kind = key[0]
    if kind == "o":
        return True
    if kind == "l":
        return any(_uncapturable(k) for k in key[2])
    if kind == "d":
        return any(_uncapturable(k) for _, k in key[1])
    return False


def _has_float(key: tuple) -> bool:
    """True when the key holds a float, ie a per-step timestep meaning one graph per step."""
    kind = key[0]
    if kind == "v":
        return isinstance(key[1], float)
    if kind == "l":
        return any(_has_float(k) for k in key[2])
    if kind == "d":
        return any(_has_float(k) for _, k in key[1])
    return False


# Private pools per graph OOM a dual-DiT family; the graphs never replay concurrently, so share.
_POOL_BOX: list = [None]

# A pool id handed to a capture after the last graph recorded into it died is a dangling pointer.
_LIVE_WRAPPERS: "weakref.WeakSet" = weakref.WeakSet()


def _drop_pool_if_unused() -> None:
    try:
        for wrapper in tuple(_LIVE_WRAPPERS):
            if wrapper.cache:
                return
        _POOL_BOX[0] = None
    except Exception:  # noqa: BLE001
        pass


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.cuda_graph: %s failed: %s", what, exc)


class _Entry:
    """One captured graph plus the static buffers it replays into."""

    __slots__ = ("graph", "static", "in_spec", "out_spec", "out_tensors")


class GraphedForward:
    """A callable that replaces one denoiser module's ``forward`` with a CUDA-graph replay.

    It walks the ``(args, kwargs)`` tree rather than naming positions, so every family goes through
    the identical object. A capture that raises poisons the wrapper rather than half-graphing."""

    def __init__(
        self,
        module: Any,
        *,
        warmup: int = WARMUP_ITERS,
        max_graphs: int = MAX_GRAPHS_PER_MODULE,
        logger: Any = None,
    ) -> None:
        self.module = module
        # The CLASS forward: the eager callable, whatever is already in the instance slot.
        self.orig = type(module).forward.__get__(module)
        try:
            update_wrapper(self, self.orig)
        except Exception:  # noqa: BLE001
            pass
        try:
            # LOAD-BEARING: H3's modular pipeline filters kwargs by ``inspect.signature(
            # transformer.forward).parameters``, so a bare ``**kwargs`` signature drops tensors.
            self.__signature__ = inspect.signature(self.orig)
        except Exception:  # noqa: BLE001
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

    def install(self) -> "GraphedForward":
        """Write ``forward`` into the instance ``__dict__``; ``__setattr__`` would inspect it."""
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
        """Disarm AND uninstall, so the call path matches a load that never built a wrapper."""
        self.enabled = False
        self.uninstall()
        return self

    def set_bypass(self, on: bool) -> "GraphedForward":
        """Run eager without dropping the captured graphs (a chunk under a step cache)."""
        self.bypassed = bool(on)
        return self

    def _release(self) -> None:
        """Return the dropped graphs' pool segments to the device; ``gc.collect`` alone leaves them in the private pool."""
        try:
            gc.collect()
            cuda = getattr(_torch(), "cuda", None)
            if cuda is not None and hasattr(cuda, "empty_cache"):
                cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def reset(self) -> "GraphedForward":
        """Drop every captured graph (weights changed under us: LoRA load, unload, adapter switch)."""
        self.cache.clear()
        self._release()
        return self

    def poison(self, exc: BaseException) -> "GraphedForward":
        """Record why capture failed and run eager for the rest of this wrapper's life."""
        self.capture_error = {
            "type": type(exc).__name__,
            "msg": str(exc)[:4000],
            "traceback": traceback.format_exc()[-6000:],
        }
        self.poisoned = True
        # ``__call__`` short-circuits to eager on ``poisoned`` before it reads the cache, so every
        # entry is now unreachable and only pins its statics, outputs and slice of the pool for the
        # life of the load. Freeing is left to the caller's ``_release``, which runs once the failed
        # capture's own frames are gone, so both go back in one pass.
        self.cache.clear()
        _drop_pool_if_unused()
        return self

    def free(self) -> "GraphedForward":
        """Drop the graphs and the slot."""
        self.reset()
        self.uninstall()
        self.enabled = False
        _LIVE_WRAPPERS.discard(self)
        return self

    def describe(self) -> dict:
        """A JSON-safe summary for the status payload, without the capture traceback."""
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

    def _eager(self, args: tuple, kwargs: dict) -> Any:
        self.stats["eager_calls"] += 1
        return self.orig(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if (
            not self.enabled
            or self.bypassed
            or self.poisoned
            # Belt to the caller's per-chunk bypass: one step's cond and uncond share a key.
            or getattr(self.module, "_unsloth_step_cache", None)
        ):
            return self._eager(args, kwargs)

        # True or absent builds an output dataclass, which a replay hands back frozen.
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
                # The recorded graphs keep replaying: one rare new key is no reason to drop them.
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
            captured = None
            try:
                captured = self._capture(args, kwargs)
            except Exception as exc:  # noqa: BLE001 - a failed capture must never fail the render
                self.poison(exc)
                self.stats["fallbacks"] += 1
                if self.logger is not None:
                    self.logger.warning(
                        "diffusion.cuda_graph: capture failed on %s (%s: %s); this load runs eager",
                        type(self.module).__name__,
                        type(exc).__name__,
                        exc,
                    )
                # The handled exception keeps _capture's frame, and with it the statics and the pool slice,
                # alive through the eager retry, which then OOMs on a tight card. capture_error has the text.
                exc.__traceback__ = None
            if captured is None:
                self._release()
                return self._eager(args, kwargs)
            entry = captured
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
        # Cloned: every replay writes the SAME buffers, which the pipeline holds across steps.
        return _rebuild(entry.out_spec, [t.clone() for t in entry.out_tensors])

    def _capture(self, args: tuple, kwargs: dict) -> _Entry:
        torch = _torch()
        entry = _Entry()
        live: list = []
        entry.in_spec = _flatten((args, kwargs), live)

        bad = [(i, str(t.device)) for i, t in enumerate(live) if t.device.type != "cuda"]
        if bad:
            self.stats["refused_host_tensor"] += 1
            raise RuntimeError(
                f"{len(bad)} input tensor(s) are not on cuda ({bad[:4]}); a host tensor read "
                f"inside a captured region is baked in at its recorded value"
            )

        # A tensor made under ``torch.inference_mode()``, which renders run in, refuses ``copy_``.
        with torch.inference_mode(False):
            entry.static = [torch.empty_like(t) for t in live]
        for dst, src in zip(entry.static, live):
            dst.copy_(src)
        static_args, static_kwargs = _rebuild(entry.in_spec, entry.static)

        # Side stream: a workspace first created DURING capture is only valid while recording.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.orig(*static_args, **static_kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        # ``pool = None`` is identical to omitting the argument, so both captures take one path.
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
    """Whether this load may be graphed, and a short reason when it may not. Cheapest first."""
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
        # Onload hooks move weights host to card, so a graph's baked pointers go stale.
        return False, "offload active"

    if cache_active:
        return False, "step cache active"

    mode = str(speed_mode or "").strip().lower()
    if mode not in ("default", "max"):
        return False, f"speed tier {mode or 'off'}"

    # Lazy: ``diffusion_speed`` imports this module, so a module-level import here is a cycle.
    from .diffusion_speed import _denoiser_dits, _denoiser_unet

    if _denoiser_unet(pipe) is not None:
        # A whole-compiled U-Net serves from ``_compiled_call_impl``, which a slot swap bypasses.
        return False, "denoiser is a U-Net"

    if not _denoiser_dits(pipe):
        return False, "no denoiser transformer"

    try:
        cuda = getattr(_torch(), "cuda", None)
        if cuda is None or not hasattr(cuda, "CUDAGraph") or not cuda.is_available():
            return False, "torch.cuda unavailable"
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "availability probe", exc)
        return False, "torch.cuda unavailable"

    if not bool(getattr(family, "supports_cuda_graph", family_default)):
        return False, "family opts out"

    return True, "eligible"


def install_cuda_graphs(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm one ``GraphedForward`` per denoiser module; the first denoising step captures."""
    from .diffusion_speed import _denoiser_dits

    handles: list = []
    for module in _denoiser_dits(pipe):
        try:
            handles.append(GraphedForward(module, max_graphs = max_graphs, logger = logger).enable())
        except Exception as exc:  # noqa: BLE001 - a second expert may fail without failing the load
            _warn(logger, f"install on {type(module).__name__}", exc)

    installed = tuple(handles)
    try:
        pipe._unsloth_cuda_graphs = installed
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "handle stash", exc)
    if installed and logger is not None:
        logger.info("diffusion.cuda_graph: armed on %d denoiser module(s)", len(installed))
    return installed


def set_bypass(handles: Any, on: bool) -> None:
    """Run eager without dropping graphs, for every handle."""
    for handle in handles or ():
        try:
            handle.set_bypass(on)
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def reset_all(handles: Any) -> None:
    """Drop every captured graph, for every handle (the weights changed).

    Drops the pool token with the last graph: a stale token dies on the allocator's
    "use_count > 0 INTERNAL ASSERT FAILED", raised after ``torch.cuda.graph`` entered its side
    stream, so the thread is left off the default stream too."""
    for handle in handles or ():
        try:
            handle.reset()
        except Exception:  # noqa: BLE001 - optimisation only
            pass
    _drop_pool_if_unused()


def uninstall_all(handles: Any, *, logger: Any = None) -> None:
    """Free every handle and forget the shared pool if nothing else holds a graph. Idempotent."""
    for handle in handles or ():
        try:
            if logger is not None:
                logger.debug("diffusion.cuda_graph: %s", handle.describe())
            handle.free()
        except Exception:  # noqa: BLE001
            pass
    _drop_pool_if_unused()


def stats(handles: Any) -> dict:
    """JSON-safe aggregate over the handles, for the status payload."""
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
        except Exception:  # noqa: BLE001
            pass
    return out
