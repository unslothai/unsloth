# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention (``unsloth_int8``) for the families whose attention goes through diffusers' attention dispatch.

The strided processors (FLUX.3, MiniMax-H3, MiniMax-H3-Image) call the kernel themselves. Every other DiT processor
calls ``dispatch_attention_fn(q, k, v, ..., backend=self._attention_backend)`` with ``[B, S, H, D]`` views, which is
the kernel's own layout. So no family needs a processor of its own: ``install`` gives each processor that calls the
dispatch an ``Int8Route`` as its backend, and the dispatch wrapper installed here sends the calls it can take to
``torch.ops.unsloth_studio.int8_attention_nhd``. Everything else goes to the backend the processor had before:

* masked or causal calls, dropout, a non-default scale, GQA, LSE requests, extra attention kwargs, context parallel;
* gradient calls, a dtype or head dim the card check did not pass, another device;
* calls with fewer query or key tokens than the race found the kernel to win at.

``Int8Route`` is a ``str`` equal to the incumbent backend's name, so code that reads ``_attention_backend`` without
knowing it (``AttentionBackendName(...)``, ``.value``, comparisons) sees the incumbent. The wrapper is a plain Python
function and the kernel an opaque custom op with a fake, so a compiled block traces with no graph break; the route's
fields are guarded constants.

Which loads route (``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL`` is read by ``diffusion_int8_attention.mode``):

* ``0``: never; ``UNSLOTH_DIFFUSION_INT8_ATTENTION_DISPATCH=0`` turns off only this route.
* ``1``, or the load asked for ``unsloth_int8`` by name: every DiT whose processors call the dispatch, wherever the
  kernel passes its accuracy check on this card (no race, except the Triton stand-in).
* unset: the DiT classes in ``AUTO_DITS`` on NVIDIA, where the kernel passes the check and wins the timed race
  against the processor's own backend by more than ``RACE_MARGIN``; from ``RACE_TOKENS[1]`` tokens on at least. Where
  it loses at the race's sizes, a second race at ``LONG_RACE_TOKENS`` (video and large-image lengths) can still route
  the calls from that length on: the int8 products win more the longer the sequence (A100: 0.96x at 4096 tokens,
  1.37x at 27k).
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Optional

DISPATCH_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_DISPATCH"
_DIFFUSERS_DISPATCH = "diffusers.models.attention_dispatch"
_ROUTED_ATTR = "_unsloth_int8_dispatch"
# DiT classes the unset switch races on: those whose teacher-forced per-step denoiser error with the kernel stays
# within the error of the INT8 weights Studio ships for them, on every output. Left out, so they route only by name
# or with the env switch: Z-Image-Turbo (rel L2 0.035 against 0.024 for its INT8 weights, G4) and LTX-2 (video 0.044
# against 0.046, but its audio output 0.010 against 0.008).
AUTO_DITS = frozenset(
    {
        "QwenImage21Transformer2DModel",
        "WanTransformer3DModel",
    }
)
# processors that already run the kernel themselves
_STRIDED_FLAG = "_unsloth_int8"

# (large, small) token counts of the second race, for a card where the kernel loses at RACE_TOKENS
LONG_RACE_TOKENS = (16384, 8192)

_LOCK = threading.Lock()
_LONG_VERDICTS: dict = {}
_STATE: dict = {"original": None}
# eager calls by outcome ("int8" or why the incumbent ran); compiled calls are counted when they are traced
ROUTE_STATS: dict = {}


def enabled() -> bool:
    raw = (os.environ.get(DISPATCH_ENV) or "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _backend_name(backend: Any) -> str:
    if backend is None:
        return "native"
    return str(getattr(backend, "value", backend))


class Int8Route(str):
    """A processor's attention backend while it routes to the INT8 kernel: equal to the incumbent's name, with the
    incumbent itself and what the card check and race decided. ``cards`` maps each CUDA device index the kernel
    passed on to its token floor (a layer split or CFG replica runs the same processor on a second card);
    ``min_tokens`` is the lowest of them, ``index`` the load's card. Built by ``install`` only."""

    def __new__(
        cls,
        incumbent: Any,
        min_tokens: int,
        dtype: Any,
        index: int,
        head_dim: int,
        impl: Optional[str] = None,
        cards: Optional[dict] = None,
    ) -> "Int8Route":
        if isinstance(incumbent, Int8Route):
            incumbent = incumbent.incumbent
        route = str.__new__(cls, _backend_name(incumbent))
        route.incumbent = incumbent
        route.cards = (
            {int(k): int(v) for k, v in cards.items()} if cards else {int(index): int(min_tokens)}
        )
        route.min_tokens = min(route.cards.values())
        route.dtype = dtype
        route.index = int(index)
        route.head_dim = int(head_dim)
        route.impl = impl
        return route

    @property
    def value(self) -> str:
        return str.__str__(self)

    @property
    def name(self) -> str:
        return str(getattr(self.incumbent, "name", None) or self.value).upper()

    def __reduce__(self) -> tuple:
        return (
            Int8Route,
            (
                self.incumbent,
                self.min_tokens,
                self.dtype,
                self.index,
                self.head_dim,
                self.impl,
                self.cards,
            ),
        )

    def __repr__(self) -> str:
        return f"Int8Route({self.value!r}, cards={self.cards})"


def _arg(
    args: tuple,
    kwargs: dict,
    index: int,
    name: str,
    default: Any = None,
) -> Any:
    return args[index] if len(args) > index else kwargs.get(name, default)


def _reason(route: Int8Route, args: tuple, kwargs: dict) -> Optional[str]:
    """Why this dispatch call keeps the incumbent, or None when the kernel takes it."""
    import torch

    query = _arg(args, kwargs, 0, "query")
    key = _arg(args, kwargs, 1, "key")
    value = _arg(args, kwargs, 2, "value")
    if not (torch.is_tensor(query) and torch.is_tensor(key) and torch.is_tensor(value)):
        return "not tensors"
    if _arg(args, kwargs, 3, "attn_mask") is not None:
        return "mask"
    # cheapest decisive check next: an eager family below the token floor pays only these few lines per call
    if (
        query.dim() == 4
        and key.dim() == 4
        and (query.shape[1] < route.min_tokens or key.shape[1] < route.min_tokens)
    ):
        return "short"
    if _arg(args, kwargs, 5, "is_causal", False):
        return "causal"
    if _arg(args, kwargs, 4, "dropout_p", 0.0):
        return "dropout"
    if _arg(args, kwargs, 7, "enable_gqa", False):
        return "gqa"
    if _arg(args, kwargs, 8, "attention_kwargs"):
        return "attention kwargs"
    if kwargs.get("parallel_config") is not None:
        return "context parallel"
    if len(args) > 9 or any(name not in _KNOWN for name in kwargs):
        return "unknown arguments"
    if query.dim() != 4 or key.dim() != 4 or tuple(key.shape) != tuple(value.shape):
        return "not [B, S, H, D]"
    if (
        query.shape[0] != key.shape[0]
        or query.shape[2] != key.shape[2]
        or query.shape[3] != route.head_dim
        or key.shape[3] != route.head_dim
    ):
        return "shapes"
    scale = _arg(args, kwargs, 6, "scale")
    if scale is not None and abs(float(scale) * route.head_dim**0.5 - 1.0) > 1e-6:
        return "scale"
    if not (query.dtype == key.dtype == value.dtype == route.dtype):
        return "dtype"
    if not (_on_card(query, route) and _on_card(key, route) and _on_card(value, route)):
        return "device"
    if query.device != key.device or query.device != value.device:
        return "device"
    floor = route.cards.get(query.device.index, route.min_tokens)
    if query.shape[1] < floor or key.shape[1] < floor:
        return "short"
    if torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    ):
        # no backward in the kernel; autograd cannot see into the op
        return "needs gradients"
    return None


def _on_card(x: Any, route: Int8Route) -> bool:
    """Whether a tensor is on a card the route was checked on."""
    return bool(x.is_cuda) and x.device.index in route.cards


_KNOWN = frozenset(
    {
        "query",
        "key",
        "value",
        "attn_mask",
        "dropout_p",
        "is_causal",
        "scale",
        "enable_gqa",
        "attention_kwargs",
        "backend",
        "parallel_config",
    }
)


def _note(outcome: str) -> None:
    ROUTE_STATS[outcome] = ROUTE_STATS.get(outcome, 0) + 1


def routed_dispatch_attention_fn(*args: Any, **kwargs: Any) -> Any:
    """diffusers' ``dispatch_attention_fn`` with the INT8 kernel for the calls of routed processors it can take."""
    import torch

    backend = kwargs.get("backend")
    if isinstance(backend, Int8Route):
        kwargs["backend"] = backend.incumbent
        reason = _reason(backend, args, kwargs)
        if not torch.compiler.is_compiling():
            _note(reason or "int8")
        if reason is None:
            return torch.ops.unsloth_studio.int8_attention_nhd(
                _arg(args, kwargs, 0, "query"),
                _arg(args, kwargs, 1, "key"),
                _arg(args, kwargs, 2, "value"),
            )
    return _STATE["original"](*args, **kwargs)


routed_dispatch_attention_fn._unsloth_int8_dispatch = True  # type: ignore[attr-defined]


def original_dispatch() -> Any:
    """diffusers' own ``dispatch_attention_fn`` (None when diffusers has none)."""
    if _STATE["original"] is not None:
        return _STATE["original"]
    try:
        from diffusers.models import attention_dispatch
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(attention_dispatch, "dispatch_attention_fn", None)
    if fn is None or getattr(fn, _ROUTED_ATTR, False):
        return None
    _STATE["original"] = fn
    return fn


def install_dispatch_route() -> int:
    """Point every loaded diffusers / Studio module's ``dispatch_attention_fn`` at the routing wrapper (the processors
    read it from their module's globals at call time). Calls whose backend is not an ``Int8Route`` pass through
    unchanged. Idempotent; returns the number of modules that now hold the wrapper."""
    original = original_dispatch()
    if original is None:
        return 0
    count = 0
    with _LOCK:
        for name, module in list(sys.modules.items()):
            if module is None or not (
                name.startswith("diffusers.") or name.startswith(__package__ or "core.inference")
            ):
                continue
            try:
                current = getattr(module, "dispatch_attention_fn", None)
            except Exception:  # noqa: BLE001 - lazy modules
                continue
            if current is original:
                try:
                    setattr(module, "dispatch_attention_fn", routed_dispatch_attention_fn)
                except Exception:  # noqa: BLE001
                    continue
            if getattr(module, "dispatch_attention_fn", None) is routed_dispatch_attention_fn:
                count += 1
    return count


def _calls_dispatch(proc: Any) -> bool:
    """Whether the processor's ``__call__`` calls ``dispatch_attention_fn`` (directly, or through a module attribute
    or a function-level import of it)."""
    call = getattr(type(proc), "__call__", None)
    code = getattr(call, "__code__", None)
    if code is None:
        return False
    names = set(code.co_names)
    for const in code.co_consts:
        inner = getattr(const, "co_names", None)
        if inner:
            names.update(inner)
    return "dispatch_attention_fn" in names


def routable_modules(dit: Any) -> list:
    """The attention modules of a DiT whose processor can take an ``Int8Route``."""
    modules = getattr(dit, "modules", None)
    if not callable(modules):
        return []
    out = []
    for m in modules():
        proc = getattr(m, "processor", None)
        if proc is None or not hasattr(m, "heads") or not hasattr(proc, "_attention_backend"):
            continue
        if hasattr(type(proc), _STRIDED_FLAG):
            continue
        if getattr(proc, "_parallel_config", None) is not None:
            continue
        if not _calls_dispatch(proc):
            continue
        out.append(m)
    return out


def _dits(pipe: Any) -> list:
    from .diffusion_attention import _attention_dits
    try:
        return list(_attention_dits(pipe))
    except Exception:  # noqa: BLE001
        return []


def remove(pipe: Any) -> int:
    """Every routed processor back to its incumbent backend. Returns how many were routed."""
    count = 0
    for dit in _dits(pipe):
        modules = getattr(dit, "modules", None)
        if not callable(modules):
            continue
        for m in modules():
            proc = getattr(m, "processor", None)
            backend = getattr(proc, "_attention_backend", None)
            if isinstance(backend, Int8Route):
                proc._attention_backend = backend.incumbent
                count += 1
    return count


def routed_count(pipe: Any) -> int:
    """Engagement census: attention modules whose dispatch calls may run the kernel."""
    return sum(
        1
        for dit in _dits(pipe)
        for m in (dit.modules() if callable(getattr(dit, "modules", None)) else ())
        if isinstance(getattr(getattr(m, "processor", None), "_attention_backend", None), Int8Route)
    )


def status(pipe: Any = None) -> dict:
    routes: dict = {}
    for dit in _dits(pipe) if pipe is not None else ():
        for m in dit.modules():
            backend = getattr(getattr(m, "processor", None), "_attention_backend", None)
            if isinstance(backend, Int8Route):
                key = f"{type(dit).__name__} heads={m.heads} head_dim={backend.head_dim} vs {backend.value} cards={backend.cards} impl={backend.impl}"
                routes[key] = routes.get(key, 0) + 1
    return {
        "enabled": enabled(),
        "modules": sum(routes.values()),
        "routes": routes,
        "calls": dict(ROUTE_STATS),
    }


def install(
    pipe: Any,
    target: Any,
    *,
    requested: Any = None,
    logger: Any = None,
) -> int:
    """Route the pipeline's DiT attention dispatch calls to the INT8 kernel where this card and the switch allow it
    (module docstring). Returns the number of attention modules routed; never raises."""
    try:
        return _install(pipe, target, requested, logger)
    except Exception as exc:  # noqa: BLE001 -- an optimisation: the processors keep their backend
        remove(pipe)
        if logger is not None:
            logger.info("diffusion.int8_attention: dispatch route skipped (%s)", exc)
        return 0


def long_verdict(
    device: Any, dtype: Any, heads: int, head_dim: int, incumbent: Any, name: str
) -> Optional[int]:
    """The token count from which the kernel beats the incumbent by more than ``RACE_MARGIN`` at
    ``LONG_RACE_TOKENS``, or None. Timed like the card race (interleaved, median); cached per card, dtype, geometry
    and incumbent; an out-of-memory probe counts as a loss."""
    import torch

    from . import diffusion_int8_attention as I8

    key = (str(device), str(dtype), int(heads), int(head_dim), name)
    if key in _LONG_VERDICTS:
        return _LONG_VERDICTS[key]
    wins: dict = {}
    for tokens in LONG_RACE_TOKENS:
        try:
            with torch.no_grad():
                q, k, v = I8.probe_inputs(device, dtype, tokens, heads, head_dim, seed = tokens)
                ms = I8._median_ms(
                    {
                        "incumbent": lambda: incumbent(q, k, v),
                        "int8": lambda: I8.int8_attention(q, k, v),
                    }
                )
            wins[tokens] = ms["int8"] < I8.RACE_MARGIN * ms["incumbent"]
        except torch.OutOfMemoryError:
            wins[tokens] = False
        finally:
            q = k = v = None
            torch.cuda.empty_cache()
    large, small = LONG_RACE_TOKENS
    found = None
    if wins[large]:
        found = (small + int(I8.RACE_TOKENS[0])) // 2 if wins[small] else (large + small) // 2
    _LONG_VERDICTS[key] = found
    return found


def _secondary_cards(primary: int) -> tuple:
    """The other CUDA cards of this load's multi-GPU selection (a layer split or a CFG replica places DiT blocks
    there after the attention pick), or none."""
    try:
        from .diffusion_multigpu import current_selection
        selection = current_selection()
    except Exception:  # noqa: BLE001
        return ()
    if selection is None or int(selection.primary) != int(primary):
        return ()
    return tuple(int(c) for c in selection.secondaries)


def _card_floor(
    device: Any,
    dtype: Any,
    heads: int,
    head_dim: int,
    run: Any,
    name: str,
    forced: bool,
    env_force: bool,
) -> tuple:
    """(verdict, token floor or None) of one card: the accuracy check, then the race (both races unless forced)."""
    from . import diffusion_int8_attention as I8

    found = I8.verdict(device, dtype, heads, head_dim, run, name, timed = not forced)
    stand_in = (
        found.get("impl") == "triton"
        and not env_force
        and I8.impl_setting() == "auto"
        and not I8._is_rocm()
    )
    if forced and stand_in and found.get("ok"):
        found = I8.verdict(device, dtype, heads, head_dim, run, name, timed = True)
    if not found.get("ok"):
        return found, None
    if forced and not stand_in:
        return found, 0
    if found.get("min_tokens") is None:
        minimum = long_verdict(device, dtype, heads, head_dim, run, name)
        return {**found, "long_min_tokens": minimum}, minimum
    return found, max(int(found["min_tokens"]), int(I8.RACE_TOKENS[-1]))


def _target_device(target: Any) -> str:
    return str(getattr(target, "torch_device", None) or getattr(target, "device", "") or "")


def _install(pipe: Any, target: Any, requested: Any, logger: Any) -> int:
    import torch

    from . import diffusion_int8_attention as I8

    remove(pipe)
    setting = I8.mode()
    if setting == "off" or not enabled():
        return 0
    forced = setting == "force" or I8.requested(requested)
    device = _target_device(target)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0
    if not forced and I8._is_rocm():
        return 0
    dev = torch.device(device)
    index = I8._device_index(dev)
    dev = torch.device("cuda", index)
    dtype = getattr(target, "dtype", None)
    if str(dtype) not in ("torch.float16", "torch.bfloat16"):
        dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(index) >= (8, 0) else torch.float16
        )
    candidates = []
    for dit in _dits(pipe):
        if not forced and type(dit).__name__ not in AUTO_DITS:
            continue
        candidates.extend(routable_modules(dit))
    if not candidates or install_dispatch_route() == 0:
        return 0
    I8.int8_attention_op()
    original = original_dispatch()
    env_force = setting == "force"
    cards = [index] + [c for c in _secondary_cards(index) if c != index]
    report: dict = {}
    routed = 0
    for m in candidates:
        proc = m.processor
        head_dim = I8._head_dim(m)
        if head_dim is None:
            continue
        incumbent = proc._attention_backend
        name = f"dispatch:{_backend_name(incumbent)}"

        def run(
            q: Any,
            k: Any,
            v: Any,
            backend: Any = incumbent,
        ) -> Any:
            return original(q, k, v, backend = backend)

        floors: dict = {}
        impl = None
        for card in cards:
            if I8.implementation(card, head_dim) is None:
                continue
            found, minimum = _card_floor(
                torch.device("cuda", card),
                dtype,
                int(m.heads),
                head_dim,
                run,
                name,
                forced,
                env_force,
            )
            report[(card, int(m.heads), head_dim, name)] = found
            if minimum is not None:
                floors[card] = minimum
                impl = impl or found.get("impl")
        if not floors:
            continue
        proc._attention_backend = Int8Route(
            incumbent,
            floors.get(index, min(floors.values())),
            dtype,
            index,
            head_dim,
            impl,
            cards = floors,
        )
        routed += 1
    if logger is not None:
        for (card, heads, head_dim, name), found in report.items():
            logger.info(
                "diffusion.int8_attention: cuda:%d heads=%d head_dim=%d vs %s: impl=%s ok=%s cos=%s ms=%s "
                "min_tokens=%s%s%s",
                card,
                heads,
                head_dim,
                name,
                found.get("impl"),
                found.get("ok"),
                found.get("cos"),
                found.get("ms"),
                found.get("min_tokens"),
                f" long race min_tokens={found['long_min_tokens']}"
                if "long_min_tokens" in found
                else "",
                " (forced)" if forced else "",
            )
        if routed:
            logger.info(
                "diffusion.int8_attention: %s on %d attention modules through the attention dispatch (cards %s)",
                I8.BACKEND,
                routed,
                ",".join(str(c) for c in cards),
            )
    return routed
