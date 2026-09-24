# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in static step skip for the diffusion transformer (``transformer_cache="static"``).

First-Block-Cache decides per step from the data, so the decision runs inside the denoiser
forward: the compile loses fullgraph and the CUDA graph runs eager. This layer decides from a
schedule fixed before the generation starts instead. The first ``head`` and last ``tail``
fraction of the steps always compute; the middle computes one step in ``every`` and a skipped
step returns the last computed output of the same CFG branch (``reuse``), or a first-order
extrapolation in timestep over the last two (``taylor1``). Nothing data-dependent runs inside
the forward, so the compile and the graph see exactly what an uncached load gives them.

Measured on Qwen-Image-2.1, 1024px, 25 steps, CFG 1: reuse / every 2 skipped 111 of 315 denoiser
calls at LPIPS 0.017 (mean) against uncached, with the CUDA graph kept.

The layer sits in the transformer INSTANCE ``__dict__["forward"]``, outermost: a GraphedForward
installed later goes UNDER it (see ``diffusion_cuda_graph._outer_layer``), so the graph only ever
sees computed steps, with their unchanged arguments and per-shape keys. torch is imported lazily.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from functools import update_wrapper
from typing import Any, Optional

from .diffusion_cache import TC_STATIC

MODE_REUSE = "reuse"
MODE_TAYLOR1 = "taylor1"
SKIP_MODES = (MODE_REUSE, MODE_TAYLOR1)

DEFAULT_MODE = MODE_TAYLOR1
DEFAULT_HEAD = 0.2
DEFAULT_TAIL = 0.1
DEFAULT_EVERY = 2

# Fewer steps leave no middle worth skipping, and a distilled few-step model has no step to spare.
STATIC_MIN_STEPS = 12
# A prefix-KV DiT (Qwen-Image-2.1, FLUX.2 klein KV) returns prompt + target rows on step 0 and
# target rows after, so the output a skip reuses must come from step 1 or later.
MIN_HEAD_STEPS = 2

# Benchmarking knobs, read at install. The request's transformer_cache_threshold is ignored for
# static: the schedule is the whole policy.
ENV_MODE = "UNSLOTH_STATIC_SKIP_MODE"
ENV_HEAD = "UNSLOTH_STATIC_SKIP_HEAD"
ENV_TAIL = "UNSLOTH_STATIC_SKIP_TAIL"
ENV_EVERY = "UNSLOTH_STATIC_SKIP_EVERY"

_SLOT = "_unsloth_static_skip"


def _torch():
    import torch
    return torch


def _is_tensor(obj: Any) -> bool:
    try:
        is_tensor = getattr(_torch(), "is_tensor", None)
    except Exception:  # noqa: BLE001 - no torch, no tensors
        return False
    return bool(callable(is_tensor) and is_tensor(obj))


def static_schedule(
    steps: Optional[int],
    *,
    head: float = DEFAULT_HEAD,
    tail: float = DEFAULT_TAIL,
    every: int = DEFAULT_EVERY,
) -> tuple:
    """Which denoise steps compute (True) and which reuse (False).

    Empty, meaning compute every step, for an unknown count or fewer than ``STATIC_MIN_STEPS``.
    ``round`` (banker's) matches the measured prototype: 25 steps -> head 5, tail 2, 9 skipped.
    """
    try:
        n = int(steps)  # type: ignore[arg-type]
        every = int(every)
    except (TypeError, ValueError):
        return ()
    if n < STATIC_MIN_STEPS or every < 2:
        return ()
    first = max(MIN_HEAD_STEPS, round(n * float(head)))
    last = max(1, round(n * float(tail)))
    return tuple(i < first or i >= n - last or (i - first) % every == 0 for i in range(n))


def static_skip_settings(env: Optional[dict] = None, logger: Any = None) -> dict:
    """The schedule knobs, from the environment. A malformed knob keeps its default."""
    env = os.environ if env is None else env
    out = {
        "mode": DEFAULT_MODE,
        "head": DEFAULT_HEAD,
        "tail": DEFAULT_TAIL,
        "every": DEFAULT_EVERY,
    }
    mode = str(env.get(ENV_MODE, "") or "").strip().lower()
    if mode in SKIP_MODES:
        out["mode"] = mode
    elif mode and logger is not None:
        logger.warning("diffusion.step_skip: ignoring %s=%r", ENV_MODE, mode)
    for key, name in (("head", ENV_HEAD), ("tail", ENV_TAIL)):
        raw = env.get(name)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
            if not 0.0 <= value < 1.0:
                raise ValueError(raw)
            out[key] = value
        except (TypeError, ValueError):
            if logger is not None:
                logger.warning("diffusion.step_skip: ignoring %s=%r", name, raw)
    if out["head"] + out["tail"] >= 1.0:
        out["head"], out["tail"] = DEFAULT_HEAD, DEFAULT_TAIL
    raw = env.get(ENV_EVERY)
    if raw not in (None, ""):
        try:
            value = int(raw)
            if value < 2:
                raise ValueError(raw)
            out["every"] = value
        except (TypeError, ValueError):
            if logger is not None:
                logger.warning("diffusion.step_skip: ignoring %s=%r", ENV_EVERY, raw)
    return out


def _call_signature(args: tuple, kwargs: dict) -> tuple:
    """What a reused output must have been computed under: the latent shape, the prefix-KV mode
    (an ``extract`` call fills the cache and returns a longer sequence) and the container asked
    for. A skip only reuses an output whose signature matches the current call."""
    hidden = kwargs.get("hidden_states", args[0] if args else None)
    if _is_tensor(hidden):
        shape = tuple(hidden.shape)
    elif isinstance(hidden, (list, tuple)) and hidden and all(_is_tensor(h) for h in hidden):
        # Z-Image passes one latent per image.
        shape = tuple(tuple(h.shape) for h in hidden)
    else:
        shape = None
    mode = kwargs.get("kv_cache_mode")
    if not isinstance(mode, (str, type(None))):
        mode = repr(mode)
    return (shape, mode, kwargs.get("return_dict", True) is False)


def _same_shape_tensors(items: list) -> bool:
    return (
        bool(items)
        and all(_is_tensor(t) for t in items)
        and len({(tuple(t.shape), t.dtype, t.device) for t in items}) == 1
    )


def _split_output(out: Any) -> tuple:
    """(noise prediction, rebuild) for a container a skip can reproduce, else (None, None).

    A bare tensor, a 1-tuple of one, a 1-tuple of a list of same-shape tensors (Z-Image, one per
    image; stacked here and split back), or a diffusers output dataclass holding one tensor (or such a
    list) field (``Transformer2DModelOutput(sample=...)``), rebuilt through its own class. Anything else,
    e.g. FLUX.2 klein KV's ``(noise, kv_cache)`` extract step, is not reusable."""
    if _is_tensor(out):
        return out, lambda v: v
    if type(out) is tuple:
        if len(out) == 1 and _is_tensor(out[0]):
            return out[0], lambda v: (v,)
        if len(out) == 1 and type(out[0]) is list and _same_shape_tensors(out[0]):
            return _torch().stack(out[0]), lambda v: (list(v.unbind(0)),)
        return None, None
    keys = getattr(out, "keys", None)
    if callable(keys) and callable(getattr(out, "to_tuple", None)):
        try:
            names = list(keys())
            if len(names) != 1:
                return None, None
            name = names[0]
            value = out[name]
        except Exception:  # noqa: BLE001 - not the mapping it looked like
            return None, None
        cls = type(out)
        if _is_tensor(value):
            return value, lambda v: cls(**{name: v})
        # Z-Image img2img / inpaint omit return_dict=False: Transformer2DModelOutput(sample=list).
        if type(value) is list and _same_shape_tensors(value):
            return _torch().stack(value), lambda v: cls(**{name: list(v.unbind(0))})
        return None, None
    return None, None


_TIMESTEP_NAMES = ("timestep", "timesteps", "t")


def _timestep_slot(signature: Any) -> tuple:
    """(name, positional index or None) of the forward's timestep parameter, ("timestep", None) when
    unknown. Z-Image calls ``transformer(x, t, cap_feats)`` positionally; FLUX / Qwen pass ``timestep=``."""
    try:
        params = list(signature.parameters.values())
    except Exception:  # noqa: BLE001
        return "timestep", None
    names = {p.name: i for i, p in enumerate(params)}
    for name in _TIMESTEP_NAMES:
        if name in names:
            index = names[name]
            positional = all(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params[: index + 1]
            )
            return name, index if positional else None
    return "timestep", None


def _timestep_of(
    args: tuple,
    kwargs: dict,
    slot: tuple = ("timestep", None),
) -> Any:
    """The call's timestep as a one-element float32 tensor on its own device (no host sync).

    The max, not the first element: a per-token timestep (Wan2.2 TI2V ``expand_timesteps``, LTX)
    is ``mask * t`` with 0 on conditioned tokens, so the first element can be 0 at every step and
    would turn taylor1 into a silent reuse. On a per-sample timestep the max is the step's t."""
    name, index = slot
    t = kwargs.get(name)
    if t is None and index is not None and index < len(args):
        t = args[index]
    try:
        if _is_tensor(t):
            return t.detach().reshape(-1).float().amax().reshape(1).clone()
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            return _torch().tensor([float(t)], dtype = _torch().float32)
    except Exception:  # noqa: BLE001 - no timestep, taylor1 falls back to reuse
        return None
    return None


def _extrapolate(v0: Any, t0: Any, v1: Any, t1: Any, t: Any) -> Any:
    """First order in timestep: v1 + (v1 - v0) * (t - t1) / (t1 - t0), 0 when t1 == t0."""
    torch = _torch()
    device = v1.device
    t0, t1, t = t0.to(device), t1.to(device), t.to(device)
    step = t1 - t0
    safe = torch.where(step != 0, step, torch.ones_like(step))
    ratio = torch.where(step != 0, (t - t1) / safe, torch.zeros_like(step))
    ratio = ratio.reshape([1] * v1.ndim) if v1.ndim else ratio.reshape(())
    v1f = v1.float()
    return (v1f + (v1f - v0.float()) * ratio).to(v1.dtype)


class _Stored:
    __slots__ = ("sig", "t", "value", "rebuild")

    def __init__(self, sig: tuple, t: Any, value: Any, rebuild: Any) -> None:
        self.sig, self.t, self.value, self.rebuild = sig, t, value, rebuild


class StaticStepSkip:
    """The outer forward layer: counts denoiser calls per CFG branch and skips the scheduled ones.

    Branch key: the open ``cache_context`` name, else the call's ordinal within the step when
    the backend signals step ends (``step_end``), else one shared counter. Step index: the number
    of step-end signals, else the per-branch call count. Unarmed (no plan) it is a pure
    passthrough, which is also what an uninstall leaves behind when someone wrapped it later.
    """

    _unsloth_outer_forward = True

    def __init__(
        self,
        module: Any,
        *,
        inner: Any = None,
        mode: str = DEFAULT_MODE,
        head: float = DEFAULT_HEAD,
        tail: float = DEFAULT_TAIL,
        every: int = DEFAULT_EVERY,
        logger: Any = None,
    ) -> None:
        # First, so nothing update_wrapper copies over can shadow the state set below.
        cls_forward = type(module).forward.__get__(module)
        try:
            update_wrapper(self, cls_forward)
        except Exception:  # noqa: BLE001
            pass
        self._t_slot = ("timestep", None)
        try:
            # Pipelines filter kwargs by inspect.signature(transformer.forward); keep the real one visible.
            self.__signature__ = inspect.signature(cls_forward)
            self._t_slot = _timestep_slot(self.__signature__)
        except Exception:  # noqa: BLE001
            pass
        self.module = module
        # The next forward down: a GraphedForward, an instance forward that was already there, or None for the class
        # forward.
        self.inner = inner
        self.mode = mode if mode in SKIP_MODES else DEFAULT_MODE
        self.head, self.tail, self.every = float(head), float(tail), int(every)
        self.logger = logger
        self.armed = True
        self.context: Any = None
        self._warned_container = False
        self.stats = {"calls": 0, "computed": 0, "skipped": 0}
        self.last_stats = dict(self.stats)
        self.reset(None)

    def reset(
        self,
        steps: Optional[int],
        *,
        step_signal: bool = False,
        keep_stats: bool = False,
    ) -> "StaticStepSkip":
        """Start a forward of ``steps`` effective denoise steps (None: compute every step). ``keep_stats``
        carries the counters over, for the later chunks of one generation."""
        self.plan = (
            static_schedule(steps, head = self.head, tail = self.tail, every = self.every)
            if self.armed
            else ()
        )
        skips = [i for i, compute in enumerate(self.plan) if not compute]
        # Nothing after the last skipped step is ever reused, so the tail stores nothing.
        self.last_skip = skips[-1] if skips else -1
        self.step_signal = bool(step_signal)
        # A generation is running even when its schedule is empty (too few steps): its calls count as computed,
        # so the status route never reports the previous generation's skips for it.
        self.counting = steps is not None
        self.steps_ended = 0
        self.ordinal = 0
        self.counters: dict = {}
        self.history: dict = {}
        if keep_stats:
            return self
        if self.stats["calls"] or self.counting:
            # The generation that just ended, kept for the status route once the per-call reset clears it. Arming a
            # new one clears it too: until its first call, or if it fails before one, status reports zeros, not the
            # previous generation's counts.
            self.last_stats = dict(self.stats)
        self.stats = {"calls": 0, "computed": 0, "skipped": 0}
        return self

    def step_end(self) -> None:
        """One denoise step finished (the pipeline's ``callback_on_step_end``)."""
        self.steps_ended += 1
        self.ordinal = 0

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "head": self.head,
            "tail": self.tail,
            "every": self.every,
            "armed": bool(self.armed),
            "planned_skips": sum(1 for c in self.plan if not c),
            "stats": dict(self.stats if self.stats["calls"] else self.last_stats),
        }

    def _forward(self, args: tuple, kwargs: dict) -> Any:
        inner = self.inner
        if inner is None:
            return type(self.module).forward(self.module, *args, **kwargs)
        return inner(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if not self.plan:
            if self.counting:
                self.stats["calls"] += 1
                self.stats["computed"] += 1
            return self._forward(args, kwargs)
        if self.context is not None:
            key = self.context
        else:
            key = f"#{self.ordinal}" if self.step_signal else None
        self.ordinal += 1
        if self.step_signal:
            step = self.steps_ended
        else:
            step = self.counters.get(key, 0)
            self.counters[key] = step + 1
        self.stats["calls"] += 1
        sig = _call_signature(args, kwargs)
        history = self.history.get(key)
        if (
            0 <= step < len(self.plan)
            and not self.plan[step]
            and history
            and history[-1].sig == sig
        ):
            try:
                out = self._skipped(history, args, kwargs)
            except Exception as exc:  # noqa: BLE001 - a skip that cannot be built computes instead
                out = None
                if self.logger is not None:
                    self.logger.debug("diffusion.step_skip: computing step %s (%s)", step, exc)
            if out is not None:
                self.stats["skipped"] += 1
                return out
        out = self._forward(args, kwargs)
        self.stats["computed"] += 1
        if step < self.last_skip:
            self._remember(key, sig, args, kwargs, out)
        return out

    def _remember(self, key: Any, sig: tuple, args: tuple, kwargs: dict, out: Any) -> None:
        value, rebuild = _split_output(out)
        if value is None:
            # Not reproducible, so this branch computes until an output that is comes back.
            self.history.pop(key, None)
            if not self._warned_container and self.logger is not None:
                self._warned_container = True
                self.logger.info(
                    "diffusion.step_skip: %s returned a %s, which a skipped step cannot "
                    "reproduce; those calls compute",
                    type(self.module).__name__,
                    type(out).__name__,
                )
            return
        # Cloned: the pipeline may keep or modify the tensor it was handed.
        stored = _Stored(
            sig, _timestep_of(args, kwargs, self._t_slot), value.detach().clone(), rebuild
        )
        keep = 2 if self.mode == MODE_TAYLOR1 else 1
        history = self.history.setdefault(key, [])
        history.append(stored)
        del history[:-keep]

    def _skipped(self, history: list, args: tuple, kwargs: dict) -> Any:
        last = history[-1]
        value = None
        if self.mode == MODE_TAYLOR1 and len(history) >= 2:
            prev = history[-2]
            t = _timestep_of(args, kwargs, self._t_slot)
            # Only same-shape outputs combine: a prefix-KV step 0 is longer than every later step.
            if (
                prev.sig == last.sig
                and tuple(prev.value.shape) == tuple(last.value.shape)
                and None not in (t, prev.t, last.t)
            ):
                value = _extrapolate(prev.value, prev.t, last.value, last.t, t)
        if value is None:
            value = last.value.clone()
        return last.rebuild(value)


def _context_wrapper(skip: StaticStepSkip, module: Any, prior: Any) -> Any:
    """An instance ``cache_context`` that records the branch name, then delegates."""
    target = prior if prior is not None else getattr(type(module), "cache_context")

    @contextlib.contextmanager
    def cache_context(*args, **kwargs):
        name = args[0] if args else kwargs.get("name")
        previous = skip.context
        skip.context = name
        try:
            if prior is not None:
                with target(*args, **kwargs):
                    yield
            else:
                with target(module, *args, **kwargs):
                    yield
        finally:
            skip.context = previous

    cache_context._unsloth_static_skip = skip
    return cache_context


def _find(pipe: Any) -> Optional[StaticStepSkip]:
    transformer = getattr(pipe, "transformer", None)
    try:
        skip = transformer.__dict__.get(_SLOT) if transformer is not None else None
    except Exception:  # noqa: BLE001 - no instance dict
        return None
    return skip if isinstance(skip, StaticStepSkip) else None


def install_static_step_skip(
    pipe: Any,
    *,
    settings: Optional[dict] = None,
    logger: Any = None,
) -> Optional[str]:
    """Put the static step skip on ``pipe.transformer``. Returns ``TC_STATIC``, or None when the
    pipe cannot take it (it then runs uncached). Never sets ``_unsloth_step_cache``: that marker
    is what runs the CUDA graph eager, and nothing here needs it. Idempotent."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        _warn(logger, "pipeline has no transformer")
        return None
    # A second denoiser takes part of the CFG trajectory this layer would not see.
    for attr in ("transformer_2", "unconditional_transformer"):
        other = getattr(pipe, attr, None)
        if other is not None and other is not transformer:
            _warn(logger, f"pipeline runs a second denoiser ({attr})")
            return None
    if not callable(getattr(type(transformer), "forward", None)):
        _warn(logger, "transformer has no forward")
        return None
    knobs = settings if settings is not None else static_skip_settings(logger = logger)
    existing = _find(pipe)
    if existing is not None:
        existing.armed = True
        existing.mode = knobs.get("mode", existing.mode)
        existing.head = float(knobs.get("head", existing.head))
        existing.tail = float(knobs.get("tail", existing.tail))
        existing.every = int(knobs.get("every", existing.every))
        existing.reset(None)
        return TC_STATIC
    try:
        slots = transformer.__dict__
        skip = StaticStepSkip(
            transformer,
            inner = slots.get("forward"),
            mode = knobs.get("mode", DEFAULT_MODE),
            head = knobs.get("head", DEFAULT_HEAD),
            tail = knobs.get("tail", DEFAULT_TAIL),
            every = knobs.get("every", DEFAULT_EVERY),
            logger = logger,
        )
        prior_ctx = slots.get("cache_context")
        skip._prior_ctx = prior_ctx
        skip._ctx = None
        if prior_ctx is not None or callable(getattr(type(transformer), "cache_context", None)):
            skip._ctx = _context_wrapper(skip, transformer, prior_ctx)
            slots["cache_context"] = skip._ctx
        # Written into the instance dict, like GraphedForward: nn.Module.__setattr__ would inspect it.
        slots["forward"] = skip
        slots[_SLOT] = skip
    except Exception as exc:  # noqa: BLE001 - best-effort, the load proceeds uncached
        _warn(logger, str(exc))
        return None
    if logger is not None:
        logger.info(
            "diffusion.step_skip: static engaged (mode=%s head=%s tail=%s every=%s)",
            skip.mode,
            skip.head,
            skip.tail,
            skip.every,
        )
    return TC_STATIC


def reset_static_step_skip(
    pipe: Any,
    steps: Optional[int],
    *,
    step_signal: bool = False,
    keep_stats: bool = False,
) -> bool:
    """Arm the schedule for one pipeline call of ``steps`` effective denoise steps. ``keep_stats``
    adds this call's counts to the running generation's instead of starting new ones."""
    skip = _find(pipe)
    if skip is None:
        return False
    skip.reset(steps, step_signal = step_signal, keep_stats = keep_stats)
    return True


def mark_step_end(pipe: Any) -> None:
    skip = _find(pipe)
    if skip is not None:
        skip.step_end()


def static_skip_stats(pipe: Any) -> Optional[dict]:
    skip = _find(pipe)
    return skip.describe() if skip is not None else None


def uninstall_static_step_skip(pipe: Any) -> bool:
    """Take the layer off and put back what it wrapped. Idempotent.

    A GraphedForward installed after the layer is its ``inner``, so it goes back into the slot.
    When something wrapped the layer after it (an offload hook), its slot is not ours to rewrite:
    the layer disarms to a passthrough in place instead."""
    skip = _find(pipe)
    if skip is None:
        return False
    skip.armed = False
    skip.reset(None)
    try:
        slots = skip.module.__dict__
        if slots.get("forward") is skip:
            if skip.inner is None:
                slots.pop("forward", None)
            else:
                slots["forward"] = skip.inner
        ctx = getattr(skip, "_ctx", None)
        if ctx is not None and slots.get("cache_context") is ctx:
            prior = getattr(skip, "_prior_ctx", None)
            if prior is None:
                slots.pop("cache_context", None)
            else:
                slots["cache_context"] = prior
        slots.pop(_SLOT, None)
    except Exception:  # noqa: BLE001 - unload must not fail; a disarmed layer is a passthrough
        return False
    return True


def _warn(logger: Any, why: str) -> None:
    if logger is not None:
        logger.warning("diffusion.step_skip: static unavailable (%s); running uncached", why)
