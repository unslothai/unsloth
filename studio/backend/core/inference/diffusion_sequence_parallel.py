# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two-card Ulysses sequence parallel (guidance-free); bit-exact only if GEMMs keep a row's bits at
any row count; swaps match by module and call kind, not call order; plan mismatch or error keeps
one card; UNSLOTH_DIFFUSION_SEQUENCE_PARALLEL: 0 never, unset from MIN_TOKENS, force any length."""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import threading
from typing import Any, Callable, Optional

SP_ENV = "UNSLOTH_DIFFUSION_SEQUENCE_PARALLEL"
# Below this many tokens per call the two swaps per layer cost more than half the step saves (images, short clips).
MIN_TOKENS = 4096
# A card waiting longer than this on the other card's half gives up (both abort, the step reruns on one card).
WAIT_S = 600.0

# The plans this module was written and checked against: (key, kind, index or name, dim, expected_dims). A class
# whose diffusers ``_cp_plan`` reads differently is not split.
SP_PLANS: dict[str, tuple] = {
    "WanTransformer3DModel": (
        ("", "input", "timestep", 1, 2),
        ("rope", "output", 0, 1, 4),
        ("rope", "output", 1, 1, 4),
        ("blocks.0", "input", "hidden_states", 1, 3),
        ("proj_out", "gather", None, 1, 3),
    ),
}
# Cuts follow the checked diffusers plan; per-token GEMMs whose row count changes (Wan 2.2 TI2V
# timestep embedding, output proj) run on all tokens on both cards for one-card bits; kinds: input /
# output cut args / outputs, gather joins an output, gather_input joins an argument before its
# module.
SP_EXEC_PLANS: dict[str, tuple] = {
    "WanTransformer3DModel": (
        ("condition_embedder", "output", 0, 1, 3),
        ("condition_embedder", "output", 1, 1, 3),
        ("rope", "output", 0, 1, 4),
        ("rope", "output", 1, 1, 4),
        ("blocks.0", "input", "hidden_states", 1, 3),
        ("proj_out", "gather_input", 0, 1, 3),
    ),
}
# Attention processor classes whose self-attention call passes ``parallel_config`` (diffusers' context parallel
# hook point): the mark Studio sets there is how a call is told apart from cross-attention.
SP_PROCESSORS = frozenset({"WanAttnProcessor"})

STATS = {
    "forwards": 0,
    "swaps": 0,
    "bytes": 0,
    "repeat_reads": 0,
    "failures": 0,
    "last_error": None,
}

_TLS = threading.local()
_PATCH_LOCK = threading.Lock()


def mode() -> str:
    """``off`` / ``auto`` (default) / ``force``."""
    raw = (os.environ.get(SP_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "disable", "disabled"):
        return "off"
    return "force" if raw == "force" else "auto"


def _torch():
    import torch
    return torch


class _Mark:
    """Set as ``processor._parallel_config`` during a split call: names the attention module, so the two cards'
    swaps pair by module (diffusers passes it only on a self-attention call)."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"_Mark({self.name!r})"


class SequenceParallelError(RuntimeError):
    pass


def _split_sizes(n: int) -> tuple[int, int]:
    first = (int(n) + 1) // 2
    return first, int(n) - first


def _chunk(x: Any, dim: int, rank: int) -> Any:
    first, second = _split_sizes(x.shape[dim])
    start, size = (0, first) if rank == 0 else (first, second)
    return x.narrow(dim, start, size).contiguous()


def read_plan(cls: Any) -> Optional[tuple]:
    """The checked plan for ``cls`` when its diffusers ``_cp_plan`` matches ``SP_PLANS``, else None."""
    expected = SP_PLANS.get(getattr(cls, "__name__", ""))
    raw = getattr(cls, "_cp_plan", None)
    if expected is None or not isinstance(raw, dict):
        return None
    found = []
    try:
        for key, spec in raw.items():
            if isinstance(spec, dict):
                for name, inner in spec.items():
                    if isinstance(inner, (list, tuple)):
                        return None
                    if getattr(inner, "split_output", False):
                        found.append(
                            (key, "output", name, int(inner.split_dim), inner.expected_dims)
                        )
                    else:
                        found.append(
                            (key, "input", name, int(inner.split_dim), inner.expected_dims)
                        )
            elif hasattr(spec, "gather_dim"):
                found.append((key, "gather", None, int(spec.gather_dim), spec.expected_dims))
            else:
                return None
    except Exception:  # noqa: BLE001 - an unreadable plan: no split
        return None
    if sorted(found, key = repr) != sorted(expected, key = repr):
        return None
    return SP_EXEC_PLANS.get(cls.__name__, tuple(found))


def refusal(module: Any, tokens: Optional[int] = None) -> Optional[str]:
    """Why a call of ``module`` is not split across two cards, or None."""
    if mode() == "off":
        return f"{SP_ENV}=0"
    torch = _torch()
    if torch.is_grad_enabled():
        return "gradients enabled"
    if torch.compiler.is_compiling():
        return "inside a compile"
    cls = type(module)
    if read_plan(cls) is None:
        return (
            f"{cls.__name__} has no checked sequence-parallel plan"
            if cls.__name__ not in SP_PLANS
            else f"{cls.__name__}'s diffusers context-parallel plan differs from the checked one"
        )
    # Also a compile moved below an offload hook (diffusion_block_graph), which clears _compiled_call_impl.
    from .diffusion_cfg_batch import compiled_denoiser

    compiled = compiled_denoiser(module)
    if compiled is not None:
        return f"compiled submodules ({compiled})"
    for name, sub in module.named_modules():
        proc = getattr(sub, "processor", None)
        if proc is None or not hasattr(sub, "heads"):
            continue
        if type(proc).__name__ not in SP_PROCESSORS:
            return f"{name} runs {type(proc).__name__}"
        if getattr(proc, "_parallel_config", None) is not None:
            return f"{name} already has a context-parallel config"
        if int(sub.heads) % 2:
            return f"{name} has an odd head count ({sub.heads})"
    if tokens is not None and mode() != "force" and int(tokens) < MIN_TOKENS:
        return f"{tokens} tokens, under {MIN_TOKENS}"
    return None


def call_tokens(module: Any, latent: Any) -> Optional[int]:
    """Tokens per sample of a call on ``latent`` (B, C, F, H, W) after patching, or None when unreadable."""
    try:
        patch = tuple(int(p) for p in module.config.patch_size)
        f, h, w = (int(s) for s in latent.shape[-3:])
        return (f // patch[0]) * (h // patch[1]) * (w // patch[2])
    except Exception:  # noqa: BLE001
        return None


class _Store:
    """The two cards' halves for one split call, keyed by (attention name, kind, publishing rank). A publisher keeps
    its entries for its last two attention names only: the other card can be at most one name behind it."""

    def __init__(self, devices: tuple):
        self.devices = devices
        self.cond = threading.Condition()
        self.entries: dict = {}
        self.names: tuple = ([], [])
        self.failed: Optional[str] = None

    def fail(self, why: str) -> None:
        with self.cond:
            if self.failed is None:
                self.failed = why
            self.cond.notify_all()

    def publish(self, rank: int, name: str, kind: str, payload: Any) -> None:
        torch = _torch()
        key = (name, kind, rank)
        with self.cond:
            if key in self.entries:
                # a repeated call on the same inputs: the first copy stands
                STATS["repeat_reads"] += 1
                return
        src = self.devices[rank]
        dst = self.devices[1 - rank]
        moved = tuple(t.to(dst, non_blocking = True) for t in payload)
        event = None
        if getattr(src, "type", None) == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(src))
        nbytes = sum(t.numel() * t.element_size() for t in moved)
        with self.cond:
            names = self.names[rank]
            if name not in names:
                names.append(name)
                while len(names) > 2:
                    old = names.pop(0)
                    for k in [k for k in self.entries if k[0] == old and k[2] == rank]:
                        del self.entries[k]
            self.entries[key] = (moved, event)
            STATS["swaps"] += 1
            STATS["bytes"] += int(nbytes)
            self.cond.notify_all()

    def fetch(self, rank: int, name: str, kind: str) -> tuple:
        torch = _torch()
        key = (name, kind, 1 - rank)
        with self.cond:
            ok = self.cond.wait_for(
                lambda: key in self.entries or self.failed is not None, timeout = WAIT_S
            )
            if self.failed is not None:
                raise SequenceParallelError(f"the other card failed ({self.failed})")
            if not ok:
                raise SequenceParallelError(f"timed out waiting for {name} {kind}")
            moved, event = self.entries[key]
        dev = self.devices[rank]
        if event is not None and getattr(dev, "type", None) == "cuda":
            stream = torch.cuda.current_stream(dev)
            stream.wait_event(event)
            for t in moved:
                t.record_stream(stream)
        return moved


class _Rank:
    __slots__ = ("store", "rank")

    def __init__(self, store: _Store, rank: int):
        self.store = store
        self.rank = rank


@contextlib.contextmanager
def _as_rank(ctx: _Rank):
    """This thread runs ``ctx``'s half. No CUDA graph records or replays meanwhile: the other card's thread issues
    CUDA work the whole time, so a block or step graph would wait out its hold on every call, and a replay would skip
    the swaps its kernels need."""
    previous = getattr(_TLS, "ctx", None)
    _TLS.ctx = ctx
    try:
        from .diffusion_bg_compile import no_capture
    except Exception:  # noqa: BLE001 - no graph machinery: nothing records
        no_capture = contextlib.nullcontext
    try:
        with no_capture():
            yield
    finally:
        _TLS.ctx = previous


def _arg(args: tuple, kwargs: dict, index: int, name: str) -> Any:
    return args[index] if len(args) > index else kwargs.get(name)


def _ulysses(ctx: _Rank, name: str, inner: Callable, args: tuple, kwargs: dict) -> Any:
    """Self-attention over both cards' tokens for half of the heads; returns this card's tokens, every head."""
    torch = _torch()
    q, k, v = (_arg(args, kwargs, i, n) for i, n in enumerate(("query", "key", "value")))
    if not all(torch.is_tensor(t) and t.dim() == 4 for t in (q, k, v)):
        raise SequenceParallelError(f"{name}: attention inputs are not [B, S, H, D]")
    if _arg(args, kwargs, 3, "attn_mask") is not None:
        raise SequenceParallelError(f"{name}: a masked self-attention cannot be split by heads")
    heads = int(q.shape[2])
    if heads % 2 or int(k.shape[2]) != heads or int(v.shape[2]) != heads:
        raise SequenceParallelError(f"{name}: {heads} query heads cannot be halved")
    r = ctx.rank
    half = heads // 2
    own = slice(r * half, (r + 1) * half)
    other = slice((1 - r) * half, (2 - r) * half)
    same = tuple(q.shape) == tuple(k.shape) == tuple(v.shape)
    if same:
        payload = (torch.stack((q[:, :, other], k[:, :, other], v[:, :, other])),)
    else:
        payload = (
            q[:, :, other].contiguous(),
            torch.stack((k[:, :, other], v[:, :, other])),
        )
    ctx.store.publish(r, name, "qkv", payload)
    got = ctx.store.fetch(r, name, "qkv")
    if same:
        pq, pk, pv = got[0].unbind(0)
    else:
        pq = got[0]
        pk, pv = got[1].unbind(0)

    def join(mine: Any, theirs: Any) -> Any:
        return torch.cat((mine, theirs) if r == 0 else (theirs, mine), dim = 1)

    full = (join(q[:, :, own], pq), join(k[:, :, own], pk), join(v[:, :, own], pv))
    rest = tuple(args[3:]) if len(args) > 3 else ()
    kw = {key: val for key, val in kwargs.items() if key not in ("query", "key", "value")}
    out = inner(*full, *rest, **kw)
    mine_n = int(q.shape[1])
    first = mine_n if r == 0 else int(pq.shape[1])
    if r == 0:
        mine_tok, their_tok = out[:, :first], out[:, first:]
    else:
        their_tok, mine_tok = out[:, :first], out[:, first:]
    ctx.store.publish(r, name, "out", (their_tok.contiguous(),))
    (theirs,) = ctx.store.fetch(r, name, "out")
    return torch.cat((mine_tok, theirs) if r == 0 else (theirs, mine_tok), dim = 2)


def _wrap_dispatch(inner: Callable) -> Callable:
    def dispatch_attention_fn(*args: Any, **kwargs: Any) -> Any:
        mark = kwargs.get("parallel_config")
        if isinstance(mark, _Mark):
            kwargs["parallel_config"] = None
            ctx = getattr(_TLS, "ctx", None)
            if ctx is not None:
                return _ulysses(ctx, mark.name, inner, args, kwargs)
        return inner(*args, **kwargs)

    dispatch_attention_fn._unsloth_sequence_parallel = True  # type: ignore[attr-defined]
    dispatch_attention_fn.__wrapped__ = inner  # type: ignore[attr-defined]
    return dispatch_attention_fn


def _dispatch_modules(modules: tuple) -> list:
    names = set()
    for module in modules:
        for sub in module.modules():
            proc = getattr(sub, "processor", None)
            if proc is not None and type(proc).__name__ in SP_PROCESSORS:
                names.add(type(proc).__module__)
    return [sys.modules[n] for n in sorted(names) if n in sys.modules]


@contextlib.contextmanager
def _patched(modules: tuple):
    """Marks every split-capable processor of ``modules`` and wraps its module's ``dispatch_attention_fn`` for the
    duration of one split call; both are put back after it."""
    marks = []
    for module in modules:
        for name, sub in module.named_modules():
            proc = getattr(sub, "processor", None)
            if proc is not None and type(proc).__name__ in SP_PROCESSORS:
                marks.append((proc, proc.__dict__.get("_parallel_config", _UNSET)))
                proc._parallel_config = _Mark(name)
    swapped = []
    with _PATCH_LOCK:
        for mod in _dispatch_modules(modules):
            current = getattr(mod, "dispatch_attention_fn", None)
            if current is None or getattr(current, "_unsloth_sequence_parallel", False):
                continue
            wrapper = _wrap_dispatch(current)
            mod.dispatch_attention_fn = wrapper
            swapped.append((mod, current, wrapper))
    try:
        yield
    finally:
        with _PATCH_LOCK:
            for mod, current, wrapper in swapped:
                if getattr(mod, "dispatch_attention_fn", None) is wrapper:
                    mod.dispatch_attention_fn = current
        for proc, previous in marks:
            if previous is _UNSET:
                proc.__dict__.pop("_parallel_config", None)
            else:
                proc._parallel_config = previous


_UNSET = object()


def _param_index(fn: Any, name: Any) -> Optional[int]:
    if isinstance(name, int):
        return name
    try:
        params = [p for p in inspect.signature(fn).parameters.values() if p.name != "self"]
    except (TypeError, ValueError):
        return None
    for i, p in enumerate(params):
        if p.name == name and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            return i
    return None


def _split_input(fn: Any, args: tuple, kwargs: dict, name: str, dim: int, dims: Any, rank: int):
    """``(args, kwargs)`` with input ``name`` cut to this rank's tokens (unchanged when absent or of another rank)."""
    torch = _torch()
    if name in kwargs:
        value = kwargs[name]
        if torch.is_tensor(value) and (dims is None or value.dim() == dims):
            kwargs = {**kwargs, name: _chunk(value, dim, rank)}
        return args, kwargs
    index = _param_index(fn, name)
    if index is not None and index < len(args):
        value = args[index]
        if torch.is_tensor(value) and (dims is None or value.dim() == dims):
            args = (*args[:index], _chunk(value, dim, rank), *args[index + 1 :])
    return args, kwargs


def _install_hooks(module: Any, plan: tuple, rank: int, store: _Store, engaged: dict) -> list:
    handles = []
    subs = dict(module.named_modules())
    for key, kind, name, dim, dims in plan:
        if key == "":
            continue
        sub = subs.get(key)
        if sub is None:
            raise SequenceParallelError(f"{type(module).__name__} has no submodule {key}")
        if kind == "input":

            def pre(
                mod,
                args,
                kwargs,
                _n = name,
                _d = dim,
                _x = dims,
                _k = key,
            ):
                engaged[_k] = engaged.get(_k, 0) + 1
                return _split_input(type(mod).forward, args, kwargs, _n, _d, _x, rank)

            handles.append(sub.register_forward_pre_hook(pre, with_kwargs = True))
        elif kind == "output":

            def post(
                mod,
                args,
                output,
                _n = name,
                _d = dim,
                _x = dims,
                _k = key,
            ):
                engaged[(_k, _n)] = engaged.get((_k, _n), 0) + 1
                seq = list(output) if isinstance(output, (tuple, list)) else None
                if seq is None or not isinstance(_n, int) or _n >= len(seq):
                    raise SequenceParallelError(f"{_k} output {_n} is not in its output")
                value = seq[_n]
                if _x is None or value.dim() == _x:
                    seq[_n] = _chunk(value, _d, rank)
                return type(output)(seq)

            handles.append(sub.register_forward_hook(post))
        elif kind == "gather_input":

            def gather_in(
                mod,
                args,
                kwargs,
                _n = name,
                _d = dim,
                _x = dims,
                _k = key,
            ):
                torch = _torch()
                engaged[_k] = engaged.get(_k, 0) + 1
                index = _param_index(type(mod).forward, _n)
                if _n in kwargs:
                    value = kwargs[_n]
                elif index is not None and index < len(args):
                    value = args[index]
                else:
                    raise SequenceParallelError(f"{_k} has no input {_n}")
                if not torch.is_tensor(value) or (_x is not None and value.dim() != _x):
                    raise SequenceParallelError(f"{_k} input {_n} is not a {_x}-dim tensor")
                store.publish(rank, f"gather:{_k}", "in", (value.contiguous(),))
                (theirs,) = store.fetch(rank, f"gather:{_k}", "in")
                full = torch.cat((value, theirs) if rank == 0 else (theirs, value), dim = _d)
                if _n in kwargs:
                    return args, {**kwargs, _n: full}
                return (*args[:index], full, *args[index + 1 :]), kwargs

            handles.append(sub.register_forward_pre_hook(gather_in, with_kwargs = True))
        else:

            def gather(
                mod,
                args,
                output,
                _d = dim,
                _k = key,
            ):
                torch = _torch()
                engaged[_k] = engaged.get(_k, 0) + 1
                if not torch.is_tensor(output):
                    raise SequenceParallelError(f"{_k} output is not a tensor")
                store.publish(rank, f"gather:{_k}", "out", (output.contiguous(),))
                (theirs,) = store.fetch(rank, f"gather:{_k}", "out")
                pair = (output, theirs) if rank == 0 else (theirs, output)
                return torch.cat(pair, dim = _d)

            handles.append(sub.register_forward_hook(gather))
    return handles


def run(
    module: Any,
    replica: Any,
    submit: Callable[[Callable[[], Any]], Any],
    wait: Callable[[Any], Any],
    *,
    primary_forward: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    replica_args: tuple,
    replica_kwargs: dict,
    devices: tuple,
) -> Any:
    """One call of ``module`` split over two cards: ``primary_forward(*args, **kwargs)`` on the caller's thread as
    rank 0, ``replica(*replica_args, **replica_kwargs)`` through ``submit`` as rank 1. Returns rank 0's output (the
    whole call's). Raises :class:`SequenceParallelError` (or the card's own error) when either rank fails."""
    plan = read_plan(type(module))
    if plan is None or read_plan(type(replica)) is None:
        raise SequenceParallelError("no checked plan")
    store = _Store(tuple(devices))
    root = [p for p in plan if p[0] == ""]
    forward = type(module).forward

    def cut(a: tuple, k: dict, rank: int) -> tuple:
        for _key, _kind, name, dim, dims in root:
            a, k = _split_input(forward, a, k, name, dim, dims, rank)
        return a, k

    a0, k0 = cut(args, kwargs, 0)
    a1, k1 = cut(replica_args, replica_kwargs, 1)
    engaged: tuple = ({}, {})
    handles = []
    try:
        handles += _install_hooks(module, plan, 0, store, engaged[0])
        handles += _install_hooks(replica, plan, 1, store, engaged[1])
        with _patched((module, replica)):

            def second() -> Any:
                try:
                    with _as_rank(_Rank(store, 1)):
                        return replica(*a1, **k1)
                except BaseException as exc:
                    store.fail(f"{type(exc).__name__}: {exc}")
                    raise

            job = submit(second)
            try:
                with _as_rank(_Rank(store, 0)):
                    out = primary_forward(*a0, **k0)
            except BaseException as exc:
                store.fail(f"{type(exc).__name__}: {exc}")
                with contextlib.suppress(BaseException):
                    wait(job)
                raise
            wait(job)
    except BaseException as exc:
        STATS["failures"] += 1
        STATS["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        for h in handles:
            h.remove()
    needed = {p[0] if p[1] != "output" else (p[0], p[2]) for p in plan if p[0] != ""}
    for rank in (0, 1):
        missing = sorted(map(str, needed - set(engaged[rank])))
        if missing:
            STATS["failures"] += 1
            raise SequenceParallelError(f"rank {rank} never reached {', '.join(missing)}")
    STATS["forwards"] += 1
    return out


def status() -> dict:
    return {"mode": mode(), **STATS}
