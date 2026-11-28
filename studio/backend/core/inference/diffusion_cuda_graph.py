# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static-buffer CUDA graph of a denoiser forward, kept in instance __dict__; not a proxy or hook."""

from __future__ import annotations

import collections
import contextlib
import contextvars
import gc
import inspect
import os
import sys
import threading
import traceback
import weakref
import types
from functools import WRAPPER_ASSIGNMENTS, update_wrapper
from typing import Any, Optional

from .diffusion_bg_compile import capture_suppressed as _bg_capture_suppressed
from .diffusion_bg_compile import eager_forced as _bg_eager_forced

BG_PLANNED_REPLAY_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_PLANNED"


def _bg_planned_replay_enabled() -> bool:
    return (os.environ.get(BG_PLANNED_REPLAY_ENV) or "").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


CUDA_GRAPH_DISABLE_ENV = "UNSLOTH_DISABLE_CUDA_GRAPH"
# Master switch for every diffusion CUDA graph (whole-forward and per-block): "0" turns them all off.
CUDA_GRAPHS_ENV = "UNSLOTH_DIFFUSION_CUDA_GRAPHS"
# =0: an offloaded denoiser never records its whole step with the copies inside (per-block graphs still may).
OFFLOAD_CUDA_GRAPH_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_CUDA_GRAPH"
# =0: a block-streamed whole-step graph records its copies into the graph pool instead of the prefetcher's slot ring.
STEP_SLOTS_ENV = "UNSLOTH_DIFFUSION_STEP_GRAPH_SLOTS"
MODEL_OFFLOAD_CUDA_GRAPH_ENV = "UNSLOTH_DIFFUSION_MODEL_OFFLOAD_CUDA_GRAPH"
# =1 adds stock onloads (costly re-records); default: stable range (diffusion_stable_arena).
MODEL_OFFLOAD_REASON = (
    "model offload re-uploads the denoiser every render at new addresses, so its step graph would record again on "
    "every render; set " + MODEL_OFFLOAD_CUDA_GRAPH_ENV + "=1 to record it anyway"
)

# Per module. Legitimate second keys exist, but each costs a pool-sized slice of VRAM.
MAX_GRAPHS_PER_MODULE = 4

# Calls that carry several requests' images (image_request_batching): only these sizes record a graph of their own,
# next to the size-1 graphs, at most MAX_BATCHED_GRAPHS per module, and only where the caller allows a recording (an
# idle warm-up, never a user's request). A full cache never evicts for them; an OOM recording one drops only them.
BATCH_GRAPH_SIZES = (2, 4)
MAX_BATCHED_GRAPHS = 2
# Extra pool a batched graph may take, in MiB; unset = what the card has free beyond BATCH_GRAPH_MARGIN_MB.
BATCH_GRAPH_POOL_ENV = "UNSLOTH_DIFFUSION_BATCH_GRAPH_POOL_MB"
BATCH_GRAPH_MARGIN_MB = 1024
_BATCH_ROWS: "contextvars.ContextVar[Optional[tuple]]" = contextvars.ContextVar(
    "unsloth_graph_batch_rows", default = None
)


_BATCH_CANVAS: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "unsloth_graph_batch_canvas", default = None
)


@contextlib.contextmanager
def batch_rows(
    rows: int,
    *,
    capture: bool = False,
    canvas: Optional[str] = None,
):
    """Calls in this context carry ``rows`` requests; ``capture`` allows a new batched graph to record. ``canvas`` (the
    batch race's shape token) labels every batched graph this context records or replays, so a verdict raced at one
    canvas is never satisfied by a same-size graph of another (graphs key on tensor shapes)."""
    token = _BATCH_ROWS.set((int(rows), bool(capture)))
    canvas_token = _BATCH_CANVAS.set(None if canvas is None else str(canvas))
    try:
        yield
    finally:
        _BATCH_CANVAS.reset(canvas_token)
        _BATCH_ROWS.reset(token)


def current_batch_rows() -> Optional[tuple]:
    return _BATCH_ROWS.get()


_FORCED_EAGER: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "unsloth_graph_forced_eager", default = False
)


@contextlib.contextmanager
def forced_eager():
    """Every graph handle runs eager in this context, whatever a caller's ``set_bypass(False)`` says (the batch race's
    eager arm: generate() re-arms the graphs before each chunk)."""
    token = _FORCED_EAGER.set(True)
    try:
        yield
    finally:
        _FORCED_EAGER.reset(token)


def eager_forced() -> bool:
    return bool(_FORCED_EAGER.get())


def _batch_pool_budget_bytes() -> Optional[int]:
    raw = (os.environ.get(BATCH_GRAPH_POOL_ENV) or "").strip()
    if not raw:
        return None
    try:
        return max(0, int(float(raw) * 2**20))
    except ValueError:
        return None


def _free_device_bytes(index: Optional[int] = None) -> Optional[int]:
    """Free bytes on card ``index`` (the call's operands), else the current card."""
    try:
        torch = _torch()
        free, _total = (
            torch.cuda.mem_get_info() if index is None else torch.cuda.mem_get_info(index)
        )
        return int(free)
    except Exception:  # noqa: BLE001 - unreadable: the recording's own OOM path still guards
        return None


def _has_reference(kwargs: Any) -> bool:
    """A call carrying edit reference states (an edit's image scope), as opposed to plain text-to-image."""
    return isinstance(kwargs, dict) and any(
        kwargs.get(name) is not None for name in _REFERENCE_KWARGS
    )


def _is_oom(exc: BaseException) -> bool:
    try:
        from .diffusion_batched import is_oom_error
        return bool(is_oom_error(exc))
    except Exception:  # noqa: BLE001
        return "out of memory" in str(exc).lower()


# Three per the PyTorch docs. The warmup forces lazy allocations OUTSIDE the capture.
WARMUP_ITERS = 3
# A resident key's first real steps are its warm-ups, so no forward is run and thrown away before the recording (three
# per key on the first render of a load); its WARMUP_ITERS-th + 1 step records. 0 keeps the side-stream warm-ups.
# Only where the render runs the key on at least WARMUP_ITERS + 1 steps (note_render_steps: a render of more steps than
# WARMUP_ITERS, one denoiser, a key that follows no step-dependent branch); any other key records on its first call
# with side-stream warm-ups, as before, so a 1-3 step render replays from the next render on.
# Exact while no first-call decision changes an output after a used step: INT8 attention keeps the sink path a used
# call ran on for the load (diffusion_int8_attention._choose_path), NVFP4 tunes before the first step (below).
REAL_WARMUP_ENV = "UNSLOTH_DIFFUSION_GRAPH_REAL_WARMUP"


def real_warmup_enabled() -> bool:
    return (os.environ.get(REAL_WARMUP_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


_TRUE_TOKENS = ("1", "true", "yes", "on")


def _torch():
    """The torch module, imported lazily so this file imports on a torch-free host."""
    import torch
    return torch


# Comma list of family names (or "all") armed whatever the family declares: for measuring a family before it opts in.
FORCE_FAMILIES_ENV = "UNSLOTH_DIFFUSION_CUDA_GRAPH_FAMILIES"
# A module attribute: zero-arg callable returning hashable per-call state that the forward branches on.
GRAPH_KEY_EXTRA_ATTR = "_unsloth_graph_key_extra"
# On an exception that hook raises to decline a call: the status reason (else the exception type names it).
GRAPH_DECLINE_ATTR = "graph_decline"
# =1: a denoiser whose text length follows the prompt records every new prompt length, on any render (the old way).
EVERY_LENGTH_ENV = "UNSLOTH_DIFFUSION_CUDA_GRAPH_EVERY_LENGTH"


def _every_length() -> bool:
    return (os.environ.get(EVERY_LENGTH_ENV) or "").strip().lower() in _TRUE_TOKENS


def _length_varies(module: Any) -> bool:
    """Whether ``module``'s graphs key on the prompt length (diffusion_dynamic_text.PROMPT_LENGTH_POLICY)."""
    if _every_length():
        return False
    try:
        from .diffusion_dynamic_text import prompt_length_varies  # noqa: PLC0415
        return bool(prompt_length_varies(module))
    except Exception:  # noqa: BLE001 - an unanswerable probe keeps today's graphs
        return False


_REFERENCE_KWARGS = ("image_cond_hidden_states", "image_cond_ids")


def _image_scope(args: tuple, kwargs: dict) -> tuple:
    """Latents key plus reference images: the call's non-text shape (prompt length excluded)."""
    first = args[0] if args else next(iter(kwargs.values()), None)
    refs = tuple((name, graph_key(kwargs[name])) for name in _REFERENCE_KWARGS if name in kwargs)
    return (graph_key(first),) + refs if refs else graph_key(first)


def _family_forced(family: Any) -> bool:
    raw = (os.environ.get(FORCE_FAMILIES_ENV) or "").strip().lower()
    if not raw:
        return False
    names = {n.strip() for n in raw.split(",") if n.strip()}
    return "all" in names or str(getattr(family, "name", "")).lower() in names


def offload_graphs_enabled() -> bool:
    return (os.environ.get(OFFLOAD_CUDA_GRAPH_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def model_offload_graphs_requested() -> bool:
    return (os.environ.get(MODEL_OFFLOAD_CUDA_GRAPH_ENV) or "").strip().lower() in _TRUE_TOKENS


def cuda_graph_disabled() -> bool:
    """Whether an env kill switch is set. Read before any torch or pipe inspection."""
    if os.environ.get(CUDA_GRAPHS_ENV, "").strip().lower() in ("0", "off", "false", "no"):
        return True
    return os.environ.get(CUDA_GRAPH_DISABLE_ENV, "").strip().lower() in _TRUE_TOKENS


def _disabled_reason() -> str:
    if os.environ.get(CUDA_GRAPHS_ENV, "").strip().lower() in ("0", "off", "false", "no"):
        return f"disabled by {CUDA_GRAPHS_ENV}=0"
    return f"disabled by {CUDA_GRAPH_DISABLE_ENV}"


# ``thread_local`` capture mode: global mode lets any thread's unsafe call break the recording. A device sync or a
# second capture still breaks it, so captures are mutually exclusive and syncs use ``synchronize_outside_capture``.
# ``hold_off_capture`` holds the lock ACROSS the call (a flag read first is a TOCTOU); depth is raised under the lock
# and is a count because one thread's captures can nest.
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_DEPTH = 0
# CFG-parallel replica work is counted from queueing until its inputs drop; a capture waits for the count to drain.
_CAPTURE_IDLE = threading.Condition(_CAPTURE_LOCK)
_OUTSIDE_JOBS = 0


class _ThreadGateState(threading.local):
    """Per-thread gate counters; class defaults keep hot-path reads cheap."""

    depth = 0
    recording = 0
    lock_held = 0
    device_held = 0
    device_card = None


# Single-card work (deferred int8 CUTLASS race) holds only its card; a None card blocks all.
_DEVICE_HOLDS: dict = {}
_DEVICE_DEPTH: dict = {}


_OUTSIDE_LOCAL = _ThreadGateState()
# A count that never drains (dead worker) must not hang captures: after this long the capture is refused and the step
# runs eager. Symmetrically, replica work that waits this long on a recording is refused and its branch runs on the source.
OUTSIDE_WAIT_S = 120.0
# Set by a refusal, cleared once the replica-side count drains: while it stands, captures refuse without waiting again.
_OUTSIDE_STUCK = [False]
# Holds of threads parked in _capturing make no CUDA calls, so other held threads may record.
_OUTSIDE_PARKED = [0]


def _graph_unsafe(module: Any) -> Optional[str]:
    """Why ``module`` must never be recorded (``_unsloth_cuda_graph_unsafe``, set by a per-step route), or None."""
    reason = (getattr(module, "__dict__", None) or {}).get("_unsloth_cuda_graph_unsafe")
    return str(reason) if reason else None


class CaptureExclusionTimeout(RuntimeError):
    """A capture or a replica-side job gave up waiting on the other side of the capture exclusion."""


class CaptureRefused(CaptureExclusionTimeout):
    """Raised by ``_capturing`` BEFORE the depth is raised and before any recording begins: nothing to clean up."""


class OutsideWorkRefused(CaptureExclusionTimeout):
    """Raised by ``held_work`` before the replica-side work runs: none of its CUDA calls were made."""


def _other_holds(own: int, parked: int) -> int:
    """Replica holds to wait for, less own held work and (if own) parked holds of others."""
    return _OUTSIDE_JOBS - own - (parked if own else 0)


def capture_refused_now() -> bool:
    """Whether a capture would be refused without waiting (a stale replica-side count is still standing)."""
    with _CAPTURE_IDLE:
        own = getattr(_OUTSIDE_LOCAL, "depth", 0)
        return _OUTSIDE_STUCK[0] and _other_holds(own, _OUTSIDE_PARKED[0]) > 0


def _wait_outside_drained(own: int) -> None:
    """Under ``_CAPTURE_IDLE``: wait for other threads' held work to drain, else raise ``CaptureRefused``."""
    drained = _other_holds(own, _OUTSIDE_PARKED[0]) <= 0
    if not drained and not _OUTSIDE_STUCK[0]:
        if own:
            _OUTSIDE_PARKED[0] += own
            _CAPTURE_IDLE.notify_all()
        try:
            drained = _CAPTURE_IDLE.wait_for(
                lambda: _other_holds(own, _OUTSIDE_PARKED[0] - own) <= 0,
                timeout = OUTSIDE_WAIT_S,
            )
        finally:
            if own:
                _OUTSIDE_PARKED[0] -= own
                _CAPTURE_IDLE.notify_all()
    if not drained:
        _OUTSIDE_STUCK[0] = True
        raise CaptureRefused(
            f"replica-side CUDA work still running after {OUTSIDE_WAIT_S:g}s; capture refused"
        )


def await_outside_drained() -> None:
    """Wait as a capture does, for the warm-ups: they may compile, and up to torch 2.13 FX tracing is flagged process-wide."""
    own = getattr(_OUTSIDE_LOCAL, "depth", 0)
    with _CAPTURE_IDLE:
        _wait_outside_drained(own)


def _capture_card(device: Any) -> Optional[int]:
    """The card a capture records on: ``device`` if given, else the current one; None when unreadable."""
    try:
        if device is None:
            return int(_torch().cuda.current_device())
        index = getattr(device, "index", device)
        return None if index is None else int(index)
    except Exception:  # noqa: BLE001
        return None


def _device_holds(card: Optional[int]) -> int:
    """Under ``_CAPTURE_IDLE``: other threads' single-card holds a capture on ``card`` must wait for (this thread's
    own hold is discounted: its capture runs serially with the held work)."""
    held = sum(_DEVICE_HOLDS.values()) if card is None else _DEVICE_HOLDS.get(card, 0)
    own = _OUTSIDE_LOCAL.device_held
    if own and (card is None or card == _OUTSIDE_LOCAL.device_card):
        held -= own
    return held


def _wait_device_holds(card: Optional[int]) -> None:
    """Under ``_CAPTURE_IDLE``: wait for the single-card work on ``card``, else raise ``CaptureRefused``."""
    if _device_holds(card) <= 0:
        return
    if not _CAPTURE_IDLE.wait_for(lambda: _device_holds(card) <= 0, timeout = OUTSIDE_WAIT_S):
        raise CaptureRefused(
            f"single-card CUDA work on cuda:{card} still running after {OUTSIDE_WAIT_S:g}s; capture deferred"
        )


@contextlib.contextmanager
def _capturing(device: Any = None):
    """Depth must rise before torch.cuda.graph is entered so this stays outermost; raises
    CaptureRefused (depth untouched) if replica work did not drain; every capture settles retired
    pools; device (default current) waits out only its card's hold_device_off_capture."""
    global _CAPTURE_DEPTH
    own = getattr(_OUTSIDE_LOCAL, "depth", 0)
    mine = getattr(_OUTSIDE_LOCAL, "recording", 0)
    card = _capture_card(device)
    seen = _retired_pools_seen()
    with _CAPTURE_IDLE:
        for _ in range(_ENTRY_SETTLE_TRIES):
            _wait_recordings_ended(mine)
            if seen is not None:
                _settle_retired_pools(seen)
            _wait_outside_drained(own)
            _wait_device_holds(card)
            if _CAPTURE_DEPTH - mine > 0 or _device_holds(card) > 0:
                continue  # a capture or a card hold entered while this one waited (the lock was released): again
            # Retirement takes _RETIRE_LOCK, so one is either counted before this check (flush again, still under
            # the capture lock) or lands once the depth is up (due for the next capture): never in between.
            with _RETIRE_LOCK:
                if _RETIRED_POOLS[1] >= _RETIRED_POOLS[0]:
                    _CAPTURE_DEPTH += 1
                    _DEVICE_DEPTH[card] = _DEVICE_DEPTH.get(card, 0) + 1
                    break
                seen = _RETIRED_POOLS[0]
        else:
            raise CaptureRefused(
                "graph pools kept retiring (or captures kept entering) while this capture entered; capture deferred"
            )
    _OUTSIDE_LOCAL.recording = mine + 1
    try:
        yield
    finally:
        _OUTSIDE_LOCAL.recording = mine
        with _CAPTURE_IDLE:
            _CAPTURE_DEPTH -= 1
            left = _DEVICE_DEPTH.get(card, 0) - 1
            if left > 0:
                _DEVICE_DEPTH[card] = left
            else:
                _DEVICE_DEPTH.pop(card, None)
            _CAPTURE_IDLE.notify_all()
        if _DEFERRED_RELEASES and not mine:
            drain_deferred_releases()


def _wait_recordings_ended(mine: int) -> None:
    """Under ``_CAPTURE_IDLE``: wait for other threads' recordings, else raise ``CaptureRefused``."""
    if _CAPTURE_DEPTH - mine <= 0:
        return
    if not _CAPTURE_IDLE.wait_for(lambda: _CAPTURE_DEPTH - mine <= 0, timeout = RETIRED_FLUSH_WAIT_S):
        raise CaptureRefused(
            f"another thread's CUDA graph capture still recording after {RETIRED_FLUSH_WAIT_S:g}s; capture deferred"
        )


def recording_here() -> bool:
    return getattr(_OUTSIDE_LOCAL, "recording", 0) > 0


def device_held_here() -> bool:
    """Whether THIS thread is inside ``hold_device_off_capture`` (only its card is held off captures)."""
    return _OUTSIDE_LOCAL.device_held > 0


def _sync_device(torch: Any, device: Any) -> None:
    if device is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device)


def synchronize_outside_capture(device: Any = None, *, refuse: bool = False) -> None:
    """Device sync that waits out another thread's recording and holds captures off while it runs.

    After ``RETIRED_FLUSH_WAIT_S`` it raises ``CaptureRefused`` if ``refuse`` (nothing recorded yet), else syncs anyway.
    Under ``hold_off_capture`` / ``hold_device_off_capture`` (whose card no capture records on) or this thread's own
    recording it syncs at once, so it never self-deadlocks.
    Not for finalizers: one can run while its thread holds the (non-reentrant) capture lock."""
    torch = _torch()
    if _OUTSIDE_LOCAL.recording or _OUTSIDE_LOCAL.lock_held or _OUTSIDE_LOCAL.device_held:
        _sync_device(torch, device)
        return
    global _OUTSIDE_JOBS
    with _CAPTURE_IDLE:
        # Check and count atomically, so no capture slips in between.
        ended = _CAPTURE_DEPTH == 0 or _CAPTURE_IDLE.wait_for(
            lambda: _CAPTURE_DEPTH == 0, timeout = RETIRED_FLUSH_WAIT_S
        )
        if not ended and refuse:
            raise CaptureRefused(
                f"another thread's CUDA graph capture still recording after {RETIRED_FLUSH_WAIT_S:g}s; capture deferred"
            )
        _OUTSIDE_JOBS += 1
    try:
        _sync_device(torch, device)
    finally:
        capture_release()
    drain_deferred_releases()


# Releases a finalizer could not run at once (the capture lock was taken, possibly by its own thread, or a capture was
# recording). A deque: append / popleft run no Python code, so a finalizer firing mid-call cannot re-enter them.
_DEFERRED_RELEASES: "collections.deque" = collections.deque()


def run_if_capture_idle(fn: Any) -> bool:
    """Runs fn under the capture lock unless recording (then False); never blocks, as finalizers may
    run under the non-reentrant lock; fn may sync and free memory: no capture can enter."""
    if _OUTSIDE_LOCAL.recording:
        return False
    if _OUTSIDE_LOCAL.lock_held:
        # Under this thread's own hold_off_capture: the lock is held here, so the depth cannot change under fn.
        if _CAPTURE_DEPTH != 0:
            return False
        fn()
        return True
    if not _CAPTURE_LOCK.acquire(blocking = False):
        return False
    try:
        if _CAPTURE_DEPTH != 0:
            return False
        fn()
        return True
    finally:
        _CAPTURE_LOCK.release()


def release_when_capture_idle(fn: Any) -> None:
    """For finalizers: run ``fn`` as ``run_if_capture_idle`` does, else queue it for the next gated point (an explicit
    ``synchronize_outside_capture`` or a capture's exit). Never blocks."""
    if not run_if_capture_idle(fn):
        _DEFERRED_RELEASES.append(fn)


def drain_deferred_releases() -> None:
    """Run the queued finalizer releases the gate allows now; the rest stay queued. Never blocks."""
    for _ in range(len(_DEFERRED_RELEASES)):
        try:
            fn = _DEFERRED_RELEASES.popleft()
        except IndexError:
            return
        try:
            done = run_if_capture_idle(fn)
        except Exception:  # noqa: BLE001 - a failed release is dropped, as the finalizer would have
            continue
        if not done:
            _DEFERRED_RELEASES.append(fn)
            return


def capture_hold() -> None:
    global _OUTSIDE_JOBS
    with _CAPTURE_IDLE:
        _OUTSIDE_JOBS += 1


def capture_release() -> None:
    global _OUTSIDE_JOBS
    with _CAPTURE_IDLE:
        _OUTSIDE_JOBS = max(0, _OUTSIDE_JOBS - 1)
        if _OUTSIDE_JOBS == 0:
            _OUTSIDE_STUCK[0] = False
        _CAPTURE_IDLE.notify_all()


@contextlib.contextmanager
def held_work():
    """Waits for another thread's recording; this thread may then record without waiting on its own hold.

    Raises ``OutsideWorkRefused`` (the body never runs) when a recording is still going after ``OUTSIDE_WAIT_S``."""
    with _CAPTURE_IDLE:
        if not _CAPTURE_IDLE.wait_for(lambda: _CAPTURE_DEPTH == 0, timeout = OUTSIDE_WAIT_S):
            raise OutsideWorkRefused(
                f"a CUDA graph capture still recording after {OUTSIDE_WAIT_S:g}s; replica work refused"
            )
    _OUTSIDE_LOCAL.depth = getattr(_OUTSIDE_LOCAL, "depth", 0) + 1
    try:
        yield
    finally:
        _OUTSIDE_LOCAL.depth -= 1


@contextlib.contextmanager
def outside_capture():
    capture_hold()
    try:
        with held_work():
            yield
    finally:
        capture_release()


@contextlib.contextmanager
def hold_off_capture():
    """Non-blocking acquire: yields True while CUDA calls are safe; on False a capture is recording or entering, so the caller must skip its CUDA call."""
    if not _CAPTURE_LOCK.acquire(blocking = False):
        yield False
        return
    held = getattr(_OUTSIDE_LOCAL, "lock_held", 0)
    _OUTSIDE_LOCAL.lock_held = (
        held + 1
    )  # synchronize_outside_capture must not re-take the lock under it
    try:
        yield _CAPTURE_DEPTH == 0
    finally:
        _OUTSIDE_LOCAL.lock_held = held
        _CAPTURE_LOCK.release()


@contextlib.contextmanager
def hold_device_off_capture(index: int):
    """``hold_off_capture`` for long work on one card (the deferred int8 CUTLASS timing race): yields True while no
    capture records or enters on card ``index``, without holding the process-wide lock, so a capture on another card
    records meanwhile; a capture on this card waits for it (``OUTSIDE_WAIT_S``, then runs eager). On False a capture
    is recording on this card (or one whose card is unknown): skip the CUDA work. The body may launch and sync on
    ``index`` only; a process-wide call (``empty_cache``) goes through ``run_if_capture_idle``. Never blocks."""
    card = int(index)
    if not _CAPTURE_LOCK.acquire(blocking = False):
        yield False
        return
    try:
        safe = not _DEVICE_DEPTH.get(card, 0) and not _DEVICE_DEPTH.get(None, 0)
        if safe:
            _DEVICE_HOLDS[card] = _DEVICE_HOLDS.get(card, 0) + 1
    finally:
        _CAPTURE_LOCK.release()
    if not safe:
        yield False
        return
    held, held_card = _OUTSIDE_LOCAL.device_held, _OUTSIDE_LOCAL.device_card
    _OUTSIDE_LOCAL.device_held = held + 1
    _OUTSIDE_LOCAL.device_card = card
    try:
        yield True
    finally:
        _OUTSIDE_LOCAL.device_held = held
        _OUTSIDE_LOCAL.device_card = held_card
        with _CAPTURE_IDLE:
            left = _DEVICE_HOLDS.get(card, 0) - 1
            if left > 0:
                _DEVICE_HOLDS[card] = left
            else:
                _DEVICE_HOLDS.pop(card, None)
            _CAPTURE_IDLE.notify_all()


# Retired pools stay cached until empty_cache, which cannot run mid-capture (OOM risk): no capture
# records until earlier retirements are flushed; counted, so mid-flush retirements stay due.
_RETIRE_LOCK = threading.Lock()
_RETIRED_POOLS = [0, 0]  # [retirements counted, retirements flushed]
_POOL_FLUSHES = [0]  # retired-pool flushes performed (diagnostics, tests)
# A deferred release owes a cuBLAS workspace clear; the next entry flush does it.
_CUBLAS_CLEAR_DUE = [False]
# Wait (not eager fallback) for another thread's recording to end: eager CUDA work mid-capture can invalidate it.
RETIRED_FLUSH_WAIT_S = OUTSIDE_WAIT_S
# Entry flush attempts before the capture is refused.
_ENTRY_SETTLE_TRIES = 8


def note_pool_retired() -> None:
    """Count a retired graph pool. Call once the graphs that held it are dropped, never before."""
    with _RETIRE_LOCK:
        _RETIRED_POOLS[0] += 1


def retired_pool_due() -> bool:
    with _RETIRE_LOCK:
        return _RETIRED_POOLS[0] != _RETIRED_POOLS[1]


def _retired_pools_seen() -> Optional[int]:
    """Retirement count a flush covers (None if none due); read before gc.collect frees them."""
    with _RETIRE_LOCK:
        seen = _RETIRED_POOLS[0]
        if seen == _RETIRED_POOLS[1]:
            return None
    gc.collect()  # outside _CAPTURE_LOCK: a finalizer may take it
    return seen


def _flush_retired_locked(seen: int) -> bool:
    """Under ``_CAPTURE_LOCK`` with no capture recording: synchronize, clear the cuBLAS workspaces a deferred release
    owed, flush, acknowledge up to ``seen``."""
    with _RETIRE_LOCK:
        clear_cublas, _CUBLAS_CLEAR_DUE[0] = _CUBLAS_CLEAR_DUE[0], False
    try:
        torch = _torch()
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            sync = getattr(cuda, "synchronize", None)
            if callable(sync):
                sync()
            if clear_cublas:
                clear = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
                if callable(clear):
                    clear()
            empty = getattr(cuda, "empty_cache", None)
            if callable(empty):
                empty()
    except Exception:  # noqa: BLE001 - not flushed: stays due, the cuBLAS clear with it
        if clear_cublas:
            with _RETIRE_LOCK:
                _CUBLAS_CLEAR_DUE[0] = True
        return False
    with _RETIRE_LOCK:
        _RETIRED_POOLS[1] = max(_RETIRED_POOLS[1], seen)
    _POOL_FLUSHES[0] += 1
    return True


def _settle_retired_pools(seen: int) -> None:
    """Under ``_CAPTURE_IDLE``, from ``_capturing``: flush the pools retired up to ``seen`` before this capture records.
    Waits (lock released) for another thread's recording to end, since that one prohibits the flush; raises
    ``CaptureRefused`` (depth untouched, nothing recorded) when it does not end in time or the flush failed."""
    with _RETIRE_LOCK:
        if _RETIRED_POOLS[1] >= seen:
            return  # another capture flushed them already: no wait on its recording
    if not _CAPTURE_IDLE.wait_for(lambda: _CAPTURE_DEPTH == 0, timeout = RETIRED_FLUSH_WAIT_S):
        raise CaptureRefused(
            "a retired graph pool could not be flushed (another capture is recording); capture deferred"
        )
    with _RETIRE_LOCK:
        if _RETIRED_POOLS[1] >= seen:
            return  # another capture flushed them while this one waited
    if not _flush_retired_locked(seen):
        raise CaptureRefused("a retired graph pool could not be flushed; capture deferred")


def flush_retired_pools() -> bool:
    """Best effort, outside any capture: hand retired pools back now rather than at the next capture's entry. False
    (still due) while another thread records."""
    seen = _retired_pools_seen()
    if seen is None:
        return True
    with hold_off_capture() as safe:
        if not safe:
            return False
        return _flush_retired_locked(seen)


def flush_now_locked(clear_cublas: bool = False) -> bool:
    """For a caller already holding ``hold_off_capture()`` (True): flush now, and acknowledge every retirement counted
    before it, so the next capture's entry does not flush again."""
    with _RETIRE_LOCK:
        seen = _RETIRED_POOLS[0]
        if clear_cublas:
            _CUBLAS_CLEAR_DUE[0] = True
    gc.collect()
    return _flush_retired_locked(seen)


def release_cached_outside_capture(clear_cublas: bool = False) -> bool:
    """Empty_cache and cuBLAS clear run under the capture lock, else owed to the next capture."""
    # Owed first, in one step with the cuBLAS debt: a capture entering from here on sees it due, whether or not this
    # flush gets the lock; a successful flush below acknowledges it.
    with _RETIRE_LOCK:
        _RETIRED_POOLS[0] += 1
        seen = _RETIRED_POOLS[0]
        if clear_cublas:
            _CUBLAS_CLEAR_DUE[0] = True
    gc.collect()  # outside _CAPTURE_LOCK: a finalizer may take it
    with hold_off_capture() as safe:
        if safe and _flush_retired_locked(seen):
            return True
    return False


def _note_dropped_pools(tokens: list) -> None:
    """Dropped pools count as retired unless a live graph still records into them."""
    try:
        if not tokens:
            return
        live = [
            getattr(entry, "pool_token", None)
            for wrapper in tuple(_LIVE_WRAPPERS)
            for entry in tuple(wrapper.cache.values())
        ]
        if any(token is None or token not in live for token in tokens):
            note_pool_retired()
    except Exception:  # noqa: BLE001 - unsure: count it, the cost is one flush
        note_pool_retired()


# Raised while a whole step warms up or records: per-block graphs then run their compute, not nested replays.
_STEP_RECORDING = [0]


@contextlib.contextmanager
def _recording_step():
    _STEP_RECORDING[0] += 1
    try:
        yield
    finally:
        _STEP_RECORDING[0] -= 1


def step_recording() -> bool:
    return _STEP_RECORDING[0] > 0


def capture_in_progress() -> bool:
    """Snapshot for reporting only; can be stale by the time the caller acts, so never gate a CUDA call on it; use hold_off_capture instead."""
    with _CAPTURE_LOCK:
        return _CAPTURE_DEPTH > 0


CAPTURE_MODE = "thread_local"


def _graph_with_flush(
    graph: Any,
    pool: Any = None,
    stream: Any = None,
):
    return _torch().cuda.graph(graph, pool = pool, stream = stream, capture_error_mode = CAPTURE_MODE)


_CAPTURE_STREAMS: dict = {}


def _capture_stream(index: int) -> Any:
    """One capture stream per card: torch's default capture stream lives on the first card that captured, so a
    capture for another card recorded none of its kernels. Pool blocks are reused only from the recording stream."""
    torch = _torch()
    stream = _CAPTURE_STREAMS.get(index)
    if stream is None:
        default = getattr(torch.cuda.graph, "default_capture_stream", None)
        if default is not None and getattr(default, "device", None) == torch.device("cuda", index):
            stream = default
        else:
            stream = torch.cuda.Stream(device = index)
        stream = _CAPTURE_STREAMS.setdefault(index, stream)
    return stream


@contextlib.contextmanager
def card_graph(
    graph: Any,
    index: int,
    pool: Any = None,
):
    """Records on a card's own stream; bare torch.cuda.graph on a second card is empty."""
    torch = _torch()
    index = int(index)
    with torch.cuda.device(index):
        with _graph_with_flush(graph, pool = pool, stream = _capture_stream(index)):
            yield


@contextlib.contextmanager
def _graph_without_flush(
    graph: Any,
    pool: Any = None,
    stream: Any = None,
):
    """torch.cuda.graph minus empty_cache and host emptyCache; same stream, CAPTURE_MODE, pool."""
    torch = _torch()
    torch.cuda.synchronize()
    if stream is None:
        stream = _capture_stream(torch.cuda.current_device())
    with torch.cuda.stream(stream):
        graph.capture_begin(pool = pool, capture_error_mode = CAPTURE_MODE)
        try:
            yield
        finally:
            graph.capture_end()


# Not ``torch.utils._pytree``: it makes an unregistered object a LEAF, which must stay visible.


def _flatten(obj: Any, out: list) -> tuple:
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


def _copy_mode(entry: Any) -> Any:
    if getattr(entry, "inference_copy", False):
        return _torch().inference_mode()
    return contextlib.nullcontext()


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
# Batch-size graphs (batch_rows > 1) record into their own pool, shared by every batched graph (both DiTs of a dual-DiT
# family included): torch frees a pool's blocks only once no graph uses it, so in the size-1 graphs' pool dropping the
# batched graphs would give back nothing. "shared" (env) records them beside the size-1 graphs instead.
BATCH_POOL_ENV = "UNSLOTH_DIFFUSION_BATCH_GRAPH_POOL"
_BATCH_POOL_BOX: list = [None]


def _batch_pool_split() -> bool:
    return (os.environ.get(BATCH_POOL_ENV) or "split").strip().lower() != "shared"


def _pool_box() -> list:
    """The pool box a capture in this context records into."""
    rows = _BATCH_ROWS.get()
    if rows is not None and rows[0] > 1 and _batch_pool_split():
        return _BATCH_POOL_BOX
    return _POOL_BOX


# A pool id handed to a capture after the last graph recorded into it died is a dangling pointer.
_LIVE_WRAPPERS: "weakref.WeakSet" = weakref.WeakSet()


def _drop_pool_if_unused() -> None:
    """Drop each shared pool id once no live graph uses it (else use_count assert); check per graph."""
    for box in (_POOL_BOX, _BATCH_POOL_BOX):
        try:
            pool = box[0]
            if pool is None:
                continue
            if any(
                getattr(entry, "pool_token", pool) == pool  # handles are fresh tuples per call
                for wrapper in tuple(_LIVE_WRAPPERS)
                for entry in tuple(wrapper.cache.values())
            ):
                continue
            box[0] = None
        except Exception:  # noqa: BLE001
            pass


# A failed capture (torch >= 2.11) pins its graph and pool forever via the host allocator.
MAX_FAILED_CAPTURES = 16
_FAILED_GRAPHS: list = []
_COLLIDED_GRAPHS: list = []
_FAILED_LOCK = threading.Lock()


def captures_exhausted() -> bool:
    """True once MAX_FAILED_CAPTURES recordings failed in this process; every later capture then runs eager."""
    return len(_FAILED_GRAPHS) >= MAX_FAILED_CAPTURES


def capture_device() -> Optional[int]:
    """The CUDA ordinal a capture records on (the current device), or None when unreadable."""
    try:
        return int(_torch().cuda.current_device())
    except Exception:  # noqa: BLE001
        return None


def live_graph_count(device_index: Optional[int]) -> int:
    """Live graphs on device (whole-step and per-block); pools stay allocated until reset."""
    total = sum(
        len(w.cache)
        for w in tuple(_LIVE_WRAPPERS)
        if getattr(w, "_unsloth_capture_device", None) == device_index
    )
    from .diffusion_block_graph import live_block_graph_count

    return total + live_block_graph_count(device_index)


_CAPTURES: dict = {}
# image and video render threads capture concurrently: an unserialized read-modify-write could lose an increment
_CAPTURES_LOCK = threading.Lock()


def note_capture(device_index: Optional[int]) -> None:
    """Count a recorded graph (whole-step or per-block) on ``device_index``; see ``capture_generation``."""
    with _CAPTURES_LOCK:
        _CAPTURES[device_index] = _CAPTURES.get(device_index, 0) + 1


def capture_generation(device_index: Optional[int]) -> int:
    """Graphs ever recorded on ``device_index``. A full cache evicts before it captures, so a replacement leaves
    ``live_graph_count`` unchanged; this count still moves."""
    return _CAPTURES.get(device_index, 0)


def failed_capture_count() -> int:
    """Failed captures, absorbed OOMs included; a render that saw one overstates its need."""
    return len(_FAILED_GRAPHS)


def refuse_if_exhausted(logger: Any = None) -> None:
    """Raise (the caller's failed-capture path then runs eager) once the failure budget is spent; log that once."""
    global _EXHAUSTED_LOGGED
    if not captures_exhausted():
        return
    with _FAILED_LOCK:
        first = not _EXHAUSTED_LOGGED
        _EXHAUSTED_LOGGED = True
    if first and logger is not None:
        logger.warning(
            "diffusion.cuda_graph: %d graph captures failed in this process; no further captures are attempted",
            len(_FAILED_GRAPHS),
        )
    raise RuntimeError(
        f"{len(_FAILED_GRAPHS)} graph captures failed in this process; capture is off"
    )


_EXHAUSTED_LOGGED = False


def retire_failed_capture(
    graph: Any,
    pool: Any,
    exc: Optional[BaseException] = None,
) -> None:
    """Caller must stop handing pool to captures; a recording begun elsewhere (already recording) is not ours to end, so only abandon our own."""
    begun_elsewhere = exc is not None and "already recording" in str(exc)
    ended_here = False if begun_elsewhere else _abandon_capture_pool(pool)
    if not begun_elsewhere:  # the generators then belong to the other thread's live capture
        _heal_generators()
    if not ended_here and not begun_elsewhere:
        try:
            graph.reset()
        except Exception:  # noqa: BLE001
            pass
    with _FAILED_LOCK:
        # A collision with another recording left nothing behind: kept alive, but not a failure the budget counts.
        (_COLLIDED_GRAPHS if begun_elsewhere else _FAILED_GRAPHS).append(graph)
    for box in (_POOL_BOX, _BATCH_POOL_BOX):
        if pool is not None and box[0] == pool:
            box[0] = None
    if not begun_elsewhere:
        note_pool_retired()  # whatever the failed recording took stays cached until the next capture's entry flush


def _nvfp4_flashinfer_linears(module: Any) -> list:
    try:
        from .diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear
    except Exception:  # noqa: BLE001 - no backend module, no NVFP4 layers to find
        return []
    found: list = []
    try:
        for name, sub in module.named_modules():
            if is_nvfp4_flashinfer_linear(sub):
                found.append((name, sub))
    except Exception:  # noqa: BLE001 - an exotic module tree is simply not an NVFP4 one
        return []
    return found


def _protect_keyed(module: Any) -> bool:
    """Whether the graph key needs the precision branch: lever armed AND the module holds NVFP4 layers."""
    try:
        from .diffusion_nvfp4_protect import module_controller
        if not module_controller(module).armed:
            return False
        return bool(_nvfp4_flashinfer_linears(module))
    except Exception:  # noqa: BLE001 - a tree we cannot walk simply keys the way it always did
        return False


def protect_graph_key(controller: Any = None) -> tuple:
    from .diffusion_nvfp4_protect import protect_graph_key as _key
    return _key(controller = controller)


def _unbaked_nvfp4_layers(layers: list) -> list:
    return [name for name, layer in layers if not getattr(layer, "activation_scales_baked", False)]


def _prewarm_token_counts(live: list) -> tuple:
    """Candidate GEMM row counts (M) from the warm-up's shapes, smallest first; bounded."""
    counts = {1}
    for tensor in live:
        try:
            shape = tuple(int(dim) for dim in tensor.shape)
        except Exception:  # noqa: BLE001 - not a shaped tensor, nothing to read
            continue
        if len(shape) < 2:
            continue
        rows = 1
        for dim in shape[:-1]:
            rows *= dim
        if rows > 0:
            counts.add(rows)
    return tuple(sorted(counts)[:8])


def _allocator_segments() -> list:
    torch = _torch()
    native = getattr(getattr(torch, "_C", None), "_cuda_memorySnapshot", None)
    if native is None:
        return torch.cuda.memory_snapshot()
    # Avoid borrowed-reference corruption in the tuple API: pytorch/pytorch#201043.
    try:
        snapshot = native(None)
    except TypeError:
        snapshot = native()
    return snapshot["segments"]


def _pool_id_bytes(pool: Any) -> int:
    """Device bytes pool id ``pool`` holds now, 0 if unreadable or None."""
    if pool is None:
        return 0
    try:
        want = tuple(pool)
        return sum(
            int(seg.get("total_size", 0))
            for seg in _allocator_segments()
            if tuple(seg.get("segment_pool_id") or ()) == want
        )
    except Exception:  # noqa: BLE001
        return 0


def _first_device_index(args: tuple, kwargs: dict) -> Optional[int]:
    torch = _torch()
    for value in (*args, *kwargs.values()):
        index = (
            getattr(getattr(value, "device", None), "index", None)
            if torch.is_tensor(value)
            else None
        )
        if index is not None:  # a CPU operand says nothing about the card
            return index
    return None


def _pool_bytes(graph: Any) -> int:
    """Device bytes the graph's private pool holds (shared by every graph of the load), 0 if unreadable."""
    try:
        pool = tuple(graph.pool())
        total = 0
        for seg in _allocator_segments():
            if tuple(seg.get("segment_pool_id") or ()) == pool:
                total += int(seg.get("total_size", 0))
        return total
    except Exception:  # noqa: BLE001
        return 0


def _current_stream() -> Any:
    try:
        return _torch().cuda.current_stream()
    except Exception:  # noqa: BLE001
        return None


def _restore_stream(stream: Any) -> None:
    if stream is None:
        return
    try:
        torch = _torch()
        if torch.cuda.current_stream() != stream:
            torch.cuda.set_stream(stream)
    except Exception:  # noqa: BLE001
        pass


def _abandon_capture_pool(pool: Any) -> bool:
    """Ends a pool capture a failed capture left open (else torch 2.6 asserts on empty_cache); only when this call ended it, never twice."""
    torch = _torch()
    end = getattr(torch._C, "_cuda_endAllocateToPool", None) or getattr(
        torch._C, "_cuda_endAllocateCurrentStreamToPool", None
    )
    if end is None or pool is None:
        return False
    try:
        device = torch.cuda.current_device()
        end(device, pool)
    except Exception:  # noqa: BLE001 - capture_end already took it off, or capture_begin never put it on
        return False
    try:
        torch._C._cuda_releasePool(device, pool)
    except Exception:  # noqa: BLE001
        pass
    return True


def _heal_generators() -> None:
    """Clears capture mode on the CUDA generators after a failed capture, or later eager RNG draws fail."""
    try:
        torch = _torch()
        for gen in getattr(torch.cuda, "default_generators", ()) or ():
            clone = getattr(gen, "clone_state", None)
            restore = getattr(gen, "graphsafe_set_state", None)
            if callable(clone) and callable(restore):
                restore(clone())
    except Exception:  # noqa: BLE001
        pass


# Offloaded: each key's first SPEED_SAMPLES replays are judged against its own eager steps (_judge_keys).
SPEED_SAMPLES = 3
# Timed eager steps before a key's capture; the capture's own warm-ups would measure the replay.
SPEED_EAGER_SAMPLES = 2
# A key's replay this much slower than its eager step (and by at least SPEED_MIN_MS) drops that key's graph.
SPEED_MARGIN = 0.03
SPEED_MIN_MS = 2.0
# Streamed: the recording holds a ring and a pool the eager step does not, so it must be at least this much faster.
SPEED_GAIN = 0.01
SPEED_CHECK_ENV = "UNSLOTH_DIFFUSION_OFFLOAD_GRAPH_SPEED_CHECK"


def speed_check_enabled() -> bool:
    return (os.environ.get(SPEED_CHECK_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _timing_events() -> tuple:
    torch = _torch()
    return torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)


def live_pool_bytes() -> int:
    """Bytes the shared graph pool holds while any graph lives (the status number)."""
    try:
        return max(
            [int(w.stats.get("pool_bytes", 0)) for w in tuple(_LIVE_WRAPPERS) if w.cache] or [0]
        )
    except Exception:  # noqa: BLE001
        return 0


def live_pool_free_bytes() -> int:
    """Every pool a live graph recorded into: a failed capture retires the box pool while earlier graphs keep it."""
    pools = {
        tuple(entry.pool_token)
        for w in tuple(_LIVE_WRAPPERS)
        for entry in tuple(w.cache.values())
        if getattr(entry, "pool_token", None) is not None
    }
    if not pools:
        return 0
    try:
        free = 0
        cuda = _torch().cuda
        device = cuda.current_device()
        for seg in _allocator_segments():
            if seg.get("device", device) != device:
                continue
            if tuple(seg.get("segment_pool_id") or ()) in pools:
                free += int(seg.get("total_size", 0)) - int(seg.get("allocated_size", 0))
        return max(0, free)
    except Exception:  # noqa: BLE001 - over-counting only makes the guard stricter
        return live_pool_bytes()


def releasable_pool_bytes(handles: Any, device: Optional[int] = None) -> int:
    """Graph-pool bytes reset_all(handles) frees on device (current when None): whole-step pools
    these handles record into (shared ones stay), plus their block pools; 0 if unreadable."""
    handles = tuple(handles or ())
    # reset() also frees a whole-step graph's per-block fallback, so its block pools count too
    nested = []
    pending = list(handles)
    while pending:
        h = pending.pop()
        if h is None or any(h is seen for seen in nested):
            continue
        nested.append(h)
        pending.append(getattr(h, "fallback_handle", None))
    handles = tuple(nested)
    own = {id(h) for h in handles}
    try:
        if device is None:
            device = int(_torch().cuda.current_device())
    except Exception:  # noqa: BLE001 - unknown: release nothing
        return 0
    mine: set = set()
    others: set = set()
    for w in tuple(_LIVE_WRAPPERS):
        if id(w) not in own and getattr(w, "_unsloth_capture_device", device) not in (None, device):
            continue  # graph pools are per device: a recording on another card keeps nothing here alive
        tokens = {
            tuple(entry.pool_token)
            for entry in tuple(w.cache.values())
            if getattr(entry, "pool_token", None) is not None
        }
        (mine if id(w) in own else others).update(tokens)
    pools = mine - others
    shared = {
        id(state)
        for h in handles
        if callable(getattr(h, "_all_shared", None))
        for state in h._all_shared()
    }
    try:
        free = 0
        if pools:
            for seg in _allocator_segments():
                if seg.get("device", device) != device:
                    continue
                if tuple(seg.get("segment_pool_id") or ()) in pools:
                    free += int(seg.get("total_size", 0)) - int(seg.get("allocated_size", 0))
        if shared:
            from .diffusion_block_graph import pool_bytes
            free += int(pool_bytes(device, only = shared))
        return max(0, free)
    except Exception:  # noqa: BLE001 - unknown: release nothing
        return 0


def _warn(logger: Any, what: str, exc: Any) -> None:
    if logger is not None:
        logger.warning("diffusion.cuda_graph: %s failed: %s", what, exc)


def _outer_layer(module: Any) -> Any:
    """The ``_unsloth_outer_forward`` layer in ``module``'s forward slot (its ``inner`` None = class forward)."""
    try:
        slot = module.__dict__.get("forward")
    except Exception:  # noqa: BLE001 - no instance dict, nothing layered
        return None
    return slot if getattr(slot, "_unsloth_outer_forward", False) is True else None


def _router_layer(module: Any) -> Any:
    try:
        slot = module.__dict__.get("forward")
    except Exception:  # noqa: BLE001
        return None
    # A static step skip installed after the router wraps it: the router is the skip's inner.
    if getattr(slot, "_unsloth_outer_forward", False) is True:
        slot = getattr(slot, "inner", None)
    return slot if getattr(type(slot), "_unsloth_cfg_router_layer", False) is True else None


class _Entry:
    """One captured graph plus the static buffers it replays into."""

    __slots__ = (
        "graph",
        "static",
        "in_spec",
        "out_spec",
        "out_tensors",
        "sticky",
        "last",
        "inference_copy",
        "pool_token",
    )


def _layer_split(module: Any) -> bool:
    from .diffusion_layer_split import is_layer_split
    return is_layer_split(module)


def _rehome_after_layer_split(pipe: Any, logger: Any) -> None:
    """Before any recording or first forward: the int8 / fused records the speed layer built on one card follow the
    split's blocks to their cards (``diffusion_int8_gemm.after_layer_split``)."""
    try:
        from .diffusion_int8_gemm import after_layer_split
        for module in _denoiser_modules(pipe):
            if _layer_split(module):
                after_layer_split(module, logger)
    except Exception as exc:  # noqa: BLE001 - the stock paths stay correct
        _warn(logger, "layer split re-home", exc)


def _layer_split_pipe(pipe: Any) -> bool:
    """Any denoiser of ``pipe`` is split across cards by block ranges."""
    try:
        return any(_layer_split(m) for m in _denoiser_modules(pipe))
    except Exception:  # noqa: BLE001
        return False


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
        call: Any = None,
        placement: Optional["OffloadPlacement"] = None,
    ) -> None:
        self.module = module
        # How the weights reach the GPU (None = resident); see OffloadPlacement.
        self.placement = placement
        # CLASS forward; a stock forward that syncs the host gets its capture-safe rewrite. `call` overrides.
        safe = None
        if call is None or placement is not None:
            try:
                from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
                safe, _ = _capture_safe(type(module))
            except Exception:  # noqa: BLE001 - the stock forward is still a correct eager callable
                safe = None
        self.capture_safe = safe is not None
        # Planner maps non-tensor args (KV cache) to tensors-only or None (eager); new sticky tensors copy.
        self.plan = getattr(safe, "__unsloth_graph_plan__", None) if safe is not None else None
        self.sticky = frozenset(getattr(safe, "__unsloth_graph_sticky__", ()) or ())
        # How a planned step enters through the offload hooks (offload_placement refuses a placement without it).
        self.placed = getattr(safe, "__unsloth_graph_placed__", None) if safe is not None else None
        if placement is not None:
            # Model offload: weights are on GPU, so the slot may hold a rewrite; block offload uses _placed_step.
            rewrite = safe is not None and self.plan is None and placement.mode == "model"
            call = safe.__get__(module) if rewrite else placement.eager
        if call is not None:
            self.orig = call
        else:
            self.orig = (safe if safe is not None else type(module).forward).__get__(module)
        try:
            update_wrapper(self, self.orig, updated = ())
            # Copy wrapped attrs but never over ours: under block offload orig may be the step skip (plan/stats).
            for name, value in dict(getattr(self.orig, "__dict__", None) or {}).items():
                self.__dict__.setdefault(name, value)
        except Exception:  # noqa: BLE001
            pass
        try:
            # LOAD-BEARING: H3's pipeline filters kwargs by forward's signature; bare **kwargs drops tensors.
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
        # Set when the module's branch-key hook declines a call (_branch_declined): the status names it.
        self.branch_refusal: Optional[str] = None
        self.cache: dict = {}
        self.cap_hit = False
        self._warmed: dict = {}
        # Resident only, per graph key: real steps run eagerly on capture-layout copies before the recording.
        self._real_warm: dict = {}
        # Keys whose real warm-ups did not finish inside one render: they record the side-stream way on their next
        # appearance, so renders 2+ replay as before.
        self._real_warm_closed: set = set()
        # The denoiser steps of the render under way (note_render_steps), None when unknown: real warm-ups need more
        # than WARMUP_ITERS of them, else the key records on its first call as before.
        self._render_steps: Optional[int] = None
        # Keys recorded before: a placement change re-records them with no warm-up (same shapes, new addresses).
        self._recorded: dict = {}
        # Offloaded only, per graph key: seen count, eager/graph spans, verdict ('' kept, else drop reason).
        self._judge: dict = {}
        self._dropped: set = set()  # keys that run eager for the load
        self._slower: Optional[str] = None
        # Per-block fallback for the calls this graph leaves eager: its arming callable and the handle it returned.
        self.fallback: Any = None
        self.fallback_handle: Any = None
        self.protect_keyed: Optional[bool] = None
        self.protect_ctl: Any = None
        self.stats = {
            "captures": 0,
            "replays": 0,
            "eager_calls": 0,
            "fallbacks": 0,
            "refused_float": 0,
            "refused_host_tensor": 0,
            "refused_object": 0,
            "refused_branch": 0,
            "cap_skips": 0,
            "invalidations": 0,
            "pool_bytes": 0,
            "planned_eager": 0,
            "shape_warmups": 0,
            "real_warmups": 0,
            "evictions": 0,
            "speed_eager": 0,
            "length_eager": 0,
            "batch_captures": 0,
            "batch_eager": 0,
            "batch_capture_failures": 0,
        }
        # Graph keys recorded inside a coalesced call -> its request count (batch_rows); every other key is size 1.
        self._batched: dict = {}
        # Batched graph key -> the race canvas tokens it served (batch_rows ``canvas``); read only beside _batched.
        self._batched_canvas: dict = {}
        # Why this load records no more batched graphs (an OOM recording one), else None.
        self._batched_off: Optional[str] = None
        self.valid_token: Any = None
        # A text-length-keyed denoiser would record a graph per prompt; each image scope records
        # lengths only in its first render, sealed once a graph replays or the render ends, then
        # unseen lengths run ungraphed (same image); a new scope gets its own window.
        self.length_varies = _length_varies(module)
        self._sealed_scopes: set = set()
        self._scope_captures: dict = {}
        self._render_scopes: set = set()
        # Keys warmed since the last render end/reset may still record after a seal (true CFG's other-length
        # negative). Not ``_warmed``: it outlives resets, so would reopen sealed scopes.
        self._open_warmups: set = set()
        _LIVE_WRAPPERS.add(self)

    @property
    def length_sealed(self) -> bool:
        return bool(getattr(self, "_sealed_scopes", None))

    def install(self) -> "GraphedForward":
        """Set ``forward`` via ``__dict__`` (``__setattr__`` would inspect it), under any step-skip layer or CFG router."""
        if self.placement is not None:
            self.placement.install(self)
            return self
        router = _router_layer(self.module)
        if router is not None:
            router.inner = self
            return self
        outer = _outer_layer(self.module)
        if outer is not None:
            outer.inner = self
            return self
        self.module.__dict__["forward"] = self
        return self

    def uninstall(self) -> "GraphedForward":
        if self.placement is not None:
            self.placement.uninstall(self)
            return self
        router = _router_layer(self.module)
        if router is not None and router.inner is self:
            router.inner = None
            return self
        outer = _outer_layer(self.module)
        if outer is not None:
            if outer.inner is self:
                outer.inner = None
            return self
        if self.module.__dict__.get("forward") is self:
            self.module.__dict__.pop("forward", None)
            return self
        # Offload hooks installed after this wrapper wrap it: unwrap it inside their chain, keeping the hooks.
        stock = type(self.module).forward.__get__(self.module)
        registry = getattr(self.module, "_diffusers_hook", None)
        for ref in list(getattr(registry, "_fn_refs", None) or ()):
            for attr in ("forward", "original_forward"):
                if getattr(ref, attr, None) is self:
                    setattr(ref, attr, stock)
        self._unlink_wrapper_metadata(stock)
        return self

    # What ``update_wrapper`` copies from the class forward through this wrapper: kept on a hook wrapper.
    _WRAPPER_METADATA = frozenset(
        {*WRAPPER_ASSIGNMENTS, "__wrapped__", "__signature__", "__dict__"}
    )

    def _unlink_wrapper_metadata(self, stock: Any) -> None:
        """update_wrapper leaves __wrapped__ naming this retired wrapper; the CFG replica builder
        refuses an unknown override, so relink to the class forward and drop the copies."""
        chain: list = []
        link = self.module.__dict__.get("forward")
        while link is not None and link is not self and len(chain) < 16:
            chain.append(link)
            link = getattr(link, "__wrapped__", None)
        if link is not self:
            return
        mine = self.__dict__
        for link in chain:
            state = getattr(link, "__dict__", None)
            if not isinstance(state, dict):
                continue
            for key in [
                k
                for k, v in state.items()
                if k in mine and k not in self._WRAPPER_METADATA and v is mine[k]
            ]:
                del state[key]
            if state.get("__wrapped__") is self:
                state["__wrapped__"] = stock

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
        if getattr(self, "fallback_handle", None) is not None:
            self.fallback_handle.set_bypass(on)
        return self

    def _release(self) -> None:
        """Return the dropped graphs' pool segments to the device; ``gc.collect`` alone leaves them in the private pool."""
        # the recordings ran on their own capture stream, whose cuBLAS workspace outlives them
        release_cached_outside_capture(clear_cublas = getattr(self, "placement", None) is not None)

    def _invalidate_for_placement(self, token: Any) -> None:
        """The recorded graphs replay the weight addresses of the placement they saw; a new placement records anew."""
        if self.cache:
            self.stats["invalidations"] += 1
            # Not reset(): empty_cache re-pays cudaMallocs; pool stays with allocator.
            dropped = [getattr(e, "pool_token", None) for e in self.cache.values()]
            self.cache.clear()
            _note_dropped_pools(dropped)  # the next capture flushes it before it records
            _drop_pool_if_unused()  # the next capture must not record into a pool no graph holds
        getattr(self, "_batched", {}).clear()
        getattr(self, "_edit_keys", set()).clear()
        self._warmed.clear()
        self.valid_token = token

    def _unbatched_count(self) -> int:
        batched = getattr(self, "_batched", {})
        return sum(1 for k in self.cache if k not in batched)

    def _eviction_victim(self, incoming_edit: bool) -> Any:
        """The least recently replayed size-1 graph a full planned cache may drop for a new size-1 key, else None: an
        edit's first; an edit never drops a text-to-image graph. Batched graphs have their own budget."""
        batched = getattr(self, "_batched", {})
        edits = getattr(self, "_edit_keys", set())
        victim = next((k for k in self.cache if k in edits and k not in batched), None)
        if victim is not None or incoming_edit:
            return victim
        return next((k for k in self.cache if k not in batched), None)

    def _batched_refusal(
        self,
        rows: tuple,
        device: Optional[int] = None,
    ) -> Optional[str]:
        """Why a call carrying ``rows`` requests may not record a new graph here, else None."""
        count, capture = rows
        if getattr(self, "_batched_off", None):
            return self._batched_off
        if not capture:
            return "no recording on a user's request"
        if getattr(self, "placement", None) is not None:
            # A failed recording under an offload placement poisons the whole load; only resident loads record them.
            return "offloaded denoiser: batched calls run eager"
        if count not in BATCH_GRAPH_SIZES:
            return f"batch size {count} has no graph"
        batched = getattr(self, "_batched", {})
        if sum(1 for k in batched if k in self.cache) >= MAX_BATCHED_GRAPHS:
            return "batched graph cap reached"
        # Priced from the size-1 graph's own pool growth, not the whole shared pool.
        need = int(self.stats.get("size1_graph_bytes", 0) or 0) * count
        budget = _batch_pool_budget_bytes()
        if budget is not None and need > budget:
            return "over the batched graph pool budget"
        free = _free_device_bytes(device)
        if free is not None and need and need > free - BATCH_GRAPH_MARGIN_MB * 2**20:
            return "not enough free memory for a batched graph"
        return None

    def drop_batched(self, size: Optional[int] = None) -> int:
        """Drop the batched graphs (of ``size`` only, when given); the size-1 graphs stay. Returns how many. Their memory
        goes back only from their own pool (the default): with ``BATCH_POOL_ENV=shared`` the size-1 graphs keep the one
        pool alive and nothing returns to the card."""
        batched = getattr(self, "_batched", {})
        keys = [k for k, v in batched.items() if size is None or v == int(size)]
        gone = [getattr(self.cache.pop(k), "pool_token", None) for k in keys if k in self.cache]
        for k in keys:
            batched.pop(k, None)
        if gone:
            _note_dropped_pools(gone)
            _drop_pool_if_unused()
            self._release()
        return len(gone)

    def _drop_batched(self, reason: Optional[str] = None) -> None:
        batched = getattr(self, "_batched", {})
        gone = [
            getattr(self.cache.pop(k), "pool_token", None) for k in list(batched) if k in self.cache
        ]
        batched.clear()
        if reason is not None:
            self._batched_off = reason
        if gone:
            _note_dropped_pools(gone)
            _drop_pool_if_unused()

    def reset(self) -> "GraphedForward":
        """Drop every captured graph (weights changed under us: LoRA load, unload, adapter switch)."""
        dropped = bool(self.cache)
        self.cache.clear()
        getattr(self, "_batched", {}).clear()
        getattr(self, "_edit_keys", set()).clear()
        if dropped:
            note_pool_retired()  # counted where it drops; the _release below flushes and acknowledges it
        self._recorded.clear()
        # Nothing recorded is left to replay: every scope's first render starts over (captures counted afresh).
        self._sealed_scopes.clear()
        self._scope_captures.clear()
        self._render_scopes.clear()
        self._open_warmups.clear()
        self._real_warm.clear()  # new weights may compile a new variant: its first real steps warm it again
        self._real_warm_closed.clear()
        if getattr(self, "fallback_handle", None) is not None:
            try:
                self.fallback_handle.reset()
            except Exception:  # noqa: BLE001
                pass
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
        if (
            getattr(self, "fallback", None) is None
            and getattr(self, "fallback_handle", None) is None
        ):
            # A pending or armed per-block fallback still keys its graphs on the resident groups' addresses: the
            # stable re-pin stays until it is known to be gone (``_engage_fallback`` / ``free``).
            _retire_stable_placement(self.placement)
        # Poisoned entries are unreachable (__call__ goes eager first) yet pin statics, outputs,
        # pool slices; caller's _release frees both once the failed capture's frames are gone.
        dropped = [getattr(e, "pool_token", None) for e in self.cache.values()]
        self.cache.clear()
        getattr(self, "_batched", {}).clear()
        _note_dropped_pools(dropped)
        _drop_pool_if_unused()
        return self

    def free(self) -> "GraphedForward":
        self.reset()
        self.fallback = None
        if getattr(self, "fallback_handle", None) is not None:
            try:
                self.fallback_handle.free()
            except Exception:  # noqa: BLE001
                pass
            self.fallback_handle = None
        self.uninstall()
        if self.placement is not None:
            self.placement.release_slots()
        _retire_stable_placement(getattr(self, "placement", None))
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
            "length_sealed": bool(getattr(self, "length_sealed", False)),
            "batched_graphs": sorted(
                v for k, v in getattr(self, "_batched", {}).items() if k in self.cache
            ),
            "batched_off": getattr(self, "_batched_off", None),
            "placement": None if self.placement is None else self.placement.mode,
            "stats": dict(self.stats),
            "fallback": None
            if getattr(self, "fallback_handle", None) is None
            else self.fallback_handle.describe(),
            "capture_error": None
            if not error
            else {
                "type": str(error.get("type")),
                "msg": str(error.get("msg")),
            },
        }

    def _judge_keys(self) -> None:
        """Drop a key graph if its timed replays run slower than its own eager steps here; events are read only after completion (no host wait)."""
        for key, state in list(self._judge.items()):
            if state["verdict"] is not None or len(state["graph"]) < SPEED_SAMPLES:
                continue
            # Event queries and the drop's syncs / frees are prohibited while another thread records: judge later.
            with hold_off_capture() as safe:
                if not safe or not all(end.query() for _, end in state["graph"]):
                    continue
                try:
                    # Best of each, so a stall on a shared card in either window cannot tip the verdict alone.
                    eager = min(start.elapsed_time(end) for start, end in state["eager"])
                    graph = min(start.elapsed_time(end) for start, end in state["graph"])
                except Exception:  # noqa: BLE001
                    state["verdict"] = ""
                    continue
                self.stats["eager_ms"] = round(eager, 3)
                self.stats["replay_ms"] = round(graph, 3)
                placement = getattr(self, "placement", None)
                streams = placement is not None and placement.streams()
                if streams:
                    kept = graph < eager * (1.0 - SPEED_GAIN)
                else:
                    kept = graph <= eager * (1.0 + SPEED_MARGIN) or graph - eager < SPEED_MIN_MS
                if kept:
                    state["verdict"] = ""
                    continue
                if graph > eager:
                    reason = f"replay measured slower than the eager step on this device ({graph:.1f} vs {eager:.1f} ms)"
                else:
                    reason = (
                        f"replay measured within {SPEED_GAIN:.0%} of the eager step on this device ({graph:.1f} vs "
                        f"{eager:.1f} ms), not worth the memory a streamed recording holds"
                    )
                state["verdict"] = reason
                ring = False
                if streams:
                    # One placement, one verdict: its other keys stream the same copies.
                    ring = bool(placement.decline(reason))
                    for other in list(self.cache):
                        if other != key:
                            self.cache.pop(other, None)
                            self._dropped.add(other)
                    if (
                        getattr(self, "fallback", None) is None
                        and getattr(self, "fallback_handle", None) is None
                    ):
                        _retire_stable_placement(
                            placement
                        )  # no per-block graphs either: nothing records again
                self._slower = reason
                self._dropped.add(key)
                if self.logger is not None:
                    self.logger.info(
                        "diffusion.cuda_graph: %s drops a graph: %s",
                        type(self.module).__name__,
                        reason,
                    )
                dropped_entry = self.cache.pop(key, None)
                if dropped_entry is not None:
                    try:
                        _torch().cuda.current_stream().synchronize()  # no replay still running on what is freed
                    except Exception:  # noqa: BLE001
                        pass
                    gone = getattr(dropped_entry, "pool_token", None)
                    del dropped_entry  # counted once the graph is gone, never before
                    _note_dropped_pools([gone])
                    _drop_pool_if_unused()
                if not self.cache:
                    self.capture_error = {"type": "Refused", "msg": reason}
                    if not ring:
                        # (with a ring, its drop at the end of this forward flushes once: a mid-forward flush fragments the cache)
                        self._release()

    def _timed_eager(self, call: Any, args: tuple, kwargs: dict, key: Any) -> Any:
        """Eager reference step on fresh copies of the caller tensors; a key first call may autotune and is not timed; next SPEED_EAGER_SAMPLES are timed."""
        self.stats["eager_calls"] += 1
        self.stats["speed_eager"] += 1
        # Copies laid out like the capture statics: compiled code guards on strides and offsets.
        live: list = []
        spec = _flatten((args, kwargs), live)
        try:
            statics = self._statics(live)
        except _torch().cuda.OutOfMemoryError:
            self._dropped.add(key)
            self._slower = "no memory for the speed probe's input copies"
            return call(*args, **kwargs)
        args, kwargs = _rebuild(spec, statics)
        state = self._judge_state(key)
        state["seen"] += 1
        if state["seen"] == 1:
            self._warmed[key] = None
            return call(*args, **kwargs)
        # Event records are prohibited while another thread records a graph: such a step goes untimed.
        start = end = None
        with hold_off_capture() as safe:
            if safe:
                start, end = _timing_events()
                start.record()
        out = call(*args, **kwargs)
        if start is not None:
            with hold_off_capture() as safe:
                if safe:
                    end.record()
                    state["eager"].append((start, end))
        return out

    def _judge_state(self, key: Any) -> dict:
        state = self._judge.get(key)
        if state is None:
            state = self._judge[key] = {"seen": 0, "eager": [], "graph": [], "verdict": None}
            while len(self._judge) > 4 * self.max_graphs:
                self._judge.pop(next(iter(self._judge)))
        return state

    def _placed_step(self, **step: Any) -> Any:
        return self.placed(self.orig, step)

    def _eager(self, args: tuple, kwargs: dict) -> Any:
        self.stats["eager_calls"] += 1
        return self.orig(*args, **kwargs)

    def _eager_as_recorded(self, args: tuple, kwargs: dict) -> Any:
        """Eager step makes plain copies of inference leaves, else dispatch guards recompile."""
        if self.plan is not None:
            return self._eager(args, kwargs)
        live: list = []
        try:
            spec = _flatten((args, kwargs), live)
        except TypeError:
            return self._eager(args, kwargs)
        picks = [
            i
            for i, t in enumerate(live)
            if callable(getattr(t, "is_inference", None)) and t.is_inference()
        ]
        if not picks:
            return self._eager(args, kwargs)
        try:
            copies = self._statics([live[i] for i in picks])
        except (
            _torch().cuda.OutOfMemoryError
        ):  # no room for the copies: run as before, at the cost of a recompile
            return self._eager(args, kwargs)
        for i, fresh in zip(picks, copies):
            live[i] = fresh
        args, kwargs = _rebuild(spec, live)
        return self._eager(args, kwargs)

    def _background_call(self, args: tuple, kwargs: dict) -> Any:
        """Replay through the call a render captures: otherwise the render's compiled code guards on other layouts and recompiles."""
        if (
            self.plan is not None
            and self.enabled
            and not self.poisoned
            and not self.bypassed
            and kwargs.get("return_dict", True) is False
            and _bg_planned_replay_enabled()
        ):
            planned = self.plan(self.module, args, kwargs)
            if planned is not None:
                call, planned_kwargs = planned
                if self.placement is not None:
                    call = self._placed_step
                live: list = []
                spec = _flatten(((), planned_kwargs), live)
                warm_args, warm_kwargs = _rebuild(spec, self._statics(live))
                return call(*warm_args, **warm_kwargs)
        return self.orig(*args, **kwargs)

    def _placement_declined(self) -> bool:
        # A declined placement refuses every later recording for the load (a merely dropped key does not).
        return getattr(getattr(self, "placement", None), "_declined", None) is not None

    def _engage_fallback(self) -> None:
        """Arm per-block graphs on a key drop or failure, between forwards (slot ring moves)."""
        fallback, self.fallback = self.fallback, None
        try:
            self.fallback_handle = fallback()
        except Exception as exc:  # noqa: BLE001
            _warn(self.logger, "per-block fallback", exc)
            self.fallback_handle = None
        if self.fallback_handle is None and (self.poisoned or self._placement_declined()):
            _retire_stable_placement(self.placement)  # nothing left that records over the placement
        if self.fallback_handle is not None and self.logger is not None:
            self.logger.info(
                "diffusion.cuda_graph: %s records per block for the steps its whole-step graph leaves eager",
                type(self.module).__name__,
            )

    def _branch_declined(self, exc: BaseException) -> None:
        """The module's branch-key hook (GRAPH_KEY_EXTRA_ATTR) raised: a deliberate decline (Krea 2's per-prompt key
        start) or an unreadable flag. Counted apart from non-tensor arguments, named in the status, logged once."""
        self.stats["refused_branch"] = self.stats.get("refused_branch", 0) + 1
        reason = branch_decline_reason(exc)
        if getattr(self, "branch_refusal", None) != reason:
            self.branch_refusal = reason
            if self.logger is not None:
                self.logger.info(
                    "diffusion.cuda_graph: %s runs eager: %s", type(self.module).__name__, reason
                )
        exc.__traceback__ = None

    def __call__(self, *args, **kwargs):
        if _bg_capture_suppressed():
            # Background compile thread: never capture off the render thread.
            return self._background_call(args, kwargs)
        if getattr(self, "fallback", None) is not None and (self.poisoned or self._dropped):
            self._engage_fallback()
        if (
            not self.enabled
            or _bg_eager_forced()
            or self.bypassed
            or _FORCED_EAGER.get()
            or self.poisoned
            # Belt to the caller's per-chunk bypass: one step's cond and uncond share a key.
            or getattr(self.module, "_unsloth_step_cache", None)
            or _layer_split(self.module)
        ):
            return self._eager(args, kwargs)

        # True or absent builds an output dataclass, which a replay hands back frozen.
        if kwargs.get("return_dict", True) is not False:
            return self._eager(args, kwargs)

        if self.placement is not None:
            # The recorded graphs replay the weight addresses of the placement they saw; a new placement records anew.
            token = self.placement.token()
            if token != self.valid_token:
                self._invalidate_for_placement(token)
            refusal = self.placement.refusal(self.stats)
            if self._judge and speed_check_enabled():
                self._judge_keys()
                if self._dropped and getattr(self, "fallback", None) is not None:
                    self._engage_fallback()
                refusal = self.placement.refusal(self.stats)
            skip = getattr(self.placement, "skip", None)
            if refusal is None and skip is not None and _skips_planned(skip):
                # Skip steps need every call eager; per-block graphs, where armed, still record below the hooks.
                self.stats["skip_eager"] = self.stats.get("skip_eager", 0) + 1
                if getattr(self, "fallback", None) is not None:
                    self._engage_fallback()
                return self._eager(args, kwargs)
            if refusal is not None:
                if self.logger is not None and getattr(self, "_refusal_logged", None) != refusal:
                    self._refusal_logged = refusal
                    self.logger.info(
                        "diffusion.cuda_graph: %s runs eager: %s",
                        type(self.module).__name__,
                        refusal,
                    )
                self.capture_error = {"type": "Refused", "msg": refusal}
                return self._eager(args, kwargs)
            if (self.capture_error or {}).get("type") == "Refused" and (
                self.cache or not self._slower
            ):
                self.capture_error = None

        call = self.orig
        # Every eager fallback below runs the caller's own call, never the planned one.
        eager_args, eager_kwargs = args, kwargs
        if self.plan is not None:
            planned = self.plan(self.module, args, kwargs)
            if planned is None:
                self.stats["planned_eager"] += 1
                return self._eager(args, kwargs)
            call, kwargs = planned
            args = ()
            if self.placement is not None:
                # The offload hooks onload the top-level weights: the step must enter through the hooked forward.
                call = self._placed_step

        if kwargs.get("kv_cache_mode") == "extract":
            # Prefix-KV prefill (FLUX.2 klein KV step 0): a capture would poison plain calls.
            self.stats["refused_object"] += 1
            return self._eager(eager_args, eager_kwargs)
        declined = False
        try:
            key = graph_key((args, kwargs))
            if self.protect_keyed is None:
                self.protect_keyed = _protect_keyed(self.module)
                if self.protect_keyed:
                    from .diffusion_nvfp4_protect import module_controller
                    self.protect_ctl = module_controller(self.module)
                if self.protect_keyed:
                    self.max_graphs *= 2
                    if self.logger is not None:
                        self.logger.info(
                            "diffusion.cuda_graph: NVFP4 per-step precision is armed on %s; graph "
                            "cap raised to %d (one graph per branch per input shape)",
                            type(self.module).__name__,
                            self.max_graphs,
                        )
            if self.protect_keyed:
                # One graph per branch: a W4A4 replay at a W4A16 step means the lever never fired.
                key = key + protect_graph_key(self.protect_ctl)
            extra = getattr(self.module, GRAPH_KEY_EXTRA_ATTR, None)
            if callable(extra):
                # Pre-hook state that picks a branch: a graph recorded on one branch must not replay the other.
                try:
                    branch = extra()
                except Exception as exc:  # noqa: BLE001 - the hook declines this call: it runs eager
                    self._branch_declined(exc)
                    declined = True
                else:
                    key = key + (("x", branch),)
            if not declined:
                entry = self.cache.get(key)
                scope = (
                    _image_scope(args, kwargs) if getattr(self, "length_varies", False) else None
                )
                # A key that also follows the step (NVFP4 protect branch, a pre-hook's branch flag) has variants a
                # later step of this render still has to record: its scope seals when the render ends (seal_lengths).
                step_keyed = bool(self.protect_keyed) or callable(extra)
        except Exception:  # noqa: BLE001 - an unhashable tree is simply not capturable
            self.stats["refused_object"] += 1
            return self._eager(eager_args, eager_kwargs)
        if declined:
            # Outside the key handler: an error in the eager forward propagates instead of running it twice.
            return self._eager(eager_args, eager_kwargs)

        rows = _BATCH_ROWS.get() if entry is None else None
        if rows is not None and rows[0] > 1:
            refusal = self._batched_refusal(rows, _first_device_index(eager_args, eager_kwargs))
            if refusal is not None:
                self.stats["batch_eager"] += 1
                return self._eager_as_recorded(eager_args, eager_kwargs)
        else:
            rows = None
        if entry is None:
            if _uncapturable(key):
                self.stats["refused_object"] += 1
                return self._eager(eager_args, eager_kwargs)
            if _has_float(key):
                self.stats["refused_float"] += 1
                return self._eager(eager_args, eager_kwargs)
            if scope is not None:
                self._render_scopes.add(scope)
            if (
                scope in getattr(self, "_sealed_scopes", ())
                and (self.placement is None or key not in self._recorded)
                and not (self.plan is not None and key in self._open_warmups)
                # a key the first render already ran (real warm-ups it did not finish) records, as it would have then
                and key not in getattr(self, "_real_warm_closed", ())
            ):
                # A prompt length first seen after the first render: no warm-up, no capture, no eviction on this
                # render. (An offloaded key it recorded before re-records after its weights moved, as before.)
                self.stats["length_eager"] += 1
                return self._eager_as_recorded(eager_args, eager_kwargs)
            if self.placement is not None and speed_check_enabled():
                if key in self._dropped:
                    return self._eager(eager_args, eager_kwargs)
                state = self._judge.get(key)
                if state is None or (
                    state["verdict"] is None and len(state["eager"]) < SPEED_EAGER_SAMPLES
                ):
                    # The eager reference for _judge_keys, timed with CUDA events and never waited on.
                    return self._timed_eager(call, args, kwargs, key)
            if (
                self.placement is None
                and self.warmup > 0
                and real_warmup_enabled()
                and self._real_warm.get(key, 0) < self.warmup
                and key not in self._real_warm_closed
                and (
                    self._real_warm.get(key, 0) > 0
                    # a key this render runs on every step, and on more steps than the warm-ups: it records on one
                    # of them (a branch key may appear on 1-3 steps only, and records on its first call as before)
                    or (
                        not step_keyed and (getattr(self, "_render_steps", None) or 0) > self.warmup
                    )
                )
                and self._real_warmup_ready(key, args, kwargs)
            ):
                # Warm-up on statics-layout copies so recording meets no new compiled variant.
                self._real_warm[key] = self._real_warm.get(key, 0) + 1
                while len(self._real_warm) > 4 * self.max_graphs:
                    self._real_warm.pop(next(iter(self._real_warm)))
                self.stats["real_warmups"] += 1
                if self.plan is not None:
                    self._warmed[key] = None
                    self._open_warmups.add(key)
                try:
                    live = []
                    spec = _flatten((args, kwargs), live)
                    warm_args, warm_kwargs = _rebuild(spec, self._statics(live))
                    return call(*warm_args, **warm_kwargs)
                except Exception as exc:  # noqa: BLE001 - like a failed capture: never fails the render
                    if rows is not None and _is_oom(exc):
                        # Out of memory warming a batched shape (as when recording one): drop only the batched
                        # graphs, keep the size-1 ones, and run this step eager like the other unrecorded shapes.
                        self._real_warm_closed.add(key)
                        exc.__traceback__ = None
                        warm_args = warm_kwargs = live = None
                        self._drop_batched(f"out of memory warming a batch of {rows[0]}")
                        self.stats["batch_capture_failures"] += 1
                        if self.logger is not None:
                            self.logger.warning(
                                "diffusion.cuda_graph: no memory to warm a batch-%d graph on %s; batched calls run "
                                "eager, the size-1 graphs keep replaying",
                                rows[0],
                                type(self.module).__name__,
                            )
                        self._release()
                        return self._eager_as_recorded(eager_args, eager_kwargs)
                    # Copies can OOM or fail: this load runs eager like after a failed capture.
                    self.poison(exc)
                    self.stats["fallbacks"] += 1
                    if self.logger is not None:
                        self.logger.warning(
                            "diffusion.cuda_graph: real warm-up failed on %s (%s: %s); this load runs eager",
                            type(self.module).__name__,
                            type(exc).__name__,
                            exc,
                        )
                    exc.__traceback__ = None
                    warm_args = warm_kwargs = live = None
                    self._release()
                    return self._eager(eager_args, eager_kwargs)
            if self.poisoned:
                # the real warm-up's NVFP4 prewarm failed: like a failed capture, this load runs eager
                return self._eager(eager_args, eager_kwargs)
            if self.plan is not None and self.stats["captures"] and key not in self._warmed:
                # A new prompt length: this real step is the new shape's warm-up; the next one records.
                self._warmed[key] = None
                while len(self._warmed) > 4 * self.max_graphs:
                    self._warmed.pop(next(iter(self._warmed)))
                self._open_warmups.add(key)
                self.stats["shape_warmups"] += 1
                # Copies like statics: compiled code guards strides/offsets, or capture fails.
                live: list = []
                spec = _flatten((args, kwargs), live)
                warm_args, warm_kwargs = _rebuild(spec, self._statics(live))
                return call(*warm_args, **warm_kwargs)
            # Batched graphs have their own budget (MAX_BATCHED_GRAPHS): they never take a size-1 key's slot.
            batched_here = rows is not None
            victim = (
                self._eviction_victim(_has_reference(kwargs))
                if not batched_here
                and self.plan is not None
                and self._unbatched_count() >= self.max_graphs
                else None
            )
            if victim is not None:
                # A planned call keys on the prompt length, so evict the least recently replayed graph: a batched one
                # first, then an edit's; an edit never evicts a text-to-image graph (it runs ungraphed instead).
                self._batched.pop(victim, None)
                getattr(self, "_edit_keys", set()).discard(victim)
                gone = getattr(self.cache.pop(victim), "pool_token", None)
                self.stats["evictions"] += 1
                _note_dropped_pools([gone])
                # The last graph on the shared pool takes the pool with it; never hand its id to the next capture.
                _drop_pool_if_unused()
            if not batched_here and self._unbatched_count() >= self.max_graphs:
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
                return self._eager_as_recorded(eager_args, eager_kwargs)
            captured = None
            stream_before = _current_stream()
            try:
                captured = self._capture(args, kwargs, call, key)
            except CaptureRefused as exc:
                _restore_stream(stream_before)
                self.stats["capture_refusals"] = self.stats.get("capture_refusals", 0) + 1
                if self.logger is not None:
                    self.logger.warning(
                        "diffusion.cuda_graph: capture deferred on %s (%s); this step runs eager",
                        type(self.module).__name__,
                        exc,
                    )
                exc.__traceback__ = None
            except Exception as exc:  # noqa: BLE001 - a failed capture must never fail the render
                # graph __exit__ raises before leaving its capture stream; restore or everything runs on it
                _restore_stream(stream_before)
                if "already recording" not in str(exc):
                    _heal_generators()  # before anything eager draws from the CUDA RNG
                if rows is not None and self.placement is None and _is_oom(exc):
                    # Out of memory recording a batched graph: drop only the batched graphs, keep the size-1 ones.
                    self._drop_batched(f"out of memory recording a batch of {rows[0]}")
                    self.stats["batch_capture_failures"] += 1
                    if self.logger is not None:
                        self.logger.warning(
                            "diffusion.cuda_graph: no memory for a batch-%d graph on %s; batched calls run eager, "
                            "the size-1 graphs keep replaying",
                            rows[0],
                            type(self.module).__name__,
                        )
                    exc.__traceback__ = None
                    self._release()
                    # The size-1 graphs keep replaying: run like the other unrecorded shapes beside them.
                    return self._eager_as_recorded(eager_args, eager_kwargs)
                if self.placement is not None:
                    self.placement.recover()  # the eager retry must not wait on the dead capture's copies
                    self.placement.release_slots()  # nothing will replay into the ring for this load
                    if self.placement.mode == "model":
                        _retire_stable_range(self.module)
                self.poison(exc)
                self.stats["fallbacks"] += 1
                if self.logger is not None:
                    self.logger.warning(
                        "diffusion.cuda_graph: capture failed on %s (%s: %s); this load runs eager",
                        type(self.module).__name__,
                        type(exc).__name__,
                        exc,
                    )
                # The traceback pins statics and the pool slice through the eager retry, which then OOMs.
                exc.__traceback__ = None
            if captured is None:
                self._release()
                return self._eager(eager_args, eager_kwargs)
            entry = captured
            self.cache[key] = entry
            if rows is not None:
                self._batched[key] = rows[0]
                canvas = _BATCH_CANVAS.get()
                self.__dict__.setdefault("_batched_canvas", {})[key] = (
                    set() if canvas is None else {canvas}
                )
                self.stats["batch_captures"] += 1
            if _has_reference(kwargs):
                self.__dict__.setdefault("_edit_keys", set()).add(key)
            if scope is not None:
                self._scope_captures[scope] = self._scope_captures.get(scope, 0) + 1
            self._unsloth_capture_device = capture_device()
            note_capture(self._unsloth_capture_device)
            if self.logger is not None:
                self.logger.debug(
                    "diffusion.cuda_graph: captured %s graph %d/%d over %d input tensor(s)",
                    type(self.module).__name__,
                    len(self.cache),
                    self.max_graphs,
                    len(entry.static),
                )

        else:
            if scope is not None:
                self._render_scopes.add(scope)
                if not step_keyed:
                    self._sealed_scopes.add(scope)
            if self.plan is not None:
                self.cache[key] = self.cache.pop(key)
            if key in getattr(self, "_batched", {}):
                now = _BATCH_ROWS.get()
                if now is None or now[0] <= 1:
                    # A size-1 call (true CFG has two rows too) replays this graph: it is a size-1 graph now.
                    self._batched.pop(key, None)
                    getattr(self, "_batched_canvas", {}).pop(key, None)
                elif _BATCH_CANVAS.get() is not None:
                    # Same shapes at another race canvas (steps, guidance): this graph serves that one too.
                    self.__dict__.setdefault("_batched_canvas", {}).setdefault(key, set()).add(
                        _BATCH_CANVAS.get()
                    )

        live: list = []
        _flatten((args, kwargs), live)
        last = entry.last
        state = self._judge.get(key) if self.placement is not None else None
        timing = (
            state is not None and state["verdict"] is None and len(state["graph"]) < SPEED_SAMPLES
        )
        start = None
        if timing:
            # Timing spans input copies and output clones (eager pays neither); off while another thread records.
            with hold_off_capture() as safe:
                if safe:
                    start, end = _timing_events()
                    start.record()
        with _copy_mode(entry):
            for index, (dst, src) in enumerate(zip(entry.static, live)):
                if last is not None and entry.sticky[index]:
                    ref = last[index]
                    if ref is not None and ref() is src:
                        continue
                    dst.copy_(src)
                    last[index] = weakref.ref(src)
                    continue
                dst.copy_(src)
        entry.graph.replay()
        # Cloned: every replay writes the SAME buffers, which the pipeline holds across steps.
        outs = [t.clone() for t in entry.out_tensors]
        if start is not None:
            with hold_off_capture() as safe:
                if safe:
                    end.record()
                    state["graph"].append((start, end))
        self.stats["replays"] += 1
        skip = getattr(self.placement, "skip", None) if self.placement is not None else None
        if skip is not None and getattr(skip, "counting", False):
            # The replay passed the (pass-through) skip layer by: count the computed step as it would have.
            skip.stats["calls"] += 1
            skip.stats["computed"] += 1
        return _rebuild(entry.out_spec, outs)

    def _real_warmup_ready(self, key: Any, args: tuple, kwargs: dict) -> bool:
        """Before a key's first real warm-up: NVFP4 GEMMs tuned at the step's row counts (an untuned
        step would run another tactic than replays); calibrating NVFP4 loads run eager."""
        if self._real_warm.get(key, 0):
            return True
        nvfp4_layers = _nvfp4_flashinfer_linears(self.module)
        if not nvfp4_layers:
            return True
        if _unbaked_nvfp4_layers(nvfp4_layers):
            return False
        live: list = []
        _flatten((args, kwargs), live)
        from .diffusion_nvfp4_linear import nvfp4_prewarm

        try:
            nvfp4_prewarm(self.module, _prewarm_token_counts(live), logger = self.logger)
        except Exception as exc:  # noqa: BLE001 - as inside a capture: never fails the render
            self.poison(exc)
            self.stats["fallbacks"] += 1
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.cuda_graph: NVFP4 prewarm failed on %s (%s: %s); this load runs eager",
                    type(self.module).__name__,
                    type(exc).__name__,
                    exc,
                )
            exc.__traceback__ = None
            return False
        return True

    def _statics(self, live: list) -> list:
        """Fresh copies of ``live``, as a capture records from (and the shape warm-up step runs on)."""
        torch = _torch()
        if self.plan is not None:
            # Inference tensors as statics: compiled code guards on dispatch keys.
            static = []
            for t in live:
                with torch.inference_mode(bool(t.is_inference())):
                    static.append(torch.empty_like(t))
        else:
            # A tensor made under ``torch.inference_mode()``, which renders run in, refuses ``copy_``.
            with torch.inference_mode(False):
                static = [torch.empty_like(t) for t in live]
        # Writing into an inference tensor is only allowed inside inference mode.
        copying = types.SimpleNamespace(
            inference_copy = self.plan is not None and any(t.is_inference() for t in static)
        )
        with _copy_mode(copying):
            for dst, src in zip(static, live):
                dst.copy_(src)
        return static

    def _capture(
        self,
        args: tuple,
        kwargs: dict,
        call: Any = None,
        key: Any = None,
    ) -> _Entry:
        torch = _torch()
        call = self.orig if call is None else call
        entry = _Entry()
        live: list = []
        entry.in_spec = _flatten((args, kwargs), live)
        entry.sticky, entry.last = None, None
        if self.sticky:
            # Flat positions of the sticky kwargs, in ``_flatten``'s order (args, then kwargs by insertion).
            marks: list = []
            _flatten(args, marks)
            sticky = [False] * len(marks)
            for name, value in kwargs.items():
                part: list = []
                _flatten(value, part)
                sticky.extend([name in self.sticky] * len(part))
            if any(sticky):
                entry.sticky = tuple(sticky)
                entry.last = [weakref.ref(t) if s else None for t, s in zip(live, sticky)]

        bad = [(i, str(t.device)) for i, t in enumerate(live) if t.device.type != "cuda"]
        if bad:
            self.stats["refused_host_tensor"] += 1
            raise RuntimeError(
                f"{len(bad)} input tensor(s) are not on cuda ({bad[:4]}); a host tensor read "
                f"inside a captured region is baked in at its recorded value"
            )

        nvfp4_layers = _nvfp4_flashinfer_linears(self.module)
        unbaked = _unbaked_nvfp4_layers(nvfp4_layers)
        if unbaked:
            raise RuntimeError(
                f"{len(unbaked)} NVFP4 linear(s) report unbaked activation scales "
                f"(first: {unbaked[0]}); a scale still being calibrated would be frozen into the "
                f"graph at whatever value this capture saw, so this load runs eager"
            )

        entry.static = self._statics(live)
        # Writing into an inference tensor is only allowed inside inference mode.
        entry.inference_copy = self.plan is not None and any(t.is_inference() for t in entry.static)
        static_args, static_kwargs = _rebuild(entry.in_spec, entry.static)

        if nvfp4_layers:
            # Before the warm-up: a captured tuning launch would bake in the default tactic.
            from .diffusion_nvfp4_linear import nvfp4_prewarm
            nvfp4_prewarm(self.module, _prewarm_token_counts(live), logger = self.logger)

        refuse_if_exhausted(
            self.logger
        )  # before the warm-ups, and before the step skip's counters are saved
        if capture_refused_now():
            raise CaptureRefused("replica-side CUDA work still running; capture refused")
        await_outside_drained()
        # Under an offload placement the capture also records the copy-stream onloads (OffloadPlacement.record).
        record = call if self.placement is None else self.placement.recorder(call)
        # Side stream: a workspace first created during capture is only valid while recording.
        recapture = self.plan is not None and self.stats["captures"] > 0
        # An offloaded key re-recorded after its weights moved is already warm: it records like a planned re-capture.
        recapture = recapture or (
            self.placement is not None and key is not None and key in self._recorded
        )
        if self.placement is not None:
            self.placement.before_capture()  # anything the recorded copies land in exists before the capture
            self.valid_token = self.placement.token()
        # A pass-through step skip under the hooks counts every call; the warm-ups and the recording are not steps.
        skip = getattr(self.placement, "skip", None) if self.placement is not None else None
        skip_stats = (
            dict(skip.stats)
            if skip is not None and isinstance(getattr(skip, "stats", None), dict)
            else None
        )
        # Keys warmed by timed eager steps record at once; a side-stream warm-up would double activations.
        warm = (
            self.placement is not None
            and key is not None
            and int((getattr(self, "_judge", {}).get(key) or {}).get("seen", 0)) >= 1
        )
        warm = warm or (
            self.placement is None
            and key is not None
            and self.warmup > 0
            and real_warmup_enabled()
            and int(getattr(self, "_real_warm", {}).get(key, 0)) >= self.warmup
        )
        with _recording_step():
            if not recapture and not warm:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    # A re-record under a new offload placement only needs the forward to have run once on it.
                    again = self.placement is not None and self.stats["captures"] > 0
                    for _ in range(min(self.warmup, 1) if again else self.warmup):
                        record(*static_args, **static_kwargs)
                torch.cuda.current_stream().wait_stream(side)
            try:
                synchronize_outside_capture(
                    refuse = True
                )  # a device sync: never inside another thread's recording
            except CaptureRefused:
                if skip_stats is not None:
                    skip.stats = skip_stats
                raise

            graph = torch.cuda.CUDAGraph()
            # An explicit pool id, never None: a capture that fails must hand its pool back (retire_failed_capture).
            box = _pool_box()
            pool = box[0]
            pool_before = _pool_id_bytes(pool)
            if pool is None and callable(getattr(torch.cuda, "graph_pool_handle", None)):
                pool = torch.cuda.graph_pool_handle()
            recorder = _graph_without_flush if recapture else _graph_with_flush
            card = next((t.device.index for t in live if t.device.index is not None), None)
            if card is None:
                card = torch.cuda.current_device()
            try:
                # _capturing hands back any retired pool first, so the no-flush re-record never records beside one.
                with _capturing(card), recorder(graph, pool = pool, stream = _capture_stream(card)):
                    out = record(*static_args, **static_kwargs)
            except BaseException as exc:
                # A refusal is raised before the recording begins: the graph and the pool were never touched.
                if not isinstance(exc, CaptureRefused):
                    retire_failed_capture(graph, pool, exc)
                raise
            finally:
                if skip_stats is not None:
                    skip.stats = skip_stats
        self.stats["pool_bytes"] = _pool_bytes(graph)
        if _BATCH_ROWS.get() is None or _BATCH_ROWS.get()[0] <= 1:
            # This size-1 graph's own pool growth: what a batch-k graph of it is priced from (k times).
            grown = max(0, int(self.stats["pool_bytes"]) - int(pool_before))
            self.stats["size1_graph_bytes"] = max(
                int(self.stats.get("size1_graph_bytes", 0)), grown
            )
        if key is not None:
            self._recorded[key] = None
            while len(self._recorded) > 4 * self.max_graphs:
                self._recorded.pop(next(iter(self._recorded)))
        box = _pool_box()
        if box[0] is None:
            try:
                box[0] = graph.pool()
            except Exception:  # noqa: BLE001 - a private pool per graph still works
                pass

        entry.graph = graph
        # The pool this graph recorded into (see _drop_pool_if_unused), not whatever the box holds by now.
        entry.pool_token = pool if pool is not None else box[0]
        out_tensors: list = []
        entry.out_spec = _flatten(out, out_tensors)
        entry.out_tensors = out_tensors
        self.stats["captures"] += 1
        return entry


class GraphedCompiledCall(GraphedForward):
    """``GraphedForward`` for a whole-module compiled denoiser (SDXL U-Net): ``Module.__call__`` serves it from
    ``_compiled_call_impl`` and never reads ``forward``, so the replay sits there and captures the compiled callable."""

    def __init__(self, module: Any, **kwargs: Any) -> None:
        compiled = getattr(module, "_compiled_call_impl", None)
        # A static skip already fronting the compiled call (front_compiled_call at compile): graph what it fronts.
        from .diffusion_step_skip import fronting_skip

        skip = fronting_skip(module)
        if skip is not None:
            compiled = skip.compiled_call
        if compiled is None:
            raise RuntimeError(f"{type(module).__name__} is not whole-module compiled")
        self.compiled = compiled
        super().__init__(module, call = compiled, **kwargs)

    def install(self) -> "GraphedCompiledCall":
        self.module._compiled_call_impl = self
        # A static step skip in the forward slot would run inside the captured call: it goes in front instead.
        from .diffusion_step_skip import front_compiled_call

        front_compiled_call(self.module)
        return self

    def uninstall(self) -> "GraphedCompiledCall":
        if getattr(self.module, "_compiled_call_impl", None) is self:
            self.module._compiled_call_impl = self.compiled
        else:
            from .diffusion_step_skip import unfront_compiled_call
            unfront_compiled_call(self.module, self, self.compiled)
        return self


def _leaf_ptrs(t: Any, out: list) -> None:
    """Data pointers of ``t``'s storage, through tensor subclasses (torchao int8 / fp8 weights) to their inner tensors."""
    flatten = getattr(type(t), "__tensor_flatten__", None)
    if flatten is not None and type(t).__name__ not in ("Tensor", "Parameter"):
        try:
            names, _ctx = t.__tensor_flatten__()
            for name in names:
                _leaf_ptrs(getattr(t, name), out)
            return
        except Exception:  # noqa: BLE001 - fall through to the plain pointer
            pass
    try:
        out.append((str(t.device), int(t.data_ptr())))
    except Exception:  # noqa: BLE001
        out.append(("?", id(t)))


def _module_tensors(module: Any) -> list:
    seen: set = set()
    out: list = []
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            out.append(t)
    return out


class OffloadPlacement:
    """Where an offloaded denoiser's replay sits and when its recorded addresses stop holding.

    ``group``: diffusers block offload driven by the event-fenced prefetch (diffusion_offload_prefetch). The replay
    replaces the module's hooked forward, so ONE graph records the whole step: the copy-stream copies of every
    streamed group from its pinned host tensor into the graph pool, the compute stream waiting on each copy's event,
    and the resident groups read in place. The pool holds the in-flight window at fixed addresses, so it plays the
    prefetch ring. Recorded again when a residency change or a host-copy swap moves an address. A prompt encode's
    release / restore does not: once armed, the re-pin lands in each group's stable range (diffusion_stable_arena).

    ``model``: whole-module CPU offload (accelerate). By default only over the stable device range
    (``diffusion_stable_arena``: every onload lands at the same addresses, so render 1's recording replays on every
    later render); ``MODEL_OFFLOAD_CUDA_GRAPH_ENV=1`` records over the stock onloads instead (see
    ``MODEL_OFFLOAD_REASON``). The hook moves the module on the eager path first and the replay sits under it, in
    ``_old_forward``. Recorded again when the onload placed the weights at new addresses; after
    ``MODEL_MAX_INVALIDATIONS`` re-records the module runs eager (each re-record costs a forward)."""

    MODEL_MAX_INVALIDATIONS = 2
    # Past MODEL_MAX_INVALIDATIONS, re-recording continues while each recording serves at least this many replays.
    MODEL_MIN_REPLAYS_PER_RECORD = 4

    def __init__(self, module: Any, mode: str):
        self.module = module
        self.mode = mode
        self.prev: Any = None
        self._epoch: Any = None
        self._fp: Any = None
        self._checked: Any = object()
        self._why: Optional[str] = None
        self._tensors: Optional[list] = None
        # Set when a key's replay did not pay on a streaming placement: nothing records again for this load.
        self._declined: Optional[str] = None
        self._slots = False
        if mode == "group":
            self.prev = module.__dict__.get("forward")
        else:
            self.prev = getattr(module, "_old_forward", None)
        if self.prev is None:
            raise RuntimeError(f"no {mode} offload forward to wrap")
        self.eager = self.prev
        # A static step skip (diffusion_step_skip) the replay would bypass: see GraphedForward.__call__.
        self.skip: Any = self.prev if getattr(self.prev, "_unsloth_outer_forward", False) else None
        # Block offload: hook chain ends in the class forward, which the capture-safe rewrite replaces.
        self._inner: Any = None
        self._inner_stock: Any = None
        self._inner_safe: Any = None
        if mode == "group":
            safe = _plain_rewrite(module)
            if safe is not None:
                ref = _innermost_class_forward(module)
                if ref is None:
                    raise RuntimeError(
                        "the block-offload hook chain does not end in the class forward"
                    )
                self._inner, self._inner_stock = ref, ref.forward
                self._inner_safe = safe.__get__(module)

    def install(self, handle: Any) -> None:
        if self.mode == "group":
            if self._inner is not None:
                self._inner.forward = self._inner_safe
            self.module.__dict__["forward"] = handle
            self._hold_carry(True)
            self._enable_slots()
        else:
            self.module._old_forward = handle

    def _hold_carry(self, hold: bool) -> None:
        """A replay bypasses the prefetcher's onloads: nothing is carried into a forward that may replay."""
        try:
            pf = self._prefetcher()
            if pf is None:
                return
            pf.graph_hold = hold
            if hold and not getattr(pf, "active", False):
                pf.drop_carried()
        except Exception:  # noqa: BLE001
            pass

    def _prefetcher(self) -> Any:
        from .diffusion_offload_prefetch import module_prefetcher
        return module_prefetcher(self.module)

    def _enable_slots(self) -> None:
        """Streamed copies use the prefetcher's slot ring at fixed addresses outside the graph pool,
        so it holds one step's activations; UNSLOTH_DIFFUSION_STEP_GRAPH_SLOTS=0 disables."""
        if (os.environ.get(STEP_SLOTS_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
            return
        if not self.streams():
            return
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "enable_slots", None)):
                pf.enable_slots()
                self._slots = bool(getattr(pf, "slot_of", None))
        except Exception:  # noqa: BLE001
            self._slots = False

    def streams(self) -> bool:
        """Weights move each step (model offload or unpinned group); recording costs memory."""
        if self.mode != "group":
            return True
        try:
            from .diffusion_memory import _offload_groups
            pf = self._prefetcher()
            for group in _offload_groups(self.module):
                if getattr(group, "_unsloth_resident", False) or getattr(
                    group, "_unsloth_pinned_top", False
                ):
                    continue
                if pf is None or pf.owns(group):
                    return True
        except Exception:  # noqa: BLE001 - unknown layout: treat as streaming (the stricter speed check)
            return True
        return False

    def before_capture(self) -> None:
        """Allocate the slot ring before a capture, so no slot is carved out of the graph pool."""
        if not self._slots:
            return
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "materialize_slots", None)):
                pf.materialize_slots()
        except Exception:  # noqa: BLE001
            pass

    def release_slots(self) -> None:
        """Hand the slot ring back once no graph of this placement can replay into it."""
        if not self._slots:
            return
        self._slots = False
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "drop_slots_after_forward", None)):
                pf.drop_slots_after_forward()
        except Exception:  # noqa: BLE001
            pass

    def decline(self, reason: str) -> bool:
        """No more recording on this placement for the load; hand the slot ring back after the current forward.
        True when that drop is scheduled (it then also releases the dropped graphs' pool)."""
        self._declined = reason
        if not self._slots:
            return False
        self._slots = False
        try:
            pf = self._prefetcher()
            if pf is not None and callable(getattr(pf, "drop_slots_after_forward", None)):
                pf.drop_slots_after_forward()
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def uninstall(self, handle: Any) -> None:
        if self._inner is not None and self._inner.forward is self._inner_safe:
            self._inner.forward = self._inner_stock
        if self.mode == "group":
            if self.module.__dict__.get("forward") is handle:
                self.module.__dict__["forward"] = self.prev
                self._hold_carry(False)
            return
        if getattr(self.module, "_old_forward", None) is handle:
            self.module._old_forward = self.prev
        if self.module.__dict__.get("forward") is handle:
            self.module.__dict__.pop("forward", None)

    def recorder(self, call: Any) -> Any:
        """What a capture records for ``call``: under block offload the prefetcher's copies join the capture."""
        if self.mode != "group":
            return call

        def record(*args: Any, **kwargs: Any) -> Any:
            from .diffusion_offload_prefetch import module_prefetcher

            pf = module_prefetcher(self.module)
            pf.begin()  # inside the capture: forks the copy stream in, and end() joins it back
            try:
                return call(*args, **kwargs)
            finally:
                pf.end()

        return record

    def record(self, *args: Any, **kwargs: Any) -> Any:
        return self.recorder(self.eager)(*args, **kwargs)

    def recover(self) -> None:
        """After a failed capture: drop what the prefetcher queued inside it (see GroupPrefetcher.abandon)."""
        if self.mode != "group":
            return
        try:
            from .diffusion_offload_prefetch import module_prefetcher
            pf = module_prefetcher(self.module)
            if pf is not None:
                pf.abandon()
        except Exception:  # noqa: BLE001
            pass

    def _fingerprint(self) -> tuple:
        out: list = []
        if self.mode == "group":
            from .diffusion_memory import _offload_groups
            from .diffusion_offload_prefetch import _group_tensors, module_prefetcher

            pf = module_prefetcher(self.module)
            for group in _offload_groups(self.module):
                resident = bool(getattr(group, "_unsloth_resident", False))
                out.append((id(group), resident, bool(pf is not None and pf.owns(group))))
                tensors = _group_tensors(group)
                if resident:
                    for t in tensors:
                        _leaf_ptrs(t, out)
                else:
                    cpu = getattr(group, "cpu_param_dict", None) or {}
                    for t in tensors:
                        _leaf_ptrs(cpu.get(t, t), out)
            # Streamed copies land in the slot ring when it is on: a new ring is a new placement.
            raws = getattr(pf, "slot_raw", None) or {}
            out.append(("slots", tuple((i, int(raws[i].data_ptr())) for i in sorted(raws))))
            return tuple(out)
        # Read every call: Module.to() replaces buffers with new tensor objects on offload/onload.
        for t in _module_tensors(self.module):
            _leaf_ptrs(t, out)
        return tuple(out)

    def token(self) -> Any:
        if self.mode == "group":
            from .diffusion_offload_prefetch import placement_epoch

            epoch = placement_epoch()
            if epoch != self._epoch:
                self._epoch = epoch
                self._fp = self._fingerprint()
            return self._fp
        self._fp = self._fingerprint()
        return self._fp

    def refusal(self, stats: dict) -> Optional[str]:
        if getattr(self, "_declined", None) is not None:
            return self._declined
        if self.mode == "model":
            failed = _stable_range_failure(getattr(self, "module", None))
            if failed is not None:
                # Stock onloads land at new addresses each render. Retire, not detach: placed weights still sit in the range.
                _retire_stable_range(self.module)
                self._declined = (
                    f"the stable device range failed ({failed}); " + MODEL_OFFLOAD_REASON
                )
                return self._declined
            # Judged once per placement, when the weights land (the fingerprint is rebuilt every call).
            if self._checked != self._fp:
                self._checked = self._fp
                moves = int(stats.get("invalidations", 0))
                records = max(1, int(stats.get("captures", 0)))
                replays = int(stats.get("replays", 0))
                self._why = None
                if (
                    moves >= self.MODEL_MAX_INVALIDATIONS
                    and replays < self.MODEL_MIN_REPLAYS_PER_RECORD * records
                ):
                    self._why = (
                        f"model offload placed the weights at new addresses on {moves} onloads, {replays} replays "
                        f"over {records} recordings; each re-record costs a step"
                    )
            return self._why
        if self._checked is not self._fp:
            from .diffusion_offload_prefetch import capture_refusal
            self._checked = self._fp
            self._why = capture_refusal(self.module)
        return self._why


def _skips_planned(skip: Any) -> bool:
    plan = getattr(skip, "plan", None) or ()
    return bool(plan) and not all(plan)


def offload_placement(module: Any) -> tuple:
    """``(placement, refusal)`` for ``module`` as it is placed now: ``(None, None)`` when nothing moves it."""
    try:
        subs = [m for m in module.modules() if m is not module]
    except Exception:  # noqa: BLE001
        subs = []
    if any(getattr(m, "_hf_hook", None) is not None for m in subs):
        return None, (
            "leaf-level (sequential) offload uploads every layer synchronously from host memory inside the forward"
        )
    hf_hook = getattr(module, "_hf_hook", None)
    compiled = getattr(module, "_compiled_call_impl", None) is not None
    if hf_hook is not None:
        if compiled:
            return (
                None,
                "whole-module compiled denoiser: the CPU-offload hook runs inside its compiled call",
            )
        if "CpuOffload" not in type(hf_hook).__name__:
            return (
                None,
                f"offload hook {type(hf_hook).__name__} moves the weights inside the forward",
            )
        # Other refusals first: stable range swaps onloads, so it attaches only to recorders.
        why = _forward_refusal(module, "model")
        if why is not None:
            return None, why
        arena = None
        if not model_offload_graphs_requested():
            # Recorded only where the onloads land at the same addresses every render (diffusion_stable_arena).
            from .diffusion_stable_arena import attach
            arena, _arena_why = attach(module)
            if arena is None:
                return None, MODEL_OFFLOAD_REASON
        try:
            return OffloadPlacement(module, "model"), None
        except Exception as exc:  # noqa: BLE001
            if arena is not None:
                arena.detach()
            return None, f"model offload hook unreadable ({exc})"
    registry = getattr(module, "_diffusers_hook", None)
    hooks = getattr(registry, "hooks", None) or {}
    if not any("offload" in str(key) for key in hooks):
        return None, None
    if compiled:
        return (
            None,
            "whole-module compiled denoiser: the offload hooks run inside its compiled call",
        )
    try:
        from diffusers.hooks import group_offloading as go
        name = getattr(go, "_GROUP_OFFLOADING", "group_offloading")
    except Exception:  # noqa: BLE001
        name = "group_offloading"
    if name not in hooks:
        return None, "an offload hook other than block offloading moves the weights"
    skip = None
    for fn_ref in getattr(registry, "_fn_refs", None) or ():
        if getattr(getattr(fn_ref, "forward", None), "_unsloth_outer_forward", False):
            # Static skip: skip-planning renders run every call through it; others pass through to graph replay.
            skip = fn_ref.forward
    why = _forward_refusal(module, "group")
    if why is not None:
        return None, why
    from .diffusion_offload_prefetch import capture_refusal

    why = capture_refusal(module)
    if why is not None and "background" not in why:
        return None, why
    try:
        placement = OffloadPlacement(module, "group")
    except Exception as exc:  # noqa: BLE001
        return None, f"block offload hook unreadable ({exc})"
    placement.skip = skip
    return placement, None


def _detach_stable_range(module: Any) -> None:
    """Give a module whose step stays eager its stock onload back (no stable range, no per-offload card sync)."""
    try:
        from .diffusion_stable_arena import arena_of
        arena = arena_of(module)
        if arena is not None:
            arena.detach()
    except Exception:  # noqa: BLE001
        pass


def _retire_stable_placement(placement: Any) -> None:
    """Drop stable addresses once no graph can replay (poisoned/freed); re-arm attaches afresh."""
    if placement is None:
        return
    if placement.mode == "model":
        _detach_stable_range(placement.module)
        return
    try:
        from .diffusion_stable_arena import drop_stable_repin
        drop_stable_repin(placement.module)
    except Exception:  # noqa: BLE001
        pass


def _stable_range_failure(module: Any) -> Optional[str]:
    try:
        from .diffusion_stable_arena import arena_of
        arena = arena_of(module)
        return None if arena is None else arena.failed
    except Exception:  # noqa: BLE001
        return None


def _retire_stable_range(module: Any) -> None:
    try:
        from .diffusion_stable_arena import arena_of
        arena = arena_of(module)
        if arena is not None:
            arena.retire()
    except Exception:  # noqa: BLE001
        pass


def _stable_range(module: Any) -> bool:
    try:
        from .diffusion_stable_arena import arena_of
        return arena_of(module) is not None
    except Exception:  # noqa: BLE001
        return False


def _forward_refusal(module: Any, mode: str) -> Optional[str]:
    """Why ``module``'s own forward cannot record under ``mode`` offload, else None (see GraphedForward.__init__)."""
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        safe, why = _capture_safe(type(module))
    except Exception:  # noqa: BLE001 - the stock forward is then what records
        return None
    if why is not None:
        return why
    if safe is None:
        return None
    if getattr(safe, "__unsloth_graph_plan__", None) is not None:
        if getattr(safe, "__unsloth_graph_placed__", None) is None:
            return f"{type(module).__name__} plans its graph steps but has no entry through the offload hooks"
        return None
    if mode == "group" and _innermost_class_forward(module) is None:
        return (
            f"{type(module).__name__} records only through its capture-safe forward, and the block-offload hook "
            "chain does not end in the class forward it would replace"
        )
    return None


def _plain_rewrite(module: Any) -> Any:
    """``module``'s capture-safe class-forward rewrite when it is a plain one (not a planned step), else None."""
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        safe, _ = _capture_safe(type(module))
    except Exception:  # noqa: BLE001
        return None
    if safe is None or getattr(safe, "__unsloth_graph_plan__", None) is not None:
        return None
    return safe


def _innermost_class_forward(module: Any) -> Any:
    registry = getattr(module, "_diffusers_hook", None)
    refs = getattr(registry, "_fn_refs", None) or ()
    if not refs:
        return None
    ref = refs[0]
    if getattr(ref, "original_forward", None) is not None:
        return None  # a hook with its own new_forward owns the innermost call
    current = getattr(ref, "forward", None)
    owner = next((k for k in type(module).__mro__ if "forward" in vars(k)), None)
    if owner is None or getattr(current, "__self__", None) is not module:
        return None
    if getattr(current, "__func__", None) is not vars(owner)["forward"]:
        return None
    return ref


def arm_after_placement(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm the graphs once placement is final, in the slot each denoiser's offload leaves free. Returns
    ``(handles, reason)``: the reason names what kept a denoiser eager, else what engaged."""
    uninstall_all(getattr(pipe, "_unsloth_cuda_graphs", ()) or (), logger = logger)
    handles: list = []
    refusals: list = []
    modes: list = []
    from .diffusion_layer_split import is_layer_split

    for module in _denoiser_modules(pipe):
        if is_layer_split(module):
            refusals.append(f"{type(module).__name__}: denoiser is split across cards")
            continue
        placement, why = offload_placement(module)
        if why is not None:
            refusals.append(f"{type(module).__name__}: {why}")
            continue
        if placement is None and getattr(pipe, "_unsloth_cuda_graph_offload_only", False):
            refusals.append(
                f"{type(module).__name__}: the family records offloaded steps only, and it stays resident"
            )
            continue
        try:
            if placement is None:
                kind = (
                    GraphedCompiledCall
                    if getattr(module, "_compiled_call_impl", None) is not None
                    else GraphedForward
                )
                handles.append(kind(module, max_graphs = max_graphs, logger = logger).enable())
                modes.append("resident")
            else:
                handles.append(
                    GraphedForward(
                        module, max_graphs = max_graphs, logger = logger, placement = placement
                    ).enable()
                )
                stable = placement.mode == "model" and _stable_range(module)
                modes.append("model_stable" if stable else placement.mode)
                if placement.mode == "group":
                    # Encode re-pins now land at the addresses this step records (diffusion_stable_arena.GroupRange).
                    from .diffusion_stable_arena import want_stable_repin
                    want_stable_repin(module)
        except Exception as exc:  # noqa: BLE001
            if placement is not None and placement.mode == "model":
                _detach_stable_range(module)
            refusals.append(f"{type(module).__name__}: {exc}")
    installed = tuple(handles)
    try:
        pipe._unsloth_cuda_graphs = installed
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "handle stash", exc)
    if refusals:
        reason = "; ".join(refusals)
    else:
        labels = {
            "resident": "resident",
            "group": "block-streamed (copies recorded in the graph)",
            "model": "model offload",
            "model_stable": "model offload over a stable device range",
        }
        reason = "captured per input shape: " + ", ".join(sorted({labels.get(m, m) for m in modes}))
    if logger is not None:
        logger.info(
            "diffusion.cuda_graph: armed on %d denoiser module(s) after placement (%s)",
            len(installed),
            reason,
        )
    return installed, reason


def _denoiser_modules(pipe: Any) -> list:
    from .diffusion_speed import _denoiser_dits, _denoiser_unet

    dits = _denoiser_dits(pipe)
    if dits:
        return dits
    unet = _denoiser_unet(pipe)
    return [unet] if unet is not None else []


PROMPT_LENGTH_REASON = "text length follows the prompt"


def prompt_length_reason(pipe: Any, family: Any = None) -> Optional[str]:
    """Why a denoiser whose text length follows the prompt (diffusion_dynamic_text.PROMPT_LENGTH_POLICY) records no
    per-block graphs, else None. A block's graph keys on its inputs, which carry the text stream, so every new prompt
    length would record every block again on that render; its whole-step graph instead records the first render's
    lengths only (``GraphedForward.length_sealed``). ``UNSLOTH_DIFFUSION_CUDA_GRAPH_FAMILIES`` and
    ``UNSLOTH_DIFFUSION_CUDA_GRAPH_EVERY_LENGTH`` still force it."""
    if _family_forced(family) or _every_length():
        return None
    try:
        from .diffusion_dynamic_text import prompt_length_varies  # noqa: PLC0415
        names = sorted(
            {type(m).__name__ for m in _denoiser_modules(pipe) if prompt_length_varies(m)}
        )
    except Exception:  # noqa: BLE001 - an unanswerable probe keeps today's graphs
        return None
    if not names:
        return None
    return (
        f"{PROMPT_LENGTH_REASON} ({', '.join(names)}): per-block graphs would record every block again on each "
        f"new prompt length, so the blocks run compiled ungraphed"
    )


# torch < 2.12, sm_89: row-wise fp8 _scaled_mm (stream-K) memsets its barrier outside capture, so replay hangs (pytorch/pytorch#177573).
FP8_ROWWISE_STREAMK_CAPS = ((8, 9),)
FP8_ROWWISE_STREAMK_FIXED = (2, 12)


def _fp8_rowwise_weight(t: Any, activation: bool = False) -> bool:
    """A torchao fp8 weight whose linear runs the row-wise ``_scaled_mm`` (dynamic activation, per-row scale)."""
    name = type(t).__name__
    if name == "LinearActivationQuantizedTensor":
        inner = getattr(t, "original_weight_tensor", None)
        return inner is not None and _fp8_rowwise_weight(inner, activation = True)
    if name == "Float8Tensor":
        scale = getattr(t, "scale", None)
        dynamic = activation or getattr(t, "act_quant_kwargs", None) is not None
    elif name == "AffineQuantizedTensor" and activation:
        impl = getattr(t, "tensor_impl", None)
        if "Float8" not in type(impl).__name__:
            return False
        scale = getattr(impl, "scale", None)
        dynamic = True
    else:
        return False
    try:
        return bool(dynamic and scale is not None and int(scale.numel()) > 1)
    except Exception:  # noqa: BLE001
        return False


def fp8_rowwise_capture_refusal(modules: Any, device: Any = None) -> Optional[str]:
    """Why ``modules`` must not be graphed on ``device``, else None (also None when unreadable)."""
    if sys.platform == "win32":
        return None  # not built on Windows
    try:
        torch = _torch()
        from .rocm_bf16 import is_rocm_torch

        if is_rocm_torch(torch):
            return None
        major, minor = (int(x) for x in str(torch.__version__).split("+")[0].split(".")[:2])
        if (major, minor) >= FP8_ROWWISE_STREAMK_FIXED:
            return None
        cap = tuple(torch.cuda.get_device_capability(device))
    except Exception:  # noqa: BLE001
        return None
    caps: dict = {}

    def _weight_cap(w: Any) -> Any:
        # Layer split: an sm_89 secondary card hangs too.
        if cap in FP8_ROWWISE_STREAMK_CAPS:
            return cap
        d = getattr(w, "device", None)
        if getattr(d, "type", None) != "cuda":
            return cap
        key = str(d)
        if key not in caps:
            try:
                caps[key] = tuple(torch.cuda.get_device_capability(d))
            except Exception:  # noqa: BLE001
                caps[key] = None
        return caps[key]

    for module in modules or ():
        try:
            weights = list(module.parameters())
        except Exception:  # noqa: BLE001
            continue
        for w in weights:
            if not _fp8_rowwise_weight(w):
                continue
            hit = _weight_cap(w)
            if hit in FP8_ROWWISE_STREAMK_CAPS:
                return (
                    f"the row-wise fp8 GEMM on this sm_{hit[0]}{hit[1]} card cannot replay in a CUDA graph under "
                    f"torch {major}.{minor} (it hangs; fixed in torch {FP8_ROWWISE_STREAMK_FIXED[0]}."
                    f"{FP8_ROWWISE_STREAMK_FIXED[1]}, pytorch/pytorch#177573)"
                )
    return None


def _graph_device(target: Any) -> Any:
    device = getattr(target, "torch_device", None)
    return device if isinstance(device, str) and device.startswith("cuda") else None


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
    offloaded: bool = False,
) -> tuple[bool, str]:
    """Whether this load may be graphed, and a short reason when it may not. Cheapest first. ``offloaded``: the
    denoiser moves (graphs armed after placement), which a family can opt into on its own (``offload_cuda_graph``)."""
    if cuda_graph_disabled():
        return False, _disabled_reason()

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

    from .diffusion_layer_split import is_layer_split

    if any(is_layer_split(m) for m in _denoiser_modules(pipe)):
        return False, "denoiser is split across cards"

    # A whole-compiled U-Net serves from ``_compiled_call_impl``: GraphedCompiledCall captures there.
    if not _denoiser_dits(pipe) and _denoiser_unet(pipe) is None:
        return False, "no denoiser transformer"

    # Sage (pip or hub build) under a replayed graph renders noise (FLUX.1-schnell, A100); ungraphed is correct.
    if any(
        getattr(m, "_unsloth_attention_backend", None) in ("sage", "sage_hub")
        for m in _denoiser_dits(pipe)
    ):
        return False, "SageAttention is not CUDA-graph safe"
    # a denoiser that routes per denoise step (H3's INT8 attention exact start) would replay one route for all
    unsafe = next((r for r in (_graph_unsafe(m) for m in _denoiser_dits(pipe)) if r), None)
    if unsafe:
        return False, unsafe

    try:
        cuda = getattr(_torch(), "cuda", None)
        if cuda is None or not hasattr(cuda, "CUDAGraph") or not cuda.is_available():
            return False, "torch.cuda unavailable"
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "availability probe", exc)
        return False, "torch.cuda unavailable"

    wanted = bool(getattr(family, "supports_cuda_graph", family_default))
    wanted = wanted or (offloaded and bool(getattr(family, "offload_cuda_graph", False)))
    if not wanted and not _family_forced(family):
        return False, str(getattr(family, "cuda_graph_decline", None) or "family opts out")

    # Host-syncing forward with no capture-safe rewrite: decline up front, it would run eager.
    try:
        from .diffusion_capture_safe import resolve as _capture_safe  # noqa: PLC0415
        for module in _denoiser_dits(pipe):
            _, why = _capture_safe(type(module))
            if why:
                return False, why
    except Exception as exc:  # noqa: BLE001 - the capture itself still poisons safely
        _warn(logger, "capture-safe probe", exc)

    why = fp8_rowwise_capture_refusal(_denoiser_modules(pipe), _graph_device(target))
    if why is not None:
        return False, why

    return True, "eligible"


def install_cuda_graphs(
    pipe: Any,
    *,
    logger: Any = None,
    max_graphs: int = MAX_GRAPHS_PER_MODULE,
) -> tuple:
    """Arm one ``GraphedForward`` per denoiser module; the first denoising step captures."""
    handles: list = []
    for module in _denoiser_modules(pipe):
        try:
            kind = (
                GraphedCompiledCall
                if getattr(module, "_compiled_call_impl", None) is not None
                else GraphedForward
            )
            handles.append(kind(module, max_graphs = max_graphs, logger = logger).enable())
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


def note_render_steps(handles: Any, steps: Any) -> None:
    """Of the render's denoiser steps, a key's first real ones are warm-ups only if more than
    WARMUP_ITERS follow, so it records now; None or several denoisers keep side-stream warm-ups."""
    handles = tuple(handles or ())
    try:
        count = int(steps) if steps is not None and len(handles) == 1 else None
    except Exception:  # noqa: BLE001 - an unreadable count is an unknown one
        count = None
    for handle in handles:
        try:
            handle._render_steps = count
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def seal_lengths(handles: Any) -> None:
    """A render ended: a denoiser whose text length follows the prompt records no new prompt length from here on
    (until ``reset``). Covers a first render that recorded but never replayed (an offloaded key still being timed)."""
    for handle in handles or ():
        try:
            handle._render_steps = None  # the next render says its own
        except Exception:  # noqa: BLE001 - optimisation only
            pass
        try:
            real = getattr(handle, "_real_warm", None)
            if real:
                # A key whose real warm-ups the render cut short records next time: side-stream
                # warm-ups if it ran fewer than a capture needs, else at once if it ran them all.
                handle._real_warm_closed.update(
                    key
                    for key, n in real.items()
                    if 0 < n <= handle.warmup and key not in handle.cache
                )
        except Exception:  # noqa: BLE001 - optimisation only
            pass
        try:
            if not getattr(handle, "length_varies", False):
                continue
            # Only a scope recorded since reset is sealed; warm-up-only renders keep it open.
            for scope in handle._render_scopes:
                if handle._scope_captures.get(scope, 0) > 0:
                    handle._sealed_scopes.add(scope)
            handle._render_scopes.clear()
            handle._open_warmups.clear()
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def set_bypass(handles: Any, on: bool) -> None:
    """Run eager without dropping graphs, for every handle; inside ``forced_eager`` they stay bypassed."""
    on = bool(on) or eager_forced()
    for handle in handles or ():
        try:
            handle.set_bypass(on)
        except Exception:  # noqa: BLE001 - optimisation only
            pass


def wrappers_over(model: Any) -> tuple:
    try:
        ids = {id(m) for m in model.modules()} if hasattr(model, "modules") else {id(model)}
    except Exception:  # noqa: BLE001
        ids = {id(model)}
    return tuple(w for w in tuple(_LIVE_WRAPPERS) if id(getattr(w, "module", None)) in ids)


def _holds_graphs(handle: Any) -> bool:
    if getattr(handle, "cache", None):
        return True
    fallback = getattr(handle, "fallback_handle", None)
    return fallback is not None and fallback is not handle and _holds_graphs(fallback)


def reset_all(handles: Any, strict: bool = False) -> None:
    """Drop every captured graph, for every handle (the weights changed).

    Drops the pool token with the last graph: a stale token dies on the allocator's
    "use_count > 0 INTERNAL ASSERT FAILED", raised after ``torch.cuda.graph`` entered its side
    stream, so the thread is left off the default stream too. ``strict``: raise when a handle failed
    to reset or still holds a graph (a caller about to move weights must not leave one that could replay)."""
    failed: list = []
    for handle in handles or ():
        try:
            handle.reset()
        except Exception as exc:  # noqa: BLE001 - optimisation only, unless strict
            failed.append(f"{type(handle).__name__}: {type(exc).__name__}: {exc}")
            continue
        if strict and _holds_graphs(handle):
            failed.append(f"{type(handle).__name__}: still holds a captured graph")
    _drop_pool_if_unused()
    if strict and failed:
        raise RuntimeError("could not drop every captured CUDA graph: " + "; ".join(failed))


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


_REFUSALS = ("refused_float", "refused_host_tensor", "refused_object", "refused_branch")


def branch_decline_reason(exc: BaseException) -> str:
    """Why a branch-key hook declined a call: its exception's ``graph_decline`` text, else the exception type."""
    reason = getattr(exc, GRAPH_DECLINE_ATTR, None)
    return str(reason) if reason else f"declined: branch key unreadable ({type(exc).__name__})"


def stats(handles: Any) -> dict:
    """JSON-safe aggregate over the handles, for the status payload."""
    out = {
        "graphs": 0,
        "captures": 0,
        "replays": 0,
        "eager_calls": 0,
        "fallbacks": 0,
        "cap_skips": 0,
        "invalidations": 0,
        "pool_bytes": 0,
        "planned_eager": 0,
        "shape_warmups": 0,
        "evictions": 0,
        "speed_eager": 0,
        **{field: 0 for field in _REFUSALS},
        "placements": [],
        "poisoned": False,
        "capture_error": None,
        # Offloaded only: the measured eager step and replay the speed check judged (ms), once judged.
        "eager_ms": None,
        "replay_ms": None,
    }
    for handle in handles or ():
        try:
            out["graphs"] += len(handle.cache)
            for field in (
                "captures",
                "replays",
                "eager_calls",
                "fallbacks",
                "cap_skips",
                "invalidations",
                "planned_eager",
                "shape_warmups",
                "evictions",
                "speed_eager",
                *_REFUSALS,
            ):
                out[field] += int(handle.stats.get(field, 0))
            out["pool_bytes"] = max(out["pool_bytes"], int(handle.stats.get("pool_bytes", 0)))
            if out["eager_ms"] is None and handle.stats.get("eager_ms") is not None:
                out["eager_ms"] = float(handle.stats["eager_ms"])
                out["replay_ms"] = float(handle.stats["replay_ms"])
            placement = getattr(handle, "placement", None)
            out["placements"].append("resident" if placement is None else placement.mode)
            if placement is not None and placement.mode == "model":
                from .diffusion_stable_arena import arena_of
                arena = arena_of(placement.module)
                if arena is not None:
                    out["stable_range"] = arena.describe()
            if placement is not None and placement.mode == "group":
                from .diffusion_stable_arena import repin_stats
                repin = repin_stats(placement.module)
                if repin is not None:
                    out["stable_repin"] = repin
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


def engagement(handles: Any) -> dict:
    """Counters for the status payload, so engaged paths can be told from never-recompiling ones:
    this load's captures, replays and ungraphed denoiser calls, plus process-wide dynamo frame
    and graph-break totals (a new frame or break after first render is a recompile or break)."""
    # A whole-step graph that dropped a key or failed hands its calls to per-block graphs it holds as
    # ``fallback_handle`` (``GraphedForward._engage_fallback``): those count too.
    units: list = []
    for handle in handles or ():
        units.append(handle)
        nested = getattr(handle, "fallback_handle", None)
        if nested is not None and nested is not handle:
            units.append(nested)
    s = stats(units)
    out = {
        field: s[field]
        for field in ("graphs", "captures", "replays", "eager_calls", "invalidations", "fallbacks")
    }
    out["segment_captures"] = out["segment_replays"] = 0
    for handle in units:
        for segment in getattr(handle, "segments", None) or ():
            try:
                out["segment_captures"] += int(segment.stats.get("segment_captures", 0))
                out["segment_replays"] += int(segment.stats.get("segment_replays", 0))
            except Exception:  # noqa: BLE001
                pass
    out["dynamo_frames"] = out["graph_breaks"] = None
    if "torch._dynamo" in sys.modules:
        try:
            from torch._dynamo.utils import counters
            out["dynamo_frames"] = int(counters["stats"].get("unique_graphs", 0))
            out["graph_breaks"] = int(sum(counters["graph_break"].values()))
        except Exception:  # noqa: BLE001
            pass
    return out


def never_engaged(handles: Any) -> Optional[str]:
    """Why the armed graphs never replayed a step; None before the first call or once any engaged."""
    if not handles:
        return None
    reasons = [h.why_off() for h in handles if callable(getattr(h, "why_off", None))]
    if reasons and len(reasons) == len(handles):
        return None if any(r is None for r in reasons) else "; ".join(reasons)
    s = stats(handles)
    refused = [
        str((h.capture_error or {}).get("msg"))
        for h in handles
        if (getattr(h, "capture_error", None) or {}).get("type") == "Refused"
    ]
    if refused and not s["captures"] and not s["replays"]:
        return f"armed, but every denoiser call so far ran eager: {refused[0]}"
    if refused and not s["graphs"]:
        return f"captured, then dropped: {refused[0]}; denoiser steps now run eager"
    if all(getattr(h, "poisoned", False) for h in handles):
        error = s["capture_error"] or {}
        return f"capture failed ({error.get('type') or 'error'}); every denoiser step runs eager"
    if s["captures"] or s["replays"] or not s["eager_calls"]:
        return None
    if s["eager_calls"] == s["speed_eager"]:
        # Every eager call so far was an offloaded step timed for the speed check, before its first capture.
        return None
    parts = [
        f"{s[field]} {label}"
        for field, label in (
            ("refused_object", "with a non-tensor argument"),
            ("refused_float", "with a float argument"),
            ("refused_host_tensor", "with a host tensor"),
            ("cap_skips", "past the graph cap"),
        )
        if s[field]
    ]
    if s["refused_branch"]:
        declined = next(
            (str(h.branch_refusal) for h in handles if getattr(h, "branch_refusal", None)),
            "declined by the denoiser's branch key",
        )
        parts.append(f"{s['refused_branch']} {declined}")
    detail = ", ".join(parts) if parts else "bypassed"
    return f"armed, but all {s['eager_calls']} denoiser call(s) so far ran eager ({detail})"


def live_status(resolved: Any, speed_optims: Any, handles: Any) -> tuple:
    """Copies, never mutates: the load-time record stays as recorded."""
    why = never_engaged(handles)
    optims = list(speed_optims or ())
    if why is None:
        return resolved, optims
    optims = [o for o in optims if o != "cuda_graph"]
    if isinstance(resolved, dict) and isinstance(resolved.get("cuda_graph"), dict):
        resolved = {
            **resolved,
            "cuda_graph": {**resolved["cuda_graph"], "value": "off", "reason": why},
        }
    return resolved, optims


# Per-block graphs (diffusion_block_graph) for the loads the whole-forward recording cannot hold.
WHOLE_REASON = "denoiser step captured per input shape, replayed bit-identically"


def _block_basics(
    target: Any, family: Any, cache_engaged: bool, speed_mode: str, family_default: bool
) -> Optional[str]:
    """The whole-forward checks that also bind per-block recording (everything except offload and capture-safety)."""
    if cuda_graph_disabled():
        return _disabled_reason()
    if getattr(target, "device", None) != "cuda":
        return f"device is {getattr(target, 'device', None) or 'unknown'}"
    backend = getattr(target, "backend", "cuda")
    if backend != "cuda":
        return f"backend is {backend}"
    if cache_engaged:
        return "step cache active"
    mode = str(speed_mode or "").strip().lower()
    if mode not in ("default", "max"):
        return f"speed tier {mode or 'off'}"
    if not bool(getattr(family, "supports_cuda_graph", family_default)):
        return "family opts out"
    try:
        torch = _torch()
        if not torch.cuda.is_available() or not hasattr(torch.cuda, "graph_pool_handle"):
            return "torch.cuda unavailable"
    except Exception:  # noqa: BLE001
        return "torch.cuda unavailable"
    return None


def arm_block_graphs(
    pipe: Any,
    applied: dict,
    *,
    target: Any,
    family: Any,
    hooked: bool,
    pinned: bool = False,
    cache_engaged: bool = False,
    speed_mode: str = "default",
    family_default: bool = True,
    logger: Any = None,
) -> tuple:
    """Whole forward if it holds, else per block; offloaded DiTs need UNSLOTH_DIFFUSION_BLOCK_GRAPHS=1."""
    handles = tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
    whole = [h for h in handles if isinstance(h, GraphedForward)]
    prior = str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
    step = [h for h in whole if h.placement is not None or h.plan is not None]
    split = _layer_split_pipe(pipe)
    if split:
        # a whole-step capture records one card; a split's blocks record per block on their own card
        step = []
        _rehome_after_layer_split(pipe, logger)
    if whole and len(step) == len(whole):
        # The whole step stays primary; where per-block graphs are the default they arm for the calls it leaves eager.
        pipe._unsloth_cuda_graph_mode = "step"
        why = _block_basics(
            target, family, cache_engaged, speed_mode, family_default
        ) or prompt_length_reason(pipe, family)
        if why is None:
            _attach_block_fallback(pipe, step, target, hooked, pinned, logger)
        return handles
    if not split and not hooked and (whole or "not capture-safe" not in prior):
        # Nothing moves the denoiser: the speed layer's decision stands, except a forward that is not capture-safe,
        # whose blocks still record.
        return handles
    why = (
        _block_basics(target, family, cache_engaged, speed_mode, family_default)
        # Ahead of the prompt-length reason: an env var forces past that one, never this.
        or fp8_rowwise_capture_refusal(_denoiser_modules(pipe), _graph_device(target))
        or (None if split else prompt_length_reason(pipe, family))
    )
    if whole:
        uninstall_all(whole, logger = logger)
    pipe._unsloth_cuda_graph_mode = None
    if why is not None:
        _set_reason(pipe, why)
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        return ()
    try:
        from .diffusion_block_graph import (
            BLOCK_GRAPHS_ENV,
            OPT_IN_REASON,
            block_graphs_disabled,
            block_graphs_requested,
            install_block_graphs,
        )
        from .diffusion_speed import _denoiser_dits
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "block graph import", exc)
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        _set_reason(pipe, "offload active" if hooked else prior or "block graphs unavailable")
        return ()
    if any(
        getattr(t, "_unsloth_attention_backend", None) in ("sage", "sage_hub")
        for t in _denoiser_dits(pipe)
    ):
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        _set_reason(pipe, "SageAttention is not CUDA-graph safe")
        return ()
    unsafe = next((r for r in (_graph_unsafe(t) for t in _denoiser_dits(pipe)) if r), None)
    if unsafe:
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        _set_reason(pipe, unsafe)
        return ()
    model_offload = any(getattr(t, "_hf_hook", None) is not None for t in _denoiser_dits(pipe))
    stays = (not hooked or bool(pinned)) and not model_offload
    if block_graphs_disabled() or not (stays or block_graphs_requested()):
        applied["cuda_graph"] = False
        pipe._unsloth_cuda_graphs = ()
        if block_graphs_disabled():
            _set_reason(pipe, "disabled by UNSLOTH_DIFFUSION_BLOCK_GRAPHS=0")
        else:
            # Keep the whole step's own refusal visible (its opt-in, or why the opt-in still could not record).
            if model_offload:
                whole_why = prior if prior and prior != "offload active" else MODEL_OFFLOAD_REASON
                _set_reason(pipe, whole_why + "; per block: set " + BLOCK_GRAPHS_ENV + "=1")
            else:
                _set_reason(pipe, OPT_IN_REASON)
        return ()
    armed: list = []
    reasons: list = []
    for transformer in _denoiser_dits(pipe):
        try:
            handle, reason = install_block_graphs(
                transformer, device = getattr(target, "torch_device", None), logger = logger
            )
        except Exception as exc:  # noqa: BLE001
            _warn(logger, "block graph install", exc)
            handle, reason = None, f"install failed ({type(exc).__name__})"
        if handle is not None:
            armed.append(handle)
        else:
            reasons.append(reason)
    pipe._unsloth_cuda_graphs = tuple(armed)
    applied["cuda_graph"] = bool(armed)
    if armed:
        where = (
            "pinned denoiser"
            if hooked and pinned
            else "offloaded denoiser"
            if hooked
            else "split denoiser (each card's blocks on that card)"
            if split
            else "denoiser"
        )
        slots = sum(int(getattr(h, "slots_mib", 0) or 0) for h in armed)
        _set_reason(
            pipe,
            f"{where} recorded per block, keyed by input shape and weight placement"
            + (f"; streamed blocks read a {slots} MiB slot ring" if slots else ""),
        )
        pipe._unsloth_cuda_graph_mode = "blocks"
    else:
        _set_reason(
            pipe, "; ".join(r for r in reasons if r) or "no denoiser block could be recorded"
        )
    return tuple(armed)


def _attach_block_fallback(
    pipe: Any, step: list, target: Any, hooked: bool, pinned: bool, logger: Any
) -> None:
    try:
        from .diffusion_block_graph import (
            block_graphs_disabled,
            block_graphs_requested,
            install_block_graphs,
        )
    except Exception as exc:  # noqa: BLE001
        _warn(logger, "block graph import", exc)
        return
    if block_graphs_disabled():
        return
    for handle in step:
        model_offload = getattr(handle.placement, "mode", None) == "model"
        stays = (not hooked or bool(pinned)) and not model_offload
        if not (stays or block_graphs_requested()):
            continue

        def arm(module: Any = handle.module) -> Any:
            armed, _why = install_block_graphs(
                module, device = getattr(target, "torch_device", None), logger = logger
            )
            if armed is not None:
                _set_reason(
                    pipe,
                    str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
                    + "; the steps it leaves eager record per block",
                )
            return armed

        handle.fallback = arm


def _set_reason(pipe: Any, reason: str) -> None:
    try:
        pipe._unsloth_cuda_graph_reason = reason
    except Exception:  # noqa: BLE001
        pass


def status_reason(pipe: Any, on: bool) -> str:
    """The cuda_graph badge text: the block layer's own sentence when it armed, else why graphs are off."""
    if on:
        mode = getattr(pipe, "_unsloth_cuda_graph_mode", None)
        reason = str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "")
        if mode == "blocks":
            return reason or WHOLE_REASON
        head, _, tail = reason.partition("; ")
        if mode == "step" and head.startswith("captured") and "resident" not in head:
            # "denoiser step captured per input shape: block-streamed (copies recorded in the graph), replayed ..."
            return f"denoiser step {head}, replayed bit-identically" + (f"; {tail}" if tail else "")
        return WHOLE_REASON + (f"; {tail}" if mode == "step" and tail else "")
    return str(getattr(pipe, "_unsloth_cuda_graph_reason", None) or "speed tier does not capture")


def held_bytes(handles: Any) -> int:
    """Device bytes the armed graphs hold outside the caching allocator's reach (pools, static buffers, slot ring)."""
    total = 0
    for handle in handles or ():
        fn = getattr(handle, "held_bytes", None)
        if callable(fn):
            try:
                total += int(fn())
            except Exception:  # noqa: BLE001
                pass
    return total
