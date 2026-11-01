# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense denoisers (offloaded too) compile in background while renders run eager; a render mid-
compile waits for it; any failure ends the attempt and the next render compiles inline;
UNSLOTH_DIFFUSION_BG_COMPILE=0 disables"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import sys
import threading
import time
from _thread import get_ident as _thread_ident
from pathlib import Path
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE"
OFFLOADED_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_OFFLOADED"
YIELD_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_YIELD"
# Seconds of compile a render lets run before it pauses it; unset = this load's fastest eager render.
SLICE_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_SLICE_S"
_SLICE_MAX_S = 30.0
# A gap between two yield points longer than this compiled a block graph (a cached block call takes milliseconds).
_GRAPH_SEGMENT_S = 0.25
_MAX_SAMPLES = 4

# ContextVars, not thread-locals: diffusion_render_thread.run copies the caller's context onto the render thread.
_FORCE_EAGER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_force_eager", default = False
)
_NO_CAPTURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_no_capture", default = False
)
_NO_RECORD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_no_record", default = False
)
_AUXILIARY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_auxiliary_pass", default = False
)
# The BackgroundCompile whose thread this is (set only on that thread): yield_point's target.
_ACTIVE: contextvars.ContextVar[Optional["BackgroundCompile"]] = contextvars.ContextVar(
    "unsloth_diffusion_bg_active", default = None
)


class _Closed(BaseException):
    """Raised at a yield point once the compile is closed (unload): ends the paused replay."""


def enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def offloaded_enabled() -> bool:
    raw = (os.environ.get(OFFLOADED_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def yield_enabled() -> bool:
    raw = (os.environ.get(YIELD_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


WARM_POOL_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_WARM_POOL"
_POOL_WARMED = False


def warm_compile_pool(logger: Any = None) -> bool:
    """Start inductor's compile worker pool ahead of the deferred compile."""
    global _POOL_WARMED
    if _POOL_WARMED or not enabled():
        return False
    raw = (os.environ.get(WARM_POOL_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    _POOL_WARMED = True
    try:
        from . import diffusion_compile_config
        from torch._inductor.async_compile import AsyncCompile

        diffusion_compile_config.apply()
        AsyncCompile.warm_pool()
        AsyncCompile.wakeup()
    except Exception as exc:  # noqa: BLE001 - the compile warms its own pool as before
        if logger is not None:
            logger.debug("diffusion.bg_compile: compile pool warm-up skipped: %s", exc)
        return False
    if logger is not None:
        logger.info(
            "diffusion.bg_compile: inductor compile workers starting ahead of the deferred compile"
        )
    return True


PREWARM_ENV = "UNSLOTH_DIFFUSION_DEFERRED_PREWARM"
_PREWARM_LOCK = threading.Lock()
_PREWARM_THREAD: Optional[threading.Thread] = None


def prewarm_enabled() -> bool:
    raw = (os.environ.get(PREWARM_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _triton_key_fn() -> Optional[Callable[[], Any]]:
    for name in ("triton.runtime.cache", "triton.compiler.compiler"):
        try:
            module = __import__(name, fromlist = ["triton_key"])
        except Exception:  # noqa: BLE001 - another Triton layout
            continue
        fn = getattr(module, "triton_key", None)
        if callable(fn):
            return fn
    return None


def warm_triton_key(logger: Any = None) -> bool:
    """Fill Triton's per-process ``triton_key()`` cache (1-1.5 s, paid by the first JIT even on a warm disk cache)."""
    fn = _triton_key_fn()
    if fn is None:
        return False
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - the first JIT call computes it as before
        if logger is not None:
            logger.debug("diffusion.bg_compile: triton key warm-up skipped: %s", exc)
        return False
    if logger is not None:
        logger.info("diffusion.bg_compile: triton key ready in %.2f s", time.perf_counter() - t0)
    return True


def warm_engage_imports(logger: Any = None) -> int:
    """Model-free first-use work of the engagement; returns how many steps ran."""
    import importlib

    done = 0
    steps: list[Callable[[], Any]] = [
        # Import only: diffusion_patch_backend._helpers() keeps its own retry (import unsloth) for the 3rd image.
        lambda: importlib.import_module("unsloth_zoo.temporary_patches.utils"),
        lambda: importlib.import_module(f"{__package__}.diffusion_vae_fused")._kernels(),
    ]
    for step in steps:
        try:
            step()
            done += 1
        except Exception as exc:  # noqa: BLE001 - the 3rd image does this inline as before
            if logger is not None:
                logger.debug("diffusion.bg_compile: engage warm-up step skipped: %s", exc)
    return done


def start_deferred_prewarm(logger: Any = None) -> bool:
    """Run the warm-ups on a side thread so none of it lands inside a render; ``PREWARM_ENV=0`` warms the pool inline."""
    global _PREWARM_THREAD
    if not prewarm_enabled():
        return warm_compile_pool(logger)
    with _PREWARM_LOCK:
        if _PREWARM_THREAD is not None:
            return False

        def run() -> None:
            t0 = time.perf_counter()
            warm_compile_pool(logger)
            warm_triton_key(logger)
            n = warm_engage_imports(logger)
            if logger is not None:
                logger.info(
                    "diffusion.bg_compile: deferred profile warm-up done in %.2f s (%d import steps)",
                    time.perf_counter() - t0,
                    n,
                )

        _PREWARM_THREAD = threading.Thread(target = run, name = "unsloth-deferred-prewarm", daemon = True)
        _PREWARM_THREAD.start()
    return True


def wait_prewarm(timeout: Optional[float] = 60.0) -> bool:
    """Join the side warm-up before the profile engages (no second pool start, no second key hash)."""
    with _PREWARM_LOCK:
        thread = _PREWARM_THREAD
    if thread is None or thread is threading.current_thread():
        return True
    thread.join(timeout)
    return not thread.is_alive()


def slice_seconds(eager_render_s: Optional[float]) -> float:
    """Compile seconds a render lets run before it pauses the compile: ``SLICE_ENV`` when set, else the load's
    fastest eager render (0 when none was timed), clamped to [0, 30]."""
    raw = (os.environ.get(SLICE_ENV) or "").strip()
    value: Optional[float] = None
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = None
    if value is None:
        value = eager_render_s if isinstance(eager_render_s, (int, float)) else 0.0
    if value != value:  # NaN
        value = 0.0
    return max(0.0, min(_SLICE_MAX_S, float(value)))


IDLE_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_IDLE_ONLY"
IDLE_GRACE_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_IDLE_S"
# Resume delay after a render: covers sending the response and a back-to-back client's next request.
_IDLE_GRACE_S = 1.0
_IDLE_GRACE_MAX_S = 60.0
# The compile thread's innermost frame unchanged this long = blocked without the GIL: the render stops waiting.
_FREEZE_QUIET_S = 0.02
_FREEZE_POLL_S = 0.002


def _innermost(ident: Optional[int]) -> Optional[tuple]:
    frame = sys._current_frames().get(ident) if ident is not None else None
    if frame is None:
        return None
    return (id(frame.f_code), frame.f_lasti, id(frame))


def idle_only_enabled() -> bool:
    """Per-block compiles run only while no render runs (default). ``=0`` restores the slice a render let run."""
    raw = (os.environ.get(IDLE_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def idle_grace_seconds() -> float:
    raw = (os.environ.get(IDLE_GRACE_ENV) or "").strip()
    try:
        value = float(raw) if raw else _IDLE_GRACE_S
    except ValueError:
        value = _IDLE_GRACE_S
    if value != value:  # NaN
        value = _IDLE_GRACE_S
    return max(0.0, min(_IDLE_GRACE_MAX_S, value))


# Mid-trace freeze via sys.monitoring PY_START (3.12+): park only the compile thread, only inside torch's compile,
# never with a lock-taking module frame (or a Studio frame inside a lock block) on its stack.
_FREEZE_TOOL_IDS = (4, 3)
_FREEZE_TOOL: list = [None]  # None = not tried, False = unavailable, else the tool id
_FREEZE_TARGET: list = [None]
# Job frozen mid-trace now (_FREEZE_TARGET is cleared by then): what _CompileLockProxy thaws.
_FROZEN_JOB: list = [None]
_FREEZE_LOCK = threading.Lock()
_FREEZE_MAX_DEPTH = 5000
# Frames here may hold a lock a render (or the compile pool) takes: never park there.
_FREEZE_DENY_ROOTS = frozenset(
    {
        "logging",
        "importlib",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "zipimport",
        "threading",
        "_threading_local",
        "concurrent",
        "multiprocessing",
        "queue",
        "subprocess",
        "selectors",
        "asyncio",
        "socket",
        "ssl",
        "warnings",
        "tqdm",
        "structlog",
        "rich",
        "filelock",
        "weakref",
        "_weakrefset",
        "gc",
    }
)
# (torch._inductor.async_compile takes no lock: it only hands kernels to the pool, whose locks compile_worker holds.)
_FREEZE_DENY_MODULES = (
    "torch._logging",
    "torch._inductor.compile_worker",
    "torch._inductor.autotune_process",
    "torch.utils._filelock",
    "core.inference.diffusion_compile_config",
    __name__,
)
# Studio frames are refused only inside their own lock-holding code, decided once per code object from its source.
_STUDIO_ROOTS = ("core.", "studio.")
_LOCKISH = re.compile(r"lock|LOCK|_IDLE\b|_cv\b|hold_off_capture|held_work|capture_hold")
# code -> True (bare acquire: whole function) or (first, last) line spans of lock-holding ``with`` blocks
_LOCK_CODE: dict = {}


def _lock_spans(code: Any) -> Any:
    hit = _LOCK_CODE.get(code)
    if hit is not None:
        return hit
    try:
        import ast
        import inspect
        import textwrap

        lines, first = inspect.getsourcelines(code)
        tree = ast.parse(textwrap.dedent("".join(lines)))
        spans = []
        bare = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                if any(_LOCKISH.search(ast.unparse(item.context_expr)) for item in node.items):
                    spans.append(
                        (node.lineno + first - 1, (node.end_lineno or node.lineno) + first - 1)
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("acquire", "capture_hold")
            ):
                bare = True
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "capture_hold"
            ):
                bare = True
        hit = True if bare else tuple(spans)
    except Exception:  # noqa: BLE001 - no source: assume it may hold one
        hit = True
    _LOCK_CODE[code] = hit
    return hit


def _holds_lock(frame: Any) -> bool:
    spans = _lock_spans(frame.f_code)
    if spans is True:
        return True
    if not spans:
        return False
    line = frame.f_lineno or 0
    return any(a <= line <= b for a, b in spans)


# Park only inside the compile: outside it runs eager Studio / diffusers code whose locks no list can name.
_FREEZE_INSIDE = ("torch._dynamo.", "torch._inductor.", "torch._functorch.")


FREEZE_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_FREEZE"


def freeze_enabled() -> bool:
    """Idle-only and ``FREEZE_ENV`` not off and the interpreter can (``=0``: stop only at block boundaries)."""
    raw = (os.environ.get(FREEZE_ENV) or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return idle_only_enabled() and _freeze_tool() is not None


def _freeze_tool() -> Optional[int]:
    tool = _FREEZE_TOOL[0]
    if tool is not None:
        return tool or None
    with _FREEZE_LOCK:
        if _FREEZE_TOOL[0] is not None:
            return _FREEZE_TOOL[0] or None
        _FREEZE_TOOL[0] = False
        mon = getattr(sys, "monitoring", None)
        if mon is None:
            return None
        try:
            # dynamo would try to trace the callback when it fires between two compiled frames on the compile thread
            from torch._dynamo.eval_frame import skip_code
            for fn in (
                _on_py_start,
                _freeze_safe,
                _freeze_refusal,
                _frame_refusal,
                _holds_lock,
                _lock_spans,
                _compile_lock_owned,
                BackgroundCompile._maybe_freeze,
                BackgroundCompile._park,
            ):
                skip_code(fn.__code__)
        except Exception:  # noqa: BLE001 - no skip hook: park at block boundaries only
            return None
        for tool_id in _FREEZE_TOOL_IDS:
            try:
                mon.use_tool_id(tool_id, "unsloth-diffusion-bg-compile")
            except Exception:  # noqa: BLE001 - id held by a profiler / debugger
                continue
            try:
                mon.register_callback(tool_id, mon.events.PY_START, _on_py_start)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    mon.free_tool_id(tool_id)
                continue
            _FREEZE_TOOL[0] = tool_id
            return tool_id
        return None


def _set_freeze_events(on: bool) -> None:
    tool = _FREEZE_TOOL[0]
    if not tool:
        return
    mon = sys.monitoring
    with contextlib.suppress(Exception):
        mon.set_events(tool, mon.events.PY_START if on else 0)


def _on_py_start(code: Any, offset: int) -> Any:
    bg = _FREEZE_TARGET[0]
    if bg is None or _thread_ident() != bg._tid:
        return None
    bg._maybe_freeze()
    return None


def _frame_refusal(frame: Any) -> Optional[str]:
    """Why ``frame`` alone forbids a park (a module or function that may hold a lock a render takes), else None."""
    name = frame.f_globals.get("__name__") or ""
    if name.partition(".")[0] in _FREEZE_DENY_ROOTS:
        return name
    for deny in _FREEZE_DENY_MODULES:
        if name == deny or name.startswith(deny + "."):
            return name
    if name.startswith(_STUDIO_ROOTS) and _holds_lock(frame):
        return f"{name}:{frame.f_code.co_name}"
    return None


def _freeze_refusal(frame: Any, memo: Optional[dict] = None) -> Optional[str]:
    """None when the compile thread may park at ``frame``, else why not. ``memo`` caches per-frame verdicts."""
    chain = []
    depth = 0
    run_code = BackgroundCompile._run.__code__
    refusal: Optional[str] = None
    inside = False
    f = frame
    while True:
        if f is None:
            refusal = "not the compile thread"
            break
        if depth >= _FREEZE_MAX_DEPTH:
            refusal = "stack depth"
            break
        if memo is not None:
            hit = memo.get(id(f))
            if hit is not None and hit[0] is f:
                refusal, inside = hit[1], hit[2]
                break
        if f.f_code is run_code:
            break
        chain.append(f)
        f = f.f_back
        depth += 1
    for g in reversed(chain):
        if refusal is None:
            refusal = _frame_refusal(g)
        if not inside:
            inside = (g.f_globals.get("__name__") or "").startswith(_FREEZE_INSIDE)
        if memo is not None:
            memo[id(g)] = (g, refusal, inside)
    if refusal is None and not inside:
        return "outside the compile"
    return refusal


def _stack_summary(frame: Any, limit: int) -> str:
    where = []
    while frame is not None and len(where) < limit:
        where.append(f"{frame.f_globals.get('__name__', '?')}:{frame.f_code.co_name}")
        frame = frame.f_back
    return " < ".join(where)


def _freeze_safe(frame: Any) -> bool:
    return _freeze_refusal(frame) is None


class _CompileLockProxy:
    """dynamo's ``compile_lock``, unchanged, except that a thread blocked on it while the background compile is frozen
    mid-trace (holding it) thaws that compile, so the wait ends when its block graph does instead of never."""

    _unsloth_bg_proxy = True

    def __init__(self, lock: Any) -> None:
        self._lock = lock

    def acquire(
        self,
        blocking: bool = True,
        timeout: float = -1,
    ) -> bool:
        if self._lock.acquire(False):
            return True
        if not blocking:
            return False
        deadline = None if timeout is None or timeout < 0 else time.perf_counter() + timeout
        with _WAITERS_LOCK:
            _LOCK_WAITERS[0] += 1
        try:
            while True:
                for bg in {
                    id(j): j for j in (_FROZEN_JOB[0], _FREEZE_TARGET[0]) if j is not None
                }.values():
                    bg._thaw_for_lock()
                if self._lock.acquire(True, 0.05):
                    return True
                if deadline is not None and time.perf_counter() >= deadline:
                    return False
        finally:
            with _WAITERS_LOCK:
                _LOCK_WAITERS[0] -= 1

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self._lock.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lock, name)


# Threads blocked on dynamo's compile_lock: the compile thread never freezes holding it while one waits.
_LOCK_WAITERS: list = [0]
_WAITERS_LOCK = threading.Lock()
_COMPILE_LOCK: list = [None]


def _compile_lock_owned() -> bool:
    lock = _COMPILE_LOCK[0]
    if lock is None:
        return False
    try:
        return bool(lock._is_owned())
    except Exception:  # noqa: BLE001 - assume held: never freeze on it while a thread waits
        return True


def _install_compile_lock_proxy() -> bool:
    try:
        from torch._dynamo import convert_frame
    except Exception:  # noqa: BLE001
        return False
    lock = getattr(convert_frame, "compile_lock", None)
    if lock is None:
        return False
    if getattr(lock, "_unsloth_bg_proxy", False):
        _COMPILE_LOCK[0] = lock._lock
        return True
    try:
        _COMPILE_LOCK[0] = lock
        convert_frame.compile_lock = _CompileLockProxy(lock)
    except Exception:  # noqa: BLE001
        return False
    return True


def yield_point() -> None:
    """Called before a guarded compiled block runs. On a background compile thread with a pause requested, parks
    here (between two block graphs, never inside a trace) until the render resumes it. Elsewhere a no-op."""
    bg = _ACTIVE.get()
    if bg is not None:
        bg._yield_point()


FLOAT_SEEDS_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_FLOAT_SEEDS"
FLOAT_SEEDS_FILE = "bg_compile_float_specializations.json"
_FLOAT_SEEDS_VERSION = 1
_SYMFLOAT_NAME = re.compile(r"^zf\d+$")


def float_seeds_enabled() -> bool:
    raw = (os.environ.get(FLOAT_SEEDS_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _tensorify_state() -> Any:
    """dynamo's set of SymFloat names to specialise on a (re)trace, or None on a torch without it."""
    try:
        from torch._dynamo.symbolic_convert import TensorifyState
    except Exception:  # noqa: BLE001 - older / newer torch: nothing to seed
        return None
    names = getattr(TensorifyState, "force_specializations", None)
    if not isinstance(names, set) or not callable(getattr(TensorifyState, "specialize", None)):
        return None
    return TensorifyState


def _code_digest(code: Any) -> str:
    """Bytecode plus constants; a nested code object (lambda, genexpr) by its own digest, never its address."""
    h = hashlib.sha256(code.co_code)
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            h.update(_code_digest(const).encode("utf-8"))
        else:
            h.update(repr(const).encode("utf-8"))
    return h.hexdigest()


def _code_signature(module: Any) -> Optional[str]:
    """The traced code a seed list belongs to: torch build plus every repeated block class's forward bytecode."""
    try:
        import torch

        from .diffusion_block_graph import repeated_blocks

        parts = [str(torch.__version__), type(module).__qualname__]
        for cls in sorted({type(b) for b in repeated_blocks(module)}, key = lambda c: c.__qualname__):
            code = getattr(getattr(cls, "forward", None), "__code__", None)
            if code is None:
                return None
            name = getattr(code, "co_qualname", code.co_name)
            parts.append(f"{cls.__module__}.{cls.__qualname__}:{name}:{_code_digest(code)}")
        if len(parts) < 3:
            return None
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 - no signature, no seeds
        return None


def _modules_signature(modules: Any) -> Optional[str]:
    modules = list(modules) if isinstance(modules, (list, tuple)) else [modules]
    sigs = [_code_signature(m) for m in modules]
    if not sigs or any(s is None for s in sigs):
        return None
    return sigs[0] if len(sigs) == 1 else hashlib.sha256("|".join(sigs).encode("utf-8")).hexdigest()


def _read_float_seeds(path: Path, signature: str) -> set:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - missing / torn file: learn again
        return set()
    if not isinstance(data, dict) or data.get("version") != _FLOAT_SEEDS_VERSION:
        return set()
    if data.get("signature") != signature:
        return set()
    names = data.get("names")
    if not isinstance(names, list):
        return set()
    return {n for n in names if isinstance(n, str) and _SYMFLOAT_NAME.match(n)}


def _write_float_seeds(path: Path, signature: str, names: set) -> bool:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp.write_text(
            json.dumps(
                {"version": _FLOAT_SEEDS_VERSION, "signature": signature, "names": sorted(names)},
                indent = 1,
            ),
            encoding = "utf-8",
        )
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001 - bookkeeping for the next load only
        with contextlib.suppress(Exception):
            tmp.unlink()
        return False


def _seed_floats(modules: Any, seed_dir: Optional[Path]) -> tuple[Optional[tuple], int]:
    """Pre-seed the SymFloat names the last compile of this key learned; returns (keep state, names added)."""
    if seed_dir is None or not float_seeds_enabled():
        return None, 0
    try:
        state = _tensorify_state()
        signature = _modules_signature(modules)
        if state is None or signature is None:
            return None, 0
        path = Path(seed_dir) / FLOAT_SEEDS_FILE
        before = set(state.force_specializations)
        seeds = _read_float_seeds(path, signature)
        for name in sorted(seeds - before):
            state.specialize(name)
        return (state, path, signature, before, seeds), len(seeds - before)
    except Exception:  # noqa: BLE001 - no seeds: the compile restarts as before
        return None, 0


def _keep_floats(seeded: Optional[tuple]) -> int:
    """Save this compile's specialised names if changed; returns how many are new."""
    if seeded is None:
        return 0
    state, path, signature, before, seeds = seeded
    try:
        names = {
            n for n in set(state.force_specializations) - before if _SYMFLOAT_NAME.match(str(n))
        }
        names |= seeds & set(state.force_specializations)
        if names and names != seeds:
            _write_float_seeds(path, signature, names)
        return len(names - seeds)
    except Exception:  # noqa: BLE001 - bookkeeping only
        return 0


class InlineFloatSeeds:
    """Float seeds for a compile inside a render (video's first clip): ``seed`` at denoise entry, ``keep`` at decode
    entry, so only the denoiser's names are saved. Never raises."""

    def __init__(
        self,
        modules: Any,
        seed_dir: Any,
        *,
        logger: Any = None,
    ) -> None:
        self.modules = list(modules) if isinstance(modules, (list, tuple)) else [modules]
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.logger = logger
        self.float_seeds = 0
        self.float_seeds_learned = 0
        self.seeded = False
        self.kept = False
        self._seeds: Optional[tuple] = None

    def seed(self) -> int:
        if self.seeded:
            return 0
        self.seeded = True
        self._seeds, self.float_seeds = _seed_floats(self.modules, self.seed_dir)
        if self.float_seeds and self.logger is not None:
            with contextlib.suppress(Exception):
                self.logger.info(
                    "diffusion.float_seeds: %d float specialisation(s) pre-seeded for the %s compile",
                    self.float_seeds,
                    "/".join(type(m).__name__ for m in self.modules),
                )
        return self.float_seeds

    def keep(self) -> None:
        if not self.seeded or self.kept:
            return
        self.kept = True
        self.float_seeds_learned = _keep_floats(self._seeds)
        self._seeds = None
        self.modules = []

    def done(self) -> bool:
        return self.kept

    def describe(self) -> dict:
        return {"float_seeds": self.float_seeds, "float_seeds_learned": self.float_seeds_learned}


def load_time_enabled() -> bool:
    """Opt-in (``=1``): render 1 eager and render 2 compiled would break "same seed twice repeats"."""
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw in ("1", "true", "yes", "on", "all")


def eager_forced() -> bool:
    """True inside a generation that must not touch the compiled callable (its compile is still in flight)."""
    return _FORCE_EAGER.get()


def capture_suppressed() -> bool:
    """True on the background compile thread: run compiled, never record a CUDA graph."""
    return _NO_CAPTURE.get()


@contextlib.contextmanager
def no_record():
    token = _NO_RECORD.set(True)
    try:
        yield
    finally:
        _NO_RECORD.reset(token)


def auxiliary() -> bool:
    return _AUXILIARY.get()


@contextlib.contextmanager
def auxiliary_pass():
    """Detailer crop inpaints: eager, uncaptured, unrecorded, uncounted. Each box is its own size, so compiling
    them recompiled per detection and used up the per-module graph cap the main render needs."""
    tokens = (_AUXILIARY.set(True), _NO_RECORD.set(True), _FORCE_EAGER.set(True))
    try:
        yield
    finally:
        _FORCE_EAGER.reset(tokens[2])
        _NO_RECORD.reset(tokens[1])
        _AUXILIARY.reset(tokens[0])


def gate_compiled_eager(modules: Any) -> Callable[[], None]:
    """Route ``Module.compile``d modules to eager ``_call_impl`` under ``force_eager``; returns the restore.
    Without it force_eager only reaches regional blocks and graph wrappers, which still call the compiled U-Net."""
    installed: list = []
    for module in modules or ():
        inner = getattr(module, "_compiled_call_impl", None)
        if inner is None:
            continue
        eager = getattr(module, "_call_impl", None)
        if eager is None:
            continue

        def gate(
            *args: Any,
            _inner: Any = inner,
            _eager: Any = eager,
            **kwargs: Any,
        ) -> Any:
            if _FORCE_EAGER.get():
                return _eager(*args, **kwargs)
            return _inner(*args, **kwargs)

        try:
            module._compiled_call_impl = gate
        except Exception:  # noqa: BLE001 - not a settable module
            continue
        installed.append((module, gate, inner))

    def restore() -> None:
        for module, gate, inner in installed:
            try:
                if getattr(module, "_compiled_call_impl", None) is gate:
                    module._compiled_call_impl = inner
            except Exception:  # noqa: BLE001
                pass
        installed.clear()

    return restore


@contextlib.contextmanager
def force_eager(on: bool = True):
    token = _FORCE_EAGER.set(bool(on))
    try:
        yield
    finally:
        _FORCE_EAGER.reset(token)


def recording(module: Any) -> bool:
    """True when a call into ``module`` now records inputs for its pending background compile; callers that reshape
    compiled calls (Z-Image CFG split) must reshape this one too, or the compile builds a graph nothing runs."""
    if not eager_forced() or _NO_RECORD.get():
        return False
    for hook in tuple((getattr(module, "_forward_pre_hooks", None) or {}).values()):
        owner = getattr(hook, "__self__", None)
        if (
            isinstance(owner, BackgroundCompile)
            and owner.module is module
            and owner.wants_samples()
        ):
            return True
    return False


def _cuda_graph_helpers() -> tuple[Callable, Callable, Callable]:
    from .diffusion_cuda_graph import graph_key
    return _flatten_sample, _rebuild_sample, graph_key


# Plain state objects in a call (Qwen-Image-2.1's ``QwenImage21KVCache``) are recorded as fresh instances of cloned tensors.
OBJECTS_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_OBJECTS"
_OBJECT_DEPTH = 3
# A prefix holding a condition image is ~2.2 GiB: such a call keeps compiling on the render.
_OBJECT_MAX_BYTES = 1 << 30


def _objects_enabled() -> bool:
    raw = (os.environ.get(OBJECTS_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _plain_state(obj: Any) -> Optional[dict]:
    import torch

    if isinstance(obj, (torch.nn.Module, torch.Tensor, type)) or callable(obj):
        return None
    cls = type(obj)
    if str(cls.__module__).split(".")[0] in ("builtins", "torch") or hasattr(cls, "__slots__"):
        return None
    state = getattr(obj, "__dict__", None)
    return state if isinstance(state, dict) else None


def _flatten_sample(
    obj: Any,
    out: list,
    depth: int = 0,
    budget: Optional[list] = None,
) -> tuple:
    import torch

    if budget is None:
        budget = [0]
    if torch.is_tensor(obj):
        out.append(obj)
        return ("t", len(out) - 1)
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), [_flatten_sample(o, out, depth, budget) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, _flatten_sample(v, out, depth, budget)) for k, v in obj.items()])
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    state = _plain_state(obj) if _objects_enabled() and depth < _OBJECT_DEPTH else None
    if state is None:
        raise TypeError(f"cannot make a static buffer for a {type(obj).__name__} in the call tree")
    first = len(out)
    spec = (
        "o",
        type(obj),
        [(k, _flatten_sample(v, out, depth + 1, budget)) for k, v in state.items()],
    )
    held = budget[1] if len(budget) > 1 else None
    if held is not None:
        held.update(range(first, len(out)))
    if depth == 0:
        # The outermost object's slice holds every nested one's: charging each level would count a tensor twice.
        budget[0] += sum(t.numel() * t.element_size() for t in out[first:])
    if budget[0] > _OBJECT_MAX_BYTES:
        raise TypeError(
            f"{type(obj).__name__} holds {budget[0] >> 20} MiB, above the {_OBJECT_MAX_BYTES >> 20} MiB a sample copies"
        )
    return spec


def _rebuild_sample(spec: tuple, tensors: list) -> Any:
    kind = spec[0]
    if kind == "t":
        return tensors[spec[1]]
    if kind == "l":
        values = [_rebuild_sample(s, tensors) for s in spec[2]]
        return tuple(values) if spec[1] else values
    if kind == "d":
        return {k: _rebuild_sample(s, tensors) for k, s in spec[1]}
    if kind == "o":
        obj = object.__new__(spec[1])
        for k, s in spec[2]:
            obj.__dict__[k] = _rebuild_sample(s, tensors)
        return obj
    return spec[1]


def _capture_armed(module: Any) -> bool:
    """Whether a live CUDA-graph wrapper sits on ``module`` (forward slot for a DiT, compiled slot for a U-Net)."""
    try:
        from .diffusion_cuda_graph import GraphedForward, _outer_layer
    except Exception:  # noqa: BLE001
        return False
    candidates = [getattr(module, "_compiled_call_impl", None), module.__dict__.get("forward")]
    gate = candidates[0]
    pair = getattr(gate, "_unsloth_bg_inner", None)
    if pair is not None:
        candidates.append(pair)
    try:
        outer = _outer_layer(module)
        if outer is not None:
            candidates.append(getattr(outer, "inner", None))
    except Exception:  # noqa: BLE001
        pass
    return any(
        isinstance(c, GraphedForward) and c.enabled and not c.poisoned and not c.bypassed
        for c in candidates
    )


class BackgroundCompile:
    """One armed denoiser: records eager inputs, then compiles from them on a daemon thread."""

    def __init__(
        self,
        module: Any,
        *,
        logger: Any = None,
        max_samples: int = _MAX_SAMPLES,
        seed_dir: Any = None,
    ) -> None:
        self.module = module
        self.logger = logger
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.float_seeds = 0
        self.float_seeds_learned = 0
        self.max_samples = int(max_samples)
        self.samples: list[tuple] = []
        self._keys: set = set()
        self._handle: Any = None
        self._gate: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # recording -> compiling -> done | failed
        self.state = "recording"
        self.error: Optional[str] = None
        self.compile_s: Optional[float] = None
        self.eager_generations = 0
        self._cv = threading.Condition(threading.Lock())
        self._pause_depth = 0
        self._parked = False
        # Bumped when the last pause is released: a parked thread leaves on it even if a new pause is already held.
        self._release_epoch = 0
        self.pauses = 0
        self.paused_s = 0.0
        self.yield_points = 0
        # _owe_graph: one graph per resume, so back-to-back renders cannot starve the compile.
        self.graph_s: Optional[float] = None
        self._last_yield: Optional[float] = None
        self._owe_graph = False
        # Idle-only: a pause stops the compile at once; it resumes idle_grace_seconds() after the last render.
        self.idle = idle_only_enabled()
        self._idle_grace_s = idle_grace_seconds()
        self._release_at = 0.0
        self._tid: Optional[int] = None
        self._freeze_ok = False
        self._frozen = False
        self._thaw = False
        self.freezes = 0
        self.pause_wait_s = 0.0
        self.thaws = 0
        self._thaw_where: Optional[str] = None
        self._refusals: dict = {}
        # frame verdicts for _freeze_refusal; holds frames, so cleared on every resume
        self._frame_memo: dict = {}
        self._freezes_at_pause = 0
        self._pause_t0 = 0.0
        self._stopped_after: Optional[float] = None
        self._stopped_how = ""
        # A frozen compile holds dynamo's compile_lock: freeze only once the proxy that thaws it is in place.
        self._lock_proxy = False

    def yields(self) -> bool:
        return bool(getattr(self.module, "_unsloth_bg_yield", False)) and yield_enabled()

    def pause(
        self,
        cancel: Any = None,
        budget_s: float = 0.0,
        poll_s: float = 0.05,
        freeze: bool = False,
    ) -> tuple[float, bool]:
        """Let the compile run ``budget_s`` more seconds, then park it at its next yield point. Returns
        ``(seconds waited, paused)``; ``paused`` means a pause is held and the caller MUST ``resume()``. A compile
        that ends inside the budget returns ``paused=False`` (the render then runs compiled). Idle-only ignores the
        budget; ``freeze`` (resident denoiser) stops it at its next Python call inside the trace."""
        t0 = time.perf_counter()
        thread = self._thread
        if thread is None or thread is threading.current_thread() or not thread.is_alive():
            return 0.0, False
        if self.idle:
            return self._pause_now(thread, cancel, freeze, t0), True
        # Ask one graph_s early: the pause lands after the block graph in flight.
        budget = max(0.0, float(budget_s or 0.0))
        lead = self.graph_s if self.graph_s is not None else budget
        deadline = t0 + (0.0 if self._pause_depth else max(0.0, budget - lead))
        while thread.is_alive() and not (cancel is not None and cancel.is_set()):
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            thread.join(min(poll_s * 5, left))
        if not thread.is_alive():
            return time.perf_counter() - t0, False
        with self._cv:
            self._pause_depth += 1
            self.pauses += 1
            self._cv.notify_all()
            while thread.is_alive() and not self._parked:
                if cancel is not None and cancel.is_set():
                    break
                self._cv.wait(poll_s)
        return time.perf_counter() - t0, True

    def _pause_now(self, thread: Any, cancel: Any, freeze: bool, t0: float) -> float:
        with self._cv:
            self._pause_depth += 1
            self.pauses += 1
            nested = self._pause_depth > 1
            if not nested:
                self._freeze_ok = (
                    bool(freeze) and self._lock_proxy and self._tid is not None and freeze_enabled()
                )
                self._freezes_at_pause = self.freezes
                self._refusals = {}
                self._pause_t0 = t0
                self._stopped_after = None
                if self._freeze_ok and not self._parked:
                    _FREEZE_TARGET[0] = self
                    _set_freeze_events(True)
            self._cv.notify_all()
            if not nested:
                # Frozen and blocked without the GIL: costs the render nothing, the armed freeze stops it on wake.
                # Unfrozen (offloaded): it would move the render's weights, so wait for the block-boundary park.
                last = _innermost(thread.ident)
                quiet_since = time.perf_counter()
                while thread.is_alive() and not self._parked:
                    if cancel is not None and cancel.is_set():
                        break
                    self._cv.wait(_FREEZE_POLL_S)
                    now = time.perf_counter()
                    here = _innermost(thread.ident)
                    if here != last:
                        last, quiet_since = here, now
                    elif self._freeze_ok and now - quiet_since >= _FREEZE_QUIET_S:
                        break
        waited = time.perf_counter() - t0
        self.pause_wait_s += waited
        return waited

    def resume(self) -> None:
        with self._cv:
            self._pause_depth = max(0, self._pause_depth - 1)
            if not self._pause_depth:
                self._release_epoch += 1
                if self.idle:
                    self._frame_memo = {}
                    self._release_at = time.perf_counter() + self._idle_grace_s
                    if (
                        self._freeze_ok
                        and self.freezes == self._freezes_at_pause
                        and not self._parked
                    ):
                        self._log_unfrozen()
                    elif self.logger is not None and (self._stopped_after or 0.0) > 0.1:
                        self.logger.info(
                            "diffusion.bg_compile: the compile stopped %.2f s into the render (%s); refusals %s",
                            self._stopped_after,
                            self._stopped_how,
                            sorted(self._refusals.items(), key = lambda kv: -kv[1])[:6],
                        )
                    if self._thaw_where is not None and self.logger is not None:
                        self.logger.info(
                            "diffusion.bg_compile: a render needed dynamo's compile lock while the compile held it "
                            "(%d thaw(s)); first from %s",
                            self.thaws,
                            self._thaw_where,
                        )
                        self._thaw_where = None
                    self._thaw = False
                    if not self._frozen:
                        # never stopped: leave no event on for later renders
                        self._disarm_freeze()
                else:
                    # the next pause() waits for a fresh park, after the block graph this release owes
                    self._parked = False
            self._cv.notify_all()

    def _log_unfrozen(self) -> None:
        if self.logger is None:
            return
        try:
            frame = sys._current_frames().get(self._tid)
            where = []
            while frame is not None and len(where) < 12:
                where.append(f"{frame.f_globals.get('__name__', '?')}:{frame.f_code.co_name}")
                frame = frame.f_back
            top = sorted(self._refusals.items(), key = lambda kv: -kv[1])[:6]
            others = []
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, f in sys._current_frames().items():
                if ident in (self._tid, _thread_ident()):
                    continue
                chain = []
                while f is not None and len(chain) < 40:
                    chain.append(f"{f.f_globals.get('__name__', '?')}:{f.f_code.co_name}")
                    f = f.f_back
                if any(c.startswith(_FREEZE_INSIDE) for c in chain):
                    others.append(f"[{names.get(ident, ident)}] " + " < ".join(chain[:8]))
            self.logger.info(
                "diffusion.bg_compile: the compile did not stop during this render (thawed %d, refusals %s); at %s; "
                "other compile threads: %s",
                self.thaws,
                top,
                " < ".join(where),
                " || ".join(others) or "none",
            )
        except Exception:  # noqa: BLE001 - diagnostic only
            pass

    def _disarm_freeze(self) -> None:
        if _FREEZE_TARGET[0] is self:
            _set_freeze_events(False)
            _FREEZE_TARGET[0] = None

    def _thaw_for_lock(self) -> None:
        """Another thread waits on dynamo's compile_lock: a compile frozen holding it runs on to the end of its frame
        (it releases the lock there) and freezes again at its next safe call."""
        if _thread_ident() == self._tid:
            return
        with self._cv:
            if self._frozen and not self._thaw:
                self.thaws += 1
                if self._thaw_where is None:
                    self._thaw_where = _stack_summary(sys._getframe(2), 14)
                self._thaw = True
                if self._pause_depth and self._freeze_ok:
                    _FREEZE_TARGET[0] = self
                    _set_freeze_events(True)
                self._cv.notify_all()

    def _maybe_freeze(self) -> None:
        if not self._pause_depth or self._parked or self._closed.is_set() or not self._freeze_ok:
            return
        if _LOCK_WAITERS[0] and _compile_lock_owned():
            return  # a thread waits on compile_lock: finish the frame that holds it first
        self._thaw = False
        frame = sys._getframe(2)  # _maybe_freeze <- _on_py_start <- the function that just started
        memo = self._frame_memo
        if len(memo) > 200_000:
            memo.clear()
        why = _freeze_refusal(frame, memo)
        if why is not None:
            self._refusals[why] = self._refusals.get(why, 0) + 1
            return
        self._park(frozen = True)

    def _park(self, frozen: bool) -> None:
        """Idle-only park until unpaused and the grace elapsed, close, or (frozen) a thaw."""
        now = time.perf_counter()
        with self._cv:
            if not self._pause_depth and now >= self._release_at:
                return
            if frozen:
                self._frozen = True
                self.freezes += 1
                _FROZEN_JOB[0] = self
                self._disarm_freeze()
            self._parked = True
            if self._pause_depth and self._stopped_after is None:
                self._stopped_after = now - self._pause_t0
                self._stopped_how = "frozen" if frozen else "block boundary"
            self._cv.notify_all()
            try:
                while not self._closed.is_set() and not (frozen and self._thaw):
                    if self._pause_depth:
                        self._cv.wait(0.5)
                        continue
                    left = self._release_at - time.perf_counter()
                    if left <= 0:
                        break
                    self._cv.wait(left)
            finally:
                self._parked = False
                self._frozen = False
                if _FROZEN_JOB[0] is self:
                    _FROZEN_JOB[0] = None
                if not frozen:
                    self._thaw = False
                end = time.perf_counter()
                self.paused_s += end - now
                self._last_yield = end

    def _yield_point(self) -> None:
        self.yield_points += 1
        if self._closed.is_set():
            raise _Closed()
        now = time.perf_counter()
        if self._last_yield is not None and now - self._last_yield > _GRAPH_SEGMENT_S:
            seg = now - self._last_yield
            self.graph_s = seg if self.graph_s is None else 0.5 * (self.graph_s + seg)
            self._owe_graph = False
        self._last_yield = now
        if self.idle:
            if self._pause_depth or time.perf_counter() < self._release_at:
                self._park(frozen = False)
                if self._closed.is_set():
                    raise _Closed()
            return
        if not self._pause_depth or self._owe_graph:
            return
        with self._cv:
            if not self._pause_depth:
                return
            self._parked = True
            self._cv.notify_all()
            epoch = self._release_epoch
            try:
                while self._release_epoch == epoch and not self._closed.is_set():
                    self._cv.wait(0.5)
            finally:
                self._parked = False
                end = time.perf_counter()
                self.paused_s += end - now
                self._last_yield = end
                self._owe_graph = True
        if self._closed.is_set():
            raise _Closed()

    def install(self) -> bool:
        try:
            self._install_gate()
            self._handle = self.module.register_forward_pre_hook(self._record, with_kwargs = True)
            return True
        except Exception as exc:  # noqa: BLE001 - no hook, no background compile
            self._finish("failed", f"pre-hook: {type(exc).__name__}: {exc}")
            self._remove_gate()
            return False

    def _install_gate(self) -> None:
        """``Module.compile`` serves every call from ``_compiled_call_impl``: route it eager under ``force_eager``."""
        inner = getattr(self.module, "_compiled_call_impl", None)
        if inner is None:
            return
        eager = self.module._call_impl

        def gate(*args: Any, **kwargs: Any) -> Any:
            if _FORCE_EAGER.get():
                return eager(*args, **kwargs)
            return inner(*args, **kwargs)

        gate._unsloth_bg_gate = True  # type: ignore[attr-defined]
        gate._unsloth_bg_inner = inner  # type: ignore[attr-defined]
        self.module._compiled_call_impl = gate
        self._gate = (gate, inner)

    def _remove_gate(self) -> None:
        pair, self._gate = self._gate, None
        if pair is None or self.module is None:
            return
        gate, inner = pair
        try:
            if getattr(self.module, "_compiled_call_impl", None) is gate:
                self.module._compiled_call_impl = inner
        except Exception:  # noqa: BLE001
            pass

    def _record(self, module: Any, args: tuple, kwargs: dict) -> None:
        if (
            not eager_forced()
            or _NO_RECORD.get()
            or len(self.samples) >= self.max_samples
            or self.state != "recording"
        ):
            return None
        try:
            import torch

            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            key = graph_key((args, kwargs))
            if key in self._keys:
                return None
            live: list = []
            held: set = set()
            spec = _flatten((args, kwargs), live, 0, [0, held])
            if _capture_armed(module):
                # Like the capture's static buffers (made outside inference mode): dynamo guards on inference-ness.
                # Tensors inside a state object are not graph inputs: they keep the inference-ness they arrived with.
                clones = []
                for index, t in enumerate(live):
                    with torch.inference_mode(index in held and t.is_inference()):
                        clones.append(torch.empty_like(t))
                for dst, src in zip(clones, live):
                    dst.copy_(src)
            else:
                clones = [t.detach().clone() for t in live]
            self._keys.add(key)
            self.samples.append(
                (
                    spec,
                    clones,
                    bool(torch.is_inference_mode_enabled()),
                    bool(torch.is_grad_enabled()),
                    torch.cuda.current_device() if torch.cuda.is_available() else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - an uncloneable tree just means no background compile
            self.state = "failed"
            self.error = f"record: {type(exc).__name__}: {exc}"
            # Not _finish: the hook is running, and removing it mid-dispatch would mutate the dict torch iterates.
            self.samples = []
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: %s cannot be recorded (%s); the next render compiles inline",
                    type(module).__name__,
                    self.error,
                )
        return None

    def wants_samples(self) -> bool:
        return (
            self.state == "recording"
            and len(self.samples) < self.max_samples
            and self._handle is not None
        )

    def _remove_hook(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass

    def pending(self) -> bool:
        return self.state in ("recording", "compiling")

    def note_eager_generation(self) -> None:
        self.eager_generations += 1

    def compiling(self) -> bool:
        thread = self._thread
        return self.state == "compiling" and thread is not None and thread.is_alive()

    def wait(
        self,
        cancel: Any = None,
        poll_s: float = 0.25,
    ) -> float:
        """Block until an in-flight background compile ends; returns the seconds waited. ``cancel`` aborts the wait."""
        t0 = time.perf_counter()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return 0.0
        while thread.is_alive():
            if cancel is not None and cancel.is_set():
                break
            thread.join(poll_s)
        return time.perf_counter() - t0

    def kick(self) -> bool:
        """Start the background compile once samples exist. Idempotent; returns whether a thread is running."""
        with self._lock:
            if self.state == "compiling":
                return True
            if self.state != "recording":
                self._remove_hook()
                return False
            if not self.samples:
                # The generation never reached the denoiser (cancelled / failed early): keep recording.
                return False
            # Before the thread starts: the compiled trace must not contain the hook.
            self._remove_hook()
            self.state = "compiling"
            if self.idle:
                # kicked from the render's finally: start once its response is out (the first block boundary waits)
                self._release_at = time.perf_counter() + self._idle_grace_s
                self._lock_proxy = (
                    self.yields() and freeze_enabled() and _install_compile_lock_proxy()
                )
            self._thread = threading.Thread(
                target = self._run, name = "unsloth-diffusion-bg-compile", daemon = True
            )
            self._thread.start()
            return True

    def abandon(self, reason: str) -> bool:
        """Give up while still recording, so the next render compiles inline. False once a compile started or ended."""
        with self._lock:
            if self.state != "recording":
                return False
            self._finish("failed", reason)
            return True

    def _finish(
        self,
        state: str,
        error: Optional[str] = None,
    ) -> None:
        self.state = state
        self.error = error
        self.samples = []
        self._remove_hook()

    def _run(self) -> None:
        t0 = time.perf_counter()
        token = _NO_CAPTURE.set(True)
        _ACTIVE.set(self if self.yields() else None)
        self._tid = threading.get_ident()
        try:
            import torch

            from . import diffusion_compile_config

            # torch 2.12+ keeps compile config per context: a fresh thread would compile without the recorded knobs.
            diffusion_compile_config.apply()
            seeds = self._seed_floats()
            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            for spec, clones, inference, grad, device in list(self.samples):
                if self._closed.is_set():
                    self._finish("failed", "closed before the compile finished")
                    return
                if device is not None:
                    torch.cuda.set_device(device)
                args, kwargs = _rebuild(spec, clones)
                mode = torch.inference_mode() if inference else torch.set_grad_enabled(grad)
                with mode:
                    self.module(*args, **kwargs)
                if torch.cuda.is_available():
                    from .diffusion_cuda_graph import synchronize_outside_capture
                    synchronize_outside_capture()
            self.compile_s = time.perf_counter() - t0
            self._keep_floats(seeds)
            self._finish("done")
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: %s compiled in the background in %.1f s%s%s; the next render runs compiled",
                    type(self.module).__name__,
                    self.compile_s,
                    f" ({self.float_seeds} float specialisation(s) pre-seeded)"
                    if self.float_seeds
                    else "",
                    f" (paused {self.pauses}x for renders, {self.paused_s:.1f} s, compiling "
                    f"{self.compile_s - self.paused_s:.1f} s)"
                    if self.pauses
                    else "",
                )
        except _Closed:
            self.compile_s = time.perf_counter() - t0
            self._finish("failed", "closed before the compile finished")
        except BaseException as exc:  # noqa: BLE001 - never kills the process; the next render compiles inline
            self.compile_s = time.perf_counter() - t0
            self._finish(
                "failed",
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}",
            )
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.bg_compile: background compile of %s failed after %.1f s (%s); the next render "
                    "compiles inline",
                    type(self.module).__name__,
                    self.compile_s,
                    self.error,
                )
            exc.__traceback__ = None
        finally:
            _NO_CAPTURE.reset(token)
            _ACTIVE.set(None)
            with self._cv:
                self._disarm_freeze()
                self._cv.notify_all()

    def _seed_floats(self) -> Optional[tuple]:
        """Pre-seed the SymFloat specialisations this key's last compile learned; returns what ``_keep_floats`` needs."""
        seeded, self.float_seeds = _seed_floats(self.module, self.seed_dir)
        return seeded

    def _keep_floats(self, seeded: Optional[tuple]) -> None:
        self.float_seeds_learned = _keep_floats(seeded)

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop recording and wait for an in-flight compile (it holds the module); used on unload."""
        self._closed.set()
        with self._cv:
            self._cv.notify_all()  # a parked compile wakes, sees closed and ends
        self._remove_hook()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: waiting for the background compile before unloading"
                )
            thread.join(timeout)
        if self.pending():
            self._finish("failed", "closed")
        # Unload uninstalls the CUDA-graph layer by identity: the gate in front of it goes first.
        self._remove_gate()
        self.module = None

    def describe(self) -> dict:
        return {
            "state": self.state,
            "samples": len(self.samples),
            "compile_s": None if self.compile_s is None else round(self.compile_s, 2),
            "eager_generations": self.eager_generations,
            "pauses": self.pauses,
            "paused_s": round(self.paused_s, 2),
            "float_seeds": self.float_seeds,
            "idle_only": self.idle,
            "freezes": self.freezes,
            "thaws": self.thaws,
            "pause_wait_s": round(self.pause_wait_s, 3),
            "error": self.error,
        }


def select_module(
    pipe: Any,
    *,
    speed_optims: Any,
    default_tier: bool,
    quantized: bool,
    gguf: bool,
    step_cache: bool,
    device: Any,
    backend: Any,
    denoiser_hooked: bool,
) -> Any:
    """The one dense default-tier CUDA denoiser whose compile may move off the render, or None.

    Not torchao (~30x slower eager), GGUF (compiles only its dequant), a step cache (toggles graphs per step) or
    max-autotune (an explicit request to pay the compile). An offloaded denoiser only with ``offloaded_enabled``."""
    if not enabled():
        return None
    if "compiled" not in tuple(speed_optims or ()) or not default_tier:
        return None
    if quantized or gguf or step_cache or (denoiser_hooked and not offloaded_enabled()):
        return None
    if device != "cuda" or backend == "rocm":
        return None
    try:
        from .diffusion_cuda_graph import _denoiser_modules
        modules = _denoiser_modules(pipe)
    except Exception:  # noqa: BLE001
        return None
    return modules[0] if len(modules) == 1 else None


def arm(
    module: Any,
    *,
    logger: Any = None,
    seed_dir: Any = None,
) -> Optional[BackgroundCompile]:
    """A recording ``BackgroundCompile`` on ``module``, or None (disabled / no hook)."""
    if module is None or not enabled():
        return None
    bg = BackgroundCompile(module, logger = logger, seed_dir = seed_dir)
    if not bg.install():
        return None
    return bg
