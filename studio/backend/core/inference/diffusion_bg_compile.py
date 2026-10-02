# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile a dense denoiser in the background while the user's renders keep running eager.

The regional / whole-module ``torch.compile`` of the denoiser is lazy: the first forward after the speed profile
engages traces and codegens every graph, inside a user render. On a dense (bf16 / fp16) denoiser eager is a usable
speed, so that compile is pure waiting: SDXL's deferred profile turned render 3 into a 66 s render on a B200 (68.9 s on
an H100) against 1.3 s eager, with no compiled bundle on disk yet.

Flow, per armed denoiser:

1. ``arm`` (under the generate lock, right after the speed optims installed the compile wrappers): a forward pre-hook
   on the denoiser starts recording. The generation then runs with ``force_eager`` set, which every compile guard
   (``diffusion_speed``) and every CUDA-graph wrapper (``diffusion_cuda_graph``) reads: they call the eager forward,
   never dynamo, never a capture. The hook clones the first few distinct input trees it sees (one per input shape:
   a CFG pair with two text lengths records two).
2. ``kick`` (after that generation returned): the hook is removed and a daemon thread replays each recorded input
   through the COMPILED callable on a side stream, under the grad / inference mode the render had, with
   ``capture_suppressed`` set so no CUDA graph is recorded off the render thread. Compiling is what it is for; the
   output is discarded.
3. Once every sample replayed, ``pending()`` turns False and the NEXT generation runs the compiled path as before:
   dynamo's cache already holds the graphs, so it pays only the CUDA-graph capture. A generation that starts while
   the compile is still running waits for it (``wait``) rather than running beside it, so the compile only ever
   overlaps the time between renders: a user reading the last image hides it, a back-to-back caller sees the
   remainder once, one render later than before.

Anything unexpected (an input tree that cannot be cloned, an exception in the warm forward) ends the background
attempt with ``pending()`` False, which is exactly the old behaviour: the next generation compiles inline. A compile
failure inside the warm is the compile guard's business, as on the render thread (it routes the denoiser eager for
the rest of the load).

Only for dense, non-offloaded denoisers: torchao-quantised ones are ~30x slower eager (they must compile before the
first step) and an offload hook moves weights per block, which two concurrent forwards would fight over.

By default only the deferred profile (generation 3) compiles in the background: it already switched eager ->
compiled, so the same seed repeats exactly as before. UNSLOTH_DIFFUSION_BG_COMPILE=1 also moves a load's first-render
compile off the render (image and video), at the cost of render 1 (eager) differing from render 2 (compiled) for one
seed. UNSLOTH_DIFFUSION_BG_COMPILE=0 disables it (the compile lands on the render, as before).
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import time
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE"
_MAX_SAMPLES = 4

# Read by the compile guards and the CUDA-graph wrappers. ContextVars, not thread-locals: the render thread runs each
# generation inside a copy of the caller's context (diffusion_render_thread.run), so a value set around a generation
# reaches the thread that actually runs the denoiser, and never leaks into the background compile thread, which starts
# with a fresh context.
_FORCE_EAGER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_force_eager", default = False
)
_NO_CAPTURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_no_capture", default = False
)


def enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def load_time_enabled() -> bool:
    """Whether a load may arm the background compile for its first render (opt-in: ``=1``).

    Off by default: a load that used to compile inside render 1 would render 1 eager and 2 compiled, so the same seed
    twice in a row no longer repeats. The deferred profile (generation 3) already switched eager -> compiled, so it
    keeps the background compile by default."""
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw in ("1", "true", "yes", "on", "all")


def eager_forced() -> bool:
    """True inside a generation that must not touch the compiled callable (its compile is still in flight)."""
    return _FORCE_EAGER.get()


def capture_suppressed() -> bool:
    """True on the background compile thread: run compiled, never record a CUDA graph."""
    return _NO_CAPTURE.get()


@contextlib.contextmanager
def force_eager(on: bool = True):
    token = _FORCE_EAGER.set(bool(on))
    try:
        yield
    finally:
        _FORCE_EAGER.reset(token)


def _cuda_graph_helpers() -> tuple[Callable, Callable, Callable]:
    from .diffusion_cuda_graph import _flatten, _rebuild, graph_key

    return _flatten, _rebuild, graph_key


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
        isinstance(c, GraphedForward) and c.enabled and not c.poisoned and not c.bypassed for c in candidates
    )


class BackgroundCompile:
    """One armed denoiser: records eager inputs, then compiles from them on a daemon thread."""

    def __init__(self, module: Any, *, logger: Any = None, max_samples: int = _MAX_SAMPLES) -> None:
        self.module = module
        self.logger = logger
        self.max_samples = int(max_samples)
        self.samples: list[tuple] = []
        self._keys: set = set()
        self._handle: Any = None
        # (gate, inner) while a whole-module compiled denoiser (SDXL's U-Net) is gated, else None.
        self._gate: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # recording -> compiling -> done | failed. pending() is True for the first two.
        self.state = "recording"
        self.error: Optional[str] = None
        self.compile_s: Optional[float] = None
        self.eager_generations = 0

    # ------------------------------------------------------------------ recording
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
        """Route a WHOLE-module compiled denoiser to its eager ``_call_impl`` under ``force_eager``.

        A regionally compiled DiT needs nothing here (its block guards read the flag), but ``Module.compile`` serves
        every call from ``_compiled_call_impl`` (whatever wraps it: the CUDA-graph layer, or the raw compiled callable
        when graphs are off), so the flag has to be read in front of it."""
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
        if not eager_forced() or len(self.samples) >= self.max_samples or self.state != "recording":
            return None
        try:
            import torch

            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            key = graph_key((args, kwargs))
            if key in self._keys:
                return None
            live: list = []
            spec = _flatten((args, kwargs), live)
            if _capture_armed(module):
                # The compiled callable will first be reached from the CUDA-graph layer's capture warm-up, which feeds
                # it STATIC buffers made outside inference mode (diffusion_cuda_graph._capture). Dynamo guards on the
                # dispatch keys an inference tensor lacks, so warming with the render's own (inference) tensors would
                # compile a graph the capture then misses and recompiles: build the buffers the same way.
                with torch.inference_mode(False):
                    clones = [torch.empty_like(t) for t in live]
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
        return None

    def _remove_hook(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ lifecycle
    def pending(self) -> bool:
        return self.state in ("recording", "compiling")

    def note_eager_generation(self) -> None:
        self.eager_generations += 1

    def compiling(self) -> bool:
        thread = self._thread
        return self.state == "compiling" and thread is not None and thread.is_alive()

    def wait(self, cancel: Any = None, poll_s: float = 0.25) -> float:
        """Block until an in-flight background compile ends; returns the seconds waited.

        A render never runs next to the compile. Measured on a B200 (SDXL, render 4 eager beside the compile): the
        render took 49-62 s instead of 1.3 s, because dynamo / inductor hold the GIL for most of a compile and make_fx
        patches ``nn.Module.__call__`` process-wide while it traces, so every module call of the eager render went
        through the tracer's wrapper. Waiting costs the same wall time with neither hazard, and is only reached when
        the user starts the next render before the compile (which runs between renders) is done. ``cancel`` (an
        Event) aborts the wait, not the compile."""
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
            # Before the thread starts: the compiled trace must not contain the recording hook.
            self._remove_hook()
            self.state = "compiling"
            self._thread = threading.Thread(
                target = self._run, name = "unsloth-diffusion-bg-compile", daemon = True
            )
            self._thread.start()
            return True

    def _finish(self, state: str, error: Optional[str] = None) -> None:
        self.state = state
        self.error = error
        self.samples = []
        self._remove_hook()

    def _run(self) -> None:
        t0 = time.perf_counter()
        token = _NO_CAPTURE.set(True)
        try:
            import torch

            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            for spec, clones, inference, grad, device in list(self.samples):
                if self._closed.is_set():
                    self._finish("failed", "closed before the compile finished")
                    return
                if device is not None:
                    torch.cuda.set_device(device)
                args, kwargs = _rebuild(spec, clones)
                # The default stream, like a render: no render runs beside this (a render waits for it), and offload
                # hooks synchronise against the current stream.
                mode = torch.inference_mode() if inference else torch.set_grad_enabled(grad)
                with mode:
                    self.module(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            self.compile_s = time.perf_counter() - t0
            self._finish("done")
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: %s compiled in the background in %.1f s; the next render runs compiled",
                    type(self.module).__name__,
                    self.compile_s,
                )
        except BaseException as exc:  # noqa: BLE001 - never kills the process; the next render compiles inline
            self.compile_s = time.perf_counter() - t0
            self._finish("failed", f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")
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

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop recording and wait for an in-flight compile (it holds the module); used on unload."""
        self._closed.set()
        self._remove_hook()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            if self.logger is not None:
                self.logger.info("diffusion.bg_compile: waiting for the background compile before unloading")
            thread.join(timeout)
        if self.pending():
            self._finish("failed", "closed")
        # Unload uninstalls the CUDA-graph layer by identity, so the gate in front of it has to go first.
        self._remove_gate()
        self.module = None

    def describe(self) -> dict:
        return {
            "state": self.state,
            "samples": len(self.samples),
            "compile_s": None if self.compile_s is None else round(self.compile_s, 2),
            "eager_generations": self.eager_generations,
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
    """The denoiser whose compile may move off the render, or None.

    One dense, resident, regionally or whole-module compiled denoiser on the default tier of a CUDA load: a torchao
    denoiser is ~30x slower eager than the compile costs, GGUF compiles only its dequant chain, a step cache toggles
    graphs per step, max-autotune is an explicit request to pay the compile, and a denoiser an offload hook moves is
    not the module a warm forward would compile against."""
    if not enabled():
        return None
    if "compiled" not in tuple(speed_optims or ()) or not default_tier:
        return None
    if quantized or gguf or step_cache or denoiser_hooked:
        return None
    if device != "cuda" or backend == "rocm":
        return None
    try:
        from .diffusion_cuda_graph import _denoiser_modules

        modules = _denoiser_modules(pipe)
    except Exception:  # noqa: BLE001
        return None
    return modules[0] if len(modules) == 1 else None


def arm(module: Any, *, logger: Any = None) -> Optional[BackgroundCompile]:
    """A recording ``BackgroundCompile`` on ``module``, or None (disabled / no hook)."""
    if module is None or not enabled():
        return None
    bg = BackgroundCompile(module, logger = logger)
    if not bg.install():
        return None
    return bg
