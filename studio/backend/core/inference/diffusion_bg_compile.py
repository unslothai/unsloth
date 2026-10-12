# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile a dense denoiser in the background while the user's renders keep running eager.

``arm`` records the first few distinct denoiser inputs of a render run under ``force_eager`` (read by every compile
guard and CUDA-graph wrapper); ``kick`` replays them through the compiled callable on a daemon thread with CUDA-graph
capture suppressed, so the next render pays only the capture. A render arriving mid-compile waits for it: a render
beside the compile was starved to 49-62 s on a B200. Any failure ends the attempt and the next render compiles inline.

Dense, non-offloaded denoisers only: torchao ones are ~30x slower eager, and offload hooks would fight two forwards.
By default only the deferred profile (generation 3, already eager -> compiled) uses it; UNSLOTH_DIFFUSION_BG_COMPILE=1
also moves a load's first-render compile (render 1 then differs from render 2 for one seed), =0 disables it.
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

# ContextVars, not thread-locals: diffusion_render_thread.run copies the caller's context onto the render thread.
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
    ) -> None:
        self.module = module
        self.logger = logger
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
                # Like the capture's static buffers (made outside inference mode): dynamo guards on inference-ness.
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
            self._thread = threading.Thread(
                target = self._run, name = "unsloth-diffusion-bg-compile", daemon = True
            )
            self._thread.start()
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
        try:
            import torch

            from . import diffusion_compile_config

            # torch 2.12+ keeps compile config per context: a fresh thread would compile without the recorded knobs.
            diffusion_compile_config.apply()
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

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop recording and wait for an in-flight compile (it holds the module); used on unload."""
        self._closed.set()
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
    """The one dense, resident, default-tier CUDA denoiser whose compile may move off the render, or None.

    Not torchao (~30x slower eager), GGUF (compiles only its dequant), a step cache (toggles graphs per step),
    max-autotune (an explicit request to pay the compile) or an offloaded denoiser."""
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
