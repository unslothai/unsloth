# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A small job service for diffusion LoRA training.

This is deliberately separate from the LLM ``TrainingBackend``: that backend's
lifecycle (LLM config build, per-run SQLite rows, matplotlib plots, transfer-to-chat-
inference) is specific to text training and would mis-handle a diffusion run. This
service does only what a diffusion job needs -- spawn the trainer subprocess, pump its
events (``model_load_*`` / ``progress`` / ``complete`` / ``error``) into an in-memory
status snapshot, and support stop -- and is polled over JSON by the route layer.

The subprocess context, target, and queues are injectable so the service can be unit
tested without real multiprocessing or torch: tests pass a fake context whose Process
runs a scripted target on a thread.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
import uuid
from typing import Any, Callable, Optional

# Spawn (not fork): a fresh interpreter, matching the LLM training worker, so CUDA/torch
# state from the parent never leaks into the trainer.
_CTX = mp.get_context("spawn")

# Terminal event types after which the pump stops.
_TERMINAL = ("complete", "error")


def _finite_or_none(value: Any) -> Optional[float]:
    """Coerce a numeric progress field to a finite float, or None. A divergent run (or a
    grad clip that returns inf) can push loss / grad_norm to NaN or +/-Infinity, and those
    are invalid in strict JSON -- FastAPI's encoder would emit the JS-only NaN/Infinity
    tokens that break a strict client parse. Nulling them here (the single service ingestion
    point both trainers feed) keeps every status snapshot and persisted record JSON-safe."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _run_diffusion_child(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    # Imported lazily so this module (and the route layer) stays torch-free at import.
    from .diffusion_lora_trainer import run_diffusion_training_process
    run_diffusion_training_process(event_queue = event_queue, stop_queue = stop_queue, config = config)


def _default_target(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    # First thing in the spawned child (before torch is imported): bind to the parent's
    # death on Linux and scrub the native path lease secret, exactly like the inference /
    # export / LLM-training workers. multiprocessing children cannot be given a
    # parent-set preexec_fn, so the child must self-bind; otherwise a Studio crash or
    # kill leaves this trainer holding the GPU. Tests inject their own target, so this
    # binding only runs for the real production spawn.
    from utils.native_path_leases import run_without_native_path_secret
    run_without_native_path_secret(
        _run_diffusion_child, event_queue = event_queue, stop_queue = stop_queue, config = config
    )


# Cap on retained metric points. When exceeded, the arrays are decimated (every other
# point dropped) so a long run stays bounded in memory while the live loss chart keeps a
# faithful shape. 4000 points comfortably covers a typical run at full resolution.
_METRIC_CAP = 4000


def _idle_state() -> dict[str, Any]:
    return {
        "active": False,
        "job_id": None,
        "status": "idle",
        "message": "",
        "step": 0,
        "total_steps": 0,
        "loss": None,
        "avg_loss": None,
        "learning_rate": None,
        "grad_norm": None,
        "num_images": None,
        "in_model_load": False,
        "output_dir": None,
        "lora_path": None,
        "catalog_path": None,
        "family": None,
        "base_model": None,
        "samples_per_second": None,
        "peak_memory_gb": None,
        "started_at": None,
        "updated_at": None,
        # Bounded, paired history arrays for the live loss chart (see _append_metric).
        "metric_steps": [],
        "metric_loss": [],
        "metric_lr": [],
        "metric_grad_norm": [],
    }


def _append_metric(
    state: dict[str, Any],
    step: Any,
    loss: Any,
    lr: Any,
    grad_norm: Any = None,
) -> None:
    """Append one (step, loss, lr, grad_norm) point to the bounded history arrays on
    ``state``.

    Only records finite, positive-step points (mirrors the LLM trainer, which logs history
    only for step > 0 with a real loss). When the arrays hit ``_METRIC_CAP`` they are
    decimated in place (keep every other point) so appends stay bounded without losing the
    curve's shape. lr / grad_norm may be None (kept as None so those series can be
    sparse while staying index-aligned with ``steps``)."""
    try:
        istep = int(step)
    except (TypeError, ValueError):
        return
    if istep <= 0 or loss is None:
        return
    floss = _finite_or_none(loss)
    if floss is None:  # non-numeric or non-finite (NaN/Inf): skip, keep the curve JSON-safe
        return
    # lr / grad_norm may be None (sparse series) or non-finite; a non-finite value is
    # nulled, not dropped, so a bad point never taints the (loss-driven) history.
    flr = _finite_or_none(lr)
    fgn = _finite_or_none(grad_norm)
    steps = state["metric_steps"]
    losses = state["metric_loss"]
    lrs = state["metric_lr"]
    gns = state["metric_grad_norm"]
    if len(steps) >= _METRIC_CAP:
        state["metric_steps"] = steps[::2]
        state["metric_loss"] = losses[::2]
        state["metric_lr"] = lrs[::2]
        state["metric_grad_norm"] = gns[::2]
        steps, losses, lrs, gns = (
            state["metric_steps"],
            state["metric_loss"],
            state["metric_lr"],
            state["metric_grad_norm"],
        )
    steps.append(istep)
    losses.append(floss)
    lrs.append(flr)
    gns.append(fgn)


class DiffusionTrainingService:
    """One diffusion LoRA training job at a time, spawned as a subprocess."""

    def __init__(
        self,
        *,
        ctx: Any = None,
        target: Optional[Callable[..., None]] = None,
    ) -> None:
        self._ctx = ctx if ctx is not None else _CTX
        self._target = target if target is not None else _default_target
        self._lock = threading.Lock()
        self._proc: Any = None
        self._stop_queue: Any = None
        self._pump: Optional[threading.Thread] = None
        self._state: dict[str, Any] = _idle_state()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def is_active(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.is_alive()

    def start(self, config: dict) -> str:
        """Validate ``config``, spawn the trainer, and start pumping its events.

        Raises ValueError for an unusable config (before any spawn) and RuntimeError if a
        job is already running. Returns the new job id."""
        # Validate cheaply BEFORE spawning so a bad request fails fast with a clear error.
        from .diffusion_lora_trainer import _config_from_dict

        _config_from_dict(config).normalized()

        # Join a finished job's pump OUTSIDE the lock: its final state writes take this
        # lock (via _apply_event / the exit handler), so joining under it would stall
        # the start for the whole timeout and then let the stale pump overwrite the new
        # job's state once the lock was released.
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                raise RuntimeError("A diffusion training job is already running.")
            pump = self._pump
        if pump is not None and pump.is_alive():
            pump.join(timeout = 5.0)

        with self._lock:
            # Re-check: another start() may have won the race while we joined.
            if self._proc is not None and self._proc.is_alive():
                raise RuntimeError("A diffusion training job is already running.")

            job_id = uuid.uuid4().hex
            event_queue = self._ctx.Queue()
            self._stop_queue = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target = self._target,
                kwargs = {
                    "event_queue": event_queue,
                    "stop_queue": self._stop_queue,
                    "config": config,
                },
                daemon = True,
            )
            self._proc.start()
            try:
                from utils.process_lifetime import adopt_pid
                adopt_pid(self._proc.pid)  # bind to parent lifetime (no zombie on exit)
            except Exception:  # noqa: BLE001 -- lifetime binding is best-effort
                pass

            now = time.time()
            self._state = _idle_state()
            self._state.update(
                active = True,
                job_id = job_id,
                status = "running",
                message = "Starting diffusion LoRA training...",
                base_model = config.get("base_model") or config.get("model_name"),
                started_at = now,
                updated_at = now,
            )
            self._pump = threading.Thread(
                target = self._pump_loop, args = (event_queue, self._proc), daemon = True
            )
            self._pump.start()
            return job_id

    def stop(self, save: bool = True) -> bool:
        """Request a clean stop: the trainer finishes the current step, then either saves
        a partial adapter (``save=True``, the default) or discards the run (``save=False``,
        matching the LLM trainer's cancel). Returns True if a stop was signalled, False if
        nothing was running."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive() or self._stop_queue is None:
                return False
            try:
                # Bare True keeps the wire format older trainers expect; the dict form
                # carries the no-save cancel flag the trainer's _check_stop understands.
                self._stop_queue.put(True if save else {"save": False})
            except Exception:  # noqa: BLE001
                return False
            self._state["message"] = (
                "Stop requested; finishing the current step and saving a partial adapter..."
                if save
                else "Cancel requested; finishing the current step (no adapter will be saved)..."
            )
            self._state["updated_at"] = time.time()
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            # Keep ``active`` honest even if the process died between events.
            snap["active"] = self._proc is not None and self._proc.is_alive()
            return snap

    # ── event pump ───────────────────────────────────────────────────────────
    def _pump_loop(self, event_queue: Any, proc: Any) -> None:
        while True:
            try:
                ev = event_queue.get(timeout = 1.0)
            except Exception:  # noqa: BLE001 -- Empty (timeout) or a closed queue
                if not proc.is_alive():
                    # Drain anything buffered, then decide if it exited cleanly.
                    drained = False
                    while True:
                        try:
                            self._apply_event(event_queue.get_nowait(), proc = proc)
                            drained = True
                        except Exception:  # noqa: BLE001
                            break
                    with self._lock:
                        if self._proc is not proc:
                            return  # superseded by a newer job; don't touch its state
                        if self._state.get("status") not in ("completed", "stopped", "error"):
                            self._state.update(
                                active = False,
                                status = "error",
                                message = "Training process exited unexpectedly.",
                                updated_at = time.time(),
                            )
                    _ = drained
                    return
                continue
            self._apply_event(ev, proc = proc)
            if ev.get("type") in _TERMINAL:
                return

    def _apply_event(
        self,
        ev: dict[str, Any],
        proc: Any = None,
    ) -> None:
        """Fold one trainer event into the status snapshot. Pure state update -- unit
        tested by feeding events directly. ``proc`` (when given) fences a stale pump:
        an event from a superseded job's process must not touch the current job's
        state."""
        etype = ev.get("type")
        with self._lock:
            if proc is not None and self._proc is not proc:
                return
            s = self._state
            s["updated_at"] = time.time()
            if etype == "model_load_started":
                s.update(in_model_load = True, status = "running", message = "Loading base model...")
                if ev.get("num_images") is not None:
                    s["num_images"] = ev.get("num_images")
            elif etype == "model_load_completed":
                s.update(in_model_load = False, message = "Training...")
            elif etype == "preparing":
                # A long precompute phase (e.g. the VAE latent cache) between model load and
                # the first step; surfaced so the UI shows visible progress instead of a
                # silent "Loading base model..." stall.
                done, total = ev.get("done"), ev.get("total")
                stage = str(ev.get("stage", "prepare")).replace("_", " ")
                s.update(
                    status = "running",
                    in_model_load = True,
                    message = (
                        f"Preparing ({stage} {done}/{total})..."
                        if done is not None and total is not None
                        else f"Preparing ({stage})..."
                    ),
                )
            elif etype == "warning":
                # Non-fatal trainer notes (e.g. torch.compile falling back to eager); keep
                # training state, surface the text.
                s["message"] = str(ev.get("message", "warning"))
            elif etype == "progress":
                # Null any non-finite float (NaN/Inf from a divergent step or an inf grad
                # norm) so the JSON status stays strict-parseable; a missing key keeps the
                # last value, a present-but-non-finite one becomes None.
                loss = _finite_or_none(ev["loss"]) if "loss" in ev else s["loss"]
                avg_loss = _finite_or_none(ev["avg_loss"]) if "avg_loss" in ev else s["avg_loss"]
                learning_rate = (
                    _finite_or_none(ev["learning_rate"])
                    if "learning_rate" in ev
                    else s["learning_rate"]
                )
                grad_norm = (
                    _finite_or_none(ev["grad_norm"]) if "grad_norm" in ev else s["grad_norm"]
                )
                s.update(
                    status = "running",
                    step = ev.get("step", s["step"]),
                    total_steps = ev.get("total_steps", s["total_steps"]),
                    loss = loss,
                    avg_loss = avg_loss,
                    learning_rate = learning_rate,
                    grad_norm = grad_norm,
                    message = "Training...",
                )
                # Fold optional perf fields (emitted by the trainers) so the UI can show
                # throughput + peak VRAM without a separate channel.
                if ev.get("samples_per_second") is not None:
                    s["samples_per_second"] = ev.get("samples_per_second")
                if ev.get("peak_memory_gb") is not None:
                    s["peak_memory_gb"] = ev.get("peak_memory_gb")
                # Retain a bounded (step, loss, lr, grad_norm) history for the live charts.
                _append_metric(
                    s,
                    ev.get("step"),
                    ev.get("loss"),
                    ev.get("learning_rate"),
                    ev.get("grad_norm"),
                )
            elif etype == "complete":
                # Reset in_model_load: a stop during model load emits complete without a
                # preceding model_load_completed, which would otherwise leave a stale
                # loading indicator after the job ended.
                s.update(
                    active = False,
                    in_model_load = False,
                    status = "stopped" if ev.get("stopped") else "completed",
                    output_dir = ev.get("output_dir"),
                    lora_path = ev.get("lora_path"),
                    message = (
                        "Stopped (partial adapter saved)."
                        if ev.get("lora_path")
                        else "Stopped (no adapter saved)."
                    )
                    if ev.get("stopped")
                    else "Training complete.",
                )
                if ev.get("catalog_path") is not None:
                    s["catalog_path"] = ev.get("catalog_path")
                if ev.get("family") is not None:
                    s["family"] = ev.get("family")
                if ev.get("base_model") is not None:
                    s["base_model"] = ev.get("base_model")
            elif etype == "error":
                # Reset in_model_load too: an error raised during model loading has no
                # model_load_completed, so the terminal state must clear it explicitly.
                s.update(
                    active = False,
                    in_model_load = False,
                    status = "error",
                    message = str(ev.get("message", "error")),
                )


_service: Optional[DiffusionTrainingService] = None
_service_lock = threading.Lock()


def get_diffusion_training_service() -> DiffusionTrainingService:
    """Process-wide singleton used by the route layer."""
    global _service
    with _service_lock:
        if _service is None:
            _service = DiffusionTrainingService()
        return _service
