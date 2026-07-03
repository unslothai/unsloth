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
        "num_images": None,
        "in_model_load": False,
        "output_dir": None,
        "lora_path": None,
        "started_at": None,
        "updated_at": None,
    }


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
                started_at = now,
                updated_at = now,
            )
            self._pump = threading.Thread(
                target = self._pump_loop, args = (event_queue, self._proc), daemon = True
            )
            self._pump.start()
            return job_id

    def stop(self) -> bool:
        """Request a clean stop (the trainer finishes the current step and saves a partial
        adapter). Returns True if a stop was signalled, False if nothing was running."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive() or self._stop_queue is None:
                return False
            try:
                self._stop_queue.put(True)
            except Exception:  # noqa: BLE001
                return False
            self._state["message"] = "Stop requested; finishing the current step..."
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
            elif etype == "progress":
                s.update(
                    status = "running",
                    step = ev.get("step", s["step"]),
                    total_steps = ev.get("total_steps", s["total_steps"]),
                    loss = ev.get("loss", s["loss"]),
                    avg_loss = ev.get("avg_loss", s["avg_loss"]),
                    learning_rate = ev.get("learning_rate", s["learning_rate"]),
                    message = "Training...",
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
                    message = "Stopped (partial adapter saved)."
                    if ev.get("stopped")
                    else "Training complete.",
                )
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
