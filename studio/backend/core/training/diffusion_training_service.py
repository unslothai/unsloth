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

import contextlib
import json
import math
import multiprocessing as mp
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

# Spawn (not fork): a fresh interpreter so parent CUDA/torch state never leaks into the trainer.
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
    # First thing in the child (before torch): self-bind to parent death and scrub the native path secret, like the other workers.
    from utils.native_path_leases import run_without_native_path_secret
    run_without_native_path_secret(
        _run_diffusion_child, event_queue = event_queue, stop_queue = stop_queue, config = config
    )


# Cap on retained metric points; over it, arrays are decimated so a long run stays bounded while the chart keeps its shape.
_METRIC_CAP = 4000


class TrainingActiveError(RuntimeError):
    """A dataset mutation was refused because a run owns the dataset (route: 409)."""


class DatasetMutationInFlight(RuntimeError):
    """A start was refused because a dataset mutation is open (route: 409)."""


def _llm_training_active() -> bool:
    """Whether the LLM trainer holds the GPU. Best-effort: an import/health failure must not
    block a diffusion start, exactly as the route's own reciprocal check fails open."""
    try:
        from core.training import get_training_backend
        return bool(get_training_backend().is_training_active())
    except Exception:  # noqa: BLE001
        return False


# ── persisted run history ──────────────────────────────────────────────────────
# Every terminal run is recorded as one JSON file (summary + scrubbed config + metric logs) for the Train tab's history. JSON, not the LLM sqlite, so diffusion runs stay off the LLM Runs page.
def _runs_dir() -> Path:
    from utils.paths.storage_roots import studio_root

    d = studio_root() / "runs" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    return d


def _resume_fields(
    output_dir: Optional[str],
    *,
    status: Optional[str] = None,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    total_steps: Optional[int] = None,
    write_error: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    source_created_at: Optional[float] = None,
) -> dict[str, Any]:
    """``can_resume`` / ``checkpoint_step`` / ``resume_blocked_reason`` for a run, read from
    the checkpoints that are actually on disk.

    Derived, not trusted: a persisted record is a snapshot of the moment the run ended, but
    the user can delete the output folder afterwards. ``started_at`` fences off bundles an
    EARLIER run of the same adapter name left in the same folder, and ``ended_at`` fences off
    the ones a LATER run put there after this one finished -- without the upper bound a
    finished run offers, and resumes, its successor's training state. ``write_error`` is the run's
    own report that a checkpoint write failed; that is sticky and blocks resume (mirroring the
    MLX trainer's ``resume_blocked``), because whatever older state is on disk predates the
    adapter that was published, so continuing from it would silently lose steps. Never raises."""
    try:
        from core.training.diffusion_checkpoint import describe_resume_state
        state = describe_resume_state(
            output_dir,
            status = status,
            started_at = started_at,
            ended_at = ended_at,
            total_steps = total_steps,
            source_checkpoint = source_checkpoint,
            source_created_at = source_created_at,
        )
    except Exception:  # noqa: BLE001 -- history must never break on a checkpoint scan
        state = {}
    if write_error:
        return {
            "can_resume": False,
            "checkpoint_step": state.get("checkpoint_step"),
            "checkpoint_path": None,
            "resume_blocked_reason": write_error,
        }
    can_resume = bool(state.get("can_resume"))
    return {
        "can_resume": can_resume,
        "checkpoint_step": state.get("checkpoint_step"),
        # The EXACT bundle this run would continue, so the client resumes the one it was shown
        # rather than "whatever is newest in that folder" (which, in a folder two runs share,
        # can be a different run's).
        "checkpoint_path": state.get("checkpoint_path") if can_resume else None,
        "resume_blocked_reason": state.get("resume_blocked_reason"),
    }


def _refresh_resume_state(rec: dict) -> dict:
    """Re-derive a persisted record's resume fields from the checkpoints on disk, in place.

    Also backfills ``output_dir`` from the stored config for a record written before that field
    existed: the UI replays that path as ``resume_from_checkpoint``, so reporting ``can_resume``
    without it would enable an action the client then refuses on its own."""
    config = rec.get("config")
    output_dir = rec.get("output_dir") or (
        config.get("output_dir") if isinstance(config, dict) else None
    )
    if output_dir and not rec.get("output_dir"):
        rec["output_dir"] = str(output_dir)
    rec.update(
        _resume_fields(
            output_dir,
            status = rec.get("status"),
            started_at = rec.get("started_at"),
            ended_at = rec.get("ended_at"),
            # Same rule as the write side, and via the same helper: a run that died during
            # model loading never got a "resumed" event to seed total_steps, and zero here
            # sends the calculation back to the checkpoint manifest's OLDER target. In EPOCH
            # mode train_steps is the request model's unused default, so recomputing from it
            # here undid the persisted answer on every read -- a 600-step checkpoint of a run
            # resolved to 1000 read as 600/500 and Resume was disabled.
            total_steps = _resolved_total_steps(rec, config if isinstance(config, dict) else {}),
            write_error = rec.get("checkpoint_write_error"),
            # What this run itself resumed from, so a resumed run that died before its first
            # save can still offer the bundle it was validated against.
            source_checkpoint = (
                config.get("resume_from_checkpoint") if isinstance(config, dict) else None
            ),
            # And WHICH bundle that was, so a slot another run has since rewritten is not
            # offered back as this run's own lineage.
            source_created_at = rec.get("resumed_source_created_at"),
        )
    )
    return rec


def list_diffusion_runs(limit: int = 20) -> list[dict]:
    """Summaries of persisted diffusion runs, newest first. The heavy per-run payload
    (metric logs, config) stays in the file; fetch it via ``get_diffusion_run``."""
    try:
        files = sorted(_runs_dir().glob("*.json"), key = lambda p: p.stat().st_mtime, reverse = True)
    except Exception:  # noqa: BLE001 -- unreadable dir -> no history
        return []
    out: list[dict] = []
    for p in files[: max(0, int(limit))]:
        try:
            rec = json.loads(p.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001 -- a corrupt record never breaks the listing
            continue
        # Skip a wrong-shape record so one bad file cannot blow up the route's DiffusionTrainingRunSummary(**r) or the whole panel.
        if not isinstance(rec, dict):
            continue
        if not (isinstance(rec.get("job_id"), str) and isinstance(rec.get("status"), str)):
            continue
        # Resume state comes from the checkpoints on disk NOW, before the config is dropped.
        # Per record, because the refresh reads counters straight out of the file: one
        # hand-edited or older record with a nonnumeric total_steps took the whole endpoint
        # down, where every other kind of corruption here is skipped.
        try:
            _refresh_resume_state(rec)
            _restate_live_job(rec)
        except Exception:  # noqa: BLE001 -- a record we cannot refresh is still a record
            continue
        rec.pop("metric_history", None)
        rec.pop("config", None)
        out.append(rec)
    return out


def get_diffusion_run(job_id: str) -> Optional[dict]:
    """The full persisted record for one run (summary + config + metric logs)."""
    # Keyed by uuid4 hex; reject anything else so a crafted id can't traverse out of the dir.
    if not re.fullmatch(r"[0-9a-f]{32}", str(job_id or "")):
        return None
    p = _runs_dir() / f"{job_id}.json"
    try:
        rec = json.loads(p.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 -- missing/corrupt record
        return None
    if not isinstance(rec, dict):
        return rec
    try:
        _refresh_resume_state(rec)
        _restate_live_job(rec)
    except Exception:  # noqa: BLE001 -- the stored record beats a 500
        return rec
    return rec


def _restate_live_job(rec: dict) -> dict:
    """Undo the interim record's pessimism for the job that is still running, in place.

    An interim record is written as interrupted because that is the outcome if Studio never
    comes back. While the process IS still here and still on that job, the honest answer is
    running -- and reporting it as errored would offer a Resume for a directory the live run is
    writing into."""
    global _service
    service = _service
    if service is None:
        return rec
    try:
        live = service.status()
    except Exception:  # noqa: BLE001 -- a status we cannot read leaves the record as written
        return rec
    if not isinstance(live, dict) or not live.get("job_id"):
        return rec
    if rec.get("job_id") == live.get("job_id") and live.get("status") == "running":
        rec["status"] = "running"
        rec["message"] = live.get("message") or ""
        # And nothing resumable, which is what the interim record's error status made
        # _refresh_resume_state derive a moment ago. _UNRESUMABLE_STATUS rejects a running job
        # for a reason: its output directory is being written right now, so the only thing a
        # Resume action there can do is race the live run and 409.
        rec["can_resume"] = False
        rec["checkpoint_path"] = None
    return rec


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
        # The optional second (EMA-averaged) adapter, when ema_decay was enabled.
        "ema_path": None,
        "catalog_path": None,
        "family": None,
        "base_model": None,
        "samples_per_second": None,
        "peak_memory_gb": None,
        # Resume state for this job: where its newest checkpoint-<N> bundle is, the step it holds,
        # why one could not be written (surfaced as the Resume action's disabled tooltip), and the
        # step a resumed run picked up from.
        "checkpoint_path": None,
        "checkpoint_step": None,
        "resume_blocked_reason": None,
        "resumed_from_step": None,
        # The manifest timestamp of the bundle this run resumed FROM. A pathname is not an
        # identity: another run can write over the same checkpoint-<N> slot, and the fallback
        # would then offer that replacement back under this run's lineage.
        "resumed_source_created_at": None,
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
    curve's shape. lr / grad_norm may be None (kept as None so those series can be sparse
    while staying index-aligned with ``steps``)."""
    try:
        istep = int(step)
    except (TypeError, ValueError):
        return
    if istep <= 0 or loss is None:
        return
    floss = _finite_or_none(loss)
    if floss is None:  # non-numeric or non-finite: skip, keep the curve JSON-safe
        return
    # lr / grad_norm may be None or non-finite; non-finite is nulled (not dropped) to stay index-aligned.
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


def _resolved_total_steps(state: dict[str, Any], cfg: dict[str, Any]) -> int:
    """The step target this run was actually going to reach.

    ``train_steps`` is only meaningful when the run was NOT configured by epochs: the request
    model defaults it to 500 and ``num_epochs`` overrides it, so falling back to it in epoch
    mode invents a target the run never had -- a 600-step checkpoint of a run resolved to 1000
    then reads as 600/500 and the resume is refused. Zero is honest there, and the checkpoint
    manifest's own target (written with the resolved count) answers instead.
    """
    live = int(state.get("total_steps") or 0)
    if live:
        return live
    if int(cfg.get("num_epochs") or 0) > 0:
        return 0
    return int(cfg.get("train_steps") or 0)


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
        # Set by reserve() while a start is in flight (before the route frees GPU models) so the load guards refuse a concurrent load. Cleared by unreserve().
        self._reserved = False
        # Dataset mutations in flight. A start refuses while any is open and a mutation refuses once a start is reserved, both under _lock, so neither slips through the other's check-then-act window.
        self._dataset_mutations = 0
        # "Stop without saving" for the CURRENT job, remembered in the parent. The trainer
        # reports it on its completion event, but a child that is killed or OOMs after the
        # request never emits one -- and the unexpected-exit path then recorded a resumable
        # error run, re-offering Resume from the very checkpoint the user asked to discard.
        self._discard_requested = False
        # Bundle paths THIS job reported writing, so a discard the child could not carry out
        # itself removes exactly those and nothing that predated the run.
        self._own_checkpoints: list[str] = []
        # Set when the child reported a discarded completion, which it emits only AFTER running
        # its own clear_own_checkpoints. That helper hands a displaced slot back, so the path
        # this run wrote to can now hold ANOTHER run's restored bundle -- and the parent-side
        # cleanup, which knows only pathnames, would delete it.
        self._child_cleared_own = False
        # Set by a checkpoint_saved event, cleared once the pump has written the record.
        self._persist_interim = False
        # Whether this job has already had a stop signalled. Only the first one reaches the
        # child, so only the first one may set the parent's disposition.
        self._stop_signalled = False
        # GPU load admissions in flight (a load between its training guard and its arbiter registration). Same two-sided rule as the dataset mutations.
        self._gpu_admissions = 0
        self._proc: Any = None
        self._stop_queue: Any = None
        self._pump: Optional[threading.Thread] = None
        self._state: dict[str, Any] = _idle_state()
        # The active job's start config, scrubbed of secrets, kept for the run record.
        self._config: dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def is_active(self) -> bool:
        with self._lock:
            if self._reserved:
                return True
            return self._proc is not None and self._proc.is_alive()

    def reserve(self) -> None:
        """Mark a diffusion-training start as in flight so the image/video load guards (which
        read is_active) refuse a concurrent load BEFORE the route frees resident GPU models.
        Without this the training becomes active only at start(), after the free, so an
        overlapping load passes its guard, acquires the GPU, and both workloads allocate VRAM.

        Compare-and-set: raise if a start is already reserved or a job is already running, so a
        second overlapping /diffusion/start is rejected (409) BEFORE it frees GPU residents,
        instead of both requests tearing down residents and racing to start() (whichever finishes
        first wins, so a double-click or a retry with different parameters could start the wrong
        config). Paired with unreserve() in a finally by the reserving caller, so a failed start
        never leaves training 'active'."""
        with self._lock:
            if self._reserved or (self._proc is not None and self._proc.is_alive()):
                raise RuntimeError("A diffusion training job is already running.")
            if self._dataset_mutations:
                raise DatasetMutationInFlight(
                    "The training images are being changed right now. Wait for that to finish, "
                    "then start the run."
                )
            if self._gpu_admissions:
                # A load already passed its training guard and is about to take the GPU. Reserving now would free residents it has not
                # registered yet. Refusing is safe: the admission is held only across the registration, not the load itself.
                raise RuntimeError(
                    "A model is being loaded onto the GPU right now. Wait for that to finish, "
                    "then start the run."
                )
            # The LLM trainer under the SAME lock, not just at the route's earlier check: several network-bound preflights separate the
            # two, so an LLM start could spawn in between. The LLM route holds gpu_load_admission() across its own spawn, so one of the two always raises.
            if _llm_training_active():
                raise RuntimeError(
                    "An LLM training job is already running. "
                    "Stop it before starting diffusion (Images) training."
                )
            self._reserved = True

    def unreserve(self) -> None:
        """Clear the reservation set by reserve(). Only touches the reservation flag, never
        _proc, so a live job stays active on success and a failed start is fully rolled back."""
        with self._lock:
            self._reserved = False

    @contextlib.contextmanager
    def dataset_mutation(self):
        """Hold the dataset interlock for one mutation, refusing if a run owns the dataset.

        The route layer used to check ``is_active()`` and only then hand the filesystem work to a
        thread, so a ``/diffusion/start`` could reserve inside that gap: the caption or the image
        then changed underneath a preflight or a live trainer, which is what the immutability rule
        exists to prevent. Registering the mutation under the same lock ``reserve()`` uses closes
        it from both sides -- this raises once a start is committed, and ``reserve()`` raises while
        a mutation is open, so neither waits on the other (a start must never block on a
        minutes-long dataset import).
        """
        with self._lock:
            if self._reserved or (self._proc is not None and self._proc.is_alive()):
                raise TrainingActiveError(
                    "Training images cannot be changed while diffusion training is active. "
                    "Stop the run before uploading, importing, editing captions, or deleting images."
                )
            self._dataset_mutations += 1
        try:
            yield
        finally:
            with self._lock:
                self._dataset_mutations = max(0, self._dataset_mutations - 1)

    @contextlib.contextmanager
    def gpu_load_admission(self):
        """Hold the GPU-admission interlock across a load's guard -> arbiter -> registration.

        The load guards read ``is_active()`` and only THEN acquire the arbiter and register the
        load, so a start reserving inside that gap freed residents the load had not registered
        yet and the trainer came up beside a brand-new pipeline. Registering the admission under
        the same lock ``reserve()`` uses closes it from both sides, exactly like
        ``dataset_mutation``: this raises once a start is reserved or running, and ``reserve()``
        raises while an admission is open, so neither waits on the other.

        The span is deliberately short. ``begin_load`` returns as soon as the load is registered
        (the download and build run on a daemon thread), and from that point
        ``_free_gpu_for_diffusion_training`` preempts the in-flight load, so holding this for the
        whole load would block starts for minutes to no purpose."""
        with self._lock:
            if self._reserved or (self._proc is not None and self._proc.is_alive()):
                raise TrainingActiveError(
                    "Diffusion training is running, so the GPU is in use. Stop the run before "
                    "loading a model."
                )
            self._gpu_admissions += 1
        try:
            yield
        finally:
            with self._lock:
                self._gpu_admissions = max(0, self._gpu_admissions - 1)

    def start(self, config: dict) -> str:
        """Validate ``config``, spawn the trainer, and start pumping its events.

        Raises ValueError for an unusable config (before any spawn) and RuntimeError if a
        job is already running. Returns the new job id."""
        # Validate cheaply BEFORE spawning so a bad request fails fast with a clear error.
        from .diffusion_lora_trainer import _config_from_dict

        _config_from_dict(config).normalized()

        # Join a finished job's pump OUTSIDE the lock: its final state writes take this lock, so joining under it would stall the start and let the stale pump overwrite the new state.
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
            self._discard_requested = False
            self._own_checkpoints = []
            self._child_cleared_own = False
            self._stop_signalled = False
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
            # AFTER the reset above, which replaces the whole state dict. The route has already
            # validated and PINNED a source bundle, so its identity is known before the child
            # runs: recording it here means a resume that dies during the model load -- before
            # the trainer can emit "resumed" -- still has the timestamp its fallback is checked
            # against, instead of trusting the pathname and offering back whatever later
            # occupies that slot.
            self._seed_source_identity(config)
            # Keep the config (minus secrets) for the persisted run record.
            self._config = {k: v for k, v in dict(config).items() if k != "hf_token"}
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
            if self._stop_signalled:
                # The child consumes the FIRST signal and acts on it. A second one cannot
                # un-save an adapter it has already exported, so honouring a later
                # stop-without-saving set a parent discard the child never carried out: the
                # run was marked discarded and its checkpoints deleted while the adapter and
                # the catalog entry it published stayed on disk. The disposition is whichever
                # one the child actually got.
                return True
            try:
                # Bare True = older wire format; the dict form carries the no-save cancel flag.
                self._stop_queue.put(True if save else {"save": False})
            except Exception:  # noqa: BLE001
                return False
            self._stop_signalled = True
            if not save:
                # Remembered here so a child that dies before its discarded completion still
                # blocks the resume the user threw away.
                self._discard_requested = True
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
                        discarding = self._discard_requested
                    if discarding:
                        self._apply_discard_intent()
                    _ = drained
                    self._persist_run_record()
                    return
                continue
            self._apply_event(ev, proc = proc)
            if self._persist_interim:
                self._persist_interim = False
                self._persist_run_record(interim = True)
            if ev.get("type") in _TERMINAL:
                # An exception raised on the current step is a terminal `error`, and the pump
                # returns here rather than through the dead-process branch -- so the discard the
                # user asked for has to be applied on this path too.
                with self._lock:
                    discarding = self._discard_requested and self._proc is proc
                    child_cleared = self._child_cleared_own
                if discarding:
                    self._apply_discard_intent(delete = not child_cleared)
                self._persist_run_record()
                return

    def _seed_source_identity(self, config: dict[str, Any]) -> None:
        """Record which bundle the route accepted, before the child can report it itself."""
        source = config.get("resume_from_checkpoint")
        if not source:
            return
        try:
            from core.training.diffusion_checkpoint import read_checkpoint
            manifest = read_checkpoint(Path(str(source)).expanduser())
        except Exception:  # noqa: BLE001 -- an unreadable bundle simply has no identity here
            manifest = None
        if isinstance(manifest, dict):
            self._state["resumed_source_created_at"] = manifest.get("created_at")

    def _apply_discard_intent(self, *, delete: bool = True) -> None:
        """Carry out a stop-without-saving the child could not report itself.

        The trainer does this on its own completion path; a child that OOMs, is killed, or dies
        on the current step never gets there. Blocking the resume is the visible half -- the
        bundles are the other one, and they hold optimizer and scheduler state, are sizeable,
        and have no delete path in the UI once the run is marked discarded.

        ``delete`` False is the case where the child DID get there. Its cleanup restores any
        bundle this run wrote over, so the paths remembered here no longer name this run's
        bundles -- deleting them then destroys the predecessor that was just handed back, which
        is another run's resume point. The state half still applies either way.
        """
        with self._lock:
            own = list(self._own_checkpoints) if delete else []
            self._state["resume_blocked_reason"] = (
                "This run was stopped without saving, so it was discarded."
            )
            self._state["checkpoint_path"] = None
            self._state["checkpoint_step"] = None
            self._own_checkpoints = []
        if not own:
            return
        try:
            from core.training.diffusion_checkpoint import discard_named_checkpoints
            discard_named_checkpoints(own)
        except Exception:  # noqa: BLE001 -- cleanup must never break the terminal transition
            pass

    def _persist_run_record(self, *, interim: bool = False) -> None:
        """Best-effort JSON record of the finished run (summary + scrubbed config + the
        bounded metric logs) into the studio runs directory. Never fatal: history is a
        convenience, not part of the training contract.

        ``interim`` writes the same record for a run that is still going, which is what makes a
        checkpoint survive Studio itself dying: the bundle is on disk but only a terminal event
        used to write the JSON that Previous runs and its Resume action are built from. The
        status recorded is the one that is true if nothing else ever happens -- the run was
        interrupted -- and the terminal write replaces it in place."""
        try:
            with self._lock:
                s = dict(self._state)
                cfg = dict(self._config)
            if not s.get("job_id"):
                return
            if interim:
                s["status"] = "error"
                s["message"] = "Studio exited while this run was training."
            elif s.get("status") not in ("completed", "stopped", "error"):
                return
            adapter = s.get("output_dir") or cfg.get("output_dir")
            record = {
                "job_id": s.get("job_id"),
                "status": s.get("status"),
                "message": s.get("message") or "",
                "family": s.get("family") or cfg.get("model_family"),
                "base_model": s.get("base_model") or cfg.get("base_model"),
                "adapter": Path(str(adapter)).name if adapter else None,
                "instance_prompt": cfg.get("instance_prompt"),
                "step": s.get("step") or 0,
                "total_steps": s.get("total_steps") or 0,
                "loss": s.get("loss"),
                "avg_loss": s.get("avg_loss"),
                "learning_rate": s.get("learning_rate"),
                "grad_norm": s.get("grad_norm"),
                "samples_per_second": s.get("samples_per_second"),
                "peak_memory_gb": s.get("peak_memory_gb"),
                "num_images": s.get("num_images"),
                "started_at": s.get("started_at"),
                "ended_at": s.get("updated_at"),
                "lora_path": s.get("lora_path"),
                "ema_path": s.get("ema_path"),
                "catalog_path": s.get("catalog_path"),
                "saved": bool(s.get("lora_path")),
                # Resume lineage + state. output_dir is what a Resume replays, so it is recorded on
                # the run rather than only inside the config. can_resume / checkpoint_step are a
                # snapshot taken now and RE-DERIVED from disk on read, so deleting the checkpoints
                # later takes the action away instead of leaving a button that 400s.
                "output_dir": str(adapter) if adapter else None,
                "resumed_from_job_id": cfg.get("resumed_from_job_id") or None,
                "resumed_from_step": s.get("resumed_from_step"),
                "resumed_source_created_at": s.get("resumed_source_created_at"),
                "checkpoint_write_error": s.get("resume_blocked_reason") or None,
                **_resume_fields(
                    str(adapter) if adapter else None,
                    status = s.get("status"),
                    started_at = s.get("started_at"),
                    # The REQUESTED target, falling back to the config the start preflight
                    # accepted: total_steps is seeded by the "resumed" event, and a run that
                    # dies during model loading never gets one. Zero there sent the resume
                    # calculation back to the checkpoint manifest's older target, which then
                    # reported "nothing left to train" for a run whose whole point was a
                    # raised target.
                    #
                    # NOT in epoch mode, though: num_epochs overrides train_steps, which then
                    # still carries the Pydantic default of 500. A run resolved to 1000 that
                    # died before its "resumed" event would report a 600-step checkpoint as
                    # 600/500 and refuse the resume. Zero there is honest, and the manifest's
                    # own target (written with the resolved count) is the right answer.
                    total_steps = _resolved_total_steps(s, cfg),
                    write_error = s.get("resume_blocked_reason"),
                    source_checkpoint = cfg.get("resume_from_checkpoint"),
                    source_created_at = s.get("resumed_source_created_at"),
                ),
                "config": cfg,
                "metric_history": {
                    "steps": s.get("metric_steps") or [],
                    "loss": s.get("metric_loss") or [],
                    "lr": s.get("metric_lr") or [],
                    "grad_norm": s.get("metric_grad_norm") or [],
                },
            }
            path = _runs_dir() / f"{s['job_id']}.json"
            path.write_text(json.dumps(record), encoding = "utf-8")
        except Exception:  # noqa: BLE001 -- persisting history must never break the run
            pass

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
                # A long precompute phase (e.g. the VAE latent cache) before the first step; surfaced so the UI shows progress instead of a silent stall.
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
                # Non-fatal trainer notes; keep training state, surface the text.
                s["message"] = str(ev.get("message", "warning"))
            elif etype == "resumed":
                # The trainer restored a checkpoint and is continuing from it. Recorded so the run
                # history shows lineage instead of an unexplained run that starts mid-way. This is
                # the step resumed FROM, not a checkpoint this run wrote, so it does not touch
                # checkpoint_step (which tracks this run's own newest bundle).
                s.update(
                    resumed_from_step = ev.get("step"),
                    resumed_source_created_at = ev.get("source_created_at"),
                    message = f"Resuming from step {ev.get('step')}...",
                )
                # Seed the LIVE counters from the same event. They are what the UI renders and
                # what an errored run persists, and until the first post-resume progress event
                # they still read 0/0 -- so a resume of step 400 of 500 shows "0/0", and an OOM
                # on the first step records a failed run at step 0 with a target of 0.
                # checkpoint_step stays untouched: this bundle is not one this run wrote.
                if ev.get("step") is not None:
                    s["step"] = int(ev["step"])
                if ev.get("total_steps") is not None:
                    s["total_steps"] = int(ev["total_steps"])
            elif etype == "checkpoint_saved":
                # Folded as each bundle lands, not only at the end, so a run that later crashes is
                # still reported as resumable. A good write also clears an earlier write's failure.
                s.update(
                    checkpoint_path = ev.get("checkpoint_path"),
                    checkpoint_step = ev.get("step"),
                    resume_blocked_reason = None,
                )
                written = ev.get("checkpoint_path")
                if written and written not in self._own_checkpoints:
                    self._own_checkpoints.append(str(written))
                # The bundle is on disk; the run JSON is not, and only a terminal event writes
                # one. Studio being killed or shut down after a periodic save therefore left a
                # resumable checkpoint that Previous runs -- which reads those JSONs and
                # nothing else -- had no entry for, so the Resume action did not exist. Written
                # as an interrupted run because that is what it IS until the run ends: the
                # terminal persist overwrites the same file, and the live job is reported as
                # running by the reader below.
                self._persist_interim = True
            elif etype == "checkpoint_failed":
                # Sticky: whatever older bundle is on disk predates the work this run did, so
                # resuming from it would silently lose steps. Mirrors the MLX resume_blocked flag.
                s["resume_blocked_reason"] = str(ev.get("message", "checkpoint write failed"))
                # Persisted for the same reason a successful write is. In memory only, a Studio
                # exit after this left the last record advertising the OLDER checkpoint as
                # resumable -- the one this service has just decided is stale -- and resuming it
                # rolls the run back past everything after it. A later checkpoint_saved clears
                # the reason and replaces the record.
                self._persist_interim = True
            elif etype == "progress":
                # Null any non-finite float so the JSON stays strict-parseable; a missing key keeps the last value.
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
                # Fold optional perf fields so the UI shows throughput + peak VRAM.
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
                # Reset in_model_load: a stop during model load emits complete with no preceding model_load_completed, leaving a stale indicator.
                s.update(
                    active = False,
                    in_model_load = False,
                    status = "stopped" if ev.get("stopped") else "completed",
                    output_dir = ev.get("output_dir"),
                    lora_path = ev.get("lora_path"),
                    ema_path = ev.get("ema_path"),
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
                # Checkpoint state is folded from the per-save events above; only the lineage is
                # (re)confirmed here, for a pump that missed the earlier ``resumed`` event.
                if ev.get("resumed_from_step") is not None:
                    s["resumed_from_step"] = ev.get("resumed_from_step")
                if ev.get("discarded"):
                    # Cancelled with "stop without saving": the user threw this run away, so its
                    # own periodic checkpoints must not keep offering to continue it.
                    #
                    # The child ran its own cleanup before emitting this, so the parent must not
                    # repeat it by pathname: a slot this run overwrote has been handed back to
                    # the bundle it displaced, and that one belongs to another run.
                    self._child_cleared_own = True
                    s["resume_blocked_reason"] = (
                        "This run was stopped without saving, so it was discarded."
                    )
            elif etype == "error":
                # Reset in_model_load too: an error during model loading has no model_load_completed.
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
