# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A small job service for diffusion LoRA training.

This is deliberately separate from the LLM ``TrainingBackend``: that backend's
lifecycle (LLM config build, per-run SQLite rows, and matplotlib plots) is specific
to text training and would mis-handle a diffusion run. This service does only what a
diffusion job needs -- spawn the trainer subprocess, pump its
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
from utils.paths.path_utils import drop_appledouble_metadata

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
    # Fresh spawned interpreter: re-apply the OS-trust-store injection, inside the secret scrub and
    # before the trainer imports diffusers.
    from utils.native_tls import activate_native_tls

    activate_native_tls()

    # Imported lazily so this module (and the route layer) stays torch-free at import.
    from .diffusion_lora_trainer import run_diffusion_training_process

    # This child never runs LogConfig.setup_logging, so it installs the Hub default here; the diffusers
    # half is done inside the trainer entrypoints.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a training run
        pass

    run_diffusion_training_process(event_queue = event_queue, stop_queue = stop_queue, config = config)


def _default_target(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    # First thing in the child (before torch): self-bind to parent death and scrub the native path
    # secret, like the other workers.
    from utils.native_path_leases import run_without_native_path_secret
    run_without_native_path_secret(
        _run_diffusion_child, event_queue = event_queue, stop_queue = stop_queue, config = config
    )


# Cap on retained metric points; over it, arrays are decimated so a long run stays bounded while the
# chart keeps its shape.
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


# One JSON file per terminal run, not the LLM sqlite, so diffusion runs stay off the LLM Runs page.
# ── persisted run history ──────────────────────────────────────────────────────
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
        # The EXACT bundle this run would continue, so the client resumes the one it was shown rather than
        # whatever is newest in a folder two runs share.
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
            # A run that died during model loading never got a "resumed" event to seed total_steps, and zero
            # here sends the calculation back to the manifest's OLDER target, so a 600-step checkpoint read as
            # 600/500 and Resume was disabled.
            total_steps = _resolved_total_steps(rec, config if isinstance(config, dict) else {}),
            write_error = rec.get("checkpoint_write_error"),
            # What this run itself resumed from, so a resumed run that died before its first save can still
            # offer the bundle it was validated against.
            source_checkpoint = (
                config.get("resume_from_checkpoint") if isinstance(config, dict) else None
            ),
            # And WHICH bundle that was, so a slot another run has since rewritten is not offered back as this
            # run's own lineage.
            source_created_at = rec.get("resumed_source_created_at"),
        )
    )
    return rec


def list_diffusion_runs(limit: int = 20) -> list[dict]:
    """Summaries of persisted diffusion runs, newest first. The heavy per-run payload
    (metric logs, config) stays in the file; fetch it via ``get_diffusion_run``."""
    try:
        # Sliced by limit below, so a companion per run would halve the history window.
        files = drop_appledouble_metadata(
            sorted(_runs_dir().glob("*.json"), key = lambda p: p.stat().st_mtime, reverse = True)
        )
    except Exception:  # noqa: BLE001 -- unreadable dir -> no history
        return []
    out: list[dict] = []
    for p in files[: max(0, int(limit))]:
        try:
            rec = json.loads(p.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001 -- a corrupt record never breaks the listing
            continue
        # Skip a wrong-shape record so one bad file cannot blow up the route's
        # DiffusionTrainingRunSummary(**r) or the whole panel.
        if not isinstance(rec, dict):
            continue
        if not (isinstance(rec.get("job_id"), str) and isinstance(rec.get("status"), str)):
            continue
        # Resume state comes from the checkpoints on disk NOW, before the config is dropped. Per record,
        # because one hand-edited nonnumeric total_steps otherwise took the whole endpoint down.
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

    An interim record is written as interrupted because that is the outcome if Unsloth never
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
        # _UNRESUMABLE_STATUS rejects a running job because its output directory is being written right now,
        # so a Resume there can only race the live run and 409.
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
        # The last per-modality losses of a joint run; None on families that train one modality.
        "video_loss": None,
        "audio_loss": None,
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
        # Where the newest checkpoint-<N> bundle is, the step it holds, why one could not be written (the
        # Resume action's disabled tooltip), and the step a resumed run picked up from.
        "checkpoint_path": None,
        "checkpoint_step": None,
        "resume_blocked_reason": None,
        "resumed_from_step": None,
        # A pathname is not an identity: another run can write over the same checkpoint-<N> slot, and the
        # fallback would offer that replacement back under this run's lineage.
        "resumed_source_created_at": None,
        "started_at": None,
        "updated_at": None,
        # Bounded, paired history arrays for the live loss chart (see _append_metric).
        "metric_steps": [],
        "metric_loss": [],
        "metric_lr": [],
        "metric_grad_norm": [],
        # Null on every other family, and on H3 steps that reported only the combined loss, so the series
        # stay index-aligned.
        "metric_video_loss": [],
        "metric_audio_loss": [],
    }


# The value series, in append order, paired index-for-index with metric_steps. One list so a new
# series cannot be added to the appends and forgotten in the decimation below, which would
# silently misalign the curves.
_METRIC_SERIES: tuple[str, ...] = (
    "metric_loss",
    "metric_lr",
    "metric_grad_norm",
    "metric_video_loss",
    "metric_audio_loss",
)


def _append_metric(
    state: dict[str, Any],
    step: Any,
    loss: Any,
    lr: Any,
    grad_norm: Any = None,
    video_loss: Any = None,
    audio_loss: Any = None,
) -> None:
    """Append one (step, loss, lr, grad_norm, video_loss, audio_loss) point to the bounded
    history arrays on ``state``.

    Only records finite, positive-step points (mirrors the LLM trainer, which logs history
    only for step > 0 with a real loss). When the arrays hit ``_METRIC_CAP`` they are
    decimated in place (keep every other point) so appends stay bounded without losing the
    curve's shape. Everything but loss may be None (kept as None so those series can be sparse
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
    # Non-finite is nulled rather than dropped, to stay index-aligned.
    values = {
        "metric_loss": floss,
        "metric_lr": _finite_or_none(lr),
        "metric_grad_norm": _finite_or_none(grad_norm),
        "metric_video_loss": _finite_or_none(video_loss),
        "metric_audio_loss": _finite_or_none(audio_loss),
    }
    steps = state["metric_steps"]
    if len(steps) >= _METRIC_CAP:
        state["metric_steps"] = steps[::2]
        for key in _METRIC_SERIES:
            state[key] = (state.get(key) or [])[::2]
        steps = state["metric_steps"]
    steps.append(istep)
    for key in _METRIC_SERIES:
        # setdefault, not [key]: a run resumed from a record written before a series existed carries a state
        # dict without it.
        state.setdefault(key, []).append(values[key])


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
        # Set by reserve() while a start is in flight (before the route frees GPU models) so the load guards
        # refuse a concurrent load. Cleared by unreserve().
        self._reserved = False
        # Dataset mutations in flight. A start refuses while any is open and a mutation refuses once a
        # start is reserved, both under _lock, so neither slips through the other's window.
        self._dataset_mutations = 0
        # The trainer reports a no-save stop on its completion event, but a child killed or OOMed after
        # the request never emits one, and the unexpected-exit path then re-offered Resume from the
        # discarded checkpoint.
        self._discard_requested = False
        # Bundle paths THIS job reported writing, so a discard the child could not carry out removes exactly
        # those and nothing that predated the run.
        self._own_checkpoints: list[str] = []
        # The child emits a discarded completion only after its own clear_own_checkpoints, which hands a
        # displaced slot back, so the parent's pathname-only cleanup would delete another run's bundle.
        self._child_cleared_own = False
        # Set by a checkpoint_saved event, cleared once the pump has written the record.
        self._persist_interim = False
        # Only the first stop reaches the child, so only the first may set the parent's disposition.
        self._stop_signalled = False
        # GPU load admissions in flight (a load between its training guard and its arbiter registration).
        # Same two-sided rule as the dataset mutations.
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
                # A load past its training guard is about to take the GPU, and reserving now would free residents
                # it has not registered yet; refusing is safe, since admission is held only across registration.
                raise RuntimeError(
                    "A model is being loaded onto the GPU right now. Wait for that to finish, "
                    "then start the run."
                )
            # Under the SAME lock as the LLM trainer, not just at the route's earlier check: several
            # network-bound preflights separate the two, and the LLM route holds gpu_load_admission() across
            # its spawn, so one of the two always raises.
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
        # Validate before spawning, and keep the normalised config: it carries the resolved family the
        # recipe overrides are keyed on, which the raw request dict need not name.
        from .diffusion_lora_trainer import _config_from_dict
        from .diffusion_train_common import train_recipe_overrides

        normalized_cfg = _config_from_dict(config).normalized()

        # Join a finished job's pump OUTSIDE the lock: its final state writes take this lock, so joining
        # under it would stall the start and let the stale pump overwrite the new state.
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
            # Keep the lease secret out of the child's env, as other orchestrators do.
            from utils.native_path_leases import native_path_secret_removed_for_child_start

            with native_path_secret_removed_for_child_start():
                self._proc.start()
            try:
                from utils.process_lifetime import adopt_pid
                adopt_pid(self._proc.pid)
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
            # AFTER the reset above: the route has already pinned a source bundle, so recording its identity
            # here means a resume that dies during the model load still has its timestamp to check against.
            self._seed_source_identity(config)
            # Record the config with the fields this family's loop REPLACES set to what it will actually run:
            # the trainer applies the same table in the child, so without this Previous runs described a
            # recipe no step ever used.
            self._config = {k: v for k, v in dict(config).items() if k != "hf_token"}
            self._config.update(train_recipe_overrides(normalized_cfg))
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
                # The child consumes the FIRST signal, so honouring a later stop-without-saving set a parent
                # discard the child never carried out: the run was marked discarded and its checkpoints deleted
                # while the adapter and catalog entry stayed on disk.
                return True
            try:
                # Bare True = older wire format; the dict form carries the no-save cancel flag.
                self._stop_queue.put(True if save else {"save": False})
            except Exception:  # noqa: BLE001
                return False
            self._stop_signalled = True
            if not save:
                # Remembered here so a child that dies before its discarded completion still blocks the resume the
                # user threw away.
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
                # An exception raised on the current step is a terminal error and returns here rather than through
                # the dead-process branch, so the discard has to be applied on this path too.
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
        checkpoint survive Unsloth itself dying: the bundle is on disk but only a terminal event
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
                s["message"] = "Unsloth exited while this run was training."
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
                "video_loss": s.get("video_loss"),
                "audio_loss": s.get("audio_loss"),
                "samples_per_second": s.get("samples_per_second"),
                "peak_memory_gb": s.get("peak_memory_gb"),
                "num_images": s.get("num_images"),
                "started_at": s.get("started_at"),
                "ended_at": s.get("updated_at"),
                "lora_path": s.get("lora_path"),
                "ema_path": s.get("ema_path"),
                "catalog_path": s.get("catalog_path"),
                "saved": bool(s.get("lora_path")),
                # output_dir is what a Resume replays, so it is recorded on the run; can_resume / checkpoint_step
                # are re-derived from disk, so deleting the checkpoints takes the action away.
                "output_dir": str(adapter) if adapter else None,
                "resumed_from_job_id": cfg.get("resumed_from_job_id") or None,
                "resumed_from_step": s.get("resumed_from_step"),
                "resumed_source_created_at": s.get("resumed_source_created_at"),
                "checkpoint_write_error": s.get("resume_blocked_reason") or None,
                **_resume_fields(
                    str(adapter) if adapter else None,
                    status = s.get("status"),
                    started_at = s.get("started_at"),
                    # total_steps is seeded by the "resumed" event and a run that dies during model loading never gets
                    # one; zero there sent the resume calculation back to the manifest's older target, reporting
                    # "nothing left to train" for a run whose point was a raised target.
                    # NOT in epoch mode: num_epochs overrides train_steps, which keeps its Pydantic default of 500, so a
                    # run resolved to 1000 reported a 600-step checkpoint as 600/500 and refused the resume.
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
                    "video_loss": s.get("metric_video_loss") or [],
                    "audio_loss": s.get("metric_audio_loss") or [],
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
                # A long precompute phase (e.g. the VAE latent cache) before the first step; surfaced so the UI
                # shows progress instead of a silent stall.
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
                # This is the step resumed FROM, not a checkpoint this run wrote, so it does not touch checkpoint_step.
                s.update(
                    resumed_from_step = ev.get("step"),
                    resumed_source_created_at = ev.get("source_created_at"),
                    message = f"Resuming from step {ev.get('step')}...",
                )
                # Seed the LIVE counters from the same event: until the first post-resume progress they read 0/0, so
                # a resume of step 400 of 500 showed "0/0" and an OOM recorded a failed run at step 0 of 0.
                if ev.get("step") is not None:
                    s["step"] = int(ev["step"])
                if ev.get("total_steps") is not None:
                    s["total_steps"] = int(ev["total_steps"])
            elif etype == "checkpoint_saved":
                # Folded as each bundle lands, not only at the end, so a run that later crashes is still reported as
                # resumable; a good write also clears an earlier failure.
                s.update(
                    checkpoint_path = ev.get("checkpoint_path"),
                    checkpoint_step = ev.get("step"),
                    resume_blocked_reason = None,
                )
                written = ev.get("checkpoint_path")
                if written and written not in self._own_checkpoints:
                    self._own_checkpoints.append(str(written))
                # Only a terminal event writes the run JSON, so being killed after a periodic save left a resumable
                # checkpoint Previous runs had no entry for.
                self._persist_interim = True
            elif etype == "checkpoint_failed":
                # Sticky: an older bundle on disk predates the work this run did, so resuming from it would
                # silently lose steps. Mirrors the MLX resume_blocked flag.
                s["resume_blocked_reason"] = str(ev.get("message", "checkpoint write failed"))
                # Persisted because, in memory only, an exit after this left the last record advertising the stale
                # older checkpoint as resumable.
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
                # Folded like the rest rather than dropped: on MiniMax-H3 the combined loss can hold steady while
                # one modality degrades, and these are the only signal that says so.
                video_loss = (
                    _finite_or_none(ev["video_loss"]) if "video_loss" in ev else s["video_loss"]
                )
                audio_loss = (
                    _finite_or_none(ev["audio_loss"]) if "audio_loss" in ev else s["audio_loss"]
                )
                s.update(
                    status = "running",
                    step = ev.get("step", s["step"]),
                    total_steps = ev.get("total_steps", s["total_steps"]),
                    loss = loss,
                    avg_loss = avg_loss,
                    learning_rate = learning_rate,
                    grad_norm = grad_norm,
                    video_loss = video_loss,
                    audio_loss = audio_loss,
                    message = "Training...",
                )
                # Fold optional perf fields so the UI shows throughput + peak VRAM.
                if ev.get("samples_per_second") is not None:
                    s["samples_per_second"] = ev.get("samples_per_second")
                if ev.get("peak_memory_gb") is not None:
                    s["peak_memory_gb"] = ev.get("peak_memory_gb")
                # Retain a bounded per-step history for the live charts.
                _append_metric(
                    s,
                    ev.get("step"),
                    ev.get("loss"),
                    ev.get("learning_rate"),
                    ev.get("grad_norm"),
                    ev.get("video_loss"),
                    ev.get("audio_loss"),
                )
            elif etype == "complete":
                # Reset in_model_load: a stop during model load emits complete with no preceding
                # model_load_completed, leaving a stale indicator.
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
                # Only the lineage is re-confirmed here, for a pump that missed the earlier "resumed" event.
                if ev.get("resumed_from_step") is not None:
                    s["resumed_from_step"] = ev.get("resumed_from_step")
                if ev.get("discarded"):
                    # A discarded run's own periodic checkpoints must not keep offering to continue it.
                    # The child ran its own cleanup before emitting this, so the parent must not repeat it by pathname:
                    # a slot this run overwrote has been handed back to another run's bundle.
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
