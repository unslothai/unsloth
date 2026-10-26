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

from core.inference.inference_gate import pauses_background_embedding
from core.training.account_jobs import (
    account_is_retired,
    init_job_owner,
    job_busy,
    job_control,
    job_pump,
    job_read,
    owned_job,
    validate_job_paths,
    worker_alive,
)
from utils.account_context import account_thread
from utils.gpu_memory_events import invalidates_gpu_memory as _invalidates_gpu_memory
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
    """NaN and Infinity are invalid in strict JSON, so non-finite values become None here."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _run_diffusion_child(*, event_queue: Any, stop_queue: Any, config: dict) -> None:
    # Fresh spawned interpreter: re-apply network injections inside the secret scrub and before the trainer imports diffusers.
    from utils.native_tls import activate_native_tls
    from utils.happy_eyeballs import activate_happy_eyeballs

    activate_native_tls()
    activate_happy_eyeballs()

    # Imported lazily so this module (and the route layer) stays torch-free at import.
    from .diffusion_lora_trainer import run_diffusion_training_process

    # Skips LogConfig.setup_logging, so this installs the Hub default; diffusers setup is in the trainer.
    try:
        from loggers.config import quiet_third_party_progress_bars
        quiet_third_party_progress_bars()
    except Exception:  # noqa: BLE001 - never let log tidying stop a training run
        pass

    run_diffusion_training_process(event_queue = event_queue, stop_queue = stop_queue, config = config)


def _default_target(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
    unsloth_stderr_mirror_path: Optional[str] = None,
) -> None:
    # First in the child, before torch: bind to parent death and scrub the native path secret; token policy precedes the returning account branch.
    if unsloth_stderr_mirror_path:
        # Before the token setup below, so its failure is captured too.
        try:
            from utils.worker_stderr import install_worker_stderr_mirror
            install_worker_stderr_mirror(unsloth_stderr_mirror_path)
        except Exception:
            pass
    if not config.get("allow_ambient", True):
        # Before any huggingface_hub import: the child env inherits the parent's token.
        import os

        from hub.utils.hf_tokens import apply_token_to_child_env, hf_token_arg, is_anonymous

        hf_token = hf_token_arg(config.get("hf_token"), allow_ambient_token = False)
        apply_token_to_child_env(os.environ, hf_token)
        if is_anonymous(hf_token):
            os.environ["HF_TOKEN_PATH"] = os.devnull

    account = config.pop("_job_account", None)
    if account is not None:
        from core.training.account_jobs import run_account_child
        from utils.native_path_leases import run_without_native_path_secret

        run_without_native_path_secret(
            run_account_child,
            account = account,
            job_module = "core.training.diffusion_training_service",
            job_target = "_run_diffusion_child",
            event_queue = event_queue,
            stop_queue = stop_queue,
            config = config,
        )
        return
    from utils.native_path_leases import run_without_native_path_secret

    run_without_native_path_secret(
        _run_diffusion_child,
        event_queue = event_queue,
        stop_queue = stop_queue,
        config = config,
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
def _runs_dir() -> Path:
    from utils.paths.storage_roots import tensorboard_root

    d = tensorboard_root() / "diffusion"
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
    """Derived from disk; write_error is sticky and blocks resume; started_at and ended_at fence off bundles from other runs. Never raises."""
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
    """Backfills output_dir for old records, else can_resume would enable a resume the client refuses."""
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
            # Without total_steps, the manifest's older target would block Resume.
            total_steps = _resolved_total_steps(rec, config if isinstance(config, dict) else {}),
            write_error = rec.get("checkpoint_write_error"),
            # Source bundle this run resumed from, so a run that died before its first save can still offer it.
            source_checkpoint = (
                config.get("resume_from_checkpoint") if isinstance(config, dict) else None
            ),
            # Identity of that source bundle; a slot another run has rewritten is not offered as lineage.
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
        # Skip wrong-shape records: one bad file must not break DiffusionTrainingRunSummary(**r) or panel.
        if not isinstance(rec, dict):
            continue
        if not (isinstance(rec.get("job_id"), str) and isinstance(rec.get("status"), str)):
            continue
        # Resume state read from disk first; per record, since one nonnumeric total_steps broke the endpoint.
        try:
            _refresh_resume_state(rec)
            _restate_live_job(rec)
        except Exception:  # noqa: BLE001 -- a record we cannot refresh is still a record
            continue
        rec.pop("metric_history", None)
        rec.pop("config", None)
        rec.pop("samples", None)
        out.append(rec)
    return out


def get_diffusion_run(job_id: str) -> Optional[dict]:
    """The full persisted record for one run (summary + config + metric logs)."""
    # Reject non-uuid4-hex ids so a crafted id cannot traverse out of the dir.
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
    """A live job reports running, not the interim interrupted state, so Resume is not offered."""
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
        "preparation": None,
        "output_dir": None,
        "lora_path": None,
        # The optional second (EMA-averaged) adapter, when ema_decay was enabled.
        "ema_path": None,
        "catalog_path": None,
        "family": None,
        "base_model": None,
        "samples_per_second": None,
        "peak_memory_gb": None,
        # Newest checkpoint-<N> path and step; why it failed to save (Resume tooltip); resumed-from step.
        "checkpoint_path": None,
        "checkpoint_step": None,
        "resume_blocked_reason": None,
        "resumed_from_step": None,
        # A path is not identity: another run can overwrite checkpoint-<N>, so the fallback lineage is wrong.
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
        "samples": [],
    }


# Past this the oldest half after the step-0 baseline is thinned, like the metrics.
_SAMPLES_CAP = 400


def _append_samples(
    state: dict[str, Any],
    step: Any,
    images: Any,
    output_dir: Optional[str] = None,
) -> None:
    """Fold one ``sample`` event; thinned rounds' images are deleted so every file on disk stays listed."""
    from core.training.diffusion_samples import SAMPLE_PATH_RE, delete_sample_files

    try:
        istep = int(step)
    except (TypeError, ValueError):
        return
    entries = state.setdefault("samples", [])
    for img in images if isinstance(images, list) else []:
        if not isinstance(img, dict) or not SAMPLE_PATH_RE.match(str(img.get("path") or "")):
            continue
        entries.append(
            {
                "step": istep,
                "path": str(img["path"]),
                "prompt": str(img.get("prompt") or ""),
                "seed": img.get("seed") if isinstance(img.get("seed"), int) else None,
            }
        )
    if len(entries) > _SAMPLES_CAP:
        steps = sorted({e["step"] for e in entries})
        keep = set(steps[:1] + steps[1:-1][::2] + steps[-1:])
        state["samples"] = [e for e in entries if e["step"] in keep]
        delete_sample_files(output_dir, [e["path"] for e in entries if e["step"] not in keep])


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
    """Only finite, positive-step points are kept; at the cap, every other point is dropped."""
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
        # setdefault: older records may lack newer series.
        state.setdefault(key, []).append(values[key])


def _resolved_total_steps(state: dict[str, Any], cfg: dict[str, Any]) -> int:
    """In epoch mode train_steps is a stale request default, so return 0 and let the manifest answer."""
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
        init_job_owner(
            self,
            lambda: worker_alive(self, pump = "_pump") or self.is_active(),
            lambda: self.stop(save = False),
            self._clear_account_result,
        )
        self._ctx = ctx if ctx is not None else _CTX
        self._target = target if target is not None else _default_target
        self._stderr_capture = None
        self._lock = threading.Lock()
        # Set by reserve() while a start is in flight (before the route frees GPU models) so the load guards
        # refuse a concurrent load. Cleared by unreserve().
        self._reserved = False
        # Two-sided under _lock: starts and dataset mutations exclude each other.
        self._dataset_mutations = 0
        # A child killed after a no-save stop never reports it; remember it here.
        self._discard_requested = False
        # Bundle paths this job wrote; a discard the child cannot finish removes only those, not older ones.
        self._own_checkpoints: list[str] = []
        # The child already cleaned up; a parent pathname cleanup could delete another run's bundle.
        self._child_cleared_own = False
        # Set by a checkpoint_saved event, cleared once the pump has written the record.
        self._persist_interim = False
        # Only the first stop reaches the child, so only the first may set the parent's disposition.
        self._stop_signalled = False
        # GPU load admissions in flight; same two-sided rule as dataset mutations.
        self._gpu_admissions = 0
        self._proc: Any = None
        self._stop_queue: Any = None
        self._pump: Optional[threading.Thread] = None
        self._state: dict[str, Any] = _idle_state()
        # The active job's start config, scrubbed of secrets, kept for the run record.
        self._config: dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _clear_account_result(self):
        self._state = _idle_state()
        self._config = {}

    def is_active(self) -> bool:
        with self._lock:
            if self._reserved:
                return True
            return self._proc is not None and self._proc.is_alive()

    @owned_job()
    def reserve(self) -> None:
        """Mark a diffusion-training start as in flight so the image/video load guards (which read
        is_active) refuse a concurrent load BEFORE the route frees resident GPU models. Without this
        the training becomes active only at start(), after the free, so an overlapping load passes
        its guard, acquires the GPU, and both workloads allocate VRAM.

        Compare-and-set: raise if a start is already reserved or a job is already running, so a
        second overlapping /diffusion/start is rejected (409) BEFORE it frees GPU residents, instead
        of both requests tearing down residents and racing to start(). Paired with unreserve() in a
        finally by the reserving caller, so a failed start never leaves training active.
        """
        with self._lock:
            if self._reserved or (self._proc is not None and self._proc.is_alive()):
                raise RuntimeError("A diffusion training job is already running.")
            if self._dataset_mutations:
                raise DatasetMutationInFlight(
                    "The training images are being changed right now. Wait for that to finish, "
                    "then start the run."
                )
            if self._gpu_admissions:
                # Refuse: reserving now would free residents this load has not registered.
                raise RuntimeError(
                    "A model is being loaded onto the GPU right now. Wait for that to finish, "
                    "then start the run."
                )
            # Checked under the same lock as the LLM trainer so one of the two always raises.
            if _llm_training_active():
                raise RuntimeError(
                    "An LLM training job is already running. "
                    "Stop it before starting diffusion (Images) training."
                )
            self._reserved = True

    @job_control
    def unreserve(self) -> None:
        """Clear the reservation set by reserve(). Only touches the reservation flag, never
        _proc, so a live job stays active on success and a failed start is fully rolled back."""
        with self._lock:
            self._reserved = False

    @contextlib.contextmanager
    def dataset_mutation(self):
        """Registers under reserve()'s lock, closing the gap between an is_active() check and the
        mutation."""
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
        """Held only until the load registers, because a start preempts an in-flight load from then on."""
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

    @_invalidates_gpu_memory("diffusion training start")
    @owned_job()
    @pauses_background_embedding
    def start(self, config: dict) -> str:
        """Validate ``config``, spawn the trainer, and start pumping its events.

        Raises ValueError for an unusable config (before any spawn) and RuntimeError if a
        job is already running. Returns the new job id."""
        validate_job_paths(config)
        # Validate before spawning; keep the normalised config, whose resolved family keys recipe overrides.
        from .diffusion_lora_trainer import _config_from_dict
        from .diffusion_train_common import train_recipe_overrides

        normalized_cfg = _config_from_dict(config).normalized()

        # Join outside the lock: the pump's final writes take it.
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
            if self.job_account is not None:
                config = {**config, "_job_account": self.job_account}
            child_kwargs = {
                "event_queue": event_queue,
                "stop_queue": self._stop_queue,
                "config": config,
            }
            # Test targets take only the three job kwargs.
            if self._target is _default_target:
                self._open_worker_stderr_capture()
                if self._stderr_capture is not None:
                    from utils.native_path_leases import STDERR_MIRROR_KWARG
                    child_kwargs[STDERR_MIRROR_KWARG] = self._stderr_capture.path
            self._proc = self._ctx.Process(
                target = self._target,
                kwargs = child_kwargs,
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
            # After the reset: records the source identity for a resume that dies during load.
            self._seed_source_identity(config)
            # Record the recipe the family loop actually runs.
            self._config = {
                k: v for k, v in dict(config).items() if k not in {"hf_token", "_job_account"}
            }
            # A resume's unset bucketing stays unset; the trainer adopts the bundle's.
            if normalized_cfg.bucketing is not None:
                self._config["bucketing"] = normalized_cfg.bucketing
            self._config.update(train_recipe_overrides(normalized_cfg))
            self._pump = account_thread(
                target = self._pump_loop, args = (event_queue, self._proc), daemon = True
            )
            self._pump.start()
            return job_id

    @_invalidates_gpu_memory("diffusion training stop")
    @job_control
    def stop(self, save: bool = True) -> bool:
        """Request a clean stop: the trainer finishes the current step, then either saves
        a partial adapter (``save=True``, the default) or discards the run (``save=False``,
        matching the LLM trainer's cancel). Returns True if a stop was signalled, False if
        nothing was running."""
        with self._lock:
            if self._proc is None or not self._proc.is_alive() or self._stop_queue is None:
                return False
            if self._stop_signalled:
                # Child consumes only the FIRST signal; a later stop-without-saving must not set a parent discard, which would delete checkpoints the child kept.
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

    def stop_for_shutdown(self, timeout: float) -> bool:
        from utils.account_context import run_as

        with self._lock:
            proc = self._proc
            pump = self._pump
            account = self._result_account

        def settled():
            # The pump writes the run record after the child exits, so wait for it too.
            return (proc is None or not proc.is_alive()) and (pump is None or not pump.is_alive())

        if settled():
            return True
        if proc is not None and proc.is_alive():
            # The signal path runs as the owner, which job_control refuses for a managed account's run.
            run_as(account, self.stop, save = True)
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if settled():
                return True
            time.sleep(0.25)
        return False

    @job_read(
        lambda self: {
            **_idle_state(),
            "status": "busy" if job_busy(self) else "idle",
            "message": "Busy" if job_busy(self) else "",
        }
    )
    def status(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._state)
            # Keep ``active`` honest even if the process died between events.
            snap["active"] = self._proc is not None and self._proc.is_alive()
            return snap

    def sample_source(self, job_id: str) -> Optional[tuple[Optional[str], list]]:
        """(output_dir, samples) for the live job ``job_id``, or None when it is not the live job or
        belongs to another account (the run record answers for a finished one)."""
        from core.training.account_jobs import job_is_foreign

        if job_is_foreign(self):
            return None
        with self._lock:
            if not job_id or self._state.get("job_id") != job_id:
                return None
            out = self._state.get("output_dir") or self._config.get("output_dir")
            return (str(out) if out else None), list(self._state.get("samples") or [])

    def _open_worker_stderr_capture(self) -> None:
        previous = self._stderr_capture
        self._stderr_capture = None
        if previous is not None:
            try:
                previous.close()
            except Exception:
                pass
        try:
            from utils.worker_stderr import WorkerStderrCapture
            self._stderr_capture = WorkerStderrCapture(prefix = "unsloth-diffusion-worker-")
        except Exception:
            self._stderr_capture = None

    def _unexpected_exit_message(self, proc: Any) -> str:
        from utils.worker_stderr import unexpected_exit_message

        text = ""
        capture = self._stderr_capture
        if capture is not None:
            try:
                text = capture.text()
            except Exception:
                text = ""
        return unexpected_exit_message(
            getattr(proc, "pid", None),
            getattr(proc, "exitcode", None),
            text,
        )

    # ── event pump ───────────────────────────────────────────────────────────
    @job_pump
    def _pump_loop(self, event_queue: Any, proc: Any) -> None:
        while True:
            try:
                ev = event_queue.get(timeout = 1.0)
            except Exception:  # noqa: BLE001 -- Empty (timeout) or a closed queue
                if not proc.is_alive():
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
                                # A worker killed mid-load / mid-preparation sends no terminal event to clear these.
                                in_model_load = False,
                                preparation = None,
                                message = self._unexpected_exit_message(proc),
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
        """Keep bundles when the child already ran cleanup: its restore put back a predecessor's bundle."""
        if account_is_retired():
            delete = False
        with self._lock:
            own = list(self._own_checkpoints) if delete else []
            samples = [e.get("path") for e in self._state.get("samples") or []] if delete else []
            sample_root = self._state.get("output_dir") or self._config.get("output_dir")
            self._state["resume_blocked_reason"] = (
                "This run was stopped without saving, so it was discarded."
            )
            self._state["checkpoint_path"] = None
            self._state["checkpoint_step"] = None
            # Their files are deleted, so listing them would serve 404s.
            self._state["samples"] = []
            self._own_checkpoints = []
        if samples:
            try:
                from core.training.diffusion_samples import discard_sample_paths
                discard_sample_paths(sample_root, samples)
            except Exception:  # noqa: BLE001 -- cleanup must never break the terminal transition
                pass
        if not own:
            return
        try:
            from core.training.diffusion_checkpoint import discard_named_checkpoints
            discard_named_checkpoints(own)
        except Exception:  # noqa: BLE001 -- cleanup must never break the terminal transition
            pass

    def _persist_run_record(self, *, interim: bool = False) -> None:
        """Best-effort and never fatal. Interim mode records the run as interrupted so a checkpoint survives if Unsloth itself dies."""
        if account_is_retired():
            return
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
                # output_dir is recorded for Resume; can_resume and checkpoint_step re-derive from disk each time.
                "output_dir": str(adapter) if adapter else None,
                "resumed_from_job_id": cfg.get("resumed_from_job_id") or None,
                "resumed_from_step": s.get("resumed_from_step"),
                "resumed_source_created_at": s.get("resumed_source_created_at"),
                "checkpoint_write_error": s.get("resume_blocked_reason") or None,
                **_resume_fields(
                    str(adapter) if adapter else None,
                    status = s.get("status"),
                    started_at = s.get("started_at"),
                    # Use the resolved total; epoch mode leaves train_steps at the default 500.
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
                "samples": s.get("samples") or [],
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
        """Events from a superseded job's process are ignored, so a stale pump cannot touch current
        state."""
        etype = ev.get("type")
        with self._lock:
            if proc is not None and self._proc is not proc:
                return
            s = self._state
            s["updated_at"] = time.time()
            if etype == "model_load_started":
                s.update(
                    in_model_load = True,
                    preparation = None,
                    status = "running",
                    message = "Loading base model...",
                )
                if ev.get("num_images") is not None:
                    s["num_images"] = ev.get("num_images")
            elif etype == "model_load_completed":
                s.update(in_model_load = False, preparation = None, message = "Training...")
            elif etype == "preparing":
                # Long precompute such as the VAE latent cache, surfaced so the UI does not look stalled.
                done, total = ev.get("done"), ev.get("total")
                stage = str(ev.get("stage", "prepare")).replace("_", " ")
                s.update(
                    status = "running",
                    in_model_load = True,
                    preparation = {"stage": ev.get("stage"), "done": done, "total": total},
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
                # Seed live counters from the resume event, else they read 0/0 until the first post-resume progress.
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
                # Persist now: only terminal events write the run JSON.
                self._persist_interim = True
            elif etype == "checkpoint_failed":
                # Sticky: resuming from an older bundle would silently lose steps.
                s["resume_blocked_reason"] = str(ev.get("message", "checkpoint write failed"))
                # Persisted: in memory only, an exit left the record advertising an older checkpoint as resumable.
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
                # Per-modality losses can reveal degradation the combined loss hides (H3).
                video_loss = (
                    _finite_or_none(ev["video_loss"]) if "video_loss" in ev else s["video_loss"]
                )
                audio_loss = (
                    _finite_or_none(ev["audio_loss"]) if "audio_loss" in ev else s["audio_loss"]
                )
                s.update(
                    status = "running",
                    step = ev.get("step", s["step"]),
                    preparation = None,
                    in_model_load = False,
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
            elif etype == "sample":
                _append_samples(
                    s,
                    ev.get("step"),
                    ev.get("images"),
                    s.get("output_dir") or self._config.get("output_dir"),
                )
            elif etype == "complete":
                # Reset in_model_load: a stop during load skips model_load_completed and would leave it stale.
                s.update(
                    active = False,
                    in_model_load = False,
                    status = "stopped" if ev.get("stopped") else "completed",
                    preparation = None,
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
                    # The child already cleaned up; never repeat by pathname, since a slot this run overwrote now holds a bundle from another run.
                    self._child_cleared_own = True
                    s["resume_blocked_reason"] = (
                        "This run was stopped without saving, so it was discarded."
                    )
                    s["samples"] = []
            elif etype == "error":
                # Reset in_model_load too: an error during model loading has no model_load_completed.
                s.update(
                    active = False,
                    in_model_load = False,
                    status = "error",
                    preparation = None,
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
