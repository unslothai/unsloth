# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training job queue: sequential, unattended execution of queued runs."""

import json
import threading
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import ValidationError

from loggers import get_logger
from models import TrainingStartRequest
from core.training.launch import (
    generate_job_id,
    launch_training,
    validate_training_request,
)
from storage import studio_db

logger = get_logger(__name__)

POLL_INTERVAL = 5.0
SETTLE_DELAY = 3.0
MAX_PENDING = 5
MAX_START_DEFERRALS = 3
RESTART_SKIP_REASON = "Server restarted while this job was running"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_request_json(request_json: str) -> str:
    try:
        data = json.loads(request_json)
    except (json.JSONDecodeError, TypeError):
        return "{}"
    if not isinstance(data, dict):
        return "{}"
    for key in ("hf_token", "wandb_token"):
        if data.get(key):
            data[key] = None
    s3_config = data.get("s3_config")
    if isinstance(s3_config, dict):
        data["s3_config"] = {
            "bucket": s3_config.get("bucket"),
            "region": s3_config.get("region"),
            "prefix": s3_config.get("prefix"),
            "use_iam_role": bool(s3_config.get("use_iam_role")),
        }
    return json.dumps(data)


def _dataset_summary(request: TrainingStartRequest) -> str:
    if request.hf_dataset:
        return request.hf_dataset
    if request.local_datasets:
        first = request.local_datasets[0].replace("\\", "/").rstrip("/").split("/")[-1]
        extra = len(request.local_datasets) - 1
        return f"{first} (+{extra})" if extra > 0 else first
    if request.s3_config is not None:
        return f"S3: {request.s3_config.bucket}"
    return "unknown"


class TrainingQueueManager:
    def __init__(self, backend = None):
        self._backend = backend
        self._wake = threading.Event()
        self._stop_runner = threading.Event()
        self._runner_thread: Optional[threading.Thread] = None
        self._runner_lock = threading.Lock()
        # Consecutive no-error start failures per item id (transient cleanup
        # races); in-memory only, a restart resets the count.
        self._start_deferrals: Dict[str, int] = {}
        # Overridable for tests.
        self.poll_interval = POLL_INTERVAL
        self.settle_delay = SETTLE_DELAY
        self.max_pending = MAX_PENDING
        self.max_start_deferrals = MAX_START_DEFERRALS

    def _get_backend(self):
        if self._backend is None:
            from core.training import get_training_backend
            self._backend = get_training_backend()
        return self._backend

    def _queue_full_error(self) -> HTTPException:
        return HTTPException(
            status_code = 409,
            detail = f"Queue is full ({self.max_pending} pending jobs). "
            "Remove an item or wait for a job to start.",
        )

    def enqueue(
        self,
        request: TrainingStartRequest,
        subject: str,
        via_api_key: bool = False,
    ) -> dict:
        # Fast fail before validation; the insert re-checks the cap atomically.
        if studio_db.count_pending_queue_items() >= self.max_pending:
            raise self._queue_full_error()

        # validate_training_request mutates the request (resume_from_checkpoint:
        # run dir -> concrete checkpoint path). Persist the pre-validation
        # payload so launch-time validation resolves the original resume target.
        request_json = request.model_dump_json()
        validate_training_request(request)

        item = studio_db.enqueue_queue_item(
            id = f"qitem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:8]}",
            request_json = request_json,
            model_name = request.model_name,
            dataset_summary = _dataset_summary(request),
            subject = subject,
            via_api_key = via_api_key,
            max_pending = self.max_pending,
        )
        if item is None:
            raise self._queue_full_error()
        logger.info("Enqueued training job %s (%s)", item["id"], request.model_name)
        # A successful enqueue proves the DB is usable, so revive the runner if
        # startup restore failed transiently (lifespan logs and continues);
        # without a waiter the wake below is a no-op and accepted work would
        # sit pending until the next restart. start_runner() is idempotent.
        self.start_runner()
        self._wake.set()
        return item

    def remove(self, item_id: str) -> bool:
        return studio_db.delete_queue_item_if_pending(item_id)

    def move(self, item_id: str, direction: str) -> bool:
        return studio_db.move_queue_item(item_id, direction)

    def pause(self, reason: str = "user") -> None:
        studio_db.set_queue_paused(True, reason)
        logger.info("Training queue paused (%s)", reason)

    def resume(self) -> None:
        studio_db.set_queue_paused(False)
        logger.info("Training queue resumed")
        # Same revival as enqueue: un-pausing accepts queued work, so the
        # runner must exist for the wake to mean anything.
        self.start_runner()
        self._wake.set()

    def state(self) -> Dict[str, Any]:
        paused, paused_reason = studio_db.get_queue_paused()
        backend = self._get_backend()
        active_job_id = backend.current_job_id if backend.is_training_active() else None
        return {
            "paused": paused,
            "paused_reason": paused_reason,
            "pending_count": studio_db.count_pending_queue_items(),
            "max_pending": self.max_pending,
            "active_job_id": active_job_id,
            "items": studio_db.list_queue_items() + studio_db.list_finished_queue_items(limit = 10),
        }

    def restore_on_startup(self) -> None:
        # Preserve a terminal run that finished before the queue runner had a
        # chance to reconcile its row. Only rows with no terminal run record
        # were interrupted by the restart.
        skipped = 0
        recovered = 0
        for item in studio_db.list_queue_items(statuses = ("starting", "running")):
            job_id = item.get("job_id")
            run = studio_db.get_run(job_id) if job_id else None
            result_status = (run or {}).get("status")
            if result_status and result_status != "running":
                studio_db.update_queue_item_status(
                    item["id"],
                    "done",
                    result_status = result_status,
                    error_message = (run or {}).get("error_message"),
                    finished_at = _now(),
                    request_json = _redact_request_json(item["request_json"]),
                )
                recovered += 1
                continue

            # Terminal transition: redact request_json like every other one
            # so orphaned queue items cannot retain credentials in studio.db.
            studio_db.update_queue_item_status(
                item["id"],
                "skipped",
                error_message = RESTART_SKIP_REASON,
                finished_at = _now(),
                request_json = _redact_request_json(item["request_json"]),
            )
            skipped += 1
        if skipped:
            logger.info("Skipped %d queue item(s) orphaned by restart", skipped)
        if recovered:
            logger.info("Recovered %d completed queue item(s) at startup", recovered)
        if studio_db.count_pending_queue_items() > 0:
            studio_db.set_queue_paused(True, "restart")
            logger.info("Training queue has pending items after restart; starting paused")
        else:
            studio_db.set_queue_paused(False)
        self.start_runner()

    def start_runner(self) -> None:
        backend = self._get_backend()
        backend.on_job_finished = self._wake.set
        with self._runner_lock:
            if self._runner_thread is not None and self._runner_thread.is_alive():
                return
            self._stop_runner.clear()
            self._runner_thread = threading.Thread(
                target = self._run_loop, daemon = True, name = "training-queue-runner"
            )
            self._runner_thread.start()
        logger.info("Training queue runner started")

    def stop_runner(self) -> None:
        self._stop_runner.set()
        self._wake.set()
        with self._runner_lock:
            runner = self._runner_thread
        if runner is not None and runner is not threading.current_thread():
            runner.join(timeout = 1.0)

    def _run_loop(self) -> None:
        while not self._stop_runner.is_set():
            self._wake.wait(self.poll_interval)
            self._wake.clear()
            if self._stop_runner.is_set():
                return
            try:
                self._tick()
            except Exception:
                logger.warning("Training queue tick failed", exc_info = True)

    def _tick(self) -> None:
        backend = self._get_backend()

        self._reconcile_running_items(backend)

        paused, _reason = studio_db.get_queue_paused()
        if paused:
            return
        if backend.is_training_active():
            return

        item = studio_db.next_pending_queue_item()
        if item is None:
            return

        # A "complete" event precedes full subprocess exit; re-check after a
        # short delay so we never spawn against a worker that is still dying.
        if self.settle_delay > 0:
            if self._stop_runner.wait(self.settle_delay):
                return
            # A pause request during the delay must win over the launch.
            paused, _reason = studio_db.get_queue_paused()
            if paused:
                return
            # The head may have changed while we slept (reorder/remove).
            item = studio_db.next_pending_queue_item()
            if item is None:
                return
        if backend.is_training_active():
            return
        if self._api_item_must_wait(item):
            return
        if self._install_must_wait(item):
            return
        if self._stop_runner.is_set():
            return

        self._launch_item(backend, item)

    def _api_item_must_wait(self, item: dict) -> bool:
        # POST /start refuses API-key starts while an inference request is in
        # flight (the training VRAM hook may unload the chat model mid-stream).
        # Items queued over the API keep that guarantee at launch time too;
        # the item stays pending and the next tick retries. UI-queued items
        # keep the UI semantics (coexist with chat or free VRAM).
        if not item.get("via_api_key"):
            return False
        try:
            from core.inference.llama_keepwarm import other_inference_request_count
            in_flight = other_inference_request_count(current_request_counted = False)
        except Exception:
            return False
        if in_flight > 0:
            logger.debug(
                "Queue item %s waiting: %d inference request(s) in flight",
                item["id"],
                in_flight,
            )
            return True
        return False

    def _install_must_wait(self, item: dict) -> bool:
        # POST /start refuses to spawn while a consented transformers install
        # stage-and-swaps the sidecar (a worker spawned mid-swap could activate
        # a half-replaced venv). Same guard here: the item stays pending and
        # the next tick retries, without consuming start deferrals.
        try:
            from utils.transformers_latest import is_install_in_progress
            if not is_install_in_progress():
                return False
        except Exception:
            return False
        logger.debug("Queue item %s waiting: transformers install in progress", item["id"])
        return True

    def _reconcile_running_items(self, backend) -> None:
        for item in studio_db.list_queue_items(statuses = ("starting", "running")):
            job_id = item.get("job_id")
            if not job_id:
                # 'starting' without a job_id: launch crashed mid-transition.
                if item["status"] == "starting":
                    studio_db.update_queue_item_status(
                        item["id"],
                        "skipped",
                        error_message = "Launch was interrupted before the job started",
                        finished_at = _now(),
                        request_json = _redact_request_json(item["request_json"]),
                    )
                continue
            if backend.is_training_active() and backend.current_job_id == job_id:
                continue
            run = studio_db.get_run(job_id)
            result_status = (run or {}).get("status") or "error"
            if result_status == "running":
                # Pump hasn't finalized the run row yet; next poll catches it.
                continue
            studio_db.update_queue_item_status(
                item["id"],
                "done",
                result_status = result_status,
                error_message = (run or {}).get("error_message"),
                finished_at = _now(),
                request_json = _redact_request_json(item["request_json"]),
            )
            logger.info("Queue item %s finished: run %s -> %s", item["id"], job_id, result_status)

    def _launch_item(self, backend, item: dict) -> None:
        if self._stop_runner.is_set():
            return
        if not studio_db.update_queue_item_status(
            item["id"], "starting", expected_status = "pending"
        ):
            return

        if self._stop_runner.is_set():
            studio_db.update_queue_item_status(item["id"], "pending", expected_status = "starting")
            return

        def _skip(reason: str) -> None:
            self._start_deferrals.pop(item["id"], None)
            studio_db.update_queue_item_status(
                item["id"],
                "skipped",
                error_message = reason,
                finished_at = _now(),
                request_json = _redact_request_json(item["request_json"]),
            )
            logger.warning("Queue item %s skipped: %s", item["id"], reason)
            self._wake.set()

        try:
            request = TrainingStartRequest.model_validate_json(item["request_json"])
        except ValidationError as e:
            _skip(f"Stored request no longer parses: {e.error_count()} validation error(s)")
            return

        try:
            resume_output_dir = validate_training_request(request)
        except HTTPException as e:
            _skip(str(e.detail))
            return
        except ValueError as e:
            _skip(str(e))
            return

        from utils.transformers_version import SidecarSwapInProgress

        job_id = generate_job_id()
        try:
            success = launch_training(
                job_id = job_id,
                request = request,
                resume_output_dir = resume_output_dir,
                subject = item.get("subject") or "queue",
                backend = backend,
            )
        except SidecarSwapInProgress as e:
            # Transient: a consented transformers install is mid-swap. Put the
            # item back for the next tick instead of skipping it permanently.
            if self._defer_start(item):
                return
            _skip(str(e))
            return
        except HTTPException as e:
            _skip(str(e.detail))
            return
        except ValueError as e:
            _skip(str(e))
            return
        except Exception as e:
            _skip(f"Failed to launch training: {e}")
            return

        if not success:
            if backend.is_training_active():
                # Lost a race with a manual /start; not the item's fault.
                studio_db.update_queue_item_status(
                    item["id"], "pending", expected_status = "starting"
                )
                logger.info("Queue item %s deferred: backend busy with a manual start", item["id"])
                return
            error = None
            try:
                error = backend.trainer.training_progress.error
            except Exception:
                pass
            if not error and self._defer_start(item):
                return
            _skip(error or "Training subprocess failed to start")
            return

        self._start_deferrals.pop(item["id"], None)
        studio_db.update_queue_item_status(
            item["id"],
            "running",
            expected_status = "starting",
            job_id = job_id,
            started_at = _now(),
        )
        logger.info("Queue item %s launched as run %s", item["id"], job_id)

    def _defer_start(self, item: dict) -> bool:
        # start_training can return False with the backend already inactive
        # while the previous pump thread is still finalizing (worker exited,
        # join timed out, run row being written). No progress error means the
        # request itself was never at fault, so put the item back at the head
        # for the next tick -- bounded, in case the backend is truly wedged.
        attempts = self._start_deferrals.get(item["id"], 0) + 1
        if attempts > self.max_start_deferrals:
            return False
        self._start_deferrals[item["id"]] = attempts
        studio_db.update_queue_item_status(item["id"], "pending", expected_status = "starting")
        logger.info(
            "Queue item %s deferred (%d/%d): backend still cleaning up",
            item["id"],
            attempts,
            self.max_start_deferrals,
        )
        return True


_queue_manager: Optional[TrainingQueueManager] = None


def get_training_queue_manager() -> TrainingQueueManager:
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = TrainingQueueManager()
    return _queue_manager
