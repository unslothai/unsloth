# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Any

import multiprocessing as mp

from ..jsonable import to_preview_jsonable
from .constants import (
    EVENT_JOB_CANCELLING,
    EVENT_JOB_CANCELLED,
    EVENT_JOB_COMPLETED,
    EVENT_JOB_ENQUEUED,
    EVENT_JOB_ERROR,
    EVENT_JOB_STARTED,
)
from .parse import apply_update, coerce_event, parse_log_message
from .types import Job
from .worker import run_job_process


_CTX = mp.get_context("spawn")


def _github_source_estimated_total(recipe: dict) -> int | None:
    seed_config = recipe.get("seed_config")
    if not isinstance(seed_config, dict):
        return None
    source = seed_config.get("source")
    if not isinstance(source, dict) or source.get("seed_type") != "github_repo":
        return None

    repos_raw = source.get("repos")
    repos = (
        [repo for repo in repos_raw if isinstance(repo, str) and repo.strip()]
        if isinstance(repos_raw, list)
        else []
    )
    item_types_raw = source.get("item_types")
    item_types = (
        [
            item
            for item in item_types_raw
            if isinstance(item, str) and item in {"issues", "pulls", "commits"}
        ]
        if isinstance(item_types_raw, list)
        else []
    )
    try:
        limit = int(source.get("limit") or 0)
    except (TypeError, ValueError):
        return None
    if not repos or not item_types or limit <= 0:
        return None
    return len(repos) * len(item_types) * limit


def _source_progress_status(job: Job) -> dict[str, Any] | None:
    progress = job.source_progress
    if progress is None:
        return None
    return {
        "source": progress.source,
        "status": progress.status,
        "repo": progress.repo,
        "resource": progress.resource,
        "page": progress.page,
        "page_items": progress.page_items,
        "fetched_items": progress.fetched_items,
        "estimated_total": progress.estimated_total,
        "percent": progress.percent,
        "rate_remaining": progress.rate_remaining,
        "retry_after_sec": progress.retry_after_sec,
        "message": progress.message,
        "updated_at": progress.updated_at,
    }


@dataclass
class Subscription:
    replay: list[dict]
    _q: queue.Queue
    _next_id: int = 0

    async def next_event(self, *, timeout_sec: float) -> dict | None:
        """Wait for next event (SSE), w/ timeout so we can check disconnects."""
        try:
            return await asyncio.to_thread(self._q.get, True, timeout_sec)
        except queue.Empty:
            return None

    def format_sse(self, event: dict) -> bytes:
        """Turn event dict into SSE bytes (id/event/data)."""
        event_id = event.get("seq")
        if event_id is None:
            self._next_id += 1
            event_id = self._next_id
        body = json.dumps(event, separators = (",", ":"), ensure_ascii = False)
        event_type = event.get("type") or "message"
        return (
            f"id: {event_id}\n" f"event: {event_type}\n" f"data: {body}\n\n"
        ).encode("utf-8")


class JobManager:
    def __init__(self) -> None:
        """Single-job runner (in-mem). Simple on purpose, not a whole platform."""
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._proc: mp.Process | None = None
        self._mp_q: Any | None = None
        self._events: deque[dict] = deque(maxlen = 5000)
        self._subs: list[queue.Queue] = []
        self._pump_thread: threading.Thread | None = None
        self._seq: int = 0

    def start(
        self,
        *,
        recipe: dict,
        run: dict,
        internal_api_key_id: int | None = None,
    ) -> str:
        """Spawn the job subprocess (one at a time, no cap).

        ``internal_api_key_id`` is the row id of a workflow-scoped
        sk-unsloth-* key minted by the route layer for local providers.
        JobManager revokes it when the job reaches a terminal state so the
        key's live window is no longer than the run.
        """
        llm_columns = recipe.get("columns") or []
        llm_column_count = 0
        if isinstance(llm_columns, list):
            for column in llm_columns:
                if not isinstance(column, dict):
                    continue
                column_type = str(column.get("column_type") or "").strip().lower()
                if column_type.startswith("llm"):
                    llm_column_count += 1
        if llm_column_count <= 0:
            llm_column_count = 1

        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                raise RuntimeError("job already running")

            job_id = uuid.uuid4().hex
            self._job = Job(job_id = job_id, status = "pending", started_at = time.time())
            self._job.progress_columns_total = llm_column_count
            self._job.source_progress_estimated_total = _github_source_estimated_total(
                recipe
            )
            self._job.internal_api_key_id = internal_api_key_id
            self._events.clear()
            self._seq = 0

            run_payload = dict(run)
            run_payload["_job_id"] = job_id
            mp_q = _CTX.Queue()
            proc = _CTX.Process(
                target = run_job_process,
                kwargs = {"event_queue": mp_q, "recipe": recipe, "run": run_payload},
                daemon = True,
            )
            proc.start()

            self._mp_q = mp_q
            self._proc = proc
            self._pump_thread = threading.Thread(target = self._pump_loop, daemon = True)
            self._pump_thread.start()

            self._emit(
                {"type": EVENT_JOB_ENQUEUED, "ts": time.time(), "job_id": job_id}
            )
            return job_id

    def cancel(self, job_id: str) -> bool:
        """Hard stop. We terminate the subprocess. Quick + reliable."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return False
            if self._proc is None or not self._proc.is_alive():
                return True
            self._job.status = "cancelling"
            self._emit(
                {"type": EVENT_JOB_CANCELLING, "ts": time.time(), "job_id": job_id}
            )
            try:
                self._proc.terminate()
            except (AttributeError, OSError):
                pass
            return True

    def get_status(self, job_id: str) -> dict | None:
        """UI friendly snapshot that we need. Alternative to sse kinda of and structured"""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            job = self._job
            return {
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.stage,
                "current_column": job.current_column,
                "completed_columns": list(job.completed_columns),
                "batch": {"idx": job.batch.idx, "total": job.batch.total},
                "progress": {
                    "done": job.progress.done,
                    "total": job.progress.total,
                    "percent": job.progress.percent,
                    "eta_sec": job.progress.eta_sec,
                    "rate": job.progress.rate,
                    "ok": job.progress.ok,
                    "failed": job.progress.failed,
                },
                "column_progress": {
                    "done": job.column_progress.done,
                    "total": job.column_progress.total,
                    "percent": job.column_progress.percent,
                    "eta_sec": job.column_progress.eta_sec,
                    "rate": job.column_progress.rate,
                    "ok": job.column_progress.ok,
                    "failed": job.column_progress.failed,
                },
                "source_progress": _source_progress_status(job),
                "model_usage": {
                    name: {
                        "model": usage.model,
                        "tokens": {
                            "input": usage.input_tokens,
                            "output": usage.output_tokens,
                            "total": usage.total_tokens,
                            "tps": usage.tps,
                        },
                        "requests": {
                            "success": usage.requests_success,
                            "failed": usage.requests_failed,
                            "total": usage.requests_total,
                            "rpm": usage.rpm,
                        },
                    }
                    for name, usage in job.model_usage.items()
                },
                "rows": job.rows,
                "cols": job.cols,
                "error": job.error,
                "has_analysis": job.analysis is not None,
                "dataset_rows": None if job.dataset is None else len(job.dataset),
                "artifact_path": job.artifact_path,
                "execution_type": job.execution_type,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            }

    def get_current_status(self) -> dict | None:
        """Single-job convenience (last/current)."""
        job_id = self.get_current_job_id()
        if job_id is None:
            return None
        return self.get_status(job_id)

    def get_current_job_id(self) -> str | None:
        """Return current job_id (or None)."""
        with self._lock:
            return None if self._job is None else self._job.job_id

    def get_analysis(self, job_id: str) -> dict | None:
        """Final profiling output (only after job completes)."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            return self._job.analysis

    def get_dataset(
        self,
        job_id: str,
        *,
        limit: int,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        """Load dataset page (offset + limit) and include total rows."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            in_memory_dataset = self._job.dataset
            artifact_path = self._job.artifact_path
            job_status = self._job.status

        if in_memory_dataset is not None:
            total = len(in_memory_dataset)
            rows = in_memory_dataset[offset : offset + limit]
            return {"dataset": rows, "total": total}
        if not artifact_path:
            if job_status in {"completed", "error", "cancelled"}:
                return {"error": "artifact path missing"}
            return None

        try:
            base_dataset_path = Path(artifact_path)
            parquet_dir = base_dataset_path / "parquet-files"
            if not parquet_dir.exists():
                return {"error": f"dataset path missing: {parquet_dir}"}

            return self._load_dataset_page(
                parquet_dir = parquet_dir, limit = limit, offset = offset
            )
        except Exception as exc:
            return {"error": f"dataset load failed: {exc}"}

    @staticmethod
    def _load_dataset_page(
        *,
        parquet_dir: Path,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        dataset_page = JobManager._load_dataset_page_with_duckdb(
            parquet_dir = parquet_dir,
            limit = limit,
            offset = offset,
        )
        if dataset_page is not None:
            return dataset_page
        return JobManager._load_dataset_page_with_data_designer(
            parquet_dir = parquet_dir,
            limit = limit,
            offset = offset,
        )

    @staticmethod
    def _load_dataset_page_with_duckdb(
        *,
        parquet_dir: Path,
        limit: int,
        offset: int,
    ) -> dict[str, Any] | None:
        parquet_glob = str((parquet_dir / "*.parquet").resolve())
        try:
            import duckdb  # type: ignore
        except Exception:
            return None

        try:
            conn = duckdb.connect(":memory:")
            try:
                total_row = conn.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [parquet_glob],
                ).fetchone()
                total = int(total_row[0] if total_row else 0)
                dataframe = conn.execute(
                    (
                        "SELECT *, row_number() OVER (PARTITION BY filename) AS __row_num__ "
                        "FROM read_parquet(?, filename=true) "
                        "ORDER BY filename, __row_num__ "
                        "LIMIT ? OFFSET ?"
                    ),
                    [parquet_glob, int(limit), int(offset)],
                ).fetchdf()
            finally:
                conn.close()
        except (RuntimeError, ValueError, duckdb.Error):
            return None

        for helper_col in ("filename", "__row_num__"):
            if helper_col in dataframe.columns:
                dataframe = dataframe.drop(columns = [helper_col])

        rows = dataframe.to_dict(orient = "records")
        return {"dataset": to_preview_jsonable(rows), "total": total}

    @staticmethod
    def _load_dataset_page_with_data_designer(
        *,
        parquet_dir: Path,
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        from data_designer.config.utils.io_helpers import read_parquet_dataset

        dataframe = read_parquet_dataset(parquet_dir)
        total = int(len(dataframe.index))
        rows = dataframe.iloc[offset : offset + limit].to_dict(orient = "records")
        return {"dataset": to_preview_jsonable(rows), "total": total}

    def subscribe(
        self, job_id: str, *, after_seq: int | None = None
    ) -> Subscription | None:
        """SSE subscribe: get replay buffer + live events stream."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            q: queue.Queue = queue.Queue(maxsize = 2000)
            self._subs.append(q)
            if after_seq is None:
                replay = list(self._events)
            else:
                replay = [e for e in self._events if int(e.get("seq") or 0) > after_seq]
            return Subscription(replay = replay, _q = q)

    def unsubscribe(self, sub: Subscription) -> None:
        """Drop SSE subscriber (client disconnected)."""
        with self._lock:
            self._subs = [q for q in self._subs if q is not sub._q]

    def _emit(self, event: dict) -> None:
        """Broadcast event to replay buffer + all subscribers."""
        self._seq += 1
        event["seq"] = self._seq
        self._events.append(event)
        stale: list[queue.Queue] = []
        for q in self._subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                stale.append(q)
        if stale:
            self._subs = [q for q in self._subs if q not in stale]

    def _snapshot(self) -> tuple[Job, mp.Process, Any] | None:
        """Grab pointers for the pump loop (avoid holding lock too long)."""
        with self._lock:
            if self._job is None or self._proc is None or self._mp_q is None:
                return None
            return self._job, self._proc, self._mp_q

    @staticmethod
    def _read_queue_with_timeout(q: Any, *, timeout_sec: float) -> dict | None:
        """Try read 1 event from mp queue. Timeout = pump stays responsive."""
        try:
            return coerce_event(q.get(timeout = timeout_sec))
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError):
            return None

    @staticmethod
    def _drain_queue(q: Any) -> list[dict]:
        """Drain mp queue fast (used on process exit)."""
        events: list[dict] = []
        while True:
            try:
                events.append(coerce_event(q.get_nowait()))
            except queue.Empty:
                return events
            except (EOFError, OSError, ValueError):
                return events

    def _pump_loop(self) -> None:
        """Background thread: consumes worker events + updates job snapshot."""
        while True:
            snap = self._snapshot()
            if snap is None:
                return
            job, proc, mp_q = snap

            event = self._read_queue_with_timeout(mp_q, timeout_sec = 0.25)
            if event is not None:
                self._handle_event(job, event)
                continue

            if proc.is_alive():
                continue

            for e in self._drain_queue(mp_q):
                self._handle_event(job, e)

            retired_job: Job | None = None
            with self._lock:
                if self._job and self._job.status in {
                    "pending",
                    "active",
                    "cancelling",
                }:
                    if self._job.status == "cancelling":
                        self._job.status = "cancelled"
                    else:
                        self._job.status = "error"
                        self._job.error = self._job.error or "process exited"
                    self._job.finished_at = time.time()
                    event_type = (
                        EVENT_JOB_CANCELLED
                        if self._job.status == "cancelled"
                        else EVENT_JOB_ERROR
                    )
                    self._emit(
                        {
                            "type": event_type,
                            "ts": time.time(),
                            "job_id": self._job.job_id,
                        }
                    )
                    retired_job = self._job
            if retired_job is not None:
                self._retire_workflow_key(retired_job)
            return

    def _handle_event(self, job: Job, event: dict) -> None:
        """Apply event -> job state + forward to SSE."""
        et = event.get("type")
        msg = event.get("message") if et == "log" else None

        terminal = False
        with self._lock:
            if self._job is None or self._job.job_id != job.job_id:
                return
            if et == EVENT_JOB_STARTED:
                self._job.status = "active"
            if et == EVENT_JOB_COMPLETED:
                self._job.status = "completed"
                self._job.finished_at = time.time()
                self._job.analysis = event.get("analysis")
                self._job.artifact_path = event.get("artifact_path")
                self._job.execution_type = event.get("execution_type")
                self._job.dataset = event.get("dataset")
                self._job.processor_artifacts = event.get("processor_artifacts")
                if self._job.progress.total and self._job.progress.total > 0:
                    self._job.progress.done = self._job.progress.total
                    self._job.progress.percent = 100.0
                terminal = True
            if et == EVENT_JOB_ERROR:
                self._job.status = "error"
                self._job.finished_at = time.time()
                self._job.error = event.get("error") or "error"
                terminal = True
            if et == EVENT_JOB_CANCELLED:
                terminal = True

            if msg:
                upd = parse_log_message(msg)
                if upd:
                    apply_update(self._job, upd)

        if terminal:
            self._retire_workflow_key(job)

        self._emit(event)

    def _retire_workflow_key(self, job: Job) -> None:
        """Revoke the workflow-scoped sk-unsloth-* key, if one was minted.

        Best-effort: revocation failures are swallowed. The key would
        expire on its own after 24h, so a missed revoke is a latency
        concern, not a correctness one.
        """
        key_id = getattr(job, "internal_api_key_id", None)
        if not key_id:
            return
        try:
            from auth import storage  # deferred: avoids circular import

            storage.revoke_internal_api_key(int(key_id))
        except Exception:
            pass
        job.internal_api_key_id = None


_JOB_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    """Singleton JobManager (we only run 1 job anyway)."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER
