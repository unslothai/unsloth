from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import multiprocessing as mp

from .parse import apply_update, coerce_event, parse_log_message
from .types import Job
from .worker import run_job_process


_CTX = mp.get_context("spawn")


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
        self._next_id += 1
        body = json.dumps(event, separators=(",", ":"), ensure_ascii=False)
        event_type = event.get("type") or "message"
        return (
            f"id: {self._next_id}\n"
            f"event: {event_type}\n"
            f"data: {body}\n\n"
        ).encode("utf-8")


class JobManager:
    def __init__(self) -> None:
        """Single-job runner (in-mem). Simple on purpose, not a whole platform."""
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._proc: mp.Process | None = None
        self._mp_q: Any | None = None
        self._events: deque[dict] = deque(maxlen=5000)
        self._subs: list[queue.Queue] = []
        self._pump_thread: threading.Thread | None = None

    def start(self, *, recipe: dict, run: dict) -> str:
        """Spawn the job subprocess (one at a time, no cap)."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                raise RuntimeError("job already running")

            job_id = uuid.uuid4().hex
            self._job = Job(job_id=job_id, status="pending", started_at=time.time())
            self._events.clear()

            mp_q = _CTX.Queue()
            proc = _CTX.Process(
                target=run_job_process,
                kwargs={"event_queue": mp_q, "recipe": recipe, "run": run},
                daemon=True,
            )
            proc.start()

            self._mp_q = mp_q
            self._proc = proc
            self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
            self._pump_thread.start()

            self._emit({"type": "job.enqueued", "ts": time.time(), "job_id": job_id})
            return job_id

    def cancel(self, job_id: str) -> bool:
        """Hard stop. We terminate the subprocess. Quick + reliable."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return False
            if self._proc is None or not self._proc.is_alive():
                return True
            self._job.status = "cancelling"
            self._emit({"type": "job.cancelling", "ts": time.time(), "job_id": job_id})
            try:
                self._proc.terminate()
            except Exception:
                pass
            return True

    def get_status(self, job_id: str) -> dict | None:
        """UI-friendly snapshot. Poll this if you don't want SSE."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            job = self._job
            return {
                "job_id": job.job_id,
                "status": job.status,
                "stage": job.stage,
                "current_column": job.current_column,
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

    def subscribe(self, job_id: str) -> Subscription | None:
        """SSE subscribe: get replay buffer + live events stream."""
        with self._lock:
            if self._job is None or self._job.job_id != job_id:
                return None
            q: queue.Queue = queue.Queue(maxsize=2000)
            self._subs.append(q)
            return Subscription(replay=list(self._events), _q=q)

    def unsubscribe(self, sub: Subscription) -> None:
        """Drop SSE subscriber (client disconnected)."""
        with self._lock:
            self._subs = [q for q in self._subs if q is not sub._q]

    def _emit(self, event: dict) -> None:
        """Broadcast event to replay buffer + all subscribers."""
        self._events.append(event)
        stale: list[queue.Queue] = []
        for q in self._subs:
            try:
                q.put_nowait(event)
            except Exception:
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
            return coerce_event(q.get(timeout=timeout_sec))
        except queue.Empty:
            return None
        except Exception:
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
            except Exception:
                return events

    def _pump_loop(self) -> None:
        """Background thread: consumes worker events + updates job snapshot."""
        while True:
            snap = self._snapshot()
            if snap is None:
                return
            job, proc, mp_q = snap

            event = self._read_queue_with_timeout(mp_q, timeout_sec=0.25)
            if event is not None:
                self._handle_event(job, event)
                continue

            if proc.is_alive():
                continue

            for e in self._drain_queue(mp_q):
                self._handle_event(job, e)

            with self._lock:
                if self._job and self._job.status in {"pending", "active", "cancelling"}:
                    if self._job.status == "cancelling":
                        self._job.status = "cancelled"
                    else:
                        self._job.status = "error"
                        self._job.error = self._job.error or "process exited"
                    self._job.finished_at = time.time()
                    self._emit(
                        {
                            "type": f"job.{self._job.status}",
                            "ts": time.time(),
                            "job_id": self._job.job_id,
                        }
                    )
            return

    def _handle_event(self, job: Job, event: dict) -> None:
        """Apply event -> job state + forward to SSE."""
        et = event.get("type")
        msg = event.get("message") if et == "log" else None

        with self._lock:
            if self._job is None or self._job.job_id != job.job_id:
                return
            if et == "job.started":
                self._job.status = "active"
            if et == "job.completed":
                self._job.status = "completed"
                self._job.finished_at = time.time()
                self._job.analysis = event.get("analysis")
                self._job.artifact_path = event.get("artifact_path")
            if et == "job.error":
                self._job.status = "error"
                self._job.finished_at = time.time()
                self._job.error = event.get("error") or "error"

            if msg:
                upd = parse_log_message(msg)
                if upd:
                    apply_update(self._job, upd)

        self._emit(event)


_JOB_MANAGER: JobManager | None = None


def get_job_manager() -> JobManager:
    """Singleton JobManager (we only run 1 job anyway)."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER

