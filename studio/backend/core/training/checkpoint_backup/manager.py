"""Non-blocking checkpoint upload scheduling and retry policy."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Optional, Protocol

from .config import CheckpointBackupConfig
from .progress import BackupProgressStore


class BackupTransport(Protocol):
    def upload_checkpoint(self, run_id: str, checkpoint_path: Path, progress: Callable) -> None: ...


@dataclass(frozen = True)
class _Upload:
    run_id: str
    step: int
    path: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointBackupManager:
    """One worker per manager; latest pending upload coalesces per run."""

    def __init__(self, config: CheckpointBackupConfig, output_root: Path,
                 transport: BackupTransport, *, checkpoint_validator: Callable[[Path], bool],
                 max_attempts: int = 3, backoff_seconds: float = 1.0) -> None:
        self.config = config
        self.output_root = output_root.resolve()
        self.transport = transport
        self.checkpoint_validator = checkpoint_validator
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self.progress = BackupProgressStore()
        self._pending: dict[str, _Upload] = {}
        self._active: dict[str, Path] = {}
        self._lock = Lock()
        self._wake = Event()
        self._cancel = Event()
        self._thread = Thread(target = self._worker, daemon = True, name = "checkpoint-backup")
        self._thread.start()

    def on_checkpoint_saved(self, run_id: str, global_step: int, checkpoint_path: Path) -> bool:
        if not self.config.enabled or global_step % self.config.interval_steps:
            return False
        path = Path(checkpoint_path).resolve()
        try:
            path.relative_to(self.output_root)
        except ValueError:
            return False
        if path.name.endswith((".part", ".tmp")) or not self.checkpoint_validator(path):
            return False
        item = _Upload(run_id, global_step, path)
        with self._lock:
            self._pending[run_id] = item
        self.progress.update(run_id, status = "queued", repo_id = self.config.repo_id,
                             checkpoint_name = path.name, checkpoint_step = global_step,
                             queued_at = _now(), max_attempts = self.max_attempts)
        self._wake.set()
        return True

    def is_pinned(self, checkpoint_path: Path) -> bool:
        resolved = Path(checkpoint_path).resolve()
        with self._lock:
            return resolved in self._active.values()

    def cancel(self) -> None:
        self._cancel.set()
        self._wake.set()

    def shutdown(self, timeout: float = 10.0) -> bool:
        self._cancel.set()
        self._wake.set()
        self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive()

    def _worker(self) -> None:
        while not self._cancel.is_set():
            self._wake.wait(0.25)
            self._wake.clear()
            with self._lock:
                item = next(iter(self._pending.values()), None)
                if item:
                    self._pending.pop(item.run_id, None)
                    self._active[item.run_id] = item.path
            if item:
                self._upload(item)
                with self._lock:
                    self._active.pop(item.run_id, None)
                    if self._pending:
                        self._wake.set()
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            self.progress.update(item.run_id, status = "cancelled", completed_at = _now())

    def _upload(self, item: _Upload) -> None:
        started = monotonic()
        self.progress.update(item.run_id, status = "preparing", started_at = _now())
        for attempt in range(1, self.max_attempts + 1):
            if self._cancel.is_set():
                self.progress.update(item.run_id, status = "cancelled", completed_at = _now())
                return
            self.progress.update(item.run_id, status = "uploading", attempt = attempt)

            def report(files_done: int, files_total: int, sent: int, total: int) -> None:
                elapsed = max(monotonic() - started, 0.001)
                speed = sent / elapsed
                self.progress.update(
                    item.run_id, status = "uploading", files_completed = files_done,
                    files_total = files_total, bytes_uploaded = sent, bytes_total = total,
                    progress_percent = sent / total * 100 if total else None,
                    upload_speed_bytes_per_second = speed,
                    eta_seconds = (total - sent) / speed if total and speed else None,
                )
            try:
                self.transport.upload_checkpoint(item.run_id, item.path, report)
            except Exception as exc:
                if attempt == self.max_attempts:
                    self.progress.update(item.run_id, status = "failed", completed_at = _now(),
                                         error_code = "upload_failed", error_message = str(exc))
                    return
                self.progress.update(item.run_id, status = "retrying", error_code = "upload_failed",
                                     error_message = str(exc))
                self._cancel.wait(min(self.backoff_seconds * (2 ** (attempt - 1)), 30.0))
            else:
                self.progress.update(item.run_id, status = "success", completed_at = _now(),
                                     last_successful_checkpoint = item.path.name,
                                     last_successful_step = item.step, error_code = None,
                                     error_message = None)
                return
