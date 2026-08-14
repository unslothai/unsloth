"""Thread-safe snapshots for checkpoint backup progress."""

from dataclasses import asdict, dataclass, replace
from threading import RLock
from typing import Literal, Optional


BackupStatus = Literal[
    "disabled", "idle", "queued", "preparing", "uploading", "retrying",
    "success", "failed", "cancelled", "authentication_required",
]
TERMINAL_STATUSES = frozenset({"success", "failed", "cancelled", "authentication_required"})


@dataclass(frozen = True)
class BackupProgress:
    """Current state; terminal means the current attempt ended, not that a run ended."""

    status: BackupStatus = "disabled"
    repo_id: Optional[str] = None
    checkpoint_name: Optional[str] = None
    checkpoint_step: Optional[int] = None
    files_completed: int = 0
    files_total: Optional[int] = None
    bytes_uploaded: int = 0
    bytes_total: Optional[int] = None
    progress_percent: Optional[float] = None
    upload_speed_bytes_per_second: Optional[float] = None
    eta_seconds: Optional[float] = None
    attempt: int = 0
    max_attempts: int = 3
    queued_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_successful_checkpoint: Optional[str] = None
    last_successful_step: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class BackupProgressStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, BackupProgress] = {}

    def snapshot(self, run_id: str) -> BackupProgress:
        with self._lock:
            return self._runs.get(run_id, BackupProgress())

    def update(self, run_id: str, **changes) -> BackupProgress:
        with self._lock:
            current = self._runs.get(run_id, BackupProgress())
            updated = replace(current, **changes)
            self._runs[run_id] = updated
            return updated
