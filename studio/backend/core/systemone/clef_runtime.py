# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torch-free parent runtime for one owned Clef worker."""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import queue as _queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .catalog import ClefCheckpoint
from .laya_runtime import Unavailable


logger = logging.getLogger(__name__)

# Safety bounds, not expected latency targets.
LOAD_WAIT_S = 300.0
RUN_WAIT_S = 300.0
CANCEL_GRACE_S = 5.0
SHUTDOWN_WAIT_S = 10.0
LOAD_CANCEL_WAIT_S = 12.0
IDLE_UNLOAD_S = 300.0
FAILURE_BACKOFF_S = 60.0
MAX_PENDING = 8
_POLL_S = 0.1

# Avoid inheriting server locks or CUDA state.
_CTX = mp.get_context("spawn")


class ClefWorkerError(RuntimeError):
    """The owned Clef child could not start, answer, or stay alive."""


class ClefWorkerCancelled(ClefWorkerError):
    """A load or decision was cancelled because its owner is retiring."""


class ClefWorkerInputError(ClefWorkerError):
    """The child rejected one request before inference."""


class ClefWorker:
    """Handle for one process this runtime spawned."""

    def __init__(self, target: Callable[..., None] | None = None) -> None:
        self._target = target
        self._process = None
        self._cmd_queue = None
        self._resp_queue = None
        self._cancel_event = None
        self._ready_event = None
        self._closed = False
        self._close_lock = threading.Lock()
        self.device: str | None = None
        self.gpu_available: bool | None = None

    def start(
        self,
        snapshot_path: Path,
        checkpoint: ClefCheckpoint,
        requested_device: str,
        cancelled: threading.Event,
    ) -> None:
        """Spawn then load a local snapshot, reaping every failed start."""
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import adopt_pid, is_process_shutting_down

        if cancelled.is_set() or is_process_shutting_down():
            raise ClefWorkerCancelled("Clef model loading was cancelled.")
        cache_env = get_hf_cache_paths().child_env({})
        try:
            with (
                child_environment_for_spawn(cache_env),
                native_path_secret_removed_for_child_start(),
            ):
                self._cmd_queue = _CTX.Queue()
                self._resp_queue = _CTX.Queue()
                self._cancel_event = _CTX.Event()
                self._ready_event = _CTX.Event()
                if self._target is None:
                    target = run_without_native_path_secret
                    args = ("core.systemone.clef_worker", "run_clef_worker", cache_env)
                else:
                    target = self._target
                    args = ()
                process = _CTX.Process(
                    target=target,
                    args=args,
                    kwargs={
                        "cmd_queue": self._cmd_queue,
                        "resp_queue": self._resp_queue,
                        "cancel_event": self._cancel_event,
                        "ready_event": self._ready_event,
                        "config": {},
                    },
                    daemon=True,
                )
                self._process = process
                process.start()
        except Exception as exc:
            self._process = None
            self._close_queues()
            raise ClefWorkerError(f"Could not start the Clef worker process: {exc}") from exc

        # Only this retained process handle is ever signalled.
        try:
            adopt_pid(process.pid)
        except Exception:
            logger.debug(
                "Could not register Clef worker %s for shutdown", process.pid, exc_info=True
            )
        if cancelled.is_set() or is_process_shutting_down():
            self.cancel()
            self.close(graceful_timeout=0.0)
            raise ClefWorkerCancelled("Clef model loading was cancelled.")

        try:
            self._send(
                {
                    "type": "load",
                    "snapshot_path": str(snapshot_path),
                    "model": checkpoint.name,
                    "requested_device": requested_device,
                }
            )
            response = self._await("loaded", LOAD_WAIT_S, cancelled, "load")
        except BaseException:
            self.close(graceful_timeout=0.0)
            raise
        self.device = str(response.get("device") or "cpu")
        self.gpu_available = response.get("gpu_available") is True

    def decide(
        self,
        checkpoint: ClefCheckpoint,
        state: Any,
        questions: dict[str, dict[str, Any]],
        images: list[bytes],
    ) -> dict[str, Any]:
        if self._closed:
            raise ClefWorkerCancelled("The Clef worker is unloading.")
        if self._cancel_event is not None:
            self._cancel_event.clear()
        self._send(
            {
                "type": "decide",
                "model": checkpoint.name,
                "state": state,
                "questions": questions,
                "images": images,
            }
        )
        response = self._await("result", RUN_WAIT_S, None, "decide")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ClefWorkerError("The Clef worker returned an invalid response.")
        return result

    def cancel(self) -> None:
        """Thread-safe signal; this never touches queues or an unowned process."""
        event = self._cancel_event
        if event is not None:
            try:
                event.set()
            except (OSError, ValueError):
                pass

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def close(self, graceful_timeout: float = SHUTDOWN_WAIT_S) -> bool:
        """Stop only this instance's process and release its queues after it is dead."""
        with self._close_lock:
            self._closed = True
            self.cancel()
            process = self._process
            if process is not None:
                try:
                    if graceful_timeout > 0 and process.is_alive() and self._cmd_queue is not None:
                        self._cmd_queue.put({"type": "shutdown"})
                except (OSError, ValueError):
                    pass
                if graceful_timeout > 0:
                    try:
                        process.join(graceful_timeout)
                    except Exception:
                        pass
                if process.is_alive():
                    try:
                        process.terminate()
                        process.join(CANCEL_GRACE_S)
                    except Exception:
                        pass
                if process.is_alive():
                    try:
                        process.kill()
                        process.join(CANCEL_GRACE_S)
                    except Exception:
                        pass
                if process.is_alive():
                    # Retain the only safe handle for later shutdown.
                    logger.error("Clef worker %s survived terminate and kill", process.pid)
                    return False
                try:
                    from utils.process_lifetime import forget_pid

                    forget_pid(process.pid)
                except Exception:
                    logger.debug("Could not forget Clef worker %s", process.pid, exc_info=True)
            self._process = None
            self._close_queues()
            return True

    def _send(self, command: dict[str, Any]) -> None:
        queue = self._cmd_queue
        if queue is None or self._closed:
            raise ClefWorkerCancelled("The Clef worker is not running.")
        try:
            queue.put(command)
        except (OSError, ValueError) as exc:
            raise ClefWorkerError(f"Could not reach the Clef worker: {exc}") from exc

    def _await(
        self,
        expected: str,
        timeout: float,
        cancelled: threading.Event | None,
        phase: str,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        cancel_deadline: float | None = None
        while True:
            if cancelled is not None and cancelled.is_set():
                self.cancel()
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + CANCEL_GRACE_S
                elif time.monotonic() >= cancel_deadline:
                    self.close(graceful_timeout=0.0)
                    raise ClefWorkerCancelled("Clef model loading was cancelled.")
            response_queue = self._resp_queue
            if response_queue is None:
                raise ClefWorkerCancelled("The Clef worker was unloaded.")
            try:
                response = response_queue.get(timeout=_POLL_S)
            except _queue.Empty:
                if not self.is_alive():
                    raise ClefWorkerError(self._crash_message(phase))
                if time.monotonic() >= deadline:
                    self.close(graceful_timeout=0.0)
                    raise ClefWorkerError(f"The Clef worker timed out while {phase}.")
                continue
            except (EOFError, OSError, ValueError) as exc:
                raise ClefWorkerError(f"Lost contact with the Clef worker: {exc}") from exc
            if not isinstance(response, dict):
                continue
            if response.get("type") == "error":
                kind = response.get("kind")
                message = str(response.get("message") or "The Clef worker failed.")
                if kind == "cancelled":
                    raise ClefWorkerCancelled(message)
                if kind == "invalid_request_error":
                    raise ClefWorkerInputError(message)
                raise ClefWorkerError(message)
            if response.get("type") == expected:
                return response

    def _crash_message(self, phase: str) -> str:
        process = self._process
        exitcode = getattr(process, "exitcode", None)
        return f"The Clef worker stopped while {phase} (exitcode={exitcode})."

    def _close_queues(self) -> None:
        for queue in (self._cmd_queue, self._resp_queue):
            try:
                if queue is not None:
                    queue.cancel_join_thread()
                    queue.close()
            except Exception:
                pass
        self._cmd_queue = None
        self._resp_queue = None


@dataclass
class _RepositoryLease:
    repo: str
    registry: Any
    owner: object


@dataclass
class _Load:
    checkpoint: ClefCheckpoint
    generation: int
    cancel: threading.Event
    done: threading.Event
    worker: ClefWorker | None = None
    thread: threading.Thread | None = None
    lease: _RepositoryLease | None = None


_state_lock = threading.RLock()
_run_lock = threading.Lock()
_admission = threading.BoundedSemaphore(MAX_PENDING)
_worker: ClefWorker | None = None
_loaded: ClefCheckpoint | None = None
_device_name: str | None = None
_loading: _Load | None = None
_lease: _RepositoryLease | None = None
_failure: tuple[ClefCheckpoint, str, float] | None = None
_generation = 0
_idle_timer: threading.Timer | None = None
_last_activity = 0.0
_shutdown_requested = False
_retiring: set[ClefWorker] = set()

# Injectable spawned-protocol worker for tests.
_WORKER_FACTORY: Callable[[], ClefWorker] = ClefWorker


def _checkpoint_files(checkpoint: ClefCheckpoint) -> tuple[str, ...]:
    files = tuple(checkpoint.files)
    if not files:
        raise FileNotFoundError(f"Clef checkpoint {checkpoint.name} has no approved file manifest.")
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise FileNotFoundError(
                f"Clef checkpoint {checkpoint.name} has an invalid file manifest."
            )
    return files


def _claim_repository_lease(checkpoint: ClefCheckpoint) -> _RepositoryLease | None:
    if checkpoint.is_local:
        return None
    from hub.utils.download_registry import get_models_registry

    registry, owner = get_models_registry(), object()
    claimed, state = registry.claim_repository_owner(checkpoint.source, owner)
    if not claimed:
        action = "is being deleted" if state == "deleting" else f"is busy ({state})"
        raise ClefWorkerError(f"The Clef model cache {action}; retry shortly.")
    return _RepositoryLease(checkpoint.source, registry, owner)


def _release_repository_lease(lease: _RepositoryLease | None) -> None:
    if lease is None:
        return
    try:
        if not lease.registry.release_repository_owner(lease.repo, lease.owner):
            logger.warning("Clef cache lease was no longer owned for %s", lease.repo)
    except Exception:
        logger.warning("Could not release Clef cache lease for %s", lease.repo, exc_info=True)


def _ensure_not_retiring() -> None:
    with _state_lock:
        if _retiring:
            raise Unavailable(
                503, "model_loading", "The previous Clef worker is still stopping; retry shortly.",
                retry_after=1,
            )


def _retire(
    worker: ClefWorker | None, lease: _RepositoryLease | None, *, graceful_timeout: float = 0.0
) -> None:
    if worker is None:
        _release_repository_lease(lease)
        return
    with _state_lock:
        _retiring.add(worker)

    def release() -> None:
        _release_repository_lease(lease)
        with _state_lock:
            _retiring.discard(worker)

    if worker.close(graceful_timeout=graceful_timeout):
        release()
        return

    def release_after_exit() -> None:
        while worker.is_alive():
            time.sleep(_POLL_S)
        worker.close(graceful_timeout=0.0)
        release()

    threading.Thread(target=release_after_exit, name="clef-retire", daemon=True).start()


def _watch_resident(worker: ClefWorker, lease: _RepositoryLease | None, generation: int) -> None:
    while worker.is_alive():
        time.sleep(1.0)
    global _worker, _loaded, _device_name, _lease, _failure, _generation
    with _state_lock:
        if _worker is not worker or _lease is not lease or _generation != generation:
            return
        checkpoint = _loaded
        _cancel_idle_locked()
        _worker = _lease = None
        _loaded = _device_name = None
        _generation += 1
        if checkpoint is not None:
            _failure = (
                checkpoint,
                "The Clef worker stopped unexpectedly.",
                time.monotonic() + FAILURE_BACKOFF_S,
            )
    _retire(worker, lease)


def _checkpoint_dir(checkpoint: ClefCheckpoint) -> Path:
    """Resolve one complete, immutable snapshot without ever downloading it."""
    files = _checkpoint_files(checkpoint)
    if checkpoint.is_local:
        root = Path(checkpoint.source).expanduser()
    else:
        if not checkpoint.revision:
            raise FileNotFoundError(f"Clef checkpoint {checkpoint.name} has no pinned revision.")
        from huggingface_hub import snapshot_download

        from utils.hf_cache_settings import active_hf_hub_cache

        root = Path(
            snapshot_download(
                checkpoint.source,
                revision=checkpoint.revision,
                cache_dir=active_hf_hub_cache(),
                local_files_only=True,
                allow_patterns=list(files),
                token=False,
            )
        )
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"Clef checkpoint {checkpoint.name} is not completely cached ({preview}{suffix}). "
            "Download it from Settings > API before serving it."
        )
    return root


def is_cached(checkpoint: ClefCheckpoint) -> bool:
    try:
        _checkpoint_dir(checkpoint)
    except Exception:
        return False
    return True


def download_plan(checkpoint: ClefCheckpoint) -> dict[str, Any]:
    """The explicit Settings downloader's pinned, network-free request plan."""
    cached = is_cached(checkpoint)
    plan: dict[str, Any] = {
        "repo": None if checkpoint.is_local else checkpoint.source,
        "files": [] if cached else list(_checkpoint_files(checkpoint)),
        "size_bytes": checkpoint.download_bytes,
        "cached": cached,
        "error": None,
        "revision": checkpoint.revision or None,
    }
    if checkpoint.is_local and not cached:
        plan["error"] = f"No complete Clef checkpoint at {checkpoint.source}"
    return plan


def loading_repo_ids() -> tuple[str, ...]:
    with _state_lock:
        loading = _loading
        if loading is not None and not loading.checkpoint.is_local:
            return (loading.checkpoint.source,)
    return ()


def _requested_device() -> str:
    # Read saved settings without probing torch/CUDA in the parent.
    from utils.systemone_settings import get_device

    return "gpu" if get_device() == "gpu" else "cpu"


def _current(load: _Load) -> bool:
    with _state_lock:
        return (
            _loading is load
            and _generation == load.generation
            and not load.cancel.is_set()
            and not _shutdown_requested
        )


def _start_loading(checkpoint: ClefCheckpoint) -> _Load | None:
    """Start one nonblocking local-only load after retiring another Clef model."""
    global _loaded, _worker, _device_name, _loading, _lease, _failure
    with _state_lock:
        switch = (
            _loaded is not None
            and _loaded != checkpoint
            and _worker is not None
            and _worker.is_alive()
        )
    if switch:
        unload()
        return _start_loading(checkpoint)
    old_worker = old_lease = None
    load = None
    error = None
    already_loaded = False
    with _state_lock:
        _ensure_not_retiring()
        if _shutdown_requested:
            raise Unavailable(503, "model_unavailable", "The Clef runtime is shutting down.")
        if _loaded == checkpoint and _worker is not None and _worker.is_alive():
            already_loaded = True
        elif _loaded is not None and (_worker is None or not _worker.is_alive()):
            old_worker, old_lease = _worker, _lease
            _worker = _lease = None
            _loaded = _device_name = None
        if (
            not already_loaded
            and _failure is not None
            and _failure[0] == checkpoint
            and time.monotonic() < _failure[2]
        ):
            error = (_failure[1], max(1.0, _failure[2] - time.monotonic()), "model_unavailable")
        elif not already_loaded and _loading is not None:
            if _loading.checkpoint == checkpoint:
                load = _loading
            else:
                error = (f"{_loading.checkpoint.name} is loading", 5, "model_loading")
        elif not already_loaded:
            load = _Load(checkpoint, _generation, threading.Event(), threading.Event())
            _loading = load
            try:
                load.thread = threading.Thread(
                    target=_load_worker, args=(load,), name="clef-systemone-load", daemon=True
                )
                load.thread.start()
            except Exception as exc:
                _loading = None
                _failure = (
                    checkpoint,
                    f"Could not start {checkpoint.name}: {type(exc).__name__}: {exc}",
                    time.monotonic() + FAILURE_BACKOFF_S,
                )
                load.done.set()
                error = (_failure[1], FAILURE_BACKOFF_S, "model_unavailable")
    _retire(old_worker, old_lease)
    if error is not None:
        raise Unavailable(503, error[2], error[0], retry_after=error[1])
    return None if already_loaded else load


def _load_worker(load: _Load) -> None:
    global _worker, _loaded, _device_name, _loading, _lease, _failure
    worker = None
    lease = None
    published = False
    watch_generation = None
    try:
        if not _current(load):
            return
        # Claim outside _state_lock: the registry calls loading_repo_ids().
        lease = _claim_repository_lease(load.checkpoint)
        if not _current(load):
            return
        with _state_lock:
            if not _current(load):
                return
            load.lease = lease
        snapshot = _checkpoint_dir(load.checkpoint)
        if not _current(load):
            return
        worker = _WORKER_FACTORY()
        with _state_lock:
            if not _current(load):
                return
            load.worker = worker
        worker.start(snapshot, load.checkpoint, _requested_device(), load.cancel)
        if not _current(load):
            return
        with _state_lock:
            if not _current(load):
                return
            _worker, _loaded, _device_name = worker, load.checkpoint, worker.device
            _lease, load.lease, _failure, _loading = lease, None, None, None
            watch_generation = _generation
            published = True
            _touch_locked(worker)
        threading.Thread(
            target=_watch_resident,
            args=(worker, lease, watch_generation),
            name="clef-watch",
            daemon=True,
        ).start()
        logger.info("Clef loaded %s on %s", load.checkpoint.name, worker.device)
    except ClefWorkerCancelled:
        pass
    except Exception as exc:
        message = f"Could not load {load.checkpoint.name}: {type(exc).__name__}: {exc}"
        logger.warning("Clef load failed: %s", message, exc_info=True)
        with _state_lock:
            if _loading is load and not load.cancel.is_set() and not _shutdown_requested:
                _failure = (load.checkpoint, message, time.monotonic() + FAILURE_BACKOFF_S)
                _loading = None
    finally:
        if not published:
            _retire(worker, lease)
        with _state_lock:
            if _loading is load:
                _loading = None
            load.lease = None
        load.done.set()


def prepare(checkpoint: ClefCheckpoint) -> None:
    """Nonblocking load registration for the Decisions GPU-arbiter handoff."""
    _start_loading(checkpoint)


def _worker_for(checkpoint: ClefCheckpoint) -> ClefWorker:
    load = _start_loading(checkpoint)
    if load is None:
        with _state_lock:
            if _loaded == checkpoint and _worker is not None and _worker.is_alive():
                return _worker
        return _worker_for(checkpoint)
    if not load.done.wait(LOAD_WAIT_S):
        raise Unavailable(
            503,
            "model_loading",
            f"{checkpoint.name} is still loading",
            retry_after=5,
        )
    with _state_lock:
        if _loaded == checkpoint and _worker is not None and _worker.is_alive():
            return _worker
        failure = _failure if _failure and _failure[0] == checkpoint else None
        if failure is not None and time.monotonic() < failure[2]:
            raise Unavailable(
                503,
                "model_unavailable",
                failure[1],
                retry_after=max(1.0, failure[2] - time.monotonic()),
            )
    raise Unavailable(503, "model_loading", f"{checkpoint.name} is reloading", retry_after=1)


def _cancel_idle_locked() -> None:
    global _idle_timer
    timer, _idle_timer = _idle_timer, None
    if timer is not None:
        timer.cancel()


def _touch_locked(worker: ClefWorker) -> None:
    global _idle_timer, _last_activity
    _cancel_idle_locked()
    _last_activity = time.monotonic()
    timer = threading.Timer(IDLE_UNLOAD_S, _idle_unload, args=(worker, _generation))
    timer.daemon = True
    _idle_timer = timer
    timer.start()


def _idle_unload(worker: ClefWorker, generation: int) -> None:
    global _worker, _loaded, _device_name, _lease, _generation, _idle_timer
    with _state_lock:
        if (
            _shutdown_requested
            or _worker is not worker
            or _generation != generation
            or time.monotonic() - _last_activity < IDLE_UNLOAD_S
        ):
            return
        if not _run_lock.acquire(blocking=False):
            timer = threading.Timer(1.0, _idle_unload, args=(worker, generation))
            timer.daemon = True
            _idle_timer = timer
            timer.start()
            return
        _cancel_idle_locked()
        lease, _lease = _lease, None
        _retiring.add(worker)
        _worker = None
        _loaded = _device_name = None
        _generation += 1
    try:
        _retire(worker, lease, graceful_timeout=SHUTDOWN_WAIT_S)
    finally:
        _run_lock.release()


def decide(
    checkpoint: ClefCheckpoint,
    state: Any,
    questions: dict[str, dict[str, Any]],
    images: list[bytes] | None = None,
) -> dict[str, Any]:
    """Return a Clef SystemOne wire object through one bounded owned worker."""
    if not _admission.acquire(blocking=False):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after=1)
    try:
        return _decide(checkpoint, state, questions, list(images or ()))
    finally:
        _admission.release()


def _decide(
    checkpoint: ClefCheckpoint,
    state: Any,
    questions: dict[str, dict[str, Any]],
    images: list[bytes],
) -> dict[str, Any]:
    worker = _worker_for(checkpoint)
    if not _run_lock.acquire(timeout=RUN_WAIT_S):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", retry_after=1)
    try:
        with _state_lock:
            if _worker is not worker or _loaded != checkpoint:
                raise Unavailable(
                    503, "model_loading", f"{checkpoint.name} is reloading", retry_after=1
                )
            _cancel_idle_locked()
        try:
            result = worker.decide(checkpoint, state, questions, images)
        except ClefWorkerInputError as exc:
            raise Unavailable(422, "invalid_request_error", str(exc)) from None
        except ClefWorkerCancelled as exc:
            raise Unavailable(503, "model_loading", str(exc), retry_after=1) from None
        except ClefWorkerError as exc:
            _forget_failed_worker(worker, checkpoint, str(exc))
            raise Unavailable(503, "model_unavailable", str(exc), retry_after=1) from None
        with _state_lock:
            if _worker is not worker or _loaded != checkpoint:
                raise Unavailable(
                    503, "model_loading", f"{checkpoint.name} is reloading", retry_after=1
                )
            _touch_locked(worker)
        return result
    finally:
        _run_lock.release()


def _forget_failed_worker(worker: ClefWorker, checkpoint: ClefCheckpoint, message: str) -> None:
    global _worker, _loaded, _device_name, _lease, _failure, _generation
    with _state_lock:
        if _worker is not worker:
            return
        _cancel_idle_locked()
        lease, _lease = _lease, None
        _retiring.add(worker)
        _worker = None
        _loaded = _device_name = None
        _generation += 1
        _failure = (
            checkpoint,
            f"Clef worker failed: {message}",
            time.monotonic() + FAILURE_BACKOFF_S,
        )
    _retire(worker, lease)


def ensure_can_unload() -> None:
    """Only a running decision blocks retirement; loads are generation-cancellable."""
    _ensure_not_retiring()
    if not _run_lock.acquire(blocking=False):
        raise Unavailable(
            409,
            "model_loading",
            "Wait for the Clef decision to finish before unloading.",
        )
    _run_lock.release()


def _invalidate(
    *, stopping: bool
) -> tuple[ClefWorker | None, _Load | None, _RepositoryLease | None, bool]:
    global \
        _worker, \
        _loaded, \
        _device_name, \
        _loading, \
        _lease, \
        _failure, \
        _generation, \
        _shutdown_requested
    with _state_lock:
        if stopping:
            _shutdown_requested = True
        _cancel_idle_locked()
        worker, loading, lease = _worker, _loading, _lease
        was_loaded = worker is not None
        for retiring in (worker, loading.worker if loading is not None else None):
            if retiring is not None:
                _retiring.add(retiring)
        _worker = _lease = None
        _loaded = _device_name = None
        _loading = None
        _failure = None
        _generation += 1
        if loading is not None:
            loading.cancel.set()
            loading_worker = loading.worker
        else:
            loading_worker = None
    if worker is not None:
        worker.cancel()
    if loading_worker is not None and loading_worker is not worker:
        loading_worker.cancel()
    return worker, loading, lease, was_loaded


def unload() -> bool:
    """Cancel a load or retire an idle worker; never interrupt a decision."""
    ensure_can_unload()
    if not _run_lock.acquire(blocking=False):
        raise Unavailable(
            409, "model_loading", "Wait for the Clef decision to finish before unloading."
        )
    try:
        worker, loading, lease, was_loaded = _invalidate(stopping=False)
        _retire(worker, lease, graceful_timeout=SHUTDOWN_WAIT_S)
        if loading is not None:
            pending_worker = loading.worker
            if pending_worker is not None and pending_worker is not worker:
                pending_worker.close(graceful_timeout=0.0)
            loading.done.wait(LOAD_CANCEL_WAIT_S)
        _ensure_not_retiring()
        return was_loaded
    finally:
        _run_lock.release()


def shutdown() -> bool:
    """Force-retire every owned worker, including a running decision or load."""
    worker, loading, lease, was_loaded = _invalidate(stopping=True)
    _retire(worker, lease)
    if loading is not None:
        loading.done.wait(LOAD_CANCEL_WAIT_S)
        pending_worker = loading.worker
        if pending_worker is not None and pending_worker is not worker:
            pending_worker.close(graceful_timeout=0.0)
    return was_loaded


def status() -> dict[str, Any]:
    with _state_lock:
        failure = _failure if _failure and time.monotonic() < _failure[2] else None
        worker_alive = _worker is not None and _worker.is_alive()
        return {
            "loaded_model": _loaded.name if _loaded and worker_alive else None,
            "device": _device_name if worker_alive else None,
            "loading_model": _loading.checkpoint.name if _loading else None,
            "installing": False,
            "error": failure[1] if failure else None,
            "error_model": failure[0].name if failure else None,
        }


# Backstop for interpreter exits that bypass the application lifespan.
atexit.register(shutdown)
