# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One owned Clef worker, shared by native llama.cpp and official PyTorch."""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import ClefCheckpoint
from .laya_runtime import Unavailable

logger = logging.getLogger(__name__)
LOAD_WAIT_S = RUN_WAIT_S = 300.0
CANCEL_GRACE_S = 5.0
LOAD_CANCEL_WAIT_S = 12.0
IDLE_UNLOAD_S = 300.0
FAILURE_BACKOFF_S = 60.0
MAX_PENDING = 8
_POLL_S = 0.1
_CTX = mp.get_context("spawn")


class ClefWorkerError(RuntimeError):
    pass


class ClefWorkerCancelled(ClefWorkerError):
    pass


class ClefWorkerInputError(ClefWorkerError):
    pass


class ClefWorker:
    """Torch-free handle for the official adapter's isolated process."""

    def __init__(self, target = None):
        self._target = target
        self._process = self._cmd_queue = self._resp_queue = self._cancel_event = None
        self._closed = False
        self._close_lock = threading.Lock()
        self.device = None
        self.gpu_available = None

    def start(self, snapshot_path, checkpoint, requested_device, cancelled):
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import (
            adopt_pid,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )

        if cancelled.is_set() or is_process_shutting_down():
            raise ClefWorkerCancelled("Clef model loading was cancelled.")
        cache_env = get_hf_cache_paths().child_env({})
        try:
            with (
                child_environment_for_spawn(cache_env),
                native_path_secret_removed_for_child_start(),
            ):
                self._cmd_queue, self._resp_queue = _CTX.Queue(), _CTX.Queue()
                self._cancel_event = _CTX.Event()
                target = self._target or run_without_native_path_secret
                args = (
                    ()
                    if self._target
                    else ("core.systemone.clef_worker", "run_clef_worker", cache_env)
                )
                self._process = _CTX.Process(
                    target = target,
                    args = args,
                    kwargs = dict(
                        cmd_queue = self._cmd_queue,
                        resp_queue = self._resp_queue,
                        cancel_event = self._cancel_event,
                    ),
                    daemon = True,
                )
                # Linux's parent-death signal follows the spawning thread's lifetime.
                spawn_on_lifetime_thread(self._process.start)
            adopt_pid(self._process.pid)
            if cancelled.is_set() or is_process_shutting_down():
                raise ClefWorkerCancelled("Clef model loading was cancelled.")
            self._send(
                dict(
                    type = "load",
                    snapshot_path = str(snapshot_path),
                    model = checkpoint.name,
                    requested_device = requested_device,
                )
            )
            loaded = self._await("loaded", LOAD_WAIT_S, cancelled)
            self.device, self.gpu_available = loaded["device"], loaded["gpu_available"]
        except BaseException:
            self.close()
            raise

    def decide(self, checkpoint, state, questions, images):
        self._send(
            dict(
                type = "decide",
                model = checkpoint.name,
                state = state,
                questions = questions,
                images = images,
            )
        )
        result = self._await("result", RUN_WAIT_S).get("result")
        if not isinstance(result, dict):
            raise ClefWorkerError("The Clef worker returned an invalid response.")
        return result

    def _send(self, command):
        if self._closed or self._cmd_queue is None:
            raise ClefWorkerCancelled("The Clef worker was unloaded.")
        self._cmd_queue.put(command)

    def _await(
        self,
        expected,
        timeout,
        cancelled = None,
    ):
        deadline = time.monotonic() + timeout
        while not self._closed:
            if cancelled is not None and cancelled.is_set():
                raise ClefWorkerCancelled("Clef model loading was cancelled.")
            if time.monotonic() >= deadline:
                raise ClefWorkerError("The Clef worker operation timed out.")
            channel = self._resp_queue
            if channel is None:
                raise ClefWorkerCancelled("The Clef worker was unloaded.")
            try:
                response = channel.get(timeout = _POLL_S)
            except queue.Empty:
                if self._closed:
                    raise ClefWorkerCancelled("The Clef worker was unloaded.")
                if not self.is_alive():
                    code = getattr(self._process, "exitcode", None)
                    raise ClefWorkerError(f"The Clef worker stopped (exitcode={code}).")
                continue
            except (EOFError, OSError, ValueError) as exc:
                raise ClefWorkerError("Lost contact with the Clef worker.") from exc
            if response.get("type") == "error":
                error = {
                    "cancelled": ClefWorkerCancelled,
                    "invalid_request_error": ClefWorkerInputError,
                }.get(response.get("kind"), ClefWorkerError)
                raise error(response.get("message") or "The Clef worker failed.")
            if response.get("type") == expected:
                return response
        raise ClefWorkerCancelled("The Clef worker was unloaded.")

    def cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def close(self, graceful_timeout = 0.0):
        from utils.process_lifetime import forget_pid
        with self._close_lock:
            self._closed = True
            self.cancel()
            process = self._process
            if process is not None and process.pid is not None:
                if graceful_timeout and self.is_alive():
                    self._cmd_queue.put({"type": "shutdown"})
                    process.join(graceful_timeout)
                for stop in (process.terminate, process.kill):
                    if self.is_alive():
                        stop()
                        process.join(CANCEL_GRACE_S)
                if self.is_alive():
                    return False
                forget_pid(process.pid)
            self._process = None
            for channel in (self._cmd_queue, self._resp_queue):
                if channel is not None:
                    channel.cancel_join_thread()
                    channel.close()
            self._cmd_queue = self._resp_queue = None
            return True


@dataclass
class _Load:
    checkpoint: ClefCheckpoint
    generation: int
    cancel: threading.Event
    done: threading.Event
    worker: Any = None
    thread: threading.Thread | None = None


_state_lock = threading.RLock()
_run_lock = threading.Lock()
_admission = threading.BoundedSemaphore(MAX_PENDING)
_worker = _loaded = _device_name = _loading = _lease = _failure = _idle_timer = None
_generation = 0
_shutdown_requested = False
_retiring = set()
_WORKER_FACTORY = None


def is_native(checkpoint):
    return len(checkpoint.files) == 1 and checkpoint.files[0].endswith(".gguf")


def _claim_repository_lease(checkpoint):
    if checkpoint.is_local:
        return None
    from hub.utils.download_registry import get_models_registry

    registry, owner = get_models_registry(), object()
    claimed, state = registry.claim_repository_owner(checkpoint.source, owner)
    if not claimed:
        action = "is being deleted" if state == "deleting" else f"is busy ({state})"
        raise ClefWorkerError(f"The Clef model cache {action}; retry shortly.")
    return registry, checkpoint.source, owner


def _release_gpu_if_idle():
    from core.inference.gpu_arbiter import DECISIONS, release_if
    def idle():
        with _state_lock:
            return _loading is None and not _retiring and (_worker is None or _device_name == "cpu")

    # Eviction/registration can hold the arbiter lock; recheck ownership off-thread.
    threading.Thread(
        target = lambda: release_if(DECISIONS, idle), name = "clef-gpu-release", daemon = True
    ).start()


def _retire(
    worker,
    lease,
    *,
    graceful_timeout = 0.0,
):
    def release():
        if lease is not None:
            registry, repo, owner = lease
            if not registry.release_repository_owner(repo, owner):
                logger.warning("Clef cache lease was no longer owned for %s", repo)
        with _state_lock:
            _retiring.discard(worker)
        _release_gpu_if_idle()

    if worker is None:
        release()
        return
    with _state_lock:
        _retiring.add(worker)
    if worker.close(graceful_timeout = graceful_timeout):
        release()
        return

    def after_exit():
        while worker.is_alive():
            time.sleep(_POLL_S)
        worker.close()
        release()

    threading.Thread(target = after_exit, name = "clef-retire", daemon = True).start()


def _ensure_not_retiring():
    with _state_lock:
        if _retiring:
            raise Unavailable(
                503,
                "model_loading",
                "The previous Clef worker is still stopping; retry shortly.",
                1,
            )


def _checkpoint_dir(checkpoint):
    files = checkpoint.files
    if not files or any(Path(f).is_absolute() or ".." in Path(f).parts for f in files):
        raise FileNotFoundError("Clef checkpoint has no valid approved file manifest.")
    if checkpoint.is_local:
        root = Path(checkpoint.source).expanduser()
    else:
        from huggingface_hub import snapshot_download
        from utils.hf_cache_settings import active_hf_hub_cache

        if len(checkpoint.revision) != 40:
            raise FileNotFoundError("Clef checkpoint has no immutable revision.")
        root = Path(
            snapshot_download(
                checkpoint.source,
                revision = checkpoint.revision,
                cache_dir = active_hf_hub_cache(),
                local_files_only = True,
                allow_patterns = list(files),
                token = False,
            )
        )
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Clef checkpoint is not completely cached ({', '.join(missing[:3])}). Download it from Settings > API before serving it."
        )
    return root / files[0] if is_native(checkpoint) else root


def is_cached(checkpoint):
    try:
        _checkpoint_dir(checkpoint)
        return True
    except (OSError, ValueError):
        return False


def download_plan(checkpoint):
    cached = is_cached(checkpoint)
    return dict(
        repo = None if checkpoint.is_local else checkpoint.source,
        files = [] if cached else list(checkpoint.files),
        size_bytes = checkpoint.download_bytes,
        cached = cached,
        error = "Local Clef checkpoint is incomplete."
        if checkpoint.is_local and not cached
        else None,
        revision = checkpoint.revision or None,
    )


def loading_repo_ids():
    with _state_lock:
        return (
            (_loading.checkpoint.source,) if _loading and not _loading.checkpoint.is_local else ()
        )


def _current(load):
    return (
        _loading is load
        and _generation == load.generation
        and not load.cancel.is_set()
        and not _shutdown_requested
    )


def _start_loading(checkpoint):
    global _loading
    with _state_lock:
        switch = _worker is not None and (_loaded != checkpoint or not _worker.is_alive())
    if switch:
        unload()
    with _state_lock:
        _ensure_not_retiring()
        if _shutdown_requested:
            raise Unavailable(503, "model_unavailable", "The Clef runtime is shutting down.")
        if _loaded == checkpoint and _worker is not None:
            return None
        if _failure and _failure[0] == checkpoint and time.monotonic() < _failure[2]:
            raise Unavailable(503, "model_unavailable", _failure[1], _failure[2] - time.monotonic())
        if _loading:
            if _loading.checkpoint != checkpoint:
                raise Unavailable(503, "model_loading", f"{_loading.checkpoint.name} is loading", 1)
            return _loading
        load = _Load(checkpoint, _generation, threading.Event(), threading.Event())
        _loading = load
        load.thread = threading.Thread(
            target = _load_worker, args = (load,), name = "clef-load", daemon = True
        )
        try:
            load.thread.start()
        except BaseException:
            _loading = None
            raise
        return load


def _load_worker(load):
    global _worker, _loaded, _device_name, _loading, _lease, _failure
    worker = lease = None
    published = False
    try:
        lease = _claim_repository_lease(load.checkpoint)
        path = _checkpoint_dir(load.checkpoint)
        with _state_lock:
            if not _current(load):
                return
        from utils.systemone_settings import get_device

        if _WORKER_FACTORY:
            worker = _WORKER_FACTORY()
        elif is_native(load.checkpoint):
            from .native_worker import NativeWorker
            worker = NativeWorker()
        else:
            worker = ClefWorker()
        with _state_lock:
            load.worker = worker
            if not _current(load):
                return
        worker.start(path, load.checkpoint, get_device(), load.cancel)
        with _state_lock:
            if not _current(load):
                return
            _worker, _loaded, _device_name, _lease = worker, load.checkpoint, worker.device, lease
            _failure = _loading = None
            _touch_locked(worker)
            published = True
    except ClefWorkerCancelled:
        pass
    except Exception as exc:
        logger.warning("Clef load failed: %s", exc)
        with _state_lock:
            if _current(load):
                _failure = (load.checkpoint, str(exc), time.monotonic() + FAILURE_BACKOFF_S)
    finally:
        if not published:
            _retire(worker, lease)
        with _state_lock:
            if _loading is load:
                _loading = None
        _release_gpu_if_idle()
        load.done.set()


def prepare(checkpoint):
    _start_loading(checkpoint)


def _worker_for(checkpoint):
    load = _start_loading(checkpoint)
    if load is not None and not load.done.wait(LOAD_WAIT_S):
        raise Unavailable(503, "model_loading", f"{checkpoint.name} is still loading", 5)
    with _state_lock:
        if _loaded == checkpoint and _worker is not None:
            return _worker
        if _failure and _failure[0] == checkpoint:
            raise Unavailable(
                503, "model_unavailable", _failure[1], max(1, _failure[2] - time.monotonic())
            )
    raise Unavailable(503, "model_loading", f"{checkpoint.name} is reloading", 1)


def _cancel_idle_locked():
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _touch_locked(worker):
    global _idle_timer
    _cancel_idle_locked()
    _idle_timer = threading.Timer(IDLE_UNLOAD_S, _idle_unload, args = (worker, _generation))
    _idle_timer.daemon = True
    _idle_timer.start()


def _idle_unload(worker, generation):
    with _state_lock:
        if _worker is not worker or _generation != generation:
            return
        if not _run_lock.acquire(blocking = False):
            _touch_locked(worker)
            return
    try:
        with _state_lock:
            if _worker is not worker or _generation != generation:
                return
        _stop(False)
    finally:
        _run_lock.release()


def decide(
    checkpoint,
    state,
    questions,
    images = None,
):
    if not _admission.acquire(blocking = False):
        raise Unavailable(529, "overloaded", "System One is busy; retry shortly", 1)
    held = False
    try:
        worker = _worker_for(checkpoint)
        held = _run_lock.acquire(timeout = RUN_WAIT_S)
        if not held:
            raise Unavailable(529, "overloaded", "System One is busy; retry shortly", 1)
        with _state_lock:
            if _worker is not worker:
                raise Unavailable(503, "model_loading", f"{checkpoint.name} is reloading", 1)
            _cancel_idle_locked()
        try:
            return worker.decide(checkpoint, state, questions, list(images or ()))
        except ClefWorkerInputError as exc:
            raise Unavailable(422, "invalid_request_error", str(exc)) from None
        except ClefWorkerCancelled as exc:
            raise Unavailable(503, "model_loading", str(exc), 1) from None
        except ClefWorkerError as exc:
            shutdown_worker(worker, checkpoint, str(exc))
            raise Unavailable(503, "model_unavailable", str(exc), 1) from None
        finally:
            with _state_lock:
                if _worker is worker:
                    _touch_locked(worker)
    finally:
        if held:
            _run_lock.release()
        _admission.release()


def shutdown_worker(worker, checkpoint, error):
    global _worker, _loaded, _device_name, _lease, _failure
    with _state_lock:
        if _worker is not worker:
            return
        lease, _lease = _lease, None
        _worker = _loaded = _device_name = None
        _failure = (checkpoint, error, time.monotonic() + FAILURE_BACKOFF_S)
    _retire(worker, lease)


def ensure_can_unload():
    _ensure_not_retiring()
    if _run_lock.locked():
        raise Unavailable(
            409, "model_loading", "Wait for the Clef decision to finish before unloading."
        )


def _invalidate(stopping):
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
        _shutdown_requested |= stopping
        _cancel_idle_locked()
        worker, load, lease = _worker, _loading, _lease
        _worker = _loaded = _device_name = _loading = _lease = _failure = None
        _generation += 1
        if load:
            load.cancel.set()
        for candidate in (worker, load.worker if load else None):
            if candidate is not None:
                _retiring.add(candidate)
                candidate.cancel()
        return worker, load, lease


def _stop(stopping):
    worker, load, lease = _invalidate(stopping)
    _retire(worker, lease)
    if load:
        if load.worker is not None:
            load.worker.close()
        load.done.wait(LOAD_CANCEL_WAIT_S)
    return worker is not None


def unload():
    ensure_can_unload()
    if not _run_lock.acquire(blocking = False):
        raise Unavailable(
            409, "model_loading", "Wait for the Clef decision to finish before unloading."
        )
    try:
        result = _stop(False)
        _ensure_not_retiring()
        return result
    finally:
        _run_lock.release()


def shutdown():
    return _stop(True)


def status():
    with _state_lock:
        alive = _worker is not None and _worker.is_alive()
        failure = _failure if _failure and time.monotonic() < _failure[2] else None
        return dict(
            loaded_model = _loaded.name if alive else None,
            device = _device_name if alive else None,
            backend = ("llama.cpp" if is_native(_loaded) else "pytorch") if alive else None,
            loading_model = _loading.checkpoint.name if _loading else None,
            installing = False,
            error = failure[1] if failure else None,
            error_model = failure[0].name if failure else None,
        )


atexit.register(shutdown)
