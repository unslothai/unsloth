# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whisper (Transformers) dictation in a spawn child, and the handle for it.

A CUDA context costs ~700 MiB on a current card and is never returned while the
process that created it lives, so an in-process load left the backend that much
heavier for good: unloading the model frees the weights and nothing else (the
rule is written down in core/inference/gpu_arbiter.py). whisper.cpp and
llama.cpp dictation already serve from their own processes; this puts the
Transformers engine on the same footing, so an idle unload ends a process and
the context goes with it.

Both sides live here: ``run_stt_worker`` is the child entrypoint (imported by
name in the child, so it must stay at module scope for Windows spawn), and
``WhisperWorker`` is the parent-side handle the sidecar holds instead of a
model object.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# Spawn, never fork: a forked CUDA context is unusable, and Windows and macOS
# have no fork to begin with.
_CTX = mp.get_context("spawn")

_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)

# Both bounds exist only to break a hang; a large-v3 load off a cold disk and a
# 30 minute transcription are both legitimately slow.
_LOAD_TIMEOUT_SECONDS = 600.0
_TRANSCRIBE_TIMEOUT_SECONDS = 600.0
# How long a cancelled command gets to come back on its own before the child is
# killed. Generation stops within a token, but a load inside from_pretrained
# reaches no checkpoint at all, and training is waiting for that memory.
_CANCEL_GRACE_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_POLL_SECONDS = 0.1

# Errors the child may report that the parent must re-raise as themselves; any
# other failure crosses as a RuntimeError carrying the child's message.
_FORWARDED_ERRORS = (
    "SttLoadCancelledError",
    "SttTranscriptionCancelledError",
    "SttModelNotDownloadedError",
    "SttModelCompatibilityError",
    "SttUnavailableError",
)


class SttWorkerError(RuntimeError):
    """The dictation worker crashed, hung, or could not be reached."""


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------


def _ensure_backend_on_path() -> None:
    if _BACKEND_PATH not in sys.path:
        sys.path.insert(0, _BACKEND_PATH)


class _CancelCriteria:
    """Stops generation as soon as the parent sets the shared cancel event."""

    def __init__(self, cancel_event) -> None:
        self._cancel_event = cancel_event

    def __call__(self, *_args, **_kwargs) -> bool:
        return self._cancel_event.is_set()


def _raise_if_cancelled(cancel_event) -> None:
    from core.inference.stt_sidecar import SttLoadCancelledError

    if cancel_event is not None and cancel_event.is_set():
        raise SttLoadCancelledError("STT model loading was cancelled so training could start.")


def load_whisper(snapshot_path: str, device: str, dtype_name: str, cancel_event = None) -> tuple:
    """Load a Whisper model + processor from the local Hub cache. Child side.

    local_files_only keeps the Model Hub the only download path; a cache miss
    raises so the parent can surface SttModelNotDownloadedError.
    """
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = getattr(torch, dtype_name, None) or torch.float32
    processor = WhisperProcessor.from_pretrained(snapshot_path, local_files_only = True)
    _raise_if_cancelled(cancel_event)
    # use_safetensors forces the pickle-free load path even if a
    # pytorch_model.bin somehow reached the cache; the selector and the
    # completeness check already exclude pickle weights upstream.
    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot_path, torch_dtype = dtype, local_files_only = True, use_safetensors = True
    )
    _raise_if_cancelled(cancel_event)
    model.to(torch.device(device))
    _raise_if_cancelled(cancel_event)
    model.eval()
    return model, processor


def transcribe_window(model, processor, pcm: bytes, generate_kwargs: dict, cancel_event = None):
    """Run Whisper over one window of 16 kHz mono float32 PCM. Child side.

    Feeds a pre-decoded array so nothing here touches the Transformers audio
    path (torchcodec/ffmpeg); the parent decodes and windows.
    """
    import numpy as np
    import torch

    segment = np.frombuffer(pcm, dtype = np.float32)
    kwargs = dict(generate_kwargs)
    if cancel_event is not None:
        from transformers import StoppingCriteriaList

        kwargs["stopping_criteria"] = StoppingCriteriaList([_CancelCriteria(cancel_event)])
    from core.inference.stt_sidecar import _TARGET_SAMPLE_RATE

    inputs = processor(segment, sampling_rate = _TARGET_SAMPLE_RATE, return_tensors = "pt")
    features = inputs.input_features.to(model.device)
    target_dtype = getattr(model, "dtype", None)
    if target_dtype is not None:
        features = features.to(target_dtype)
    with torch.no_grad():
        generated = model.generate(features, **kwargs)
    text = processor.batch_decode(generated, skip_special_tokens = True)
    return text[0] if text else ""


def _error_response(exc: BaseException) -> dict:
    """Describe a failure so the parent can re-raise the same class.

    The exception itself is not sent: an arbitrary Transformers or torch error
    may not pickle, and a queue that fails to serialise costs the caller its
    whole timeout instead of an error.
    """
    from core.inference.stt_sidecar import _is_missing_local_model_error

    kind = type(exc).__name__
    message = str(exc) or kind
    if kind not in _FORWARDED_ERRORS and _is_missing_local_model_error(exc):
        # A local-cache miss is the one generic error with a specific meaning
        # upstream: the model is not downloaded, not a broken runtime.
        kind = "SttModelNotDownloadedError"
        message = "The dictation model is not downloaded."
    return {"type": "error", "kind": kind, "error": message}


def _send(resp_queue, response: dict) -> None:
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("STT worker could not answer the backend: %s", exc)


def run_stt_worker(*, cmd_queue, resp_queue, cancel_event, config: Optional[dict] = None) -> None:
    """Child entrypoint: hold one Whisper model and answer transcription commands.

    Returning ends the process, which is the only way to give the CUDA context
    back, so a failed load and an unload both exit rather than idle.
    """
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _ensure_backend_on_path()
    try:
        from loggers.config import LogConfig

        LogConfig.setup_logging(
            service_name = "unsloth-studio-stt-worker",
            env = os.getenv("ENVIRONMENT_TYPE", "production"),
        )
    except Exception as exc:  # noqa: BLE001 - logging setup must not fail dictation
        logger.debug("STT worker logging setup failed: %s", exc)

    engine = None
    while True:
        try:
            command = cmd_queue.get(timeout = 1.0)
        except _queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if not isinstance(command, dict):
            continue
        kind = command.get("type")
        try:
            if kind == "load":
                model, processor = load_whisper(
                    command["snapshot_path"],
                    command["device"],
                    command["dtype"],
                    cancel_event,
                )
                engine = (model, processor)
                generation_config = getattr(model, "generation_config", None)
                is_multilingual = getattr(generation_config, "is_multilingual", None)
                _send(
                    resp_queue,
                    {
                        "type": "loaded",
                        "device": command["device"],
                        "is_multilingual": (
                            is_multilingual if isinstance(is_multilingual, bool) else None
                        ),
                    },
                )
            elif kind == "transcribe":
                if engine is None:
                    raise SttWorkerError("The dictation worker has no model loaded.")
                cancellable = bool(command.get("cancellable"))
                text = transcribe_window(
                    engine[0],
                    engine[1],
                    command["audio"],
                    command.get("generate_kwargs") or {},
                    cancel_event if cancellable else None,
                )
                if cancellable and cancel_event.is_set():
                    from core.inference.stt_sidecar import SttTranscriptionCancelledError

                    raise SttTranscriptionCancelledError("Transcription cancelled.")
                _send(resp_queue, {"type": "text", "text": text})
            elif kind == "shutdown":
                _send(resp_queue, {"type": "shutdown_ack"})
                return
            else:
                _send(resp_queue, {"type": "error", "kind": "SttWorkerError",
                                   "error": f"Unknown command '{kind}'."})
        except BaseException as exc:  # noqa: BLE001 - every failure is reported, then handled
            _send(resp_queue, _error_response(exc))
            if kind == "load":
                # Nothing is resident after a failed load, and the attempt may
                # already have taken a context; exiting returns it.
                return


# ---------------------------------------------------------------------------
# Parent process
# ---------------------------------------------------------------------------


def _raise_worker_error(response: dict) -> None:
    from core.inference import stt_sidecar

    kind = response.get("kind")
    message = response.get("error") or "The dictation worker failed."
    error_type = getattr(stt_sidecar, kind, None) if kind in _FORWARDED_ERRORS else None
    raise (error_type or SttWorkerError)(message)


class WhisperWorker:
    """Parent-side handle for one Whisper model living in one child process.

    Stands in for the ``(model, processor)`` pair the sidecar used to hold, so
    the sidecar keeps its lifecycle, its lock and its idle timer unchanged.
    Not thread-safe by itself: the sidecar serialises every call under its
    model lock, exactly as it did for in-process inference.
    """

    def __init__(self) -> None:
        self._process = None
        self._cmd_queue = None
        self._resp_queue = None
        self._cancel_event = None
        self.device: Optional[str] = None
        # Read by the sidecar the way it read the model's, so an English-only
        # checkpoint still drops the task/language kwargs it rejects.
        self.generation_config = SimpleNamespace(is_multilingual = None)

    def start(
        self,
        snapshot_path: str,
        device: str,
        dtype_name: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Spawn the child and load the model, or raise having killed it."""
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import adopt_pid

        cache_env = get_hf_cache_paths().child_env({})
        with (
            child_environment_for_spawn(cache_env),
            native_path_secret_removed_for_child_start(),
        ):
            self._cmd_queue = _CTX.Queue()
            self._resp_queue = _CTX.Queue()
            self._cancel_event = _CTX.Event()
            self._process = _CTX.Process(
                # The shared shim binds the child to this process's lifetime and
                # applies the Hub cache environment before any import.
                target = run_without_native_path_secret,
                args = ("core.inference.stt_transformers_worker", "run_stt_worker", cache_env),
                kwargs = {
                    "cmd_queue": self._cmd_queue,
                    "resp_queue": self._resp_queue,
                    "cancel_event": self._cancel_event,
                    "config": {},
                },
                daemon = True,
            )
            self._process.start()
        adopt_pid(self._process.pid)  # terminate_all backstop for graceful exits
        logger.info("STT worker started (pid=%s) for %s on %s", self._process.pid,
                    snapshot_path, device)
        try:
            self._send(
                {
                    "type": "load",
                    "snapshot_path": str(snapshot_path),
                    "device": device,
                    "dtype": dtype_name,
                }
            )
            response = self._await("loaded", _LOAD_TIMEOUT_SECONDS, cancel_event, "load")
        except BaseException:
            self.close()
            raise
        self.device = response.get("device") or device
        self.generation_config = SimpleNamespace(
            is_multilingual = response.get("is_multilingual")
        )

    def transcribe_window(
        self,
        pcm: bytes,
        generate_kwargs: dict,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Transcribe one decoded window and return its text."""
        if self._cancel_event is not None:
            self._cancel_event.clear()
        self._send(
            {
                "type": "transcribe",
                "audio": pcm,
                "generate_kwargs": dict(generate_kwargs),
                "cancellable": cancel_event is not None,
            }
        )
        response = self._await(
            "text", _TRANSCRIBE_TIMEOUT_SECONDS, cancel_event, "transcribe"
        )
        text = response.get("text")
        return text if isinstance(text, str) else ""

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def cancel(self) -> None:
        """Ask the child to stop the command it is running."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def close(self) -> None:
        """Stop the child, which is what actually returns its accelerator memory."""
        process = self._process
        self._process = None
        self.cancel()  # unblock a generation before asking the loop to exit
        if process is not None:
            try:
                if process.is_alive() and self._cmd_queue is not None:
                    self._cmd_queue.put({"type": "shutdown"})
            except (OSError, ValueError):
                pass
            process.join(_SHUTDOWN_TIMEOUT_SECONDS)
            if process.is_alive():
                logger.warning("STT worker %s did not exit; terminating", process.pid)
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(3)
            try:
                from utils.process_lifetime import forget_pid

                forget_pid(process.pid)
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail an unload
                logger.debug("Could not forget STT worker pid %s: %s", process.pid, exc)
        for handle in (self._cmd_queue, self._resp_queue):
            try:
                if handle is not None:
                    # The feeder thread must not outlive the queue it feeds.
                    handle.cancel_join_thread()
                    handle.close()
            except Exception:  # noqa: BLE001 - a closed queue is best effort
                pass
        self._cmd_queue = None
        self._resp_queue = None

    def _send(self, command: dict) -> None:
        if self._cmd_queue is None:
            raise SttWorkerError("The dictation worker is not running.")
        try:
            self._cmd_queue.put(command)
        except (OSError, ValueError) as exc:
            raise SttWorkerError(f"Could not reach the dictation worker: {exc}") from exc

    def _await(
        self,
        expected: str,
        timeout: float,
        cancel_event: Optional[threading.Event],
        phase: str,
    ) -> dict:
        """Wait for one response, mirroring cancellation and watching for death."""
        deadline = time.monotonic() + timeout
        cancel_deadline: Optional[float] = None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.cancel()
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                elif time.monotonic() >= cancel_deadline:
                    # A load inside from_pretrained never sees the event, and the
                    # memory is wanted now, so stop asking and end the process.
                    self.close()
                    self._raise_cancelled(phase)
            try:
                response = self._resp_queue.get(timeout = _POLL_SECONDS)
            except _queue.Empty:
                if not self.is_alive():
                    raise SttWorkerError(self._crash_message(phase))
                if time.monotonic() >= deadline:
                    self.close()
                    raise SttWorkerError(
                        "The dictation worker stopped responding; try recording again."
                    )
                continue
            except (EOFError, OSError, ValueError) as exc:
                raise SttWorkerError(f"Lost contact with the dictation worker: {exc}") from exc
            if not isinstance(response, dict):
                continue
            if response.get("type") == "error":
                _raise_worker_error(response)
            if response.get("type") == expected:
                return response

    @staticmethod
    def _raise_cancelled(phase: str) -> None:
        from core.inference.stt_sidecar import (
            SttLoadCancelledError,
            SttTranscriptionCancelledError,
        )

        if phase == "load":
            raise SttLoadCancelledError(
                "STT model loading was cancelled so training could start."
            )
        raise SttTranscriptionCancelledError("Transcription cancelled.")

    def _crash_message(self, phase: str) -> str:
        process = self._process
        exitcode = getattr(process, "exitcode", None)
        detail = f"exitcode={exitcode}"
        if isinstance(exitcode, int) and exitcode < 0:
            import signal

            try:
                name = signal.Signals(-exitcode).name
            except ValueError:
                name = f"SIG{-exitcode}"
            detail = f"signal={name}"
            if name == "SIGKILL":
                # The usual cause on a busy box, and the one the user can act on.
                detail += "; the system may have killed it under memory pressure"
        what = "loading the dictation model" if phase == "load" else "transcribing"
        return f"The dictation worker stopped while {what} ({detail})."
