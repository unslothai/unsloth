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

# spawn, never fork: a forked CUDA context is unusable, and Windows/macOS have no fork
_CTX = mp.get_context("spawn")

_BACKEND_PATH = str(Path(__file__).resolve().parent.parent.parent)

_LOAD_TIMEOUT_SECONDS = 600.0
_TRANSCRIBE_TIMEOUT_SECONDS = 600.0
# A load inside from_pretrained never sees the cancel, so kill after a grace period.
_CANCEL_GRACE_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_POLL_SECONDS = 0.1
_TIMESTAMPS_PER_SECOND = 50

_FORWARDED_ERRORS = (
    "SttLoadCancelledError",
    "SttTranscriptionCancelledError",
    "SttModelNotDownloadedError",
    "SttModelCompatibilityError",
    "SttUnavailableError",
)


class SttWorkerError(RuntimeError):
    """The dictation worker crashed, hung, or could not be reached."""


class SttWorkerSpawnError(SttWorkerError):
    """No child could be created at all: this host forbids or cannot spawn.

    Distinct because it says nothing about the model or the device, so the
    caller answers it by loading in process rather than by giving up.
    """


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


def load_whisper(
    snapshot_path: str,
    device: str,
    dtype_name: str,
    cancel_event = None,
) -> tuple:
    """A cache miss raises, so the parent can report it as SttModelNotDownloadedError."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dtype = getattr(torch, dtype_name, None) or torch.float32
    processor = WhisperProcessor.from_pretrained(snapshot_path, local_files_only = True)
    _raise_if_cancelled(cancel_event)
    # use_safetensors forces the pickle-free load path even if a .bin reached the cache.
    model = WhisperForConditionalGeneration.from_pretrained(
        snapshot_path, torch_dtype = dtype, local_files_only = True, use_safetensors = True
    )
    _raise_if_cancelled(cancel_event)
    model.to(torch.device(device))
    _raise_if_cancelled(cancel_event)
    model.eval()
    return model, processor


def transcribe_window(
    model,
    processor,
    pcm: bytes,
    generate_kwargs: dict,
    cancel_event = None,
):
    """Takes pre-decoded PCM so the Transformers audio path (torchcodec, ffmpeg) is never used."""
    import numpy as np
    import torch

    segment = np.frombuffer(pcm, dtype = np.float32)
    kwargs = dict(generate_kwargs)
    return_timestamps = bool(kwargs.get("return_timestamps"))
    if return_timestamps:
        kwargs["force_unique_generate_call"] = True
    if cancel_event is not None:
        from transformers import StoppingCriteriaList
        kwargs["stopping_criteria"] = StoppingCriteriaList([_CancelCriteria(cancel_event)])
    from core.inference.stt_sidecar import _TARGET_SAMPLE_RATE

    inputs = processor(
        segment,
        sampling_rate = _TARGET_SAMPLE_RATE,
        return_tensors = "pt",
        return_attention_mask = return_timestamps,
    )
    features = inputs.input_features.to(model.device)
    target_dtype = getattr(model, "dtype", None)
    if target_dtype is not None:
        features = features.to(target_dtype)
    if return_timestamps:
        kwargs["attention_mask"] = inputs.attention_mask.to(model.device)
    with torch.no_grad():
        generated = model.generate(features, **kwargs)
    consumed = len(segment)
    if return_timestamps:
        raw_tokens = [int(token) for token in generated[0]]
        indices, end = _complete_segment_indices(raw_tokens, model.generation_config)
        if end:
            consumed = end * _TARGET_SAMPLE_RATE // _TIMESTAMPS_PER_SECOND
        generated = [[raw_tokens[index] for index in indices]]
    text = processor.batch_decode(generated, skip_special_tokens = True)
    return (text[0] if text else ""), consumed


class _MlxSttModel:
    def __init__(self, model) -> None:
        self.model = model


def load_mlx_stt(
    snapshot_path: str,
    device: str,
    cancel_event = None,
) -> _MlxSttModel:
    from core.inference.stt_mlx import ships_custom_code
    from core.inference.stt_sidecar import SttModelCompatibilityError

    if ships_custom_code(snapshot_path):
        raise SttModelCompatibilityError(
            "This dictation model ships its own code, which is not run."
        )
    import mlx.core as mx
    from mlx_audio.stt import load

    if device == "cpu":
        mx.set_default_device(mx.cpu)
    model = load(str(snapshot_path))
    _raise_if_cancelled(cancel_event)
    return _MlxSttModel(model)


# Whisper language codes -> the names Qwen3-ASR lists in config.support_languages.
_MLX_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def _mlx_language(model, code: Optional[str]) -> Optional[str]:
    """The checkpoint's own name for a Whisper language code, or None to let it detect one."""
    name = _MLX_LANGUAGE_NAMES.get(code or "", "").lower()
    supported = getattr(getattr(model, "config", None), "support_languages", None) or []
    return next((lang for lang in supported if name and str(lang).lower() == name), None)


def transcribe_mlx_window(
    engine: _MlxSttModel,
    pcm: bytes,
    language: Optional[str] = None,
) -> tuple:
    import numpy as np

    segment = np.frombuffer(pcm, dtype = np.float32)
    hint = _mlx_language(engine.model, language)
    result = engine.model.generate(segment, **({"language": hint} if hint else {}))
    text = getattr(result, "text", result)
    return (text if isinstance(text, str) else ""), len(segment)


def _complete_segment_indices(tokens: list, generation_config) -> tuple:
    timestamp_begin = generation_config.no_timestamps_token_id + 1
    indexed_tokens = [
        (index, token)
        for index, token in enumerate(tokens)
        if token < generation_config.eos_token_id or token >= timestamp_begin
    ]
    is_timestamp = [token >= timestamp_begin for _, token in indexed_tokens]
    pairs = [i for i in range(1, len(indexed_tokens)) if is_timestamp[i - 1] and is_timestamp[i]]
    if not pairs or is_timestamp[-2:] == [False, True]:
        return [index for index, _ in indexed_tokens], 0
    return (
        [index for index, _ in indexed_tokens[: pairs[-1]]],
        indexed_tokens[pairs[-1] - 1][1] - timestamp_begin,
    )


def _supports_segment_timestamps(generation_config) -> bool:
    return all(
        isinstance(getattr(generation_config, field, None), int)
        for field in ("no_timestamps_token_id", "eos_token_id")
    )


def _error_response(exc: BaseException) -> dict:
    """Sends the class name and message, since an unpicklable exception would cost the caller the
    timeout."""
    from core.inference.stt_sidecar import _is_missing_local_model_error

    kind = type(exc).__name__
    message = str(exc) or kind
    if kind not in _FORWARDED_ERRORS and _is_missing_local_model_error(exc):
        kind = "SttModelNotDownloadedError"
        message = "The dictation model is not downloaded."
    return {"type": "error", "kind": kind, "error": message}


def _send(resp_queue, response: dict) -> None:
    try:
        resp_queue.put(response)
    except (OSError, ValueError) as exc:
        logger.error("STT worker could not answer the backend: %s", exc)


def run_stt_worker(
    *,
    cmd_queue,
    resp_queue,
    cancel_event,
    ready_event = None,
    config: Optional[dict] = None,
) -> None:
    """Returning ends the process, the only way to give back the CUDA context; a failed load exits too."""
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

    # Signal readiness via a shared Event before model work: Queue.put can be lost if the child
    # crashes, and on Windows exit codes cannot tell a crash from a failed bootstrap.
    if ready_event is not None:
        ready_event.set()

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
            if kind == "load" and command.get("runtime") == "mlx":
                engine = load_mlx_stt(command["snapshot_path"], command["device"], cancel_event)
                _send(
                    resp_queue,
                    {
                        "type": "loaded",
                        "device": command["device"],
                        "is_multilingual": None,
                        "supports_timestamps": False,
                    },
                )
            elif kind == "load":
                model, processor = load_whisper(
                    command["snapshot_path"],
                    command["device"],
                    command["dtype"],
                    cancel_event,
                )
                engine = (model, processor)
                generation_config = getattr(model, "generation_config", None)
                is_multilingual = getattr(generation_config, "is_multilingual", None)
                supports_timestamps = _supports_segment_timestamps(generation_config)
                _send(
                    resp_queue,
                    {
                        "type": "loaded",
                        "device": command["device"],
                        "is_multilingual": (
                            is_multilingual if isinstance(is_multilingual, bool) else None
                        ),
                        "supports_timestamps": supports_timestamps,
                    },
                )
            elif kind == "transcribe":
                if engine is None:
                    raise SttWorkerError("The dictation worker has no model loaded.")
                cancellable = bool(command.get("cancellable"))
                if isinstance(engine, _MlxSttModel):
                    text, consumed = transcribe_mlx_window(
                        engine,
                        command["audio"],
                        (command.get("generate_kwargs") or {}).get("language"),
                    )
                else:
                    text, consumed = transcribe_window(
                        engine[0],
                        engine[1],
                        command["audio"],
                        command.get("generate_kwargs") or {},
                        cancel_event if cancellable else None,
                    )
                if cancellable and cancel_event.is_set():
                    from core.inference.stt_sidecar import SttTranscriptionCancelledError
                    raise SttTranscriptionCancelledError("Transcription cancelled.")
                _send(resp_queue, {"type": "text", "text": text, "consumed": consumed})
            elif kind == "shutdown":
                _send(resp_queue, {"type": "shutdown_ack"})
                return
            else:
                _send(
                    resp_queue,
                    {
                        "type": "error",
                        "kind": "SttWorkerError",
                        "error": f"Unknown command '{kind}'.",
                    },
                )
        except BaseException as exc:  # noqa: BLE001 - every failure is reported, then handled
            _send(resp_queue, _error_response(exc))
            if kind == "load":
                return


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
        self._ready_event = None
        self._answered = False
        # A child that ignored terminate and kill: kept for accounting, never reused.
        self.survived_kill = False
        self.device: Optional[str] = None
        self.generation_config = SimpleNamespace(is_multilingual = None)

    def start(
        self,
        snapshot_path: str,
        device: str,
        dtype_name: str,
        cancel_event: Optional[threading.Event] = None,
        runtime: Optional[str] = None,
    ) -> None:
        """Spawn the child and load the model, or raise having killed it."""
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.process_lifetime import adopt_pid, is_process_shutting_down

        # Refuse to spawn once shutdown has begun; no shutdown step unloads this sidecar.
        if is_process_shutting_down():
            raise SttWorkerSpawnError(
                "Unsloth is shutting down; not starting the dictation worker."
            )
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
                self._process = _CTX.Process(
                    target = run_without_native_path_secret,
                    args = ("core.inference.stt_transformers_worker", "run_stt_worker", cache_env),
                    kwargs = {
                        "cmd_queue": self._cmd_queue,
                        "resp_queue": self._resp_queue,
                        "cancel_event": self._cancel_event,
                        "ready_event": self._ready_event,
                        "config": {},
                    },
                    daemon = True,
                )
                # Local handle: a concurrent teardown can clear self._process while start() returns.
                _spawned_proc = self._process
                _spawned_proc.start()
        except Exception as exc:  # noqa: BLE001 - any refusal to spawn reads the same
            self._process = None
            self._close_queues()
            raise SttWorkerSpawnError(
                f"Could not start the dictation worker process: {exc}"
            ) from exc
        adopt_pid(_spawned_proc.pid)
        if is_process_shutting_down():
            logger.info("shutdown began during the spawn; reaping the new dictation worker")
            try:
                if _spawned_proc.is_alive():
                    _spawned_proc.terminate()
                    _spawned_proc.join(5)
                if _spawned_proc.is_alive():
                    _spawned_proc.kill()
                    _spawned_proc.join(5)
            except Exception:  # noqa: BLE001 - the reap is best-effort
                pass
            self._process = None
            self._close_queues()
            raise SttWorkerSpawnError(
                "Unsloth is shutting down; not starting the dictation worker."
            )
        logger.info(
            "STT worker started (pid=%s) for %s on %s", _spawned_proc.pid, snapshot_path, device
        )
        try:
            load_command = {
                "type": "load",
                "snapshot_path": str(snapshot_path),
                "device": device,
                "dtype": dtype_name,
            }
            if runtime is not None:
                load_command["runtime"] = runtime
            self._send(load_command)
            response = self._await("loaded", _LOAD_TIMEOUT_SECONDS, cancel_event, "load")
        except BaseException as exc:
            # Classify before close(), which reaps the child and drops the handle.
            stillborn = self._never_bootstrapped(exc)
            self.close()
            if stillborn:
                raise SttWorkerSpawnError(
                    f"The dictation worker process could not start: {exc}"
                ) from exc
            raise
        self.device = response.get("device") or device
        self.generation_config = SimpleNamespace(
            is_multilingual = response.get("is_multilingual"),
            supports_timestamps = response.get("supports_timestamps") is True,
        )

    def transcribe_window(
        self,
        pcm: bytes,
        generate_kwargs: dict,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, int]:
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
        response = self._await("text", _TRANSCRIBE_TIMEOUT_SECONDS, cancel_event, "transcribe")
        text = response.get("text")
        return (text if isinstance(text, str) else ""), response["consumed"]

    def is_alive(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

    def _never_bootstrapped(self, exc: BaseException) -> bool:
        """A child that dies before ready_event is a host that cannot spawn, not a native crash
        during load."""
        if self._answered or not isinstance(exc, SttWorkerError):
            return False
        ready = self._ready_event
        if ready is not None and ready.is_set():
            return False
        process = self._process
        if process is None or process.is_alive():
            return False
        exitcode = getattr(process, "exitcode", None)
        return isinstance(exitcode, int) and exitcode > 0

    def cancel(self) -> None:
        """Ask the child to stop the command it is running."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def close(self, graceful_timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        """A child that survives kill keeps its handle and pid record, so terminate_all can still
        find it."""
        process = self._process
        self.cancel()  # unblock a generation before asking the loop to exit
        if process is not None:
            try:
                if graceful_timeout > 0 and process.is_alive() and self._cmd_queue is not None:
                    self._cmd_queue.put({"type": "shutdown"})
            except (OSError, ValueError):
                pass
            if graceful_timeout > 0:
                process.join(graceful_timeout)
            if process.is_alive():
                logger.warning("STT worker %s did not exit; terminating", process.pid)
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(3)
            if process.is_alive():
                logger.error(
                    "STT worker %s survived terminate and kill; keeping its handle and its "
                    "pid record so the sweep can still reach it",
                    process.pid,
                )
                # Recorded on the handle: _await may close the worker and raise over this answer.
                self.survived_kill = True
                return False
            try:
                from utils.process_lifetime import forget_pid
                forget_pid(process.pid)
            except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail an unload
                logger.debug("Could not forget STT worker pid %s: %s", process.pid, exc)
        self._process = None
        self._close_queues()
        return True

    def _close_queues(self) -> None:
        for handle in (self._cmd_queue, self._resp_queue):
            try:
                if handle is not None:
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
        self, expected: str, timeout: float, cancel_event: Optional[threading.Event], phase: str
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
                    self.close(graceful_timeout = 0.0)
                    self._raise_cancelled(phase)
            try:
                response = self._resp_queue.get(timeout = _POLL_SECONDS)
            except _queue.Empty:
                if not self.is_alive():
                    raise SttWorkerError(self._crash_message(phase))
                if time.monotonic() >= deadline:
                    if cancel_deadline is not None:
                        # A late cancel still owes the caller the phase's own error (409 load, 499 transcription).
                        self.close(graceful_timeout = 0.0)
                        self._raise_cancelled(phase)
                    self.close()
                    raise SttWorkerError(
                        "The dictation worker stopped responding; try recording again."
                    )
                continue
            except (EOFError, OSError, ValueError) as exc:
                raise SttWorkerError(f"Lost contact with the dictation worker: {exc}") from exc
            if not isinstance(response, dict):
                continue
            self._answered = True
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
            raise SttLoadCancelledError("STT model loading was cancelled so training could start.")
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
                detail += "; the system may have killed it under memory pressure"
        what = "loading the dictation model" if phase == "load" else "transcribing"
        return f"The dictation worker stopped while {what} ({detail})."


class InProcessWhisperEngine:
    """The pre-spawn engine, kept for hosts where no child can be created.

    A sandboxed deployment or a frozen POSIX build may refuse spawn outright,
    and dictation worked there before this module existed, so it still has to.
    Presents the same surface as WhisperWorker, so the sidecar cannot tell the
    two apart.

    CPU only, and deliberately so: the reason the model moved out of process is
    that an accelerator context is never given back while the process holding
    it lives. A CPU load in the backend takes no context, so this fallback
    costs the backend nothing permanent.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self.device: Optional[str] = None
        self.generation_config = SimpleNamespace(is_multilingual = None)

    def start(
        self,
        snapshot_path: str,
        device: str,
        dtype_name: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        if device != "cpu":
            raise SttWorkerError("The in-process dictation fallback runs on the CPU only.")
        model, processor = load_whisper(str(snapshot_path), "cpu", dtype_name, cancel_event)
        self._model = model
        self._processor = processor
        self.device = "cpu"
        generation_config = getattr(model, "generation_config", None)
        is_multilingual = getattr(generation_config, "is_multilingual", None)
        self.generation_config = SimpleNamespace(
            is_multilingual = is_multilingual if isinstance(is_multilingual, bool) else None,
            supports_timestamps = _supports_segment_timestamps(generation_config),
        )
        logger.info("STT model loaded in process on the CPU from %s", snapshot_path)

    def transcribe_window(
        self,
        pcm: bytes,
        generate_kwargs: dict,
        cancel_event: Optional[threading.Event] = None,
    ) -> tuple[str, int]:
        if self._model is None:
            raise SttWorkerError("The dictation worker has no model loaded.")
        result = transcribe_window(
            self._model, self._processor, pcm, dict(generate_kwargs), cancel_event
        )
        if cancel_event is not None and cancel_event.is_set():
            from core.inference.stt_sidecar import SttTranscriptionCancelledError
            raise SttTranscriptionCancelledError("Transcription cancelled.")
        return result

    def is_alive(self) -> bool:
        return self._model is not None

    def cancel(self) -> None:
        """Nothing to signal: the caller's own event stops the generation."""

    def close(self, graceful_timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> bool:
        self._model = None
        self._processor = None
        self.device = None
        return True
