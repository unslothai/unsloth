# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owned llama.cpp adapter for Clef's text-only SystemOne endpoint."""

from __future__ import annotations

import math
import mmap
import re
import secrets
import subprocess
import threading
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from .owned_runtime import (
    CANCEL_GRACE_S,
    LOAD_WAIT_S,
    RUN_WAIT_S,
    ClefWorkerCancelled,
    ClefWorkerError,
    ClefWorkerInputError,
)

_ROUTE = b"/v1/systemone"
_POLL_S = 0.1
NATIVE_MAX_CONTEXT_TOKENS = 2_048


@lru_cache(maxsize = 16)
def _mapped_contains(path: str, mtime_ns: int, size: int, marker: bytes) -> bool:
    try:
        with (
            open(path, "rb") as source,
            mmap.mmap(source.fileno(), 0, access = mmap.ACCESS_READ) as data,
        ):
            return size > 0 and data.find(marker) >= 0
    except OSError:
        return False


def _contains(path: Path, marker: bytes) -> bool:
    try:
        path = path.resolve(strict = True)
        stat = path.stat()
    except OSError:
        return False
    return stat.st_size > 0 and _mapped_contains(str(path), stat.st_mtime_ns, stat.st_size, marker)


def supports_systemone(binary: str | Path) -> bool:
    """Inspect the executable and, for shared builds, its linked server implementation."""
    path = Path(binary).resolve()
    if _contains(path, _ROUTE):
        return True
    for name in ("libllama-server-impl.so", "libllama-server-impl.dylib", "llama-server-impl.dll"):
        if _contains(path, name.encode()):
            return any(
                _contains(directory / name, _ROUTE)
                for directory in (path.parent, path.parent.parent / "lib")
            )
    return False


def _resolve_binary() -> str | None:
    from core.inference.llama_cpp import LlamaCppBackend
    binary = LlamaCppBackend._find_llama_server_binary()
    return (LlamaCppBackend._exec_path_for_launch(binary) or binary) if binary else None


def native_availability() -> dict[str, bool | str | None]:
    binary = _resolve_binary()
    if binary is None:
        return {"available": False, "reason": "llama-server is not installed.", "binary": None}
    if not supports_systemone(binary):
        return {
            "available": False,
            "reason": "This llama-server build does not support /v1/systemone.",
            "binary": binary,
        }
    return {"available": True, "reason": None, "binary": binary}


def _request_gap(questions: object, images: object) -> str | None:
    if images:
        return "Unsloth's native Clef bundle does not support images without a projector; use the PyTorch runtime."
    if not isinstance(questions, Mapping):
        return "questions must be an object"
    for question in questions.values():
        if not isinstance(question, Mapping):
            return "questions must map string ids to objects"
        if question.get("instructions") == "":
            return "Native llama.cpp Clef cannot serve empty instructions."
        if (
            question.get("type") == "score"
            and isinstance(question.get("criteria"), list)
            and len(question["criteria"]) < 2
        ):
            return "Native llama.cpp Clef scores require at least two criteria."
    return None


def supports_request(questions: object, images: object) -> bool:
    return _request_gap(questions, images) is None


def _wire_questions(questions: object, images: object) -> dict[str, dict[str, Any]]:
    if gap := _request_gap(questions, images):
        raise ClefWorkerInputError(gap)
    result = {}
    for key, question in questions.items():  # type: ignore[union-attr]
        if not isinstance(key, str):
            raise ClefWorkerInputError("questions must map string ids to objects")
        copied = dict(question)
        if copied.get("instructions") is None:
            copied["instructions"] = key
        result[key] = copied
    return result


def _detail(response: Any) -> str:
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, Mapping):
        value = body.get("error") or body.get("detail") or body.get("message")
        if isinstance(value, Mapping):
            value = value.get("message") or value.get("detail")
        if isinstance(value, str) and value:
            return value[:1000]
    return str(getattr(response, "text", "")).strip()[:1000] or f"HTTP {response.status_code}"


def _validate_result(
    data: object, checkpoint: Any, questions: Mapping[str, object]
) -> dict[str, Any]:
    if not isinstance(data, Mapping) or not isinstance(data.get("answers"), Mapping):
        raise ClefWorkerError("Native llama.cpp Clef returned an invalid decision response.")
    try:
        for name in questions:
            answer = data["answers"].get(name)
            if not isinstance(answer, Mapping):
                raise ValueError(f"missing answer for {name}")
            values = answer.get("probabilities")
            values = values.values() if isinstance(values, Mapping) else [answer.get("noul")]
            if any(isinstance(value, bool) for value in values):
                raise ValueError(f"invalid probabilities for {name}")
            values = [float(value) for value in values]
            if not values or any(
                not math.isfinite(value) or not 0 <= value <= 1 for value in values
            ):
                raise ValueError(f"invalid probabilities for {name}")
            if "probabilities" in answer and not math.isclose(
                sum(values), 1, rel_tol = 1e-4, abs_tol = 1e-4
            ):
                raise ValueError(f"probabilities for {name} do not sum to one")
    except (TypeError, ValueError) as exc:
        raise ClefWorkerError(
            f"Native llama.cpp Clef returned invalid probabilities: {exc}"
        ) from None
    result = dict(data)
    result["model"] = checkpoint.name
    return result


class NativeWorker:
    """One private loopback llama-server process; the parent owns selection/fallback."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._client: httpx.Client | None = None
        self._port: int | None = None
        self._api_key: str | None = None
        self._cancelled, self._lock = threading.Event(), threading.RLock()
        self._external_cancel: threading.Event | None = None
        self._closed = False
        self.device: str | None = None
        self.gpu_available: bool | None = None

    def start(
        self,
        snapshot_path: Path,
        checkpoint: Any,
        requested_device: str,
        cancelled: threading.Event,
    ) -> None:
        if self._is_cancelled(cancelled):
            raise ClefWorkerCancelled("Native Clef model loading was cancelled.")
        if not snapshot_path.is_file():
            raise ClefWorkerError("The native Clef GGUF is not available in the local model cache.")
        availability = native_availability()
        if not availability["available"]:
            raise ClefWorkerError(str(availability["reason"]))
        binary = str(availability["binary"])
        from core.inference.llama_cpp import LlamaCppBackend
        from utils.native_path_leases import child_env_without_native_path_secret
        from utils.process_lifetime import (
            adopt_pid,
            child_popen_kwargs,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )
        from utils.subprocess_compat import windows_hidden_subprocess_kwargs

        if is_process_shutting_down():
            raise ClefWorkerCancelled("Studio is shutting down; native Clef was not started.")
        use_gpu, port, key = (
            requested_device == "gpu",
            LlamaCppBackend._find_free_port(),
            secrets.token_urlsafe(32),
        )
        env = child_env_without_native_path_secret(
            LlamaCppBackend._llama_server_env_for_binary(binary)
        )
        device = None
        if use_gpu:
            devices = LlamaCppBackend._enumerated_gpu_devices(binary, env)
            if not devices:
                raise ClefWorkerError(
                    "The native llama-server has no usable GPU under the reserved device visibility."
                )
            device = devices[0]
        else:
            env.update(CUDA_VISIBLE_DEVICES = "", HIP_VISIBLE_DEVICES = "-1")
            LlamaCppBackend._clear_device_placement_env(env)
        command = [
            binary,
            "-m",
            str(snapshot_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            key,
            "--parallel",
            "1",
            "-c",
            str(NATIVE_MAX_CONTEXT_TOKENS),
            "-b",
            str(NATIVE_MAX_CONTEXT_TOKENS),
            "-ub",
            str(NATIVE_MAX_CONTEXT_TOKENS),
            "-ngl",
            "-1" if use_gpu else "0",
            *(["--device", device] if device else []),
        ]
        self._external_cancel, self._port, self._api_key = cancelled, port, key
        self._client = httpx.Client(timeout = LOAD_WAIT_S, trust_env = False)
        try:
            self._process = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    command,
                    stdin = subprocess.DEVNULL,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL,
                    env = env,
                    **windows_hidden_subprocess_kwargs(),
                    **child_popen_kwargs(),
                )
            )
            adopt_pid(self._process.pid)
            if is_process_shutting_down() or self._is_cancelled(cancelled):
                raise ClefWorkerCancelled("Native Clef model loading was cancelled.")
            self._wait_for_health(cancelled)
        except BaseException:
            self.close()
            raise
        self.device, self.gpu_available = (device, True) if device else ("cpu", False)

    def decide(
        self, checkpoint: Any, state: Any, questions: dict[str, dict[str, Any]], images: list[bytes]
    ) -> dict[str, Any]:
        wire_questions = _wire_questions(questions, images)
        if self._is_cancelled():
            raise ClefWorkerCancelled("Native Clef worker was cancelled.")
        if not self.is_alive() or not all((self._client, self._port, self._api_key)):
            raise ClefWorkerError("Native Clef worker is not running.")
        done, outcome = threading.Event(), {}

        def post() -> None:
            try:
                outcome["response"] = self._client.post(
                    f"http://127.0.0.1:{self._port}/v1/systemone",
                    json = {"model": checkpoint.name, "state": state, "questions": wire_questions},
                    headers = {"Authorization": f"Bearer {self._api_key}"},
                    timeout = RUN_WAIT_S,
                )
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                done.set()

        threading.Thread(target = post, name = "native-clef-request", daemon = True).start()
        while not done.wait(_POLL_S):
            if self._is_cancelled():
                self.close()
                raise ClefWorkerCancelled("Native Clef decision was cancelled.")
        if self._is_cancelled():
            raise ClefWorkerCancelled("Native Clef decision was cancelled.")
        if error := outcome.get("error"):
            raise ClefWorkerError(
                f"Native Clef request failed: {type(error).__name__}: {error}"
            ) from error
        response, status = outcome["response"], int(outcome["response"].status_code)
        if not 200 <= status < 300:
            detail = _detail(response)
            # llama.cpp reports this input-bound failure as HTTP 500.
            overflow = status == 500 and re.fullmatch(
                r"input \(\d+ tokens\) is too large to process\. increase the physical batch size "
                r"\(current batch size: \d+\)",
                detail,
            )
            if overflow or (status not in {401, 403} and 400 <= status < 500):
                raise ClefWorkerInputError(detail)
            raise ClefWorkerError(f"Native Clef server returned HTTP {status}: {detail}")
        try:
            return _validate_result(response.json(), checkpoint, wire_questions)
        except ValueError as exc:
            raise ClefWorkerError("Native Clef server returned invalid JSON.") from exc

    def cancel(self) -> None:
        self._cancelled.set()
        self.close()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def close(self, graceful_timeout: float = 0.0) -> bool:
        with self._lock:
            self._closed = True
            self._cancelled.set()
            client, self._client, process = self._client, None, self._process
            try:
                if client is not None:
                    client.close()
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout = max(0.0, graceful_timeout))
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout = CANCEL_GRACE_S)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            if process is not None and process.poll() is None:
                return False
            if process is not None:
                from utils.process_lifetime import forget_pid
                forget_pid(process.pid)
            self._process, self._port, self._api_key = None, None, None
            return True

    def _wait_for_health(self, cancelled: threading.Event) -> None:
        deadline = time.monotonic() + LOAD_WAIT_S
        while time.monotonic() < deadline:
            if self._is_cancelled(cancelled):
                raise ClefWorkerCancelled("Native Clef model loading was cancelled.")
            if not self.is_alive():
                raise ClefWorkerError("Native Clef server exited during startup.")
            try:
                response = self._client.get(
                    f"http://127.0.0.1:{self._port}/health",
                    headers = {"Authorization": f"Bearer {self._api_key}"},
                    timeout = 2.0,
                )
                if response.status_code == 200:
                    return
                if response.status_code not in (503,):
                    raise ClefWorkerError(
                        f"Native Clef health returned HTTP {response.status_code}."
                    )
            except httpx.TransportError:
                pass
            time.sleep(_POLL_S)
        raise ClefWorkerError("Native Clef server did not become healthy in time.")

    def _is_cancelled(self, extra: threading.Event | None = None) -> bool:
        if self._closed or self._cancelled.is_set() or (extra is not None and extra.is_set()):
            return True
        if self._external_cancel is not None and self._external_cancel.is_set():
            return True
        from utils.process_lifetime import is_process_shutting_down

        return is_process_shutting_down()
