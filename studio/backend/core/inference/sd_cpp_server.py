# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent ``sd-server`` (stable-diffusion.cpp) process manager.

The native diffusion tier used to shell out to one-shot ``sd-cli`` per image, which
reloaded the multi-GB GGUF from disk every generation. ``sd-server`` (the upstream
``examples/server`` target) loads the model once at spawn and serves many generations
over HTTP, exactly like the chat backend's persistent ``llama-server``. This manager
owns ONLY the process + HTTP lifecycle; the backend (``sd_cpp_backend.py``) still owns
asset resolution, request validation, and the public Studio surface.

Shape mirrors ``core/rag/embed_llama_server.py``:
  * ``start``      -- pick a free loopback port, spawn the server (model loads here),
                      drain stdout on a daemon thread, poll until ready.
  * ``img_gen``    -- POST ``/sdcpp/v1/img_gen`` (the whole batch in one request),
                      poll the async job to a terminal state, return image bytes.
  * ``stop``       -- SIGTERM -> wait -> SIGKILL, join the drain thread (idempotent).

Readiness is real: upstream ``main.cpp`` loads the model BEFORE it binds the port and
prints ``listening on:``, so a 200 from ``GET /v1/models`` (a trivial handler) means the
model is loaded; a load failure exits the process before listening and is surfaced with
the captured log tail. (The richer ``/sdcpp/v1/capabilities`` handler can block in some
builds, so it is not used for readiness.) The job JSON has no per-step field, so step progress is recovered by
parsing the server's stdout (the same ``N/M`` lines ``sd-cli`` emits), routed to the
active generation's callback.

Import-light on purpose (no torch / diffusers / PIL), so selecting the native tier on
a CPU box never drags the GPU stack into the process.
"""

from __future__ import annotations

import atexit
import base64
import logging
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import httpx

from core.inference.sd_cpp_args import SdCppModelFiles, build_sd_cpp_server_command
from core.inference.sd_cpp_engine import SdCppCancelled, runtime_env
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import adopt_pid, child_popen_kwargs, forget_pid
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# httpx transport errors meaning "the server is gone / connection refused" -- treated
# as "not ready yet" while polling readiness, and as a fatal "server died" mid-request.
_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)

# Readiness probe. Upstream binds the port only AFTER the model is loaded, so any 200
# means ready. We use /v1/models (a trivial, always-fast handler) rather than
# /sdcpp/v1/capabilities: the capabilities handler can block in some builds (it enumerates
# model metadata), which would stall readiness even though the server is up.
_READY_PATH = "/v1/models"
# Native async sdcpp API.
_IMG_GEN_PATH = "/sdcpp/v1/img_gen"
_JOBS_PATH = "/sdcpp/v1/jobs"

_TERMINAL_OK = "completed"
_TERMINAL_FAIL = "failed"
_TERMINAL_CANCELLED = "cancelled"

# After a cancel is requested, how long to let the server reflect it in job status before
# abandoning the poll. The native cancel is best-effort, so without this cap a server that
# ignores/loses the cancel would keep this call (and the backend's generate lock) alive
# until the job finishes naturally, blocking a superseding load from swapping the model.
_CANCEL_GRACE_S = 5.0


class SdCppServer:
    """A resident ``sd-server`` subprocess plus the HTTP client that drives it."""

    def __init__(
        self,
        binary: str,
        *,
        host: str = "127.0.0.1",
    ) -> None:
        self.binary = binary
        self.host = host
        self.port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        # Fixed-size, thread-safe tail buffer: the drain thread appends while lifecycle /
        # request threads read it for diagnostics, so a deque(maxlen) is safer and cheaper
        # than a list with manual slicing.
        self._tail: deque[str] = deque(maxlen = 200)
        self._stdout_thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()
        # Set (lock-free) by stop() so a blocking start()/readiness wait can be aborted
        # promptly without waiting on the lifecycle lock start() holds.
        self._abort = threading.Event()
        # Set for the duration of a generation so the continuous stdout drain can feed
        # the active request's step-progress callback; cleared in img_gen's finally.
        self._step_listener: Optional[Callable[[str], None]] = None
        self._client = httpx.Client(timeout = 30.0)
        self._scratch_dir: Optional[str] = None
        self._stopped = False
        atexit.register(self.stop)

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def start(
        self,
        files: SdCppModelFiles,
        *,
        vae_format: Optional[str] = None,
        offload: Optional[list[str]] = None,
        native_speed: Optional[str] = None,
        threads: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
        startup_timeout: float = 600.0,
    ) -> None:
        """Spawn the server (which loads the model) and block until it is ready.

        Raises ``RuntimeError`` (with the captured log tail) if the process exits during
        startup or never answers within ``startup_timeout``. Holds the lifecycle lock so
        a concurrent start/stop can't interleave.
        """
        with self._lifecycle_lock:
            self._stopped = False
            self._abort.clear()
            port = self._find_free_port()
            # An empty scratch dir for sd-server's LoRA / upscaler / embeddings scans
            # (it recursively iterates them per request and errors on a missing dir).
            self._scratch_dir = tempfile.mkdtemp(prefix = "sdcpp_dirs_")
            cmd = build_sd_cpp_server_command(
                self.binary,
                files,
                host = self.host,
                port = port,
                vae_format = vae_format,
                offload = list(offload or []),
                native_speed = native_speed,
                threads = threads,
                scratch_dir = self._scratch_dir,
                verbose = True,  # sd-server prints the per-step sampling lines we parse
            )
            run_env = runtime_env(self.binary, child_env_without_native_path_secret())
            if env:
                run_env.update(env)
            logger.info("starting sd-server: %s", " ".join(cmd))
            self._tail = []
            self._spawn_error: Optional[Exception] = None
            spawned = threading.Event()

            # Spawn INSIDE the drain thread, which then reads stdout for the process's whole
            # lifetime. child_popen_kwargs() sets PR_SET_PDEATHSIG, which on Linux is bound to
            # the CREATING THREAD -- so the child must be created by a thread that outlives it,
            # or a transient spawner thread ending would kill the server. The drain thread is
            # exactly that long-lived owner; it dies only when the process exits or the
            # interpreter goes away (the case we DO want to reap the GPU-resident server).
            def _own_process() -> None:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        text = True,
                        errors = "replace",
                        env = run_env,
                        **windows_hidden_subprocess_kwargs(),
                        **child_popen_kwargs(),
                    )
                except Exception as exc:  # noqa: BLE001 -- surface the spawn failure to start()
                    self._spawn_error = exc
                    spawned.set()
                    return
                self._process = proc
                self.port = port
                adopt_pid(proc.pid)  # so a global shutdown sweep also reaps it
                spawned.set()
                self._drain_stdout(proc)
                # stdout closed == the process exited; reap it so it is not left a zombie
                # until the next stop()/reload.
                try:
                    proc.wait(timeout = 5)
                except Exception:  # noqa: BLE001
                    pass

            self._stdout_thread = threading.Thread(
                target = _own_process, daemon = True, name = "sd-server-owner"
            )
            self._stdout_thread.start()
            spawned.wait()
            if self._spawn_error is not None:
                self._dispose()
                raise RuntimeError(f"failed to spawn sd-server: {self._spawn_error}")
            if not self._wait_ready(startup_timeout):
                tail = "\n".join(list(self._tail)[-30:])
                aborted = self._abort.is_set()
                self._kill_locked()
                self._dispose()
                if aborted:
                    raise SdCppCancelled("sd-server startup was cancelled.")
                raise RuntimeError("sd-server failed to become ready. Last output:\n" + tail[:2000])

    def _wait_ready(
        self,
        timeout: float,
        interval: float = 0.5,
    ) -> bool:
        """Poll ``/v1/models`` until 200; bail early if the process exits.

        Upstream binds the port only AFTER the model is loaded, so a 200 here is a true
        ready signal (no half-loaded race)."""
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}{_READY_PATH}"
        while time.monotonic() < deadline:
            # A concurrent stop() (unload / superseding load) sets _abort so this wait can
            # bail without holding the model-load hostage for the full startup_timeout.
            if self._abort.is_set():
                logger.info("sd-server startup aborted before ready")
                return False
            if not self.is_alive():
                code = None if self._process is None else self._process.returncode
                logger.error("sd-server exited early during load (code %s)", code)
                return False
            try:
                if self._client.get(url, timeout = 2.0).status_code == 200:
                    return True
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                pass
            time.sleep(interval)
        logger.error("sd-server readiness timed out after %ss", timeout)
        return False

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        """Drain stdout so the pipe never deadlocks; keep a tail for diagnostics and
        feed each line to the active generation's step callback."""
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                self._tail.append(line)  # deque(maxlen) discards the oldest automatically
                logger.debug("[sd-server] %s", line)
                cb = self._step_listener
                if cb is not None:
                    try:
                        cb(line)
                    except Exception:  # noqa: BLE001 -- a progress callback must never break drain
                        pass
        except Exception:  # noqa: BLE001 -- drain thread must never raise (pipe closed at teardown)
            pass

    def stop(self) -> None:
        """Terminate the server (SIGTERM -> SIGKILL), join the drain, and release the HTTP
        client + atexit handler. Idempotent."""
        # Signal abort BEFORE contending for the lifecycle lock: a concurrent start() holds
        # that lock for the whole (up to startup_timeout) readiness wait, so setting the
        # event lets that wait bail immediately instead of stop() blocking behind it.
        self._abort.set()
        self._stopped = True
        with self._lifecycle_lock:
            self._kill_locked()
            self._dispose()

    def _dispose(self) -> None:
        """Release per-instance resources (on stop / failed start). The backend never
        reuses a disposed server, so this closes the pooled httpx client and drops the
        atexit handler that would otherwise pin every reloaded instance for the session."""
        try:
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._scratch_dir:
            shutil.rmtree(self._scratch_dir, ignore_errors = True)
            self._scratch_dir = None

    def _kill_locked(self) -> None:
        proc = self._process
        if proc is None:
            return
        pid = proc.pid
        try:
            proc.terminate()
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            logger.warning("sd-server did not exit on SIGTERM; killing")
            try:
                proc.kill()
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001 -- best-effort teardown
                pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("error terminating sd-server: %s", exc)
        finally:
            forget_pid(pid)
            self._process = None
            self.port = None
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout = 2)
                self._stdout_thread = None

    # ── generation ───────────────────────────────────────────────────────────

    def img_gen(
        self,
        payload: dict[str, Any],
        *,
        on_step: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.4,
        submit_timeout: float = 60.0,
        total_timeout: float = 1800.0,
    ) -> list[bytes]:
        """Submit one async ``img_gen`` job, poll it to completion, return image bytes.

        ``on_step`` receives each server stdout line (for the step bar). ``cancel_event``,
        when set, cancels the job via the native endpoint and raises ``SdCppCancelled``.
        Raises ``RuntimeError`` on submit/poll failures (including the server dying), with
        the log tail attached.
        """
        # If the server was already stopped for a cancel/unload/superseding load that set
        # the cancel event before this submit began, report it as a cancellation (which the
        # route maps to a client-state 409) rather than a generic "server died" 500.
        if self._stopped or not self.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                raise SdCppCancelled("sd-server generation was cancelled.")
            raise RuntimeError("sd-server is not running.")

        self._step_listener = on_step
        job_id: Optional[str] = None
        try:
            # Submit -> 202 Accepted + job id.
            try:
                resp = self._client.post(
                    f"{self.base_url}{_IMG_GEN_PATH}", json = payload, timeout = submit_timeout
                )
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as exc:
                raise RuntimeError(self._died_message("img_gen submit", exc)) from exc
            if resp.status_code == 429:
                raise RuntimeError("sd-server job queue is full (HTTP 429).")
            if resp.status_code not in (200, 202):
                raise RuntimeError(
                    f"sd-server img_gen submit -> {resp.status_code}: {resp.text[:500]}"
                )
            try:
                job = resp.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"sd-server img_gen returned a non-JSON submit response: {exc}"
                ) from exc
            if not isinstance(job, dict):
                raise RuntimeError(
                    f"sd-server img_gen returned an unexpected submit response type: {type(job)}"
                )
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"sd-server img_gen returned no job id: {job}")

            # Poll the job to a terminal state.
            deadline = time.monotonic() + total_timeout
            cancel_sent_at: Optional[float] = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    if cancel_sent_at is None:
                        self.cancel(job_id)
                        cancel_sent_at = time.monotonic()
                    elif time.monotonic() - cancel_sent_at > _CANCEL_GRACE_S:
                        # The best-effort cancel was not reflected in job status within the
                        # grace window; abandon the poll so the caller can stop the server
                        # instead of holding the generate lock until the job finishes.
                        raise SdCppCancelled("sd-server generation was cancelled.")
                if not self.is_alive():
                    # If we're unwinding a cancel (e.g. unload killed the server), surface a
                    # clean cancellation rather than a generic "server died" error.
                    if cancel_event is not None and cancel_event.is_set():
                        raise SdCppCancelled("sd-server generation was cancelled.")
                    raise RuntimeError(self._died_message("img_gen poll", None))
                if time.monotonic() > deadline:
                    self.cancel(job_id)
                    raise RuntimeError(f"sd-server generation timed out after {total_timeout}s")
                try:
                    jr = self._client.get(f"{self.base_url}{_JOBS_PATH}/{job_id}", timeout = 10.0)
                except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                    time.sleep(poll_interval)
                    continue
                if jr.status_code in (404, 410):
                    raise RuntimeError(f"sd-server job {job_id} is gone (HTTP {jr.status_code}).")
                if jr.status_code != 200:
                    time.sleep(poll_interval)
                    continue
                try:
                    jd = jr.json()
                except ValueError as exc:
                    raise RuntimeError(f"sd-server job status was not JSON: {exc}") from exc
                if not isinstance(jd, dict):
                    raise RuntimeError(
                        f"sd-server job status returned an unexpected response type: {type(jd)}"
                    )
                status = jd.get("status")
                if status == _TERMINAL_OK:
                    return self._decode_images(jd)
                if status == _TERMINAL_FAIL:
                    err = jd.get("error") or {}
                    raise RuntimeError(
                        "sd-server generation failed: "
                        f"{err.get('code', 'error')}: {err.get('message', '')}".strip()
                    )
                if status == _TERMINAL_CANCELLED:
                    raise SdCppCancelled("sd-server generation was cancelled.")
                time.sleep(poll_interval)
        finally:
            self._step_listener = None

    def cancel(self, job_id: str) -> None:
        """Best-effort native cancel of an in-flight job."""
        try:
            self._client.post(f"{self.base_url}{_JOBS_PATH}/{job_id}/cancel", timeout = 5.0)
        except Exception:  # noqa: BLE001 -- cancel is best-effort
            pass

    @staticmethod
    def _decode_images(job: dict[str, Any]) -> list[bytes]:
        # Defensive against an unexpected response shape (a misbehaving/older server):
        # verify each level is the type we index before calling dict/list methods.
        result = job.get("result") if isinstance(job, dict) else None
        images = result.get("images") if isinstance(result, dict) else None
        items = [it for it in images if isinstance(it, dict)] if isinstance(images, list) else []
        out: list[bytes] = []
        for item in sorted(items, key = lambda d: d.get("index", 0)):
            b64 = item.get("b64_json")
            if not b64:
                continue
            try:
                out.append(base64.b64decode(b64))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"sd-server returned an undecodable image: {exc}") from exc
        if not out:
            raise RuntimeError("sd-server completed the job but returned no images.")
        return out

    def _died_message(self, where: str, exc: Optional[Exception]) -> str:
        tail = "\n".join(list(self._tail)[-20:])
        base = f"sd-server connection lost during {where}"
        if not self.is_alive():
            code = None if self._process is None else self._process.returncode
            base += f" (process exited, code {code})"
        if exc is not None:
            base += f": {exc}"
        if tail:
            base += "\nLast output:\n" + tail[:1500]
        return base
