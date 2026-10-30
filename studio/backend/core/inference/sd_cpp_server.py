# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent sd-server; ready on 200 from GET /v1/models (not capabilities, which can block);
routes POST /sdcpp/v1/img_gen and POST /sdcpp/v1/vid_gen; import-light, no torch/diffusers/PIL"""

from __future__ import annotations

import atexit
import base64
import logging
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import httpx

from core.inference.sd_cpp_args import SdCppModelFiles, build_sd_cpp_server_command
from core.inference.sd_cpp_engine import (
    NATIVE_GENERATION_TIMEOUT_S,
    SdCppCancelled,
    _diagnostic_tail,
    _sd_cpp_command_for_log,
    _sd_cpp_command_summary,
    _verbose_native_logs,
    binary_accelerator,
    runtime_env,
)
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    is_process_shutting_down,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

# "server gone / connection refused": not-ready while polling, fatal mid-request.
_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)

# Port binds only after the model loads; the capabilities endpoint can block.
_READY_PATH = "/v1/models"
_IMG_GEN_PATH = "/sdcpp/v1/img_gen"
_VID_GEN_PATH = "/sdcpp/v1/vid_gen"
_JOBS_PATH = "/sdcpp/v1/jobs"

# Compute-buffer lines excluded: they vary per generation, params stay until exit.
_TOTAL_PARAMS_VRAM_RE = re.compile(
    r"total params memory size\s*=\s*[0-9.]+\s*MB\s*\(\s*VRAM\s+([0-9]+(?:\.[0-9]+)?)\s*MB\b",
    re.IGNORECASE,
)

_TERMINAL_OK = "completed"
_TERMINAL_FAIL = "failed"
_TERMINAL_CANCELLED = "cancelled"

# Grace for the best-effort native cancel to show in job status; without the cap a lost cancel would hold the generate
# lock until the job ends.
_CANCEL_GRACE_S = 5.0


def _socket_owner_is_always_named() -> bool:
    """Windows names every socket's owner: an unnamed one there is a psutil quirk (#15022)."""
    return sys.platform == "win32"


def _has_ancestor(
    pid: int,
    ancestor_pid: int,
    *,
    max_depth: int = 8,
) -> bool:
    """Walks ancestors, since the listener may be a wrapper or shell child of our spawned process."""
    try:
        import psutil
        proc = psutil.Process(pid)
        for _ in range(max_depth):
            proc = proc.parent()
            if proc is None:
                return False
            if proc.pid == ancestor_pid:
                return True
    except Exception:  # noqa: BLE001 -- gone / no permission: treat as unknown
        return False
    return False


class SdCppServerUnsupported(RuntimeError):
    """The sd-server has no route / mode for this job kind; nothing ran, so the caller can fall back to sd-cli."""


def _route_unsupported(status_code: int, text: str, kind: str) -> bool:
    # 404: no route in this build; 400 "does not support": the loaded model lacks the mode (runtime.cpp).
    if status_code == 404:
        return True
    return status_code == 400 and f"does not support {kind}" in (text or "")


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
        self._resident_params_vram_gb: Optional[float] = None
        # Bounded tail buffer shared by the drain thread (appends) and readers (diagnostics).
        self._tail: deque[str] = deque(maxlen = 200)
        self._stdout_thread: Optional[threading.Thread] = None
        self._lifecycle_lock = threading.Lock()
        # Set lock-free by stop() so a blocking start()/readiness wait bails promptly.
        self._abort = threading.Event()
        # Set during a generation so the stdout drain feeds the step-progress callback.
        self._step_listener: Optional[Callable[[str], None]] = None
        # trust_env=False: loopback-only client must not route through HTTP(S)_PROXY.
        self._client = httpx.Client(timeout = 30.0, trust_env = False)
        self._scratch_dir: Optional[str] = None
        self._stopped = False
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def lora_dir(self) -> Optional[str]:
        """Server rescans this dir per request; reference staged files via the lora field, not
        prompt tags."""
        return self._scratch_dir

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
        extra_args: Optional[list[str]] = None,
    ) -> None:
        """Spawns the server and blocks until ready under the lifecycle lock; extra_args go last so they win, which pins the CPU-backend restart off the GPU."""
        with self._lifecycle_lock:
            # A stop() that raced in before the lock already closed the client; do not leak a process.
            if self._stopped or self._abort.is_set():
                raise SdCppCancelled("sd-server start was cancelled before launch.")
            self._abort.clear()
            port = self._find_free_port()
            # empty scratch dir for sd-server's LoRA/upscaler/embeddings scans (errors if missing)
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
                extra_args = list(extra_args or []),
                accelerator = binary_accelerator(self.binary),
            )
            run_env = runtime_env(self.binary, child_env_without_native_path_secret())
            if env:
                run_env.update(env)
            if _verbose_native_logs():
                logger.info("starting sd-server: %s", " ".join(_sd_cpp_command_for_log(cmd)))
            else:
                logger.info(
                    "starting sd-server: %s",
                    _sd_cpp_command_summary(cmd, default_mode = "server"),
                )
            # Clear in place; reassigning [] would drop the maxlen bound and grow unbounded.
            self._tail.clear()
            self._resident_params_vram_gb = None
            self._spawn_error: Optional[Exception] = None
            spawned = threading.Event()

            # Spawn in the drain thread: PR_SET_PDEATHSIG is bound to the creating thread on Linux.
            def _own_process() -> None:
                # Nothing in _graceful_shutdown unloads this engine, so gate every spawn on shutdown.
                if is_process_shutting_down():
                    self._spawn_error = RuntimeError(
                        "Unsloth is shutting down; not starting sd-server"
                    )
                    spawned.set()
                    return
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout = subprocess.PIPE,
                        stderr = subprocess.STDOUT,
                        text = True,
                        encoding = "utf-8",
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
                # Recheck after recording: covers the window between the gate and adopt_pid.
                if is_process_shutting_down():
                    logger.info("shutdown began during the spawn; killing the new sd-server")
                    try:
                        proc.kill()
                        proc.wait(timeout = 5)
                    except Exception:  # noqa: BLE001 - the reap is best-effort
                        pass
                    # Keep the record if kill/wait raised: it is the last thing that could reap the child.
                    if proc.poll() is not None:
                        forget_pid(proc.pid)
                    else:
                        logger.warning(
                            "sd-server pid %s survived the shutdown kill; leaving it adopted",
                            proc.pid,
                        )
                    self._process = None
                    self._spawn_error = RuntimeError(
                        "Unsloth is shutting down; not starting sd-server"
                    )
                    spawned.set()
                    return
                spawned.set()
                self._drain_stdout(proc)
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
                tail = _diagnostic_tail(self._tail, keep = 30)
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
        """A 200 is a true ready signal: upstream binds the port only after the model has loaded."""
        deadline = time.monotonic() + timeout
        url = f"{self.base_url}{_READY_PATH}"
        while time.monotonic() < deadline:
            # stop() sets _abort so the wait bails early instead of holding the load for startup_timeout.
            if self._abort.is_set():
                logger.info("sd-server startup aborted before ready")
                return False
            if not self.is_alive():
                code = None if self._process is None else self._process.returncode
                logger.error("sd-server exited early during load (code %s)", code)
                return False
            try:
                # Connection: close keeps a probe answered by a squatter out of the generation pool.
                resp = self._client.get(url, headers = {"Connection": "close"}, timeout = 2.0)
                if resp.status_code == 200 and self._port_is_ours():
                    return True
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                pass
            time.sleep(interval)
        logger.error("sd-server readiness timed out after %ss", timeout)
        return False

    def _port_is_ours(self) -> bool:
        """Best-effort: fails only when the port listener is definitely another pid, since another local server can answer /v1/models 200; unknown passes."""
        proc = self._process
        if proc is None or proc.pid is None:
            return True
        # A child that already exited cannot be the server that answered.
        poll = getattr(proc, "poll", None)
        if callable(poll) and poll() is not None:
            return False
        try:
            import psutil
        except Exception:  # noqa: BLE001 -- optional dependency
            return True
        windows = _socket_owner_is_always_named()

        def _listens(conn) -> bool:
            laddr = getattr(conn, "laddr", None)
            if not laddr or getattr(laddr, "port", None) != self.port:
                return False
            if conn.status == psutil.CONN_LISTEN:
                return True
            # psutil maps a TCP state it cannot decode to NONE, seen on live Windows listeners (#15022).
            return (
                windows
                and conn.status == getattr(psutil, "CONN_NONE", "NONE")
                and not getattr(conn, "raddr", None)
                and getattr(conn, "type", socket.SOCK_STREAM) == socket.SOCK_STREAM
            )

        # Another user's socket shows no pid on Linux/macOS: our readable tree not listening = foreign.
        tree_says_foreign = False
        try:
            if not isinstance(proc, subprocess.Popen):
                raise TypeError("not a child we spawned")
            root = psutil.Process(proc.pid)
            tree = [root, *root.children(recursive = True)]
            readable = True
            for member in tree:
                try:
                    read = getattr(member, "net_connections", None) or member.connections
                    if any(_listens(conn) for conn in read(kind = "inet")):
                        return True
                except psutil.NoSuchProcess:
                    continue
                except Exception:  # noqa: BLE001 -- access denied / platform quirk
                    readable = False
            tree_says_foreign = readable and not windows
        except Exception as exc:  # noqa: BLE001 -- platform quirk: fall back below
            if isinstance(exc, getattr(psutil, "NoSuchProcess", ())):
                return False
        try:
            for conn in psutil.net_connections(kind = "inet"):
                if not _listens(conn):
                    continue
                owner = conn.pid
                # pid 0 is the System Idle Process on Windows: psutil's other "unknown owner".
                if owner is None or (windows and owner == 0):
                    continue  # not visible to us; do not punish a healthy start
                if owner == proc.pid or _has_ancestor(owner, proc.pid):
                    return True
                logger.error(
                    "sd-server readiness port %s is held by pid %s, not our child %s; "
                    "refusing to use it",
                    self.port,
                    owner,
                    proc.pid,
                )
                return False
        except Exception as exc:  # noqa: BLE001 -- permissions / platform quirks
            logger.debug("sd-server port ownership check unavailable: %s", exc)
        if tree_says_foreign:
            logger.error(
                "sd-server readiness port %s answered but our child %s is not listening on it; "
                "refusing to use it",
                self.port,
                proc.pid,
            )
            return False
        return True

    def _drain_stdout(self, proc: subprocess.Popen) -> None:
        """Drain stdout so the pipe never deadlocks; keep a tail for diagnostics and
        feed each line to the active generation's step callback."""
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                self._tail.append(line)
                match = _TOTAL_PARAMS_VRAM_RE.search(line)
                if match is not None:
                    value = float(match.group(1)) / 1024.0
                    if value > 0:
                        self._resident_params_vram_gb = value
                logger.debug("[sd-server] %s", line)
                cb = self._step_listener
                if cb is not None:
                    try:
                        cb(line)
                    except Exception:  # noqa: BLE001 -- a progress callback must never break drain
                        pass
        except Exception:  # noqa: BLE001 -- drain thread must never raise (pipe closed at teardown)
            pass

    def reclaimable_params_vram_gb(
        self, physical_gpu_id: Optional[int]
    ) -> Optional[dict[int, float]]:
        """Fails closed (None) when device attribution is missing or the process has exited."""
        process = self._process
        if physical_gpu_id is None or process is None or process.poll() is not None:
            return None
        value = self._resident_params_vram_gb
        if value is None or value <= 0:
            return None
        if self._process is not process or process.poll() is not None:
            return None
        return {int(physical_gpu_id): float(value)}

    def stop(self) -> None:
        """Terminate the server (SIGTERM -> SIGKILL), join the drain, and release the HTTP
        client + atexit handler. Idempotent."""
        # Set abort BEFORE taking the lock so a start() readiness wait bails instead of blocking.
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

    def img_gen(
        self,
        payload: dict[str, Any],
        *,
        on_step: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.4,
        submit_timeout: float = 60.0,
        total_timeout: float = NATIVE_GENERATION_TIMEOUT_S,
    ) -> list[bytes]:
        """A set cancel_event cancels the job natively and raises SdCppCancelled; on_step gets
        stdout lines."""
        return self._run_job(
            "img_gen",
            _IMG_GEN_PATH,
            payload,
            self._decode_images,
            on_step = on_step,
            cancel_event = cancel_event,
            poll_interval = poll_interval,
            submit_timeout = submit_timeout,
            total_timeout = total_timeout,
        )

    def vid_gen(
        self,
        payload: dict[str, Any],
        *,
        on_step: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.4,
        submit_timeout: float = 60.0,
        total_timeout: float = NATIVE_GENERATION_TIMEOUT_S,
    ) -> bytes:
        """``img_gen``'s contract for one ``vid_gen`` job, returning the container bytes (audio muxed); raises
        ``SdCppServerUnsupported`` when the server cannot do video."""
        return self._run_job(
            "vid_gen",
            _VID_GEN_PATH,
            payload,
            self._decode_video,
            on_step = on_step,
            cancel_event = cancel_event,
            poll_interval = poll_interval,
            submit_timeout = submit_timeout,
            total_timeout = total_timeout,
        )

    def _run_job(
        self,
        kind: str,
        path: str,
        payload: dict[str, Any],
        decode: Callable[[dict[str, Any]], Any],
        *,
        on_step: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        poll_interval: float = 0.4,
        submit_timeout: float = 60.0,
        total_timeout: float = NATIVE_GENERATION_TIMEOUT_S,
    ) -> Any:
        """Submits and polls one async job; a cancel raises SdCppCancelled (route 409), not a 500; submit or poll failures raise RuntimeError with the log tail."""
        if self._stopped or not self.is_alive():
            if cancel_event is not None and cancel_event.is_set():
                raise SdCppCancelled("sd-server generation was cancelled.")
            raise RuntimeError("sd-server is not running.")

        self._step_listener = on_step
        job_id: Optional[str] = None
        try:
            try:
                resp = self._client.post(
                    f"{self.base_url}{path}", json = payload, timeout = submit_timeout
                )
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as exc:
                raise RuntimeError(self._died_message(f"{kind} submit", exc)) from exc
            if resp.status_code == 429:
                raise RuntimeError("sd-server job queue is full (HTTP 429).")
            if resp.status_code not in (200, 202):
                if _route_unsupported(resp.status_code, resp.text, kind):
                    raise SdCppServerUnsupported(
                        f"this sd-server does not serve {kind} (HTTP {resp.status_code}): "
                        f"{resp.text[:300]}"
                    )
                raise RuntimeError(
                    f"sd-server {kind} submit -> {resp.status_code}: {resp.text[:500]}"
                )
            try:
                job = resp.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"sd-server {kind} returned a non-JSON submit response: {exc}"
                ) from exc
            if not isinstance(job, dict):
                raise RuntimeError(
                    f"sd-server {kind} returned an unexpected submit response type: {type(job)}"
                )
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"sd-server {kind} returned no job id: {job}")

            deadline = time.monotonic() + total_timeout
            cancel_sent_at: Optional[float] = None
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    if cancel_sent_at is None:
                        self.cancel(job_id)
                        cancel_sent_at = time.monotonic()
                    elif time.monotonic() - cancel_sent_at > _CANCEL_GRACE_S:
                        # sd-server will not interrupt the job; stop it so the GPU is freed with the generate lock.
                        self.stop()
                        raise SdCppCancelled("sd-server generation was cancelled.")
                if not self.is_alive():
                    # Unwinding a cancel (e.g. unload killed the server): clean cancellation.
                    if cancel_event is not None and cancel_event.is_set():
                        raise SdCppCancelled("sd-server generation was cancelled.")
                    raise RuntimeError(self._died_message(f"{kind} poll", None))
                if time.monotonic() > deadline:
                    # sd-server will not interrupt an in-flight job; the backend reloads on the next generate.
                    self.cancel(job_id)
                    self.stop()
                    raise RuntimeError(f"sd-server generation timed out after {total_timeout}s")
                try:
                    jr = self._client.get(f"{self.base_url}{_JOBS_PATH}/{job_id}", timeout = 10.0)
                except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                    time.sleep(poll_interval)
                    continue
                except RuntimeError as exc:
                    # A concurrent stop() closes the client, giving a plain RuntimeError, not a transport error.
                    if cancel_event is not None and cancel_event.is_set():
                        raise SdCppCancelled("sd-server generation was cancelled.") from exc
                    raise
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
                    return decode(jd)
                if status == _TERMINAL_FAIL:
                    err = jd.get("error") or {}
                    message = (
                        "sd-server generation failed: "
                        f"{err.get('code', 'error')}: {err.get('message', '')}".strip()
                    )
                    if kind == "vid_gen":
                        # The job error is generic; the cause (OOM, abort) is only in the log.
                        tail = _diagnostic_tail(self._tail)
                        if tail:
                            message += "\nLast output:\n" + tail[:1500]
                    raise RuntimeError(message)
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
    def _decode_video(job: dict[str, Any]) -> bytes:
        result = job.get("result") if isinstance(job, dict) else None
        b64 = result.get("b64_json") if isinstance(result, dict) else None
        if not b64:
            raise RuntimeError("sd-server completed the job but returned no video.")
        try:
            return base64.b64decode(b64)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sd-server returned an undecodable video: {exc}") from exc

    @staticmethod
    def _decode_images(job: dict[str, Any]) -> list[bytes]:
        # Type-check each level before indexing (guards a misbehaving/older server).
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
        tail = _diagnostic_tail(self._tail)
        base = f"sd-server connection lost during {where}"
        if not self.is_alive():
            code = None if self._process is None else self._process.returncode
            base += f" (process exited, code {code})"
        if exc is not None:
            base += f": {exc}"
        if tail:
            base += "\nLast output:\n" + tail[:1500]
        return base
