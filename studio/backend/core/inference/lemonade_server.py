# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Private Lemonade serving FastFlowLM on AMD NPU over authenticated loopback; no torch or
transformers imports; FLM_MODEL_PATH stops other FLM installs deleting models"""

from __future__ import annotations

import json
import logging
import re
import secrets
import socket
import subprocess
import threading
import time
import weakref
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import httpx

from utils.log_retention import prune_log_dir
from utils.native_path_leases import child_env_without_native_path_secret
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    is_process_shutting_down,
    spawn_on_lifetime_thread,
    terminate_pid,
)
from utils.subprocess_compat import windows_hidden_subprocess_kwargs

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
)

_CONFIG_OVERRIDES: dict[str, Any] = {
    "broadcast": False,
    "auto_check_model_updates": False,
    "auto_update_models": False,
    "log_file": "disabled",
    # backend_output() raises it over /internal/set, which persists; a crash must not leave debug.
    "log_level": "info",
    "host": "127.0.0.1",
    "max_loaded_models": 1,
}


# The freed port can be taken before lemond binds it; only a collision exit is retried.
_START_ATTEMPTS = 3
_PORT_TAKEN = re.compile(
    r"address already in use|eaddrinuse|only one usage of each socket address|errno 98\b|errno 48\b"
    # A localized Windows message still carries the code; never a bare 10048, which can be a port.
    r"|(?:winerror|errno|error)\s*:?\s*10048\b",
    re.IGNORECASE,
)


# lemond's "<date> <time> [Info] (Router) <message>"; backend output is tagged Process. Anchored:
# chat content can carry the same text further in.
_LEMOND_LINE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[(\w+)\] \(([^)]*)\) ")
# lemond's own line once /internal/set has taken effect.
_BACK_AT_INFO = re.compile(r"Logging configuration updated \(level=info\b")
_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")


class LemonadeUnavailable(RuntimeError):
    """lemond is not running or stopped answering."""


def _kill_with_the_parent() -> None:
    """Use SIGKILL, not SIGTERM: under SIGTERM a loaded NPU model kept the device held for seconds."""
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno = True).prctl(1, signal.SIGKILL)


class OutputWindow:
    __slots__ = ("server", "start", "shared", "__weakref__")

    def __init__(self, server: LemonadeServer, start: int) -> None:
        self.server = server
        self.start = start
        self.shared = False


class LemonadeServer:
    def __init__(
        self,
        binary: Path,
        *,
        cache_dir: Path,
        config_dir: Path,
        flm_model_dir: Path,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.binary = Path(binary)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.log_path: Optional[Path] = None
        self._log_file = None
        self._log_lock = threading.Lock()
        self.cache_dir = Path(cache_dir)
        self.config_dir = Path(config_dir)
        self.flm_model_dir = Path(flm_model_dir)
        self.port: Optional[int] = None
        self.api_key: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque[str] = deque(maxlen = 200)
        # Never reset, so a mark taken before a restart still counts the new lines.
        self._lines_seen = 0
        self._tail_lock = threading.Lock()
        self._windows: weakref.WeakSet[OutputWindow] = weakref.WeakSet()
        self._load_window: Optional[OutputWindow] = None
        self._record_kept = True
        # Set once a load is over, until lemond is seen back at info.
        self._filter_until: Optional[float] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # stop() sets it without the lock, which start() holds while waiting for readiness.
        self._stop_requested = threading.Event()
        self._client = httpx.Client(timeout = 30.0, trust_env = False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def log_tail(self, lines: int = 30) -> str:
        return "\n".join(list(self._tail)[-lines:])

    def output_mark(self) -> int:
        with self._tail_lock:
            return self._lines_seen

    def open_window(self) -> "OutputWindow":
        """A mark for one request. lemond's lines carry no request id, so a window that overlaps
        another request's is ``shared`` and its output can't be pinned on either. Released when
        the caller drops it."""
        with self._tail_lock:
            window = OutputWindow(self, self._lines_seen)
            others = list(self._windows)
            for other in others:
                other.shared = True
            window.shared = bool(others)
            self._windows.add(window)
        return window

    @contextmanager
    def _other_operation(self):
        """Keep overlapping loads and chats from claiming this operation's console output."""
        window = self.open_window()
        try:
            yield
        finally:
            with self._tail_lock:
                self._windows.discard(window)

    def output_since(
        self,
        mark: int,
        lines: int = 40,
        settle: float = 0.5,
    ) -> list[str]:
        """What lemond printed after ``mark``, at most ``lines``. It logs a failure just before
        answering, so this waits up to ``settle`` seconds for the drain to go quiet."""
        now = time.monotonic()
        deadline = now + settle
        quiet_for = min(0.2, settle)
        seen = self._lines_seen
        changed = now
        while time.monotonic() < deadline:
            time.sleep(0.05)
            if self._lines_seen != seen:
                seen = self._lines_seen
                changed = time.monotonic()
            # Nothing yet means the drain may still be behind: keep waiting to the deadline.
            elif seen > mark and time.monotonic() - changed >= quiet_for:
                break
        # One snapshot: a line drained between the two reads would shift the window.
        with self._tail_lock:
            tail = list(self._tail)
            count = min(self._lines_seen - mark, lines)
        return tail[-count:] if count > 0 else []

    @contextmanager
    def backend_output(self, window: OutputWindow):
        """Keep the backend's own output (FastFlowLM's reason for a failed load) while this is open.
        lemond only relays it for a backend started at debug, where lemond also logs chat content,
        and flm prints part of every prompt, so both stay off outside a load, and a load that
        overlaps a chat (``window`` is shared) goes without."""
        if window.shared:
            yield
            return
        # Up before debug, down after it: a chat during either level change is already filtered.
        self._filter_until = None
        self._load_window = window
        self._set_log_level("debug")
        try:
            yield
        finally:
            # A dead lemond prints no chat; its pipe may still hold flm's reason, so keep capturing.
            if self.is_alive():
                # No deadline until the request is back: it can outlast one, with lemond at debug.
                self._filter_until = float("inf")
                self._load_window = None
                if self._set_log_level("info") or not self.is_alive():
                    self._filter_until = time.monotonic() + 5.0

    def _keeps(self, line: str) -> bool:
        match = _LEMOND_LINE.match(line)
        if match:
            severity, tag = match.groups()
            own = self._record_kept = tag != "Process" and severity not in ("Trace", "Debug")
        else:
            # A request preview can continue across unprefixed lines: they follow their record.
            own = self._record_kept
        window = self._load_window
        if window is not None and not window.shared:
            return True
        until = self._filter_until
        if window is None and until is None:
            return own
        # Chat joined or debug may linger: lemond's own info+ lines only (debug runs onto untagged lines).
        back = bool(match) and own and _BACK_AT_INFO.search(line)
        if until is not None and (back or time.monotonic() > until):
            self._filter_until = None
        return bool(match) and own

    def _set_log_level(self, level: str) -> bool:
        if not self.is_alive():
            return False
        try:
            response = self._client.post(
                f"{self.base_url}/internal/set",
                json = {"log_level": level},
                headers = self.headers(),
                timeout = 10.0,
            )
            failure = None if response.status_code == 200 else f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 -- the load goes ahead either way
            failure = str(exc)
        if failure:
            logger.warning("Could not set Lemonade's log level to %s: %s", level, failure)
        return failure is None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _write_config(self) -> None:
        self.config_dir.mkdir(parents = True, exist_ok = True)
        path = self.config_dir / "config.json"
        try:
            config = json.loads(path.read_text(encoding = "utf-8"))
            if not isinstance(config, dict):
                config = {}
        except (OSError, ValueError):
            config = {}
        config.update(_CONFIG_OVERRIDES)
        config["models_dir"] = str(self.cache_dir / "models")
        config.setdefault("flm", {})["prefer_system"] = False
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(config, indent = 2) + "\n", encoding = "utf-8")
        tmp.replace(path)

    def start(self, timeout: float = 60.0) -> None:
        """Start lemond and wait for authenticated /v1/health to confirm readiness and ownership."""
        with self._lock:
            if self.is_alive():
                return
            self._stop_requested.clear()
            if is_process_shutting_down():
                raise LemonadeUnavailable("Unsloth is shutting down; not starting Lemonade.")
            self.cache_dir.mkdir(parents = True, exist_ok = True)
            self.flm_model_dir.mkdir(parents = True, exist_ok = True)
            self._write_config()
            for attempt in range(1, _START_ATTEMPTS + 1):
                if self._start_once(timeout):
                    return
                tail = self.log_tail()
                if (
                    attempt < _START_ATTEMPTS
                    and not self._stop_requested.is_set()
                    and _PORT_TAKEN.search(tail)
                ):
                    logger.warning(
                        "Lemonade lost its port to another process; retrying on a new one."
                    )
                    if is_process_shutting_down():
                        raise LemonadeUnavailable(
                            "Unsloth is shutting down; not starting Lemonade."
                        )
                    continue
                raise LemonadeUnavailable(f"Lemonade did not start. Last output:\n{tail}")

    def _start_once(self, timeout: float) -> bool:
        """Spawn lemond on a fresh port and wait for it; False (with the child gone) if it never got ready."""
        port = self._find_free_port()
        api_key = secrets.token_urlsafe(32)
        env = child_env_without_native_path_secret()
        # Lemonade CI mode forces debug even when config.json specifies info.
        env.pop("LEMONADE_CI_MODE", None)
        env["LEMONADE_API_KEY"] = api_key
        env["FLM_MODEL_PATH"] = str(self.flm_model_dir)
        env["FLM_DISABLE_UPDATE_CHECK"] = "1"
        cmd = [
            str(self.binary),
            str(self.cache_dir),
            str(self.config_dir),
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--no-broadcast",
            "--log-file",
            "disabled",
        ]
        logger.info("Starting Lemonade: %s", " ".join(cmd))
        with self._tail_lock:
            self._tail.clear()
        self._load_window = self._filter_until = None
        log_file = self._open_log_file(port, cmd)

        def _spawn() -> subprocess.Popen:
            return subprocess.Popen(
                cmd,
                cwd = str(self.binary.parent),
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                stdin = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                env = env,
                **windows_hidden_subprocess_kwargs(),
                **child_popen_kwargs(_kill_with_the_parent),
            )

        try:
            proc = spawn_on_lifetime_thread(_spawn)
            adopt_pid(proc.pid)
            self._process = proc
            self.port = port
            self.api_key = api_key
            # Shutdown may have swept children before this process was adopted.
            if is_process_shutting_down():
                self._kill_locked()
                raise LemonadeUnavailable("Unsloth is shutting down; not starting Lemonade.")
        except BaseException:
            self._close_log_file(log_file)
            raise
        self._drain_thread = threading.Thread(
            target = self._drain, args = (proc, log_file), daemon = True, name = "lemond-drain"
        )
        self._drain_thread.start()
        if not self._wait_ready(timeout):
            # After the kill, which joins the drain thread, so the tail has the last lines.
            self._kill_locked()
            return False
        return True

    def _open_log_file(
        self,
        port: int,
        cmd: Optional[list[str]] = None,
    ):
        if self.log_dir is None:
            return None
        try:
            self.log_dir.mkdir(parents = True, exist_ok = True)
            path = self.log_dir / f"lemonade-{int(time.time())}-port-{port}.log"
            # Append: a restart in the same second on the same port reuses the name.
            handle = open(path, "a", encoding = "utf-8", buffering = 1)
        except OSError as exc:
            logger.warning("Could not open a Lemonade log file: %s", exc)
            with self._log_lock:
                self.log_path = None
            return None
        # After the open, so the cap counts this file and can never delete it.
        prune_log_dir(self.log_dir, "lemonade-*.log", protect = path)
        logger.info("Lemonade output -> %s", path)
        with self._log_lock:
            self._log_file = handle
            self.log_path = path
        if cmd:
            self.note(f"Starting Lemonade: {' '.join(cmd)}")
        return handle

    def note(self, message: str) -> None:
        """A line of Unsloth Studio's own in the Lemonade log, in lemond's format, so the file reads
        on its own: what was asked of Lemonade and how it ended."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_log(f"{stamp} [Info] (Unsloth Studio) {message}")

    def _write_log(
        self,
        line: str,
        handle = None,
    ) -> None:
        with self._log_lock:
            target = handle if handle is not None else self._log_file
            if target is None or target.closed:
                return
            try:
                target.write(line + "\n")
            except (OSError, ValueError):
                pass

    def _drain(
        self,
        proc: subprocess.Popen,
        log_file = None,
    ) -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = _ANSI_COLOR.sub("", raw).rstrip()
                if line and self._keeps(line):
                    with self._tail_lock:
                        self._tail.append(line)
                        self._lines_seen += 1
                    logger.debug("[lemond] %s", line)
                    if log_file is not None:
                        self._write_log(line, log_file)
        except Exception:  # noqa: BLE001 -- the pipe closes at teardown
            pass
        finally:
            self._close_log_file(log_file)

    def _close_log_file(self, log_file) -> None:
        if log_file is None:
            return
        with self._log_lock:
            # A restart may already have opened the next file; only close this one.
            if self._log_file is log_file:
                self._log_file = None
            log_file.close()

    def _wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive() or self._stop_requested.is_set():
                return False
            try:
                response = self._client.get(
                    f"{self.base_url}/v1/health", headers = self.headers(), timeout = 2.0
                )
                if response.status_code == 200:
                    return True
            except (*_TRANSPORT_ERRORS, httpx.TimeoutException):
                pass
            time.sleep(0.25)
        return False

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict] = None,
        timeout: float = 60.0,
    ) -> httpx.Response:
        if not self.is_alive():
            raise LemonadeUnavailable("Lemonade is not running.")
        try:
            # The load owns its window; of the reads only a catalog listing makes lemond print.
            other = path != "/v1/load" and (
                method == "POST" or path.partition("?")[0] == "/v1/models"
            )
            with self._other_operation() if other else nullcontext():
                return self._client.request(
                    method,
                    f"{self.base_url}{path}",
                    json = json_body,
                    headers = self.headers(),
                    timeout = timeout,
                )
        except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as exc:
            detail = ""
            if not self.is_alive():
                detail = " (process exited)"
                if path != "/v1/load":
                    detail += f". Last output:\n{self.log_tail()}"
            raise LemonadeUnavailable(f"Lemonade request {path} failed: {exc}{detail}") from exc

    @contextmanager
    def stream(
        self,
        method: str,
        path: str,
        *,
        json_body: dict,
        timeout: float = 3600.0,
    ):
        """``httpx.Client.stream`` against lemond, for server-sent progress."""
        if not self.is_alive():
            raise LemonadeUnavailable("Lemonade is not running.")
        with (
            self._other_operation(),
            self._client.stream(
                method,
                f"{self.base_url}{path}",
                json = json_body,
                headers = self.headers(),
                timeout = httpx.Timeout(timeout, connect = 10.0, read = timeout),
            ) as response,
        ):
            yield response

    def stop(self) -> None:
        """Unload, then stop lemond and every process under it. Idempotent."""
        self._stop_requested.set()
        with self._lock:
            if self.is_alive():
                try:
                    self.request("POST", "/v1/unload", json_body = {}, timeout = 10.0)
                except Exception:  # noqa: BLE001 -- the kill below still runs
                    pass
            self._kill_locked()

    def _kill_locked(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            # Kill descendants too: flm.exe survives its parent on Windows.
            terminate_pid(proc.pid, timeout = 10.0, owner_verified = True)
            proc.wait(timeout = 10)
        except Exception as exc:  # noqa: BLE001 -- best-effort teardown
            logger.warning("Error stopping Lemonade: %s", exc)
        finally:
            self._process = None
            self.port = None
            self.api_key = None
            if self._drain_thread is not None:
                self._drain_thread.join(timeout = 2)
                self._drain_thread = None

    def close(self) -> None:
        self.stop()
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
