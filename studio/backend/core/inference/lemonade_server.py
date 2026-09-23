# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Private Lemonade process serving FastFlowLM on the AMD NPU.

Uses authenticated loopback and owns the child process tree. Runtime, config and models
stay under Studio's home; a separate FLM_MODEL_PATH prevents other FastFlowLM installs
from deleting these models (lemonade-sdk/lemonade#3375). No torch or transformers imports.
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import httpx

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

# Disable LAN discovery and automatic updates for the private runtime.
_CONFIG_OVERRIDES: dict[str, Any] = {
    "broadcast": False,
    "auto_check_model_updates": False,
    "auto_update_models": False,
    "log_file": "disabled",
    "host": "127.0.0.1",
    "max_loaded_models": 1,
}


class LemonadeUnavailable(RuntimeError):
    """lemond is not running or stopped answering."""


def _kill_with_the_parent() -> None:
    """Override the Linux parent-death signal with SIGKILL.

    With a loaded model, SIGTERM held the NPU for about 9 s; SIGKILL released it within 1 s.
    """
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno = True).prctl(1, signal.SIGKILL)


class LemonadeServer:
    def __init__(
        self, binary: Path, *, cache_dir: Path, config_dir: Path, flm_model_dir: Path
    ) -> None:
        self.binary = Path(binary)
        self.cache_dir = Path(cache_dir)
        self.config_dir = Path(config_dir)
        self.flm_model_dir = Path(flm_model_dir)
        self.port: Optional[int] = None
        self.api_key: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._tail: deque[str] = deque(maxlen = 200)
        self._drain_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
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
            if is_process_shutting_down():
                raise LemonadeUnavailable("Unsloth is shutting down; not starting Lemonade.")
            self.cache_dir.mkdir(parents = True, exist_ok = True)
            self.flm_model_dir.mkdir(parents = True, exist_ok = True)
            self._write_config()
            port = self._find_free_port()
            api_key = secrets.token_urlsafe(32)
            env = child_env_without_native_path_secret()
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
            self._tail.clear()

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

            proc = spawn_on_lifetime_thread(_spawn)
            adopt_pid(proc.pid)
            self._process = proc
            self.port = port
            self.api_key = api_key
            # Shutdown may have swept children before this process was adopted.
            if is_process_shutting_down():
                self._kill_locked()
                raise LemonadeUnavailable("Unsloth is shutting down; not starting Lemonade.")
            self._drain_thread = threading.Thread(
                target = self._drain, args = (proc,), daemon = True, name = "lemond-drain"
            )
            self._drain_thread.start()
            if not self._wait_ready(timeout):
                tail = self.log_tail()
                self._kill_locked()
                raise LemonadeUnavailable(f"Lemonade did not start. Last output:\n{tail}")

    def _drain(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if line:
                    self._tail.append(line)
                    logger.debug("[lemond] %s", line)
        except Exception:  # noqa: BLE001 -- the pipe closes at teardown
            pass

    def _wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive():
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
            return self._client.request(
                method,
                f"{self.base_url}{path}",
                json = json_body,
                headers = self.headers(),
                timeout = timeout,
            )
        except (*_TRANSPORT_ERRORS, httpx.TimeoutException) as exc:
            detail = (
                "" if self.is_alive() else f" (process exited). Last output:\n{self.log_tail()}"
            )
            raise LemonadeUnavailable(f"Lemonade request {path} failed: {exc}{detail}") from exc

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
        return self._client.stream(
            method,
            f"{self.base_url}{path}",
            json = json_body,
            headers = self.headers(),
            timeout = httpx.Timeout(timeout, connect = 10.0, read = timeout),
        )

    def stop(self) -> None:
        """Unload, then stop lemond and every process under it. Idempotent."""
        with self._lock:
            if self.is_alive():
                try:
                    self._client.post(
                        f"{self.base_url}/v1/unload",
                        json = {},
                        headers = self.headers(),
                        timeout = 10.0,
                    )
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
