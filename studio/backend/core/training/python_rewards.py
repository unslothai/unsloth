# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Python rewards: user code scored in one long-lived worker inside the OS sandbox.

The worker is launched through the same ``os_sandbox`` path as the chat Python tool (MXC on
Windows, bubblewrap on Linux, Seatbelt on macOS) in ``auto`` mode, so a host without OS
isolation falls back to the software safeguards and the run records which one it got.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from core.training import reward_worker_main as _worker_io

logger = logging.getLogger(__name__)

READY_TIMEOUT_SECONDS = 120.0
BATCH_TIMEOUT_SECONDS = 300.0
HEARTBEAT_SECONDS = 10.0
_POLL_SECONDS = 0.002
_WORKER_SOURCE = Path(__file__).with_name("reward_worker_main.py")
# TRL passes these next to the dataset columns; none of them mean anything outside the trainer.
_TRAINER_KWARGS = frozenset(
    {"trainer_state", "completion_ids", "log_extra", "log_metric", "environments"}
)


class PythonRewardError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    return str(value)


def _safe_env(workdir: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": workdir,
        "TMP": workdir,
        "TEMP": workdir,
        "TMPDIR": workdir,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    for key in ("SystemRoot", "PATHEXT", "VIRTUAL_ENV"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def _unsloth_helpers_path() -> Optional[str]:
    import importlib.util

    try:
        spec = importlib.util.find_spec("unsloth_zoo")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for root in spec.submodule_search_locations:
        path = os.path.join(root, "rl_environments.py")
        if os.path.isfile(path):
            return path
    return None


def _kill_tree(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 10,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check = False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - fall through to a plain kill
        pass
    try:
        proc.kill()
        proc.wait(timeout = 10)
    except Exception:  # noqa: BLE001
        pass


class RewardWorker:
    """One sandboxed process holding every Python reward of a run."""

    def __init__(
        self,
        specs: list[dict],
        *,
        mode: str = "auto",
        batch_timeout: float = BATCH_TIMEOUT_SECONDS,
    ):
        self.specs = [{"name": s["name"], "entry": s["entry"], "code": s["code"]} for s in specs]
        self.mode = mode
        self.batch_timeout = batch_timeout
        self.isolation: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._proc = None
        self._prepared = None
        self._workdir: Optional[str] = None
        self._closed = threading.Event()

    def _heartbeat(self) -> None:
        # Without this a crashed trainer would leave the worker polling forever.
        while not self._closed.wait(HEARTBEAT_SECONDS):
            try:
                Path(self._workdir, "alive").touch()
            except (OSError, TypeError):
                return

    def start(self) -> "RewardWorker":
        from core.inference import os_sandbox

        if self._workdir is not None:
            raise PythonRewardError("This reward worker was already started.")
        self._workdir = tempfile.mkdtemp(prefix = "unsloth-rewards-")
        Path(self._workdir, "alive").touch()
        shutil.copyfile(_WORKER_SOURCE, os.path.join(self._workdir, "worker.py"))
        with open(os.path.join(self._workdir, "rewards.json"), "w", encoding = "utf-8") as f:
            json.dump({"specs": self.specs, "unsloth_helpers": _unsloth_helpers_path()}, f)
        try:
            self._prepared = os_sandbox.prepare_tool_launch(
                os_sandbox.ToolLaunchPlan(
                    argv = (sys.executable, "-u", os.path.join(self._workdir, "worker.py")),
                    workdir = self._workdir,
                    env = _safe_env(self._workdir),
                    preexec_fn = None if os.name == "nt" else os.setsid,
                    requested_mode = self.mode,
                    execution_kind = "python",
                )
            )
            popen_kwargs: dict[str, Any] = {
                "cwd": self._prepared.workdir,
                "env": self._prepared.env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": self._prepared.close_fds,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                popen_kwargs["preexec_fn"] = self._prepared.preexec_fn
                if self._prepared.pass_fds:
                    popen_kwargs["pass_fds"] = tuple(self._prepared.pass_fds)
            self._proc = os_sandbox.spawn_prepared_launch(self._prepared, **popen_kwargs)
            record = self._prepared.execution_record
            self.isolation = {
                "backend": self._prepared.backend,
                "os_isolation": bool(record and record.os_isolation),
            }
            ready = self._wait_for("ready.json", READY_TIMEOUT_SECONDS)
        except BaseException:
            self.close()
            raise
        if ready.get("errors"):
            errors = ready["errors"]
            self.close()
            raise PythonRewardError(
                "; ".join(f"{name}: {message}" for name, message in sorted(errors.items()))
            )
        threading.Thread(target = self._heartbeat, daemon = True).start()
        atexit.register(self.close)
        logger.info("Python reward worker started (%s)", self.isolation)
        return self

    def _wait_for(self, filename: str, timeout: float) -> dict:
        path = os.path.join(self._workdir, filename)
        deadline = time.monotonic() + timeout
        while not os.path.exists(path):
            if self._proc.poll() is not None:
                raise PythonRewardError(
                    f"The reward worker exited (code {self._proc.returncode}). {self.log_tail()}".strip()
                )
            if time.monotonic() > deadline:
                raise PythonRewardError(f"The reward worker did not answer within {timeout:.0f} s.")
            time.sleep(_POLL_SECONDS)
        return _worker_io.take(path)

    def log_tail(self, limit: int = 2000) -> str:
        try:
            text = Path(self._workdir, "worker.log").read_text("utf-8", errors = "replace")
        except (OSError, TypeError):
            return ""
        return text[-limit:].strip()

    def score(
        self,
        names: list[str],
        prompts: Any,
        completions: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, list[Optional[float]]]:
        request = {
            "rewards": names,
            "prompts": _jsonable(prompts),
            "completions": _jsonable(completions),
            "kwargs": {
                k: _jsonable(v) for k, v in kwargs.items() if k not in _TRAINER_KWARGS
            },
        }
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise PythonRewardError("The reward worker is not running.")
            seq = self._seq
            self._seq += 1
            path = os.path.join(self._workdir, f"req-{seq}.json")
            _worker_io.put(path, request)
            try:
                result = self._wait_for(f"res-{seq}.json", self.batch_timeout)
            except PythonRewardError:
                _kill_tree(self._proc)
                raise
        if result.get("errors"):
            raise PythonRewardError(
                "; ".join(f"{name}: {message}" for name, message in sorted(result["errors"].items()))
            )
        return result["scores"]

    def close(self) -> None:
        self._closed.set()
        proc, self._proc = self._proc, None
        if proc is not None and self._workdir:
            try:
                Path(self._workdir, "stop").touch()
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001
                _kill_tree(proc)
        if self._prepared is not None:
            self._prepared.cleanup()
            self._prepared = None
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors = True)
            self._workdir = None


def _as_turns(values: Any, role: str) -> Any:
    # With thinking rendered into the prompt text, TRL hands rewards plain strings; notebook
    # rewards read completion[0]["content"], so give them the conversational shape either way.
    if not isinstance(values, list):
        return values
    return [[{"role": role, "content": v}] if isinstance(v, str) else v for v in values]


def make_python_reward_funcs(
    specs: list[dict], worker: RewardWorker
) -> list[Callable[..., list[Optional[float]]]]:
    def make(name: str):
        def reward(prompts = None, completions = None, **kwargs):
            return worker.score(
                [name], _as_turns(prompts, "user"), _as_turns(completions, "assistant"), kwargs
            )[name]

        reward.__name__ = name.replace("-", "_")
        return reward

    return [make(s["name"]) for s in specs]


_preview_lock = threading.Lock()
_preview: dict[str, Any] = {"key": None, "worker": None, "timer": None}
PREVIEW_IDLE_SECONDS = 120.0


def _close_preview() -> None:
    with _preview_lock:
        worker = _preview["worker"]
        _preview.update(key = None, worker = None, timer = None)
    if worker is not None:
        worker.close()


def preview_python_scores(
    specs: list[dict],
    text: str,
    row: dict[str, Any],
    prompt: Optional[str] = None,
) -> tuple[dict[str, Optional[float]], dict[str, Any]]:
    """Score one reply for the Try it box. The worker is kept warm for a couple of minutes,
    since the box re-scores on every edit."""
    key = json.dumps([[s["name"], s["entry"], s["code"]] for s in specs], sort_keys = True)
    with _preview_lock:
        if _preview["timer"] is not None:
            _preview["timer"].cancel()
        worker = _preview["worker"] if _preview["key"] == key else None
        stale = _preview["worker"] if worker is None else None
        if stale is not None:
            _preview.update(key = None, worker = None)
    if stale is not None:
        stale.close()
    if worker is None:
        worker = RewardWorker(specs, batch_timeout = 30.0).start()
        with _preview_lock:
            _preview.update(key = key, worker = worker)
    prompts = [[{"role": "user", "content": prompt or ""}]]
    completions = [[{"role": "assistant", "content": text}]]
    kwargs = {k: [v] for k, v in (row or {}).items()}
    try:
        scores = worker.score([s["name"] for s in specs], prompts, completions, kwargs)
    except PythonRewardError:
        _close_preview()
        raise
    finally:
        timer = threading.Timer(PREVIEW_IDLE_SECONDS, _close_preview)
        timer.daemon = True
        with _preview_lock:
            _preview["timer"] = timer
        timer.start()
    return {name: values[0] for name, values in scores.items()}, worker.isolation
