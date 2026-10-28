# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Python rewards: user code scored in one long-lived worker inside the OS sandbox.

The worker is launched through the same ``os_sandbox`` path as the chat Python tool (MXC on
Windows, bubblewrap on Linux, Seatbelt on macOS). A training run asks for ``required`` unless the
user opted into ``auto`` (software safeguards), and records which one it got in its RL metrics.
"""

from __future__ import annotations

import atexit
import itertools
import json
import logging
import math
import os
import re
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
from core.training.rewards import PREFILL_COLUMN, PROMPT_TURNS_COLUMN, TOOLS_COLUMN

try:
    import resource as _resource
except ImportError:  # Windows
    _resource = None

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
# The bundled exact-answer rule's reference_extract.
_HASH_ANSWER = re.compile(r"####\s*(.+?)\s*$", re.DOTALL | re.MULTILINE)


class PythonRewardError(RuntimeError):
    pass


ISOLATION_FIELD = "grpo_python_reward_isolation"
# The sandbox's remediation is written for the chat tools; these say the same for a reward.
_TOOL_WORDING = (
    ("To isolate tool calls, ", "To isolate them, "),
    ("; Unsloth picks it up within a minute, no restart needed.", ", then try again."),
)


def sandbox_refusal(reason: str, remediation: str = "") -> str:
    """The one refusal Try it, Start and training show when Python rewards need an OS sandbox
    this host cannot start: the fix in plain words, no internal error codes."""
    from core.inference.os_sandbox import _FALLBACK_NOTE

    detail = (remediation or "").replace(_FALLBACK_NOTE, "").strip()
    for old, new in _TOOL_WORDING:
        detail = detail.replace(old, new)
    if not detail:
        # "OS_ISOLATION_UNAVAILABLE: ... SandboxUnavailableError: why" -> "why"
        detail = re.sub(r"^(?:[A-Z_]+:\s*)?(?:.*?\b\w+Error:\s*)?", "", str(reason)).strip()
        detail = detail[:1].upper() + detail[1:]
        if detail and not detail.endswith("."):
            detail += "."
    return (
        f"Python rewards need the OS sandbox, and this system has none. {detail} To run them "
        "with software safeguards only, which do not stop a reward from reading your files, "
        "writing outside its folder or using the network, set Python reward isolation to Auto "
        f'({ISOLATION_FIELD}: "auto").'
    ).replace("  ", " ")


def python_isolation_refusal(mode: str) -> Optional[str]:
    """What a ``required`` run would be refused with on this host, or None. The same capability
    check the reward worker's launch makes, so Start and Try it agree."""
    if mode != "required":
        return None
    from core.inference import os_sandbox

    capability = os_sandbox.capability_snapshot(
        execution_kind = "python", selected_executable = sys.executable
    )
    if capability.available:
        return None
    return sandbox_refusal(capability.reason, capability.remediation)


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


def _final_answers(kwargs: dict[str, Any]) -> dict[str, Any]:
    # GSM8K answers end in "#### 72". The notebooks map them to the bare answer before TRL sees
    # them and the bundled rules extract it, so Python rewards get it too; the full text stays
    # under answer_raw.
    answers = kwargs.get("answer")
    if not isinstance(answers, (list, tuple)):
        return kwargs
    parsed = []
    for value in answers:
        match = _HASH_ANSWER.search(value) if isinstance(value, str) else None
        parsed.append(match.group(1) if match else value)
    return {"answer_raw": list(answers), **kwargs, "answer": parsed}


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


def _kill_group(proc) -> None:
    # The worker leads its own group (setsid in the preexec), so a reward's children stay in it
    # after the worker exits cleanly; _kill_tree skips an exited worker.
    if os.name == "nt" or proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


# Reward code is user code: one worker must not fill the disk or memory. No CPU cap, since one
# worker scores a whole run; the batch timeout bounds a stuck call.
REWARD_FSIZE_BYTES = 100 * 1024 * 1024
REWARD_NOFILE = 4096
REWARD_NPROC = 10000
# Resolved before the fork: the settings store cannot be read in the child.
_reward_as_bytes: Optional[int] = None


def _reward_worker_preexec() -> None:
    os.setsid()
    if _resource is None:
        return
    limits = [
        (_resource.RLIMIT_FSIZE, REWARD_FSIZE_BYTES),
        (_resource.RLIMIT_NOFILE, REWARD_NOFILE),
        (getattr(_resource, "RLIMIT_NPROC", None), REWARD_NPROC),
        (_resource.RLIMIT_AS, _reward_as_bytes),
    ]
    for limit, value in limits:
        if limit is None or value is None:
            continue
        try:
            _soft, hard = _resource.getrlimit(limit)
            target = value if hard == _resource.RLIM_INFINITY else min(value, hard)
            _resource.setrlimit(limit, (target, target))
        except (ValueError, OSError):
            pass


class RewardWorker:
    """One sandboxed process holding every Python reward of a run."""

    def __init__(
        self,
        specs: list[dict],
        *,
        mode: str = "auto",
        batch_timeout: float = BATCH_TIMEOUT_SECONDS,
        strict: bool = True,
    ):
        self.specs = [{"name": s["name"], "entry": s["entry"], "code": s["code"]} for s in specs]
        self.mode = mode
        # Training refuses a reward that does not load; Try it scores the others and names it.
        self.strict = strict
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
        global _reward_as_bytes
        try:
            from utils import sandbox_memory_limit
            _reward_as_bytes = sandbox_memory_limit.memory_limit_bytes()
        except Exception:  # noqa: BLE001 - no memory cap rather than no worker
            _reward_as_bytes = None
        try:
            self._prepared = os_sandbox.prepare_tool_launch(
                os_sandbox.ToolLaunchPlan(
                    argv = (sys.executable, "-u", os.path.join(self._workdir, "worker.py")),
                    workdir = self._workdir,
                    env = _safe_env(self._workdir),
                    preexec_fn = None if os.name == "nt" else _reward_worker_preexec,
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
        if ready.get("errors") and self.strict:
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
        self, names: list[str], prompts: Any, completions: Any, kwargs: dict[str, Any]
    ) -> dict[str, list[Optional[float]]]:
        scores, errors = self.score_each(names, prompts, completions, kwargs)
        if errors:
            raise PythonRewardError(
                "; ".join(f"{name}: {message}" for name, message in sorted(errors.items()))
            )
        return scores

    def score_each(
        self, names: list[str], prompts: Any, completions: Any, kwargs: dict[str, Any]
    ) -> tuple[dict[str, list[Optional[float]]], dict[str, str]]:
        """Scores of the rewards that ran, and each failing reward's own error."""
        request = {
            "rewards": names,
            "prompts": _jsonable(prompts),
            "completions": _jsonable(completions),
            "kwargs": {
                k: _jsonable(v)
                for k, v in _final_answers(kwargs).items()
                if k not in _TRAINER_KWARGS
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
        return result.get("scores") or {}, result.get("errors") or {}

    def close(self) -> None:
        self._closed.set()
        # The hook holds this worker (and its reward code) until exit; Try it starts many.
        atexit.unregister(self.close)
        proc, self._proc = self._proc, None
        if proc is not None and self._workdir:
            try:
                Path(self._workdir, "stop").touch()
                proc.wait(timeout = 5)
            except Exception:  # noqa: BLE001
                _kill_tree(proc)
            _kill_group(proc)
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


def _without_padding(turns: Any) -> Any:
    # Arrow's None for fields only other rows' messages have; Try it never sees them.
    if not isinstance(turns, list):
        return turns
    from core.training.rl import _drop_nulls
    return [
        {**_drop_nulls(m), "content": m.get("content")} if isinstance(m, dict) else m for m in turns
    ]


def make_python_reward_funcs(
    specs: list[dict], worker: RewardWorker
) -> list[Callable[..., list[Optional[float]]]]:
    def make(name: str):
        def reward(
            prompts = None,
            completions = None,
            **kwargs,
        ):
            # A prompt rendered to text (render_grpo_prompts) comes back as its turns, with the
            # assistant prefill TRL would have put in front of the completion.
            turns = kwargs.pop(PROMPT_TURNS_COLUMN, None)
            tools = kwargs.pop(TOOLS_COLUMN, None)
            if isinstance(tools, list) and "tools" not in kwargs:
                kwargs["tools"] = [json.loads(t) if isinstance(t, str) else t for t in tools]
            prefills = kwargs.pop(PREFILL_COLUMN, None)
            if isinstance(turns, list) and isinstance(prompts, list) and len(turns) == len(prompts):
                prompts = [json.loads(t) if isinstance(t, str) else t for t in turns]
            if isinstance(prompts, list):
                prompts = [_without_padding(p) for p in prompts]
            if isinstance(prefills, list) and isinstance(completions, list):
                completions = [
                    p + c if isinstance(c, str) and isinstance(p, str) else c
                    for p, c in zip(prefills, completions)
                ] + completions[len(prefills) :]
            return worker.score(
                [name], _as_turns(prompts, "user"), _as_turns(completions, "assistant"), kwargs
            )[name]

        reward.__name__ = name.replace("-", "_")
        return reward

    return [make(s["name"]) for s in specs]


_preview_lock = threading.Lock()
# Warm Try it workers by key: at most one idle per account, and none is closed while it scores.
_previews: dict[str, dict[str, Any]] = {}
PREVIEW_IDLE_SECONDS = 120.0
# Workers one account may have scoring older code while a new edit asks for another.
PREVIEWS_IN_FLIGHT = 2
_preview_seq = itertools.count()


def _drop_preview(key: str, entry: dict[str, Any]) -> None:
    with _preview_lock:
        if _previews.get(key) is not entry or entry["busy"]:
            return
        del _previews[key]
    if entry["worker"] is not None:
        entry["worker"].close()


def _close_preview() -> None:
    with _preview_lock:
        entries = list(_previews.values())
        _previews.clear()
    for entry in entries:
        if entry["timer"] is not None:
            entry["timer"].cancel()
        if entry["worker"] is not None:
            entry["worker"].close()


def _start_preview_worker(specs: list[dict], mode: str) -> RewardWorker:
    from core.inference.os_sandbox import SandboxUnavailableError
    try:
        return RewardWorker(specs, mode = mode, batch_timeout = 30.0, strict = False).start()
    except SandboxUnavailableError as exc:
        raise PythonRewardError(sandbox_refusal(str(exc), exc.remediation)) from exc


def preview_python_scores(
    specs: list[dict],
    text: str,
    row: dict[str, Any],
    prompt: Any = None,
    owner: Optional[str] = None,
    mode: str = "required",
    system_prompt: Optional[str] = None,
) -> tuple[dict[str, Optional[float]], dict[str, Any], dict[str, str]]:
    """Score one reply for the Try it box. The worker is kept warm for a couple of minutes,
    since the box re-scores on every edit. ``mode`` is the os_sandbox mode, as in training:
    ``required`` refuses a host without OS isolation, ``auto`` is the opt-in to software safeguards.
    Returns the scores, the worker's isolation and each failing reward's error."""
    # Keyed by account: a warm worker keeps the reward's module state, which must not carry one
    # account's rows into another account's scores.
    key = json.dumps(
        [owner, mode, [[s["name"], s["entry"], s["code"]] for s in specs]], sort_keys = True
    )
    stale = []
    with _preview_lock:
        entry = _previews.get(key)
        if entry is None:
            # A slow reward and quick edits would otherwise start a worker per edit.
            if sum(o["owner"] == owner and o["busy"] > 0 for o in _previews.values()) >= (
                PREVIEWS_IN_FLIGHT
            ):
                raise PythonRewardError(
                    "An earlier Try it is still scoring; try again when it finishes."
                )
            # The account's other idle workers hold older code.
            for other_key, other in list(_previews.items()):
                if other["owner"] == owner and not other["busy"]:
                    del _previews[other_key]
                    stale.append(other)
            entry = {
                "owner": owner,
                "worker": None,
                "timer": None,
                "busy": 0,
                "seq": next(_preview_seq),
            }
            entry["start"] = threading.Lock()
            _previews[key] = entry
        if entry["timer"] is not None:
            entry["timer"].cancel()
            entry["timer"] = None
        entry["busy"] += 1
    for other in stale:
        if other["timer"] is not None:
            other["timer"].cancel()
        if other["worker"] is not None:
            other["worker"].close()
    from core.training.rl import _REWARD_CALL_KEYS, _row_tools, prompt_turns

    prompts = [prompt_turns(prompt, system_prompt)]
    completions = [[{"role": "assistant", "content": text}]]
    # As format_rl_dataset drops them: TRL passes these names to the reward itself.
    kwargs = {k: [v] for k, v in (row or {}).items() if k not in _REWARD_CALL_KEYS}
    if "tools" in kwargs:
        # Decoded and unpadded, as training passes them.
        kwargs["tools"] = [_row_tools(row)]
    failed = True
    try:
        with entry["start"]:
            if entry["worker"] is None:
                entry["worker"] = _start_preview_worker(specs, mode)
        worker = entry["worker"]
        scores, errors = worker.score_each([s["name"] for s in specs], prompts, completions, kwargs)
        failed = False
    finally:
        with _preview_lock:
            entry["busy"] -= 1
            superseded = any(
                o["owner"] == owner and o["seq"] > entry["seq"] for o in _previews.values()
            )
            if (failed or (superseded and not entry["busy"])) and _previews.get(key) is entry:
                # A failed or superseded worker is not reused; whoever finishes last closes it.
                del _previews[key]
            idle = not entry["busy"]
            mapped = _previews.get(key) is entry
            if idle and mapped:
                timer = threading.Timer(PREVIEW_IDLE_SECONDS, _drop_preview, (key, entry))
                timer.daemon = True
                entry["timer"] = timer
                timer.start()
        if idle and not mapped and entry["worker"] is not None:
            entry["worker"].close()
    return {name: values[0] for name, values in scores.items()}, worker.isolation, errors
