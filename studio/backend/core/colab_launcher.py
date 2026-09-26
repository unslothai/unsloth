# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rent a Colab GPU with Google's ``colab`` CLI, run Unsloth Studio on it and link it here.

The CLI is Linux/macOS only, so on Windows every call goes through WSL. Lessons from
unslothai/scripts notebook_cloud_run.py that shape this module: ``--config``/``--auth`` are
global flags, ``colab exec --timeout`` is per cell so every call carries its own wall deadline,
``colab status`` output (not its exit code) is the only liveness signal, ``--env`` is never
passed, and a session that failed half way is always stopped because it bills until it is.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from loggers import get_logger
from storage import linked_instances_db

logger = get_logger(__name__)

GPUS = ("T4", "L4", "A100", "H100")
STAGES = ("allocating", "installing", "starting", "linking", "ready")
LINK_MARKER = "UNSLOTH_LINK "
INSTALL_OK = "UNSLOTH_STEP install ok"
PROBE_MARKER = "UNSLOTH_PROBE "
SESSION_PREFIX = "unsloth-"
CONFIG_DIR = "~/.config/unsloth-studio/colab"
HISTORY_DIR = "~/.config/colab-cli/history"

REPO_URL = os.environ.get("UNSLOTH_COLAB_REPO", "https://github.com/unslothai/unsloth.git")
REPO_BRANCH = os.environ.get("UNSLOTH_COLAB_BRANCH", "main")

ALLOCATE_TIMEOUT = 900
INSTALL_CELL_TIMEOUT = 3600
START_CELL_TIMEOUT = 900
STOP_TIMEOUT = 300
PROBE_TIMEOUT = 120
LOG_LINES = 200

_KEY_RE = re.compile(r"sk-unsloth-[A-Za-z0-9]+")
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

CAPACITY_MARKERS = (
    "toomanyassignments",
    "precondition failed",
    "no quota for",
    "no accelerator quota",
    "resource exhausted",
    "session count",
    "quota exceeded",
)
AUTH_MARKERS = ("401", "403", "unauthenticated", "unauthorized", "invalid credentials", "reauthentication")

# Runs on the VM, inside the kernel. Same install as the Colab notebook.
INSTALL_SCRIPT = """
import os, subprocess
if not os.path.isdir("/content/unsloth/.git"):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", {branch!r}, {repo!r}, "/content/unsloth"], check=True)
proc = subprocess.Popen(["bash", "studio/setup.sh", "--local"], cwd="/content/unsloth", stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print("UNSLOTH_STEP install " + ("ok" if rc == 0 else "failed rc=%d" % rc), flush=True)
"""

# Detached on the VM so `colab exec` returns while the server and tunnel keep running. Writes the
# URL and a fresh API key to a 0600 file; the admin password is randomised and never leaves the VM.
LAUNCHER_SCRIPT = """
import json, os, secrets, sys, time
from pathlib import Path
sys.path.insert(0, "/content/unsloth/studio/backend")
LINK = "/content/.unsloth-link.json"

def emit(data):
    fd = os.open(LINK + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(LINK + ".tmp", LINK)

try:
    from auth import storage
    storage.ensure_default_admin()
    if storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
        storage.update_password(storage.DEFAULT_ADMIN_USERNAME, secrets.token_urlsafe(24))
    import colab
    from run import run_server
    app = run_server(host="0.0.0.0", port=8888, frontend_path=Path("/content/unsloth/studio/frontend/dist"),
                     silent=True, cloudflare=False)
    port = getattr(app.state, "server_port", None) or 8888
    for _ in range(180):
        if colab._is_studio_healthy(port):
            break
        time.sleep(1)
    else:
        raise RuntimeError("Unsloth Studio did not become healthy on port %d" % port)
    url = colab.start_cloudflare_tunnel(port)
    if not url:
        raise RuntimeError("the Cloudflare tunnel did not produce a URL")
    key, _ = storage.create_api_key(username=storage.DEFAULT_ADMIN_USERNAME, name="unsloth-desktop-link")
    emit({"url": url, "api_key": key})
except BaseException as exc:
    emit({"error": "%s: %s" % (type(exc).__name__, exc)})
    raise
while True:
    time.sleep(3600)
"""

START_SCRIPT = """
import json, os, subprocess, sys, time
LINK, LOG, LAUNCHER = "/content/.unsloth-link.json", "/content/unsloth-studio.log", "/content/unsloth-launch.py"
if os.path.exists(LINK):
    os.remove(LINK)
with open(LAUNCHER, "w") as f:
    f.write({launcher!r})
venv = os.path.expanduser("~/.unsloth/studio/unsloth_studio/bin/python")
python = venv if os.path.exists(venv) else sys.executable
subprocess.Popen([python, LAUNCHER], cwd="/content/unsloth/studio/backend", stdin=subprocess.DEVNULL,
                 stdout=open(LOG, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
deadline = time.time() + 600
while time.time() < deadline and not os.path.exists(LINK):
    time.sleep(2)
if os.path.exists(LINK):
    with open(LINK) as f:
        data = json.load(f)
    os.remove(LINK)
else:
    data = {{"error": "Unsloth Studio did not report a link within 10 minutes"}}
if "error" in data and os.path.exists(LOG):
    print(open(LOG, errors="replace").read()[-4000:])
print("UNSLOTH_LINK " + json.dumps(data), flush=True)
"""

# Read-only: never runs `colab sessions` without a saved credential, because the oauth2 provider
# would then start its interactive sign-in and wait on stdin.
PROBE_SCRIPT = r"""
command -v uv >/dev/null 2>&1 && echo "UNSLOTH_PROBE uv=yes"
c=$(command -v colab 2>/dev/null) || { echo "UNSLOTH_PROBE cli=missing"; exit 0; }
echo "UNSLOTH_PROBE cli=$c"
py=$(head -n1 "$c" | sed -n 's/^#!\([^ ]*\).*/\1/p')
if [ -n "$py" ] && ! "$py" -c "import jupyter_kernel_client as j, sys; sys.exit(0 if hasattr(j, 'KernelClient') and hasattr(j, 'JupyterSubprotocol') else 3)" >/dev/null 2>&1; then
  echo "UNSLOTH_PROBE kernel_client=bad"
fi
if [ -f "$HOME/.config/colab-cli/token.json" ]; then auth=oauth2
elif [ -f "${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}" ]; then auth=adc
else echo "UNSLOTH_PROBE auth=none"; exit 0; fi
echo "UNSLOTH_PROBE auth=$auth"
out=$(timeout 90 colab --auth "$auth" sessions </dev/null 2>&1); rc=$?
echo "UNSLOTH_PROBE sessions_rc=$rc"
printf '%s\n' "$out" | tail -n 8
"""


class LaunchError(Exception):
    pass


class Cancelled(Exception):
    pass


@dataclass
class ShellResult:
    returncode: int
    output: str
    timed_out: bool = False


@dataclass
class Runner:
    kind: str  # "native" | "wsl"
    distro: Optional[str] = None

    def argv(self, script: str) -> list[str]:
        if self.kind == "wsl":
            return [
                shutil.which("wsl.exe") or "wsl.exe",
                *(["-d", self.distro] if self.distro else []),
                "-e", "bash", "-lc", script,
            ]
        return ["bash", "-lc", script]


def detect_runner() -> Optional[Runner]:
    if sys.platform == "win32":
        if not (shutil.which("wsl.exe") or shutil.which("wsl")):
            return None
        return Runner("wsl", os.environ.get("UNSLOTH_COLAB_WSL_DISTRO") or None)
    if not shutil.which("bash"):
        return None
    return Runner("native")


def _clean(line: str) -> str:
    return _KEY_RE.sub("sk-unsloth-***", _ANSI_RE.sub("", line.replace("\x00", "")).rstrip())


def run_shell(
    runner: Runner,
    script: str,
    *,
    stdin: Optional[str] = None,
    timeout: float,
    on_line: Optional[Callable[[str], None]] = None,
    cancel: Optional[threading.Event] = None,
) -> ShellResult:
    """Run *script* under bash with a wall deadline, streaming lines to *on_line*."""
    kwargs: dict = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.Popen(
            runner.argv(script),
            stdin = subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1", "TERM": "dumb"},
            **kwargs,
        )
    except OSError as exc:
        raise LaunchError(f"Could not run bash: {exc}") from exc

    killed = {"timeout": False}

    def _kill(reason: str) -> None:
        if proc.poll() is None:
            killed[reason] = True
            _terminate(proc)

    timer = threading.Timer(timeout, _kill, args = ("timeout",))
    timer.daemon = True
    timer.start()
    watcher = None
    if cancel is not None:
        def _watch() -> None:
            while proc.poll() is None:
                if cancel.wait(0.5):
                    _kill("cancel")
                    return
        watcher = threading.Thread(target = _watch, daemon = True)
        watcher.start()
    if stdin is not None:
        try:
            proc.stdin.write(stdin.encode())
            proc.stdin.close()
        except OSError:
            pass
    tail: deque[str] = deque(maxlen = 400)
    try:
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors = "replace").replace("\x00", "").rstrip("\r\n")
            line = line.split("\r")[-1]  # progress bars redraw with bare \r
            tail.append(line)
            if on_line:
                on_line(line)
        proc.wait()
    finally:
        timer.cancel()
    return ShellResult(proc.returncode if proc.returncode is not None else -1, "\n".join(tail), killed["timeout"])


def _terminate(proc: subprocess.Popen) -> None:
    try:
        if os.name != "nt":
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout = 15)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def classify(text: str) -> Optional[str]:
    low = (text or "").lower()
    if any(m in low for m in CAPACITY_MARKERS):
        return "capacity"
    if any(m in low for m in AUTH_MARKERS):
        return "auth"
    return None


def setup_commands(state: str, *, has_uv: bool = True) -> list[str]:
    install = [] if has_uv else ["curl -LsSf https://astral.sh/uv/install.sh | sh"]
    install.append("uv tool install google-colab-cli")
    signin = ["colab --auth oauth2 sessions"]
    if state == "unsupported":
        return ["wsl --install -d Ubuntu-24.04"] + install + signin
    if state == "missing_cli":
        return install + signin
    if state == "kernel_client":
        return [
            "uv tool install --force google-colab-cli --with "
            "'jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git'"
        ]
    if state == "signed_out":
        return signin
    return []


def capability(runner: Optional[Runner] = None) -> dict:
    """Whether a usable, signed-in ``colab`` CLI is reachable, plus the one-time setup if not."""
    runner = runner or detect_runner()
    base = {
        "runner": runner.kind if runner else None,
        "distro": runner.distro if runner else None,
        "auth": None,
        "detail": None,
    }
    if runner is None:
        state = "unsupported"
        return {
            **base,
            "state": state,
            "ready": False,
            "message": "The Colab CLI runs on Linux and macOS. On Windows it needs WSL, which was not found.",
            "setup": setup_commands(state, has_uv = False),
        }
    try:
        result = run_shell(runner, PROBE_SCRIPT, timeout = PROBE_TIMEOUT)
    except LaunchError as exc:
        return {**base, "state": "unsupported", "ready": False, "message": str(exc), "setup": setup_commands("unsupported", has_uv = False)}
    facts: dict[str, str] = {}
    other: list[str] = []
    for line in result.output.splitlines():
        line = _clean(line)
        if line.startswith(PROBE_MARKER):
            key, _, value = line[len(PROBE_MARKER):].partition("=")
            facts[key] = value
        elif line.strip():
            other.append(line)
    detail = "\n".join(other[-8:]) or None
    has_uv = facts.get("uv") == "yes"
    if not facts:
        state = "unsupported"
        message = (
            "Could not run bash in WSL. Install a Linux distribution, or set "
            "UNSLOTH_COLAB_WSL_DISTRO to one that has bash."
            if runner.kind == "wsl"
            else "Could not run bash."
        )
    elif facts.get("cli") == "missing":
        state, message = "missing_cli", "The Colab CLI (google-colab-cli) is not installed."
    elif facts.get("kernel_client") == "bad":
        state = "kernel_client"
        message = "The Colab CLI has the wrong jupyter-kernel-client, so running code on the VM would fail."
    elif facts.get("auth") in (None, "none"):
        state, message = "signed_out", "The Colab CLI is installed but not signed in to Google."
    elif facts.get("sessions_rc") != "0":
        state = "signed_out"
        message = "The Colab CLI's saved Google sign-in was rejected. Sign in again."
    else:
        state, message = "ready", "The Colab CLI is installed and signed in."
    return {
        **base,
        "auth": facts.get("auth") if facts.get("auth") not in (None, "none") else None,
        "detail": detail,
        "state": state,
        "ready": state == "ready",
        "message": message,
        "setup": setup_commands(state, has_uv = has_uv),
    }


def parse_link(output: str) -> dict:
    """The ``UNSLOTH_LINK {...}`` line the start script prints; raises LaunchError on a reported error."""
    payload = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(LINK_MARKER):
            try:
                payload = json.loads(line[len(LINK_MARKER):])
            except ValueError:
                payload = None
    if not isinstance(payload, dict):
        raise LaunchError("The Colab VM never reported a link for Unsloth Studio.")
    if payload.get("error"):
        raise LaunchError(f"Unsloth Studio did not start on the Colab VM: {_clean(str(payload['error']))}")
    url, key = payload.get("url"), payload.get("api_key")
    if not isinstance(url, str) or not url.startswith("https://") or not isinstance(key, str) or not key:
        raise LaunchError("The Colab VM reported an incomplete link.")
    return {"url": url, "api_key": key}


def session_alive(output: str, session: str) -> bool:
    """`colab status -s NAME` exits 0 for a missing session too; only a real status line counts."""
    return bool(
        re.search(r"\[%s\]\s*\S" % re.escape(session), output or "")
        and re.search(r"Hardware:\s*[A-Za-z0-9-]+", output or "")
    )


def _colab(session: str, auth: Optional[str], *args: str) -> str:
    # Session names are validated to [a-z0-9_-], so the unquoted ~ path is safe and still expands.
    parts = ["colab", "--config", f"{CONFIG_DIR}/{session}.json"]
    if auth:
        parts += ["--auth", shlex.quote(auth)]
    parts += [shlex.quote(a) for a in args]
    return f"mkdir -p {CONFIG_DIR} && " + " ".join(parts)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LaunchJob:
    id: str
    name: str
    gpu: str
    session: str
    stage: str = "allocating"
    state: str = "running"  # running | ready | failed | cancelled
    error: Optional[str] = None
    setup: list[str] = field(default_factory = list)
    instance_id: Optional[str] = None
    started_at: str = field(default_factory = _now)
    finished_at: Optional[str] = None
    log: deque = field(default_factory = lambda: deque(maxlen = LOG_LINES))
    cancel: threading.Event = field(default_factory = threading.Event)
    _log_lock: threading.Lock = field(default_factory = threading.Lock, repr = False)

    def add(self, line: str) -> None:
        with self._log_lock:
            self.log.append(line)

    def public(self) -> dict:
        with self._log_lock:
            log = list(self.log)
        return {
            "id": self.id,
            "name": self.name,
            "gpu": self.gpu,
            "session": self.session,
            "stage": self.stage,
            "state": self.state,
            "error": self.error,
            "setup": list(self.setup),
            "instance_id": self.instance_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": log,
        }


_lock = threading.Lock()
_job: Optional[LaunchJob] = None


def current_job() -> Optional[dict]:
    with _lock:
        return _job.public() if _job else None


def start_launch(gpu: str, name: str) -> dict:
    global _job
    gpu = (gpu or "").strip().upper()
    if gpu not in GPUS:
        raise ValueError(f"GPU must be one of {', '.join(GPUS)}.")
    name = linked_instances_db.validate_name(name)
    if linked_instances_db.get_instance_by_name(name) is not None:
        raise ValueError(f"A linked instance named '{name}' already exists.")
    session = f"{SESSION_PREFIX}{name}"
    if linked_instances_db.get_colab_session(session) is not None:
        raise ValueError(f"A Colab session for '{name}' is already running. Stop it first.")
    with _lock:
        if _job and _job.state == "running":
            raise ValueError("A Colab launch is already in progress.")
        _job = LaunchJob(id = uuid.uuid4().hex, name = name, gpu = gpu, session = session)
        job = _job
    threading.Thread(target = _run_job, args = (job,), name = "colab-launch", daemon = True).start()
    return job.public()


def cancel_launch() -> Optional[dict]:
    with _lock:
        job = _job
    if job and job.state == "running":
        job.cancel.set()
    return current_job()


def _step(job: LaunchJob, runner: Runner, script: str, *, stdin: Optional[str] = None, timeout: float) -> ShellResult:
    if job.cancel.is_set():
        raise Cancelled()

    def on_line(line: str) -> None:
        line = _clean(line)
        if line and not line.startswith(LINK_MARKER):
            job.add(line)

    result = run_shell(runner, script, stdin = stdin, timeout = timeout, on_line = on_line, cancel = job.cancel)
    if job.cancel.is_set():
        raise Cancelled()
    if result.timed_out:
        raise LaunchError(f"Timed out after {int(timeout)}s while {job.stage}.")
    return result


def _run_job(job: LaunchJob) -> None:
    runner = detect_runner()
    allocated = False
    try:
        cap = capability(runner)
        if not cap["ready"]:
            job.setup = cap["setup"]
            raise LaunchError(cap["message"])
        auth = cap["auth"]
        # Recorded before `colab new` returns: a timed-out allocation still bills and must be stoppable.
        linked_instances_db.record_colab_session(
            job.session, job.name, job.gpu, auth = auth, runner = runner.kind, distro = runner.distro
        )
        allocated = True
        job.add(f"Allocating a Colab {job.gpu} VM ({job.session})...")
        result = _step(job, runner, _colab(job.session, auth, "new", "-s", job.session, "--gpu", job.gpu), timeout = ALLOCATE_TIMEOUT)
        if result.returncode != 0:
            kind = classify(result.output)
            if kind == "capacity":
                raise LaunchError(
                    f"Colab has no {job.gpu} for this account right now. Try a smaller GPU, or stop "
                    "other Colab sessions (browser tabs count too) and retry."
                )
            if kind == "auth":
                job.setup = setup_commands("signed_out")
                raise LaunchError("Colab rejected the saved Google sign-in. Sign in again.")
            raise LaunchError(f"colab new failed: {_last_line(result.output)}")

        job.stage = "installing"
        install = INSTALL_SCRIPT.format(repo = REPO_URL, branch = REPO_BRANCH)
        result = _step(
            job, runner,
            _colab(job.session, auth, "exec", "-s", job.session, "-", "--timeout", str(INSTALL_CELL_TIMEOUT)),
            stdin = install, timeout = INSTALL_CELL_TIMEOUT + 300,
        )
        # `colab exec` exits 0 even when the cell raised; only the marker counts.
        if INSTALL_OK not in result.output:
            raise LaunchError(f"Installing Unsloth Studio on the VM failed: {_last_line(result.output)}")

        job.stage = "starting"
        start = START_SCRIPT.format(launcher = LAUNCHER_SCRIPT)
        # The printed link carries the API key, so drop the CLI's local history of this session.
        script = (
            _colab(job.session, auth, "exec", "-s", job.session, "-", "--timeout", str(START_CELL_TIMEOUT))
            + f"; rc=$?; rm -f {HISTORY_DIR}/{job.session}.jsonl; exit $rc"
        )
        result = _step(job, runner, script, stdin = start, timeout = START_CELL_TIMEOUT + 120)
        link = parse_link(result.output)

        job.stage = "linking"
        from core.inference import linked_instances as linked

        base_url = linked.normalize_base_url(link["url"])
        instance = linked_instances_db.create_instance(job.name, base_url, link["api_key"])
        job.instance_id = instance["id"]
        linked_instances_db.set_colab_session_instance(job.session, instance["id"])
        job.add(f"Linked as @{job.name} ({base_url}).")
        job.stage = "ready"
        job.state = "ready"
        allocated = False
    except Cancelled:
        job.state = "cancelled"
        job.add("Cancelled.")
    except Exception as exc:
        job.state = "failed"
        job.error = _clean(str(exc)) or type(exc).__name__
        job.add(f"Failed: {job.error}")
        logger.warning("Colab launch %s failed: %s", job.session, job.error)
    finally:
        if allocated:
            job.add(f"Stopping {job.session} so it does not keep billing...")
            try:
                _stop_session(job.session, drop_link = True)
                job.add("Stopped.")
            except Exception as exc:
                job.add(
                    f"Could not stop {job.session}: {_clean(str(exc))}. Stop it from colab.research.google.com."
                )
        job.finished_at = _now()


def _last_line(output: str) -> str:
    lines = [_clean(l) for l in (output or "").splitlines() if l.strip()]
    return lines[-1][:300] if lines else "no output"


def _stop_session(session: str, *, drop_link: bool) -> None:
    record = linked_instances_db.get_colab_session(session)
    runner = detect_runner()
    if record and record.get("runner") == "wsl":
        runner = Runner("wsl", record.get("distro"))
    if runner is None:
        raise LaunchError("The Colab CLI is not reachable from this machine.")
    auth = record.get("auth") if record else None
    result = run_shell(runner, _colab(session, auth, "stop", "-s", session), timeout = STOP_TIMEOUT)
    if result.timed_out or (result.returncode != 0 and "not found" not in result.output.lower()):
        status = run_shell(runner, _colab(session, auth, "status", "-s", session), timeout = PROBE_TIMEOUT)
        if status.timed_out or session_alive(status.output, session):
            raise LaunchError(f"colab stop failed: {_last_line(result.output)}")
    if record and drop_link and record.get("instance_id"):
        linked_instances_db.delete_instance(record["instance_id"])
        try:
            from core.inference import linked_instances as linked
            linked.forget(record["instance_id"])
        except Exception:
            pass
    linked_instances_db.delete_colab_session(session)


def list_sessions() -> list[dict]:
    return linked_instances_db.list_colab_sessions()


def stop_session(session: str) -> None:
    """Stop the VM and remove its link: the tunnel URL and key die with the VM."""
    if linked_instances_db.get_colab_session(session) is None:
        raise KeyError(session)
    with _lock:
        job = _job
    if job and job.session == session and job.state == "running":
        # The job's own teardown stops the VM.
        job.cancel.set()
        return
    _stop_session(session, drop_link = True)


@atexit.register
def _stop_unfinished_launch() -> None:
    job = _job
    if job and job.state == "running":
        job.cancel.set()
        for _ in range(60):
            if job.finished_at:
                break
            time.sleep(1)
