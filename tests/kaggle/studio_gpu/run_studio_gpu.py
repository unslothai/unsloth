# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Unsloth Studio, end to end, on a real CUDA GPU.

Unsloth has no CUDA coverage anywhere in CI. Every Unsloth workflow in this
repo runs on ``ubuntu-latest``, ``macos-15`` or ``windows-latest``; macOS
gives Metal some hardware, and the CUDA path is exercised by nothing. The
existing inference smoke deliberately uses a 270M GGUF *because* it has to
decode on a CPU. This payload is the other half: it runs Unsloth on a Kaggle
T4 and asserts the three things that only a GPU can answer.

What it asserts
---------------
**A. GGUF inference is on the GPU, and tool calling works.** A model is
loaded with an explicit manual GPU-layer pin and then the payload tries three
independent ways to catch a CPU fallback -- process-level VRAM from
nvidia-smi, llama.cpp's own offload line, and device-wide VRAM growth. See
``gpu_assert.offload_verdict``: "the model returned text" is not evidence,
and no evidence at all is a failure rather than a pass. Then tool calling,
through ``/v1/chat/completions`` with ``tool_choice: "required"``, asserting
the model came back with ``finish_reason == "tool_calls"`` and a parseable
argument object.

**B. A LoRA training run finishes and leaves an adapter.** Started through
``POST /api/train/start`` -- the same call Unsloth's own Train button makes --
and judged on three things, none of which is the phase alone: the run reaches
phase ``completed``, its ``metric_history`` holds a loss for every step it
claimed to take, and ``adapter_model.safetensors`` exists on disk above a size
floor. A run that formats its dataset down to zero rows reaches ``completed``
too, which is why the step count is checked; a save that silently no-ops
leaves a config and no weights, which is why the file is checked.

**C. GGUF export runs against a CUDA llama.cpp build and the result loads.**
The adapter from B is exported to GGUF, the output is checked for the GGUF
magic rather than merely for existence, and then it is loaded back into
Unsloth and asked to generate. "It loads" is asserted by loading it, not by
its file size.

Then the repo's existing ``tests/studio/playwright_chat_ui.py`` is driven
against the same server, so the browser path is exercised by the driver that
already exists rather than by new automation. It runs LAST because its final
phase clicks "Stop server" and asserts the port closes.

Kaggle specifics
----------------
``/kaggle/working`` is 19.5 GB and is also what ``kernels output`` ships
back; the home directory and ``/tmp`` share a ~1 TB overlay. So
``UNSLOTH_STUDIO_HOME``, the HF cache, the llama.cpp build and every model
live on the overlay, and only the evidence -- kilobytes of JSON, a log tail
and screenshots -- is ever written under ``/kaggle/working``.

No credential is printed. The bootstrap password is read from Unsloth's own
auth directory, handed to ``/api/auth/login``, and never logged, never
returned, and never written to the report or the evidence bundle.

Usage:
    python run_studio_gpu.py --outdir /kaggle/working/studio_gpu_out \\
        --repo-root ~/unsloth --studio-home ~/studio_home
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import secrets as secrets_module
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gpu_assert import (  # noqa: E402
    gguf_magic_ok,
    install_kind,
    llama_cpp_marker,
    is_cuda_install,
    offload_verdict,
    parse_compute_apps,
)
from studio_client import (  # noqa: E402
    Studio,
    StudioError,
    adapter_verdict,
    export_verdict,
    health_is_ready,
    newest_gguf,
    nonfinite_losses,
    trained_steps,
    training_verdict,
    wait_for,
)

RESULT_PREFIX = "T4_SMOKE_REPORT "
EVIDENCE_PREFIX = "STUDIO_GPU_EVIDENCE_B64 "

# Base64 characters per printed line. Large enough that a multi-megabyte
# bundle is not tens of thousands of lines, small enough that no single line
# is unreasonable for a notebook cell output to carry.
EVIDENCE_CHUNK = 4096

# Ceiling on the evidence bundle. It travels back inside the executed
# notebook's cell output, which is the only channel the shared launcher
# collects, so it competes with the notebook itself for the download.
MAX_EVIDENCE_BYTES = 2_500_000

# Last resort when the redacted logs alone are over the cap: keep the tail of
# each, which is where a failure's traceback is, rather than dropping them.
MAX_LOG_TAIL_BYTES = 400_000

# Free space the payload refuses to start without. A CUDA llama.cpp bundle,
# a torch stack, two models and a merged 16-bit export do not fit in much
# less, and running out halfway produces a failure that reads like a code bug.
MIN_FREE_GB = 25.0

# The CUDA llama.cpp bundle is a large download and an extraction, and it runs
# once per session before anything needs it. Generous, because the cost of
# being too tight is a red that reads like a selection bug.
LLAMA_CPP_INSTALL_TIMEOUT_S = 900.0
# How long to let VRAM fall after an unload before calling it the baseline.
# 12 x 2.5s bounds the wait at 30s, which is well past the ~3s a llama-server
# takes to exit on the models this harness loads, without stalling the run if
# something else on the card is holding memory.
VRAM_SETTLE_POLL_S = 2.5
VRAM_SETTLE_SAMPLES = 12
# Driver readings jitter by a few MiB between samples; only a real drop counts.
VRAM_SETTLE_TOLERANCE_MIB = 16.0

CANARY = "__UNSLOTH_STUDIO__!!!"

# The tool the model is asked to call. Deliberately trivial and deliberately
# not answerable from parametric knowledge, so "it replied with prose" and
# "it emitted a tool call" cannot be confused.
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def log(msg: str) -> None:
    print(f"[studio-gpu] {msg}", flush = True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, answering an absent binary the way a failure answers.

    Every caller here judges on ``returncode``, and a box with no nvidia-smi
    at all -- a session Kaggle handed no driver -- otherwise raised
    FileNotFoundError out of ``environment()`` inside ``finish()``, so the run
    ended with a traceback and no report instead of a preflight that says
    there is no GPU.
    """
    try:
        return subprocess.run(cmd, capture_output = True, text = True, **kw)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", f"{type(exc).__name__}: {exc}")


def nvidia_used_mib() -> float | None:
    """Device-wide VRAM in use, summed over every visible GPU."""
    proc = run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        timeout = 60,
    )
    if proc.returncode != 0:
        return None
    total = 0.0
    seen = False
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            total += float(line)
            seen = True
        except ValueError:
            continue
    return total if seen else None


def cli_run_gpu_failure(
    apps_before: dict[int, int] | None,
    apps_after: dict[int, int] | None,
    baseline: float | None,
    settled: float | None,
) -> tuple[str | None, dict]:
    """Did a model actually reach the card? Returns (failure or None, detail).

    WHICH RULER, and this one has now been wrong in both directions.

    The device total is a SHARED reading. On kernel
    unsloth-probe-studio-full2-815a0c it was sampled too early and read 0.0 MiB
    on a server that did have the weights; the fix was to sample after a served
    completion. On unsloth-probe-full-concurrent-417238 it read **-182.0 MiB**
    -- and the same report carried `compute_apps {"6841": 2628}`, so the model
    was on the card and 2.6 GB of it. That run was the first with
    --studio-concurrent, so a training leg shared the card and freed memory
    inside the window. A shared counter cannot attribute, and subtracting two of
    its samples is not a measurement of THIS process.

    So the verdict comes off the pids that APPEARED during the window, which is
    per-process and immune to a co-tenant. A pid already on the card before the
    launch is excluded, or a co-tenant holding gigabytes would satisfy the claim
    on its own -- which is the same failure in a new costume.

    The device delta is still recorded, and is still the fallback for an
    nvidia-smi that answers a total but cannot enumerate processes.
    """
    detail: dict = {}
    if baseline is not None and settled is not None:
        detail["vram_delta_mib"] = round(settled - baseline, 1)

    if apps_after is None:
        if baseline is None or settled is None:
            return "nvidia-smi did not answer, so GPU use is unmeasured", detail
        if settled - baseline < 200.0:
            return (
                f"device VRAM grew by {settled - baseline:.1f} MiB across the "
                f"launch and a served completion, and nvidia-smi could not "
                f"enumerate processes to attribute it -- `unsloth run` served "
                f"from the CPU"
            ), detail
        return None, detail

    before = apps_before or {}
    appeared = {pid: mib for pid, mib in apps_after.items() if pid not in before}
    detail["compute_apps_appeared"] = appeared
    grew = sum(appeared.values())
    detail["process_vram_mib"] = grew
    if grew < 200.0:
        return (
            f"no process appeared on the GPU holding more than {grew} MiB "
            f"across the launch and a served completion (before "
            f"{sorted(before)}, after {sorted(apps_after)}) -- `unsloth run` "
            f"served from the CPU"
        ), detail
    return None, detail


def nvidia_compute_apps() -> dict[int, int] | None:
    proc = run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        timeout = 60,
    )
    if proc.returncode != 0:
        return None
    return parse_compute_apps(proc.stdout)


def visible_device_indices() -> list[int] | None:
    """The physical card indices CUDA_VISIBLE_DEVICES exposes, or None if unset.

    An empty string is a deliberate "no cards", which is different from unset
    and must not read as "all of them".
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            # A UUID form (GPU-xxxx) selects a card this cannot map to an
            # nvidia-smi row index. Returning None would report every card as
            # usable, which is the failure being fixed, so an unparseable entry
            # counts as one card rather than as all of them.
            out.append(-1)
    return out


def gpu_inventory() -> list[str]:
    """The cards THIS PROCESS can use, not the cards the box has.

    `nvidia-smi` enumerates PHYSICAL devices and ignores CUDA_VISIBLE_DEVICES,
    and reading it as "what is available" produced a false claim on kernel
    unsloth-probe-full-concurrent-417238. build_kernel.py pins every payload
    with `CUDA_VISIBLE_DEVICES = str(gpu_index)`, and under --studio-concurrent
    that includes Studio -- so the run recorded `cards_visible: 2` and
    `tensor_split_over_two_cards: True` for a server that had ONE card, and sent
    `tensor_split: [1.0, 1.0]` asking llama.cpp to split across a device that
    was not there. It loaded anyway, so the assertion passed green.

    That field exists precisely to stop a check keeping its name while testing
    less. It was sized from the wrong instrument and did exactly that.
    """
    proc = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        timeout = 60,
    )
    if proc.returncode != 0:
        return []
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    visible = visible_device_indices()
    if visible is None:
        return rows
    # Index into the physical rows where we can; an index nvidia-smi does not
    # have, or the UUID sentinel, still counts as one card so the COUNT stays
    # right even when the description cannot be recovered.
    return [rows[i] if 0 <= i < len(rows) else "visible GPU (details unavailable)" for i in visible]


def log_paths(server_log: Path, studio_home: Path) -> list[Path]:
    """Every file Unsloth or a llama-server child may be writing to.

    llama.cpp's offload line lands in whichever of these the server's stderr
    was wired to, and which one that is depends on how Unsloth was started, so
    both are read.
    """
    candidates = [server_log]
    log_dir = studio_home / "logs"
    if log_dir.is_dir():
        candidates += sorted(log_dir.rglob("*.log"))
    return candidates


def log_marks(server_log: Path, studio_home: Path) -> dict[str, int]:
    """Current size of every log, to read forward from later.

    Taken immediately before a model load so the evidence gathered afterwards
    belongs to THAT load. Without it, a reload whose own log line never
    appeared inherited the previous load's ``offloaded N/M layers`` and the
    verdict passed on evidence from a different model.
    """
    marks: dict[str, int] = {}
    for path in log_paths(server_log, studio_home):
        try:
            marks[str(path)] = path.stat().st_size
        except OSError:
            continue
    return marks


def studio_log_text(
    server_log: Path,
    studio_home: Path,
    tail_bytes: int = 4_000_000,
    *,
    since: dict[str, int] | None = None,
) -> str:
    """Everything Unsloth and its llama-server children wrote, as one string.

    ``since`` is a ``log_marks()`` snapshot; each file is then read from the
    offset it had then, so only what this load produced comes back. A file
    that has since shrunk was rotated or truncated, and is read whole.
    """
    parts: list[str] = []
    for path in log_paths(server_log, studio_home):
        start = (since or {}).get(str(path), 0)
        try:
            with open(path, "rb") as fh:
                fh.seek(0, io.SEEK_END)
                size = fh.tell()
                if start > size:
                    start = 0
                fh.seek(max(start, size - tail_bytes))
                parts.append(fh.read().decode("utf-8", errors = "replace"))
        except OSError:
            continue
    return "\n".join(parts)


def llama_server_pids() -> list[int]:
    """PIDs of the llama-server children Unsloth started, from /proc.

    ``GET /api/inference/status`` does not carry one: ``InferenceStatusResponse``
    declares neither ``llama_server_pid`` nor ``pid``, and FastAPI drops
    anything the response model does not declare, so the process-level VRAM
    probe never had a pid to match and could never contribute evidence.
    Reading the process table gives it one back.
    """
    pids: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return pids
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors = "replace")
        except OSError:
            continue
        argv0 = cmdline.split("\x00", 1)[0]
        if "llama-server" in Path(argv0).name:
            pids.append(int(entry.name))
    return pids


class Payload:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents = True, exist_ok = True)
        self.art_dir = self.outdir / "playwright"
        self.studio_home = Path(args.studio_home).expanduser().resolve()
        self.repo_root = Path(args.repo_root).expanduser().resolve()
        self.server_log = self.outdir / "studio.log"
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.studio = Studio(self.base_url)
        self.proc: subprocess.Popen | None = None
        self.assertions: list[dict] = []
        self.failures: list[str] = []
        self.started = time.time()
        # Every secret this run has minted or read. Unsloth's startup banner
        # prints the bootstrap password to stdout, and stdout here is
        # studio.log, and studio.log travels home in the evidence bundle. The
        # value is ephemeral and local to a kernel that is destroyed minutes
        # later, but "ephemeral" is not the same as "fine to publish in a CI
        # artifact", so every log that leaves this machine is scrubbed of it.
        self.secrets: set[str] = set()

        # Resolved and registered HERE, before anything can log it. `auto`
        # mints a fresh one per run rather than carrying a constant, because a
        # constant in a repo is a credential whether or not it is ever reachable.
        if self.args.studio_password == "auto":
            self.args.studio_password = "ci-" + secrets_module.token_urlsafe(18)
        if self.args.studio_password:
            self.secrets.add(self.args.studio_password)

    # ---------------------------------------------------------------- report

    def record(self, name: str, passed: bool, detail: dict) -> bool:
        # Per-assertion wall clock, because this payload is now the longest
        # thing in the kernel and nothing said where the time went. Measured on
        # unsloth-probe-full-concurrent-417238: 1487.5s across 19 assertions,
        # and the report carried no breakdown at all, so every question about
        # shortening Studio was guesswork.
        #
        # It is elapsed-since-the-previous-record, NOT a timer around the
        # assertion body, and the name says so. The assertions run back to
        # back, so the two are the same to within the bookkeeping between them;
        # the first entry measures from process start, which is the setup
        # before any assertion and is worth seeing rather than hiding.
        #
        # Both reads are `getattr` with a default: `record` is driven directly
        # by several CPU guards against stub objects that have no clock, and a
        # hard `self.started` turns every one of them into an AttributeError
        # about timing instead of a result about the rule they test.
        now = time.time()
        started = getattr(self, "started", now)
        previous = getattr(self, "_last_record_at", started)
        self._last_record_at = now
        entry = {
            "name": name,
            "passed": bool(passed),
            "seconds_since_previous": round(now - previous, 1),
            "at_seconds": round(now - started, 1),
            **detail,
        }
        self.assertions.append(entry)
        for reason in detail.get("failures", []) or []:
            self.failures.append(f"{name}: {reason}")
        log(f"{'PASS' if passed else 'FAIL'} {name}")
        for reason in detail.get("failures", []) or []:
            log(f"     {reason}")
        return passed

    def report(self) -> dict:
        return {
            "label": self.args.label,
            "model": self.args.chat_model,
            "passed": not self.failures,
            "failures": self.failures,
            "assertions": self.assertions,
            "environment": self.environment(),
            "config": {
                "chat_model": self.args.chat_model,
                "chat_variant": self.args.chat_variant,
                "train_model": self.args.train_model,
                "max_steps": self.args.max_steps,
                "quantization": self.args.quantization,
                "gpu_layers": self.args.gpu_layers,
            },
            "seconds": round(time.time() - self.started, 1),
        }

    def environment(self) -> dict:
        env: dict = {"gpus": gpu_inventory()}
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                env["gpu_name"] = torch.cuda.get_device_name(0)
                env["gpu_capability"] = ".".join(
                    str(x) for x in torch.cuda.get_device_capability(0)
                )
        except Exception as exc:  # noqa: BLE001
            env["torch_error"] = f"{type(exc).__name__}: {exc}"
        marker = llama_cpp_marker(self.studio_home)
        env["llama_cpp_install_kind"] = install_kind(marker)
        return env

    # ------------------------------------------------------------- preflight

    def preflight(self) -> bool:
        failures: list[str] = []
        detail: dict = {}

        gpus = gpu_inventory()
        detail["gpus"] = gpus
        if not gpus:
            failures.append("nvidia-smi listed no GPU, so this is not a GPU session")

        try:
            import torch
            detail["cuda_available"] = bool(torch.cuda.is_available())
            if not torch.cuda.is_available():
                failures.append("torch.cuda.is_available() is False")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"could not import torch: {type(exc).__name__}: {exc}")

        self.studio_home.mkdir(parents = True, exist_ok = True)
        free_gb = shutil.disk_usage(self.studio_home).free / 1e9
        detail["studio_home"] = str(self.studio_home)
        detail["free_gb"] = round(free_gb, 1)
        if free_gb < MIN_FREE_GB:
            failures.append(
                f"only {free_gb:.1f} GB free at {self.studio_home}, below the "
                f"{MIN_FREE_GB} GB floor. On Kaggle this usually means "
                f"UNSLOTH_STUDIO_HOME landed under /kaggle/working (19.5 GB) "
                f"instead of the home/tmp overlay"
            )

        driver = self.repo_root / "tests" / "studio" / "playwright_chat_ui.py"
        detail["chat_driver"] = str(driver)
        if not driver.is_file():
            failures.append(f"the existing chat UI driver is not where it should be: {driver}")

        detail["failures"] = failures
        return self.record("preflight", not failures, detail)

    # ---------------------------------------------------------------- server

    def studio_command(self) -> list[str]:
        """The `unsloth` entry point of the interpreter running this payload.

        NOT ``shutil.which("unsloth")``. This payload runs under the Unsloth
        venv's Python and that venv's ``bin`` is not on PATH, so a global
        ``unsloth`` anywhere on PATH would win the lookup and the run would
        measure some other installation instead of the checkout under test.
        The console script sits next to ``sys.executable``; if it is missing,
        the same interpreter runs the module directly.
        """
        bin_dir = Path(sys.executable).parent
        for name in ("unsloth", "unsloth.exe"):
            candidate = bin_dir / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                head = [str(candidate)]
                break
        else:
            head = [sys.executable, "-c", "from unsloth_cli import app; app()"]
        cmd = head + ["studio", "-H", "127.0.0.1", "-p", str(self.args.port)]
        if self.args.studio_password:
            # The HEADLESS path, and it is a feature rather than a convenience
            # here: `--password` sets the INITIAL admin password when none is
            # set yet, which is exactly the shape a server started by a script
            # is in. Without it the only way in is the bootstrap password
            # Studio seeds into a file and prints to its own log, and a
            # deployment that has to read a log to log in is not one anybody
            # scripts twice.
            #
            # The value is generated per run and registered as a secret before
            # this is ever called, so it is scrubbed out of every log and
            # evidence bundle that leaves the machine. It is still visible in
            # this session's process list, which the flag's own help says; that
            # is acceptable for a single-tenant CI kernel and would not be on a
            # shared host.
            cmd += ["--password", self.args.studio_password]
        return cmd

    def start_server(self) -> bool:
        # An absent auth directory is what re-seeds the bootstrap password;
        # `reset-password` does not. Same thing the repo's own
        # boot-studio-api-only.sh does, for the same reason.
        shutil.rmtree(self.studio_home / "auth", ignore_errors = True)

        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(self.studio_home)
        env.setdefault("HF_HOME", str(self.studio_home / "cache" / "huggingface"))
        env["PYTHONUNBUFFERED"] = "1"
        env["UNSLOTH_DISABLE_STATISTICS"] = "1"

        cmd = self.studio_command()
        log(f"starting Unsloth: {' '.join(cmd)}")
        # Append, never truncate. assert_chat_ui() restarts the server to
        # re-seed the account, and a "wb" here threw away every backend log
        # from the inference, training and export assertions before the
        # evidence bundle was built -- so a GPU failure followed by the normal
        # UI phase shipped an artifact containing only the UI session.
        handle = open(self.server_log, "ab")
        self.proc = subprocess.Popen(
            cmd,
            cwd = str(self.repo_root),
            env = env,
            stdout = handle,
            stderr = subprocess.STDOUT,
        )

        ok, last, reason = wait_for(
            probe = lambda: self.studio.get("/api/health", auth = False)[1],
            accept = health_is_ready,
            deadline_s = self.args.health_deadline,
            interval_s = 2.0,
            alive = self.server_alive,
        )
        detail: dict = {"health": last if isinstance(last, dict) else str(last)[:400]}
        failures = []
        if not ok:
            failures.append(f"Unsloth never became ready: {reason}")
            failures.append("last 40 lines of the server log: " + self.log_tail(40))
        detail["failures"] = failures
        return self.record("studio_ready", ok, detail)

    def remember_bootstrap(self) -> str | None:
        """Learn the bootstrap password so it can be scrubbed, not so it can be used."""
        path = self.studio_home / "auth" / ".bootstrap_password"
        try:
            value = path.read_text(encoding = "utf-8").strip()
        except OSError:
            return None
        if value:
            self.secrets.add(value)
        return value or None

    def scrub(self, text: str) -> str:
        for secret in self.secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        return text

    def log_tail(self, lines: int) -> str:
        # Unsloth's startup banner prints the bootstrap password, and this tail
        # is put in front of a human on a pull request when startup fails.
        self.remember_bootstrap()
        try:
            text = self.server_log.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            return "(no server log)"
        return self.scrub(" | ".join(text.splitlines()[-lines:]))

    def authenticate(self) -> bool:
        failures: list[str] = []
        if self.args.studio_password:
            # Log in with the password we PASSED, which is the assertion: if
            # --password had been ignored, Studio would have seeded a bootstrap
            # password instead and this login would fail. A run that fell back
            # to the bootstrap on failure would pass while proving the flag does
            # nothing, so there is deliberately no fallback.
            try:
                self.studio.login(self.args.studio_password)
            except StudioError as exc:
                failures.append(
                    f"--password was passed to `unsloth studio` and logging in "
                    f"with it failed, so the flag did not take effect: {exc}"
                )
            return self.record(
                "authenticate", not failures, {"source": "--password", "failures": failures}
            )

        path = self.studio_home / "auth" / ".bootstrap_password"
        if not path.is_file():
            failures.append(f"no bootstrap password was seeded at {path}")
            return self.record("authenticate", False, {"failures": failures})
        # Read, use, drop. The value is never logged and never reaches the
        # report; it is remembered only so it can be scrubbed out of the logs
        # that do leave this machine.
        password = self.remember_bootstrap() or ""
        try:
            self.studio.login(password)
        except StudioError as exc:
            failures.append(str(exc))
        return self.record(
            "authenticate", not failures, {"source": "bootstrap", "failures": failures}
        )

    def server_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop_server(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout = 60)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    # ----------------------------------------------------------- assertion A

    def settled_baseline(self) -> float | None:
        """VRAM after anything already loaded has been evicted and freed.

        `POST /load` with `force` unloads the previous model as its first act,
        so sampling the baseline just before the request puts that release
        INSIDE the measured window. Run 7 read `device_vram_delta_mib: -1866.0`
        for the GGUF export probe: a 531 MB model loading while a 3004 MiB chat
        model left, reported as negative growth and scored as a failure to
        reach the GPU. The load was fine; the ruler was wrong.

        Unloading first and waiting for the number to stop falling makes the
        delta measure one thing. Best-effort throughout: an unload that fails
        leaves the old baseline behaviour rather than aborting the probe, and
        the delta is only ever evidence, never the sole assertion.
        """
        code, body = self.studio.get("/api/inference/status")
        active = None
        if code == 200 and isinstance(body, dict):
            # Field names from InferenceStatusResponse, not from guesswork.
            # `model_identifier` is documented as the LOADABLE identifier,
            # which is what /unload's `model_path` wants; `active_model` is a
            # display string and only a fallback. Run 8 read three names that
            # the response has never carried, so `active` was always None, no
            # unload was ever sent, and the delta came back byte-identical to
            # run 7 -- a fix that ran and did nothing.
            active = body.get("model_identifier") or body.get("active_model")
            if not active:
                loaded = body.get("loaded")
                if isinstance(loaded, list) and loaded:
                    active = loaded[0]
        if isinstance(active, str) and active:
            try:
                self.studio.post(
                    "/api/inference/unload",
                    {"model_path": active, "force_cancel_active": True},
                    timeout = self.args.load_timeout,
                )
            except StudioError:
                pass
        # Freeing is asynchronous: llama-server exits and the driver reclaims
        # afterwards, so an immediate read still sees the old model resident.
        # Settle on two consecutive samples that did not drop.
        previous = nvidia_used_mib()
        for _ in range(VRAM_SETTLE_SAMPLES):
            time.sleep(VRAM_SETTLE_POLL_S)
            current = nvidia_used_mib()
            if previous is None or current is None:
                return current
            if current >= previous - VRAM_SETTLE_TOLERANCE_MIB:
                return current
            previous = current
        return previous

    def load_model(self, model_path: str, *, variant: str | None, label: str) -> dict:
        """Load a GGUF with an explicit GPU pin and judge whether it landed there."""
        before = self.settled_baseline()
        body = {
            "model_path": model_path,
            "is_lora": False,
            "max_seq_length": 2048,
            # Manual, not Auto. Auto (`gpu_layers: -1`) delegates placement to
            # llama.cpp's fitter, and a fitter that decided on zero layers is
            # indistinguishable in the response from one that filled the card.
            # Pinning states the intent, so `status.gpu_layers` coming back
            # different is itself a finding.
            "gpu_memory_mode": "manual",
            "gpu_layers": self.args.gpu_layers,
            "force": True,
        }
        if variant:
            body["gguf_variant"] = variant

        failures: list[str] = []
        status_body: dict = {}
        # Everything already in the logs belongs to an EARLIER load. Only what
        # is written past this mark is evidence about this one.
        marks = log_marks(self.server_log, self.studio_home)
        try:
            self.studio.expect("POST", "/api/inference/load", body, timeout = self.args.load_timeout)
        except StudioError as exc:
            failures.append(str(exc))
        else:
            code, status_body = self.studio.get("/api/inference/status")
            if code != 200 or not isinstance(status_body, dict):
                failures.append(f"GET /api/inference/status returned HTTP {code}")
                status_body = {}

        after = nvidia_used_mib()
        delta = None if (before is None or after is None) else after - before

        # The status response declares no pid field, so the processes are
        # discovered here instead. See llama_server_pids().
        server_pids = llama_server_pids()
        verdict = offload_verdict(
            server_pid = None,
            server_pids = server_pids,
            compute_apps = nvidia_compute_apps(),
            log_text = studio_log_text(self.server_log, self.studio_home, since = marks),
            device_vram_delta_mib = delta,
            status = status_body,
        )
        failures += verdict["failures"]

        requested = self.args.gpu_layers
        effective = status_body.get("gpu_layers")
        if isinstance(effective, int) and effective not in (requested, -1):
            # Not a failure on its own: llama.cpp clamps a pin to the model's
            # block count, which is the common and correct reason for this.
            verdict["evidence"].append(
                f"requested gpu_layers={requested}, Unsloth reports {effective} "
                f"(a clamp to the model's layer count looks like this)"
            )

        return {
            "model": model_path,
            "variant": variant,
            "requested_gpu_layers": requested,
            "effective_gpu_layers": effective,
            "llama_server_pids": server_pids,
            "device_vram_delta_mib": None if delta is None else round(delta, 1),
            "evidence": verdict["evidence"],
            "positives": verdict["positives"],
            "failures": failures,
            "label": label,
        }

    def chat(self, messages: list[dict], **extra) -> tuple[int, object]:
        body = {
            "model": "default",
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0.0,
            "seed": 3407,
            "enable_thinking": False,
            **extra,
        }
        return self.studio.post("/v1/chat/completions", body, timeout = self.args.chat_timeout)

    def assert_gpu_inference(self) -> bool:
        detail = self.load_model(
            self.args.chat_model,
            variant = self.args.chat_variant,
            label = "chat",
        )
        if not detail["failures"]:
            code, payload = self.chat([{"role": "user", "content": "What is 1+1? Answer briefly."}])
            text = ""
            if code == 200 and isinstance(payload, dict):
                choices = payload.get("choices") or [{}]
                text = (choices[0].get("message") or {}).get("content") or ""
            detail["generated"] = text[:200]
            if code != 200:
                detail["failures"].append(f"/v1/chat/completions returned HTTP {code}")
            elif not text.strip():
                detail["failures"].append("the model on the GPU returned empty content")
        return self.record("gpu_inference", not detail["failures"], detail)

    def assert_server_flags(self) -> bool:
        """Reload with a quantized KV cache and a two-card split, and check what
        was APPLIED rather than what was asked for.

        Studio's own status distinguishes the two, which is the point: it
        carries `cache_type_kv` for the live server alongside a coverage field
        and a reason for anything it could not apply. So a request that was
        silently dropped is visible, and a check that only asserted "the load
        succeeded" would pass on exactly that.

        `tensor_split` over two T4s is the flag the brief asks about and the
        one nothing here has ever exercised. The split is sized to the cards
        that are VISIBLE, because --studio-concurrent pins this half to one
        card so it can share with a training leg, and the report says which of
        the two it did: a single-card run records
        `tensor_split_over_two_cards: false` with a note rather than passing
        under the same name. A check that keeps its name while quietly testing
        less is the failure this file exists against.

        The context length is pinned to `--studio-ctx` (2048 by default) rather
        than left at the model default, because an unconstrained context on a
        14.56GB card is how a KV-cache test turns into an OOM about something
        else.
        """
        failures: list[str] = []
        cards = gpu_inventory()
        detail: dict = {"requested": {}, "cards_visible": len(cards)}
        body = {
            "model_path": self.args.chat_model,
            "is_lora": False,
            "max_seq_length": self.args.studio_ctx,
            "gpu_memory_mode": "manual",
            "gpu_layers": self.args.gpu_layers,
            "force": True,
            # q8_0 rather than a fancier width: it is the one every llama.cpp
            # build supports, so a refusal here is about the MODEL's cache
            # layout and not about the binary.
            "cache_type_kv": "q8_0",
            # One weight per VISIBLE card, even. The values are relative
            # weights, not byte counts.
            #
            # Sized rather than hardcoded to two, because --studio-concurrent
            # pins this half to a single card so it can share with a training
            # leg: build_kernel.py's run_one sets CUDA_VISIBLE_DEVICES to the
            # card it was admitted on. Sending [1.0, 1.0] to a one-card server
            # asks llama.cpp to split across a device that is not there, and
            # what comes back is a failure about the load rather than about the
            # flag.
            "tensor_split": [1.0] * max(1, len(cards)),
        }
        if self.args.chat_variant:
            body["gguf_variant"] = self.args.chat_variant
        detail["requested"] = {
            k: body[k] for k in ("max_seq_length", "cache_type_kv", "tensor_split")
        }
        # STATED, not silent. Under --studio-concurrent this half runs on one
        # card so it can share with a training leg, and a split over one device
        # is not the two-card flag the brief asks about. Recording it as
        # exercised when it was not is how a check keeps its name and loses its
        # meaning; a reader of this report can see which machine it ran on.
        detail["tensor_split_over_two_cards"] = len(cards) >= 2
        if len(cards) < 2:
            detail["tensor_split_note"] = (
                f"only {len(cards)} card visible, so the two-card tensor_split "
                f"was NOT exercised; the KV-cache type, the context pin and the "
                f"GPU residency below still were"
            )

        try:
            self.studio.expect("POST", "/api/inference/load", body, timeout = self.args.load_timeout)
        except StudioError as exc:
            failures.append(f"loading with server flags failed: {exc}"[:600])
            detail["failures"] = failures
            return self.record("server_flags", False, detail)

        code, status_body = self.studio.get("/api/inference/status")
        detail["status_code"] = code
        status = status_body if isinstance(status_body, dict) else {}
        applied = {
            k: status.get(k)
            for k in (
                "cache_type_kv",
                "context_length",
                "max_context_length",
                "gpu_layers",
                "is_mlx",
            )
        }
        detail["applied"] = applied

        # 1. the KV cache type actually in force. Reported rather than asserted
        #    equal: a model whose cache layout cannot be quantized is entitled
        #    to refuse, and Studio says so. What is NOT acceptable is silence.
        got_cache = (applied.get("cache_type_kv") or "").lower()
        if not got_cache:
            failures.append(
                "the status reports no cache_type_kv at all after a load that "
                "asked for q8_0, so there is no way to tell whether the request "
                "was honoured, downgraded or ignored"
            )
        elif got_cache != "q8_0":
            # Not a failure by itself -- but it must come with a reason.
            reason = status.get("kv_quant_reason") or status.get("mlx_kv_quant_reason")
            detail["cache_downgrade_reason"] = reason
            if not reason:
                failures.append(
                    f"q8_0 was requested and {got_cache!r} is in force, with no "
                    f"reason given. A silent downgrade is the failure this check "
                    f"exists for"
                )

        # 2. the context length is the one that was pinned. llama-server admits
        #    a prompt on n_ctx alone, so a server running at the model default
        #    behaves differently from one at 2048 and the difference is
        #    invisible in a chat response.
        ctx = applied.get("context_length")
        if ctx is None:
            failures.append("the status reports no context_length")
        elif int(ctx) > self.args.studio_ctx:
            failures.append(
                f"context_length is {ctx}, above the {self.args.studio_ctx} that "
                f"was requested, so the pin did not take"
            )

        # 3. it is still on the GPU. A tensor split that fell back to CPU would
        #    otherwise report a healthy server and prove nothing about either
        #    card.
        used = nvidia_used_mib()
        detail["gpu_used_mib"] = used
        if used is not None and used < 200:
            failures.append(
                f"only {used} MiB of GPU memory is in use after a "
                f"{len(cards)}-card tensor_split load, so this is running on "
                f"the CPU"
            )

        detail["failures"] = failures
        return self.record("server_flags", not failures, detail)

    def assert_compaction(self) -> bool:
        """A conversation past the window must COMPACT, and a short one must not.

        The context length is pinned to `--studio-ctx` by `assert_server_flags`
        immediately above, so this is the one place the payload knows what the
        window is and can overflow it deliberately.

        Studio reports the fit on the completion itself, as `context_truncated`
        with `dropped_messages`, so the claim is readable without a browser.
        The rule is a PAIR, and the second half is what makes the first mean
        anything:

        * a conversation built past the budget comes back 200 with
          `dropped_messages > 0` -- it was shortened, not refused;
        * a two-message conversation comes back with nothing dropped.

        Asserting only the first passes on a server that reports truncation
        unconditionally, which is indistinguishable from working and is the
        failure this file keeps being caught by. Asserting only the second
        passes on a server that never compacts at all and returns a
        context-length error instead.
        """
        failures: list[str] = []
        detail: dict = {}

        def _dropped(body) -> int | None:
            if not isinstance(body, dict):
                return None
            truncation = body.get("context_truncated")
            if not isinstance(truncation, dict):
                return 0
            try:
                return int(truncation.get("dropped_messages") or 0)
            except (TypeError, ValueError):
                return 0

        # Distinct filler, so nothing can be deduplicated or cached into
        # fitting. Sized past `studio_ctx` by a wide margin rather than a
        # narrow one: the budget subtracts the reply reserve and the template's
        # own framing, so a prompt that merely equals the context length is not
        # reliably over it.
        long_messages = []
        for i in range(40):
            long_messages.append(
                {
                    "role": "user",
                    "content": f"Fact {i}: the {i}th token of this transcript is "
                    + " ".join(f"w{i}x{j}" for j in range(40)),
                }
            )
            long_messages.append({"role": "assistant", "content": f"Noted fact {i}."})
        long_messages.append({"role": "user", "content": "Reply with one word."})

        # `context_overflow` defaults to "error", and that default is CORRECT:
        # a conversation past the window comes back 400 with
        # code=context_length_exceeded so a client's own trim loop can see it.
        # Compaction is a policy you ask for, and asking for it is the thing
        # under test. Measured the hard way on kernel
        # unsloth-probe-studio-full2-815a0c, where this assertion failed a
        # documented default. "truncate_oldest" is the policy that applies to a
        # plain chat; "truncate_middle" is limited to client-tool and
        # response_format passthrough (studio/backend/models/inference.py).
        code, body = self.chat(long_messages, max_tokens = 32, context_overflow = "truncate_oldest")
        detail["context_overflow"] = "truncate_oldest"
        detail["long_status"] = code
        detail["long_messages"] = len(long_messages)
        detail["long_truncation"] = (
            (body or {}).get("context_truncated") if isinstance(body, dict) else None
        )
        long_dropped = _dropped(body)
        if code != 200:
            failures.append(
                f"an over-length conversation returned HTTP {code} instead of "
                f"being compacted: {str(body)[:300]}"
            )
        elif not long_dropped:
            failures.append(
                f"a {len(long_messages)}-message conversation against a "
                f"{self.args.studio_ctx}-token window dropped nothing, so no "
                f"compaction happened and the reply used a prompt nobody sized"
            )

        # The POLICY control: the same over-length conversation with the
        # default `context_overflow` must be REFUSED. Without this, the check
        # above passes on a server that compacts everything regardless of what
        # was asked for, and the field name in the request would be decorative.
        code, body = self.chat(long_messages, max_tokens = 32)
        detail["long_status_default_policy"] = code
        if code != 400:
            failures.append(
                f"with the default context_overflow the same over-length "
                f"conversation returned HTTP {code} rather than 400. The "
                f"documented default is to refuse with "
                f"code=context_length_exceeded so a client's own trim loop can "
                f"see it, and if everything is compacted anyway then asking "
                f"for truncate_oldest above proved nothing"
            )

        # The LENGTH control, and it is not optional either: without it a
        # server that always claims truncation passes the first check.
        code, body = self.chat([{"role": "user", "content": "Say hi."}], max_tokens = 16)
        detail["short_status"] = code
        short_dropped = _dropped(body)
        detail["short_dropped"] = short_dropped
        if code != 200:
            failures.append(f"the short control returned HTTP {code}")
        elif short_dropped:
            failures.append(
                f"a two-message conversation reported {short_dropped} dropped "
                f"messages, so the field is not responsive to length and the "
                f"check above proves nothing"
            )

        detail["failures"] = failures
        return self.record("compaction", not failures, detail)

    def assert_api_key(self) -> bool:
        """Mint an API key and DRIVE it, which is the half that can be wrong.

        Creating a key proves the endpoint returns a string. Whether that
        string authenticates anything is a separate question, and the answer
        that matters: an API key Studio issues and then rejects is worse than
        no API key, because the failure surfaces in a user's integration rather
        than here.

        Three claims, in order of what each rules out:

        1. the key is minted and returned ONCE (the raw value is not
           retrievable later, by design);
        2. a request carrying ONLY that key succeeds -- the session bearer
           token is set aside for the call, or this would pass on the token and
           say nothing about the key;
        3. a request carrying a corrupted key FAILS. Without that, a server
           that ignores the header entirely passes claim 2.
        """
        failures: list[str] = []
        detail: dict = {}
        raw_key = None
        try:
            code, body = self.studio.post(
                "/api/auth/api-keys", {"name": f"kaggle-ci-{int(time.time())}"}
            )
            detail["create_status"] = code
            if code >= 400 or not isinstance(body, dict):
                failures.append(f"could not create an API key: {code} {str(body)[:200]}")
            else:
                raw_key = body.get("key")
                detail["has_key"] = bool(raw_key)
                detail["key_metadata"] = {
                    k: v
                    for k, v in (body.get("api_key") or {}).items()
                    if k in ("id", "name", "is_active", "expires_at")
                }
                if not raw_key:
                    failures.append("the create response carried no raw key")
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"creating an API key raised: {type(exc).__name__}: {exc}"[:300])

        if raw_key:
            # Registered before it is used anywhere, so it cannot reach a log.
            self.secrets.add(raw_key)
            saved = self.studio.token
            try:
                # The key AS the bearer, with the session token set aside. If
                # the session token were left in place this would pass on the
                # token and prove nothing.
                self.studio.token = raw_key
                code, body = self.studio.get("/api/auth/api-keys")
                detail["key_auth_status"] = code
                if code >= 400:
                    failures.append(
                        f"the API key Studio just issued does not authenticate: "
                        f"{code} {str(body)[:200]}"
                    )

                # And a corrupted key must be REJECTED. Without this, a server
                # that ignores the header entirely passes the check above.
                self.studio.token = raw_key[:-4] + "0000" if len(raw_key) > 8 else "bogus"
                code, _ = self.studio.get("/api/auth/api-keys")
                detail["bad_key_status"] = code
                if code < 400:
                    failures.append(
                        f"a corrupted API key was accepted ({code}), so the "
                        f"check above passes whatever is sent"
                    )
            except BaseException as exc:  # noqa: BLE001
                failures.append(f"driving the API key raised: {type(exc).__name__}: {exc}"[:300])
            finally:
                self.studio.token = saved

        detail["failures"] = failures
        return self.record("api_key", not failures, detail)

    def assert_tool_calling(self) -> bool:
        failures: list[str] = []
        detail: dict = {}
        code, payload = self.chat(
            [{"role": "user", "content": "What is the weather in Paris right now?"}],
            tools = [WEATHER_TOOL],
            tool_choice = "required",
            max_tokens = 256,
        )
        detail["http_status"] = code
        if code != 200 or not isinstance(payload, dict):
            failures.append(f"tool-calling completion returned HTTP {code}")
            detail["failures"] = failures
            return self.record("tool_calling", False, detail)

        choice = (payload.get("choices") or [{}])[0]
        detail["finish_reason"] = choice.get("finish_reason")
        calls = (choice.get("message") or {}).get("tool_calls") or []
        detail["tool_calls"] = [c.get("function", {}).get("name") for c in calls]

        if choice.get("finish_reason") != "tool_calls":
            failures.append(
                f"finish_reason was {choice.get('finish_reason')!r}, not 'tool_calls', "
                f"so the model answered in prose instead of calling the tool"
            )
        if not calls:
            failures.append("no tool_calls were returned")
        else:
            fn = calls[0].get("function") or {}
            if fn.get("name") != "get_weather":
                failures.append(f"called {fn.get('name')!r} rather than 'get_weather'")
            try:
                arguments = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                failures.append(f"tool-call arguments were not JSON: {exc}")
            else:
                detail["arguments"] = arguments
                if not arguments.get("city"):
                    failures.append("the tool call carried no 'city' argument")

        detail["failures"] = failures
        return self.record("tool_calling", not failures, detail)

    def assert_code_execution(self) -> bool:
        """The python tool must RUN, and the proof is a file on disk.

        `assert_tool_calling` above proves the model can EMIT a call. That is a
        different claim: a weather tool is never executed by Studio at all, the
        caller is expected to run it. The local `python` tool is executed by
        Studio itself, in a per-session sandbox, and the interesting failure is
        the loop offering the tool and never running it -- which looks
        identical from the response text, because the model will happily
        narrate a result it never received.

        So the evidence is not the prose. A token is minted here, the model is
        asked to write it to a file, and this reads it back out of
        `<studio home>/sandbox`, where `sandbox_root()` puts the per-session
        working directories. Nothing in the reply can fake that; only an
        executed `open(...).write(...)` puts those bytes on this disk.

        Two settings are not incidental and must not be "simplified":

        * `permission_mode = "off"`. `routes/inference.py` REJECTS a local
          python/terminal tool under `ask`, and under `auto` or an omitted
          default, with a 400 -- there is no confirmation channel here. The
          run would fail on configuration and look like a broken tool.
        * `tool_choice` is left alone. Forcing it would prove the schema is
          reachable, not that the loop runs what it selected, and the file is
          the claim either way.

        The filename is not asserted, only the CONTENT: a small local model
        rewording a path is not a Studio defect, and any file carrying the
        token was written by code that ran.
        """
        failures: list[str] = []
        detail: dict = {}

        token = "unslothcodeexec" + secrets_module.token_hex(8)
        sandbox = self.studio_home / "sandbox"
        detail["sandbox_root"] = str(sandbox)
        before = set(sandbox.rglob("*")) if sandbox.exists() else set()

        code, payload = self.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Use the python tool to run exactly this code, then reply "
                        "with the single word done:\n\n"
                        f"open({token + '.txt'!r}, 'w').write({token!r})"
                    ),
                }
            ],
            enable_tools = True,
            enabled_tools = ["python"],
            permission_mode = "off",
            max_tokens = 512,
        )
        detail["http_status"] = code
        if code != 200 or not isinstance(payload, dict):
            failures.append(
                f"the code-execution completion returned HTTP {code}: {str(payload)[:300]}"
            )
        else:
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            detail["finish_reason"] = choice.get("finish_reason")
            detail["reply"] = (message.get("content") or "")[:300]

        # The claim, read off the filesystem rather than off the reply.
        written = []
        if sandbox.exists():
            for path in sandbox.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    body = path.read_text(encoding = "utf-8", errors = "replace")
                except OSError:
                    continue
                if token in body or token in path.name:
                    written.append(str(path.relative_to(sandbox)))
        detail["files_carrying_the_token"] = written
        detail["new_sandbox_entries"] = sorted(
            str(p.relative_to(sandbox))
            for p in ((set(sandbox.rglob("*")) - before) if sandbox.exists() else set())
        )[:20]

        if not written:
            failures.append(
                "no file under the sandbox carries the token, so the python "
                "tool was offered but never executed -- whatever the reply says"
            )

        detail["failures"] = failures
        return self.record("code_execution", not failures, detail)

    def assert_web_search(self) -> bool:
        """The web_search tool must be EXECUTED, and the log is where that shows.

        Same shape as `assert_code_execution` and a different instrument,
        because a web search leaves nothing on disk to read back. Studio logs
        `execute_tool: name=...` from INSIDE `execute_tool`, so the line is
        emitted by execution rather than by selection -- a loop that offered
        the tool and never ran it produces no such line, which is the failure
        worth catching.

        What is deliberately NOT asserted: that the search returned results.
        `_web_search` fans out through ddgs with no API key, and a provider
        rate-limiting a Kaggle egress IP is a fact about the day, not a Studio
        defect. Failing on it would put a red in front of every PR for
        something no reader could act on. The reply and the result count are
        recorded so a human can see which happened.
        """
        failures: list[str] = []
        detail: dict = {}

        # TWO attempts, and the second is what makes a failure diagnosable.
        # `enabled_tools = ["web_search"]` is one name out of ALL_TOOLS, and
        # `routes/inference.py` also reads a request naming only hosted-tool
        # names as a provider-hosted ask. Omitting `enabled_tools` selects
        # every local tool instead, which is a different path through the same
        # loop. If the first fails and the second executes, the fault is in the
        # single-name selection; if neither does, the model will not call the
        # tool however it is offered. Reporting "the loop offered web_search
        # and never ran it" off one attempt was a guess dressed as a finding:
        # nothing in that run showed the tool had been offered at all.
        marker = "execute_tool: name=web_search"
        prompt = (
            "Search the web for the current version of the Linux kernel, "
            "then answer in one sentence."
        )
        attempts: list[dict] = []
        for label, selection in (
            ("named", {"enabled_tools": ["web_search"]}),
            ("all_local_tools", {}),
        ):
            before = ""
            try:
                before = self.server_log.read_text(encoding = "utf-8", errors = "replace")
            except OSError:
                pass

            code, payload = self.chat(
                [{"role": "user", "content": prompt}],
                enable_tools = True,
                permission_mode = "off",
                # Forced BY NAME. `tool_choice: "required"` was tried first and
                # changed nothing -- kernel unsloth-probe-studio-r3-0b85d4
                # returned the same parametric answer, "The current version of
                # the Linux kernel is 6.10", with executions 0. A bare
                # "required" says only that some tool must be called; the dict
                # form pins the function, and
                # `chat_template_helpers.forced_tool_name` reads exactly this
                # shape. Without a force this measures whether a 2B model
                # DECIDES to search, which is a model property rather than a
                # Studio one.
                tool_choice = {"type": "function", "function": {"name": "web_search"}},
                max_tokens = 512,
                **selection,
            )
            record = {"selection": label, "http_status": code}
            if code != 200 or not isinstance(payload, dict):
                record["error"] = str(payload)[:200]
            else:
                choice = (payload.get("choices") or [{}])[0]
                record["finish_reason"] = choice.get("finish_reason")
                record["reply"] = ((choice.get("message") or {}).get("content") or "")[:200]

            try:
                after = self.server_log.read_text(encoding = "utf-8", errors = "replace")
            except OSError:
                after = ""
            # Only what THIS request wrote, so an earlier attempt's tool call
            # cannot be read as this one's evidence.
            fresh = after[len(before) :]
            record["executions"] = fresh.count(marker)
            # Any tool at all, which separates "the loop ran and chose
            # something else" from "the loop never ran".
            record["any_tool_executions"] = fresh.count("execute_tool: name=")
            attempts.append(record)
            if record["executions"]:
                break

        detail["attempts"] = attempts
        detail["executions"] = sum(a["executions"] for a in attempts)
        if not detail["executions"]:
            failures.append(
                f"{marker!r} never appeared in the server log for either "
                f"selection, so web_search was not executed however it was "
                f"offered: {attempts}"
            )

        detail["failures"] = failures
        return self.record("web_search", not failures, detail)

    # ----------------------------------------------------------- assertion B

    def assert_training(self) -> bool:
        failures: list[str] = []
        detail: dict = {}

        dataset = self.studio_home / "assets" / "datasets" / "uploads" / "studio_gpu_canary.jsonl"
        dataset.parent.mkdir(parents = True, exist_ok = True)
        shutil.copyfile(_HERE / "train_canary.jsonl", dataset)
        detail["dataset"] = str(dataset)

        body = {
            "model_name": self.args.train_model,
            "training_type": "LoRA/QLoRA",
            "format_type": "alpaca",
            "local_datasets": [str(dataset)],
            "project_name": "kaggle-t4-studio-gpu",
            "load_in_4bit": True,
            "max_seq_length": 512,
            "max_steps": self.args.max_steps,
            "num_epochs": 0,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            # A string, not a float: TrainingStartRequest.learning_rate is
            # declared as str and a float here is a 422.
            "learning_rate": "2e-4",
            "lora_r": 8,
            "lora_alpha": 16,
            # Above max_steps on purpose. An intermediate checkpoint would
            # double the disk this run needs and add nothing: the assertion is
            # about the final adapter.
            "save_steps": 10_000,
            "random_seed": 3407,
        }

        try:
            started = self.studio.expect(
                "POST", "/api/train/start", body, timeout = self.args.chat_timeout
            )
        except StudioError as exc:
            detail["failures"] = [str(exc)]
            return self.record("lora_training", False, detail)

        detail["job_id"] = started.get("job_id") if isinstance(started, dict) else None
        detail["start_status"] = started.get("status") if isinstance(started, dict) else None
        if detail["start_status"] != "queued":
            failures.append(
                f"/api/train/start answered status={detail['start_status']!r} rather than "
                f"'queued': {started.get('error') or started.get('message')}"
            )
            detail["failures"] = failures
            return self.record("lora_training", False, detail)

        terminal_reason = {"why": ""}

        def _accept(status) -> bool:
            done, why = training_verdict(status)
            if why:
                terminal_reason["why"] = why
            return done

        ok, last, reason = wait_for(
            probe = lambda: self.studio.get("/api/train/status")[1],
            accept = _accept,
            deadline_s = self.args.train_deadline,
            interval_s = 5.0,
            # An Unsloth that died mid-training answers every status request
            # with an error, which wait_for retries -- for the whole 1200s
            # deadline, out of a 70-minute session, before reporting it as
            # slowness rather than as death.
            alive = self.server_alive,
        )
        status = last if isinstance(last, dict) else {}
        detail["phase"] = status.get("phase")
        detail["steps_with_loss"] = trained_steps(status)
        detail["nonfinite_losses"] = len(nonfinite_losses(status))
        detail["output_dir"] = (status.get("details") or {}).get("output_dir")

        if not ok:
            failures.append(f"training never reached a terminal phase: {reason}")
        elif terminal_reason["why"]:
            failures.append(terminal_reason["why"])
        else:
            # `completed` is the worker's own bookkeeping. These two are the
            # run's output, and they are what a false green would have to
            # forge.
            if detail["nonfinite_losses"]:
                # A T4 has no bf16, so this trains in fp16, and an fp16 run
                # that diverges still reaches `completed` and still saves an
                # adapter. NaN is not a loss.
                failures.append(
                    f"{detail['nonfinite_losses']} of the logged losses are NaN or "
                    f"infinite, so the run diverged rather than trained"
                )
            if detail["steps_with_loss"] < self.args.max_steps:
                failures.append(
                    f"the run reported completed but logged a finite loss for only "
                    f"{detail['steps_with_loss']} of {self.args.max_steps} steps, so it did "
                    f"not train what it claimed to"
                )
            adapter_ok, adapter_failures, adapter_detail = adapter_verdict(detail["output_dir"])
            detail.update(adapter_detail)
            if not adapter_ok:
                failures += adapter_failures

        detail["failures"] = failures
        return self.record("lora_training", not failures, detail)

    # ----------------------------------------------------------- assertion C

    def install_llama_cpp(self) -> bool:
        """Put a CUDA llama.cpp under STUDIO_HOME before anything asks for one.

        Four hardware runs reported ``install_kind=None`` and failed the export
        assertion for it, and the reason was never subtle: nothing had ever
        installed a llama.cpp at all. The export route falls back to whatever
        it can find, so the assertion was measuring an absence.

        Its own assertion rather than a step inside the export one, because
        "the CUDA bundle would not install on this box" and "the CUDA bundle
        installed and the export against it failed" are different findings and
        only the second is about exporting.

        A failure here is NOT fatal to the run. The export assertion still
        executes and still reports the install_kind it found, so a box where
        the bundle cannot be installed produces the same honest red it did
        before rather than skipping the export entirely.
        """
        detail: dict = {}
        installer = self.repo_root / "studio" / "install_llama_prebuilt.py"
        detail["installer"] = str(installer)
        if not installer.is_file():
            return self.record(
                "llama_cpp_install",
                False,
                {**detail, "failures": [f"the installer is not where it should be: {installer}"]},
            )

        install_dir = self.studio_home / "llama.cpp"
        # What was there BEFORE. build_kernel.py runs `install.sh --local`,
        # which its own comment says puts a llama.cpp on disk, so "None after
        # install.sh" and "a CPU bundle after install.sh" are different bugs
        # and this step would hide the difference by fixing both. Recording the
        # prior state keeps the original question answerable.
        detail["install_kind_before"] = install_kind(llama_cpp_marker(self.studio_home))
        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(self.studio_home)
        # `run` already applies capture_output and text; passing them again is
        # a TypeError, and it is one no test here would reach.
        try:
            proc = run(
                [sys.executable, str(installer), "--install-dir", str(install_dir)],
                env = env,
                timeout = LLAMA_CPP_INSTALL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return self.record(
                "llama_cpp_install",
                False,
                {
                    **detail,
                    "failures": [
                        f"the llama.cpp installer did not finish within "
                        f"{LLAMA_CPP_INSTALL_TIMEOUT_S:.0f}s"
                    ],
                },
            )
        detail["returncode"] = proc.returncode
        # The selection log is the whole diagnostic when a CUDA host gets a CPU
        # bundle: install_llama_prebuilt.py explains its choice line by line.
        detail["stdout_tail"] = self.scrub(proc.stdout or "")[-2000:]
        detail["stderr_tail"] = self.scrub(proc.stderr or "")[-2000:]

        kind = install_kind(llama_cpp_marker(self.studio_home))
        detail["llama_cpp_install_kind"] = kind

        failures: list[str] = []
        if proc.returncode != 0:
            failures.append(f"the llama.cpp installer exited {proc.returncode}")
        elif not is_cuda_install(kind):
            # Reaching here means the installer SUCCEEDED and still chose a
            # non-CUDA bundle on a box with a working NVIDIA driver, which is
            # the selection regression this leg exists to catch. Distinct from
            # the installer failing, so it is worded distinctly.
            failures.append(
                f"the installer succeeded but selected install_kind={kind!r} on a CUDA "
                f"box; see stdout_tail for its linux_cuda_selection lines"
            )
        detail["failures"] = failures
        return self.record("llama_cpp_install", not failures, detail)

    def assert_gguf_export(self, adapter_dir: str | None) -> bool:
        failures: list[str] = []
        detail: dict = {"adapter_dir": adapter_dir}

        marker = llama_cpp_marker(self.studio_home)
        kind = install_kind(marker)
        detail["llama_cpp_install_kind"] = kind
        if not is_cuda_install(kind):
            # Stated as a failure rather than a skip. The whole point of
            # running this leg on a T4 is that the CUDA bundle is the one that
            # gets installed here; a CPU bundle on a machine with a working
            # NVIDIA driver means the selection in install_llama_prebuilt.py
            # demoted itself, which is exactly the regression no other job can
            # see.
            failures.append(
                f"the llama.cpp install on this box is install_kind={kind!r}, not a CUDA "
                f"bundle, so the export ran against the same CPU build every other CI job "
                f"already covers"
            )

        if not adapter_dir:
            failures.append("no adapter to export: assertion B did not produce one")
            detail["failures"] = failures
            return self.record("gguf_export", False, detail)

        try:
            self.studio.expect(
                "POST",
                "/api/export/load-checkpoint",
                {"checkpoint_path": adapter_dir, "max_seq_length": 512, "load_in_4bit": True},
                timeout = self.args.export_timeout,
            )
        except StudioError as exc:
            failures.append(f"load-checkpoint failed: {exc}")
            detail["failures"] = failures
            return self.record("gguf_export", False, detail)

        code, before = self.studio.get("/api/export/status")
        baseline_seq = before.get("last_op_seq") if isinstance(before, dict) else None
        baseline_seq = baseline_seq if isinstance(baseline_seq, int) else -1
        detail["baseline_seq"] = baseline_seq

        save_dir = "kaggle_t4_studio_gpu"
        try:
            self.studio.expect(
                "POST",
                "/api/export/export/gguf",
                {
                    "save_directory": save_dir,
                    "quantization_method": self.args.quantization,
                    "push_to_hub": False,
                },
                timeout = self.args.export_timeout,
            )
        except StudioError as exc:
            failures.append(f"export request failed: {exc}")
            detail["failures"] = failures
            return self.record("gguf_export", False, detail)
        except OSError as exc:
            # The route is blocking (routes/export.py awaits the whole export
            # in a thread), and the transport timeout is SHORTER than the
            # export deadline. A merge-convert-quantize that runs past it left
            # the backend still working while this raised an unhandled
            # TimeoutError and crashed the payload. The operation is in
            # flight, not failed: fall through to the status poll, which
            # already knows how to tell "finished" from "never started".
            detail["export_request_timeout"] = f"{type(exc).__name__}: {exc}"
            log(f"the export request timed out in transport ({type(exc).__name__}); still polling")

        terminal_reason = {"why": ""}

        def _accept(status) -> bool:
            done, why = export_verdict(status, baseline_seq)
            if why:
                terminal_reason["why"] = why
            return done

        ok, last, reason = wait_for(
            probe = lambda: self.studio.get("/api/export/status")[1],
            accept = _accept,
            deadline_s = self.args.export_deadline,
            interval_s = 10.0,
            alive = self.server_alive,
        )
        status = last if isinstance(last, dict) else {}
        detail["last_op_status"] = status.get("last_op_status")
        detail["last_op_output_path"] = status.get("last_op_output_path")

        if not ok:
            failures.append(f"the export never finished: {reason}")
        elif terminal_reason["why"]:
            failures.append(terminal_reason["why"])

        gguf = None
        if not failures:
            search_root = Path(status.get("last_op_output_path") or "")
            if not search_root.is_dir():
                search_root = self.studio_home / "exports" / save_dir
            gguf = newest_gguf(search_root)
            detail["search_root"] = str(search_root)
            if gguf is None:
                failures.append(f"the export reported success but no .gguf is under {search_root}")
            else:
                detail["gguf"] = str(gguf)
                detail["gguf_bytes"] = gguf.stat().st_size
                if not gguf_magic_ok(gguf):
                    failures.append(
                        f"{gguf.name} does not start with the GGUF magic, so the file is "
                        f"truncated or is not a GGUF at all"
                    )

        if not failures and gguf is not None:
            # "and loads" is asserted by loading it. A size check would pass
            # for a file llama.cpp cannot open.
            reload_detail = self.load_model(str(gguf), variant = None, label = "exported")
            detail["reload"] = reload_detail
            if reload_detail["failures"]:
                failures += [
                    f"the exported GGUF did not load on the GPU: {f}"
                    for f in reload_detail["failures"]
                ]
            else:
                code, payload = self.chat(
                    [{"role": "user", "content": "What is the Unsloth Studio Kaggle canary?"}]
                )
                text = ""
                if code == 200 and isinstance(payload, dict):
                    choices = payload.get("choices") or [{}]
                    text = (choices[0].get("message") or {}).get("content") or ""
                detail["exported_generated"] = text[:200]
                detail["canary_found"] = CANARY in text
                if code != 200:
                    failures.append(f"the exported GGUF returned HTTP {code} on generation")
                elif not text.strip():
                    failures.append("the exported GGUF loaded but generated nothing")

        detail["failures"] = failures
        return self.record("gguf_export", not failures, detail)

    # Every tab in Studio, and the backing endpoint its first render calls.
    # A tab whose router failed to import does not render a broken panel; the
    # route is simply absent and the request 404s, which is what this looks for.
    # Named by tab so a failure says which one, and kept as a mapping so a new
    # tab that is not covered here is visible as an omission rather than
    # invisible.
    TAB_ENDPOINTS = (
        # NOT /jobs/current: that legitimately 404s when no job is running,
        # so it would be red on correct behaviour. This one is unconditional
        # and lives in the same router package, so an import failure takes it
        # down with everything else.
        ("data_designer", "/api/data-recipe/seed/github/env-token"),
        ("image_creation", "/api/inference/images/status"),
        ("image_creation_gallery", "/api/inference/images/gallery"),
        ("video_creation", "/api/inference/video/status"),
        ("video_creation_gallery", "/api/inference/video/gallery"),
        ("image_training", "/api/train/diffusion/status"),
        ("image_training_runs", "/api/train/diffusion/runs"),
        ("training", "/api/train/status"),
        ("models", "/api/models/local"),
        ("datasets", "/api/datasets/local"),
    )

    def assert_tabs(self) -> bool:
        """Every tab's backing endpoint answers.

        This is a smoke and it is honest about being one: it does not click
        through the UI, it asks each tab's own first request. What it catches
        is the failure that actually happens -- a route module that raised on
        import, so the router was never mounted and the tab renders an error
        the moment it is opened. That is invisible to every other assertion
        here, all of which touch inference, training and export only.

        A 404 is the specific signal, so it is separated from other statuses
        in the record rather than folded into "not 200": a 500 is a live route
        with a broken handler and a 404 is a route that does not exist, and
        they lead a reader to different places.

        An endpoint answering 200 with a non-object is a failure too. Several
        of these are declared with a `response_model`, so a bare string or null
        means something upstream is substituting for the real handler.
        """
        failures: list[str] = []
        detail: dict = {"checked": len(self.TAB_ENDPOINTS)}
        results: dict = {}

        for name, path in self.TAB_ENDPOINTS:
            try:
                code, body = self.studio.get(path)
            except BaseException as exc:  # noqa: BLE001
                results[name] = {"path": path, "error": f"{type(exc).__name__}: {exc}"[:200]}
                failures.append(f"{name}: {path} raised {type(exc).__name__}")
                continue
            entry = {"path": path, "status": code, "type": type(body).__name__}
            results[name] = entry
            if code == 404:
                failures.append(
                    f"{name}: {path} is 404, so its router was never mounted -- "
                    f"the tab renders an error the moment it is opened"
                )
            elif code >= 400:
                failures.append(f"{name}: {path} returned HTTP {code}: {str(body)[:200]}")
            elif not isinstance(body, (dict, list)):
                failures.append(
                    f"{name}: {path} answered 200 with {type(body).__name__}, not a "
                    f"JSON object, so something is standing in for the handler"
                )

        detail["tabs"] = results
        detail["failures"] = failures
        return self.record("tabs", not failures, detail)

    def assert_lora_vs_base(self, gguf: str | None) -> bool:
        """The exported model must answer DIFFERENTLY from the base it came from.

        This is the comparison an export check cannot make on its own. The GGUF
        assertion proves a file was produced, carries the magic, loads on the
        GPU and generates -- and every one of those is true of an export that
        silently merged nothing and shipped the base weights. A no-op merge is
        the regression here, and it is invisible to file size, to the magic and
        to "it generated text".

        Greedy decoding at temperature 0 makes it visible: identical weights
        answer identically, so a difference is the adapter.

        **The determinism control is not optional and comes first.** If the
        SAME weights, loaded twice, do not reproduce their own answer, then a
        difference between two models says nothing, and this reports that it
        could not compare rather than passing on the noise. Two loads of the
        base, then one of the export: the claim is only made once the
        instrument has been shown to be steady.

        The canary is REPORTED rather than asserted. Studio's training run is a
        handful of steps and whether that is enough to learn a specific string
        is a property of the run length, not of the export path -- asserting it
        would be a red about training tuning wearing an export label.
        """
        failures: list[str] = []
        detail: dict = {"gguf": gguf}
        prompt = [{"role": "user", "content": "What is the Unsloth Studio Kaggle canary?"}]

        if not gguf:
            failures.append(
                "no exported GGUF to compare: the export assertion did not "
                "produce one, so there is nothing to hold against the base"
            )
            detail["failures"] = failures
            return self.record("lora_vs_base", False, detail)

        def _say(model: str, variant, label: str) -> tuple[str | None, list]:
            loaded = self.load_model(model, variant = variant, label = label)
            if loaded["failures"]:
                return None, [f"{label} did not load: {loaded['failures'][0]}"]
            code, payload = self.chat(prompt)
            if code != 200 or not isinstance(payload, dict):
                return None, [f"{label} returned HTTP {code} on generation"]
            choices = payload.get("choices") or [{}]
            return ((choices[0].get("message") or {}).get("content") or ""), []

        base_one, problems = _say(self.args.chat_model, self.args.chat_variant, "base")
        failures += problems
        base_two, problems = _say(self.args.chat_model, self.args.chat_variant, "base_again")
        failures += problems
        detail["base_first"] = (base_one or "")[:200]
        detail["base_second"] = (base_two or "")[:200]

        if base_one is not None and base_two is not None:
            detail["base_reproduces_itself"] = base_one == base_two
            if base_one != base_two:
                failures.append(
                    "the base model did not reproduce its own greedy answer "
                    "across two loads, so a difference against the export "
                    "would be noise rather than the adapter"
                )

        if not failures:
            tuned, problems = _say(gguf, None, "exported")
            failures += problems
            detail["exported_said"] = (tuned or "")[:200]
            detail["canary_in_exported"] = CANARY in (tuned or "")
            detail["canary_in_base"] = CANARY in (base_one or "")
            if tuned is not None:
                detail["differs_from_base"] = tuned != base_one
                if tuned == base_one:
                    failures.append(
                        "the exported model gave the base model's answer "
                        "verbatim, so the adapter reached neither the merge "
                        "nor the GGUF -- which every other export check passes"
                    )

        detail["failures"] = failures
        return self.record("lora_vs_base", not failures, detail)

    def assert_image_generation(self) -> bool:
        """The image tab, end to end: load, generate, and download the PNG.

        Last priority and the smallest possible run -- 256x256 (the schema's
        floor) at 2 steps -- because the claim is that the path executes, not
        that the picture is good.

        "Nothing errored" is not the check. A diffusion pipeline that fails
        mid-way still writes a gallery record and still returns 200, and a
        pipeline whose weights never loaded produces a FLAT image, which is a
        valid PNG. So the evidence is the file:

        * the bytes start with the PNG magic, so what the download endpoint
          serves is really a PNG rather than a JSON error with a 200 on it;
        * the IHDR chunk says 256x256, read out of the header rather than
          taken from the gallery record -- the record repeats what was asked
          for, and the file says what was made;
        * the image is not one flat colour. A pipeline that produced nothing
          returns a uniform frame, which compresses to almost nothing, so this
          is checked on the decoded extrema where PIL is available and on a
          compressed-size floor where it is not. Both are recorded, so a
          reader can see which one ruled.
        """
        import urllib.error
        import urllib.request

        failures: list[str] = []
        detail: dict = {"model": self.args.image_model}
        want = 256

        try:
            code, body = self.studio.post(
                "/api/inference/images/load",
                {"model_path": self.args.image_model},
                timeout = self.args.export_timeout,
            )
            detail["load_status"] = code
            if code >= 400:
                failures.append(f"images/load returned HTTP {code}: {str(body)[:300]}")
                detail["failures"] = failures
                return self.record("image_generation", False, detail)

            # The load is ASYNCHRONOUS. It answers 200 having only accepted the
            # request, and generating against it answers 409 "No diffusion
            # model is loaded." -- measured on kernel
            # unsloth-probe-studio-r3-0b85d4, where load_status was 200 and
            # generate_status was 409 twelve lines later. So wait for
            # `images/status` to say `loaded`, and carry `load-progress` while
            # waiting: a download that stalls or errors is then reported as
            # what it is instead of arriving as a generation failure.
            deadline = time.time() + self.args.export_deadline
            status: dict = {}
            progress: dict = {}
            while time.time() < deadline:
                _, status_body = self.studio.get("/api/inference/images/status")
                status = status_body if isinstance(status_body, dict) else {}
                if status.get("loaded"):
                    break
                _, progress_body = self.studio.get("/api/inference/images/load-progress")
                progress = progress_body if isinstance(progress_body, dict) else {}
                if progress.get("phase") == "error":
                    break
                time.sleep(5.0)
            detail["load_status_body"] = {
                k: status.get(k) for k in ("loaded", "repo_id", "family", "device", "model_kind")
            }
            detail["load_progress"] = {k: progress.get(k) for k in ("phase", "fraction", "error")}
            if not status.get("loaded"):
                failures.append(
                    f"the diffusion model never reported loaded within "
                    f"{self.args.export_deadline}s: status={detail['load_status_body']} "
                    f"progress={detail['load_progress']}"
                )
                detail["failures"] = failures
                return self.record("image_generation", False, detail)

            code, body = self.studio.post(
                "/api/inference/images/generate",
                {
                    "prompt": "a red square on a blue background",
                    "width": want,
                    "height": want,
                    "steps": 2,
                    "guidance": 0.0,
                    "seed": 3407,
                },
                timeout = self.args.export_timeout,
            )
            detail["generate_status"] = code
            images = (body or {}).get("images") if isinstance(body, dict) else None
            detail["generated_count"] = len(images or [])
            if code >= 400 or not images:
                failures.append(
                    f"images/generate returned HTTP {code} with {len(images or [])} "
                    f"images: {str(body)[:300]}"
                )
                detail["failures"] = failures
                return self.record("image_generation", False, detail)

            record = images[0]
            detail["record"] = {k: record.get(k) for k in ("id", "width", "height", "steps")}
            image_id = record.get("id")

            # The PNG itself, fetched raw. The JSON client decodes to utf-8,
            # which would corrupt the bytes the whole check is about.
            request = urllib.request.Request(
                f"http://127.0.0.1:{self.args.port}/api/inference/images/"
                f"gallery/{image_id}/file"
            )
            if self.studio.token:
                request.add_header("Authorization", f"Bearer {self.studio.token}")
            with urllib.request.urlopen(request, timeout = 120) as response:
                png = response.read()
                detail["content_type"] = response.headers.get("Content-Type")
            detail["png_bytes"] = len(png)

            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                failures.append(
                    f"the download did not start with the PNG magic, so what it "
                    f"served is not a PNG: {png[:16]!r}"
                )
            else:
                # IHDR is fixed-offset: 8 magic, 4 length, 4 type, then width
                # and height as big-endian uint32.
                width = int.from_bytes(png[16:20], "big")
                height = int.from_bytes(png[20:24], "big")
                detail["png_size"] = [width, height]
                if (width, height) != (want, want):
                    failures.append(
                        f"the PNG header says {width}x{height}, not {want}x{want} -- "
                        f"the file is not what was asked for, whatever the record says"
                    )

                flat = None
                try:
                    from PIL import Image  # noqa: PLC0415

                    with Image.open(io.BytesIO(png)) as opened:
                        bands = opened.convert("RGB").getextrema()
                    detail["extrema"] = bands
                    flat = all(low == high for low, high in bands)
                    detail["flatness_source"] = "pixels"
                except Exception as exc:  # noqa: BLE001
                    detail["pil_error"] = f"{type(exc).__name__}: {exc}"[:200]
                    # A uniform 256x256 frame compresses to well under 2 KB.
                    flat = len(png) < 2000
                    detail["flatness_source"] = "compressed size"
                if flat:
                    failures.append(
                        f"the image is one flat colour ({detail['flatness_source']}), "
                        f"which is what a pipeline whose weights never loaded "
                        f"produces -- and it is a perfectly valid PNG"
                    )
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"the image path raised: {type(exc).__name__}: {exc}"[:300])
        finally:
            # Always, or a diffusion pipeline holds the card for whatever runs
            # next and that failure lands on the wrong assertion.
            try:
                self.studio.post("/api/inference/images/unload", {}, timeout = 120)
            except BaseException:  # noqa: BLE001
                pass

        detail["failures"] = failures
        return self.record("image_generation", not failures, detail)

    def assert_cloudflare(self) -> bool:
        """`unsloth run --cloudflare`: a public URL that serves, and refuses.

        This is the only assertion in the payload that reaches the public
        internet, so what it claims is deliberately narrow and what it refuses
        to claim is stated:

        1. cloudflared is fetched and a quick tunnel is established, and the
           URL printed is a real `*.trycloudflare.com` host rather than the
           `api.trycloudflare.com` that appears in cloudflared's own FAILURE
           lines -- that is a live trap, and Studio's own regex carries the
           same negative lookahead for it;
        2. the tunnel SERVES: `/api/health` answers through the public URL,
           which is what separates "a URL was printed" from "a URL that works";
        3. the tunnel REFUSES an unauthenticated request. A public URL onto a
           CI machine is only defensible if it is behind auth, and this is the
           check that says so rather than assuming it.

        `--host 0.0.0.0` is not incidental. Studio raises a quick tunnel for
        WILDCARD binds; on 127.0.0.1 there is nothing to publish and no URL is
        printed, which would read as a broken feature.

        A tunnel that cannot be established AT ALL is reported rather than
        failed, and the reason is carried from the log. Kaggle egress to
        cloudflared's release host is not something this repo controls, and the
        directive asks for this "if possible". The narrowness is enforced: the
        excuse applies only when NO url was printed, and never once one was.
        """
        failures: list[str] = []
        detail: dict = {}
        port = self.args.port + 2
        log_path = self.outdir / "unsloth_cloudflare.log"
        detail["port"] = port

        head = self.studio_command()[:1] or [sys.executable]
        if head[0] == sys.executable:
            head = [sys.executable, "-c", "from unsloth_cli import app; app()"]
        cmd = head + [
            "run",
            "--model",
            self.args.chat_model,
            "--port",
            str(port),
            # WILDCARD. A loopback bind publishes nothing and prints no URL.
            "--host",
            "0.0.0.0",
            "--api-only",
            "--cloudflare",
            "--start-api-key-marker",
            "--max-seq-length",
            str(self.args.studio_ctx),
        ]
        if self.args.chat_variant:
            cmd += ["--gguf-variant", self.args.chat_variant]
        detail["command"] = " ".join(cmd)

        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(self.studio_home)
        env.setdefault("HF_HOME", str(self.studio_home / "cache" / "huggingface"))
        env["PYTHONUNBUFFERED"] = "1"
        env["UNSLOTH_DISABLE_STATISTICS"] = "1"

        handle = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            cwd = str(self.repo_root),
            env = env,
            stdout = handle,
            stderr = subprocess.STDOUT,
        )

        api_key = None
        url = None
        try:
            deadline = time.time() + self.args.health_deadline
            while time.time() < deadline:
                if proc.poll() is not None:
                    failures.append(
                        f"`unsloth run --cloudflare` exited with code {proc.returncode}"
                    )
                    break
                text = log_path.read_text(encoding = "utf-8", errors = "replace")
                if api_key is None and "UNSLOTH_START_API_KEY:" in text:
                    api_key = text.split("UNSLOTH_START_API_KEY:", 1)[1].split("\n", 1)[0].strip()
                    if api_key:
                        self.secrets.add(api_key)
                if url is None:
                    # The SAME negative lookahead Studio's own matcher uses.
                    # `api.trycloudflare.com` appears in cloudflared's failure
                    # lines and is never a usable tunnel.
                    found = re.search(r"https://(?!api\.)[A-Za-z0-9-]+\.trycloudflare\.com", text)
                    if found:
                        url = found.group(0)
                if url and api_key:
                    break
                time.sleep(2.0)

            detail["tunnel_url_seen"] = bool(url)
            if url is None:
                # Reported, not failed, and ONLY here: nothing was published,
                # so there is nothing to have gone wrong with. The reason comes
                # off the log rather than being assumed.
                tail = log_path.read_text(encoding = "utf-8", errors = "replace")[-600:]
                detail["no_tunnel_reason"] = self.scrub(tail)
                detail["reported_not_failed"] = True
            else:
                # The host is a secret in the sense that matters here: it is a
                # live public route to this machine, and the artifact is read
                # by people who are not running it.
                self.secrets.add(url)
                public = Studio(url, timeout = 30.0)
                code, body = public.get("/api/health", auth = False)
                detail["public_health_status"] = code
                # A tunnel that resolves but serves Cloudflare's own error page
                # answers 530, which is a URL that does not work.
                if code != 200:
                    failures.append(
                        f"the quick tunnel URL answered HTTP {code} on /api/health, "
                        f"so a URL was published that does not serve"
                    )

                # And it must REFUSE. A public URL onto a CI box is only
                # defensible behind auth, so this is asserted, not assumed.
                code, _ = public.post(
                    "/v1/chat/completions",
                    {
                        "model": "default",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8,
                    },
                    auth = False,
                )
                detail["public_unauthenticated_status"] = code
                if code < 400:
                    failures.append(
                        f"an UNAUTHENTICATED request through the public tunnel "
                        f"was accepted ({code}), so the quick tunnel exposes "
                        f"this machine's inference to anyone with the URL"
                    )
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"driving the tunnel raised: {type(exc).__name__}: {exc}"[:300])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout = 60)
            except subprocess.TimeoutExpired:
                proc.kill()
            handle.close()

        detail["failures"] = failures
        return self.record("cloudflare", not failures, detail)

    # ------------------------------------------------------ existing drivers

    def assert_cli_run(self) -> bool:
        """`unsloth run`: a model server started from the CLI, driven by its key.

        This is the headless path a user scripts, and nothing in CI covers it.
        It is a different launch from `unsloth studio`: `run` starts the
        backend, waits for health, mints an API key IN-PROCESS, and then loads
        the model over HTTP. Any of those four can break without the others
        noticing, and the command still prints a banner.

        Four claims, and each rules out a way the previous one passes hollow:

        1. the server becomes healthy on the port it was given;
        2. the model reaches the GPU -- measured as device VRAM growth across
           the launch, not as a line in the banner. `--api-only` on a card the
           chat-UI phase has already emptied makes that delta this launch's;
        3. the key the command minted AUTHENTICATES a real completion, and the
           completion is non-empty. A key that is printed and rejected is
           worse than no key, because the failure surfaces in a user's
           integration rather than here;
        4. a CORRUPTED key is refused. Without it, a server ignoring the header
           entirely passes claim 3.

        `--start-api-key-marker` is how the key is obtained: it prints
        `UNSLOTH_START_API_KEY: <key>`, which is the mechanism `unsloth start`
        itself uses. The value is registered as a secret the moment it is read,
        before anything scrubs a log on the way into the evidence bundle.
        """
        failures: list[str] = []
        detail: dict = {}
        port = self.args.port + 1
        log_path = self.outdir / "unsloth_run.log"
        detail["port"] = port

        baseline = nvidia_used_mib()
        detail["vram_before_mib"] = baseline
        # The per-process reading is taken alongside the device one because the
        # device one is only valid when this payload owns the card, and under
        # --studio-concurrent it does not. See the verdict below.
        apps_before = nvidia_compute_apps() or {}
        detail["compute_apps_before"] = apps_before

        head = self.studio_command()[:1] or [sys.executable]
        if head[0] == sys.executable:
            head = [sys.executable, "-c", "from unsloth_cli import app; app()"]
        cmd = head + [
            "run",
            "--model",
            self.args.chat_model,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            # Headless: no UI to serve and no browser to open, which is the
            # shape this path is for.
            "--api-only",
            # Never a public URL from CI. --secure would imply one.
            "--no-cloudflare",
            "--start-api-key-marker",
            "--max-seq-length",
            str(self.args.studio_ctx),
        ]
        if self.args.chat_variant:
            cmd += ["--gguf-variant", self.args.chat_variant]
        detail["command"] = " ".join(cmd)

        env = dict(os.environ)
        env["UNSLOTH_STUDIO_HOME"] = str(self.studio_home)
        env.setdefault("HF_HOME", str(self.studio_home / "cache" / "huggingface"))
        env["PYTHONUNBUFFERED"] = "1"
        env["UNSLOTH_DISABLE_STATISTICS"] = "1"

        handle = open(log_path, "ab")
        proc = subprocess.Popen(
            cmd,
            cwd = str(self.repo_root),
            env = env,
            stdout = handle,
            stderr = subprocess.STDOUT,
        )

        api_key = None
        client = Studio(f"http://127.0.0.1:{port}")
        try:
            deadline = time.time() + self.args.health_deadline
            while time.time() < deadline:
                if proc.poll() is not None:
                    failures.append(f"`unsloth run` exited early with code {proc.returncode}")
                    break
                text = log_path.read_text(encoding = "utf-8", errors = "replace")
                if api_key is None and "UNSLOTH_START_API_KEY:" in text:
                    api_key = text.split("UNSLOTH_START_API_KEY:", 1)[1].split("\n", 1)[0].strip()
                    if api_key:
                        # Before anything else reads this file.
                        self.secrets.add(api_key)
                if api_key and health_is_ready(client.get("/api/health", auth = False)[1]):
                    break
                time.sleep(2.0)

            detail["saw_api_key"] = bool(api_key)
            if not api_key:
                failures.append(
                    "`unsloth run` never printed UNSLOTH_START_API_KEY, so it "
                    "did not reach the point where it mints a key"
                )

            if api_key:
                client.token = api_key
                code, body = client.post(
                    "/v1/chat/completions",
                    {
                        "model": "default",
                        "messages": [{"role": "user", "content": "Say hello in one word."}],
                        "max_tokens": 32,
                        "temperature": 0.0,
                    },
                    timeout = self.args.chat_timeout,
                )
                detail["completion_status"] = code
                text = ""
                if code == 200 and isinstance(body, dict):
                    text = ((body.get("choices") or [{}])[0].get("message") or {}).get(
                        "content"
                    ) or ""
                detail["generated"] = text[:200]
                if code != 200:
                    failures.append(
                        f"the key `unsloth run` minted did not authenticate a "
                        f"completion: HTTP {code} {str(body)[:200]}"
                    )
                elif not text.strip():
                    failures.append("the CLI-served model returned empty content")

                # And a corrupted key must be refused, or the check above
                # passes on a server that ignores the header.
                client.token = api_key[:-4] + "0000" if len(api_key) > 8 else "bogus"
                code, _ = client.post(
                    "/v1/chat/completions",
                    {
                        "model": "default",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 8,
                    },
                    timeout = self.args.chat_timeout,
                )
                detail["bad_key_status"] = code
                if code < 400:
                    failures.append(
                        f"a corrupted API key was accepted ({code}), so the "
                        f"check above passes whatever is sent"
                    )

            # AFTER a completion has come back, not after the API-key banner.
            # `unsloth run` prints the key while it is still starting, and on
            # kernel unsloth-probe-studio-full2-815a0c the sample landed before
            # llama-server had the weights anywhere: 0.0 MiB of growth on a
            # launch whose own log says
            # `Starting llama-server: ... -ngl -1 --fit off`, which is Studio
            # asking for every layer on the card. A served completion is the
            # only cheap proof the weights are resident, so the ruler goes
            # after it.
            settled = nvidia_used_mib()
            detail["vram_after_mib"] = settled
            apps_after = nvidia_compute_apps()
            detail["compute_apps"] = apps_after
            failure, measured = cli_run_gpu_failure(
                apps_before,
                apps_after,
                baseline,
                settled,
            )
            detail.update(measured)
            if failure:
                failures.append(failure)
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"driving `unsloth run` raised: {type(exc).__name__}: {exc}"[:300])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout = 60)
            except subprocess.TimeoutExpired:
                proc.kill()
            handle.close()

        detail["failures"] = failures
        return self.record("cli_run", not failures, detail)

    def assert_chat_ui(self) -> bool:
        """Drive the repo's own chat UI driver. Runs last: it stops the server."""
        driver = self.repo_root / "tests" / "studio" / "playwright_chat_ui.py"
        failures: list[str] = []
        detail: dict = {"driver": str(driver)}

        # RE-SEED the account before handing over. The driver's first UI step is
        # "change-password through UI (Setup your account)", which waits for
        # #new-password on the forced-change form -- and authenticate() has to
        # retire the bootstrap password over the API to get past that same gate,
        # so by now Unsloth shows an ordinary login and the field never appears.
        # Kernel unsloth-t4-ci-412345d2 failed the API assertions on the gate;
        # 9ddd8ae4 fixed those and failed the driver on a stale password;
        # 9c1a3b (this run) fixed the password and failed the driver on the form
        # being gone. The two needs are opposites -- the API wants the change
        # DONE, the driver wants it PENDING -- so they cannot share one account.
        #
        # A restart is the cheap way to give the driver what it expects:
        # start_server() removes $STUDIO_HOME/auth, which is what re-seeds the
        # bootstrap password, and this assertion runs last and stops the server
        # anyway, so nothing after it needs the API session.
        self.stop_server()
        if not self.start_server():
            failures.append("Unsloth did not come back after the restart that re-seeds the account")
            detail["failures"] = failures
            return self.record("chat_ui_driver", False, detail)
        detail["reseeded"] = True

        current = self.remember_bootstrap() or ""
        if not current:
            failures.append("no bootstrap password was re-seeded, so the driver cannot log in")
            detail["failures"] = failures
            return self.record("chat_ui_driver", False, detail)
        self.secrets.add(current)

        self.art_dir.mkdir(parents = True, exist_ok = True)
        rotated = "KaggleT4-Studio-" + os.urandom(8).hex()
        self.secrets.add(rotated)
        env = dict(os.environ)
        env.update(
            {
                "BASE_URL": self.base_url,
                # The freshly re-seeded bootstrap password: the driver rotates
                # it through the UI and then asserts the old one stops working.
                # Passed through the environment of one child, written nowhere.
                "STUDIO_OLD_PW": current,
                "STUDIO_NEW_PW": rotated,
                "PW_ART_DIR": str(self.art_dir),
                "GGUF_REPO": self.args.chat_model,
                "GGUF_VARIANT": self.args.chat_variant or "",
                # Not strict: STRICT turns a set of cosmetic soft checks
                # (typeahead ordering, theme cycling, nav labels) into hard
                # failures, and none of them is what this leg is spending GPU
                # quota to learn.
                "STUDIO_UI_STRICT": "0",
                # The defaults are sized for a warm ubuntu-latest runner with
                # a cached 270M model. Here the model is larger and the box is
                # busier.
                "STUDIO_UI_WALL_TIMEOUT_S": str(self.args.ui_wall_timeout),
                "STUDIO_UI_LOAD_TIMEOUT_MS": "600000",
                "STUDIO_UI_TURN_TIMEOUT_MS": "180000",
                "PYTHONUNBUFFERED": "1",
            }
        )

        log("running tests/studio/playwright_chat_ui.py")
        try:
            proc = subprocess.run(
                [sys.executable, str(driver)],
                cwd = str(self.repo_root),
                env = env,
                capture_output = True,
                text = True,
                timeout = self.args.ui_wall_timeout + 300,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -9
            out = (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
            err = "the driver outlived its own wall-clock watchdog"

        (self.outdir / "playwright_chat_ui.log").write_text(
            (out or "") + "\n----- stderr -----\n" + (err or ""), encoding = "utf-8"
        )
        detail["returncode"] = rc
        detail["stdout_tail"] = self.scrub(" | ".join((out or "").splitlines()[-25:]))
        detail["stderr_tail"] = self.scrub(" | ".join((err or "").splitlines()[-25:]))
        detail["screenshots"] = sorted(p.name for p in self.art_dir.glob("*.png"))

        if rc != 0:
            failures.append(
                f"playwright_chat_ui.py exited {rc} "
                f"({'wall-clock watchdog' if rc == 2 else 'assertion failure'})"
            )
        detail["failures"] = failures
        return self.record("chat_ui_driver", not failures, detail)

    # -------------------------------------------------------------- evidence

    def redacted(self, path: Path) -> bytes:
        """A log file with every credential this run knows about removed.

        Unsloth's startup banner prints the bootstrap password, and the chat
        driver's own log echoes what it was given. Both files leave this
        machine as a CI artifact, so both are rewritten on the way out. The
        replacement is a fixed marker rather than a same-length blank, so the
        redaction is visible to whoever reads the artifact.
        """
        self.remember_bootstrap()
        return self.scrub(path.read_text(encoding = "utf-8", errors = "replace")).encode("utf-8")

    def emit_evidence(self, passed: bool) -> None:
        """Ship the artifacts back inside the notebook's own cell output.

        The shared launcher collects executed notebooks and the kernel log and
        nothing else, so a PNG on the Kaggle filesystem is a PNG nobody will
        ever see. Encoding the bundle into stdout is what gets it home. Logs
        and the report always travel; screenshots only on failure, because on
        a pass they are megabytes nobody reads.
        """

        def _pack(*, with_screenshots: bool, log_tail_bytes: int | None = None) -> bytes:
            buf = io.BytesIO()
            with tarfile.open(fileobj = buf, mode = "w:gz") as tar:
                for name in (
                    "studio_gpu_report.json",
                    "unsloth_cloudflare.log",
                    "playwright_chat_ui.log",
                    "studio.log",
                    # `unsloth run` prints the API key it mints; redacted() is
                    # what keeps it out of the artifact, and a file nobody
                    # packs is a file nobody redacts either.
                    "unsloth_run.log",
                ):
                    path = self.outdir / name
                    if path.is_file():
                        scrubbed = self.redacted(path)
                        if (
                            log_tail_bytes is not None
                            and name != "studio_gpu_report.json"
                            and len(scrubbed) > log_tail_bytes
                        ):
                            scrubbed = (
                                b"[earlier lines dropped to fit the evidence cap]\n"
                                + (scrubbed[-log_tail_bytes:])
                            )
                        info = tarfile.TarInfo(name)
                        info.size = len(scrubbed)
                        tar.addfile(info, io.BytesIO(scrubbed))
                if with_screenshots:
                    for shot in sorted(self.art_dir.glob("*.png")):
                        if buf.tell() > MAX_EVIDENCE_BYTES:
                            break
                        tar.add(shot, arcname = f"playwright/{shot.name}")
            return buf.getvalue()

        blob = _pack(with_screenshots = not passed)
        if len(blob) > MAX_EVIDENCE_BYTES:
            # Screenshots are what blow the cap, so screenshots are what goes.
            # The earlier version rebuilt with the report ALONE while saying it
            # was shipping logs, which discarded studio.log and
            # playwright_chat_ui.log in exactly the failing runs that need
            # them.
            log(f"evidence bundle is {len(blob)} bytes, over the cap; shipping logs only")
            blob = _pack(with_screenshots = False)
        if len(blob) > MAX_EVIDENCE_BYTES:
            log(f"the logs alone are {len(blob)} bytes; shipping their tails")
            blob = _pack(with_screenshots = False, log_tail_bytes = MAX_LOG_TAIL_BYTES)

        encoded = base64.b64encode(blob).decode("ascii")
        chunks = [encoded[i : i + EVIDENCE_CHUNK] for i in range(0, len(encoded), EVIDENCE_CHUNK)]
        for index, chunk in enumerate(chunks):
            print(f"{EVIDENCE_PREFIX}{index + 1}/{len(chunks)} {chunk}", flush = True)

    # ------------------------------------------------------------------ main

    def execute(self) -> int:
        if not self.preflight():
            return self.finish()
        # Before the server, so the export route sees a llama.cpp that was
        # already there rather than one that appeared underneath it. Its
        # result is recorded and deliberately not checked: a box where the
        # bundle will not install should still run every other assertion and
        # report the export failing for the reason it actually failed.
        self.install_llama_cpp()
        if not self.start_server():
            return self.finish()
        if not self.authenticate():
            return self.finish()

        # Before the GPU work: it needs nothing but a logged-in session, and
        # putting it after a 20-minute training run would mean a training
        # failure hides whether API keys work at all.
        self.assert_api_key()

        # Before any model work: these are pure route checks that need only a
        # logged-in session, and putting them after a 20-minute training run
        # would mean a training failure hides whether the tabs exist at all.
        self.assert_tabs()

        gpu_ok = self.assert_gpu_inference()
        if gpu_ok:
            self.assert_tool_calling()
            self.assert_code_execution()
            self.assert_web_search()
        else:
            # Tool calling on a CPU fallback would be a green tick for a
            # question nobody asked. Skip it and say so.
            self.record(
                "tool_calling",
                False,
                {"failures": ["skipped: the model was not on the GPU, so this proves nothing"]},
            )
            self.record(
                "code_execution",
                False,
                {"failures": ["skipped: the model was not on the GPU, so this proves nothing"]},
            )
            self.record(
                "web_search",
                False,
                {"failures": ["skipped: the model was not on the GPU, so this proves nothing"]},
            )

        # AFTER the GPU inference assertions and BEFORE training: it reloads
        # the chat model with different flags, so running it earlier would
        # change the model the inference checks measured, and running it after
        # training would put a reload between the adapter and the export.
        if gpu_ok:
            self.assert_server_flags()
            # AFTER the reload, because that is what pins the window to
            # --studio-ctx: a compaction check against an unknown context
            # length cannot say whether the prompt was over it.
            self.assert_compaction()
        else:
            self.record(
                "compaction",
                False,
                {
                    "failures": [
                        "skipped: the model was not on the GPU, so the window "
                        "the check overflows was never pinned"
                    ]
                },
            )
            self.record(
                "server_flags",
                False,
                {
                    "failures": [
                        "skipped: the model was not on the GPU, so a "
                        "tensor_split check proves nothing"
                    ]
                },
            )

        trained = self.assert_training()
        adapter_dir = None
        for entry in self.assertions:
            if entry["name"] == "lora_training":
                adapter_dir = entry.get("output_dir")
        self.assert_gguf_export(adapter_dir if trained else None)

        # Straight after, and it reads the export's own recorded path: the
        # comparison is only meaningful against the file that assertion made.
        exported_gguf = None
        for entry in self.assertions:
            if entry["name"] == "gguf_export":
                exported_gguf = entry.get("gguf")
        self.assert_lora_vs_base(exported_gguf)

        # BEFORE the UI driver, because assert_chat_ui ends by stopping the
        # server and every request below it would then be refused at the
        # socket. Measured on kernel unsloth-probe-studio-full2-815a0c, where
        # this assertion reported `URLError: Connection refused` and read as a
        # broken image path on a server that had simply been shut down.
        # Placed after all the language work regardless: a diffusion pipeline
        # is the largest single thing this payload puts on a T4, and it is
        # unloaded in a finally either way.
        if self.args.image_generation:
            self.assert_image_generation()

        if not self.args.skip_ui:
            self.assert_chat_ui()

        # LAST, and the order is the design. `unsloth run` starts a SECOND
        # backend against the same studio home, and two backends sharing one
        # home's state is not a configuration anybody ships. assert_chat_ui
        # ends by stopping the server, so by here the port is free, the card is
        # empty, and the VRAM delta below measures this launch alone.
        self.assert_cli_run()

        # LAST of all, and the only thing here that touches the public
        # internet. Its own launch rather than a flag on the one above,
        # because the tunnel needs a WILDCARD bind and assert_cli_run's claim
        # is about a loopback server.
        if self.args.cloudflare_check:
            self.assert_cloudflare()
        else:
            self.record(
                "cloudflare",
                False,
                {"failures": ["skipped: --no-cloudflare-check was passed"]},
            )
        return self.finish()

    def finish(self) -> int:
        report = self.report()
        (self.outdir / "studio_gpu_report.json").write_text(
            json.dumps(report, indent = 2), encoding = "utf-8"
        )
        self.stop_server()
        # Evidence FIRST, then the report line. The launcher's extract_reports
        # keeps the first report it sees for each label|model, so a crash in
        # emit_evidence after a pass was printed published PASS and left the
        # corrected failing report to be ignored.
        try:
            self.emit_evidence(report["passed"])
        except Exception as exc:  # noqa: BLE001
            self.failures.append(f"evidence packaging failed: {type(exc).__name__}: {exc}")
            report = self.report()
            (self.outdir / "studio_gpu_report.json").write_text(
                json.dumps(report, indent = 2), encoding = "utf-8"
            )
        print(RESULT_PREFIX + json.dumps(report), flush = True)
        return 0 if report["passed"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required = True)
    ap.add_argument("--label", default = "studio-gpu")
    ap.add_argument("--repo-root", required = True, help = "the unsloth checkout under test")
    ap.add_argument("--studio-home", required = True, help = "UNSLOTH_STUDIO_HOME for this run")
    ap.add_argument("--port", type = int, default = 18902)
    ap.add_argument(
        # 2048 constrains the runs, as the brief asks. It also keeps the
        # KV-cache check about the cache: an unconstrained context on a 14.56GB
        # card turns it into an OOM about something else.
        "--studio-ctx",
        dest = "studio_ctx",
        type = int,
        default = 2048,
        help = "context length to pin the llama.cpp server to",
    )
    ap.add_argument(
        "--image-model",
        dest = "image_model",
        default = "unsloth/sdxl-turbo",
        help = "diffusion repo for the image-generation assertion",
    )
    ap.add_argument(
        # OFF by default: it is the last-priority item and it pulls a diffusion
        # checkpoint the rest of the payload has no use for. A dispatch that
        # wants it says so.
        "--image-generation",
        dest = "image_generation",
        action = "store_true",
        default = False,
        help = "load a diffusion model and generate one 256x256 image at 2 steps",
    )
    ap.add_argument(
        # On by default because the directive asks for it, and off-able because
        # it is the one assertion here that reaches the public internet.
        "--no-cloudflare-check",
        dest = "cloudflare_check",
        action = "store_false",
        default = True,
        help = "skip the quick-tunnel assertion (it opens a public URL)",
    )
    ap.add_argument(
        # Empty means "use the bootstrap password", which is the behaviour this
        # payload had before. A caller passes `auto` to have one generated.
        "--studio-password",
        default = "",
        help = "start Studio with --password and log in with it; 'auto' generates one",
    )
    # MTP-GGUF, not the plain GGUF: multi-token prediction is a distinct
    # serving path in llama.cpp, and a leg pointed at the plain repo cannot
    # tell whether it works.
    ap.add_argument("--chat-model", default = "unsloth/Qwen3.5-2B-MTP-GGUF")
    ap.add_argument("--chat-variant", default = "UD-Q4_K_XL")
    ap.add_argument("--train-model", default = "unsloth/Qwen3.5-2B")
    ap.add_argument("--max-steps", type = int, default = 8)
    ap.add_argument("--quantization", default = "q8_0")
    ap.add_argument(
        "--gpu-layers",
        type = int,
        default = 99,
        help = "manual GPU layer pin. Above any small model's block count on purpose: "
        "llama.cpp clamps it, so this means 'all of them'",
    )
    ap.add_argument("--health-deadline", type = float, default = 420.0)
    ap.add_argument("--train-deadline", type = float, default = 1200.0)
    ap.add_argument("--export-deadline", type = float, default = 1200.0)
    ap.add_argument("--load-timeout", type = float, default = 900.0)
    ap.add_argument("--export-timeout", type = float, default = 900.0)
    ap.add_argument("--chat-timeout", type = float, default = 300.0)
    ap.add_argument("--ui-wall-timeout", type = float, default = 1200.0)
    ap.add_argument(
        "--skip-ui",
        action = "store_true",
        help = "do not drive playwright_chat_ui.py (for debugging the API assertions alone)",
    )
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    payload = Payload(args)
    try:
        return payload.execute()
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        payload.failures.append(f"payload crashed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return payload.finish()


if __name__ == "__main__":
    raise SystemExit(main())
