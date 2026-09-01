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


def nvidia_compute_apps() -> dict[int, int] | None:
    proc = run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        timeout = 60,
    )
    if proc.returncode != 0:
        return None
    return parse_compute_apps(proc.stdout)


def gpu_inventory() -> list[str]:
    proc = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        timeout = 60,
    )
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


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

    # ---------------------------------------------------------------- report

    def record(self, name: str, passed: bool, detail: dict) -> bool:
        entry = {"name": name, "passed": bool(passed), **detail}
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
        return head + ["studio", "-H", "127.0.0.1", "-p", str(self.args.port)]

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
        path = self.studio_home / "auth" / ".bootstrap_password"
        failures: list[str] = []
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
        return self.record("authenticate", not failures, {"failures": failures})

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

    # ------------------------------------------------------ existing drivers

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
                for name in ("studio_gpu_report.json", "playwright_chat_ui.log", "studio.log"):
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

        gpu_ok = self.assert_gpu_inference()
        if gpu_ok:
            self.assert_tool_calling()
        else:
            # Tool calling on a CPU fallback would be a green tick for a
            # question nobody asked. Skip it and say so.
            self.record(
                "tool_calling",
                False,
                {"failures": ["skipped: the model was not on the GPU, so this proves nothing"]},
            )

        trained = self.assert_training()
        adapter_dir = None
        for entry in self.assertions:
            if entry["name"] == "lora_training":
                adapter_dir = entry.get("output_dir")
        self.assert_gguf_export(adapter_dir if trained else None)

        if not self.args.skip_ui:
            self.assert_chat_ui()
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
    ap.add_argument("--chat-model", default = "unsloth/Qwen3.5-2B-GGUF")
    ap.add_argument("--chat-variant", default = "UD-Q4_K_XL")
    ap.add_argument("--train-model", default = "unsloth/Qwen2.5-0.5B-Instruct")
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
