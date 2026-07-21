# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` — launch a coding agent against a running Unsloth server."""

import atexit
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple, NoReturn, Optional
from urllib.parse import urlencode, urlparse

import click
import typer

from unsloth_cli._inference import (
    _USER_AGENT,
    _studio_token,
    ensure_studio_backend_path,
    find_studio_server,
    is_loopback_url,
    urlopen_no_redirect,
    verify_studio_identity,
)

start_app = typer.Typer(
    help = "Start a coding agent against a running Unsloth server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

_CODEX_PROFILE = "unsloth_api"
_CODEX_ENV_KEY = "UNSLOTH_STUDIO_AUTH_TOKEN"
_HERMES_ENV_KEY = "UNSLOTH_API_KEY"
_HERMES_PROVIDER = "unsloth"
# Skip the installer's interactive setup wizard: `unsloth start hermes` runs
# this hint unattended and then writes its own session-scoped Hermes config, so
# the wizard's global API-key/model prompts would block the launch and point the
# user at a different (global) provider than the one Unsloth just configured.
# Both installers expose a skip flag: `-SkipSetup` (PowerShell) and
# `--skip-setup` (POSIX; passed to the piped script via `bash -s --`). Pin both
# the fetched script and the repository checkout it performs to the same full
# commit so a later change to either upstream branch cannot silently replace
# code that Unsloth executes with the user's privileges.
_HERMES_INSTALL_COMMIT = "f1af945f6c576eccb126fa955edc9be258b33020"
_HERMES_INSTALL_BASE = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/"
    f"{_HERMES_INSTALL_COMMIT}/scripts"
)
_HERMES_WINDOWS_INSTALL_HINT = (
    f"& ([scriptblock]::Create((irm {_HERMES_INSTALL_BASE}/install.ps1)))"
    f" -SkipSetup -Commit {_HERMES_INSTALL_COMMIT}"
)
_HERMES_POSIX_INSTALL_HINT = (
    f"curl -fsSL {_HERMES_INSTALL_BASE}/install.sh | bash -s --"
    f" --skip-setup --commit {_HERMES_INSTALL_COMMIT}"
)
# Hermes refuses to initialize when the model window is under 64,000 tokens; its
# error message points at the model.context_length / auxiliary.compression
# overrides in config.yaml. write_hermes_config claims this value for smaller
# windows and scales the compaction threshold back down to the real window.
_HERMES_MIN_CONTEXT = 65536
_PI_PROVIDER = "unsloth"
_SUBAGENT_NAME = "unsloth"
_SUBAGENT_DESCRIPTION = (
    "Local subagent powered by Unsloth for quick debugging, fast implementation, and research. "
    "Use when the user asks to spawn an Unsloth or local agent."
)
_SUBAGENT_INSTRUCTIONS = (
    "You are a local coding subagent powered by Unsloth. Complete the assigned task directly, "
    "use the available tools when useful, verify your work, and return a concise result to the "
    "parent agent."
)
_PI_SUBAGENT_EXTENSION = Path(__file__).parent.parent / "pi_subagent.ts"
# OpenCode selects a model by "<providerID>/<modelID>" and honors a user
# disabled_providers list. Register the session provider under a dedicated id a
# user's disable list would never target, so the model is always selectable
# without the wrapper having to reconstruct (and override) OpenCode's full,
# multi-layer disabled_providers resolution.
_OPENCODE_PROVIDER = "unsloth-studio"
_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}
_CLAUDE_ENV_UNSET = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")

# Shared by every agent command; only the config/env/command differ.
_MODEL_OPTION = typer.Option(
    None, "--model", "-m", help = "Model for the agent; defaults to the one loaded in Unsloth."
)
_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar = "UNSLOTH_API_KEY",
    help = (
        "Unsloth API key. For a local Unsloth it is minted automatically and "
        "remembered per server. For a remote server, pass one with --api-key "
        "(or UNSLOTH_API_KEY); it is remembered for next time."
    ),
)
_LAUNCH_OPTION = typer.Option(
    True,
    "--launch/--no-launch",
    help = "--no-launch prints the env and command instead (remote shells, WSL).",
)
_SERVE_OPTION = typer.Option(
    True,
    "--serve/--no-serve",
    help = (
        "If no Unsloth server is running, auto-start one for --model and keep it available "
        "after the agent exits. --no-serve keeps the old behavior of erroring out."
    ),
)
# Model-load knobs mirrored from `unsloth run`; only used when --model triggers a
# load on the server. Server-startup flags (--host/--port/--cloudflare/...) do not
# apply here because `unsloth start` attaches to an already-running server.
_GGUF_VARIANT_OPTION = typer.Option(
    None, "--gguf-variant", help = "GGUF quant variant to load (e.g. UD-Q4_K_XL)."
)
_CONTEXT_OPTION = typer.Option(
    0,
    "--max-seq-length",
    "--context-length",
    help = "Context length in tokens for the load (0 = model default).",
)
_LOAD_4BIT_OPTION = typer.Option(
    True, "--load-in-4bit/--no-load-in-4bit", help = "Load hub models in 4-bit (ignored for GGUF)."
)
_TENSOR_PARALLEL_OPTION = typer.Option(
    False,
    "--tensor-parallel/--no-tensor-parallel",
    help = "Split a GGUF across GPUs by tensor instead of by layer (multi-GPU only).",
)
# One normalized "run tools without prompting" switch. Each agent spells this
# differently and it's easy to forget which is which, so accept every spelling and
# route to the agent's own mechanism in _yolo_command_flags / the config writers.
_YOLO_OPTION = typer.Option(
    False,
    "--yolo",
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    help = (
        "Auto-approve all tool actions for this session; routed to the agent's own "
        "flag/config. Any of the three spellings works for any agent."
    ),
)
_PERSIST_OPTION = typer.Option(
    False,
    "--persist/--no-persist",
    help = (
        "Keep this agent's Unsloth-managed session dir so you can resume it later. "
        "codex/openclaw/hermes/pi have their whole home relocated into an Unsloth dir "
        "that is a throwaway temp dir (wiped on exit) by default; with --persist it "
        "lives under the Unsloth agents dir and survives, so their own resume can reopen "
        "it. claude and opencode keep sessions in your own stores (~/.claude, "
        "~/.local/share/opencode), so they already resume regardless. To reopen a "
        "session, pass the agent's own resume command through, e.g. "
        "`unsloth start codex --persist resume` or `claude --resume <id>`; those flow to "
        "the agent unchanged."
    ),
)
_AS_SUBAGENT_OPTION = typer.Option(
    False,
    "--as-subagent",
    help = "Keep the coding agent's current model and add Unsloth as a local subagent.",
)

# Per-agent CLI flag for "run tools without prompting". OpenCode (native --auto is
# command-scoped, handled below) and OpenClaw (config-only) are absent from this prefix map.
_YOLO_COMMAND_FLAGS = {
    "claude": ["--dangerously-skip-permissions"],
    "codex": ["--dangerously-bypass-approvals-and-sandbox"],
    "hermes": ["--yolo"],
    # Pi never prompts per tool call; its only approval gate is project trust, so -a
    # (trust project resources) is the closest "don't ask me" equivalent.
    "pi": ["--approve"],
}


def _yolo_command_flags(agent: str, yolo: bool) -> list:
    # .get so a config-based agent (or a typo) yields no flag instead of a KeyError.
    return _YOLO_COMMAND_FLAGS.get(agent, []) if yolo else []


# Subcommands that reject --auto (OpenCode exposes it only on the default TUI and `run`),
# so `opencode serve --auto` is never emitted. Includes console/generate, hidden from
# `opencode --help` but still registered. Unknown first positionals are TUI paths -> --auto.
_OPENCODE_NON_AUTO_SUBCOMMANDS = frozenset(
    "completion acp mcp attach debug providers auth agent upgrade uninstall serve web "
    "models stats export import github pr session plugin plug db console generate".split()
)
_OPENCODE_GLOBAL_BOOLEAN_OPTIONS = frozenset(
    "-h --help -v --version --print-logs --pure --mdns".split()
)
_OPENCODE_GLOBAL_VALUE_OPTIONS = frozenset(
    "--log-level --port --hostname --mdns-domain --cors".split()
)
_OPENCODE_NATIVE_AUTO_MIN_VERSION = (1, 17, 12)


def _opencode_supports_native_auto() -> bool:
    executable = _which_with_install_dirs("opencode")
    if executable is None:
        # No local binary: a --no-launch recipe may run elsewhere, and _run installs the
        # current release on launch -- either way assume native --auto is available.
        return True
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            timeout = 10,
            stderr = subprocess.DEVNULL,
        )
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return bool(match) and tuple(int(part) for part in match.groups()) >= (
        _OPENCODE_NATIVE_AUTO_MIN_VERSION
    )


def _opencode_subcommand(args: list[str]) -> Optional[str]:
    """Return an explicit OpenCode subcommand after supported global options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None
        if arg in _OPENCODE_GLOBAL_BOOLEAN_OPTIONS:
            index += 1
            continue
        if arg in _OPENCODE_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in _OPENCODE_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        # A non-global option (e.g. --session) is a TUI flag; stop before its value is
        # mistaken for a subcommand.
        if arg.startswith("-"):
            return None
        return arg
    return None


def _opencode_native_auto_args(args: list[str], yolo: bool) -> tuple[list[str], bool]:
    """Add OpenCode's native --auto when the selected command supports it."""
    routed = list(args)
    if not yolo:
        return routed, False
    if _opencode_subcommand(routed) in _OPENCODE_NON_AUTO_SUBCOMMANDS:
        return routed, False
    separator = routed.index("--") if "--" in routed else len(routed)
    # --mini's runMini TUI forces auto=false and never forwards --auto, so appending it is
    # useless; fall back to the config permission block so --yolo still auto-approves.
    if any(arg == "--mini" or arg.startswith("--mini=") for arg in routed[:separator]):
        return routed, False
    if "--auto" not in routed[:separator]:
        routed.insert(separator, "--auto")
    return routed, True


def _hermes_install_hint() -> str:
    return _HERMES_WINDOWS_INSTALL_HINT if os.name == "nt" else _HERMES_POSIX_INSTALL_HINT


def _hermes_resume_oneshot_args(args: list[str]) -> list[str]:
    """Route resumed one-shot prompts through Hermes' session-aware chat command."""
    has_resume = any(
        arg in ("--resume", "-r", "--continue", "-c")
        or arg.startswith(("--resume=", "--continue="))
        or (len(arg) > 2 and arg.startswith(("-r", "-c")))
        for arg in args
    )
    if not has_resume:
        return args

    rewritten = list(args)
    for index, arg in enumerate(rewritten):
        if arg in ("-z", "--oneshot"):
            rewritten[index] = "-q"
        elif len(arg) > 2 and arg.startswith("-z"):
            # argparse accepts attached short-option values (`-zPROMPT` and
            # `-z=PROMPT`); preserve the value byte-for-byte when switching to -q.
            rewritten[index] = f"-q{arg[2:]}"
        elif arg.startswith("--oneshot="):
            rewritten[index] = f"--query={arg.partition('=')[2]}"
        else:
            continue
        if any(item == "--usage-file" or item.startswith("--usage-file=") for item in args):
            raise typer.BadParameter(
                "Hermes cannot resume a one-shot session with --usage-file; remove that option."
            )
        prefix = ["chat", "-Q"]
        if "--yolo" not in rewritten:
            prefix.append("--yolo")
        if "--accept-hooks" not in rewritten:
            prefix.append("--accept-hooks")
        rewritten = prefix + rewritten
        return rewritten
    return args


class LoadOptions(NamedTuple):
    """Model-load knobs forwarded to /api/inference/load when --model triggers a load."""

    gguf_variant: Optional[str] = None
    max_seq_length: int = 0
    load_in_4bit: bool = True
    tensor_parallel: bool = False


def _split_repo_variant(model: str) -> tuple:
    """Split ``org/name:QUANT`` into ``(repo, variant)`` -> ``("org/name", "QUANT")``.

    ``unsloth run`` and llama.cpp accept ``--model org/name:QUANT`` as shorthand for
    ``--model org/name --gguf-variant QUANT``. Mirror that here so a ``:variant`` suffix
    resolves against the already-loaded ``org/name`` (which /v1/models lists without the
    suffix) instead of trying to load a repo id containing ``:`` -- which Hugging Face
    rejects, and which would evict a model another session is using. Local paths, Windows
    drive letters, and ids without a ``:`` pass through unchanged.
    """
    s = (model or "").strip()
    if not s or s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():  # Windows drive, e.g. C:\models\x
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant or "/" in variant:
        return s, None
    return repo, variant


def _display_model_spec(model: str, variant: Optional[str]) -> str:
    """Return a user-facing model name that includes the selected GGUF variant."""
    repo, inline_variant = _split_repo_variant(model)
    selected_variant = variant or inline_variant
    return f"{repo}:{selected_variant}" if selected_variant else model


def _subagent_model_id(
    base: str,
    key: str,
    entry: dict,
    requested_model: Optional[str],
    requested_variant: Optional[str],
) -> str:
    """Return an API model id that preserves the selected GGUF variant.

    Coding-agent model definitions outlive the initial load. If Studio later
    unloads the model, a bare repository id may resolve to a different cached
    quant. Include the explicit or currently loaded variant so an automatic
    reload selects the same weights.
    """
    model_id = str(entry["id"])
    _, inline_variant = _split_repo_variant(requested_model or "")
    variant = requested_variant or inline_variant
    if not variant:
        try:
            status = _http_json("GET", f"{base}/api/inference/status", key)
        except Exception:
            status = {}
        if status.get("is_gguf"):
            variant = status.get("gguf_variant")
    return (
        _display_model_spec(model_id, str(variant))
        if variant and _is_hub_model_id(model_id)
        else model_id
    )


def _fail(message: str) -> NoReturn:
    typer.echo(message, err = True)
    raise typer.Exit(code = 1)


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode())
        return body.get("detail") or body["error"]["message"]
    except Exception:
        return str(exc)


def _http_json(
    method: str,
    url: str,
    token: str,
    payload = None,
    timeout = 30,
    error = None,
):
    """On HTTPError: raise if `error` is None, else fail with `error` plus the server's detail."""
    request = urllib.request.Request(
        url,
        data = None if payload is None else json.dumps(payload).encode(),
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method = method,
    )
    try:
        # No redirects: a 3xx would leak this bearer token to an unvetted base.
        with urlopen_no_redirect(request, timeout = timeout) as response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        if error is None:
            raise
        _fail(f"{error}: {_http_error_detail(exc)}")
    except (urllib.error.URLError, TimeoutError) as exc:
        if error is None:
            raise
        _fail(f"{error}: {getattr(exc, 'reason', None) or exc}")


# A server that WE auto-started (never one we merely found). Kept at module scope so
# failure paths and the atexit backstop can tear it down without threading a handle
# through all six agent commands. Only one agent runs per process, so one slot is enough.
_auto_served_server: Optional[subprocess.Popen] = None
# Model download + load can be slow; give the auto-started server room before giving up.
_SERVER_START_TIMEOUT_S = 900
_DOWNLOAD_POLL_INTERVAL_S = 1.0
_START_API_KEY_PREFIX = "UNSLOTH_START_API_KEY: "
_START_API_KEY_MARKER_ENV = "_UNSLOTH_START_API_KEY_MARKER"


def _format_download_bytes(value: int) -> str:
    value = max(0, int(value))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            precision = 0 if unit in ("B", "KiB") else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024
    return "0 B"


def _format_download_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


class _DownloadProgressDisplay:
    """Render download progress without making redirected output noisy."""

    def __init__(self) -> None:
        self._samples: list[tuple[float, int]] = []
        self._shown = False
        self._last_bucket = -1
        self._last_line_length = 0
        self._last_expected = 0
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def update(self, progress: dict) -> None:
        downloaded = max(0, int(progress.get("downloaded_bytes") or 0))
        completed = max(0, int(progress.get("completed_bytes") or 0))
        expected = max(0, int(progress.get("expected_bytes") or 0))
        self._last_expected = max(self._last_expected, expected)
        fraction = float(progress.get("progress") or 0)
        if downloaded <= 0:
            return
        # The hub endpoint can report a fully cached snapshot as 99% when an
        # older synchronous load has no download manifest. No incomplete bytes
        # means no transfer is occurring, so do not label model startup as a download.
        if completed >= downloaded > 0:
            return

        now = time.monotonic()
        if self._samples and downloaded < self._samples[-1][1]:
            self._samples.clear()
        self._samples.append((now, downloaded))
        cutoff = now - 15.0
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.pop(0)

        rate = 0.0
        if len(self._samples) >= 2:
            elapsed = self._samples[-1][0] - self._samples[0][0]
            delta = self._samples[-1][1] - self._samples[0][1]
            if elapsed >= 1.0 and delta > 0:
                rate = delta / elapsed

        if expected > 0:
            # Trust the endpoint's capped value. It deliberately reports at
            # most 99% while bytes still live in an incomplete file, even when
            # that sparse file's logical size already equals the final blob.
            fraction = min(1.0, max(0.0, fraction))
            percent = min(100, max(0, int(fraction * 100)))
            filled = min(24, int(fraction * 24))
            bar = "=" * filled + ">" + "." * max(0, 23 - filled) if filled < 24 else "=" * 24
            line = (
                f"Downloading model [{bar}] {percent:3d}% "
                f"{_format_download_bytes(downloaded)} / {_format_download_bytes(expected)}"
            )
            bucket = percent // 10
            if rate > 0:
                line += f" | {_format_download_bytes(rate)}/s"
                if downloaded < expected:
                    line += f" | ETA {_format_download_eta((expected - downloaded) / rate)}"
        else:
            line = f"Downloading model: {_format_download_bytes(downloaded)}"
            bucket = downloaded // (1024**3)
            if rate > 0:
                line += f" | {_format_download_bytes(rate)}/s"

        if self._interactive:
            padding = " " * max(0, self._last_line_length - len(line))
            typer.echo(f"\r{line}{padding}", nl = False)
            sys.stdout.flush()
            self._last_line_length = len(line)
        elif not self._shown or bucket > self._last_bucket:
            typer.echo(line)
            self._last_bucket = bucket
        self._shown = True

    def close(self) -> None:
        if self._interactive and self._shown:
            typer.echo()
        self._last_line_length = 0

    def complete(self) -> None:
        """Finish a displayed transfer after the model load confirms success."""
        if not self._shown:
            return
        downloaded = self._samples[-1][1] if self._samples else 0
        expected = max(downloaded, getattr(self, "_last_expected", 0))
        self.update(
            {
                "downloaded_bytes": expected,
                "expected_bytes": expected,
                "progress": 1.0,
            }
        )


def _normalized_variant(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


class _ModelDownloadProgress:
    """Best-effort polling of the model download endpoints."""

    def __init__(self, base: str, key: str, model: str, variant: Optional[str]) -> None:
        self._base = base
        self._key = key
        self._model = model
        self._variant = variant or ""
        self._expected_bytes = 0
        self._display = _DownloadProgressDisplay()
        self._configured = False
        self._disabled = not _is_hub_model_id(model)
        self._progress_prefix = "/api/hub"

    def _configure(self) -> None:
        self._configured = True
        if self._disabled:
            return
        # GGUF repos need the selected quant's size. The generic repo endpoint
        # totals every quant in the repository and would report a misleading
        # percentage, so resolve the variant first and otherwise show bytes only.
        if self._variant or "gguf" in self._model.lower():
            try:
                params = urlencode({"repo_id": self._model})
                try:
                    info = _http_json(
                        "GET",
                        f"{self._base}/api/hub/gguf-variants?{params}",
                        self._key,
                        timeout = 10,
                    )
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    self._progress_prefix = "/api/models"
                    info = _http_json(
                        "GET",
                        f"{self._base}/api/models/gguf-variants?{params}",
                        self._key,
                        timeout = 10,
                    )
                self._variant = self._variant or str(info.get("default_variant") or "")
                wanted = _normalized_variant(self._variant)
                for item in info.get("variants") or []:
                    quant = _normalized_variant(item.get("quant"))
                    filename = _normalized_variant(item.get("filename"))
                    if wanted and (wanted == quant or wanted in filename):
                        self._expected_bytes = int(
                            item.get("download_size_bytes") or item.get("size_bytes") or 0
                        )
                        break
            except Exception:
                # Older servers may not expose the variant endpoint. Byte progress
                # is still useful, and load errors remain owned by the load request.
                pass

    def poll(self) -> None:
        if not self._configured:
            self._configure()
        if self._disabled:
            return
        try:
            if self._variant or "gguf" in self._model.lower():
                params = urlencode(
                    {
                        "repo_id": self._model,
                        "variant": self._variant,
                        "expected_bytes": self._expected_bytes,
                    }
                )
                url = f"{self._base}{self._progress_prefix}/gguf-download-progress?{params}"
            else:
                url = (
                    f"{self._base}{self._progress_prefix}/download-progress?"
                    f"{urlencode({'repo_id': self._model})}"
                )
            try:
                reading = _http_json("GET", url, self._key, timeout = 10)
            except urllib.error.HTTPError as exc:
                if exc.code != 404 or self._progress_prefix == "/api/models":
                    raise
                self._progress_prefix = "/api/models"
                self.poll()
                return
            self._display.update(reading)
        except Exception:
            # Progress is an enhancement. Never turn an unsupported endpoint or
            # a transient polling failure into a model-load failure.
            self._disabled = True

    def close(self) -> None:
        self._display.close()

    def complete(self) -> None:
        self._display.complete()


def _load_model_with_progress(
    base: str, key: str, model: str, load: LoadOptions, payload: dict
) -> dict:
    """Run the blocking load request while polling its download progress."""
    result: list[tuple[bool, object]] = []
    done = threading.Event()

    def _load() -> None:
        try:
            value = _http_json(
                "POST",
                f"{base}/api/inference/load",
                key,
                payload,
                timeout = 3600,
                error = "Model load failed",
            )
            result.append((True, value))
        except BaseException as exc:
            result.append((False, exc))
        finally:
            done.set()

    threading.Thread(target = _load, name = "unsloth-model-load", daemon = True).start()
    progress = _ModelDownloadProgress(base, key, model, load.gguf_variant)
    loading_announced = False
    try:
        while not done.wait(_DOWNLOAD_POLL_INTERVAL_S):
            if not loading_announced:
                typer.echo(f"Loading model: {_display_model_spec(model, load.gguf_variant)}")
                loading_announced = True
            progress.poll()
        ok, value = result[0]
        if not ok:
            assert isinstance(value, BaseException)
            raise value
        progress.complete()
        return value if isinstance(value, dict) else {}
    finally:
        progress.close()


def _studio_healthy(base: str, timeout: float = 3.0) -> bool:
    request = urllib.request.Request(f"{base}/api/health", headers = {"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            return json.loads(response.read(65536).decode() or "{}").get("status") == "healthy"
    except Exception:
        return False


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding = "utf-8", errors = "replace").splitlines()[-lines:])
    except OSError:
        return "(no server log)"


def _shutdown_server(server: Optional[subprocess.Popen]) -> None:
    # Idempotent teardown of a server WE started, plus its own children (llama-server,
    # cloudflared). A no-op once the process is already gone.
    if server is None or server.poll() is not None:
        return
    if os.name == "nt":
        # terminate()/kill() reach only the parent `unsloth run`; taskkill /T walks the
        # whole tree so the llama-server child doesn't keep the port and GPU (matches the
        # taskkill /T /F pattern already used in unsloth/dataprep/synthetic.py).
        try:
            subprocess.run(
                ["taskkill", "/PID", str(server.pid), "/T", "/F"],
                capture_output = True,
                timeout = 15,
                check = False,
            )
            server.wait(timeout = 5)
        except Exception:
            with contextlib.suppress(Exception):
                server.kill()
        return
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except OSError:
        server.terminate()
    try:
        server.wait(timeout = 15)
    except Exception:
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except OSError:
            server.kill()


def _shutdown_auto_served() -> None:
    global _auto_served_server
    server, _auto_served_server = _auto_served_server, None
    if server is not None and server.poll() is None:
        typer.echo("Stopping the auto-started Unsloth server…")
        _shutdown_server(server)


def _keep_auto_served() -> bool:
    """Release ownership so a successfully started server survives this CLI."""
    global _auto_served_server
    server, _auto_served_server = _auto_served_server, None
    atexit.unregister(_shutdown_auto_served)
    return server is not None and server.poll() is None


def _start_studio_server(base: str, model: str, load: LoadOptions) -> subprocess.Popen:
    """Spawn `unsloth run` for `model`, wait until it is fully ready, and return it."""
    global _auto_served_server
    unsloth = shutil.which("unsloth") or "unsloth"
    parsed = urlparse(base)
    # --disable-tools = passthrough mode (relay the agent's own tools); --no-cloudflare =
    # loopback only, no tunnel. Mirrors .github/scripts/serve-unsloth-run.sh.
    command = [
        unsloth,
        "run",
        "-H",
        parsed.hostname or "127.0.0.1",
        "-p",
        str(parsed.port or 8888),
        "--disable-tools",
        "--no-cloudflare",
        "--model",
        model,
    ]
    if load.gguf_variant:
        command += ["--gguf-variant", load.gguf_variant]
    if load.max_seq_length:
        command += ["--context-length", str(load.max_seq_length)]
    if not load.load_in_4bit:
        command += ["--no-load-in-4bit"]
    if load.tensor_parallel:
        command += ["--tensor-parallel"]

    log_path = Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log"
    typer.echo("Starting Unsloth server")
    typer.echo(f"Model: {_display_model_spec(model, load.gguf_variant)}")
    typer.echo(f"Server log: {log_path}")
    # 0600: the `unsloth run` banner in this log carries the minted sk-unsloth- key, and
    # the tempdir is world-traversable. Unlink first so a stale looser-mode file (pid
    # reuse) can't survive with its old permissions.
    log_path.unlink(missing_ok = True)
    log = os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
    # Own session/process group so a mid-session Ctrl+C (cancel a turn) doesn't reach the
    # server. It stays available after a successful agent session and is torn down on
    # startup or launch failure.
    child_env = os.environ.copy()
    # Pass the marker out of band so an older launcher ignores it instead of
    # treating an unknown CLI option as a llama-server argument. New launchers
    # consume and preserve it across any Studio re-exec.
    child_env[_START_API_KEY_MARKER_ENV] = "1"
    kwargs: dict = {
        "stdout": log,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": child_env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        server = subprocess.Popen(command, **kwargs)
    finally:
        log.close()  # Popen dup'd the fd; drop the parent's copy
    _auto_served_server = server
    atexit.register(_shutdown_auto_served)

    deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
    progress: Optional[_ModelDownloadProgress] = None
    early_key_seen = False
    try:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                tail = _log_tail(log_path)
                _shutdown_auto_served()
                _fail(f"The Unsloth server stopped before it was ready. Last log lines:\n{tail}")
            tail = _log_tail(log_path, lines = 400)
            if progress is None:
                marker = re.search(
                    rf"^{re.escape(_START_API_KEY_PREFIX)}(sk-unsloth-[^\s]+)$",
                    tail,
                    flags = re.MULTILINE,
                )
                if marker:
                    early_key_seen = True
                    progress = _ModelDownloadProgress(
                        base,
                        marker.group(1),
                        model,
                        load.gguf_variant,
                    )
            if progress is not None:
                progress.poll()
            # New children emit an early key marker, so wait for the final model
            # banner. The fallback preserves compatibility with an older child
            # that only prints its API key after loading has completed.
            ready_signal = "Model loaded:" in tail if early_key_seen else "sk-unsloth-" in tail
            if _studio_healthy(base) and ready_signal:
                if progress is not None:
                    progress.complete()
                    progress.close()
                    progress = None
                return server
            time.sleep(2.0)
    finally:
        if progress is not None:
            progress.close()
    _shutdown_auto_served()
    _fail(
        f"The Unsloth server didn't become ready within {_SERVER_START_TIMEOUT_S}s. See {log_path}."
    )


def _effective_base(base: str) -> str:
    # `unsloth run` binds to `parsed.port or 8888` and serves at the root, so normalize
    # UNSLOTH_STUDIO_URL to plain scheme://host:port. A portless http://127.0.0.1 would
    # otherwise launch on 8888 but poll port 80, and a path like /studio would poll
    # /studio/api/health (404) -- either way hitting the startup timeout. IPv6 literals
    # stay bracketed.
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    if ":" in host:  # bare IPv6 literal (urlparse strips the brackets)
        host = f"[{host}]"
    return f"{parsed.scheme or 'http'}://{host}:{parsed.port or 8888}"


def _require_studio(
    model: Optional[str] = None,
    load: Optional[LoadOptions] = None,
    *,
    serve: bool = False,
    launch: bool = True,
) -> tuple:
    """Return (base, server). server is a Popen only when WE auto-started it."""
    base = find_studio_server()
    if base is not None:
        return base, None
    expected = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888").rstrip("/")
    # Auto-start a local server only for an interactive launch with a model to serve, and
    # only for a plain-HTTP loopback target: never stand in for an explicit remote
    # UNSLOTH_STUDIO_URL, and never for an https:// one -- `unsloth run` serves plain
    # HTTP, so the health poll against https would spin until the startup timeout.
    if (
        serve
        and launch
        and model
        and is_loopback_url(expected)
        and urlparse(expected).scheme == "http"
    ):
        # Normalize to the port unsloth run actually binds, so the health poll and the
        # returned base hit the same server we launch (not a portless :80).
        expected = _effective_base(expected)
        return expected, _start_studio_server(expected, model, load or LoadOptions())
    model_hint = "" if model else " Pass --model to have it start one for you, or"
    _fail(
        f"No running Unsloth server found at {expected}.{model_hint} start one with "
        "`unsloth studio`, or point UNSLOTH_STUDIO_URL at a remote server."
    )


def _key_cache_path() -> Path:
    ensure_studio_backend_path()
    from utils.paths import auth_root
    return auth_root() / "agent_api_key.json"


def _read_cache(cache: Path) -> dict:
    try:
        data = json.loads(cache.read_text(encoding = "utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _server_buckets(servers: dict, base: str) -> dict:
    # Normalise a server's entry to {"saved": [...], "minted": [...]}, tolerating a
    # corrupt/legacy value (bare string/list -> treated as minted, behind the handshake).
    entry = servers.get(base) if isinstance(servers, dict) else None
    if isinstance(entry, list):
        return {"saved": [], "minted": [k for k in entry if isinstance(k, str)]}
    if not isinstance(entry, dict):
        return {"saved": [], "minted": []}

    def _strs(name: str) -> list:
        value = entry.get(name)
        return [k for k in value if isinstance(k, str)] if isinstance(value, list) else []

    return {"saved": _strs("saved"), "minted": _strs("minted")}


def _cached_keys(cache: Path, base: str, source: str) -> list:
    # Keys are scoped per server. `source` splits user-supplied --api-key keys
    # ("saved", trusted for that base) from auto-minted ones ("minted", replayed
    # only after the identity check). Legacy unscoped caches are ignored.
    return _server_buckets(_read_cache(cache).get("servers", {}), base)[source]


def _write_private_json(path: Path, data: dict) -> None:
    # O_CREAT with 0o600 so a file holding an API key is never world-readable,
    # even briefly (existing files keep whatever perms the user set).
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, indent = 2) + "\n")


def _read_json_object(path: Path) -> Optional[dict]:
    # {} when missing, None when it can't be parsed as an object (so the caller
    # leaves a user-managed file untouched rather than clobbering it).
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (ValueError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _subdict(parent: dict, key: str) -> dict:
    child = parent.get(key)
    if not isinstance(child, dict):
        child = parent[key] = {}
    return child


def _remember_key(cache: Path, base: str, key: str, source: str) -> None:
    data = _read_cache(cache)
    servers = data.get("servers")
    if not isinstance(servers, dict):
        servers = data["servers"] = {}
    buckets = _server_buckets(servers, base)
    other = "minted" if source == "saved" else "saved"
    buckets[source] = ([key] + [k for k in buckets[source] if k != key])[:8]
    buckets[other] = [k for k in buckets[other] if k != key]  # a key has one provenance
    new_entry = {"saved": buckets["saved"], "minted": buckets["minted"]}
    if servers.get(base) == new_entry:
        return
    servers[base] = new_entry
    # Collapse legacy unscoped fields.
    data.pop("keys", None)
    data.pop("key", None)
    try:
        _write_private_json(cache, data)
    except OSError:
        pass  # worst case the next launch mints another key


def _key_accepted(base: str, key: str) -> bool:
    # Only a genuine auth rejection (401/403) means "this key is bad -- skip it and try
    # the next cached key or mint a fresh one". A 5xx or a network blip is a server-side
    # outage, not a bad key: fail with a clean message (never a traceback) instead of
    # silently discarding a working key and minting extras against a struggling server.
    try:
        _http_json("GET", f"{base}/v1/models", key)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return False
        _fail(
            f"Unsloth server error while checking an API key ({exc.code}). "
            "The server may be starting up or unhealthy; try again shortly."
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        _fail(
            "Couldn't reach the Unsloth server while checking an API key: "
            f"{getattr(exc, 'reason', None) or exc}"
        )


def _agent_api_key(
    base: str,
    explicit: Optional[str],
    *,
    auto_started: bool = False,
) -> str:
    cache = _key_cache_path()
    if explicit:
        if not auto_started or _key_accepted(base, explicit):
            _remember_key(cache, base, explicit, "saved")
            return explicit
        # The server was auto-started for this run, so an exported
        # UNSLOTH_API_KEY meant for some other server must not fail the
        # launch: the loopback mint path below is guaranteed to work.
        # (An explicit key that the fresh server accepts, e.g. one persisted
        # in this Unsloth home's auth db, is still honored above.)

    # Replay a key the user saved for *this exact* server first (scoped per base,
    # so it only goes back there -- including a remote/SSH-tunnelled Unsloth whose
    # secret the local handshake can't match). Skip ones the server rejects.
    for key in _cached_keys(cache, base, "saved"):
        if _key_accepted(base, key):
            _remember_key(cache, base, key, "saved")
            return key

    # Beyond here we auto-mint or replay an auto-minted key. find_studio_server()
    # trusts a base after only a health check, so both are limited to a loopback
    # server we can cryptographically confirm is ours.
    if not is_loopback_url(base):
        _fail(
            f"No saved API key for {base} and automatic minting only runs against "
            "a local Unsloth. Create an API key in Unsloth → Settings → API and "
            "pass it with --api-key (it is remembered per server), or set "
            "UNSLOTH_API_KEY."
        )
    if not verify_studio_identity(base):
        _fail(
            f"Couldn't verify that {base} is your Unsloth (it may be running as a "
            "different OS user, or another process took the port). Create an API "
            "key in Unsloth → Settings → API and pass it with --api-key, or set "
            "UNSLOTH_API_KEY."
        )

    # Identity verified: replay a previously auto-minted key, else mint a new one.
    for key in _cached_keys(cache, base, "minted"):
        if _key_accepted(base, key):
            _remember_key(cache, base, key, "minted")
            return key

    # Self-issue a JWT (signed with the local secret) and mint a key.
    token = _studio_token()
    if token is None:
        _fail(
            "Couldn't authenticate with the Unsloth server automatically. Create "
            "an API key in Unsloth → Settings → API and pass it with --api-key, "
            "or set UNSLOTH_API_KEY."
        )
    key = _http_json(
        "POST",
        f"{base}/api/auth/api-keys",
        token,
        {"name": "Coding agents (unsloth start)"},
        error = "Couldn't create an API key",
    )["key"]
    _remember_key(cache, base, key, "minted")
    return key


def _loaded_models(base: str, key: str) -> list:
    return _http_json("GET", f"{base}/v1/models", key, error = "Couldn't list models").get("data", [])


_HF_REPO_ID_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _is_hub_model_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if "\\" in text:
        return False
    if text.startswith(("/", "./", "../", "~")):
        return False
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return False
    # A hub id is exactly "namespace/name" over a restricted charset. Anything with
    # extra path segments (e.g. a server-side relative path such as
    # models/Llama/Foo.gguf on a remote Unsloth) is not a hub id and must not be
    # casefold-matched against a differently cased path on a case-sensitive
    # filesystem. This is host independent, unlike the existence probe below which
    # cannot see a path that only exists on the server.
    parts = text.split("/")
    if len(parts) != 2:
        return False
    if any(part in ("", ".", "..") or not _HF_REPO_ID_SEGMENT_RE.match(part) for part in parts):
        return False
    try:
        if Path(os.path.expanduser(text)).exists():
            return False
    except OSError:
        return False
    return True


def _model_id_matches(
    actual: object,
    requested: object,
    *,
    allow_casefold: bool = True,
) -> bool:
    if actual == requested:
        return True
    # Case-insensitive matching is only safe when the local existence probe in
    # _is_hub_model_id is authoritative, i.e. against a loopback Unsloth on this host.
    # Against a remote Unsloth a two-segment string is indistinguishable from a
    # server-side relative path (e.g. Models/Foo vs models/foo), so casefolding it
    # could attach to the wrong model on a case-sensitive server; defer to an exact
    # match there and let the load endpoint resolve the requested path.
    if not allow_casefold:
        return False
    if not (_is_hub_model_id(actual) and _is_hub_model_id(requested)):
        return False
    return str(actual).casefold() == str(requested).casefold()


def _resolve_model(
    base: str,
    key: str,
    requested: Optional[str],
    load: LoadOptions = LoadOptions(),
) -> dict:
    models = _loaded_models(base, key)
    load_requested = False
    # Only casefold-match ids against a loopback Unsloth, where _is_hub_model_id's
    # local existence probe can actually reject a server-side path; see the note there.
    allow_casefold = is_loopback_url(base)
    # /v1/models reports the model id but not the active GGUF variant or runtime load
    # settings, so an id match alone can hide the wrong quant (Q8_0 serving while the
    # user asked for UD-Q4_K_XL). When the user passed any explicit load knob, defer to
    # /api/inference/load: the server's already-loaded dedup answers "already_loaded"
    # without reloading when the variant AND settings match, so a second session running
    # the same command still attaches without evicting the first.
    load_has_overrides = bool(
        load.gguf_variant or load.max_seq_length or not load.load_in_4bit or load.tensor_parallel
    )
    # /v1/models also lists cached-but-unloaded catalog entries (loaded == False);
    # matching one would skip /api/inference/load and leave the agent pointed at a
    # model that is not resident, so only attach to an entry that is actually loaded.
    match = (
        None
        if requested and load_has_overrides
        else next(
            (
                m
                for m in models
                if _model_id_matches(m.get("id"), requested, allow_casefold = allow_casefold)
                and m.get("loaded") is not False
            ),
            None,
        )
    )
    if requested and match is None:
        load_requested = True
        active = next((m for m in models if m.get("loaded") is not False), None)
        active_id = active.get("id") if active else None
        if active_id and not _model_id_matches(
            active_id,
            requested,
            allow_casefold = allow_casefold,
        ):
            typer.echo(f"Switching the Unsloth server from {active_id} to {requested}.")
            typer.echo("This unloads the current model for every attached session.")
        # Mirror `unsloth run`'s load knobs; keep the default payload as just
        # model_path so a bare `--model` load is unchanged.
        payload = {"model_path": requested}
        if load.gguf_variant:
            payload["gguf_variant"] = load.gguf_variant
        if load.max_seq_length:
            payload["max_seq_length"] = load.max_seq_length
        if not load.load_in_4bit:
            payload["load_in_4bit"] = False
        if load.tensor_parallel:
            payload["tensor_parallel"] = True
        loaded = _load_model_with_progress(base, key, requested, load, payload)
        if loaded.get("status") == "already_loaded":
            typer.echo(f"Reusing loaded model: {_display_model_spec(requested, load.gguf_variant)}")
        # Unsloth registers the model under a canonical id (resolved identifier,
        # casing) that /v1/models echoes but which may differ from the path we
        # passed; match on the id the load reports so we don't silently fall
        # through to models[0] and connect to a different loaded model.
        wanted = {requested}
        if isinstance(loaded, dict):
            wanted |= {loaded.get("model"), loaded.get("display_name")} - {None}
        models = _loaded_models(base, key)
        match = next(
            (
                m
                for m in models
                if m.get("loaded") is not False
                and any(
                    _model_id_matches(m.get("id"), w, allow_casefold = allow_casefold) for w in wanted
                )
            ),
            None,
        )
    if match is not None:
        if requested and not load_requested:
            typer.echo(f"Reusing loaded model: {_display_model_spec(requested, load.gguf_variant)}")
        return match
    if requested:
        # We asked Unsloth to load it and it didn't surface in /v1/models; don't
        # silently hand back an unrelated loaded model.
        _fail(
            f"Unsloth didn't report '{requested}' as loaded. Double-check the model "
            "id, or load it from the model dropdown in the UI."
        )
    if not models:
        _fail(
            "No model is loaded in Unsloth. Load one from the model dropdown in "
            "the UI, or pass --model <hf-id-or-path> to load it from here."
        )
    resident = next((m for m in models if m.get("loaded") is not False), None)
    if resident is None:
        _fail(
            "No model is currently resident in Unsloth. Pass --model <hf-id-or-path> "
            "to reload one, or load it from the model dropdown in the UI."
        )
    return resident


def _require_gguf_for_codex(base: str, key: str, model_id: str) -> None:
    # Codex always streams, and Unsloth only streams /v1/responses from llama-server.
    try:
        status = _http_json("GET", f"{base}/api/inference/status", key)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return  # older server without the endpoint; don't block the launch
        raise
    if status.get("is_gguf"):
        return
    hint = model_id if "gguf" in model_id.lower() else f"{model_id}-GGUF"
    _fail(
        f"Codex needs a GGUF model served by llama-server, but {model_id} is on "
        f"the transformers backend. Try: unsloth start codex --model {hint}"
    )


_DYNAMIC_SECTIONS_FLAG = "--exclude-dynamic-system-prompt-sections"


def _claude_settings_overlay(model_id: str) -> str:
    # Session-only `claude --settings` overlay (command-line tier, no ~/.claude write):
    # suppress the attribution header, and pin availableModels to the served model so a
    # user allowlist can't reject it. The pin must be non-empty; [] is ignored.
    return json.dumps(
        {"env": {"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"}, "availableModels": [model_id]}
    )


def _claude_version() -> Optional[tuple]:
    # None = no local `claude` (a --no-launch printout for another machine; assume a
    # current build). An unparseable version is treated as too old for the new flags.
    executable = _which_with_install_dirs("claude")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output = True, text = True, timeout = 10
        )
        # Pull the X.Y.Z out of the output rather than assuming it is the first token.
        # claude prints it first today ("2.1.98 (Claude Code)"), but a format change
        # (e.g. "claude version 2.1.98") shouldn't silently drop the optimization flags;
        # no match falls through to "too old", same as an unparseable version.
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
        return tuple(int(part) for part in match.groups()) if match else (0,)
    except Exception:
        return (0,)


def _claude_flags(model_id: str) -> list:
    # KV-cache-preserving flags: move per-session context out of the system prompt and pass
    # the session overlay. claude < 2.1.98 rejects unknown flags; no local binary means a
    # printout for another machine, so assume a current build.
    version = _claude_version()
    if version is not None and version < (2, 1, 98):
        return []
    return [_DYNAMIC_SECTIONS_FLAG, "--settings", _claude_settings_overlay(model_id)]


def _merge_codex_config(existing: str, base: str) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{_CODEX_PROFILE}"\n'
    # Drop the provider table and any stale [model_providers.unsloth_api.*] subtables.
    stale = (_PROVIDER_HEADER, _PROVIDER_HEADER[:-1] + ".")
    text = "".join(c for c in chunks if not c.startswith(stale))
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + (
        f"{_PROVIDER_HEADER}\n"
        'name = "Unsloth Studio"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
        f'env_key = "{_CODEX_ENV_KEY}"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
    )


# Keep custom-model behavior aligned with Codex's own unknown-model fallback. This
# Apache-2.0 prompt is copied from openai/codex rust-v0.144.0 models-manager/prompt.md.
_CODEX_FALLBACK_PROMPT = Path(__file__).parent.parent / "codex_fallback_prompt.md"
_CODEX_MODEL_CATALOG_MIN_VERSION = (0, 110, 0)


def _codex_supports_model_catalog() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A --no-launch recipe may be copied to another machine; assume a current Codex.
        return True
    try:
        output = subprocess.check_output(
            [executable, "--version"], text = True, timeout = 10, stderr = subprocess.DEVNULL
        )
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return bool(match) and tuple(int(part) for part in match.groups()) >= (
        _CODEX_MODEL_CATALOG_MIN_VERSION
    )


def _codex_model_catalog(model: dict) -> dict:
    """Return conservative metadata for an Unsloth model unknown to Codex's built-in catalog."""
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served by Unsloth Studio",
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _CODEX_FALLBACK_PROMPT.read_text(encoding = "utf-8"),
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }
    if window:
        entry["context_window"] = int(window)
        entry["max_context_window"] = int(window)
    return {"models": [entry]}


def write_codex_config(base: str, model: dict, home: Path) -> None:
    home.mkdir(parents = True, exist_ok = True)

    config = home / "config.toml"
    existing = config.read_text(encoding = "utf-8") if config.exists() else ""
    merged = _merge_codex_config(existing, base)
    if merged != existing:
        config.write_text(merged, encoding = "utf-8")
        typer.echo(f"Updated {config}")

    # oss_provider here too: codex --oss picks the provider from it, and the
    # profile layer must beat a user-set value (e.g. "ollama") in config.toml.
    profile_text = (
        f'oss_provider = "{_CODEX_PROFILE}"\n'
        f'model_provider = "{_CODEX_PROFILE}"\n'
        f"model = {json.dumps(model['id'])}\n"
    )
    if _codex_supports_model_catalog() and _CODEX_FALLBACK_PROMPT.is_file():
        catalog = home / "model-catalog.json"
        catalog_text = json.dumps(_codex_model_catalog(model), indent = 2) + "\n"
        if not catalog.exists() or catalog.read_text(encoding = "utf-8") != catalog_text:
            catalog.write_text(catalog_text, encoding = "utf-8")
            typer.echo(f"Updated {catalog}")
        # Resolve relative to the profile file. This also survives WSL launching a Windows
        # Codex binary, where a Linux absolute path inside TOML would not be usable.
        profile_text += f"model_catalog_json = {json.dumps(catalog.name)}\n"

    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def write_codex_subagent_config(base: str, model: dict, home: Path) -> Path:
    """Write a session-scoped Codex custom agent without replacing the main model."""
    home.mkdir(parents = True, exist_ok = True)
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    catalog_name = "unsloth-model-catalog.json"
    text = (
        f"name = {json.dumps(_SUBAGENT_NAME)}\n"
        f"description = {json.dumps(_SUBAGENT_DESCRIPTION)}\n"
        f"developer_instructions = {json.dumps(_SUBAGENT_INSTRUCTIONS)}\n"
        f"model_provider = {json.dumps(_CODEX_PROFILE)}\n"
        f"model = {json.dumps(model_id)}\n"
    )
    if _codex_supports_model_catalog() and _CODEX_FALLBACK_PROMPT.is_file():
        catalog = home / catalog_name
        catalog_text = json.dumps(_codex_model_catalog(model), indent = 2) + "\n"
        if not catalog.exists() or catalog.read_text(encoding = "utf-8") != catalog_text:
            catalog.write_text(catalog_text, encoding = "utf-8")
            typer.echo(f"Updated {catalog}")
        text += f"model_catalog_json = {json.dumps(catalog_name)}\n"
    if window:
        text += f"model_context_window = {int(window)}\n"
    text += (
        f"\n{_PROVIDER_HEADER}\n"
        'name = "Unsloth Studio"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
        f'env_key = "{_CODEX_ENV_KEY}"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
    )
    path = home / f"{_SUBAGENT_NAME}.toml"
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        path.write_text(text, encoding = "utf-8")
        typer.echo(f"Updated {path}")
    return path


def _agent_config_path(path: Path, command: list) -> str:
    """Translate a generated config path when a Windows agent runs through WSL."""
    return _wsl_windows_path(path) if _wsl_windows_executable(command) else str(path)


def _codex_subagent_flags(path: Path) -> list[str]:
    config_path = _agent_config_path(path, ["codex"])
    return [
        "--enable",
        "multi_agent",
        "-c",
        f"agents.{_SUBAGENT_NAME}.description={json.dumps(_SUBAGENT_DESCRIPTION)}",
        "-c",
        f"agents.{_SUBAGENT_NAME}.config_file={json.dumps(config_path)}",
    ]


def _wsl_windows_executable(command: list) -> Optional[str]:
    if os.name == "nt" or not os.environ.get("WSL_DISTRO_NAME"):
        return None
    executable = shutil.which(command[0])
    if executable and executable.startswith("/mnt/"):
        return executable
    return None


def _wsl_windows_path(path: Path) -> str:
    try:
        translated = subprocess.check_output(["wslpath", "-w", str(path)], text = True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate WSL path {path}: {exc}")
    if not translated:
        _fail(f"Could not translate WSL path {path}")
    return translated


def _looks_like_path(value: str) -> bool:
    # A var only wants the WSLENV /p flag if its value is a filesystem path: an
    # absolute POSIX path (/...), a UNC path (\\...), or a drive-qualified Windows
    # path (C:...). Scalar knobs (e.g. a numeric context window) must pass through
    # untranslated, so they get no flag.
    return bool(value) and (value.startswith(("/", "\\")) or (len(value) >= 2 and value[1] == ":"))


def _wsl_bridge_names(env: dict, unset_env: tuple) -> tuple:
    # Build the WSLENV share list for a Windows shim reached from WSL. Path-valued
    # vars get /p so WSLENV translates them to the Windows path the /mnt shim can
    # actually open; a cleared var carries no value to translate.
    names = [name + ("/p" if _looks_like_path(value) else "") for name, value in env.items()]
    names.extend(unset_env)
    return tuple(dict.fromkeys(names))


def _merge_wslenv(current: str, names: tuple) -> str:
    # Index WSLENV entries by bare var name, preserving first-seen order. The vars we
    # bridge are applied last so our entry wins: a user's pre-existing unflagged "HOME"
    # is upgraded to "HOME/p" (rather than left as-is), since WSLENV ignores a duplicate
    # name and a bare entry would leave the path untranslated for a Windows shim.
    ordered = []
    by_name = {}
    for entry in (*current.split(":"), *names):
        if not entry:
            continue
        base = entry.split("/", 1)[0]
        if base not in by_name:
            ordered.append(base)
        by_name[base] = entry
    return ":".join(by_name[base] for base in ordered)


def _powershell_quote(arg: str) -> str:
    # PowerShell reads single-quoted strings literally (an embedded ' is doubled), so
    # JSON args such as `--settings {"env":...}` survive intact. list2cmdline's
    # backslash-escaped double quotes are cmd.exe syntax and PowerShell mis-parses them.
    if arg and re.fullmatch(r"[A-Za-z0-9_./:=+-]+", arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def _print_env(
    env: dict,
    command: list,
    unset_env: tuple = (),
    wsl_env_bridge: tuple = (),
) -> None:
    if os.name == "nt":
        for name in unset_env:
            typer.echo(f"Remove-Item Env:{name} -ErrorAction SilentlyContinue")
        for name, value in env.items():
            # PowerShell: ` is the escape char, and $ triggers expansion inside "".
            escaped = value.replace("`", "``").replace('"', '`"').replace("$", "`$")
            typer.echo(f'$env:{name} = "{escaped}"')
        typer.echo(" ".join(_powershell_quote(arg) for arg in command))
        return
    for name in unset_env:
        typer.echo(f"export {name}=" if wsl_env_bridge else f"unset {name}")
    for name, value in env.items():
        typer.echo(f"export {name}={shlex.quote(value)}")
    if wsl_env_bridge:
        typer.echo(
            f"export WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    # The final line is a SELF-CONTAINED one-liner (inline env, VAR=... cmd) rather than a
    # bare command. People copy just the last line, and a bare `codex`/`claude` would then
    # run against their real ~/.codex or Anthropic credentials with zero isolation -- e.g.
    # inheriting a pre-existing damaged ~/.codex state DB and blaming the recipe. Inline
    # assignments scope every var (and empty-string the conflicting ones) to this single
    # invocation, so a partial copy behaves the same as pasting the whole block.
    inline = [f"{name}=" for name in unset_env]
    inline += [f"{name}={shlex.quote(value)}" for name, value in env.items()]
    if wsl_env_bridge:
        inline.append(
            f"WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    typer.echo(" ".join((*inline, shlex.join(command))))


def _refresh_windows_path() -> None:
    # Merge Windows registry PATH hives after the current process PATH so a
    # freshly installed agent is visible without changing existing precedence.
    if os.name != "nt":
        return
    try:
        import winreg
    except Exception:
        return

    entries = []
    seen = set()

    def add_path(value: str) -> bool:
        added = False
        for entry in str(value).split(os.pathsep):
            entry = entry.strip()
            if not entry:
                continue
            key = os.path.normcase(entry).casefold()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
            added = True
        return added

    add_path(os.environ.get("PATH", ""))
    added_registry = False
    hives = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    for root, sub in hives:
        try:
            with winreg.OpenKey(root, sub) as key:
                value, _ = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if value:
            added_registry = add_path(os.path.expandvars(str(value))) or added_registry
    if added_registry:
        os.environ["PATH"] = os.pathsep.join(entries)


def _augment_path_with_install_dirs() -> None:
    # Append known install dirs to PATH so a freshly installed agent resolves without a new
    # shell: some installers write the binary but not PATH (claude drops ~/.local/bin and
    # only prints a note; npm -g shims land in %APPDATA%\npm). Appended, so precedence holds.
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        return
    candidates = [home / ".local" / "bin"]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm")
    current = os.environ.get("PATH")
    if current is None:
        # PATH unset: shutil.which() and exec*p* fall back to os.defpath (e.g. /bin:/usr/bin), so
        # keep that default instead of collapsing to just the install dirs (which would hide a
        # system-installed agent and strip the launched child's normal PATH). An explicitly empty
        # PATH is left as-is: like shutil.which, it means "search nothing", not os.defpath.
        current = os.defpath
    seen = {os.path.normcase(entry) for entry in current.split(os.pathsep) if entry}
    additions = [
        str(directory)
        for directory in candidates
        if directory.is_dir() and os.path.normcase(str(directory)) not in seen
    ]
    if additions:
        os.environ["PATH"] = os.pathsep.join([current, *additions] if current else additions)


def _which_with_install_dirs(name: str) -> Optional[str]:
    # shutil.which(name), but searching the known agent install dirs too, so a version probe
    # resolves the same binary _launch() will (it augments PATH before it runs). Without this an
    # agent present only in ~/.local/bin / %APPDATA%\npm is missed, wrongly assumed current, and
    # launched with flags an older build rejects. PATH is restored afterward: only _launch()
    # should persist the augmentation for the child process.
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    try:
        return shutil.which(name)
    finally:
        if original is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original


def _install_source(install_hint: str) -> Optional[str]:
    """The first http(s) URL an install hint fetches, or None (e.g. an npm install)."""
    match = re.search(r"https?://[^\s'\")]+", install_hint)
    return match.group(0) if match else None


def _pinned_raw_github_commit(source: str) -> Optional[str]:
    """Return the immutable full commit in a raw GitHub URL, if present."""
    match = re.match(
        r"^https://raw\.githubusercontent\.com/[^/]+/[^/]+/([0-9a-f]{40})/",
        source,
        flags = re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _install_agent(name: str, install_hint: str) -> Optional[str]:
    # Missing agent under --launch: offer to run its documented install command, then
    # re-resolve it on PATH. Consent-based (we never auto-run a remote install script
    # silently), and a non-interactive stdin cannot answer the prompt, so both the
    # no-TTY and declined cases return None and let the caller print the hint and exit.
    if not sys.stdin.isatty():
        return None
    typer.echo(f"`{name}` is not installed.")
    # Make the supply-chain risk explicit before the prompt: these are the vendors'
    # own installers (curl | bash, irm | iex, npm), run with the user's privileges,
    # and nothing checks a signature or hash on the fetched content. Naming the source
    # turns a blind "yes" into informed consent.
    source = _install_source(install_hint)
    if source:
        pinned_commit = _pinned_raw_github_commit(source)
        if pinned_commit:
            warning = (
                "Security warning: This will download and execute a third-party script "
                f"from {source} with your privileges. Unsloth pins this content to "
                f"immutable upstream commit {pinned_commit}, but does not independently "
                "verify or sandbox it. Continue only if you trust this source and commit."
            )
        else:
            warning = (
                "Security warning: This will download and execute an unverified third-party "
                f"script from {source} with your privileges. Unsloth does not pin or verify "
                "the downloaded content. Continue only if you trust this source."
            )
    else:
        warning = (
            f"This will RUN `{install_hint}` with your privileges; "
            "there is no signature or hash check."
        )
    typer.secho(warning, fg = "yellow", err = True)
    if not typer.confirm(f"Install `{name}` now with `{install_hint}`?", default = False):
        return None
    # Run each hint through its shell: PowerShell on Windows, /bin/sh elsewhere.
    # -ExecutionPolicy Bypass is process-scoped (nothing persistent) so npm's npm.ps1 and
    # irm | iex run under the Windows default Restricted policy instead of failing with a
    # PSSecurityException.
    if os.name == "nt":
        install_command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            install_hint,
        ]
    else:
        install_command = ["/bin/sh", "-c", install_hint]
    if subprocess.run(install_command).returncode != 0:
        message = f"Install command failed. Run it yourself, then re-run: {install_hint}"
        if os.name == "nt":
            # A hand-run retry can still hit the policy; point at the one-time per-user fix.
            message += (
                "\nIf it fails because running scripts is disabled (PSSecurityException), "
                "allow local scripts for your user, then retry:\n"
                "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
            )
        _fail(message)
    # Resolve the freshly installed agent without a shell restart: pull registry PATH
    # (Windows) plus well-known install dirs the installer may not have added to PATH.
    _refresh_windows_path()
    _augment_path_with_install_dirs()
    executable = shutil.which(name)
    if executable is None:
        _fail(
            f"`{name}` installed but isn't on PATH yet. Open a new shell (or add it to "
            f"PATH), then re-run. Install command: {install_hint}"
        )
    return executable


def _wsl_shim_env(command: list, env: dict, unset_env: tuple) -> tuple[dict, tuple]:
    wsl_env_bridge = _wsl_bridge_names(env, unset_env) if _wsl_windows_executable(command) else ()
    if not wsl_env_bridge:
        return env, wsl_env_bridge
    # Bridge PWD via WSLENV (PWD/p) so the Windows shim finds its project root from the
    # live cwd, not a stale inherited Linux PWD. Don't freeze env["PWD"]: a --no-launch
    # recipe must translate the live PWD when run, not when generated; _launch overrides it.
    return env, (*wsl_env_bridge, "PWD/p")


def _launch(
    command: list,
    env: dict,
    install_hint: str,
    unset_env: tuple = (),
) -> int:
    # Resolve well-known install dirs (e.g. ~/.local/bin) first, so an already-installed
    # agent not yet on PATH is found instead of prompting a needless reinstall.
    _augment_path_with_install_dirs()
    executable = shutil.which(command[0]) or _install_agent(command[0], install_hint)
    if executable is None:
        _fail(f"`{command[0]}` not found on PATH. Install it with: {install_hint}")
    env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
    child_env = dict(os.environ)
    if wsl_env_bridge:
        # Override stale inherited PWD with the real cwd so the shim resolves the project root.
        env = {**env, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_env_bridge)
        for name in unset_env:
            child_env[name] = ""
    else:
        for name in unset_env:
            child_env.pop(name, None)
    child_env.update(env)
    # Ctrl+C cancels a turn inside the agent; don't let it kill this wrapper.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        code = subprocess.run([executable, *command[1:]], env = child_env).returncode
    finally:
        signal.signal(signal.SIGINT, previous)
    # Negative returncode means killed by signal N; shells expect 128+N.
    return code if code >= 0 else 128 - code


def _connect(
    api_key: Optional[str],
    model: Optional[str],
    load: LoadOptions = LoadOptions(),
    *,
    serve: bool = False,
    launch: bool = True,
) -> tuple:
    # `--model org/name:QUANT` is shorthand for `--model org/name --gguf-variant QUANT`.
    # Split it before we match/serve so the attach path resolves against the already-loaded
    # `org/name` (listed without the suffix) instead of reloading a `:`-suffixed repo id --
    # which Unsloth rejects and which would evict a model another session is using.
    if model:
        repo, variant = _split_repo_variant(model)
        if variant:
            model = repo
            if not load.gguf_variant:
                load = load._replace(gguf_variant = variant)
    base, server = _require_studio(model, load, serve = serve, launch = launch)
    try:
        key = _agent_api_key(base, api_key, auto_started = server is not None)
        # A server we just started has exactly the requested model loaded, so resolve to
        # whatever it is serving instead of re-matching the raw --model string.
        entry = _resolve_model(base, key, None if server is not None else model, load)
    except BaseException:
        _shutdown_auto_served()
        raise
    return base, key, entry


def _run(
    base: str,
    entry: dict,
    env: dict,
    command: list,
    *,
    launch: bool,
    install_hint: str,
    unset_env: tuple = (),
    clear_screen: bool = False,
) -> None:
    # Some agents (Pi) render inline from wherever the cursor sits: their first
    # paint assumes a clean screen rather than clearing or entering the
    # alternate screen themselves. Hand them one so the session doesn't start
    # mid-scroll under our connection output. click.clear() is cross-platform
    # and a no-op when stdout is not a terminal (piped/CI), so transcripts and
    # --no-launch recipes stay intact.
    if launch and clear_screen:
        click.clear()
    typer.echo(f"Unsloth ready at {base} · model {entry['id']}")
    if not launch:
        env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env)
        _print_env(env, command, unset_env = unset_env, wsl_env_bridge = wsl_env_bridge)
        if _keep_auto_served():
            typer.echo(f"Unsloth Studio is still running at {base}.")
            typer.echo("Stop it with: unsloth studio stop")
        return
    try:
        code = _launch(command, env, install_hint = install_hint, unset_env = unset_env)
    except BaseException:
        # Startup succeeded but the agent itself could not launch. In that failure
        # path, retain the old cleanup behavior instead of orphaning a surprise server.
        _shutdown_auto_served()
        raise
    auto_started = _auto_served_server is not None
    kept = _keep_auto_served()
    if auto_started and not kept:
        typer.echo(f"The auto-started Unsloth server at {base} stopped during the session.")
        raise typer.Exit(code = code)
    if is_loopback_url(base):
        typer.echo(f"Unsloth Studio is still running at {base}.")
        typer.echo("Stop it with: unsloth studio stop")
    else:
        typer.echo(f"The remote Unsloth server is still running at {base}.")
    raise typer.Exit(code = code)


def _agents_config_root() -> Path:
    ensure_studio_backend_path()
    from utils.paths import auth_root
    return auth_root() / "agents"


@contextlib.contextmanager
def _session_config(
    agent: str,
    launch: bool,
    persist: bool = False,
):
    """Yield a private directory for an agent's session config (never the user's own).

    launch (default): an ephemeral temp dir removed after the agent process exits, so
    nothing persists. no-launch: a stable Unsloth-owned dir (the printed recipe is run
    later on this machine), reused across runs. persist (from --persist): use that same
    stable dir even for a launch, so the agent's session survives the exit and can be
    resumed next time. Either way the user's real ~/.<agent> config is left untouched.
    """
    if launch and not persist:
        path = Path(tempfile.mkdtemp(prefix = f"unsloth-{agent}-"))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors = True)
    else:
        # Never wipe this dir: a previously printed recipe may still be running
        # an agent whose sessions/state live here, and every config writer
        # merges idempotently into an existing home anyway. Writers must also
        # reset any state a previous run's flags left behind (--yolo especially),
        # since files here outlive the invocation that wrote them.
        path = _agents_config_root() / agent
        path.mkdir(parents = True, exist_ok = True, mode = 0o700)
        yield path


def write_openclaw_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    workspace_path: Optional[str] = None,
) -> None:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an 'unsloth' provider there "
            "yourself, or move the file aside and re-run.",
            err = True,
        )
        return
    before = json.dumps(config, sort_keys = True)
    # Unsloth is a generic OpenAI-compatible /v1 endpoint (the vLLM/LM Studio path).
    provider_model = {"id": model["id"], "name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        provider_model["contextWindow"] = int(window)
    models = _subdict(config, "models")
    models.setdefault("mode", "merge")
    _subdict(models, "providers")["unsloth"] = {
        "baseUrl": f"{base}/v1",
        "apiKey": key,
        "api": "openai-completions",
        "models": [provider_model],
    }
    # Pin a default model, else OpenClaw drops into its setup agent ("no models available").
    agents = _subdict(config, "agents")
    defaults = _subdict(agents, "defaults")
    _subdict(defaults, "model")["primary"] = f"unsloth/{model['id']}"
    # OPENCLAW_STATE_DIR does not relocate the workspace. Keep it beside the managed
    # config so ephemeral launches avoid ~/.openclaw and persisted sessions retain it.
    workspace = path.parent / "workspace"
    workspace.mkdir(parents = True, exist_ok = True, mode = 0o700)
    defaults["workspace"] = workspace_path or str(workspace)
    # Per-agent paths override agents.defaults.workspace and OPENCLAW_STATE_DIR. This
    # config is itself an isolated Unsloth copy, so remove stale explicit paths and let
    # OpenClaw resolve every listed agent beneath the managed defaults/state directory.
    agent_list = agents.get("list")
    if isinstance(agent_list, list):
        for agent_config in agent_list:
            if isinstance(agent_config, dict):
                agent_config.pop("workspace", None)
                agent_config.pop("agentDir", None)
    # Unauthenticated loopback gateway: without auth.mode=none the client won't open
    # the websocket. The daemon must still be started separately (`openclaw gateway`).
    gateway = _subdict(config, "gateway")
    gateway.setdefault("mode", "local")
    _subdict(gateway, "auth").setdefault("mode", "none")
    if yolo:
        # OpenClaw has no --yolo flag, and it gates tool execution on BOTH the
        # tools.exec config AND a host-local approvals file (the stricter wins), so
        # setting only the config still lets the agent prompt/deny. Set both, mirroring
        # `openclaw exec-policy preset yolo`.
        exec_policy = _subdict(_subdict(config, "tools"), "exec")
        exec_policy["host"] = "gateway"
        exec_policy["security"] = "full"
        exec_policy["ask"] = "off"
        # Approvals file in OPENCLAW_STATE_DIR (== this config's dir). ask=off means
        # nothing is ever prompted, so the runtime socket block is unnecessary here.
        approvals = path.parent / "exec-approvals.json"
        _write_private_json(
            approvals,
            {"version": 1, "defaults": {"security": "full", "ask": "off", "askFallback": "full"}},
        )
        typer.echo(f"Updated {approvals}")
    else:
        # The no-launch config dir is reused across runs, so a previous --yolo run may
        # have left auto-approval state behind. OpenClaw treats an omitted exec policy as
        # security=full, ask=off on the gateway host, so deleting the keys would keep
        # auto-approval on: a non-yolo run must WRITE a prompting policy. Only a
        # permissive/yolo policy is replaced; a stricter one set by hand survives.
        tools = config.get("tools")
        exec_policy = tools.get("exec") if isinstance(tools, dict) else None
        exec_policy = exec_policy if isinstance(exec_policy, dict) else {}
        # Match ONLY the exact fingerprint --yolo writes (host=gateway, security=full,
        # ask=off, all explicit, no mode); anything else is left untouched. host=auto or an
        # omitted host resolves to security=deny under an active sandbox, so treating those
        # as the permissive gateway default would broaden a fresh sandboxed config from
        # deny to allowlist. host=node and host=sandbox are user-set (--yolo only writes
        # gateway). tools.exec.mode is OpenClaw's normalized knob (it cannot be combined
        # with security/ask, and OpenClaw never rewrites our security/ask write into it),
        # so a mode is always a deliberate user policy; never clobber it.
        permissive = (
            "mode" not in exec_policy
            and exec_policy.get("host") == "gateway"
            and exec_policy.get("security") == "full"
            and exec_policy.get("ask") == "off"
        )
        if permissive:
            exec_policy = _subdict(_subdict(config, "tools"), "exec")
            exec_policy.pop("host", None)  # routing only; defaults to the gateway host
            exec_policy["security"] = "allowlist"  # only allowlisted commands skip approval
            exec_policy["ask"] = "on-miss"  # prompt on every non-allowlisted command
        # Drop the yolo defaults from the host approvals file (a stricter default set by
        # the user or OpenClaw is kept). With a prompting tools.exec the stricter of the
        # two layers wins, so an omitted approvals default still prompts.
        approvals = path.parent / "exec-approvals.json"
        if approvals.exists():
            state = _read_json_object(approvals)
            if state is not None:
                defaults = state.get("defaults")
                # Strip the defaults only when they are exactly the yolo fingerprint; a
                # user-managed mixed policy that merely shares a field (e.g. askFallback=full,
                # whose omitted default is deny) must be kept intact.
                yolo_defaults = (("security", "full"), ("ask", "off"), ("askFallback", "full"))
                is_yolo = isinstance(defaults, dict) and all(
                    defaults.get(k) == v for k, v in yolo_defaults
                )
                if is_yolo:
                    for k, _ in yolo_defaults:
                        del defaults[k]
                    if not defaults:
                        del state["defaults"]
                    if set(state) <= {"version"}:
                        # Nothing left but our own yolo payload: remove it.
                        approvals.unlink()
                        typer.echo(f"Removed {approvals}")
                    else:
                        # Keep approvals OpenClaw itself recorded; only the yolo defaults go.
                        _write_private_json(approvals, state)
                        typer.echo(f"Updated {approvals}")
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def write_opencode_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    as_subagent: bool = False,
) -> dict:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an '{_OPENCODE_PROVIDER}' provider "
            "there yourself, or move the file aside and re-run.",
            err = True,
        )
        return {}
    before = json.dumps(config, sort_keys = True)
    config.setdefault("$schema", "https://opencode.ai/config.json")
    # The session provider is registered under a dedicated id (_OPENCODE_PROVIDER)
    # that a user's disabled_providers list would never target, so it is always
    # selectable without this overlay having to reconstruct or override OpenCode's
    # disabled_providers resolution.
    model_entry = {"name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # A custom-provider model with no limit defaults to context 0, which silently
        # disables OpenCode's auto-compaction; declare the real window (and a sane
        # output cap) so it compacts instead of overflowing the server.
        model_entry["limit"] = {"context": window, "output": min(window // 4, 8192)}
    _subdict(config, "provider")[_OPENCODE_PROVIDER] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Unsloth Studio",
        "options": {"baseURL": f"{base}/v1", "apiKey": key},
        "models": {model["id"]: model_entry},
    }
    # OpenCode selects a model by "<providerID>/<modelID>". Normal mode pins it
    # as the session model. Subagent mode leaves the user's main and small models
    # alone, while making the same local model available to @unsloth and /models.
    opencode_model = f"{_OPENCODE_PROVIDER}/{model['id']}"
    if as_subagent:
        for field in ("model", "small_model"):
            if str(config.get(field) or "").startswith(f"{_OPENCODE_PROVIDER}/"):
                config.pop(field, None)
        managed_compaction = (
            {"auto": True, "reserved": max(1, window // 10)} if window else None
        )
        if managed_compaction and config.get("compaction") == managed_compaction:
            config.pop("compaction", None)
        _subdict(config, "agent")[_SUBAGENT_NAME] = {
            "description": _SUBAGENT_DESCRIPTION,
            "mode": "subagent",
            "model": opencode_model,
            "prompt": _SUBAGENT_INSTRUCTIONS,
        }
    else:
        config["model"] = opencode_model
        agents = config.get("agent")
        if isinstance(agents, dict):
            agents.pop(_SUBAGENT_NAME, None)
            if not agents:
                config.pop("agent", None)
    if window and not as_subagent:
        # Compact with ~10% headroom (near 90% full). The fixed 20k-token default
        # buffer over-compacts, or never settles, on a small local context.
        compaction = _subdict(config, "compaction")
        compaction["auto"] = True
        compaction["reserved"] = max(1, window // 10)
    tools = ("edit", "bash", "webfetch")
    if yolo:
        # Fallback for commands without native --auto and for the append-safe bare
        # --no-launch command (subcommand unknown yet). Rides inline (OPENCODE_CONFIG_CONTENT)
        # so it wins over a project config. TUI and `run` launches use --auto and call here
        # with yolo=False, letting OpenCode preserve explicit deny rules.
        session_permission = {t: "allow" for t in tools}
        session_permission["external_directory"] = {"*": "allow"}
        config["permission"] = dict(session_permission)
    else:
        # Undo only what --yolo wrote: our yolo sets an explicit per-tool "allow" for these
        # three tools, so flip exactly those explicit allows back to "ask". A "deny"/"ask",
        # a granular object, a string, or a "*" catch-all is the user's own rule and is left
        # untouched. We do NOT carry a permission inline for a non-yolo session: since
        # OPENCODE_CONFIG_CONTENT outranks the project opencode.json we cannot read, any
        # value forced there would override the user's project rules (weakening a project
        # deny, or auto-approving through a granular object's permissive default). Clearing
        # our own persisted yolo state is the fix; the project's own permissions are honored.
        session_permission: dict = {}
        permission = config.get("permission")
        if isinstance(permission, dict):
            for tool in tools:
                if permission.get(tool) == "allow":
                    permission[tool] = "ask"
            if permission.get("external_directory") == {"*": "allow"}:
                permission["external_directory"] = {"*": "ask"}
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")
    return session_permission


def write_hermes_config(base: str, model: dict, path: Path) -> None:
    import yaml

    config: dict = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except (yaml.YAMLError, OSError):
            typer.echo(
                f"Warning: couldn't parse {path} — configure the custom endpoint "
                "there yourself, or move the file aside and re-run.",
                err = True,
            )
            return
        if isinstance(loaded, dict):
            config = loaded
        elif loaded is not None:
            # Non-empty, non-mapping YAML is a user-managed file; leave it.
            typer.echo(
                f"Warning: couldn't parse {path} — configure the custom endpoint "
                "there yourself, or move the file aside and re-run.",
                err = True,
            )
            return
    # Hermes only reads the key for a *named* custom provider (a bare
    # `provider: custom` ignores it), so register it under providers.*.
    _subdict(config, "model").update(
        provider = f"custom:{_HERMES_PROVIDER}",
        default = model["id"],
        api_mode = "openai",
    )
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # Hermes auto-detects context from GET /v1/models, but OpenAI's schema has no
        # context field, so it can fall back to a 256k default that overflows a small
        # local model. Pin the real window (top-level model.context_length is the
        # highest-priority override) and compact at 90% of it (Hermes defaults to 50%).
        if window >= _HERMES_MIN_CONTEXT:
            _subdict(config, "model")["context_length"] = window
            _subdict(config, "compression").update(enabled = True, threshold = 0.9)
        else:
            # Below Hermes' 64,000-token floor it refuses to initialize, so claim
            # the floor and shrink the threshold so compaction still fires at 90%
            # of the REAL window (the threshold is a fraction of the claimed
            # context_length). The auxiliary override keeps the same floor check
            # from rejecting the compression model mid-session.
            _subdict(config, "model")["context_length"] = _HERMES_MIN_CONTEXT
            threshold = round(0.9 * window / _HERMES_MIN_CONTEXT, 4)
            _subdict(config, "compression").update(enabled = True, threshold = threshold)
            auxiliary = _subdict(_subdict(config, "auxiliary"), "compression")
            auxiliary["context_length"] = _HERMES_MIN_CONTEXT
    _subdict(config, "providers")[_HERMES_PROVIDER] = {
        "base_url": f"{base}/v1",
        "api_mode": "openai",
        "key_env": _HERMES_ENV_KEY,
    }
    text = yaml.safe_dump(config, sort_keys = False)
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text, encoding = "utf-8")
        typer.echo(f"Updated {path}")


def write_pi_config(base: str, key: str, model: dict, path: Path) -> None:
    config = _read_json_object(path)
    if config is None:
        typer.echo(
            f"Warning: couldn't parse {path} — add an 'unsloth' provider there "
            "yourself, or move the file aside and re-run.",
            err = True,
        )
        return
    before = json.dumps(config, sort_keys = True)
    # Pi reads custom providers from ~/.pi/agent/models.json (HOME-relocated for the
    # session). Unsloth is a generic OpenAI-compatible /v1 endpoint, and the key lives
    # in the config rather than the env (matching openclaw/opencode).
    provider_model = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # An unspecified model defaults to contextWindow 128000 / maxTokens 16384,
        # far larger than a small Unsloth context, so Pi compacts too late and overflows
        # the server. Pin the real window and a sane output cap (mirrors OpenCode).
        provider_model["contextWindow"] = window
        provider_model["maxTokens"] = min(window // 4, 8192)
    _subdict(config, "providers")[_PI_PROVIDER] = {
        "api": "openai-completions",
        "baseUrl": f"{base}/v1",
        "apiKey": key,
        "models": [provider_model],
    }
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


@start_app.command("claude", context_settings = _PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point Claude Code at the running Unsloth server and start it."""
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    model_id = entry["id"]

    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": model_id,
        # Session-only (no ~/.claude write): suppress the attribution header so
        # llama.cpp KV-cache reuse is preserved; --settings below reinforces it.
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        # Update checks, beta features, and other background requests either
        # stall against a local server or evict the conversation from
        # llama-server's KV-cache slots, so turn off everything nonessential.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        # A local server streams in bursts; disable the full-screen TUI redraw so the
        # terminal doesn't flicker between tokens.
        "CLAUDE_CODE_NO_FLICKER": "1",
    }
    # Claude Code auto-compacts against its native (~600k token) window; a local
    # model's context is usually far smaller, so size the window to the loaded
    # model's real context length. Otherwise the conversation overflows the
    # server's window (silent truncation) long before Claude decides to compact.
    # codex/openclaw get the same value through their config (model_context_window
    # / contextWindow); Claude has no config file, so it rides on the env var.
    window = entry.get("context_length") or entry.get("max_context_length")
    if window:
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(int(window))
        # Compact at 90% of that window; the override only takes effect once the
        # window is set, and it can only lower the threshold, so it just guarantees
        # headroom before the server's context limit instead of relying on Claude's
        # default (which is tuned for its native 200K/1M window).
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "90"
    # --yolo (or its aliases) maps to Claude's own --dangerously-skip-permissions.
    # IS_SANDBOX is left unset on purpose: Claude refuses bypass mode as root unless a
    # sandbox is detected, and we don't want to falsely claim one on the user's host.
    # claude keeps its history in ~/.claude/projects, which --settings/env never
    # relocate, so a session already survives exit; resume it with `claude --continue`
    # or `--resume <id>` passed through.
    command = [
        "claude",
        "--model",
        model_id,
        *_claude_flags(model_id),
        *_yolo_command_flags("claude", yolo),
        *ctx.args,
    ]
    install_hint = (
        "irm https://claude.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://claude.ai/install.sh | bash"
    )
    _run(
        base,
        entry,
        env,
        command,
        launch = launch,
        install_hint = install_hint,
        unset_env = _CLAUDE_ENV_UNSET,
    )


@start_app.command("codex", context_settings = _PASSTHROUGH)
def codex(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point OpenAI Codex at the running Unsloth server and start it."""
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    # This preflight runs after _connect may have auto-started a server but before _run
    # takes over its lifecycle, so tear the server down here if it rejects the model
    # (e.g. a transformers-backend model) rather than leaving it on the atexit backstop.
    try:
        _require_gguf_for_codex(base, key, entry["id"])
    except BaseException:
        _shutdown_auto_served()
        raise
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        with _session_config("codex-subagent", launch, persist = persist) as home:
            agent_config = write_codex_subagent_config(base, subagent_model, home)
            command = [
                "codex",
                *_codex_subagent_flags(agent_config),
                *_yolo_command_flags("codex", yolo),
                *ctx.args,
            ]
            typer.echo(
                "Unsloth is available as the `unsloth` local agent. "
                "Ask Codex to spawn an Unsloth or local agent."
            )
            _run(
                base,
                subagent_model,
                {_CODEX_ENV_KEY: key},
                command,
                launch = launch,
                install_hint = "npm install -g @openai/codex",
            )
        return
    command = [
        "codex",
        "--oss",
        "--profile",
        _CODEX_PROFILE,
        *_yolo_command_flags("codex", yolo),
        *ctx.args,
    ]
    with _session_config("codex", launch, persist = persist) as home:
        write_codex_config(base, entry, home)
        env = {_CODEX_ENV_KEY: key, "CODEX_HOME": str(home)}
        _run(base, entry, env, command, launch = launch, install_hint = "npm install -g @openai/codex")


@start_app.command("openclaw", context_settings = _PASSTHROUGH)
def openclaw(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point OpenClaw at the running Unsloth server and start it."""
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    openclaw_args = list(ctx.args)
    # Default a bare `unsloth start openclaw` to the local TUI. Anything the caller
    # passes through is forwarded verbatim so OpenClaw parses it under its own grammar
    # (openclaw [global-flags] <command> [options]): an explicit subcommand, a global
    # flag that must precede the command such as --profile/--dev, or a tui option. We
    # cannot reinterpret those safely because a leading "--flag value" is ambiguous
    # between a global (`--profile test`) and a tui option (`--message hi`); prepending
    # `tui --local` would break the global form, so only the empty case is defaulted.
    if not openclaw_args:
        openclaw_args = ["tui", "--local"]
    command = ["openclaw", *openclaw_args]
    install_hint = (
        "iwr -useb https://openclaw.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://openclaw.ai/install.sh | bash"
    )
    with _session_config("openclaw", launch, persist = persist) as cfg:
        config_path = cfg / "openclaw.json"
        workspace_path = None
        if _wsl_windows_executable(command):
            workspace_path = _wsl_windows_path(cfg / "workspace")
        # key lives in the config, not the env; --yolo writes the exec policy here too.
        write_openclaw_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo,
            workspace_path = workspace_path,
        )
        # Scope both config and state so OpenClaw never touches the user's ~/.openclaw.
        env = {"OPENCLAW_CONFIG_PATH": str(config_path), "OPENCLAW_STATE_DIR": str(cfg)}
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("opencode", context_settings = _PASSTHROUGH)
def opencode(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point OpenCode at the running Unsloth server and start it."""
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        route_native_auto = yolo and _opencode_supports_native_auto()
        opencode_args, native_auto = _opencode_native_auto_args(
            list(ctx.args), route_native_auto
        )
        command = ["opencode", *opencode_args]
        with _session_config("opencode-subagent", launch, persist = persist) as cfg:
            config_path = cfg / "opencode.json"
            session_permission = write_opencode_config(
                base,
                key,
                subagent_model,
                config_path,
                yolo = yolo and not native_auto,
                as_subagent = True,
            )
            env = {"OPENCODE_CONFIG": str(config_path)}
            if session_permission:
                env["OPENCODE_CONFIG_CONTENT"] = json.dumps(
                    {"permission": session_permission}
                )
            typer.echo("Unsloth is available as @unsloth and in /models.")
            _run(
                base,
                subagent_model,
                env,
                command,
                launch = launch,
                install_hint = "npm install -g opencode-ai",
            )
        return
    opencode_model = f"{_OPENCODE_PROVIDER}/{entry['id']}"
    # The inline OPENCODE_CONFIG_CONTENT below pins the model in the highest-priority
    # layer, so the session model is forced without a --model flag. Only add --model for
    # an interactive bare launch (a convenience so the TUI opens on our model). It is
    # omitted for passthrough (inserting it before a subcommand can be misparsed) and for
    # --no-launch, where the printed command is consumed by drivers that append a
    # subcommand such as `run <prompt>`; a leading --model would land before that
    # subcommand and break it. Those paths rely on the inline pin instead.
    native_auto = False
    route_native_auto = yolo and _opencode_supports_native_auto()
    if ctx.args:
        opencode_args, native_auto = _opencode_native_auto_args(list(ctx.args), route_native_auto)
        command = ["opencode", *opencode_args]
    elif launch:
        opencode_args, native_auto = _opencode_native_auto_args(
            ["--model", opencode_model],
            route_native_auto,
        )
        command = ["opencode", *opencode_args]
    else:
        # Append-safe base: `opencode --auto run ...` parses as the TUI with a project
        # "run", not the run subcommand. Command unknown here, so keep the config fallback.
        command = ["opencode"]
    # opencode keeps sessions in ~/.local/share/opencode (never relocated), so resume
    # already survives exit; reopen the last one by passing `opencode --continue` through.
    with _session_config("opencode", launch, persist = persist) as cfg:
        config_path = cfg / "opencode.json"
        # OPENCODE_CONFIG is an overlay (loaded between the user's global and project
        # configs), so this adds the Unsloth provider/model for the session without
        # changing the user's default model. Key lives in the config, not the env.
        session_permission = write_opencode_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo and not native_auto,
        )
        # A project's own opencode.json outranks OPENCODE_CONFIG, so the session model pin
        # would silently lose to a repo config. Carry it in OPENCODE_CONFIG_CONTENT, which
        # outranks project config; the API key stays in the private file, never the env.
        # Only the config fallback carries a permission. Native --auto omits it (auto-approve
        # asks, keep explicit denies); a non-yolo session omits it too, honoring project rules.
        # opencode filters every provider (a config-defined custom one included) through
        # its enabled_providers allowlist and disabled_providers denylist, and a model pin
        # does not bypass that gate -- a filtered provider resolves to ModelNotFoundError.
        # To guarantee the session model loads without reading or modifying the user's real
        # config, scope THIS session to our provider alone: allowlist _OPENCODE_PROVIDER and
        # clear the denylist. These arrays are replaced (not merged) by higher layers, so
        # setting them in the highest-priority inline overlay neutralizes any user allowlist
        # or denylist for the launch. It is session-only: it lives in OPENCODE_CONFIG_CONTENT
        # for this invocation and never touches the user's config files, so their normal
        # `opencode` is unchanged; only this session is limited to the Unsloth provider.
        # small_model is opencode's separate model for lightweight tasks; pin it to the
        # session model too, or a user/project small_model on another (now filtered)
        # provider would resolve a not-found error mid-session. The session serves one
        # model, so the session model is the only valid target here anyway.
        inline_config: dict = {
            "model": opencode_model,
            "small_model": opencode_model,
            "enabled_providers": [_OPENCODE_PROVIDER],
            "disabled_providers": [],
        }
        if session_permission:
            inline_config["permission"] = session_permission
        env = {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_CONFIG_CONTENT": json.dumps(inline_config),
        }
        _run(base, entry, env, command, launch = launch, install_hint = "npm install -g opencode-ai")


@start_app.command("hermes", context_settings = _PASSTHROUGH)
def hermes(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point Hermes (Nous Research) at the running Unsloth server and start it."""
    native_args = [*_yolo_command_flags("hermes", yolo), *ctx.args]
    command = ["hermes", *_hermes_resume_oneshot_args(native_args)]
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    install_hint = _hermes_install_hint()
    with _session_config("hermes", launch, persist = persist) as home:
        # HERMES_HOME relocates hermes' whole home dir (config.yaml, sessions, state)
        # like CODEX_HOME, so the user's ~/.hermes is left untouched for the session.
        write_hermes_config(base, entry, home / "config.yaml")
        env = {_HERMES_ENV_KEY: key, "HERMES_HOME": str(home)}
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("pi", context_settings = _PASSTHROUGH)
def pi(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point Pi (coding agent) at the running Unsloth server and start it."""
    base, key, entry = _connect(
        api_key,
        model,
        LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel),
        serve = serve,
        launch = launch,
    )
    install_hint = "npm install -g --ignore-scripts @earendil-works/pi-coding-agent"
    if as_subagent:
        if not _PI_SUBAGENT_EXTENSION.is_file():
            _fail(f"Missing Pi subagent extension: {_PI_SUBAGENT_EXTENSION}")
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        window = subagent_model.get("context_length") or subagent_model.get(
            "max_context_length"
        )
        extension = _agent_config_path(_PI_SUBAGENT_EXTENSION, ["pi"])
        command = [
            "pi",
            "--extension",
            extension,
            *_yolo_command_flags("pi", yolo),
            *ctx.args,
        ]
        env = {
            "UNSLOTH_PI_SUBAGENT_BASE_URL": f"{base}/v1",
            "UNSLOTH_PI_SUBAGENT_API_KEY": key,
            "UNSLOTH_PI_SUBAGENT_MODEL": subagent_id,
        }
        if window:
            window = int(window)
            env["UNSLOTH_PI_SUBAGENT_CONTEXT_WINDOW"] = str(window)
            env["UNSLOTH_PI_SUBAGENT_MAX_TOKENS"] = str(min(window // 4, 8192))
        typer.echo(
            "Unsloth is available as a local agent and in /model. "
            "Ask Pi to spawn an Unsloth or local agent."
        )
        _run(
            base,
            subagent_model,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            clear_screen = True,
        )
        return
    # Pi defaults to the google provider, so pin our provider/model on the command
    # line; the custom OpenAI-compatible endpoint itself is only configurable via
    # ~/.pi/agent/models.json.
    command = [
        "pi",
        "--provider",
        _PI_PROVIDER,
        "--model",
        entry["id"],
        *_yolo_command_flags("pi", yolo),
        *ctx.args,
    ]
    # --ignore-scripts matches Pi's documented install recipe (its README notes Pi needs
    # no install scripts), so accepting the prompt skips dependency lifecycle scripts.
    with _session_config("pi", launch, persist = persist) as home:
        # Pi resolves its config dir from PI_CODING_AGENT_DIR first (getAgentDir() prefers
        # it over $HOME/.pi/agent), so pin it at the session dir: an inherited
        # PI_CODING_AGENT_DIR in the user's shell would otherwise send Pi to their real
        # config and skip our provider/key. HOME is relocated too so any other ~/.pi paths
        # stay in the session. The key rides in the config rather than the env.
        pi_agent_dir = home / ".pi" / "agent"
        write_pi_config(base, key, entry, pi_agent_dir / "models.json")
        env = {"HOME": str(home), "PI_CODING_AGENT_DIR": str(pi_agent_dir)}
        if os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"):
            # Node resolves ~/.pi via USERPROFILE (then HOMEDRIVE + HOMEPATH) on Windows,
            # not HOME. Set them whenever Pi may run as a Windows process: native Windows,
            # or a /mnt Windows shim launched from WSL (the WSLENV bridge then translates
            # the path). Otherwise the Windows process falls back to the user's real
            # %USERPROFILE%\.pi. splitdrive yields no drive off a POSIX path, so
            # HOMEDRIVE/HOMEPATH stay unset there.
            env["USERPROFILE"] = str(home)
            drive, tail = os.path.splitdrive(str(home))
            if drive:
                env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
        # Pi paints inline from the current cursor position (no alternate screen,
        # no clear on first render), so give it the clean screen it assumes.
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            clear_screen = True,
        )
