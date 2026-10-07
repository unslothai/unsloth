# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` — launch a coding agent against a running Unsloth server."""

import atexit
import base64
import contextlib
import errno
import functools
import hashlib
import http.client
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
from typing import Literal, NamedTuple, NoReturn, Optional
from urllib.parse import urlencode, urlparse

import click
import typer
from typer.core import TyperCommand

from studio.backend.utils.coding_agents import (
    deepseek_harness_executables_on_path,
    is_deepseek_harness_executable,
)
from unsloth_cli._inference import (
    _USER_AGENT,
    _studio_token,
    ensure_studio_backend_path,
    find_studio_server,
    is_loopback_url,
    raise_for_deferred_error,
    require_completed_padded_body,
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
# Codex drops an SSE stream after 5 quiet minutes, and llama-server sends nothing during prompt
# processing (a CPU host can take ~460s on Codex's first turn). Each retry lands on a fresh slot and
# restarts from zero, so it never converges. 20 minutes outlasts a slow first turn.
_CODEX_STREAM_IDLE_TIMEOUT_MS = 1_200_000
_HERMES_ENV_KEY = "UNSLOTH_API_KEY"
_HERMES_PROVIDER = "unsloth"
# Skip the installer's interactive wizard: Unsloth writes its own session-scoped Hermes config.
# Pin the script and its checkout to one commit so neither branch can swap the code we run.
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
# Hermes refuses windows under 64,000 tokens; write_hermes_config claims this and scales compaction.
_HERMES_MIN_CONTEXT = 65536
_DSH_PROVIDER = "unsloth"
_DSH_ENV_KEY = "UNSLOTH_API_KEY"
_DSH_PATCH_FILE = "unsloth.patch.yml"
_DSH_PACKAGE = "@deepseek-ai/dsh"
# dsh reads DSH_PERMISSION_MODE via ??, so pin it both ways or a parent's danger mode leaks in.
_DSH_SAFE_PERMISSION_MODE = "workspace-write"
_DSH_YOLO_PERMISSION_MODE = "danger-full-access"
_PI_PROVIDER = "unsloth"
_SUBAGENT_NAME = "unsloth"
_SUBAGENT_DESCRIPTION = (
    "Local coding subagent powered by Unsloth for debugging, implementation, and codebase "
    "research. Use when the user asks to spawn an Unsloth or local agent."
)
_SUBAGENT_INSTRUCTIONS = (
    "You are a local coding subagent powered by Unsloth. Complete the assigned task directly, "
    "use the available tools when useful, verify your work, and return a concise result to the "
    "parent agent."
)
_SUBAGENT_PLAN_DESCRIPTION = (
    "Read-only local coding subagent powered by Unsloth for planning and codebase research. "
    "Use this local agent when Claude is in plan mode."
)
_SUBAGENT_PLAN_INSTRUCTIONS = (
    "You are a read-only local coding subagent powered by Unsloth. Investigate the assigned "
    "task with read-only tools, produce a concrete plan or answer, and return a concise result "
    "to the parent agent. Do not modify files."
)
_CLAUDE_SUBAGENT_MCP_MODULE = "unsloth_cli.claude_subagent_mcp"
_CLAUDE_SUBAGENT_SETTINGS_ENV = "UNSLOTH_CLAUDE_SUBAGENT_SETTINGS"
_CLAUDE_SUBAGENT_TOOL = "mcp__plugin_unsloth-local-agent_unsloth__unsloth_agent"
_CLAUDE_SUBAGENT_PLAN_TOOL = "mcp__plugin_unsloth-local-agent_unsloth__unsloth_plan_agent"
_CODEX_SUBAGENT_MCP_MODULE = "unsloth_cli.codex_subagent_mcp"
_CODEX_SUBAGENT_MCP_SERVER = "unsloth_local_agent"
_CODEX_SUBAGENT_MCP_TOOL = "spawn_local_agent"
_CODEX_SUBAGENT_CONFIG_ENV = "UNSLOTH_CODEX_SUBAGENT_CONFIG"
_CODEX_PARENT_OVERLAY_MANIFEST = ".unsloth-parent-overlay.json"
_CODEX_EPHEMERAL_STALE_SECONDS = 24 * 60 * 60
_CODEX_EPHEMERAL_HEARTBEAT_SECONDS = 60
_CODEX_SUBAGENT_TOOL_DESCRIPTION = (
    f"{_SUBAGENT_DESCRIPTION} Use this tool instead of the built-in spawn_agent tool for those "
    "requests. Other subagent requests may use the built-in tools normally."
)
_CODEX_SUBAGENT_ROUTING_INSTRUCTIONS = (
    "When the user asks to spawn an Unsloth agent or local agent, you must call the "
    "spawn_local_agent MCP tool once with the complete task. Do not answer, simulate the "
    "result, call wait, or use a built-in subagent before calling the tool. Use built-in "
    "subagents for other delegation requests."
)
_PI_SUBAGENT_EXTENSION = Path(__file__).parent.parent / "pi_subagent.ts"
_PI_USER_RESOURCE_DIRS = ("extensions", "skills", "prompts", "themes", "npm", "git")
_PI_USER_RESOURCE_SETTINGS = ("packages", "extensions", "skills", "prompts", "themes")
_PI_USER_VERBATIM_SETTINGS = ("npmCommand",)
_PI_USER_RESOURCES_MANIFEST = ".unsloth-user-resources.json"
# A dedicated provider id avoids colliding with a user's providers.
_OPENCODE_PROVIDER = "unsloth-studio"
# OpenCode sends min(limit.output, this) as max_tokens unless the env var below raises it.
_OPENCODE_OUTPUT_TOKEN_MAX = 32_000
_OPENCODE_OUTPUT_TOKEN_MAX_ENV = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"
_VIBE_PROVIDER = "unsloth-studio"
_VIBE_MODEL_ALIAS = "unsloth"
_VIBE_ENV_KEY = "UNSLOTH_API_KEY"
# Both installers put binaries in ~/.local/bin; Vibe's exits 1 and uv's leaves PATH stale without it.
_VIBE_POSIX_INSTALL_HINT = (
    'curl -LsSf https://mistral.ai/vibe/install.sh | PATH="$HOME/.local/bin:$PATH" bash'
)
_VIBE_WINDOWS_INSTALL_HINT = (
    "irm https://astral.sh/uv/install.ps1 | iex; "
    '$env:Path = "$HOME\\.local\\bin;$env:Path"; uv tool install mistral-vibe'
)
_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


class _PassthroughCommand(TyperCommand):
    """Preserve the option separator when forwarding arguments to an agent."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        raw_args = list(args)
        try:
            separator = raw_args.index("--")
        except ValueError:
            return super().parse_args(ctx, args)
        trailing_count = len(raw_args) - separator - 1
        remaining = super().parse_args(ctx, args)
        insert_at = max(0, len(remaining) - trailing_count)
        if insert_at >= len(remaining) or remaining[insert_at] != "--":
            remaining.insert(insert_at, "--")
            ctx.args = remaining
        return remaining


# provider routing overrides ANTHROPIC_BASE_URL and would bypass the local server (#9864).
_CLAUDE_ENV_UNSET = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CODE_USE_FOUNDRY",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
    "CLAUDE_CODE_USE_MANTLE",
)
_CODEX_ENV_UNSET = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")
# OpenClaw tries CODEX_API_KEY before OPENAI_API_KEY, and its exec tool inherits CODEX_ACCESS_TOKEN.
_OPENCLAW_ENV_UNSET = ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")

# Help is grouped into rich panels (Model / Server / Session).
_PANEL_MODEL = "Model"
_PANEL_SERVER = "Server"
_PANEL_SAMPLING = "Sampling"
_PANEL_SESSION = "Agent session"

_MODEL_OPTION = typer.Option(
    None,
    "--model",
    "-m",
    rich_help_panel = _PANEL_MODEL,
    help = "Model for the agent, or a bare `org/name(:variant)` positional. "
    "Defaults to the one loaded in Unsloth.",
)
_GGUF_VARIANT_OPTION = typer.Option(
    None,
    "--gguf-variant",
    rich_help_panel = _PANEL_MODEL,
    help = "GGUF quant variant to load (e.g. UD-Q4_K_XL). Defaults to UD-Q4_K_XL for "
    "unsloth/* GGUF repos, else Q4_K_M.",
)
_CONTEXT_OPTION = typer.Option(
    0,
    "--max-seq-length",
    "--context-length",
    rich_help_panel = _PANEL_MODEL,
    help = "Context length in tokens for the load (0 = model default).",
)
_LOAD_4BIT_OPTION = typer.Option(
    True,
    "--load-in-4bit/--no-load-in-4bit",
    rich_help_panel = _PANEL_MODEL,
    help = "Load hub models in 4-bit (ignored for GGUF).",
)
_TENSOR_PARALLEL_OPTION = typer.Option(
    False,
    "--tensor-parallel/--no-tensor-parallel",
    rich_help_panel = _PANEL_MODEL,
    help = "Split a GGUF across GPUs by tensor instead of by layer (multi-GPU only).",
)
_GPU_MEMORY_MODE_OPTION = typer.Option(
    None,
    "--gpu-memory-mode",
    rich_help_panel = _PANEL_MODEL,
    help = (
        "GPU memory strategy for GGUF models loaded by this command. Auto lets "
        "Unsloth manage placement. Manual with default layers and context delegates "
        "placement and sizing to llama.cpp --fit. Omit when attaching to preserve "
        "the running model's mode."
    ),
)

# Tool flags only configure a server `unsloth start` auto-starts (--serve).
_SERVE_OPTION = typer.Option(
    True,
    "--serve/--no-serve",
    rich_help_panel = _PANEL_SERVER,
    help = "If no Unsloth server is running, auto-start one for --model and keep it "
    "available after the agent exits. --no-serve errors out instead.",
)
_ENABLE_TOOLS_OPTION = typer.Option(
    None,
    "--enable-tools/--disable-tools",
    rich_help_panel = _PANEL_SERVER,
    help = "Server-side tools (web search, code execution) for the auto-started server. "
    "Default off so the agent's own tools are relayed unchanged.",
)
_TOOL_CALL_HEALING_OPTION = typer.Option(
    None,
    "--enable-tool-call-healing/--disable-tool-call-healing",
    rich_help_panel = _PANEL_SERVER,
    help = "Promote text-form tool calls from small GGUFs back into structured calls. On by "
    "default; when the flag is omitted an inherited UNSLOTH_DISABLE_TOOL_CALL_HEALING is kept.",
)
_TOOL_CALL_NUDGING_OPTION = typer.Option(
    None,
    "--enable-tool-call-nudging/--disable-tool-call-nudging",
    rich_help_panel = _PANEL_SERVER,
    help = "Retry once with a nudge when a non-streaming passthrough tool call can't be healed. "
    "On by default; when the flag is omitted an inherited UNSLOTH_TOOL_CALL_NUDGE is kept.",
)
_REASONING_OPTION = typer.Option(
    None,
    "--reasoning",
    rich_help_panel = _PANEL_SERVER,
    help = (
        "Reasoning mode for this agent session. Defaults to auto so the model's chat "
        "template decides; use 'on' or 'off' to override it."
    ),
)
_REASONING_EFFORT_OPTION = typer.Option(
    None,
    "--reasoning-effort",
    rich_help_panel = _PANEL_SERVER,
    help = (
        "Reasoning effort for this agent session, e.g. 'medium'. The "
        "levels are the model's own, so pass one its chat template accepts. Default: "
        "unset, which keeps the template's level."
    ),
)
# Sampling rides in the agent's own config; one the agent cannot send pins the auto-started server.
_TEMPERATURE_OPTION = typer.Option(
    None,
    "--temperature",
    min = 0.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the sampling temperature. Default: unset (per-model recommendation).",
)
_TOP_P_OPTION = typer.Option(
    None,
    "--top-p",
    min = 0.0,
    max = 1.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin top-p (nucleus) sampling. Default: unset (per-model recommendation).",
)
_TOP_K_OPTION = typer.Option(
    None,
    "--top-k",
    min = -1,
    max = 100,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin top-k sampling. Default: unset (per-model recommendation).",
)
_MIN_P_OPTION = typer.Option(
    None,
    "--min-p",
    min = 0.0,
    max = 1.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin min-p sampling threshold. Default: unset (per-model recommendation).",
)
_REPETITION_PENALTY_OPTION = typer.Option(
    None,
    "--repetition-penalty",
    min = 1.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the repetition penalty. Default: unset (per-model recommendation).",
)
_PRESENCE_PENALTY_OPTION = typer.Option(
    None,
    "--presence-penalty",
    min = 0.0,
    max = 2.0,
    rich_help_panel = _PANEL_SAMPLING,
    help = "Pin the presence penalty. Default: unset (per-model recommendation).",
)
_MAX_TOKENS_OPTION = typer.Option(
    None,
    "--max-tokens",
    min = 1,
    rich_help_panel = _PANEL_SAMPLING,
    help = (
        "Most tokens the agent may generate in one response. Default: a quarter of the "
        "context window, up to 32,000. Capped at half the window so the conversation "
        "keeps room."
    ),
)

_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar = "UNSLOTH_API_KEY",
    rich_help_panel = _PANEL_SESSION,
    help = "Unsloth API key. For a local Unsloth it is minted automatically and "
    "remembered per server. For a remote server, pass one with --api-key "
    "(or UNSLOTH_API_KEY); it is remembered for next time.",
)
_LAUNCH_OPTION = typer.Option(
    True,
    "--launch/--no-launch",
    rich_help_panel = _PANEL_SESSION,
    help = "--no-launch prints the env and command instead (remote shells, WSL).",
)
# Accept every agent's spelling of "run tools without prompting" and route to its own mechanism.
_YOLO_OPTION = typer.Option(
    False,
    "--yolo",
    "--dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    rich_help_panel = _PANEL_SESSION,
    help = "Auto-approve all tool actions for this session; routed to the agent's own "
    "flag/config. Any of the three spellings works for any agent.",
)
_PERSIST_OPTION = typer.Option(
    False,
    "--persist/--no-persist",
    rich_help_panel = _PANEL_SESSION,
    help = (
        "Keep this agent's Unsloth-managed session dir so you can resume it later. "
        "codex/openclaw/hermes/pi/dsh have their whole home relocated into an Unsloth dir "
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
    rich_help_panel = _PANEL_SESSION,
    help = "Keep the coding agent's current model and add Unsloth as a local subagent.",
)
_APP_OPTION = typer.Option(
    False,
    "--app",
    rich_help_panel = _PANEL_SESSION,
    help = (
        "Add Unsloth to the agent's desktop app instead of starting the CLI: writes the "
        "provider and its own API key into your config (backed up first) and leaves your "
        "default model alone, so you pick Unsloth in the app's model picker. Re-run after "
        "loading another model."
    ),
)
_CODEX_APP_OPTION = typer.Option(
    False,
    "--app",
    rich_help_panel = _PANEL_SESSION,
    help = (
        "Switch the Codex desktop app to Unsloth instead of starting the CLI, open it, and "
        "switch it back when this command exits or Unsloth stops (the app only offers one "
        "provider's models at a time)."
    ),
)

# OpenCode (command-scoped --auto, below) and OpenClaw (config-only) are absent here.
_YOLO_COMMAND_FLAGS = {
    "claude": ["--dangerously-skip-permissions"],
    "codex": ["--dangerously-bypass-approvals-and-sandbox"],
    "hermes": ["--yolo"],
    # Pi's only approval gate is project trust, so -a is the closest equivalent.
    "pi": ["--approve"],
    "vibe": ["--auto-approve"],
}


def _yolo_command_flags(agent: str, yolo: bool) -> list:
    return _YOLO_COMMAND_FLAGS.get(agent, []) if yolo else []


# Subcommands that reject --auto, including console/generate (hidden but registered). Unknown first
# positionals are TUI paths and get --auto.
_OPENCODE_NON_AUTO_SUBCOMMANDS = frozenset(
    "completion acp mcp attach debug providers auth agent upgrade uninstall serve web "
    "models stats export import github pr session plugin plug db console generate".split()
)
_OPENCODE_V2_SUBCOMMANDS = frozenset(
    "acp api debug console auth mcp plugin models export import mini run service pair serve".split()
)
_OPENCODE_V2_STANDALONE_SUBCOMMANDS = frozenset("api models export import mini run".split())
_OPENCODE_GLOBAL_BOOLEAN_OPTIONS = frozenset(
    "-h --help -v --version --print-logs --pure --mdns --standalone --wizard".split()
)
_OPENCODE_GLOBAL_VALUE_OPTIONS = frozenset(
    "--log-level --port --hostname --mdns-domain --cors --server --completions --cpu-profile".split()
)
_OPENCODE_NATIVE_AUTO_MIN_VERSION = (1, 17, 12)


def _opencode_command() -> tuple[str, bool]:
    resolved_v2 = _which_with_install_dirs("opencode2")
    if resolved_v2:
        return resolved_v2, True
    return "opencode", False


def _opencode_supports_native_auto(command: str = "opencode") -> bool:
    if Path(command).stem.lower() == "opencode2":
        return True
    executable = _which_with_install_dirs(command)
    if executable is None:
        # No local binary: assume a current release (a --no-launch recipe may run elsewhere).
        return True
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return False
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return bool(match) and tuple(int(part) for part in match.groups()) >= (
        _OPENCODE_NATIVE_AUTO_MIN_VERSION
    )


def _opencode_subcommand(args: list[str]) -> tuple[Optional[str], Optional[int]]:
    """Return an explicit OpenCode subcommand and its index after global options."""
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return None, None
        if arg in _OPENCODE_GLOBAL_BOOLEAN_OPTIONS:
            index += 1
            continue
        if arg in _OPENCODE_GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in _OPENCODE_GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        # A non-global option is a TUI flag; stop before its value is mistaken for a subcommand.
        if arg.startswith("-"):
            return None, None
        return arg, index
    return None, None


def _opencode_native_auto_args(
    args: list[str],
    yolo: bool,
    *,
    v2: bool = False,
) -> tuple[list[str], bool]:
    """Add OpenCode's native --auto when the selected command supports it."""
    routed = list(args)
    if not yolo:
        return routed, False
    subcommand, _ = _opencode_subcommand(routed)
    if v2 and subcommand in _OPENCODE_V2_SUBCOMMANDS and subcommand != "run":
        return routed, False
    if not v2 and subcommand in _OPENCODE_NON_AUTO_SUBCOMMANDS:
        return routed, False
    separator = routed.index("--") if "--" in routed else len(routed)
    # --mini forces auto=false and never forwards --auto: use the config permission block instead.
    if any(arg == "--mini" or arg.startswith("--mini=") for arg in routed[:separator]):
        return routed, False
    if "--auto" not in routed[:separator]:
        routed.insert(separator, "--auto")
    return routed, True


def _opencode_v2_standalone_args(args: list[str]) -> list[str]:
    """Keep session-only config on the V2 server that consumes it."""
    routed = list(args)
    separator = routed.index("--") if "--" in routed else len(routed)
    head = routed[:separator]
    if (
        "--standalone" in head
        or "--server" in head
        or any(arg.startswith("--server=") for arg in head)
    ):
        return routed
    subcommand, subcommand_index = _opencode_subcommand(routed)
    if (
        subcommand in _OPENCODE_V2_SUBCOMMANDS
        and subcommand not in _OPENCODE_V2_STANDALONE_SUBCOMMANDS
    ):
        return routed
    insert_at = (
        subcommand_index + 1 if subcommand in _OPENCODE_V2_STANDALONE_SUBCOMMANDS else separator
    )
    routed.insert(insert_at, "--standalone")
    return routed


def _hermes_install_hint() -> str:
    return _HERMES_WINDOWS_INSTALL_HINT if os.name == "nt" else _HERMES_POSIX_INSTALL_HINT


def _vibe_install_hint() -> str:
    return _VIBE_WINDOWS_INSTALL_HINT if os.name == "nt" else _VIBE_POSIX_INSTALL_HINT


def _npm_install_hint(package: str, *, ignore_scripts: bool = False) -> str:
    parts = ["npm", "install", "-g"]
    if os.name != "nt":
        # No home (bare container UID): fall back to npm's own prefix.
        try:
            parts.extend(("--prefix", str(Path.home() / ".local")))
        except (RuntimeError, OSError):
            pass
    if ignore_scripts:
        parts.append("--ignore-scripts")
    parts.append(package)
    if os.name == "nt":
        return " ".join(_powershell_quote(part) for part in parts)
    return shlex.join(parts)


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
            # argparse accepts attached short-option values; preserve the value byte-for-byte.
            rewritten[index] = f"-q{arg[2:]}"
        elif arg.startswith("--oneshot="):
            rewritten[index] = f"--query={arg.partition('=')[2]}"
        else:
            continue
        if any(item == "--usage-file" or item.startswith("--usage-file=") for item in args):
            raise typer.BadParameter(
                "Hermes cannot resume a one-shot session with --usage-file; remove that option."
            )
        # `-z` sets HERMES_YOLO_MODE / HERMES_ACCEPT_HOOKS for a one-shot; re-add them so a resumed `-z`
        # does not stall on an approval nobody can answer.
        prefix = ["chat", "-Q"]
        if "--yolo" not in rewritten:
            prefix.append("--yolo")
        if "--accept-hooks" not in rewritten:
            prefix.append("--accept-hooks")
        rewritten = prefix + rewritten
        return rewritten
    return args


_DSH_LAUNCHER_ARGS = frozenset(
    "--profile --patch --dump-config --dump-default-config -V --version plugin web".split()
)


_DSH_NO_PROFILE_ARGS = frozenset("-V --version plugin".split())


def _dsh_command(args: list[str], patch: Optional[str] = None) -> list[str]:
    head = args[0] if args else ""
    if head in _DSH_LAUNCHER_ARGS or head.startswith(("--profile=", "--patch=")):
        command = ["dsh", *args]
    else:
        command = ["dsh", "web", *args]
    if patch is not None and command[1] not in _DSH_NO_PROFILE_ARGS:
        # `dsh <name>` expands to `--profile <name>` only when the name comes first.
        at = 1 if command[1].startswith("-") else 2
        command[at:at] = ["--patch", patch]
    return command


class LoadOptions(NamedTuple):
    """Model-load knobs forwarded to /api/inference/load when --model triggers a load."""

    gguf_variant: Optional[str] = None
    max_seq_length: int = 0
    load_in_4bit: bool = True
    tensor_parallel: bool = False
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = None
    # Names the user actually typed: --context-length 0 equals the default yet must be sent. Appended
    # last to keep positional callers working.
    supplied: frozenset = frozenset()

    def overrides(self) -> frozenset:
        """Fields that must reach the load: typed explicitly, or differing from default."""
        differing = {
            name
            for name, default in (
                ("gguf_variant", None),
                ("max_seq_length", 0),
                ("load_in_4bit", True),
                ("tensor_parallel", False),
                ("gpu_memory_mode", None),
            )
            if getattr(self, name) != default
        }
        return frozenset(differing) | frozenset(self.supplied)


_LOAD_OPTION_PARAMS = (
    "gguf_variant",
    "max_seq_length",
    "load_in_4bit",
    "tensor_parallel",
    "gpu_memory_mode",
)


def _supplied_load_params(ctx) -> frozenset:
    """Which load knobs Click saw on the command line. The context must be PASSED IN: Typer invokes callbacks with no active click context, so click.get_current_context() is None. Unaskable gives an empty set, and `overrides()` falls back to comparing values."""
    getter = getattr(ctx, "get_parameter_source", None)
    if getter is None:
        return frozenset()
    supplied = set()
    for name in _LOAD_OPTION_PARAMS:
        try:
            source = getter(name)
        except Exception:
            continue
        # By member NAME: Typer vendors its own click, and click 8.3 reordered the IntEnum.
        if getattr(source, "name", None) == "COMMANDLINE":
            supplied.add(name)
    return frozenset(supplied)


def _load_options(
    ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
) -> LoadOptions:
    """Build LoadOptions for an agent command, recording what was typed."""
    return LoadOptions(
        gguf_variant,
        max_seq_length,
        load_in_4bit,
        tensor_parallel,
        gpu_memory_mode,
        _supplied_load_params(ctx),
    )


class ServerOptions(NamedTuple):
    """Start flags: carried fields ride in the agent's requests; the rest configure an auto-started server."""

    enable_tools: Optional[bool] = None
    tool_call_healing: Optional[bool] = None
    tool_call_nudging: Optional[bool] = None
    reasoning: Optional[Literal["on", "off", "auto"]] = None
    reasoning_effort: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    carried: frozenset = frozenset()
    # An inherited UNSLOTH_SAMPLING_* pin would override these.
    unpinned: frozenset = frozenset()

    def sent_by_agent(self) -> frozenset:
        # A request cannot ask for the template default back, so auto stays a server setting.
        unsent = {"reasoning"} if self.reasoning == "auto" else set()
        if self.reasoning_effort not in _STUDIO_REASONING_EFFORTS:
            unsent.add("reasoning_effort")
        return self.carried - unsent

    def request_body(self) -> dict:
        sent = self.sent_by_agent()
        body = {
            name: getattr(self, name)
            for name in _SAMPLING_FIELDS
            if name in sent and getattr(self, name) is not None
        }
        if "reasoning" in sent and self.reasoning in ("on", "off"):
            body["enable_thinking"] = self.reasoning == "on"
        if "reasoning_effort" in sent:
            body["reasoning_effort"] = self.reasoning_effort
        return body


_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
)
_REASONING_FIELDS = frozenset({"reasoning", "reasoning_effort"})
_ALL_REQUEST_FIELDS = frozenset(_SAMPLING_FIELDS) | _REASONING_FIELDS
_REQUEST_BODY_KEYS = (*_SAMPLING_FIELDS, "enable_thinking", "reasoning_effort")
_STUDIO_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "max", "xhigh")
_CODEX_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def _codex_reasoning_effort(
    reasoning: Optional[str], reasoning_effort: Optional[str]
) -> Optional[str]:
    if reasoning == "off":
        return "none"
    if reasoning_effort in _CODEX_REASONING_EFFORTS:
        return reasoning_effort
    return None


def _merge_request_body(container: dict, key: str, body: dict) -> None:
    current = container.get(key)
    current = (
        {k: v for k, v in current.items() if k not in _REQUEST_BODY_KEYS}
        if isinstance(current, dict)
        else {}
    )
    current.update(body)
    if current:
        container[key] = current
    else:
        container.pop(key, None)


def _split_repo_variant(model: str) -> tuple:
    """Split ``org/name:QUANT`` into ``("org/name", "QUANT")``. ``unsloth run`` and llama.cpp accept ``--model org/name:QUANT`` as shorthand for ``--model org/name --gguf-variant QUANT``, so mirror that here and a ``:variant`` suffix resolves against the already-loaded ``org/name`` (which the loaded listing shows without the suffix) instead of trying to load a repo id containing ``:``, which Hugging Face rejects and which would evict a model another session is using. Local paths, Windows drive letters and ids without a ``:`` pass through unchanged."""
    s = (model or "").strip()
    if not s or s.startswith(("/", "./", "../", "~")) or s == ".":
        return s, None
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return s, None
    if ":" not in s:
        return s, None
    repo, _, variant = s.rpartition(":")
    if not repo or not variant or "/" in variant:
        return s, None
    return repo, variant


def _looks_like_model(token: str) -> bool:
    """True for a bare `org/name(:variant)` hub id that is not a flag or a local path. Reuses `_is_hub_model_id`, so a relative dir like `owner/repo` that actually exists is left for the agent instead of being taken as a model; a non-existent `org/name` is treated as a hub id."""
    if not token or token.startswith("-") or " " in token:
        return False
    repo, _ = _split_repo_variant(token)
    return _is_hub_model_id(repo)


def _consume_positional_model(model: Optional[str], args: list) -> tuple:
    """Route a leading `org/name` positional to --model when --model was not given. Only the FIRST token is considered so an option value like `--profile owner/repo` is never stolen, and only when --model is absent so an explicit --model always wins. Returns (model, remaining_args) with the consumed token removed from the passthrough."""
    args = list(args)
    if model or not args or not _looks_like_model(args[0]):
        return model, args
    return args[0], args[1:]


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
    """Return an API model id that preserves the selected GGUF variant. Coding-agent model definitions outlive the initial load, so if Unsloth later unloads the model a bare repository id may resolve to a different cached quant; include the explicit or currently loaded variant so an automatic reload selects the same weights."""
    model_id = str(entry["id"])
    _, inline_variant = _split_repo_variant(requested_model or "")
    variant = requested_variant or inline_variant
    if not variant:
        try:
            status = _http_json("GET", f"{base}/api/inference/status", key)
        except Exception:
            status = {}
            typer.echo(
                "Warning: could not verify the loaded GGUF variant; a later reload "
                "may pick a different cached quant. Pass :variant to pin it.",
                err = True,
            )
        if status.get("is_gguf"):
            variant = status.get("gguf_variant")
    if variant and _is_hub_model_id(model_id):
        return _display_model_spec(model_id, str(variant))
    if variant:
        # A path load is advertised as a bare basename, so the quant cannot be recorded.
        typer.echo(
            f"Warning: {model_id} loaded from a path, so the subagent config cannot "
            f"pin the {variant} quant; a reload may choose a different one. Load the "
            "model by repository id to pin it.",
            err = True,
        )
    return model_id


def _fail(message: str) -> NoReturn:
    typer.echo(message, err = True)
    raise typer.Exit(code = 1)


def _reject_as_subagent(agent: str, args: list) -> None:
    # Reject early, or the flag reaches the agent binary after the model loaded.
    if any(arg == "--as-subagent" or arg.startswith("--as-subagent=") for arg in args):
        _fail(f"--as-subagent is not supported for {agent}.")


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode())
        return body.get("detail") or body["error"]["message"]
    except Exception:
        return str(exc)


def _fail_request(exc: Exception, error: str) -> NoReturn:
    """Fail with `error` plus whatever the server or the transport gave as a reason."""
    if isinstance(exc, urllib.error.HTTPError):
        _fail(f"{error}: {_http_error_detail(exc)}")
    _fail(f"{error}: {getattr(exc, 'reason', None) or exc}")


def _http_json(
    method: str,
    url: str,
    token: str,
    payload = None,
    timeout = 30,
    error = None,
):
    """On a failed request: raise if `error` is None, else fail with `error` plus the reason."""
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
            body = json.loads(response.read().decode() or "{}")
        # A padded /load or /unload commits its 200 early; raise a late in-band failure as HTTPError.
        return raise_for_deferred_error(url, body)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        if error is None:
            raise
        _fail_request(exc, error)


# Only a server WE auto-started, at module scope so failure paths and atexit can tear it down.
_auto_served_server: Optional[subprocess.Popen] = None
# Caps time since the download last advanced, not total elapsed.
_SERVER_START_TIMEOUT_S = 900
_DOWNLOAD_POLL_INTERVAL_S = 1.0
_DOWNLOAD_POLL_MAX_BACKOFF_S = 60.0
# Each listed repo costs a cache scan per poll.
_LOAD_DOWNLOAD_LIST_INTERVAL_S = 5.0
_START_API_KEY_PREFIX = "UNSLOTH_START_API_KEY: "
_START_PORT_PREFIX = "UNSLOTH_START_PORT: "
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
        self._source: Optional[str] = None
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def update(
        self,
        progress: dict,
        source: Optional[str] = None,
    ) -> None:
        if source != self._source:
            # A new repo is a new transfer; else redirected output stays silent for a later base download.
            self._source = source
            self._samples.clear()
            self._last_expected = 0
            # Keep `_shown`: `complete()` and `close()` gate on it.
            self._last_bucket = -1
        downloaded = max(0, int(progress.get("downloaded_bytes") or 0))
        completed = max(0, int(progress.get("completed_bytes") or 0))
        expected = max(0, int(progress.get("expected_bytes") or 0))
        self._last_expected = max(self._last_expected, expected)
        fraction = float(progress.get("progress") or 0)
        if downloaded <= 0:
            return
        # A fully cached snapshot can report 99% with no incomplete bytes; not a download.
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
            # The endpoint caps at 99% while bytes remain in an incomplete file; trust it.
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
        self._downloaded_bytes = 0
        self._failures = 0
        self._retry_at = 0.0
        self._display = _DownloadProgressDisplay()
        self._configured = False
        # A local model has no Hub reading, but a remote base it names is still listed.
        self._disabled = not _is_hub_model_id(model)
        self._progress_prefix = "/api/hub"
        self._repo_bytes: dict[str, int] = {}
        self._companions: list[str] = []
        self._finished: dict[str, dict] = {}
        self._listed_at: Optional[float] = None
        self._active_repo = model

    def _is_gguf(self) -> bool:
        return bool(self._variant) or "gguf" in self._model.lower()

    def _configure(self) -> None:
        self._configured = True
        if self._disabled:
            return
        # The repo endpoint totals every quant; resolve the variant first, else show bytes only.
        if self._is_gguf():
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
                # Older servers lack this endpoint.
                pass

    def _companion_repos(self) -> list[str]:
        now = time.monotonic()
        if self._listed_at is not None and now - self._listed_at < _LOAD_DOWNLOAD_LIST_INTERVAL_S:
            return self._companions
        self._listed_at = now
        try:
            url = f"{self._base}{self._progress_prefix}/active-downloads"
            listing = _http_json("GET", url, self._key, timeout = 10)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._listed_at = float("inf")
            return self._companions
        except Exception:
            return self._companions
        for item in listing.get("downloads") or []:
            repo = str(item.get("repo_id") or "")
            if not (item.get("owner") == "load" or item.get("load_attached")):
                continue
            if repo and repo.lower() != self._model.lower() and repo not in self._companions:
                self._companions.append(repo)
        return self._companions

    def _read(
        self,
        repo: str,
        gguf: bool = False,
    ) -> dict:
        if gguf:
            params = urlencode(
                {"repo_id": repo, "variant": self._variant, "expected_bytes": self._expected_bytes}
            )
            url = f"{self._base}{self._progress_prefix}/gguf-download-progress?{params}"
        else:
            params = urlencode({"repo_id": repo})
            url = f"{self._base}{self._progress_prefix}/download-progress?{params}"
        return _http_json("GET", url, self._key, timeout = 10)

    def poll(self) -> None:
        if not self._configured:
            self._configure()
        if time.monotonic() < self._retry_at:
            return
        try:
            readings = {}
            try:
                if not self._disabled:
                    readings[self._model] = self._read(self._model, gguf = self._is_gguf())
            except urllib.error.HTTPError as exc:
                if exc.code != 404 or self._progress_prefix == "/api/models":
                    raise
                self._progress_prefix = "/api/models"
                self.poll()
                return
            for repo in self._companion_repos():
                if repo in self._finished:
                    readings[repo] = self._finished[repo]
                    continue
                try:
                    readings[repo] = item = self._read(repo)
                except Exception:
                    continue
                if (
                    0
                    < int(item.get("expected_bytes") or 0)
                    <= int(item.get("completed_bytes") or 0)
                ):
                    self._finished[repo] = item
            # Follow whichever repo grew most; a repo first seen already large has not grown.
            growth = 0
            for repo, item in readings.items():
                current = max(0, int(item.get("downloaded_bytes") or 0))
                if current - self._repo_bytes.get(repo, current) > growth:
                    growth, self._active_repo = current - self._repo_bytes[repo], repo
                self._repo_bytes[repo] = max(self._repo_bytes.get(repo, 0), current)
            self._downloaded_bytes = max(self._downloaded_bytes, sum(self._repo_bytes.values()))
            self._failures = 0
            self._retry_at = 0.0
            active = self._active_repo if self._active_repo in readings else self._model
            if active in readings:
                self._display.update(readings[active], active)
        except Exception:
            # `_start_studio_server` reads `downloaded_bytes` for liveness, so back off but never stop probing.
            self._failures += 1
            self._retry_at = time.monotonic() + min(
                2.0**self._failures, _DOWNLOAD_POLL_MAX_BACKOFF_S
            )

    @property
    def downloaded_bytes(self) -> int:
        return self._downloaded_bytes

    def close(self) -> None:
        self._display.close()

    def complete(self) -> None:
        self._display.complete()


def _load_model_with_progress(
    base: str, key: str, model: str, load: LoadOptions, payload: dict
) -> dict:
    """Run the blocking load request while polling its download progress."""
    load_url = f"{base}/api/inference/load"
    result: list[tuple[bool, object]] = []
    done = threading.Event()

    def _load() -> None:
        try:
            value = _http_json(
                "POST",
                load_url,
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
            # A pad-only or half-written body fails json.loads: report the incomplete padded 200.
            if isinstance(value, ValueError):
                require_completed_padded_body(load_url, None)
            raise value
        progress.complete()
        # `_http_json` decodes a blank body as `{}`, which would look like a completed load.
        return require_completed_padded_body(load_url, value)
    finally:
        progress.close()


def _studio_healthy(base: str, timeout: float = 3.0) -> bool:
    request = urllib.request.Request(f"{base}/api/health", headers = {"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            return json.loads(response.read(65536).decode() or "{}").get("status") == "healthy"
    except Exception:
        return False


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding = "utf-8", errors = "replace")
    except OSError:
        return "(no server log)"


def _log_tail(path: Path, lines: int = 20) -> str:
    return "\n".join(_read_log(path).splitlines()[-lines:])


def _redacted_log_tail(path: Path, lines: int = 20) -> str:
    """Tail with minted keys removed; only for tails shown on the terminal."""
    return re.sub(r"sk-unsloth-\S+", "sk-unsloth-[redacted]", _log_tail(path, lines))


def _shutdown_server(server: Optional[subprocess.Popen]) -> None:
    # Idempotent teardown of a server WE started, plus its children (llama-server, cloudflared).
    if server is None or server.poll() is not None:
        return
    if os.name == "nt":
        # taskkill /T walks the whole tree so llama-server does not keep the port and GPU.
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


def _start_studio_server(
    base: str,
    model: str,
    load: LoadOptions,
    server: ServerOptions = ServerOptions(),
) -> tuple:
    """Spawn `unsloth run` for `model`, wait until it is fully ready, and return (base, server)."""
    global _auto_served_server
    # Windows: run via this interpreter, since which() resolves the policy-denied unsloth.exe (#8490).
    if sys.platform == "win32":
        # Local import: a top-level import would be circular.
        from unsloth_cli.commands.studio import _managed_cli_argv
        launch_head = _managed_cli_argv(Path(sys.executable))
    else:
        launch_head = [shutil.which("unsloth") or "unsloth"]
    parsed = urlparse(base)
    # Mirrors .github/scripts/serve-unsloth-run.sh. Healing/nudging travel via env, not new flags an
    # older re-exec'd run would pass to llama-server.
    command = [
        *launch_head,
        "run",
        "-H",
        parsed.hostname or "127.0.0.1",
        "-p",
        str(parsed.port or 8888),
        "--enable-tools" if server.enable_tools else "--disable-tools",
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
    if load.gpu_memory_mode is not None:
        command += ["--gpu-memory-mode", load.gpu_memory_mode]

    log_path = Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log"
    typer.echo("Starting Unsloth server")
    typer.echo(f"Model: {_display_model_spec(model, load.gguf_variant)}")
    typer.echo(f"Server log: {log_path}")
    # 0600: the log carries the minted sk-unsloth- key. Unlink first so a stale looser file cannot survive.
    log_path.unlink(missing_ok = True)
    log = os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
    # Own process group so a mid-session Ctrl+C does not reach the server.
    child_env = os.environ.copy()
    # Env, not a CLI flag: older managed llama-server ignores an unknown env var.
    child_env["LLAMA_ARG_REASONING"] = server.reasoning or "auto"
    # Always written, or an inherited value pins a level; 'default' is llama.cpp's own sentinel.
    child_env["LLAMA_ARG_REASONING_EFFORT"] = server.reasoning_effort or "default"
    # Env, not a flag, so an older launcher ignores it.
    child_env[_START_API_KEY_MARKER_ENV] = "1"
    # Env so it survives a re-exec into an older venv; only written when the flag was set explicitly.
    if server.tool_call_healing is not None:
        child_env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] = "0" if server.tool_call_healing else "1"
    elif "UNSLOTH_DISABLE_TOOL_CALL_HEALING" not in child_env:
        child_env["UNSLOTH_DISABLE_TOOL_CALL_HEALING"] = "0"
    if server.tool_call_nudging is not None:
        child_env["UNSLOTH_TOOL_CALL_NUDGE"] = "1" if server.tool_call_nudging else "0"
    elif "UNSLOTH_TOOL_CALL_NUDGE" not in child_env:
        child_env["UNSLOTH_TOOL_CALL_NUDGE"] = "1"
    # `unsloth run` reads UNSLOTH_SAMPLING_* as a hard override; only set fields the operator gave.
    for _sampling_name in _SAMPLING_FIELDS:
        _sampling_env = f"UNSLOTH_SAMPLING_{_sampling_name.upper()}"
        _sampling_value = getattr(server, _sampling_name)
        if _sampling_value is not None:
            child_env[_sampling_env] = str(_sampling_value)
        elif _sampling_name in server.unpinned:
            child_env.pop(_sampling_env, None)
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
        log.close()
    _auto_served_server = server
    atexit.register(_shutdown_auto_served)

    deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
    progress: Optional[_ModelDownloadProgress] = None
    downloaded_bytes = 0
    early_key_seen = False
    port_followed = False
    try:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                # The early key marker lands here before load finishes; redact it.
                tail = _redacted_log_tail(log_path)
                _shutdown_auto_served()
                _fail(f"The Unsloth server stopped before it was ready. Last log lines:\n{tail}")
            tail = _log_tail(log_path, lines = 400)
            # `unsloth run` falls forward off a taken port; it prints the port once, so read the whole log.
            if not port_followed:
                bound_port = re.search(
                    rf"^{re.escape(_START_PORT_PREFIX)}(\d+)$",
                    _read_log(log_path),
                    flags = re.MULTILINE,
                )
                if bound_port:
                    port_followed = True
                    base = _effective_base(base, int(bound_port.group(1)))
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
            # The cap measures time since bytes last arrived; log growth does not count (health polls and
            # heartbeats keep it growing while nothing loads).
            bytes_now = progress.downloaded_bytes if progress is not None else 0
            if bytes_now > downloaded_bytes:
                deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
            downloaded_bytes = bytes_now
            # New children emit an early key marker; older ones print the key only after load.
            ready_signal = "Model loaded:" in tail if early_key_seen else "sk-unsloth-" in tail
            if _studio_healthy(base) and ready_signal:
                if progress is not None:
                    progress.complete()
                    progress.close()
                    progress = None
                return base, server
            time.sleep(2.0)
    finally:
        if progress is not None:
            progress.close()
    _shutdown_auto_served()
    _fail(
        "The Unsloth server didn't become ready and made no progress for "
        f"{_SERVER_START_TIMEOUT_S}s. See {log_path}."
    )


def _effective_base(base: str, port: Optional[int] = None) -> str:
    # `unsloth run` binds `parsed.port or 8888` at the root: normalize to scheme://host:port.
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    if ":" in host:  # bare IPv6 literal (urlparse strips the brackets)
        host = f"[{host}]"
    return f"{parsed.scheme or 'http'}://{host}:{port or parsed.port or 8888}"


def _require_studio(
    model: Optional[str] = None,
    load: Optional[LoadOptions] = None,
    *,
    serve: bool = False,
    launch: bool = True,
    server_options: ServerOptions = ServerOptions(),
) -> tuple:
    """Return (base, server). server is a Popen only when WE auto-started it."""
    sent = server_options.sent_by_agent()
    server_options = server_options._replace(
        unpinned = frozenset(name for name in sent if getattr(server_options, name) is not None),
        **dict.fromkeys(sent),
    )
    # What the agent cannot send itself only reaches a server WE launch, then applies to every client.
    _pinned = [
        "--" + _name.replace("_", "-")
        for _name in _SAMPLING_FIELDS
        if getattr(server_options, _name) is not None
    ]
    _reasoning_pins = [
        f"{_flag} {_value}"
        for _flag, _value in (
            ("--reasoning", server_options.reasoning),
            ("--reasoning-effort", server_options.reasoning_effort),
        )
        if _value is not None
    ]
    base = find_studio_server()
    if base is not None:
        if _pinned:
            typer.echo(
                f"Warning: an Unsloth server is already running at {base}, and this agent "
                f"cannot send {', '.join(_pinned)} itself; they apply only when this command "
                "starts the server, so the running server keeps its current sampling. Stop it "
                "with `unsloth studio stop` and re-run to apply them.",
                err = True,
            )
        if _reasoning_pins:
            typer.echo(
                f"Warning: an Unsloth server is already running at {base}, and this agent "
                f"cannot send {', '.join(_reasoning_pins)} itself; it takes effect only when "
                "this command starts the server, so the running server keeps its current "
                "reasoning mode. Stop it with `unsloth studio stop` and re-run to apply the "
                "override.",
                err = True,
            )
        _tool_flags = [
            _on if _value else _off
            for _value, _on, _off in (
                (server_options.enable_tools, "--enable-tools", "--disable-tools"),
                (
                    server_options.tool_call_healing,
                    "--enable-tool-call-healing",
                    "--disable-tool-call-healing",
                ),
                (
                    server_options.tool_call_nudging,
                    "--enable-tool-call-nudging",
                    "--disable-tool-call-nudging",
                ),
            )
            if _value is not None
        ]
        if _tool_flags:
            typer.echo(
                f"Warning: an Unsloth server is already running at {base}; "
                f"{', '.join(_tool_flags)} takes effect only when this command starts the "
                "server, so the running server keeps its current tool settings. Stop it with "
                "`unsloth studio stop` and re-run to apply it.",
                err = True,
            )
        return base, None
    expected = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888").rstrip("/")
    # Auto-start only for an interactive launch to a plain-HTTP loopback target: never stand in for an
    # explicit remote URL, and `unsloth run` serves no https.
    if (
        serve
        and launch
        and model
        and is_loopback_url(expected)
        and urlparse(expected).scheme == "http"
    ):
        # Normalize to the port unsloth run binds, not a portless :80.
        expected = _effective_base(expected)
        load = load or LoadOptions()
        _server_wide = _pinned + [pin for pin in _reasoning_pins if pin != "--reasoning auto"]
        if _server_wide:
            typer.echo(
                f"Warning: this agent cannot send {', '.join(_server_wide)} itself, so the "
                "server this command starts applies them to every client until it stops.",
                err = True,
            )
        _dropped = [
            f"UNSLOTH_SAMPLING_{_name.upper()}"
            for _name in _SAMPLING_FIELDS
            if _name in server_options.unpinned
            and os.environ.get(f"UNSLOTH_SAMPLING_{_name.upper()}")
        ]
        if _dropped:
            typer.echo(
                f"Warning: this agent's flags replace the inherited {', '.join(_dropped)}, so "
                "the server this command starts drops those pins for every client.",
                err = True,
            )
        # Leave a bare GGUF repo's variant unset: the server's quant preference and fallback pick better.
        return _start_studio_server(expected, model, load, server_options)
    model_hint = "" if model else " Pass --model to have it start one for you, or"
    _fail(
        f"No running Unsloth server found at {expected}.{model_hint} start one with "
        "`unsloth studio`, or point UNSLOTH_STUDIO_URL at a remote server."
    )


def _studio_auth_root() -> Path:
    from unsloth_cli.commands.studio import STUDIO_HOME
    return STUDIO_HOME / "auth"


def _key_cache_path() -> Path:
    return _studio_auth_root() / "agent_api_key.json"


def _read_cache(cache: Path) -> dict:
    try:
        data = json.loads(cache.read_text(encoding = "utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _server_buckets(servers: dict, base: str) -> dict:
    # Tolerate a corrupt or legacy value (a bare string or list is treated as minted).
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
    # Per-server keys: "saved" (user --api-key, trusted) vs "minted" (replayed only after the identity
    # check). Legacy unscoped caches are ignored.
    return _server_buckets(_read_cache(cache).get("servers", {}), base)[source]


def _write_private_json(path: Path, data: dict) -> None:
    # O_CREAT with 0o600 so a file holding an API key is never world-readable, even briefly.
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(data, indent = 2) + "\n")


def _write_private_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(text)


def _read_yaml_object(path: Path) -> Optional[dict]:
    import yaml

    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding = "utf-8"))
    except (yaml.YAMLError, OSError):
        return None
    if data is None:
        return {}
    return data if isinstance(data, dict) else None


def _read_json_object(path: Path) -> Optional[dict]:
    # None when unparseable, so the caller leaves a user-managed file untouched.
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
    buckets[other] = [k for k in buckets[other] if k != key]
    new_entry = {"saved": buckets["saved"], "minted": buckets["minted"]}
    if servers.get(base) == new_entry:
        return
    servers[base] = new_entry
    data.pop("keys", None)
    data.pop("key", None)
    try:
        _write_private_json(cache, data)
    except OSError:
        pass


def _loaded_models_response(
    base: str,
    key: str,
    timeout = 30,
) -> dict:
    """Raw listing of what this server has resident. Startup and key checks need no more.

    /v1/models answers the same question but waits for disk and media discovery first,
    which on a slow scan folder outlasts the deadline below.
    """
    try:
        answer = _http_json("GET", f"{base}/api/inference/loaded-models", key, timeout = timeout)
        if isinstance(answer.get("data"), list):
            return answer
        # An older Studio answers an unknown /api path with 200 {"error": ...}; no "data" list means no route.
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    # Never on an auth or server error. Older Studios answer /v1/models.
    return _http_json("GET", f"{base}/v1/models", key, timeout = timeout)


def _key_accepted(base: str, key: str) -> bool:
    # Only 401/403 means a bad key; a 5xx or blip must not discard a working key and mint extras.
    try:
        _loaded_models_response(base, key)
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
        # An auto-started server makes the loopback mint path certain, so an unrelated exported key must not fail.

    # Replay a key saved for this exact server first (works for a remote or tunnelled Unsloth).
    for key in _cached_keys(cache, base, "saved"):
        if _key_accepted(base, key):
            _remember_key(cache, base, key, "saved")
            return key

    # find_studio_server() trusts a base after only a health check, so minting and replaying minted keys
    # are limited to a loopback server we can confirm is ours.
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

    token = _studio_token()
    if token is None:
        _fail(
            "Couldn't authenticate with the Unsloth server automatically. Create "
            "an API key in Unsloth → Settings → API and pass it with --api-key, "
            "or set UNSLOTH_API_KEY."
        )
    # Older releases could mint for a managed account, which cannot see the owner's model.
    owned = {
        entry.get("key_prefix")
        for entry in _http_json(
            "GET", f"{base}/api/auth/api-keys", token, error = "Couldn't list API keys"
        ).get("api_keys", [])
    }
    for key in _cached_keys(cache, base, "minted"):
        if key[len("sk-unsloth-") :][:8] in owned and _key_accepted(base, key):
            _remember_key(cache, base, key, "minted")
            return key

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
    try:
        return _loaded_models_response(base, key).get("data", [])
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        _fail_request(exc, "Couldn't list models")


def _model_loaded_state(base: str, key: str, model_id: object) -> Optional[bool]:
    """Return whether the model is loaded, or None if the listing is unavailable."""
    try:
        models = _loaded_models_response(base, key, timeout = 5).get("data", [])
    except Exception:
        return None
    return any(m.get("id") == model_id and m.get("loaded") is not False for m in models)


def _model_still_loaded(base: str, key: str, model_id: object) -> bool:
    return _model_loaded_state(base, key, model_id) is True


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
    # Extra path segments mean a server-side path, never casefold-matched against a hub id.
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


def _is_model_path(value: str) -> bool:
    """Mirrors core.inference.model_ids._looks_like_path: a repo id is exactly ``org/model``, and anything else with a separator, drive, prefix or .gguf is a path. Deliberately not named _looks_like_path: that name is taken further down by the WSLENV classifier, which only matches absolute paths and would shadow this one."""
    if value.lower().endswith(".gguf"):
        return True
    if value.startswith(("/", "\\", "./", "../", ".\\", "..\\", "~")):
        return True
    if len(value) >= 2 and value[1] == ":":
        return True
    return value.count("/") >= 2 or "\\" in value


def _public_model_id(value: Optional[str]) -> Optional[str]:
    """The id Unsloth advertises for a model loaded by path. The loaded listing never echoes a host path: it reports the file or directory name with any .gguf suffix stripped (core.inference.model_ids.public_model_id), so a path we asked to load has to be matched by that name too."""
    if not value or not _is_model_path(value):
        return None
    name = os.path.basename(value.replace("\\", "/").rstrip("/"))
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    return name or None


def _public_model_ids(value: Optional[str]) -> set:
    """Return a path's basename ID and HF cache repo ID for old and current servers."""
    ids = {_public_model_id(value)} - {None}
    if ids:
        parts = value.replace("\\", "/").split("/")
        for index, part in enumerate(parts):
            if part.startswith("models--") and parts[index + 1 : index + 2] == ["snapshots"]:
                ids.add(part[len("models--") :].replace("--", "/"))
                break
    return ids


def _model_id_matches(
    actual: object,
    requested: object,
    *,
    allow_casefold: bool = True,
) -> bool:
    if actual == requested:
        return True
    # Casefold only against a loopback Unsloth, where the local existence probe is authoritative.
    if not allow_casefold:
        return False
    if not (_is_hub_model_id(actual) and _is_hub_model_id(requested)):
        return False
    return str(actual).casefold() == str(requested).casefold()


def _inference_status(base: str, key: str) -> dict:
    """Runtime state of the resident model. {} means "cannot prove anything" (older server), never "nothing is set"."""
    try:
        return _http_json("GET", f"{base}/api/inference/status", key)
    except Exception:
        return {}


_OTHER_ACCOUNT_RESIDENT = (
    "The model loaded in Unsloth belongs to another account, so this API key cannot use it. "
    "Pass --model <hf-id-or-path> to load one: the same model and quant shares it, anything "
    "else unloads it for every session using it."
)


def _resident_load_target(models: list, status: dict, allow_casefold: bool):
    """(identifier to post, id it is advertised as) for the running model. The loaded listing shows only the sanitized basename while _same_loaded_identifier compares resident paths exactly, so the load must carry the identifier status reports."""
    if status.get("is_diffusion"):
        # An image runtime cannot serve chat; targeting it would tear down the diffusion server.
        _fail(
            "Unsloth is serving an image model, which cannot serve chat, so there are no "
            "settings to apply. Re-run with --model naming the chat model to load."
        )
    active_id = status.get("active_model")
    entry = None
    if active_id:
        entry = next(
            (
                m
                for m in models
                if _model_id_matches(m.get("id"), active_id, allow_casefold = allow_casefold)
                and m.get("loaded") is not False
            ),
            None,
        )
    if entry is None and not status:
        # Only when there is no status at all (older server); catalog order is not evidence.
        loaded = [m for m in models if m.get("loaded") is not False]
        if len(loaded) == 1:
            entry = loaded[0]
        elif loaded:
            _fail(
                "This Unsloth cannot say which model is serving chat, and more than one is "
                "loaded. Re-run with --model naming the one these settings are for."
            )
    public_id = active_id or (entry or {}).get("id")
    if not public_id:
        if status.get("yours") is False:
            _fail(_OTHER_ACCOUNT_RESIDENT)
        if status:
            # Returning empty would drop the knobs silently.
            _fail(
                "No chat model is currently loaded, so there are no settings to apply. "
                "Re-run with --model naming the model to load."
            )
        return None, None
    identifier = status.get("model_identifier")
    if identifier:
        return identifier, public_id
    # A path lease redacts the internal path, so only a hub id can be posted back.
    if _is_hub_model_id(public_id):
        return public_id, public_id
    _fail(
        f"Unsloth is serving '{public_id}' from a local path it does not expose, so these "
        "settings cannot be applied by attaching. Re-run with --model naming that path."
    )


# Status field -> load field. "requested_" values are what the load was invoked with.
_RESIDENT_RUNTIME_FIELDS = {
    "cache_type_kv": "cache_type_kv",
    "chat_template_override": "chat_template_override",
    "disable_vision": "disable_vision",
    "gpu_memory_mode": "gpu_memory_mode",
    "gpu_layers": "gpu_layers",
    "n_cpu_moe": "n_cpu_moe",
    "tensor_split": "tensor_split",
    "tensor_parallel": "tensor_parallel",
    "speculative_type": "speculative_type",
    "spec_draft_n_max": "spec_draft_n_max",
    # Null when the runtime refused. Scheme and width together: a bare width reloads as mx.quantize.
    "mlx_kv_quant_requested": "mlx_kv_quant",
    "mlx_kv_bits_requested": "mlx_kv_bits",
    # LoadRequest defaults this to True; null on GGUF.
    "load_in_4bit": "load_in_4bit",
    # max_seq_length defaults to 0, which would reset a custom context.
    "requested_context_length": "max_seq_length",
    "requested_gpu_ids": "gpu_ids",
    "requested_parallel_slots": "n_parallel",
    "requested_n_batch": "n_batch",
    "requested_n_ubatch": "n_ubatch",
    "requested_load_mode": "load_mode",
    "requested_ctx_checkpoints": "ctx_checkpoints",
    "requested_cache_ram": "cache_ram",
    "requested_spec_draft_cache_type": "spec_draft_cache_type",
    "requested_llama_extra_args": "llama_extra_args",
}


def _resident_runtime_payload(status: dict, payload: dict) -> dict:
    """The resident's own settings for knobs this load did not name. None means "never set" for every one of these, so it is dropped rather than sent: omitting a field is what lets the server inherit, while sending null would pin the absence. An explicit empty list is kept, since that is a real "launch with none"."""
    if not status:
        return {}
    carried = {}
    for field, load_field in _RESIDENT_RUNTIME_FIELDS.items():
        if load_field in payload:
            continue
        value = status.get(field)
        if value is None:
            continue
        carried[load_field] = value
    return carried


def _load_settings_differ(status: dict, load: LoadOptions, overrides: frozenset) -> bool:
    """Whether applying these settings can restart the resident. Unproven equality counts as a difference: a silent restart is worse than a spurious warning."""
    if not status:
        return True
    # A pending spec probe / drafter retry makes the server reload anyway, so equality is no no-op proof.
    if any(
        status.get(field)
        for field in (
            "spec_probe_retry_pending",
            "spec_dflash_retry_pending",
            "spec_fallback_binary_changed",
        )
    ):
        return True
    for name in overrides:
        if name == "gguf_variant":
            resident = status.get("gguf_variant") if status.get("is_gguf") else None
            # Casefold, not _normalized_variant: must agree with the preload gate.
            if (
                not resident
                or str(resident).strip().lower() != str(load.gguf_variant).strip().lower()
            ):
                return True
        elif name == "max_seq_length":
            # Requested, not resolved: llama.cpp clamps n_ctx at fit time.
            resident = status.get("requested_context_length")
            if resident is None or int(resident) != int(load.max_seq_length):
                return True
        elif name == "load_in_4bit":
            # GGUF reports null for 4-bit, which would read as a difference.
            if status.get("is_gguf"):
                continue
            resident = status.get("load_in_4bit")
            if resident is None or bool(resident) != bool(load.load_in_4bit):
                return True
        elif name == "tensor_parallel":
            # llama.cpp only. The standard load never forwards it.
            if not status.get("is_gguf"):
                continue
            # Re-asking for an arch-gate-normalized tensor mode is a no-op; turning it OFF needs a real reload.
            if status.get("tensor_parallel_dropped_by_arch_gate"):
                if load.tensor_parallel:
                    continue
                return True
            if bool(status.get("tensor_parallel")) != bool(load.tensor_parallel):
                return True
        elif name == "gpu_memory_mode":
            if not status.get("is_gguf"):
                continue
            # Paravirtual hosts and CPU fallbacks hide placement differences (as resident-config-match.ts does).
            if status.get("gpu_placement_paravirtual") or status.get("cpu_fallback_reason"):
                continue
            if status.get("gpu_memory_mode") != load.gpu_memory_mode:
                return True
            # Manual to manual is a real no-op.
    return False


def _resolve_model(
    base: str,
    key: str,
    requested: Optional[str],
    load: LoadOptions = LoadOptions(),
    preload_check = None,
    infer_resident: bool = True,
) -> dict:
    models = _loaded_models(base, key)
    load_requested = False
    # Only casefold against a loopback Unsloth; see _is_hub_model_id.
    allow_casefold = is_loopback_url(base)
    # An id match can hide the wrong quant; with any explicit knob, defer to /api/inference/load, whose
    # dedup answers already_loaded when variant and settings match.
    overrides = load.overrides()
    load_has_overrides = bool(overrides)
    # `requested` may be a server path here, so this is the id to show and match on.
    attach_public_id = None
    status_snapshot = None
    # Computed once so the gate, warning, consent refusal and force_reload agree.
    inferred_differs = False
    if requested is None and load_has_overrides and infer_resident:
        status_snapshot = _inference_status(base, key)
        requested, attach_public_id = _resident_load_target(models, status_snapshot, allow_casefold)
        inferred_differs = _load_settings_differ(status_snapshot, load, overrides)
        # Only attach to an entry that is actually loaded: older servers list cached-but-unloaded entries too.
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
        # The gate must not reject a request the resident model already satisfies.
        active = next((m for m in models if m.get("loaded") is not False), None)
        # On the inferred path catalog order can name a different entry.
        if attach_public_id is not None:
            active = next(
                (
                    m
                    for m in models
                    if _model_id_matches(
                        m.get("id"), attach_public_id, allow_casefold = allow_casefold
                    )
                    and m.get("loaded") is not False
                ),
                None,
            ) or {"id": attach_public_id}
        if preload_check is not None:
            # An explicit knob lets the server's dedupe answer; only the quant is checked below.
            other_overrides = bool(overrides - {"gguf_variant"})
            wanted_ids = {requested} | _public_model_ids(requested)
            resident_serves_request = not other_overrides and any(
                m.get("loaded") is not False
                and any(
                    _model_id_matches(m.get("id"), want, allow_casefold = allow_casefold)
                    for want in wanted_ids
                )
                for m in models
            )
            # A proven no-op evicts nothing, so skip the gate.
            if attach_public_id is not None and not inferred_differs:
                resident_serves_request = True
            # Confirm a path request against the loaded identifier, not only the basename.
            if resident_serves_request and _is_model_path(requested):
                try:
                    status = _http_json("GET", f"{base}/api/inference/status", key)
                except Exception:
                    status = {}
                loaded_paths = {
                    str(status.get(field))
                    for field in ("model_identifier", "gguf_path", "model_path")
                    if status.get(field)
                }
                wanted_path = os.path.abspath(os.path.expanduser(requested))
                resident_serves_request = any(
                    os.path.abspath(os.path.expanduser(path)) == wanted_path
                    for path in loaded_paths
                )
            if resident_serves_request and load.gguf_variant:
                try:
                    status = _http_json("GET", f"{base}/api/inference/status", key)
                except Exception:
                    status = {}
                resident_variant = status.get("gguf_variant") if status.get("is_gguf") else None
                # Casefold, not _normalized_variant: a mistyped Q4KM would still reload.
                resident_serves_request = (
                    bool(resident_variant)
                    and str(resident_variant).strip().lower()
                    == str(load.gguf_variant).strip().lower()
                )
            if not resident_serves_request:
                preload_check(base, key, requested, load.gguf_variant)
        active_id = active.get("id") if active else None
        announced_switch = False
        # Public IDs can collide and status hides paths: wait for the load result.
        switch_unknown = False
        if attach_public_id is not None:
            # An inferred attach never switches model.
            if inferred_differs:
                typer.echo(f"Applying new load settings to {attach_public_id}.")
                typer.echo("This unloads the current model for every attached session.")
                announced_switch = True
        elif active_id and not _model_id_matches(
            active_id,
            requested,
            allow_casefold = allow_casefold,
        ):
            if any(
                _model_id_matches(active_id, listed, allow_casefold = allow_casefold)
                for listed in _public_model_ids(requested)
            ):
                switch_unknown = True
            else:
                typer.echo(f"Switching the Unsloth server from {active_id} to {requested}.")
                typer.echo("This unloads the current model for every attached session.")
                announced_switch = True
        elif active_id and load.gguf_variant:
            # The listing does not always carry a variant, so ask the status endpoint.
            try:
                status = _http_json("GET", f"{base}/api/inference/status", key)
            except Exception:
                status = {}
            resident = status.get("gguf_variant") if status.get("is_gguf") else None
            if resident and _normalized_variant(resident) != _normalized_variant(load.gguf_variant):
                typer.echo(
                    f"Switching the Unsloth server from {active_id}:{resident} "
                    f"to {requested}:{load.gguf_variant}."
                )
                typer.echo("This unloads the current model for every attached session.")
                announced_switch = True
        elif not active_id and _inference_status(base, key).get("yours") is False:
            typer.echo(
                "Switching the Unsloth server from another account's model to "
                f"{_display_model_spec(requested, load.gguf_variant)}."
            )
            typer.echo(
                "This unloads it for every attached session, unless it is the same model and quant."
            )
        # Mirror `unsloth run`'s load knobs. Membership decides, not truthiness: a reset must be sent.
        payload = {"model_path": requested}
        if "gguf_variant" in overrides and load.gguf_variant:
            direct_file = attach_public_id is not None and str(requested).lower().endswith(".gguf")
            if direct_file:
                # from_identifier consults a variant only for a directory, so drop an inapplicable quant.
                resident_variant = status_snapshot.get("gguf_variant")
                same = (
                    bool(resident_variant)
                    and str(resident_variant).strip().lower()
                    == str(load.gguf_variant).strip().lower()
                )
                if not same:
                    _fail(
                        f"'{attach_public_id}' was loaded from a single .gguf file, so "
                        f"--gguf-variant {load.gguf_variant} cannot select a different quant. "
                        "Re-run with --model naming the repository to switch quants."
                    )
            else:
                payload["gguf_variant"] = load.gguf_variant
        elif attach_public_id is not None and status_snapshot.get("is_gguf"):
            # Re-send the running quant, or a context change would auto-pick a different one. Not for .gguf paths.
            resident_variant = status_snapshot.get("gguf_variant")
            if resident_variant and not str(requested).lower().endswith(".gguf"):
                payload["gguf_variant"] = resident_variant
        if "max_seq_length" in overrides:
            payload["max_seq_length"] = load.max_seq_length
        if "load_in_4bit" in overrides:
            payload["load_in_4bit"] = load.load_in_4bit
        if "tensor_parallel" in overrides:
            payload["tensor_parallel"] = load.tensor_parallel
        if "gpu_memory_mode" in overrides and load.gpu_memory_mode is not None:
            payload["gpu_memory_mode"] = load.gpu_memory_mode
            # On an inferred attach already in manual, -1 would drop the pinned layer count.
            already_manual = (
                attach_public_id is not None and status_snapshot.get("gpu_memory_mode") == "manual"
            )
            if load.gpu_memory_mode == "manual" and not already_manual:
                payload["gpu_layers"] = -1
        if (
            attach_public_id is not None
            and inferred_differs
            and status_snapshot.get("requires_trust_remote_code")
        ):
            # A reload cannot reproduce trust_remote_code consent, and a rejected load leaves nothing resident.
            _fail(
                f"'{attach_public_id}' was loaded with trust_remote_code, which an attach "
                "cannot re-authorize. Re-run with --model naming it to apply these settings."
            )
        if attach_public_id is not None:
            # An inferred reload is a full load: carry the resident's values for knobs the user did not name.
            payload.update(_resident_runtime_payload(status_snapshot, payload))
            # Force the reload only when status proved a difference: the server treats 0 as no preference.
            if status_snapshot and inferred_differs:
                payload["force_reload"] = True
        try:
            loaded = _load_model_with_progress(base, key, requested, load, payload)
        except Exception:
            # Not BaseException: Ctrl+C must stay immediate.
            if announced_switch and _model_still_loaded(base, key, active_id):
                typer.echo(f"Nothing was unloaded; {active_id} is still serving.", err = True)
            # Report an unannounced eviction only if the listing confirms it.
            if switch_unknown and _model_loaded_state(base, key, active_id) is False:
                typer.echo(f"{active_id} was unloaded for every attached session.", err = True)
            raise
        if loaded.get("status") == "already_loaded":
            # `requested` may be a server path.
            shown = attach_public_id or requested
            typer.echo(f"Reusing loaded model: {_display_model_spec(shown, load.gguf_variant)}")
        elif switch_unknown:
            typer.echo(f"Loaded {requested} in place of {active_id}.")
            typer.echo("This unloaded the previous model for every attached session.")
        # Match public IDs and load-response names; the listing may omit paths.
        wanted = ({requested, attach_public_id} - {None}) | _public_model_ids(requested)
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
        # Do not silently hand back an unrelated loaded model.
        _fail(
            f"Unsloth didn't report '{requested}' as loaded. Double-check the model "
            "id, or load it from the model dropdown in the UI."
        )
    resident = next((m for m in models if m.get("loaded") is not False), None)
    if resident is None:
        if _inference_status(base, key).get("yours") is False:
            _fail(_OTHER_ACCOUNT_RESIDENT)
        # Which of the two a server sends depends only on its version.
        _fail(
            "No model is loaded in Unsloth. Load one from the model dropdown in "
            "the UI, or pass --model <hf-id-or-path> to load it from here."
        )
    return resident


_HF_OFFLINE_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _hf_offline() -> bool:
    return any(
        os.environ.get(var, "").strip().lower() in _HF_OFFLINE_TRUE_VALUES
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _hub_gguf_files(repo: str) -> Optional[list]:
    if _hf_offline():
        return None
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    try:
        request = urllib.request.Request(
            f"{endpoint}/api/models/{repo}",
            headers = {"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout = 10) as response:
            info = json.loads(response.read().decode() or "{}")
    except Exception:
        return None
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        return None
    names = [s.get("rfilename") for s in siblings if isinstance(s, dict)]
    ggufs = [n for n in names if isinstance(n, str) and n.lower().endswith(".gguf")]
    return [n for n in ggufs if not _is_auxiliary_gguf(n)]


# Mirrors hub.utils.gguf._DRAFTER_KINDS / _DRAFTER_DIR_KINDS; dflash/ is also a family name.
_DRAFTER_KINDS = ("mtp", "dspark", "dflash", "eagle3")
_DRAFTER_DIR_KINDS = ("mtp", "dspark")


def _is_auxiliary_gguf(filename: str) -> bool:
    # Mirrors detect_gguf_model_remote: drafters match by basename prefix or exact parent dir, never
    # substring (kind names double as family names).
    p = filename.lower().replace("\\", "/")
    parts = [segment for segment in p.split("/") if segment]
    if not parts:
        return False
    name, parents = parts[-1], parts[:-1]
    if "mmproj" in p:
        return True
    if any(name.startswith(f"{kind}-") for kind in _DRAFTER_KINDS):
        return True
    if any(kind in parents for kind in _DRAFTER_DIR_KINDS):
        return True
    stem = name.rsplit(".", 1)[0]
    return not parents and stem.endswith(("-be", "_be"))


def _direct_gguf_is_companion(path: str) -> bool:
    """Whether the server refuses this .gguf path as a model in its own right. A strict subset of detect_gguf_model / gguf_variants._direct_gguf_loads: projector and drafter prefixes read off the basename, companion-only folders off the immediate parent, the same context the server reads, so nothing loadable is refused here. Big-endian is left out on purpose: that check needs quant context the CLI cannot mirror."""
    parts = [segment for segment in path.replace("\\", "/").split("/") if segment]
    if not parts:
        return False
    name = parts[-1].lower()
    if not name.endswith(".gguf"):
        return False
    # Root-independent refusals only; a drafter folder is not.
    if "mmproj" in name:
        return True
    return any(name.startswith(f"{kind}-") for kind in _DRAFTER_KINDS)


def _path_syntax_is_native(path: str) -> bool:
    """Whether *path* is spelled the way this OS spells paths. A Windows path read from WSL, or a POSIX one read from Windows, parses into something this process cannot judge (``C:\\models\\m.gguf`` has parent ``.`` here), so its absence locally says nothing about the server's disk."""
    windows_drive = len(path) >= 2 and path[1] == ":" and path[0].isalpha()
    if os.name == "nt":
        return True
    return not windows_drive and "\\" not in path


def _direct_gguf_companion_is_uncertain(path: str) -> bool:
    """Whether only the server can say if this path is a companion. detect_gguf_model reads drafter folders relative to the registered model root, so ``/models/MTP/foo-Q8_0.gguf`` is refused or loaded depending on where that root sits, a question only the server can answer since this process does not know its roots."""
    parts = [segment for segment in path.replace("\\", "/").split("/") if segment]
    return any(segment.lower() in _DRAFTER_DIR_KINDS for segment in parts[:-1])


# Mirrors model_config._extract_quant_label's pattern; change in lockstep.
_QUANT_LABEL_RE = re.compile(
    r"(UD-)?"
    r"(MXFP[0-9]+(?:_[A-Z0-9]+)*"
    r"|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?"
    r"|P?TQ[0-9]+_[0-9]+"
    r"|Q[0-9]+_K_[A-Z]+"
    r"|P?Q[0-9]+_[0-9]+(?:_G[0-9]+)?"
    r"|Q[0-9]+_K"
    r"|BF16|F16|F32)"
    r"(-[0-9]+(?:\.[0-9]+)?bpw)?",
    re.IGNORECASE,
)


def _direct_gguf_variant_labels(path: str) -> tuple:
    """The labels the server's direct-file resolver accepts for *path*. Mirrors _direct_gguf_for_variant: the quant label read from the basename first, the immediate parent only when the basename carries none, plus the shard-stripped stem itself. The basename wins the disagreement: a Q8_0/foo-Q4_K_M.gguf answers Q4_K_M, so its parent must not vouch for Q8_0 here while the load resolves nothing and evicts."""
    norm = path.replace("\\", "/").rstrip("/")
    name = norm.rsplit("/", 1)[-1]
    stem = re.sub(r"-\d{3,}-of-\d{3,}$", "", name.rsplit(".", 1)[0])
    match = _QUANT_LABEL_RE.search(stem)
    if match is None and "/" in norm:
        match = _QUANT_LABEL_RE.search(norm.rsplit("/", 2)[-2])
    labels = [stem]
    if match is not None:
        label = f"{match.group(1) or ''}{match.group(2)}{match.group(3) or ''}"
        labels.append(label)
        # The resolver also accepts the hub-style bpw-stripped spelling.
        stripped = re.sub(r"-[0-9]+(?:\.[0-9]+)?bpw$", "", label, flags = re.IGNORECASE)
        if stripped != label:
            labels.append(stripped)
    else:
        labels.append(stem.split("-")[-1])
    return tuple(labels)


# Mirrors hub.utils.gguf._BIG_ENDIAN_GGUF_FILENAME_RE; change in lockstep.
_BIG_ENDIAN_FILENAME_RE = re.compile(r"(^|[-_])be(?:[._-]|$)", re.IGNORECASE)


def _direct_gguf_is_big_endian(path: str) -> bool:
    """Mirrors hub.utils.gguf.is_big_endian_gguf_path over the same one-parent context detect_gguf_model reads; change in lockstep. A quant-named parent exempts a bare -be basename (that file loads); a be marker at or after a basename quant does not."""
    norm = path.replace("\\", "/").rstrip("/")
    parts = [segment for segment in norm.split("/") if segment]
    name = parts[-1]
    stem = name.rsplit(".", 1)[0].lower()
    quant_stem = re.sub(r"-\d{3,}-of-\d{3,}$", "", stem)
    match = _QUANT_LABEL_RE.search(quant_stem)
    parent = parts[-2].lower() if len(parts) > 1 else ""
    if match is None and parent:
        match = _QUANT_LABEL_RE.search(parent)
    quant_key = (
        f"{match.group(1) or ''}{match.group(2)}{match.group(3) or ''}".lower()
        if match is not None
        else quant_stem.split("-")[-1]
    )
    quant_index = stem.find(quant_key) if quant_key else -1
    quant_in_parent_only = (
        bool(parent)
        and quant_index < 0
        and (
            (quant_key and quant_key in parent)
            or (not quant_key and _QUANT_LABEL_RE.search(parent))
        )
    )
    for be in _BIG_ENDIAN_FILENAME_RE.finditer(stem):
        if quant_index >= 0 and quant_index < be.start():
            return True
        tail = stem[be.end() :].lstrip("._-")
        if not tail or _QUANT_LABEL_RE.search(tail) is None:
            return not quant_in_parent_only
    return False


# Mirrors gguf_variants._DIRECT_SPLIT_RE / _GGUF_SPLIT_FILE_RE; change in lockstep. Five digits exactly.
_DIRECT_SPLIT_FILE_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$", re.IGNORECASE
)


def _direct_gguf_file_is_ready(path: str) -> bool:
    """Whether a CLI-visible direct .gguf file can actually serve a load. Mirrors the backend's completeness rules: zero bytes is an interrupted copy, and a split needs every sibling index present and non-empty. Unknowable reports ready, so a path this process cannot judge never blocks the load."""

    def _split_set_complete(candidate: Path) -> Optional[bool]:
        match = _DIRECT_SPLIT_FILE_RE.match(candidate.name.rsplit(".", 1)[0])
        if match is None:
            return None
        total = int(match.group("total"))
        if total < 2:
            return None
        sibling = re.compile(
            re.escape(match.group("stem"))
            + r"-(\d{"
            + str(len(match.group("index")))
            + r"})-of-"
            + re.escape(match.group("total"))
            + r"\.gguf$",
            re.IGNORECASE,
        )
        found = {
            int(m.group(1))
            for q in candidate.parent.iterdir()
            if (m := sibling.match(q.name)) and q.is_file() and q.stat().st_size > 0
        }
        return found >= set(range(1, total + 1))

    try:
        p = Path(os.path.expanduser(path))
        # A broken symlink still loads by its .gguf suffix and fails after teardown.
        if p.is_symlink() and not p.exists():
            return False
        if not p.is_file():
            return True
        if p.stat().st_size == 0:
            return False
        whole = _split_set_complete(p)
        if whole is not False:
            return True
        # Mirror _local_gguf_load_path: a symlinked shard may point at a full set.
        if p.is_symlink():
            target = p.resolve()
            return _split_set_complete(target) is not False
        return False
    except OSError:
        return True


def _answer_offers_variant(
    variants: list,
    variant: str,
    strict: bool = False,
) -> bool:
    """Whether a live variants answer can resolve *variant* to a file. Mirrors llama.cpp's resolution, case-insensitively: quant label first, then the whole-token filename fallback, which is as loose as it gets since a separator-differing label resolves to nothing there either. A row missing both fields cannot be disproven, so it vouches. ``strict`` drops the filename-token tier: the LOCAL resolver takes only exact labels, so a local answer must not vouch for a shorter token (Q4 inside model-Q4_K_M.gguf) the load would never resolve."""
    wanted = str(variant).strip().lower()
    if not wanted:
        return True
    token = re.compile(r"(?<![a-z0-9])" + re.escape(wanted) + r"(?![a-z0-9])")
    for row in variants:
        if not isinstance(row, dict):
            return True
        # A torn local row cannot vouch in strict mode; hub partials stay resumable.
        if strict and row.get("partial") is True:
            continue
        quant = row.get("quant")
        filename = row.get("filename")
        if not isinstance(quant, str) and not isinstance(filename, str):
            return True
        if isinstance(quant, str):
            label = quant.strip().lower()
            if wanted in (label, re.sub(r"-[0-9]+(?:\.[0-9]+)?bpw$", "", label)):
                return True
        if isinstance(filename, str):
            # Full relative spelling only, never a nested file's bare basename.
            stem = re.sub(r"-\d{3,}-of-\d{3,}$", "", filename.rsplit(".", 1)[0]).lower()
            if wanted == stem:
                return True
            # Any whole quant token of the basename (F16-checkpoint-Q4_K_M -> Q4_K_M).
            if any(
                wanted
                in (
                    f"{m.group(1) or ''}{m.group(2)}".lower(),
                    # Keeps the bpw modifier the hub extractor drops.
                    f"{m.group(1) or ''}{m.group(2)}{m.group(3) or ''}".lower(),
                )
                for m in _QUANT_LABEL_RE.finditer(stem.rsplit("/", 1)[-1])
            ):
                return True
        if strict:
            if not isinstance(quant, str):
                return True
            continue
        if isinstance(filename, str) and token.search(filename.lower()):
            return True
    return False


class _GgufAgent(NamedTuple):
    label: str
    command: str


_CODEX_GGUF_AGENT = _GgufAgent("Codex", "codex")
_CLAUDE_GGUF_AGENT = _GgufAgent("Claude Code", "claude")


def _fail_gguf_variant_missing(model_id: str, variant: str, variants: list) -> NoReturn:
    offered = [
        row.get("quant")
        for row in variants
        if isinstance(row, dict) and isinstance(row.get("quant"), str) and row.get("quant")
    ]
    message = f"{model_id} has no GGUF variant {variant}."
    if offered:
        message += " Available: " + ", ".join(dict.fromkeys(offered))
    _fail(message)


def _fail_agent_needs_gguf(agent: _GgufAgent, model_id: str) -> NoReturn:
    message = (
        f"{agent.label} needs a GGUF model served by llama-server, " f"but {model_id} is not one."
    )
    guess = f"{model_id}-GGUF"
    if "gguf" not in model_id.lower() and _is_hub_model_id(guess) and _hub_gguf_files(guess):
        message += f" Try: unsloth start {agent.command} --model {guess}"
    _fail(message)


def _preflight_agent_gguf(
    agent: _GgufAgent,
    model: Optional[str],
    *,
    serve: bool = True,
    launch: bool = True,
) -> None:
    # Hub-listing preflight for the auto-start path only; a running server is asked instead. Only a
    # complete listing with no .gguf rejects.
    if not (serve and launch and model):
        return
    expected = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888").rstrip("/")
    if not is_loopback_url(expected) or urlparse(expected).scheme != "http":
        return
    if find_studio_server() is not None:
        return
    repo, _ = _split_repo_variant(model)
    # A bare foo.gguf naming no local file is a shorthand, canonicalized like any owner-less name.
    if "/" not in repo and (not _is_model_path(repo) or repo.lower().endswith(".gguf")):
        try:
            if Path(os.path.expanduser(repo)).exists():
                return
        except OSError:
            return
        repo = f"unsloth/{repo}"
    if not _is_hub_model_id(repo):
        return
    files = _hub_gguf_files(repo)
    if files is not None and not files:
        _fail_agent_needs_gguf(agent, repo)


def _attach_gguf_check(
    agent: _GgufAgent,
    base: str,
    key: str,
    model: Optional[str],
    variant: Optional[str] = None,
) -> None:
    # Ask the server before the load evicts the resident model. A live empty list is definitive.
    if not model:
        return
    repo, inline_variant = _split_repo_variant(model)
    variant = variant or inline_variant
    # Only the hub-id shape (owner/name.gguf) is probed.
    bare_missing_gguf = False
    if repo.lower().endswith(".gguf") and not _is_model_path(repo.rsplit(".", 1)[0]):
        try:
            bare_missing_gguf = not Path(os.path.expanduser(repo)).exists()
        except OSError:
            bare_missing_gguf = False
    if (
        repo.lower().endswith(".gguf")
        and not bare_missing_gguf
        and not (
            repo.count("/") == 1
            and not repo.startswith(("/", ".", "~"))
            and ":" not in repo
            and "\\" not in repo
        )
    ):
        # Companion or big-endian builds fall through to transformers and unload llama-server first.
        refused = _direct_gguf_is_companion(repo) or _direct_gguf_is_big_endian(repo)
        if refused:
            _fail_agent_needs_gguf(agent, repo)
        uncertain = _direct_gguf_companion_is_uncertain(repo)
        # A direct file loads as itself; a matching sibling is never substituted.
        if not refused and not uncertain:
            # Only when this machine really shares the server's filesystem: loopback may be an SSH or
            # container forward.
            if is_loopback_url(base) and verify_studio_identity(base):
                # A .gguf-named directory is scanned, not loaded.
                try:
                    is_gguf_dir = Path(os.path.expanduser(repo)).is_dir()
                except OSError:
                    is_gguf_dir = False
                if not is_gguf_dir:
                    # On loopback this reads the server's filesystem. An unreadable parent defers.
                    try:
                        probe = Path(os.path.expanduser(repo))
                        missing = (
                            # A Windows path seen from WSL parses to nonsense, so defer.
                            _path_syntax_is_native(repo)
                            and not probe.is_symlink()
                            and probe.parent.is_dir()
                            and not probe.exists()
                        )
                    except OSError:
                        missing = False
                    if missing:
                        _fail(
                            f"{repo} does not exist. Check the path before "
                            f"pointing {agent.label} at it."
                        )
                    if not _direct_gguf_file_is_ready(repo):
                        _fail(
                            f"{repo} is incomplete (zero bytes or a split missing shards); "
                            f"re-download or re-copy it before pointing {agent.label} at it."
                        )
                    # Only a spelling this OS can judge; an explicit variant goes to the probe.
                    if not variant and _path_syntax_is_native(repo):
                        return
    # Mirrors the load's shorthand precedence: the raw name, then unsloth/<name> only if the raw resolves
    # to nothing. A live answer settles that exact id.
    candidates = [repo]
    if "/" not in repo and (not _is_model_path(repo) or bare_missing_gguf):
        candidates.append(f"unsloth/{repo}")
    for candidate in candidates:
        try:
            info = _http_json(
                "GET", f"{base}/api/models/gguf-variants?{urlencode({'repo_id': candidate})}", key
            )
        except Exception:
            continue
        variants = info.get("variants") if isinstance(info, dict) else None
        if isinstance(variants, list):
            # A cleanable row is an empty leftover <quant>/ folder, never weights.
            variants = [
                row for row in variants if not (isinstance(row, dict) and row.get("cleanable"))
            ]
        # A raw answer the server calls non-local settles nothing (older servers lack the flag).
        if (
            "/" not in candidate
            and candidate != candidates[-1]
            and isinstance(info, dict)
            and "resolved_locally" in info
            and not info["resolved_locally"]
        ):
            continue
        # The server's answer comes from the load resolver itself, so it settles the round, even empty.
        if isinstance(variants, list) and isinstance(info, dict):
            offered = info.get("loadable_variants")
            # No allow-list means a direct file, which loads as itself whatever the quant.
            if variant and offered is None and isinstance(info.get("loadable"), bool):
                if not info["loadable"]:
                    _fail_agent_needs_gguf(agent, candidate)
                return
            if variant and isinstance(offered, list):
                wanted_variant = str(variant).strip().lower()
                if not any(
                    isinstance(q, str) and q.strip().lower() == wanted_variant for q in offered
                ):
                    _fail_gguf_variant_missing(candidate, variant, variants)
                return
            if not variant and isinstance(info.get("loadable"), bool):
                if not info["loadable"]:
                    _fail_agent_needs_gguf(agent, candidate)
                return
        if isinstance(variants, list) and variants:
            # llama.cpp kills the resident model before resolving the quant, so settle it here. Local answers
            # take exact labels only; owner/name.gguf is exempted like the direct-file branch.
            hub_shaped = (
                candidate.count("/") == 1
                and not candidate.startswith(("/", ".", "~"))
                and ":" not in candidate
                and "\\" not in candidate
            )
            local_answer = bool(info.get("resolved_locally")) or (
                not hub_shaped and (_is_model_path(candidate) or "/" not in candidate)
            )
            # Only torn local rows: llama-server would fail after teardown.
            if local_answer and all(
                isinstance(row, dict) and row.get("partial") is True for row in variants
            ):
                _fail(
                    f"{candidate} has only incomplete GGUF weights on the server; "
                    f"finish or re-copy the download before pointing {agent.label} at it."
                )
            # A variantless local load picks from the top level; subdirectory-only rows need the variant.
            if (
                local_answer
                and not variant
                and variants
                and all(
                    isinstance(row, dict)
                    and isinstance(row.get("filename"), str)
                    and "/" in row["filename"]
                    for row in variants
                )
            ):
                offered = ", ".join(
                    dict.fromkeys(
                        row["quant"]
                        for row in variants
                        if isinstance(row.get("quant"), str) and row["quant"]
                    )
                )
                _fail(
                    f"{candidate} keeps its GGUF weights in quant subdirectories, which a "
                    "variantless load cannot pick. Pass --gguf-variant"
                    + (f" (available: {offered})." if offered else ".")
                )
            if variant and not _answer_offers_variant(variants, variant, strict = local_answer):
                _fail_gguf_variant_missing(candidate, variant, variants)
            return
        if isinstance(variants, list):
            # Explicit local syntax settles on a live empty answer; marker-less names keep deferring (older
            # servers read them as hub ids). A bare foo.gguf waits for the canonical form.
            if bare_missing_gguf and candidate != candidates[-1]:
                continue
            if not _is_model_path(repo) and not info.get("resolved_locally"):
                try:
                    if Path(os.path.expanduser(repo)).exists():
                        return
                except OSError:
                    return
            _fail_agent_needs_gguf(agent, candidate)


def _require_gguf_for_agent(agent: _GgufAgent, base: str, key: str, model_id: str) -> None:
    # Only a definite "no" rejects: callers shut the auto-served server down on any exception.
    try:
        status = _http_json("GET", f"{base}/api/inference/status", key)
    except urllib.error.HTTPError:
        # Not evidence: get_status 500s from its own probes with a GGUF resident.
        return
    except (OSError, ValueError, http.client.HTTPException):
        # Named explicitly so typer.Exit and real bugs still surface.
        return
    if not isinstance(status, dict):
        return
    is_gguf = status.get("is_gguf")
    # A current server always sends is_gguf; absent means "not that endpoint", never "non-GGUF".
    if is_gguf is None or is_gguf:
        return
    # is_gguf defaults to False on an idle server.
    if not (status.get("active_model") or status.get("model_identifier")):
        return
    _fail_agent_needs_gguf(agent, model_id)


_DYNAMIC_SECTIONS_FLAG = "--exclude-dynamic-system-prompt-sections"


def _claude_settings_overlay(model_id: str, local_env: Optional[dict] = None) -> str:
    # Command-tier pins beat user/project settings, which Claude applies after the process env.
    settings_env = {name: "" for name in _CLAUDE_ENV_UNSET}
    settings_env.update(local_env or {})
    settings_env.update(
        {
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
            "CLAUDE_CODE_SUBAGENT_MODEL": "inherit",
        }
    )
    return json.dumps(
        {
            "env": settings_env,
            "availableModels": [model_id],
        }
    )


def _write_claude_settings(path: Path, model_id: str, local_env: dict) -> Path:
    overlay = _claude_settings_overlay(model_id, local_env)
    digest = hashlib.sha256(overlay.encode("utf-8")).hexdigest()[:16]
    settings = path / f"settings-{digest}.json"
    _write_private_text(settings, overlay)
    return settings


def _claude_version() -> Optional[tuple]:
    # None means no local `claude`: assume a current build. Unparseable means too old.
    executable = _which_with_install_dirs("claude")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            env = _probe_env(),
        )
        # Do not assume the version is the first token.
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
        return tuple(int(part) for part in match.groups()) if match else (0,)
    except Exception:
        return (0,)


def _claude_flags(model_id: str, settings: Optional[str] = None) -> list:
    # claude < 2.1.98 rejects the dynamic-sections flag but supports --settings.
    version = _claude_version()
    settings_flags = ["--settings", settings or _claude_settings_overlay(model_id)]
    if version is not None and version < (2, 1, 98):
        return settings_flags
    return [_DYNAMIC_SECTIONS_FLAG, *settings_flags]


def _claude_local_command(model_id: str, settings: str, yolo: bool, passthrough: list) -> list:
    local_args = [
        "--model",
        model_id,
        *_claude_flags(model_id, settings),
        *_yolo_command_flags("claude", yolo),
    ]
    forwarded = list(passthrough)
    separator = forwarded.index("--") if "--" in forwarded else len(forwarded)
    before_separator = forwarded[:separator]
    forwarded_settings = []
    remaining = []
    index = 0
    while index < len(before_separator):
        arg = before_separator[index]
        if arg == "--settings" and index + 1 < len(before_separator):
            forwarded_settings.extend(before_separator[index : index + 2])
            index += 2
            continue
        if arg.startswith("--settings="):
            forwarded_settings.append(arg)
        else:
            remaining.append(arg)
        index += 1
    return [
        "claude",
        *forwarded_settings,
        *local_args,
        *remaining,
        *forwarded[separator:],
    ]


def _claude_local_env(
    base: str,
    key: str,
    entry: dict,
    extra_body: Optional[dict] = None,
) -> dict:
    """Build the local endpoint, cache, display, and compaction environment."""
    model_id = entry["id"]
    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": model_id,
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        # Per-tool countdown reminders change the system prefix on local models.
        "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "off",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_NO_FLICKER": "1",
    }
    window = entry.get("context_length") or entry.get("max_context_length")
    if window:
        # claude assumes 200k for an unknown model id and clamps AUTO_COMPACT_WINDOW to [100k, that].
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(window))
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(int(window))
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = "90"
    if extra_body:
        env["CLAUDE_CODE_EXTRA_BODY"] = json.dumps(extra_body)
    return env


def _codex_provider_table(base: str, key: Optional[str] = None) -> str:
    # The desktop app gets no shell env, so it needs the key itself.
    auth = (
        f"experimental_bearer_token = {json.dumps(key)}" if key else f'env_key = "{_CODEX_ENV_KEY}"'
    )
    return (
        f"{_PROVIDER_HEADER}\n"
        'name = "Unsloth Studio"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
        f"{auth}\n"
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        f"stream_idle_timeout_ms = {_CODEX_STREAM_IDLE_TIMEOUT_MS}\n"
    )


_CODEX_PROVIDER_TABLES = (_PROVIDER_HEADER, _PROVIDER_HEADER[:-1] + ".")


def _merge_codex_config(
    existing: str,
    base: str,
    key: Optional[str] = None,
) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)
    if key is None and not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{_CODEX_PROFILE}"\n'
    text = "".join(c for c in chunks if not c.startswith(_CODEX_PROVIDER_TABLES))
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + _codex_provider_table(base, key)


# Aligned with Codex's unknown-model fallback; copied from openai/codex rust-v0.144.0
# models-manager/prompt.md (Apache-2.0).
_CODEX_FALLBACK_PROMPT = Path(__file__).parent.parent / "codex_fallback_prompt.md"
_CODEX_MODEL_CATALOG_MIN_VERSION = (0, 110, 0)
_CODEX_PATCH_LINE_ENDINGS_MIN_VERSION = (0, 148, 0)
# Older Codex sends no reasoning for a model without reasoning summaries; older Pi has no samplingParams.
_CODEX_REASONING_REQUEST_MIN_VERSION = (0, 145, 0)
_PI_SAMPLING_PARAMS_MIN_VERSION = (0, 84, 0)


def _codex_executable_version(executable: str) -> Optional[tuple[int, int, int]]:
    try:
        output = subprocess.check_output(
            [executable, "--version"],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            stderr = subprocess.DEVNULL,
            env = _probe_env(),
        )
    except Exception:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return tuple(int(part) for part in match.groups()) if match else None


def _codex_supports_model_catalog() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        # A --no-launch recipe may run elsewhere; assume a current Codex.
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_MODEL_CATALOG_MIN_VERSION


def _agent_version_at_least(command: str, minimum: tuple) -> bool:
    executable = _which_with_install_dirs(command)
    if executable is None:
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= minimum


def _codex_supports_patch_line_endings() -> bool:
    executable = _which_with_install_dirs("codex")
    if executable is None:
        return True
    version = _codex_executable_version(executable)
    return version is not None and version >= _CODEX_PATCH_LINE_ENDINGS_MIN_VERSION


def _codex_model_catalog(model: dict, visibility: str = "none") -> dict:
    """Return conservative metadata for an Unsloth model unknown to Codex's built-in catalog."""
    model_id = model["id"]
    window = model.get("context_length") or model.get("max_context_length")
    entry = {
        "slug": model_id,
        "display_name": model_id,
        "description": "Model served by Unsloth Studio",
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": visibility,
        "supported_in_api": True,
        "priority": 99,
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": _CODEX_FALLBACK_PROMPT.read_text(encoding = "utf-8"),
        "supports_reasoning_summaries": False,
        "supports_reasoning_summary_parameter": False,
        "support_verbosity": False,
        "default_verbosity": None,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10_000},
        "supports_parallel_tool_calls": False,
        "experimental_supported_tools": [],
    }
    if window:
        entry["context_window"] = int(window)
        entry["max_context_window"] = int(window)
    return {"models": [entry]}


def write_codex_config(
    base: str,
    model: dict,
    home: Path,
    reasoning_effort: Optional[str] = None,
) -> None:
    home.mkdir(parents = True, exist_ok = True)

    config = home / "config.toml"
    existing = config.read_text(encoding = "utf-8") if config.exists() else ""
    merged = _merge_codex_config(existing, base)
    if merged != existing:
        config.write_text(merged, encoding = "utf-8")
        typer.echo(f"Updated {config}")

    # codex --oss picks the provider from oss_provider; the profile must beat a user-set value.
    profile_text = (
        f'oss_provider = "{_CODEX_PROFILE}"\n'
        f'model_provider = "{_CODEX_PROFILE}"\n'
        f"model = {json.dumps(model['id'])}\n"
    )
    if _codex_supports_patch_line_endings():
        profile_text += (
            "suppress_unstable_features_warning = true\n"
            "features.apply_patch_preserve_line_endings = true\n"
        )
    if _codex_supports_model_catalog() and _CODEX_FALLBACK_PROMPT.is_file():
        catalog = home / "model-catalog.json"
        catalog_text = json.dumps(_codex_model_catalog(model), indent = 2) + "\n"
        if not catalog.exists() or catalog.read_text(encoding = "utf-8") != catalog_text:
            catalog.write_text(catalog_text, encoding = "utf-8")
            typer.echo(f"Updated {catalog}")
        # Relative to the profile file, which survives WSL launching a Windows Codex.
        profile_text += f"model_catalog_json = {json.dumps(catalog.name)}\n"

    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
    if reasoning_effort:
        profile_text += f"model_reasoning_effort = {json.dumps(reasoning_effort)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def write_codex_subagent_bridge(
    base: str,
    key: str,
    model: dict,
    home: Path,
    *,
    yolo: bool,
    reasoning_effort: Optional[str] = None,
) -> Path:
    """Write private config for an explicit local Codex child launched through MCP."""
    child_home = home / "child"
    write_codex_config(base, model, child_home, reasoning_effort)
    path = home / "subagent.json"
    _write_private_json(
        path,
        {
            "api_key": key,
            "codex_home": str(child_home),
            "bypass_permissions": yolo,
        },
    )
    return path


def _wsl_windows_user_profile(executable: str) -> Path:
    """Return the Windows user profile as a path accessible from WSL."""
    profile = os.environ.get("USERPROFILE", "").strip()
    if not profile:
        try:
            profile = subprocess.check_output(
                ["cmd.exe", "/d", "/c", "echo %USERPROFILE%"],
                text = True,
                encoding = "utf-8",
                # The path is the value: a corrupted home is worse than a loud failure.
                errors = "strict",
                stderr = subprocess.DEVNULL,
                cwd = str(Path(executable).parent),
            ).strip()
        except UnicodeDecodeError as exc:
            _fail(
                f"Could not read the Windows user profile for Codex ({exc}); "
                "set USERPROFILE in the WSL environment, for example through WSLENV."
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            _fail(f"Could not find the Windows user profile for Codex: {exc}")
    if not profile or profile == "%USERPROFILE%":
        _fail("Could not find the Windows user profile for Codex.")
    if profile.startswith("/"):
        return Path(profile)
    try:
        translated = subprocess.check_output(
            ["wslpath", "-u", profile],
            text = True,
            encoding = "utf-8",
            errors = "replace",
            stderr = subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate Windows user profile {profile}: {exc}")
    if not translated:
        _fail(f"Could not translate Windows user profile {profile}.")
    return Path(translated)


def _codex_source_home(*, ignore_configured: bool = False) -> Path:
    configured = None if ignore_configured else os.environ.get("CODEX_HOME")
    if configured:
        if _wsl_windows_executable(["codex"]) and _looks_like_path(configured):
            if not configured.startswith("/"):
                try:
                    configured = subprocess.check_output(
                        ["wslpath", "-u", configured],
                        text = True,
                        encoding = "utf-8",
                        errors = "replace",
                        stderr = subprocess.DEVNULL,
                    ).strip()
                except (OSError, subprocess.CalledProcessError) as exc:
                    _fail(f"Could not translate Windows CODEX_HOME {configured}: {exc}")
                if not configured:
                    _fail("Could not translate Windows CODEX_HOME.")
        return Path(configured).expanduser()
    executable = _wsl_windows_executable(["codex"])
    if executable:
        return _wsl_windows_user_profile(executable) / ".codex"
    return Path.home() / ".codex"


def _is_junction(path: Path) -> bool:
    # Path.is_junction() was added in Python 3.12.
    if hasattr(path, "is_junction"):
        return path.is_junction()
    try:
        return (
            getattr(os.lstat(path), "st_reparse_tag", None) == 0xA0000003
        )  # IO_REPARSE_TAG_MOUNT_POINT
    except OSError:
        return False


def _is_directory_link(path: Path) -> bool:
    # lstat reads the link, so FILE_ATTRIBUTE_DIRECTORY answers even when dangling.
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & 0x10)
    except OSError:
        return False


def _remove_overlay_entry(path: Path) -> None:
    if _is_junction(path):
        path.rmdir()
    elif os.name == "nt" and path.is_symlink() and _is_directory_link(path):
        # DeleteFileW refuses a directory entry; rmdir drops the link.
        path.rmdir()
    elif path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _create_directory_junction(source: Path, target: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(target), str(source)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 30,
            check = False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def write_codex_parent_overlay(overlay: Path) -> Path:
    """Add local-agent routing without replacing the cloud parent's configuration."""
    overlay.mkdir(parents = True, exist_ok = True, mode = 0o700)

    manifest_path = overlay / _CODEX_PARENT_OVERLAY_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        manifest = None
    source_home = _codex_source_home()
    overlay_key = str(overlay.resolve(strict = False))
    source_key = str(source_home.resolve(strict = False))
    if source_key == overlay_key:
        previous_source = manifest.get("source_home") if isinstance(manifest, dict) else None
        if isinstance(previous_source, str) and previous_source:
            candidate = Path(previous_source).expanduser()
            if str(candidate.resolve(strict = False)) != overlay_key:
                source_home = candidate
            else:
                source_home = _codex_source_home(ignore_configured = True)
        else:
            source_home = _codex_source_home(ignore_configured = True)
        source_key = str(source_home.resolve(strict = False))
    same_source = isinstance(manifest, dict) and manifest.get("source_home") == source_key
    if same_source:
        managed_entries = manifest.get("entries", [])
        if not isinstance(managed_entries, list):
            managed_entries = []
        for name in managed_entries:
            if isinstance(name, str) and name not in {"", ".", ".."} and Path(name).name == name:
                _remove_overlay_entry(overlay / name)
    else:
        # A reused overlay must never mix two Codex homes; legacy overlays have no manifest, so rebuild.
        for target in list(overlay.iterdir()):
            _remove_overlay_entry(target)

    # Symlinks, else directory junctions, keep user state shared; copy only when both are unavailable.
    fallback_dirs = {"agents", "skills", "rules", "plugins", "marketplaces", "sessions"}
    entries = []
    if source_home.is_dir():
        for source in source_home.iterdir():
            if source.name in {
                "AGENTS.md",
                "AGENTS.override.md",
                _CODEX_PARENT_OVERLAY_MANIFEST,
            }:
                continue
            target = overlay / source.name
            _remove_overlay_entry(target)
            try:
                target.symlink_to(source, target_is_directory = source.is_dir())
                entries.append(source.name)
            except OSError:
                if source.is_file():
                    shutil.copy2(source, target)
                    entries.append(source.name)
                elif source.is_dir():
                    if _create_directory_junction(source, target):
                        entries.append(source.name)
                    elif source.name in fallback_dirs:
                        shutil.copytree(source, target)
                        entries.append(source.name)

    _write_private_json(
        manifest_path,
        {"source_home": source_key, "entries": sorted(entries)},
    )

    inherited = ""
    instruction_name = "AGENTS.md"
    for candidate in (source_home / "AGENTS.override.md", source_home / "AGENTS.md"):
        try:
            text = candidate.read_text(encoding = "utf-8")
        except FileNotFoundError:
            continue
        except OSError as exc:
            _fail(f"Could not preserve Codex instructions from {candidate}: {exc}")
        if text.strip():
            inherited = text.rstrip()
            instruction_name = candidate.name
            break

    other_name = "AGENTS.md" if instruction_name == "AGENTS.override.md" else "AGENTS.override.md"
    other = overlay / other_name
    if other.is_file() or other.is_symlink():
        other.unlink()
    routing = _CODEX_SUBAGENT_ROUTING_INSTRUCTIONS
    combined = f"{inherited}\n\n{routing}\n" if inherited else f"{routing}\n"
    _write_private_text(overlay / instruction_name, combined)
    return overlay


def _agent_config_path(path: Path, command: list) -> str:
    """Translate a generated config path when a Windows agent runs through WSL."""
    return _wsl_windows_path(path) if _wsl_windows_executable(command) else str(path)


def _opencode_subagent_inline_config(
    path: Path,
    permission: dict,
    command: str = "opencode",
    v2: bool = False,
) -> dict:
    """Keep the local provider visible without hiding the parent's allowed providers."""
    inline: dict = {}
    inherited = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if inherited:
        try:
            parsed = json.loads(inherited)
        except ValueError:
            _fail("OPENCODE_CONFIG_CONTENT is not valid JSON.")
        if not isinstance(parsed, dict):
            _fail("OPENCODE_CONFIG_CONTENT must contain a JSON object.")
        inline.update(parsed)

    def merge_provider_filters(effective_config: dict) -> None:
        enabled = effective_config.get("enabled_providers")
        if isinstance(enabled, list):
            inherited_enabled = inline.get("enabled_providers")
            if not isinstance(inherited_enabled, list):
                inherited_enabled = []
            providers = [
                provider
                for provider in [*inherited_enabled, *enabled]
                if provider != _OPENCODE_PROVIDER
            ]
            inline["enabled_providers"] = list(dict.fromkeys([*providers, _OPENCODE_PROVIDER]))
        disabled = effective_config.get("disabled_providers")
        if isinstance(disabled, list) and _OPENCODE_PROVIDER in disabled:
            inline["disabled_providers"] = [
                provider for provider in disabled if provider != _OPENCODE_PROVIDER
            ]

    # V2 makes these filters policies that global/project rules still outrank; launch says so.
    merge_provider_filters(inline)
    effective = inline

    executable = None if v2 else _which_with_install_dirs(command)
    if v2:
        legacy_depth = inline.pop("subagent_depth", None)
        if (
            isinstance(legacy_depth, int)
            and not isinstance(legacy_depth, bool)
            and legacy_depth > 0
        ):
            experimental = inline.get("experimental")
            if not isinstance(experimental, dict):
                experimental = {}
            experimental.setdefault("subagent_depth", legacy_depth)
            inline["experimental"] = experimental
    elif executable is None:
        typer.echo(
            f"Warning: OpenCode is not installed, so provider filters could not be checked. "
            f"The target configuration must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    else:
        env = _probe_env(OPENCODE_CONFIG = _agent_config_path(path, [command]))
        try:
            resolved = subprocess.run(
                [executable, "debug", "config"],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 15,
                env = env,
            )
        except Exception as exc:
            _fail(f"Could not inspect OpenCode provider filters: {exc}")
        if resolved.returncode != 0:
            detail = resolved.stderr.strip() or resolved.stdout.strip()
            _fail(f"Could not inspect OpenCode provider filters: {detail or 'unknown error'}")
        try:
            effective = json.loads(resolved.stdout)
        except ValueError:
            _fail("Could not inspect OpenCode provider filters: invalid JSON response.")
        if not isinstance(effective, dict):
            _fail("Could not inspect OpenCode provider filters: expected a JSON object.")

        merge_provider_filters(effective)

    if not v2:
        depth = effective.get("subagent_depth")
        inline["subagent_depth"] = (
            depth if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0 else 1
        )
    if permission:
        inline["permission"] = permission
    return inline


def _b64_path(path: Path) -> str:
    """Path as base64, so it can cross a shell without being expanded."""
    return base64.b64encode(str(path).encode("utf-8")).decode("ascii")


_CLAUDE_PLAN_GATE_SCRIPT = '''\
"""Deny the editing agent while the parent session is in plan mode."""
import json, sys

try:
    mode = (json.load(sys.stdin) or {}).get("permission_mode")
except Exception:
    sys.exit(0)  # fail open: a hook error must never block the parent session
if mode == "plan":
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            "Plan mode is active. Call the read-only Unsloth plan agent "
            "(unsloth_plan_agent) instead of unsloth_agent."
        ),
    }}))
sys.exit(0)
'''


def write_claude_subagent_plugin(path: Path, server_env: dict) -> Path:
    """Write a session plugin that exposes the local Claude child through MCP."""
    plugin = path / "unsloth-local-agent"
    command = sys.executable
    args = ["-m", _CLAUDE_SUBAGENT_MCP_MODULE]
    mcp_env = dict(server_env)
    base = server_env.get("UNSLOTH_CLAUDE_SUBAGENT_BASE_URL")
    key = server_env.get("UNSLOTH_CLAUDE_SUBAGENT_API_KEY")
    model_id = server_env.get("UNSLOTH_CLAUDE_SUBAGENT_MODEL")
    if base and key and model_id:
        entry = {
            "id": model_id,
            "context_length": int(
                server_env.get("UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW", "0") or 0
            ),
        }
        local_env = _claude_local_env(base, key, entry)
        if "CLAUDE_CODE_EXTRA_BODY" in server_env:
            local_env["CLAUDE_CODE_EXTRA_BODY"] = server_env["CLAUDE_CODE_EXTRA_BODY"]
        settings = _write_claude_settings(plugin, model_id, local_env)
        mcp_env[_CLAUDE_SUBAGENT_SETTINGS_ENV] = str(settings)
    if _wsl_windows_executable(["claude"]):
        command = "wsl.exe"
        args = [
            "-d",
            os.environ["WSL_DISTRO_NAME"],
            "--",
            sys.executable,
            "-m",
            _CLAUDE_SUBAGENT_MCP_MODULE,
        ]
        mcp_env["WSLENV"] = _merge_wslenv(
            os.environ.get("WSLENV", ""),
            (
                *_wsl_bridge_names(server_env, ()),
                *([_CLAUDE_SUBAGENT_SETTINGS_ENV] if base and key and model_id else []),
            ),
        )
    _write_private_json(
        plugin / ".claude-plugin" / "plugin.json",
        {
            "name": "unsloth-local-agent",
            "version": "1.0.0",
            "description": _SUBAGENT_DESCRIPTION,
            "author": {"name": "Unsloth AI"},
        },
    )
    _write_private_json(
        plugin / ".mcp.json",
        {
            "mcpServers": {
                "unsloth": {
                    "type": "stdio",
                    "command": command,
                    "args": args,
                    "env": mcp_env,
                }
            }
        },
    )
    # Replace plan mode's dead end with a reason naming the read-only tool. Skipped under the WSL bridge.
    gate = plugin / "hooks" / "plan_gate.py"
    if command == "wsl.exe":
        # A persisted plugin dir may still hold a gate from an earlier non-WSL run.
        for stale in (gate, plugin / "hooks" / "hooks.json"):
            stale.unlink(missing_ok = True)
    else:
        _write_private_text(gate, _CLAUDE_PLAN_GATE_SCRIPT)
        _write_private_json(
            plugin / "hooks" / "hooks.json",
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": _CLAUDE_SUBAGENT_TOOL,
                            "hooks": [
                                {
                                    "type": "command",
                                    # runpy: a missing gate is exit 1 (fails open), not exit 2 (blocks every tool). base64 because the
                                    # path goes through a shell ($(..), backticks, %VAR%).
                                    "command": (
                                        f'"{sys.executable}" -c '
                                        f'"import base64,runpy; runpy.run_path('
                                        f"base64.b64decode('{_b64_path(gate)}').decode())\""
                                    ),
                                    # A hook with no timeout stalls the parent unbounded.
                                    "timeout": 10,
                                }
                            ],
                        }
                    ]
                }
            },
        )
    skill = plugin / "skills" / "local-agent" / "SKILL.md"
    skill.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    skill.write_text(
        "---\n"
        "description: Delegate a task to the local agent powered by Unsloth. Use when the "
        "user asks to spawn an Unsloth agent or local agent.\n"
        "---\n\n"
        "Call the Unsloth local agent tool once with the complete task. In plan mode, call "
        "the read-only Unsloth plan agent instead. Return its result to the user without "
        "claiming that the cloud parent completed the local work.\n",
        encoding = "utf-8",
    )
    return plugin


def _codex_subagent_flags(path: Path) -> list[str]:
    command = sys.executable
    package_root = str(Path(__file__).resolve().parents[2])
    bootstrap = (
        f"import sys;sys.path.insert(0,{json.dumps(package_root)});"
        f"from {_CODEX_SUBAGENT_MCP_MODULE} import main;main()"
    )
    args = ["-c", bootstrap, str(path)]
    if _wsl_windows_executable(["codex"]):
        command = "wsl.exe"
        args = [
            "-d",
            os.environ["WSL_DISTRO_NAME"],
            "--",
            sys.executable,
            "-c",
            bootstrap,
            str(path),
        ]
    server = (
        "{ "
        f"command = {json.dumps(command)}, "
        f"args = {json.dumps(args)}, "
        f"required = true, enabled_tools = [{json.dumps(_CODEX_SUBAGENT_MCP_TOOL)}], "
        'default_tools_approval_mode = "approve", '
        "startup_timeout_sec = 15, tool_timeout_sec = 3600 }"
    )
    return ["-c", f"mcp_servers.{_CODEX_SUBAGENT_MCP_SERVER}={server}"]


def _wsl_windows_executable(command: list) -> Optional[str]:
    if os.name == "nt" or not os.environ.get("WSL_DISTRO_NAME"):
        return None
    executable = shutil.which(command[0])
    if executable and executable.startswith("/mnt/"):
        return executable
    return None


def _wsl_windows_path(path: Path) -> str:
    try:
        translated = subprocess.check_output(
            ["wslpath", "-w", str(path)], text = True, encoding = "utf-8", errors = "replace"
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _fail(f"Could not translate WSL path {path}: {exc}")
    if not translated:
        _fail(f"Could not translate WSL path {path}")
    return translated


def _looks_like_path(value: str) -> bool:
    # Only filesystem paths get the WSLENV /p flag; scalars must pass untranslated.
    return bool(value) and (value.startswith(("/", "\\")) or (len(value) >= 2 and value[1] == ":"))


def _wsl_bridge_names(env: dict, unset_env: tuple) -> tuple:
    names = [name + ("/p" if _looks_like_path(value) else "") for name, value in env.items()]
    names.extend(unset_env)
    return tuple(dict.fromkeys(names))


def _merge_wslenv(current: str, names: tuple) -> str:
    # Our entries win: WSLENV ignores a duplicate name, so upgrade a bare "HOME" to "HOME/p".
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
    # PowerShell single quotes are literal; list2cmdline's escaping is cmd.exe syntax.
    if arg and re.fullmatch(r"[A-Za-z0-9_./:=+-]+", arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def _print_env(
    env: dict,
    command: list,
    unset_env: tuple = (),
    wsl_env_bridge: tuple = (),
    cwd_env: tuple = (),
) -> None:
    if os.name == "nt":
        for name in unset_env:
            typer.echo(f"Remove-Item Env:{name} -ErrorAction SilentlyContinue")
        for name, value in env.items():
            # PowerShell: ` is the escape char, and $ triggers expansion inside "".
            escaped = value.replace("`", "``").replace('"', '`"').replace("$", "`$")
            typer.echo(f'$env:{name} = "{escaped}"')
        for name in cwd_env:
            typer.echo(f"$env:{name} = (Get-Location).Path")
        typer.echo(" ".join(_powershell_quote(arg) for arg in command))
        return
    for name in unset_env:
        typer.echo(f"export {name}=" if wsl_env_bridge else f"unset {name}")
    for name, value in env.items():
        typer.echo(f"export {name}={shlex.quote(value)}")
    for name in cwd_env:
        typer.echo(f'export {name}="$PWD"')
    if wsl_env_bridge:
        typer.echo(
            f"export WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    # The last line is self-contained (inline env): people copy just it, and a bare command would run
    # against their real credentials.
    inline = [f"{name}=" for name in unset_env]
    inline += [f"{name}={shlex.quote(value)}" for name, value in env.items()]
    inline += [f'{name}="$PWD"' for name in cwd_env]
    if wsl_env_bridge:
        inline.append(
            f"WSLENV={shlex.quote(_merge_wslenv(os.environ.get('WSLENV', ''), wsl_env_bridge))}"
        )
    typer.echo(" ".join((*inline, shlex.join(command))))


def _refresh_windows_path() -> None:
    # Registry PATH after the process PATH so a fresh install is visible without changing precedence.
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


def _managed_node_tools() -> Optional[tuple[Path, Path, bool]]:
    # Best-effort: any failure means "no managed Node", never a broken launch.
    try:
        # Discovery only: must not create the cache tree (the launch may target a remote server).
        ensure_studio_backend_path(seed_cache_env = False)
        from utils.node_runtime import managed_node_binary, resolve_node_executable
        node = Path(managed_node_binary())
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None
    npm = node.with_name("npm.cmd" if os.name == "nt" else "npm")
    try:
        usable = node.is_file() and npm.is_file()
        if os.name != "nt":
            usable = usable and os.access(node, os.X_OK) and os.access(npm, os.X_OK)
    except OSError:
        return None
    if not usable:
        return None
    try:
        resolved = resolve_node_executable()
        preferred = bool(resolved) and Path(resolved).resolve() == node.resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        preferred = False
    return node, npm, preferred


def _augment_path_with_install_dirs() -> None:
    # User dirs appended, managed Node prepended; a missing home must not drop the managed Node.
    try:
        home = Path.home()
    except (RuntimeError, OSError):
        home = None
    candidates = [home / ".local" / "bin", home / ".opencode" / "bin"] if home is not None else []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm")
    managed_node = _managed_node_tools()
    if managed_node is not None and not managed_node[2]:
        candidates.append(managed_node[0].parent)
    current = os.environ.get("PATH")
    if current is None:
        # PATH unset falls back to os.defpath like shutil.which; an empty PATH means search nothing.
        current = os.defpath
    preferred_node = (
        str(managed_node[0].parent) if managed_node is not None and managed_node[2] else None
    )
    if preferred_node:
        preferred_key = os.path.normcase(preferred_node)
        current = os.pathsep.join(
            entry
            for entry in current.split(os.pathsep)
            if not entry or os.path.normcase(entry) != preferred_key
        )
    seen = {os.path.normcase(entry) for entry in current.split(os.pathsep) if entry}
    additions = [
        str(directory)
        for directory in candidates
        if directory.is_dir() and os.path.normcase(str(directory)) not in seen
    ]
    if preferred_node or additions:
        parts = [part for part in (preferred_node, current, *additions) if part]
        os.environ["PATH"] = os.pathsep.join(parts)


def _probe_env(**extra: str) -> dict:
    """Environment for probes that RUN a resolved shim. _which_with_install_dirs restores PATH before returning, so a shim backed by Unsloth's managed Node would not find that node when executed."""
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    env = os.environ.copy()
    if original is None:
        os.environ.pop("PATH", None)
    else:
        os.environ["PATH"] = original
    env.update(extra)
    return env


def _prefer_windows_cmd_sibling(executable: Optional[str]) -> Optional[str]:
    """Prefer the sibling .cmd when Windows resolved an extensionless npm/pnpm shim. cmd-shim writes ``to``, ``to.cmd`` and ``to.ps1``, and shutil.which can return the extensionless POSIX shim, which CreateProcess rejects with WinError 193. Measured on windows-latest: 3.12.0 probes the bare name before PATHEXT (gh-109590) and 3.12.1 onwards do not, and a PATHEXT holding "." reaches the same place on any version. Substituted only when the file opens with a shebang, so a real PE keeps priority over a stale wrapper beside it; matched on not-a-Windows-suffix so a dotted bin name is caught too."""
    if executable is None or os.name != "nt":
        return executable
    if Path(executable).suffix.lower() in {".exe", ".com", ".cmd", ".bat", ".ps1"}:
        return executable
    with contextlib.suppress(OSError):
        with open(executable, "rb") as resolved_file:
            if resolved_file.read(2) == b"#!":
                # Only matters on case-sensitive volumes; no writer emits .bat.
                for extension in (".cmd", ".CMD"):
                    sibling = Path(executable + extension)
                    if sibling.is_file():
                        return str(sibling)
    return executable


def _which_with_install_dirs(name: str) -> Optional[str]:
    # shutil.which(name) over the agent install dirs too, so the version probe matches what _launch()
    # runs. PATH is restored afterwards.
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    try:
        # Callers spawn this result directly, so the shim rescue is needed here too.
        return _prefer_windows_cmd_sibling(shutil.which(name))
    finally:
        if original is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = original


def _which_deepseek_harness_with_install_dirs() -> Optional[str]:
    """Find the first valid DeepSeek Harness even when another ``dsh`` shadows it."""
    original = os.environ.get("PATH")
    _augment_path_with_install_dirs()
    try:
        for executable in deepseek_harness_executables_on_path():
            executable = _prefer_windows_cmd_sibling(executable)
            if executable is not None and is_deepseek_harness_executable(executable):
                return executable
        return None
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


def _npm_executable() -> Optional[str]:
    managed_node = _managed_node_tools()
    if not (managed_node and managed_node[2]):
        executable = _prefer_windows_cmd_sibling(shutil.which("npm"))
        if executable and not _wsl_windows_executable([executable]):
            return executable
        if executable:
            # WSL inherits the Windows PATH, so the rejected shim may shadow a native npm.
            for directory in os.get_exec_path():
                candidate = _prefer_windows_cmd_sibling(shutil.which("npm", path = directory))
                if candidate and not _wsl_windows_executable([candidate]):
                    return candidate

    return str(managed_node[1]) if managed_node is not None else None


def _install_command(install_hint: str) -> tuple[list[str], Optional[dict]]:
    if not re.match(r"^\s*npm(?:\s|$)", install_hint):
        if os.name == "nt":
            return (
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    install_hint,
                ],
                None,
            )
        return ["/bin/sh", "-c", install_hint], None

    npm = _npm_executable()
    if npm is None:
        _fail(
            "npm is required to install this agent, but no native system npm or usable "
            "Unsloth-managed Node installation was found. Install Node.js with npm, "
            "then re-run."
        )
    args = shlex.split(install_hint)
    env = dict(os.environ)
    # dirname, not Path().parent: tests override os.name. Empty must not put cwd on PATH.
    npm_dir = os.path.dirname(npm)
    current_path = env.get("PATH", "")
    if npm_dir:
        env["PATH"] = os.pathsep.join([npm_dir, current_path]) if current_path else npm_dir
    if os.name == "nt":
        command = "& " + " ".join(_powershell_quote(arg) for arg in [npm, *args[1:]])
        return (
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            env,
        )
    return [npm, *args[1:]], env


def _install_agent(name: str, install_hint: str) -> Optional[str]:
    # Consent-based install; no TTY or a decline returns None and the caller prints the hint.
    if not sys.stdin.isatty():
        return None
    typer.echo(f"`{name}` is not installed.")
    # Name the vendor installer source before the prompt: nothing checks its signature.
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
    install_command, install_env = _install_command(install_hint)
    try:
        result = subprocess.run(install_command, env = install_env)
    except OSError as exc:
        _fail(
            f"Could not run the install command: {exc}. "
            f"Run it yourself, then re-run: {install_hint}"
        )
    if result.returncode != 0:
        message = f"Install command failed. Run it yourself, then re-run: {install_hint}"
        if os.name == "nt":
            message += (
                "\nIf it fails because running scripts is disabled (PSSecurityException), "
                "allow local scripts for your user, then retry:\n"
                "  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned"
            )
        _fail(message)
    _refresh_windows_path()
    _augment_path_with_install_dirs()
    executable = shutil.which(name)
    if executable is None:
        _fail(
            f"`{name}` installed but isn't on PATH yet. Open a new shell (or add it to "
            f"PATH), then re-run. Install command: {install_hint}"
        )
    return executable


def _resolve_or_install_agent(name: str, install_hint: str, resolver) -> str:
    executable = resolver(name)
    invalid_executable = None
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    executable = _install_agent(name, install_hint)
    if executable is not None:
        if name != "dsh" or is_deepseek_harness_executable(executable):
            return executable
        invalid_executable = executable
    if name == "dsh":
        executable = _which_deepseek_harness_with_install_dirs()
        if executable is not None:
            return executable

    if invalid_executable is not None:
        _fail(
            f"`{invalid_executable}` is not DeepSeek Harness. Install DeepSeek Harness "
            f"with: {install_hint}"
        )
    _fail(f"`{name}` not found on PATH. Install it with: {install_hint}")


def _require_agent_for_launch(name: str, install_hint: str, launch: bool) -> Optional[str]:
    if not launch:
        return None
    return _resolve_or_install_agent(name, install_hint, _which_with_install_dirs)


def _wsl_shim_env(
    command: list,
    env: dict,
    unset_env: tuple,
    cwd_env: tuple = (),
) -> tuple[dict, tuple]:
    if not _wsl_windows_executable(command):
        return env, ()
    wsl_env_bridge = _wsl_bridge_names(env, unset_env)
    if not wsl_env_bridge and not cwd_env:
        return env, ()
    # Bridge PWD via WSLENV (PWD/p) so the Windows shim reads the live cwd; never freeze env["PWD"].
    return env, tuple(dict.fromkeys((*wsl_env_bridge, *(f"{name}/p" for name in cwd_env), "PWD/p")))


_NPM_CMD_SHIM_HEAD = (
    "@ECHO off\n"
    "GOTO start\n"
    ":find_dp0\n"
    "SET dp0=%~dp0\n"
    "EXIT /b\n"
    ":start\n"
    "SETLOCAL\n"
    "CALL :find_dp0\n"
)
_NPM_NODE_CMD_SHIM_PREFIX = (
    re.escape(_NPM_CMD_SHIM_HEAD)
    + r"(?P<environment>(?:@SET [^=\r\n]+=[^\r\n]+\n)*)"
    + re.escape(
        '\nIF EXIST "%dp0%\\node.exe" (\n'
        + '  SET "_prog=%dp0%\\node.exe"\n'
        + ") ELSE (\n"
        + '  SET "_prog=node"\n'
    )
)
_NPM_NODE_CMD_SHIM_SUFFIX = (
    r"(?P<node_args>[^\r\n]*?)[ \t]+" + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*'
)
_NPM_NODE_CMD_SHIMS = (
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            "  SET PATHEXT=%PATHEXT:;.JS;=;%\n"
            + ")\n\n"
            + 'endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
    re.compile(
        _NPM_NODE_CMD_SHIM_PREFIX
        + re.escape(
            ")\n\n"
            + "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "
            + "set PATHEXT=%PATHEXT:;.JS;=;% & "
            + '"%_prog%"'
        )
        + _NPM_NODE_CMD_SHIM_SUFFIX,
        re.IGNORECASE,
    ),
)
_NPM_NATIVE_CMD_SHIM = re.compile(
    re.escape(_NPM_CMD_SHIM_HEAD) + r'"%dp0%\\(?P<target>[^"\r\n]+)"[ \t]+%\*',
    re.IGNORECASE,
)
_NPM_NODE_SHEBANG = re.compile(
    r"^#!\s*(?:/usr/bin/env\s+(?:-S\s+)?((?:[^ \t=]+=[^ \t=]+\s+)*))?([^ \t]+)(.*)$"
)
_NPM_SHEBANG_DOLLAR = re.compile(r"\$\{?([^$@#?\- \t{}:]+)\}?")


def _npm_batch_environment(declarations: str) -> str:
    lines = []
    for declaration in declarations.split():
        name, separator, value = declaration.partition("=")
        name = name.strip()
        value = value.strip()
        if separator and name and value:
            value = _NPM_SHEBANG_DOLLAR.sub(lambda match: f"%{match.group(1)}%", value)
            lines.append(f"@SET {name}={value}\n")
    return "".join(lines)


def _windows_expand_environment(value: str, environment: dict) -> str:
    folded = {name.casefold(): item for name, item in environment.items()}
    return re.sub(
        r"%([^%\r\n]+)%",
        lambda match: folded.get(match.group(1).casefold(), ""),
        value,
    )


def _npm_node_shim_metadata(target: Path, match, environment: dict) -> Optional[tuple]:
    environment_block = match.group("environment") or ""
    node_args_text = (match.group("node_args") or "").strip()
    known_node_suffix = target.suffix.lower() in {".js", ".cjs", ".mjs"}
    if not environment_block and not node_args_text and known_node_suffix:
        return [], {}

    first_line = target.read_text(encoding = "utf-8").splitlines()[0]
    shebang = _NPM_NODE_SHEBANG.fullmatch(first_line)
    if shebang is None or Path(shebang.group(2)).name.casefold() not in {"node", "node.exe"}:
        return None
    declarations = shebang.group(1) or ""
    if _npm_batch_environment(declarations).casefold() != environment_block.casefold():
        return None
    if (shebang.group(3) or "").strip() != node_args_text:
        return None
    try:
        node_args = shlex.split(node_args_text) if node_args_text else []
    except ValueError:
        return None

    updates = {}
    expanded_environment = dict(environment)
    for line in environment_block.splitlines():
        name, value = line.removeprefix("@SET ").split("=", 1)
        expanded = _windows_expand_environment(value, expanded_environment)
        expanded_environment[name] = expanded
        updates[name] = expanded
    return node_args, updates


def _apply_windows_environment(environment: dict, updates: dict) -> None:
    for name, value in updates.items():
        existing = next((key for key in environment if key.casefold() == name.casefold()), None)
        if existing is not None and existing != name:
            del environment[existing]
        environment[name] = value


def _resolved_launch_command(
    executable: str,
    arguments: list,
    environment: Optional[dict] = None,
) -> list:
    """Return an argv that preserves arguments through standard Windows npm shims."""
    # _launch resolves with raw shutil.which, so rescue here too.
    executable = _prefer_windows_cmd_sibling(executable)
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        # cmd.exe splits CR/LF in `%*` and PowerShell rewrites quotes; match complete cmd-shim templates.
        with contextlib.suppress(OSError, UnicodeError, IndexError):
            shim = Path(executable)
            contents = shim.read_text(encoding = "utf-8").replace("\r\n", "\n").strip()
            for pattern in _NPM_NODE_CMD_SHIMS:
                match = pattern.fullmatch(contents)
                if match is None:
                    continue
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if not target.is_file() or not any(
                    part.casefold() == "node_modules" for part in target.parts
                ):
                    continue
                metadata = _npm_node_shim_metadata(target, match, environment or os.environ)
                if metadata is None:
                    continue
                node_args, environment_updates = metadata
                bundled_node = shim.parent / "node.exe"
                node = str(bundled_node) if bundled_node.is_file() else shutil.which("node.exe")
                if node:
                    if environment is not None:
                        _apply_windows_environment(environment, environment_updates)
                    return [node, *node_args, str(target), *arguments]

            match = _NPM_NATIVE_CMD_SHIM.fullmatch(contents)
            if match is not None:
                relative = Path(*re.split(r"[\\/]+", match.group("target")))
                target = (shim.parent / relative).resolve()
                if (
                    target.is_file()
                    and any(part.casefold() == "node_modules" for part in target.parts)
                    and target.suffix.lower() in {".exe", ".com"}
                ):
                    return [str(target), *arguments]
    return [executable, *arguments]


def _launch(
    command: list,
    env: dict,
    install_hint: str,
    unset_env: tuple = (),
    cwd_env: tuple = (),
) -> int:
    # Resolve install dirs first so an installed agent not on PATH is found.
    _augment_path_with_install_dirs()
    executable = _resolve_or_install_agent(command[0], install_hint, shutil.which)
    env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env, cwd_env)
    env = {**env, **{name: os.getcwd() for name in cwd_env}}
    child_env = dict(os.environ)
    if wsl_env_bridge:
        env = {**env, "PWD": os.getcwd()}
        child_env["WSLENV"] = _merge_wslenv(child_env.get("WSLENV", ""), wsl_env_bridge)
        for name in unset_env:
            child_env[name] = ""
    else:
        for name in unset_env:
            child_env.pop(name, None)
    child_env.update(env)
    if os.name != "nt" and not wsl_env_bridge:
        # Some Node CLIs use PWD for project-root discovery instead of process.cwd().
        child_env["PWD"] = os.getcwd()
    # A no-op handler, not SIG_IGN: exec preserves an ignored signal but resets a caught one.
    previous = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        launch_command = _resolved_launch_command(executable, command[1:], child_env)
        code = subprocess.run(launch_command, env = child_env).returncode
    finally:
        signal.signal(signal.SIGINT, previous)
    # Killed by signal N; shells expect 128+N.
    return code if code >= 0 else 128 - code


def _connect(
    api_key: Optional[str],
    model: Optional[str],
    load: LoadOptions = LoadOptions(),
    *,
    serve: bool = False,
    launch: bool = True,
    server_options: ServerOptions = ServerOptions(),
    preload_check = None,
) -> tuple:
    # Split `org/name:QUANT` before matching, or a :-suffixed id would evict another session's model.
    if model:
        repo, variant = _split_repo_variant(model)
        if variant:
            model = repo
            if not load.gguf_variant:
                load = load._replace(gguf_variant = variant)
    base, server = _require_studio(
        model, load, serve = serve, launch = launch, server_options = server_options
    )
    try:
        key = _agent_api_key(base, api_key, auto_started = server is not None)
        # A server we just started serves exactly the requested model; only an attach can evict.
        entry = _resolve_model(
            base,
            key,
            None if server is not None else model,
            load,
            preload_check = None if server is not None else preload_check,
            # That server was started from these knobs; inferring would reload what was just loaded.
            infer_resident = server is None,
        )
        status = _inference_status(base, key) if model else {}
        # A GGUF can be active while the resolved entry is another resident model.
        if status.get("memory_warning") and any(
            _model_id_matches(
                (entry or {}).get("id"), status_id, allow_casefold = is_loopback_url(base)
            )
            for status_id in (status.get("active_model"), status.get("model_identifier"))
            if status_id
        ):
            typer.echo(f"Warning: {status['memory_warning']}", err = True)
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
    cwd_env: tuple = (),
) -> None:
    # Some agents (Pi) paint from the cursor assuming a clean screen. click.clear() is a no-op off a TTY.
    if launch and clear_screen:
        click.clear()
    typer.echo(f"Unsloth ready at {base} · model {entry['id']}")
    if not launch:
        env, wsl_env_bridge = _wsl_shim_env(command, env, unset_env, cwd_env)
        _print_env(
            env,
            command,
            unset_env = unset_env,
            wsl_env_bridge = wsl_env_bridge,
            cwd_env = cwd_env,
        )
        if _keep_auto_served():
            typer.echo(f"Unsloth Studio is still running at {base}.")
            typer.echo("Stop it with: unsloth studio stop")
        return
    try:
        code = _launch(
            command,
            env,
            install_hint = install_hint,
            unset_env = unset_env,
            cwd_env = cwd_env,
        )
    except BaseException:
        # Tear the server down rather than orphan it.
        _shutdown_auto_served()
        raise
    auto_started = _auto_served_server is not None
    kept = _keep_auto_served()
    if auto_started and not kept:
        typer.echo(f"The auto-started Unsloth server at {base} stopped during the session.")
        raise typer.Exit(code = code)
    if code:
        typer.echo(f"The agent exited with code {code}.")
    if is_loopback_url(base):
        typer.echo(f"Unsloth Studio is still running at {base}.")
        typer.echo("Stop it with: unsloth studio stop")
    else:
        typer.echo(f"The remote Unsloth server is still running at {base}.")
    raise typer.Exit(code = code)


def _agents_config_root() -> Path:
    return _studio_auth_root() / "agents"


@contextlib.contextmanager
def _temporary_agent_config(prefix: str):
    # Nothing else prunes Unsloth's auth tree; the lock spares live sessions.
    temp_root = _agents_config_root() / ".tmp"
    with contextlib.ExitStack() as stack:
        try:
            temp_root.mkdir(parents = True, exist_ok = True, mode = 0o700)
            path = stack.enter_context(_short_ephemeral_session(temp_root, prefix))
        except OSError:
            # Attaching needs no local auth tree; the OS prunes the system temp dir.
            path = Path(tempfile.mkdtemp(prefix = prefix))
            stack.callback(shutil.rmtree, path, ignore_errors = True)
        yield path


# codex-subagent nests CODEX_HOME under <home>/parent, so it needs the short root too.
_CODEX_SHORT_HOME_AGENTS = ("codex", "codex-subagent")


def _ephemeral_session_parent(agent: str) -> Optional[Path]:
    """Return a non-system-temp parent when an agent needs one."""
    if os.name != "nt" or agent not in _CODEX_SHORT_HOME_AGENTS:
        return None
    # Codex's nested plugin checkout can exceed Windows path limits under %TEMP%, and Codex refuses PATH
    # helpers below the system temp dir; keep the home short and private.
    root = Path.home() / ".unsloth" / ".tmp"
    root.mkdir(parents = True, exist_ok = True, mode = 0o700)
    return root


def _ephemeral_session_prefix(agent: str, parent: Optional[Path]) -> str:
    """Return the platform-specific prefix for an ephemeral agent home."""
    if agent in _CODEX_SHORT_HOME_AGENTS and parent is not None:
        return "u-codex-"
    return f"unsloth-{agent}-"


@contextlib.contextmanager
def _locked_file(path: Path, blocking: bool = True):
    """Yield whether an advisory lock was acquired for the first byte of path."""
    handle = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    if not blocking:
                        break
                    # LK_LOCK gives up after ~10s; poll LK_NBLCK so slow cleanup cannot fail a concurrent launch.
                    time.sleep(0.05)
        else:
            import fcntl
            mode = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(handle.fileno(), mode)
                acquired = True
            except BlockingIOError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _reclaim_stale_ephemeral_sessions(parent: Path, prefix: str) -> None:
    """Remove abandoned session homes while preserving locked live sessions."""
    for path in parent.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        active_lock = path / ".active.lock"
        try:
            modified = active_lock.stat().st_mtime if active_lock.exists() else path.stat().st_mtime
        except FileNotFoundError:
            continue
        # If only the wrapper is killed, its child may still use CODEX_HOME; wait a full day.
        if time.time() - modified < _CODEX_EPHEMERAL_STALE_SECONDS:
            continue
        try:
            with _locked_file(active_lock, blocking = False) as stale:
                pass
        except FileNotFoundError:
            continue
        if stale:
            shutil.rmtree(path, ignore_errors = True)


def _refresh_ephemeral_session_marker(path: Path, stop: threading.Event) -> None:
    """Keep the stale grace period relative to wrapper death, not session start."""
    while not stop.wait(_CODEX_EPHEMERAL_HEARTBEAT_SECONDS):
        with contextlib.suppress(OSError):
            os.utime(path, None)


@contextlib.contextmanager
def _short_ephemeral_session(parent: Path, prefix: str = "u-codex-"):
    """Create a session home whose lock makes crash cleanup concurrency-safe."""
    path = None
    active_lock = contextlib.ExitStack()
    heartbeat_stop = None
    heartbeat = None
    try:
        with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
            if not cleanup_lock:
                raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
            _reclaim_stale_ephemeral_sessions(parent, prefix)
            path = Path(tempfile.mkdtemp(prefix = prefix, dir = parent))
            locked = active_lock.enter_context(_locked_file(path / ".active.lock"))
            if not locked:
                raise RuntimeError(f"Could not lock ephemeral session home: {path}")
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target = _refresh_ephemeral_session_marker,
                args = (path / ".active.lock", heartbeat_stop),
                name = "unsloth-agent-home-heartbeat",
                daemon = True,
            )
            heartbeat.start()
        yield path
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout = 1)
        try:
            with _locked_file(parent / ".cleanup.lock") as cleanup_lock:
                if not cleanup_lock:
                    raise RuntimeError(f"Could not lock ephemeral session root: {parent}")
                # Release only after deletion is serialized with startup scavenging.
                active_lock.close()
                if path is not None:
                    shutil.rmtree(path, ignore_errors = True)
        finally:
            active_lock.close()


@contextlib.contextmanager
def _session_config(
    agent: str,
    launch: bool,
    persist: bool = False,
):
    """Yield a private directory for an agent's session config (never the user's own). launch (the default) uses an ephemeral temp dir removed after the agent process exits, so nothing persists; no-launch uses a stable Unsloth-owned dir, since the printed recipe is run later on this machine; persist (from --persist) uses that same stable dir even for a launch, so the agent's session survives the exit and can be resumed. Either way the user's real ~/.<agent> config is left untouched."""
    if launch and not persist:
        # Windows codex keeps #7519's short home (MAX_PATH).
        parent = _ephemeral_session_parent(agent)
        prefix = _ephemeral_session_prefix(agent, parent)
        if parent is not None:
            with _short_ephemeral_session(parent, prefix) as path:
                yield path
        else:
            with _temporary_agent_config(prefix) as path:
                yield path
    else:
        # Never wipe: a printed recipe may still be running here. Writers must reset state a previous run's
        # flags (--yolo) left behind.
        path = _agents_config_root() / agent
        path.mkdir(parents = True, exist_ok = True, mode = 0o700)
        yield path


def _studio_embedding_model(base: str, key: str) -> Optional[str]:
    """Studio's configured embedding model, or None when this server cannot say. Not a model name: a name this server will not serve, beside fallback "none", is the one combination OpenClaw cannot degrade out of, so the caller writes provider "none" instead."""
    try:
        info = _http_json("GET", f"{base}/api/settings/embedding-model", key, timeout = 10)
    # typer.Exit is a RuntimeError subclass: the broad catch would swallow a deliberate abort.
    except (typer.Exit, typer.Abort, click.exceptions.Exit, click.exceptions.Abort):
        raise
    except Exception:  # noqa: BLE001 - an older or unreachable server still gets a working config
        return None
    name = info.get("embedding_model") if isinstance(info, dict) else None
    if not isinstance(name, str) or not name.strip():
        return None
    # OpenClaw sends this to /v1/embeddings verbatim.
    return name.strip()


def _openclaw_provider(
    base: str,
    key: str,
    model: dict,
    max_tokens: Optional[int] = None,
) -> dict:
    provider_model = {"id": model["id"], "name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        provider_model["contextWindow"] = window
        # Unset, OpenClaw caps every reply at 8192 whatever the window.
        provider_model["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        provider_model["maxTokens"] = max_tokens
    return {
        "baseUrl": f"{base}/v1",
        "apiKey": key,
        "api": "openai-completions",
        "models": [provider_model],
    }


def write_openclaw_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    workspace_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
    request_body: Optional[dict] = None,
    max_tokens: Optional[int] = None,
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
    models = _subdict(config, "models")
    models.setdefault("mode", "merge")
    _subdict(models, "providers")["unsloth"] = _openclaw_provider(base, key, model, max_tokens)
    # Memory search defaults to openai, so a local session reaches OpenAI unless this is written.
    search = _subdict(_subdict(config, "memory"), "search")
    if embedding_model:
        search.update(
            {
                "provider": "openai-compatible",
                "model": embedding_model,
                "fallback": "none",
                "remote": {"baseUrl": f"{base}/v1", "apiKey": key},
            }
        )
    else:
        # "none" is keyword-only mode: no network call.
        search.update({"provider": "none", "fallback": "none"})
        search.pop("model", None)
        search.pop("remote", None)
    # ORed with OPENCLAW_LOAD_SHELL_ENV: a persisted true re-enables the login-shell key import.
    _subdict(_subdict(config, "env"), "shellEnv")["enabled"] = False
    # Else OpenClaw drops into its setup agent ("no models available").
    agents = _subdict(config, "agents")
    defaults = _subdict(agents, "defaults")
    _subdict(defaults, "model")["primary"] = f"unsloth/{model['id']}"
    # OpenClaw applies extra_body last and reads only camelCase keys.
    model_ref = f"unsloth/{model['id']}"
    model_settings = defaults.get("models")
    if request_body or (isinstance(model_settings, dict) and model_ref in model_settings):
        params = _subdict(_subdict(_subdict(defaults, "models"), model_ref), "params")
        _merge_request_body(params, "extra_body", request_body or {})
    # Do not seed bootstrap files or init a repo in the user's project.
    defaults["skipBootstrap"] = True
    # OPENCLAW_STATE_DIR does not relocate the workspace; keep the managed fallback for direct callers.
    if workspace_path is None:
        workspace = path.parent / "workspace"
        workspace.mkdir(parents = True, exist_ok = True, mode = 0o700)
        workspace_path = str(workspace)
    defaults["workspace"] = workspace_path
    # Per-agent paths override the defaults; remove stale ones from this isolated copy.
    agent_list = agents.get("list")
    if isinstance(agent_list, list):
        for agent_config in agent_list:
            if isinstance(agent_config, dict):
                agent_config.pop("workspace", None)
                agent_config.pop("agentDir", None)
    # Without auth.mode=none the client will not open the websocket.
    gateway = _subdict(config, "gateway")
    gateway.setdefault("mode", "local")
    _subdict(gateway, "auth").setdefault("mode", "none")
    if yolo:
        # OpenClaw gates exec on BOTH tools.exec and the approvals file (stricter wins); set both, like
        # `openclaw exec-policy preset yolo`.
        exec_policy = _subdict(_subdict(config, "tools"), "exec")
        exec_policy["host"] = "gateway"
        exec_policy["security"] = "full"
        exec_policy["ask"] = "off"
        # ask=off means nothing is prompted, so the runtime socket block is unnecessary.
        approvals = path.parent / "exec-approvals.json"
        _write_private_json(
            approvals,
            {"version": 1, "defaults": {"security": "full", "ask": "off", "askFallback": "full"}},
        )
        typer.echo(f"Updated {approvals}")
    else:
        # An omitted exec policy means security=full, ask=off, so a non-yolo run must WRITE a prompting
        # policy. Only replace our permissive one.
        tools = config.get("tools")
        exec_policy = tools.get("exec") if isinstance(tools, dict) else None
        exec_policy = exec_policy if isinstance(exec_policy, dict) else {}
        # Match ONLY the exact fingerprint --yolo writes; anything else, including any `mode`, is the user's.
        permissive = (
            "mode" not in exec_policy
            and exec_policy.get("host") == "gateway"
            and exec_policy.get("security") == "full"
            and exec_policy.get("ask") == "off"
        )
        if permissive:
            exec_policy = _subdict(_subdict(config, "tools"), "exec")
            exec_policy.pop("host", None)
            exec_policy["security"] = "allowlist"
            exec_policy["ask"] = "on-miss"
        # Drop only the yolo defaults from the approvals file.
        approvals = path.parent / "exec-approvals.json"
        if approvals.exists():
            state = _read_json_object(approvals)
            if state is not None:
                defaults = state.get("defaults")
                # Only the exact yolo fingerprint; a mixed user policy sharing a field stays.
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
                        approvals.unlink()
                        typer.echo(f"Removed {approvals}")
                    else:
                        _write_private_json(approvals, state)
                        typer.echo(f"Updated {approvals}")
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def opencode_output_limit(window: int, max_tokens: Optional[int] = None) -> int:
    if max_tokens:
        return max(1, min(int(max_tokens), window // 2))
    return max(1, min(window // 4, _OPENCODE_OUTPUT_TOKEN_MAX))


def _agent_output_limit(window: int, max_tokens: Optional[int]) -> int:
    output = opencode_output_limit(window, max_tokens)
    if max_tokens and output < max_tokens:
        typer.echo(
            f"Warning: --max-tokens {max_tokens} leaves too little of the {window:,}-token "
            f"context for the conversation; using {output:,}.",
            err = True,
        )
    return output


def opencode_compaction_reserved(window: int, output: int) -> int:
    return max(1, min(output, max(window // 10, 8192)))


def _opencode_output_env(model: dict, max_tokens: Optional[int]) -> dict:
    """Lift OpenCode's output ceiling when --max-tokens exceeds it; re-emit an inherited one so a --no-launch recipe keeps it."""
    window = model.get("context_length") or model.get("max_context_length")
    if not max_tokens:
        return {}
    if not window:
        typer.echo(
            "Warning: Studio did not report the model's context length, so --max-tokens is ignored.",
            err = True,
        )
        return {}
    output = _agent_output_limit(int(window), max_tokens)
    raw = os.environ.get(_OPENCODE_OUTPUT_TOKEN_MAX_ENV, "")
    inherited = int(raw) if raw.isdigit() and int(raw) > 0 else None
    ceiling = inherited or _OPENCODE_OUTPUT_TOKEN_MAX
    if output <= ceiling and inherited is None:
        return {}
    return {_OPENCODE_OUTPUT_TOKEN_MAX_ENV: str(max(output, ceiling))}


def _opencode_provider(
    base: str,
    key: str,
    model: dict,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
) -> dict:
    model_entry = {"name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        output = opencode_output_limit(window, max_tokens)
        # Without a limit OpenCode assumes context 0; without input it ignores compaction.reserved.
        model_entry["limit"] = {"context": window, "input": window, "output": output}
    provider_options = {"baseURL": f"{base}/v1", "apiKey": key}
    if request_body:
        # OpenCode 1.x reads the effort only as reasoningEffort; 2.x sends only the provider body.
        model_entry["options"] = {
            "reasoningEffort" if name == "reasoning_effort" else name: value
            for name, value in request_body.items()
        }
        if "temperature" in request_body:
            model_entry["temperature"] = True
        provider_options["body"] = request_body
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Unsloth Studio",
        "options": provider_options,
        "models": {model["id"]: model_entry},
    }


def write_opencode_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
    as_subagent: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
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
    window = model.get("context_length") or model.get("max_context_length")
    reserved = None
    if window:
        window = int(window)
        reserved = opencode_compaction_reserved(window, opencode_output_limit(window, max_tokens))
    _subdict(config, "provider")[_OPENCODE_PROVIDER] = _opencode_provider(
        base, key, model, max_tokens, request_body
    )
    # Subagent mode leaves the user's main/small models alone.
    opencode_model = f"{_OPENCODE_PROVIDER}/{model['id']}"
    if as_subagent:
        for field in ("model", "small_model"):
            if str(config.get(field) or "").startswith(f"{_OPENCODE_PROVIDER}/"):
                config.pop(field, None)
        managed = {reserved, max(1, window // 10)} if window else set()
        compaction = config.get("compaction")
        if (
            isinstance(compaction, dict)
            and compaction.keys() == {"auto", "reserved"}
            and compaction["auto"] is True
            and compaction["reserved"] in managed
        ):
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
        # The fixed 20k-token default buffer over-compacts on a small local context.
        compaction = _subdict(config, "compaction")
        compaction["auto"] = True
        compaction["reserved"] = reserved
    tools = ("edit", "bash", "webfetch", *(("task",) if as_subagent else ()))
    if yolo:
        # Inline fallback for commands without native --auto; outranks a project config. TUI and `run` use
        # --auto so OpenCode keeps explicit deny rules.
        session_permission = {t: "allow" for t in tools}
        session_permission["external_directory"] = {"*": "allow"}
        config["permission"] = dict(session_permission)
    else:
        # Undo only the explicit per-tool "allow" our --yolo wrote. Never carry a permission inline for
        # non-yolo: OPENCODE_CONFIG_CONTENT outranks the project config we cannot read.
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


def _set_hermes_provider(
    config: dict,
    base: str,
    model: dict,
    request_body: Optional[dict] = None,
) -> None:
    # Hermes only reads the key for a NAMED custom provider.
    _subdict(config, "model").update(
        provider = f"custom:{_HERMES_PROVIDER}",
        default = model["id"],
        api_mode = "openai",
    )
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # OpenAI's /v1/models has no context field, so Hermes may assume 256k; pin the window, compact at 90%.
        if window >= _HERMES_MIN_CONTEXT:
            _subdict(config, "model")["context_length"] = window
            if config["model"].get("ollama_num_ctx") == _HERMES_MIN_CONTEXT:
                del config["model"]["ollama_num_ctx"]
            _subdict(config, "compression").update(enabled = True, threshold = 0.9)
        else:
            # Below the 64,000 floor claim the floor and shrink the threshold so it fires at 90% of the real window.
            _subdict(config, "model")["context_length"] = _HERMES_MIN_CONTEXT
            # Current Hermes needs ollama_num_ctx too; context_length alone no longer passes.
            _subdict(config, "model")["ollama_num_ctx"] = _HERMES_MIN_CONTEXT
            threshold = round(0.9 * window / _HERMES_MIN_CONTEXT, 4)
            _subdict(config, "compression").update(enabled = True, threshold = threshold)
            auxiliary = _subdict(_subdict(config, "auxiliary"), "compression")
            auxiliary["context_length"] = _HERMES_MIN_CONTEXT
    _subdict(config, "providers")[_HERMES_PROVIDER] = {
        "base_url": f"{base}/v1",
        "api_mode": "openai",
        "key_env": _HERMES_ENV_KEY,
        **({"extra_body": request_body} if request_body else {}),
    }


def write_hermes_config(
    base: str,
    model: dict,
    path: Path,
    request_body: Optional[dict] = None,
) -> None:
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
    _set_hermes_provider(config, base, model, request_body)
    text = yaml.safe_dump(config, sort_keys = False)
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text, encoding = "utf-8")
        typer.echo(f"Updated {path}")


def _vibe_env(
    base: str,
    model: dict,
    request_body: Optional[dict] = None,
) -> dict:
    """Vibe settings as VIBE_* env vars: that layer outranks user and project config.toml, so
    nothing is written to the user's Vibe config."""
    entry = {"name": model["id"], "provider": _VIBE_PROVIDER, "alias": _VIBE_MODEL_ALIAS}
    # Vibe sends its own temperature (0.2 by default) with every request.
    temperature = (request_body or {}).get("temperature")
    if temperature is not None:
        entry["temperature"] = float(temperature)
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        # Vibe compacts at 200k tokens by default, past most local windows.
        entry["auto_compact_threshold"] = max(1, int(int(window) * 0.9))
    provider = {
        "name": _VIBE_PROVIDER,
        "api_base": f"{base}/v1",
        "api_key_env_var": _VIBE_ENV_KEY,
        "api_style": "openai",
        "backend": "generic",
    }
    return {
        "VIBE_PROVIDERS": json.dumps([provider]),
        "VIBE_MODELS": json.dumps([entry]),
        "VIBE_ACTIVE_MODEL": _VIBE_MODEL_ALIAS,
        # An inherited allowlist or resumed cloud session would select a hosted model.
        "VIBE_ALLOWED_MODELS": json.dumps(["re:" + re.escape(model["id"])]),
        "VIBE_ENABLE_TELEMETRY": "false",
        "VIBE_ENABLE_UPDATE_CHECKS": "false",
        "VIBE_ENABLE_AUTO_UPDATE": "false",
        "VIBE_ENABLE_NOTIFICATIONS": "false",
    }


def write_pi_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    *,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
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
    # Pi reads custom providers from ~/.pi/agent/models.json (HOME-relocated).
    provider_model = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # Pi's defaults (128000 / 16384) overflow a small context; pin the real window.
        provider_model["contextWindow"] = window
        provider_model["maxTokens"] = _agent_output_limit(window, max_tokens)
    elif max_tokens:
        provider_model["maxTokens"] = max_tokens
    if request_body:
        provider_model["samplingParams"] = request_body
    _subdict(config, "providers")[_PI_PROVIDER] = {
        "api": "openai-completions",
        "baseUrl": f"{base}/v1",
        "apiKey": key,
        "models": [provider_model],
    }
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def _link_user_dir(source: Path, target: Path) -> bool:
    """Expose source at target. True once target resolves to source."""
    # Refresh links, but preserve real session directories.
    if target.is_symlink() or _is_junction(target):
        _remove_overlay_entry(target)
    if target.exists() or not source.is_dir():
        return False
    target.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    try:
        target.symlink_to(source, target_is_directory = True)
    except OSError:
        if not _create_directory_junction(source, target):
            typer.echo(f"Warning: couldn't link {source} into the Pi session.", err = True)
            return False
    return True


def _pi_local_entry(
    entry: str,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> str:
    """Re-anchor a user path from the original Pi agent directory."""
    value = entry.strip()
    if not value or value == "." or value.startswith("file:"):
        # "" and "." would name the whole agent directory.
        return entry
    if value == "~" or value.startswith(("~/", "~" + os.sep)):
        target = os.path.join(home, value[2:])
    else:
        # Pi stores local packages relative to its agent directory.
        target = os.path.join(source, value)
    target = os.path.normpath(target)
    if agents_skills is not None:
        # Pi reads ~/.agents/skills through HOME, which moved.
        user_root, session_root = agents_skills
        try:
            inside = os.path.relpath(target, user_root)
        except ValueError:  # on another Windows drive
            inside = os.pardir
        if inside == os.curdir:
            return session_root
        if inside != os.pardir and not inside.startswith(os.pardir + os.sep):
            return os.path.join(session_root, inside)
    try:
        relative = os.path.relpath(target, source)
    except ValueError:  # on another Windows drive
        return target
    # Session-relative only where the link landed.
    if relative.split(os.sep)[0] in linked:
        return relative
    return target


def _pi_settings_entries(
    key: str,
    entries,
    source: Path,
    home: Path,
    linked: frozenset,
    agents_skills = None,
) -> list:
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        if key == "packages":
            spec = entry.get("source") if isinstance(entry, dict) else entry
            if isinstance(spec, str) and not spec.strip().startswith(
                ("npm:", "git:", "github:", "http:", "https:", "ssh:")
            ):
                spec = _pi_local_entry(spec, source, home, linked, agents_skills)
                entry = {**entry, "source": spec} if isinstance(entry, dict) else spec
        elif isinstance(entry, str):
            prefix = entry[:1] if entry.startswith(("!", "+", "-")) else ""
            pattern = entry[len(prefix) :]
            if not prefix and "*" not in entry and "?" not in entry:
                entry = _pi_local_entry(entry, source, home, linked, agents_skills)
            elif not pattern.strip().startswith("~"):  # Pi does not expand ~ in patterns
                # The agent directory moved; keep the original too, it still matches basenames.
                anchored = prefix + _pi_local_entry(
                    pattern,
                    source,
                    home,
                    linked,
                    agents_skills,
                )
                if anchored != entry:
                    result.append(entry)
                    entry = anchored
        result.append(entry)
    return result


def _clear_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Undo what an earlier launch linked and copied, leaving session state alone."""
    targets = [agent_dir / name for name in _PI_USER_RESOURCE_DIRS]
    targets.append(home / ".agents" / "skills")
    for target in targets:
        if target.is_symlink() or _is_junction(target):
            _remove_overlay_entry(target)
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if not previous:
        return
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        return
    before = json.dumps(settings, sort_keys = True)
    for key, copied in previous.items():
        own = settings.get(key)
        if key in _PI_USER_VERBATIM_SETTINGS:
            # An argument vector, not entries: subtracting drops whatever the two share.
            if own == copied:
                settings.pop(key, None)
        elif isinstance(copied, list) and isinstance(own, list):
            rest = [item for item in own if item not in copied]
            if rest:
                settings[key] = rest
            else:
                settings.pop(key, None)
        elif own == copied:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    manifest_path.unlink(missing_ok = True)


def write_pi_user_resources(agent_dir: Path, home: Path) -> None:
    """Expose selected user Pi resources inside an isolated session."""
    if _wsl_windows_executable(["pi"]):
        # Windows Pi cannot follow WSL links into mounted drives; drop those.
        _clear_pi_user_resources(agent_dir, home)
        return
    user_home = Path.home()
    configured = os.environ.get("PI_CODING_AGENT_DIR")
    configured = configured.strip() if configured else ""
    # Pi resolves a relative override from the launch directory.
    source = (
        Path(os.path.abspath(os.path.expanduser(configured)))
        if configured
        else user_home / ".pi" / "agent"
    )
    if source.resolve(strict = False) == agent_dir.resolve(strict = False):
        # Do not treat this session as its own resource source.
        source = user_home / ".pi" / "agent"
    if configured and not source.is_dir():
        typer.echo(
            f"Warning: PI_CODING_AGENT_DIR points at {source}, which is not a directory; "
            "no Pi extensions or packages will load in this session.",
            err = True,
        )
    linked = frozenset(
        name for name in _PI_USER_RESOURCE_DIRS if _link_user_dir(source / name, agent_dir / name)
    )
    # HOME is relocated, so link Pi's other global skill directory too.
    user_skills = user_home / ".agents" / "skills"
    session_skills = home / ".agents" / "skills"
    agents_skills = (
        (str(user_skills), str(session_skills))
        if _link_user_dir(user_skills, session_skills)
        else None
    )

    user_settings_path = source / "settings.json"
    user_settings = _read_json_object(user_settings_path)
    if user_settings is None:
        typer.echo(
            f"Warning: couldn't parse {user_settings_path}; "
            "Pi packages listed there won't load in this session.",
            err = True,
        )
        user_settings = {}
    settings_path = agent_dir / "settings.json"
    settings = _read_json_object(settings_path)
    if settings is None:
        typer.echo(
            f"Warning: couldn't parse {settings_path}; your Pi packages won't load in this session.",
            err = True,
        )
        return
    manifest_path = agent_dir / _PI_USER_RESOURCES_MANIFEST
    previous = _read_json_object(manifest_path)
    if previous is None:
        # Provenance is lost, so removed entries cannot be reconciled.
        typer.echo(
            f"Warning: couldn't parse {manifest_path}; Pi resources copied by an earlier "
            "launch stay in this session even if you removed them since.",
            err = True,
        )
        previous = {}
    before = json.dumps(settings, sort_keys = True)
    copied = {}
    for key in _PI_USER_RESOURCE_SETTINGS:
        entries = _pi_settings_entries(
            key,
            user_settings.get(key),
            source,
            user_home,
            linked,
            agents_skills,
        )
        stale = previous.get(key) if isinstance(previous.get(key), list) else []
        own = settings.get(key)
        if own is not None and not isinstance(own, list):
            # Pi types these as arrays; leave a shape we do not understand alone.
            typer.echo(
                f"Warning: {settings_path} has a non-list {key!r}; "
                "leaving it as is, so your Pi entries for it won't load in this session.",
                err = True,
            )
            continue
        own = [item for item in own or [] if item not in stale and item not in entries]
        if entries or own:
            # Pi keeps the FIRST duplicate package, so session entries lead; patterns stay user-first.
            settings[key] = own + entries if key == "packages" else entries + own
        else:
            settings.pop(key, None)
        if entries:
            copied[key] = entries
    for key in _PI_USER_VERBATIM_SETTINGS:
        # Pi runs every package lookup through npmCommand.
        value = user_settings.get(key)
        own = settings.get(key)
        if own is not None and own != previous.get(key):
            continue
        if isinstance(value, list) and value and all(isinstance(arg, str) for arg in value):
            settings[key] = value
            copied[key] = value
        else:
            settings.pop(key, None)
    if json.dumps(settings, sort_keys = True) != before:
        _write_private_json(settings_path, settings)
    if copied != previous:
        if copied:
            _write_private_json(manifest_path, copied)
        else:
            manifest_path.unlink(missing_ok = True)


def write_pi_subagent_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    approve: bool = False,
    max_tokens: Optional[int] = None,
    request_body: Optional[dict] = None,
) -> None:
    """Write private bootstrap data for the bundled Pi extension."""
    window = model.get("context_length") or model.get("max_context_length")
    window = int(window) if window else 32768
    _write_private_json(
        path,
        {
            "baseUrl": f"{base}/v1",
            "apiKey": key,
            "model": model["id"],
            "contextWindow": window,
            "maxTokens": _agent_output_limit(window, max_tokens),
            "approve": approve,
            **({"samplingParams": request_body} if request_body else {}),
        },
    )


def write_dsh_patch(
    base: str,
    model: dict,
    path: Path,
    request_body: Optional[dict] = None,
) -> None:
    """Write the dsh loader patch that points the booted profile at Unsloth.

    dsh 0.1.7 dropped `settings.yaml`: it now imports a leftover one into the profile only
    after the first boot has settled, so that boot still runs on the DeepSeek default. A
    `--patch` overlay is read at boot on every dsh release this supports, and the file is
    Unsloth's own, so it is rewritten whole rather than merged.
    """
    import yaml

    model_entry = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        model_entry["contextWindow"] = window
        model_entry["maxTokens"] = opencode_output_limit(window)
    compat = {"supportsDeveloperRole": False, "maxTokensField": "max_tokens"}
    if request_body:
        # dsh sends template kwargs only for a model that declares reasoning levels.
        model_entry["reasoningEfforts"] = {
            "off": None,
            "low": "low",
            "medium": "medium",
            "high": "high",
        }
        compat["thinkingFormat"] = "chat-template"
        compat["chatTemplateKwargs"] = request_body
    entries = [
        {
            "id": "llm-pi-ai",
            "name": "@deepseek-ai/dsh-llm-pi-ai",
            "config": {
                "providers": {
                    _DSH_PROVIDER: {
                        "displayName": "Unsloth Studio",
                        "api": "openai-completions",
                        "baseURL": f"{base}/v1",
                        "apiKeyEnv": _DSH_ENV_KEY,
                        # pi-ai reads an unknown base URL as OpenAI itself.
                        "compat": compat,
                        "models": [model_entry],
                    }
                }
            },
        },
        {
            "id": "agent-default-model",
            "name": "@deepseek-ai/dsh-agent-default-model",
            "config": {"provider": _DSH_PROVIDER, "model": model["id"]},
        },
    ]
    text = yaml.safe_dump(entries, sort_keys = False)
    if not path.exists() or path.read_text(encoding = "utf-8") != text:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(text, encoding = "utf-8")
        typer.echo(f"Updated {path}")


class _AppTarget(NamedTuple):
    agent: str
    label: str
    path: Path
    updates: object = None


def _app_state_path(agent: str, path: Optional[Path] = None) -> Path:
    # One state per settings file, so each profile keeps its own backup and key.
    name = (
        agent if path is None else f"{agent}-{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    )
    return _agents_config_root() / "app" / f"{name}.json"


def _read_app_file(path: Path) -> Optional[str]:
    if path.is_symlink():
        _fail(
            f"{path} is a symlink, so Unsloth won't rewrite it. Replace the link with a "
            "plain file and re-run, or add the Unsloth provider there yourself."
        )
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with open(fd, encoding = "utf-8", newline = "") as handle:
        return handle.read()


def _app_file_mode(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mode & 0o777
    except FileNotFoundError:
        return None


def _write_app_file(path: Path, text: str) -> None:
    # Never world-readable, and the rename replaces a planted symlink instead of following it.
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd, temp = tempfile.mkstemp(prefix = f".{path.name}.", dir = path.parent)
    try:
        with open(fd, "w", encoding = "utf-8", newline = "") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


_JSONC_COMMENT = re.compile(r'("(?:\\.|[^"\\])*")|//[^\n]*|/\*.*?\*/', re.S)
_JSONC_TRAILING_COMMA = re.compile(r'("(?:\\.|[^"\\])*")|,(?=\s*[\]}])')


def _load_app_config(path: Path, text: Optional[str]) -> Optional[dict]:
    if text is None or not text.strip():
        return {}
    if path.suffix == ".toml":
        try:
            import tomllib  # novermin
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None
        data = tomllib.loads(text)
    elif path.suffix == ".yaml":
        import yaml
        data = yaml.safe_load(text)
    else:
        # OpenCode reads JSONC and OpenClaw JSON5.
        text = _JSONC_COMMENT.sub(lambda m: m.group(1) or "", text)
        data = json.loads(_JSONC_TRAILING_COMMA.sub(lambda m: m.group(1) or "", text))
    if not isinstance(data, dict):
        raise ValueError("not a settings object")
    return data


def _parse_app_config(path: Path, text: Optional[str]) -> Optional[dict]:
    try:
        return _load_app_config(path, text)
    except Exception:
        _fail(f"Couldn't parse {path}, so it was left as is. Fix it and re-run.")


def _set_app_values(text: Optional[str], path: Path, updates: list) -> Optional[str]:
    config = _parse_app_config(path, text)
    before = json.dumps(config, sort_keys = True)
    for keys, value in updates:
        parent = config
        for depth, key in enumerate(keys[:-1]):
            if parent.get(key) is None:
                parent[key] = {}
            elif not isinstance(parent[key], dict):
                _fail(
                    f"{'.'.join(keys[: depth + 1])} in {path} isn't a table, so it was left as is."
                )
            parent = parent[key]
        parent[keys[-1]] = value
    if json.dumps(config, sort_keys = True) == before:
        return text
    if path.suffix != ".yaml":
        return json.dumps(config, indent = 2, ensure_ascii = False) + "\n"
    import yaml

    # Rewrite only the touched top-level blocks so the rest keeps its comments.
    candidate = text or ""
    for top in dict.fromkeys(keys[0] for keys, _ in updates):
        block = yaml.safe_dump({top: config[top]}, sort_keys = False, allow_unicode = True)
        match = re.search(rf"(?m)^{re.escape(top)}:[^\n]*\n(?:[ \t]+[^\n]*\n)*", candidate)
        if match:
            candidate = candidate[: match.start()] + block + candidate[match.end() :]
        else:
            candidate += ("\n" if candidate and not candidate.endswith("\n") else "") + block
    with contextlib.suppress(Exception):
        if _load_app_config(path, candidate) == config:
            return candidate
    return yaml.safe_dump(config, sort_keys = False, allow_unicode = True)


def _set_toml_key(text: str, name: str, line: str) -> tuple:
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=[^\r\n]*", chunks[0])
    if match is None:
        return f"{line}\n{text}", None
    chunks[0] = chunks[0][: match.start()] + line + chunks[0][match.end() :]
    return "".join(chunks), match.group(0)


def _unset_toml_key(text: str, name: str, line: str, previous: Optional[str]) -> Optional[str]:
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=([^\r\n]*)(\r?\n)?", chunks[0])
    if match is None or match.group(0).strip() != line:
        return None
    restored = "" if previous is None else previous + (match.group(2) or "")
    chunks[0] = chunks[0][: match.start()] + restored + chunks[0][match.end() :]
    return "".join(chunks)


def _same_app_config(path: Path, text: str, other: str) -> bool:
    if text.strip() == other.strip():
        return True
    try:
        parsed = _load_app_config(path, text)
        return parsed is not None and parsed == _load_app_config(path, other)
    except Exception:
        return False


def _revoke_app_key(base: str, key_id: object) -> bool:
    if not is_loopback_url(base) or not verify_studio_identity(base):
        return False
    token = _studio_token()
    if token is None:
        return False
    try:
        _http_json("DELETE", f"{base}/api/auth/api-keys/{int(key_id)}", token)
    except urllib.error.HTTPError as exc:
        return exc.code == 404
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
    return True


def _app_api_key(target: _AppTarget, base: str, explicit: Optional[str], state: dict) -> tuple:
    if explicit:
        return explicit, None
    previous = state.get("key")
    if (
        previous
        and state.get("key_id") is not None
        and state.get("base") == base
        and _key_accepted(base, previous)
    ):
        return previous, state["key_id"]
    token = _studio_token() if verify_studio_identity(base) else None
    if token is None:
        _fail(
            f"Couldn't create an API key for the {target.label} automatically. Create one "
            "in Unsloth → Settings → API and pass it with --api-key."
        )
    answer = _http_json(
        "POST",
        f"{base}/api/auth/api-keys",
        token,
        {"name": target.label},
        error = "Couldn't create an API key",
    )
    return answer["key"], answer["api_key"]["id"]


def _open_app_config(target: _AppTarget, base: str, explicit_key: Optional[str]) -> tuple:
    if not is_loopback_url(base) and not explicit_key:
        _fail(
            f"{base} isn't on this machine. --app saves the key in your {target.label} "
            "config, so pass the key for that server with --api-key."
        )
    state_path = _app_state_path(target.agent, None if target.agent == "codex" else target.path)
    state = _read_json_object(state_path) or {}
    if state.get("path") != str(target.path):
        state = {}
    text = _read_app_file(target.path)
    _parse_app_config(target.path, text)
    key, key_id = _app_api_key(target, base, explicit_key, state)
    return state_path, state, text, key, key_id


def _save_app_config(
    target: _AppTarget,
    state_path: Path,
    state: dict,
    new_state: dict,
    text: Optional[str],
    new_text: str,
) -> None:
    new_state["backup"] = state.get("backup")
    if text is not None and not state:
        backup = target.path.with_name(target.path.name + ".unsloth-backup")
        _write_app_file(backup, text)
        new_state["backup"] = str(backup)
        typer.echo(f"Backed up {target.path} to {backup}")
    _write_private_json(state_path, new_state)
    if new_text != text:
        _write_app_file(target.path, new_text)
        typer.echo(f"Updated {target.path}")
    if state.get("key_id") is not None and state.get("key") != new_state["key"]:
        _revoke_app_key(state["base"], state["key_id"])


def _app_server_notice(base: str) -> None:
    if _keep_auto_served():
        typer.echo(f"Unsloth Studio is still running at {base}; the app needs it running.")
        typer.echo("Stop it with: unsloth studio stop")


def _revoke_new_app_key(base: str, key_id: object, state: dict) -> None:
    # A key minted for a failed setup would otherwise stay live and unreachable.
    if key_id is not None and key_id != state.get("key_id"):
        _revoke_app_key(base, key_id)


def _add_app_provider(
    target: _AppTarget, base: str, explicit_key: Optional[str], entry: dict
) -> None:
    state, key_id = {}, None
    try:
        state_path, state, text, key, key_id = _open_app_config(target, base, explicit_key)
        new_text = _set_app_values(
            text, target.path, target.updates(text, target.path, base, key, entry)
        )
        new_state = {"path": str(target.path), "base": base, "key": key, "key_id": key_id}
        _save_app_config(target, state_path, state, new_state, text, new_text)
    except BaseException:
        _revoke_new_app_key(base, key_id, state)
        _shutdown_auto_served()
        raise
    typer.echo(
        f"Added {entry['id']} from Unsloth at {base} to the {target.label}. Your default "
        "model is unchanged: pick Unsloth in the app's model picker (restart the app if it "
        "is open)."
    )
    if key_id is not None:
        typer.echo(
            f"It uses its own API key, '{target.label}', which you can revoke in "
            "Unsloth → Settings → API."
        )
    typer.echo(f"After loading another model, re-run: unsloth start {target.agent} --app")
    _app_server_notice(base)


_CODEX_APP_CATALOG = "unsloth-model-catalog.json"


def _codex_app_lines(model_id: str) -> dict:
    return {
        "model_provider": f'model_provider = "{_CODEX_PROFILE}"',
        "model": f"model = {json.dumps(model_id)}",
        "model_catalog_json": f"model_catalog_json = {json.dumps(_CODEX_APP_CATALOG)}",
    }


def _codex_app_build(text: Optional[str], base: str, key: str, entry: dict, previous) -> tuple:
    previous = dict(previous or {})
    tables = [
        c for c in re.split(r"(?m)^(?=\[)", text or "") if c.startswith(_CODEX_PROVIDER_TABLES)
    ]
    previous.setdefault("table", "".join(tables) or None)
    merged = _merge_codex_config(text or "", base, key)
    for name, line in _codex_app_lines(entry["id"]).items():
        merged, old = _set_toml_key(merged, name, line)
        previous.setdefault(name, old)
    return merged, previous


def _codex_app_restore(text: str, state: dict) -> tuple:
    skipped = []
    chunks = re.split(r"(?m)^(?=\[)", text)
    ours = _codex_provider_table(state["base"], state["key"]).strip()

    def is_ours(chunk: str) -> bool:
        # An editor on Windows may have rewritten the file with CRLF line endings.
        return chunk.replace("\r\n", "\n").strip() == ours

    if any(is_ours(chunk) for chunk in chunks):
        table = state["previous"].get("table") or ""
        text = "".join(table if is_ours(chunk) else chunk for chunk in chunks)
    else:
        skipped.append(_PROVIDER_HEADER)
    for name, line in _codex_app_lines(state["model"]).items():
        restored = _unset_toml_key(text, name, line, state["previous"].get(name))
        if restored is None:
            skipped.append(name)
        else:
            text = restored
    return text, skipped


def _codex_app_target() -> _AppTarget:
    # The desktop app never sees CODEX_HOME.
    path = _codex_source_home(ignore_configured = True) / "config.toml"
    return _AppTarget("codex", "Codex app", path)


_CODEX_APP_HEALTH_INTERVAL_S = 5.0
_CODEX_APP_HEALTH_MISSES = 3


def _codex_app_owner() -> dict:
    try:
        import psutil
    except ImportError:
        return {"pid": os.getpid(), "started": None}
    return {"pid": os.getpid(), "started": psutil.Process().create_time()}


def _codex_app_owner_alive(owner: object) -> bool:
    try:
        pid = int(owner["pid"])
        if owner["started"] is None:
            # psutil is optional, so fall back to the PID alone.
            from unsloth_cli.commands.studio import _pid_alive
            return _pid_alive(pid)
        import psutil

        return abs(psutil.Process(pid).create_time() - float(owner["started"])) < 1.0
    except Exception:
        return False


def _switch_codex_app_back(state_path: Path, state: dict) -> list:
    path = Path(state["path"])
    text = _read_app_file(path)
    skipped = []
    if text is not None:
        new_text, skipped = _codex_app_restore(text, state)
        backup = Path(state["backup"]) if state.get("backup") else None
        original = _read_app_file(backup) if backup else None
        if original is not None and _same_app_config(path, new_text, original):
            new_text = original
        if state.get("created") and _same_app_config(path, new_text, ""):
            path.unlink()
        elif new_text != text:
            _write_app_file(path, new_text)
        if original is not None and new_text == original:
            if state.get("mode") is not None:
                os.chmod(path, state["mode"])
            backup.unlink(missing_ok = True)
    if "model_catalog_json" not in skipped:
        path.with_name(_CODEX_APP_CATALOG).unlink(missing_ok = True)
    revoked = state.get("key_id") is None or _revoke_app_key(state.get("base", ""), state["key_id"])
    state_path.unlink(missing_ok = True)
    notes = [
        f"Left {name} in {path} as is: it changed while Codex was on Unsloth." for name in skipped
    ]
    if not revoked:
        notes.append(
            "Couldn't reach Unsloth to revoke the app's key; delete 'Codex app' in "
            "Unsloth → Settings → API."
        )
    return notes


def _recover_codex_app() -> Optional[dict]:
    state_path = _app_state_path("codex")
    state = _read_json_object(state_path) or {}
    if not state.get("path"):
        return None
    if _codex_app_owner_alive(state.get("owner")):
        return state
    notes = _switch_codex_app_back(state_path, state)
    typer.echo(
        "Switched the Codex app back from an earlier `unsloth start codex --app` that did not "
        "exit cleanly."
    )
    for note in notes:
        typer.echo(note)
    return None


def _open_codex_app() -> None:
    if sys.platform == "darwin":
        with contextlib.suppress(OSError):
            if (
                subprocess.run(["open", "-b", "com.openai.codex"], capture_output = True).returncode
                == 0
            ):
                return
    typer.echo("Open the Codex app now.")


def _hold_codex_app(base: str) -> None:
    def stop(signum, frame):
        raise KeyboardInterrupt

    previous = {}
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            previous[getattr(signal, name)] = signal.signal(getattr(signal, name), stop)
    try:
        misses = 0
        while misses < _CODEX_APP_HEALTH_MISSES:
            time.sleep(_CODEX_APP_HEALTH_INTERVAL_S)
            misses = 0 if _studio_healthy(base) else misses + 1
        typer.echo(f"Unsloth at {base} stopped answering.")
    except KeyboardInterrupt:
        pass
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _codex_app_session(base: str, explicit_key: Optional[str], entry: dict) -> None:
    target = _codex_app_target()
    catalog = target.path.with_name(_CODEX_APP_CATALOG)
    state, key_id = {}, None
    try:
        state_path, state, text, key, key_id = _open_app_config(target, base, explicit_key)
        new_text, previous = _codex_app_build(text, base, key, entry, None)
        _parse_app_config(target.path, new_text)
        # The Codex app lists only the active provider's catalog.
        _write_app_file(catalog, json.dumps(_codex_model_catalog(entry, "list"), indent = 2) + "\n")
        new_state = {
            "path": str(target.path),
            "base": base,
            "model": entry["id"],
            "key": key,
            "key_id": key_id,
            "owner": _codex_app_owner(),
            "created": text is None,
            "mode": _app_file_mode(target.path),
            "previous": previous,
        }
        _save_app_config(target, state_path, {}, new_state, text, new_text)
    except BaseException:
        _revoke_new_app_key(base, key_id, state)
        _shutdown_auto_served()
        raise
    try:
        _open_codex_app()
        typer.echo(
            f"The Codex app is on {entry['id']} from Unsloth at {base} until this command exits "
            "(Ctrl+C). If the app was already open, quit and reopen it: it reads the model list "
            "only at startup."
        )
        _hold_codex_app(base)
    finally:
        # A second Ctrl+C must not cut the switch back short.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            notes = _switch_codex_app_back(state_path, new_state)
            _shutdown_auto_served()
            # After SIGHUP the terminal may be gone.
            with contextlib.suppress(OSError):
                typer.echo("Switched the Codex app back; restart it if a window is still open.")
                for note in notes:
                    typer.echo(note)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)


def _check_app_flags(
    requested: bool,
    args: list,
    as_subagent: bool = False,
) -> None:
    if requested and (args or as_subagent):
        _fail("--app sets up the desktop app, so it takes no agent arguments or --as-subagent.")


def _opencode_app_target(
    max_tokens: Optional[int] = None, request_body: Optional[dict] = None
) -> _AppTarget:
    # OpenCode Desktop imports the login shell env; merges config.json, opencode.json, opencode.jsonc.
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opencode"
    names = ("opencode.jsonc", "opencode.json", "config.json")
    path = next((root / n for n in names if os.path.lexists(root / n)), root / "opencode.json")

    def updates(text, path, base, key, entry):
        provider = _opencode_provider(base, key, entry, max_tokens, request_body)
        changes = [(("provider", _OPENCODE_PROVIDER), provider)]
        config = _parse_app_config(path, text) or {}
        enabled = config.get("enabled_providers")
        if isinstance(enabled, list) and _OPENCODE_PROVIDER not in enabled:
            changes.append((("enabled_providers",), [*enabled, _OPENCODE_PROVIDER]))
        # disabled_providers wins over enabled_providers.
        disabled = config.get("disabled_providers")
        if isinstance(disabled, list) and _OPENCODE_PROVIDER in disabled:
            changes.append(
                (("disabled_providers",), [p for p in disabled if p != _OPENCODE_PROVIDER])
            )
        return changes

    return _AppTarget("opencode", "OpenCode app", path, updates)


def _openclaw_app_updates(
    text,
    path,
    base,
    key,
    entry,
    max_tokens = None,
) -> list:
    config = _parse_app_config(path, text) or {}
    agents = config.get("agents")
    defaults = agents.get("defaults") if isinstance(agents, dict) else None
    defaults = defaults if isinstance(defaults, dict) else {}
    ref = f"unsloth/{entry['id']}"
    changes = [
        (("models", "providers", "unsloth"), _openclaw_provider(base, key, entry, max_tokens))
    ]
    # Never create an allowlist: the picker then hides everything else.
    policy = defaults.get("modelPolicy")
    allowed = policy.get("allow") if isinstance(policy, dict) else None
    meta = config.get("meta")
    migrations = meta.get("migrations") if isinstance(meta, dict) else None
    migrated = isinstance(migrations, dict) and migrations.get("modelPolicyAllowlist")
    # An empty allow list or models map allows every model.
    if isinstance(allowed, list):
        if allowed and ref not in allowed:
            changes.append((("agents", "defaults", "modelPolicy", "allow"), [*allowed, ref]))
    elif policy is None and not migrated and isinstance(defaults.get("models"), dict):
        if defaults["models"] and ref not in defaults["models"]:
            changes.append((("agents", "defaults", "models", ref), {}))
    return changes


def _openclaw_app_target(max_tokens: Optional[int] = None) -> _AppTarget:
    path = Path.home() / ".openclaw" / "openclaw.json"
    updates = functools.partial(_openclaw_app_updates, max_tokens = max_tokens)
    return _AppTarget("openclaw", "OpenClaw app", path, updates)


def _hermes_app_target(request_body: Optional[dict] = None) -> _AppTarget:
    # Hermes Desktop follows the sticky active profile.
    home = Path.home() / ".hermes"
    if sys.platform == "win32":
        # hermes_constants._get_platform_default_hermes_home.
        local = os.environ.get("LOCALAPPDATA", "").strip()
        home = (Path(local) if local else Path.home() / "AppData" / "Local") / "hermes"
    try:
        profile = (home / "active_profile").read_text(encoding = "utf-8").strip()
    except OSError:
        profile = ""
    path = home / "config.yaml"
    if profile != "default" and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", profile):
        path = home / "profiles" / profile / "config.yaml"

    def updates(text, path, base, key, entry):
        settings: dict = {}
        _set_hermes_provider(settings, base, entry, request_body)
        provider = settings["providers"][_HERMES_PROVIDER]
        del provider["key_env"]
        provider["api_key"] = key
        window = entry.get("context_length") or entry.get("max_context_length")
        if window:
            # model.context_length only covers the default model.
            context = max(int(window), _HERMES_MIN_CONTEXT)
            provider["models"] = {entry["id"]: {"context_length": context}}
        return [(("providers", _HERMES_PROVIDER), provider)]

    return _AppTarget("hermes", "Hermes app", path, updates)


@start_app.command("claude", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point Claude Code at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    install_hint = (
        "irm https://claude.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://claude.ai/install.sh | bash"
    )
    # Before the install prompt: no point fetching a tool the run cannot use.
    _preflight_agent_gguf(_CLAUDE_GGUF_AGENT, model, serve = serve, launch = launch)
    _require_agent_for_launch("claude", install_hint, launch)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        preload_check = functools.partial(_attach_gguf_check, _CLAUDE_GGUF_AGENT),
        server_options = server_options,
    )
    # Before the launch owns the server, so a rejection tears it down, not atexit.
    try:
        _require_gguf_for_agent(_CLAUDE_GGUF_AGENT, base, key, entry["id"])
    except BaseException:
        _shutdown_auto_served()
        raise
    model_id = entry["id"]
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        window = subagent_model.get("context_length") or subagent_model.get("max_context_length")
        server_env = {
            "UNSLOTH_CLAUDE_SUBAGENT_BASE_URL": base,
            "UNSLOTH_CLAUDE_SUBAGENT_API_KEY": key,
            "UNSLOTH_CLAUDE_SUBAGENT_MODEL": subagent_id,
            "UNSLOTH_CLAUDE_SUBAGENT_BYPASS_PERMISSIONS": "1" if yolo else "0",
        }
        if window:
            server_env["UNSLOTH_CLAUDE_SUBAGENT_CONTEXT_WINDOW"] = str(int(window))
        if server_options.request_body():
            server_env["CLAUDE_CODE_EXTRA_BODY"] = json.dumps(server_options.request_body())
        with _session_config("claude-subagent", launch, persist = persist) as config:
            plugin = write_claude_subagent_plugin(config, server_env)
            command = [
                "claude",
                "--plugin-dir",
                _agent_config_path(plugin, ["claude"]),
                # Before ctx.args (a forwarded `--` would make later flags positional); `=` form since the flag is
                # variadic.
                f"--allowedTools={_CLAUDE_SUBAGENT_TOOL},{_CLAUDE_SUBAGENT_PLAN_TOOL}",
                *_yolo_command_flags("claude", yolo),
                *ctx.args,
            ]
            typer.echo(
                "Unsloth is available as a local agent. "
                "Ask Claude to spawn an Unsloth or local agent."
            )
            _run(
                base,
                subagent_model,
                {},
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return

    env = _claude_local_env(base, key, entry, server_options.request_body())
    # --yolo maps to --dangerously-skip-permissions. IS_SANDBOX stays unset: never falsely claim a sandbox.
    # History lives in ~/.claude/projects, so `claude --continue` works.
    with _session_config("claude", launch, persist = persist) as config:
        settings = _write_claude_settings(config, model_id, env)
        command = _claude_local_command(
            model_id,
            _agent_config_path(settings, ["claude"]),
            yolo,
            ctx.args,
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


@start_app.command("codex", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def codex(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
    app: bool = _CODEX_APP_OPTION,
):
    """Point OpenAI Codex at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _check_app_flags(app, ctx.args, as_subagent)
    running = _recover_codex_app()
    if app and running is not None:
        _fail(
            "The Codex app is already on Unsloth for another `unsloth start codex --app` "
            f"(PID {running['owner']['pid']}). Stop that one first."
        )
    install_hint = _npm_install_hint("@openai/codex")
    # Before the install prompt: no point fetching a tool the run cannot use.
    _preflight_agent_gguf(_CODEX_GGUF_AGENT, model, serve = serve, launch = launch)
    _require_agent_for_launch("codex", install_hint, launch and not app)
    codex_effort = _codex_reasoning_effort(reasoning, reasoning_effort)
    if codex_effort and not _agent_version_at_least("codex", _CODEX_REASONING_REQUEST_MIN_VERSION):
        codex_effort = None
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        # The app reads no per-launch flags, so the server applies them.
        carried = _REASONING_FIELDS if codex_effort and not app else frozenset(),
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        preload_check = functools.partial(_attach_gguf_check, _CODEX_GGUF_AGENT),
        server_options = server_options,
    )
    # Tear down an auto-started server here if it rejects the model, rather than at atexit.
    try:
        _require_gguf_for_agent(_CODEX_GGUF_AGENT, base, key, entry["id"])
    except BaseException:
        _shutdown_auto_served()
        raise
    if app:
        return _codex_app_session(base, api_key, entry)
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        with _session_config("codex-subagent", launch, persist = persist) as home:
            bridge_config = write_codex_subagent_bridge(
                base,
                key,
                subagent_model,
                home,
                yolo = yolo,
                reasoning_effort = codex_effort,
            )
            parent_home = write_codex_parent_overlay(home / "parent")
            command = [
                "codex",
                *_codex_subagent_flags(bridge_config),
                *_yolo_command_flags("codex", yolo),
                *ctx.args,
            ]
            typer.echo(
                "Unsloth is available as a local agent. "
                "Ask Codex to spawn an Unsloth or local agent."
            )
            _run(
                base,
                subagent_model,
                {"CODEX_HOME": str(parent_home)},
                command,
                launch = launch,
                install_hint = install_hint,
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
        write_codex_config(base, entry, home, codex_effort)
        env = {_CODEX_ENV_KEY: key, "CODEX_HOME": str(home)}
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = _CODEX_ENV_UNSET,
        )


@start_app.command("openclaw", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def openclaw(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    app: bool = _APP_OPTION,
):
    """Point OpenClaw at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _reject_as_subagent("openclaw", ctx.args)
    _check_app_flags(app, ctx.args)
    install_hint = (
        "iwr -useb https://openclaw.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://openclaw.ai/install.sh | bash"
    )
    _require_agent_for_launch("openclaw", install_hint, launch and not app)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = frozenset() if app else _ALL_REQUEST_FIELDS,
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    if app:
        return _add_app_provider(_openclaw_app_target(max_tokens), base, api_key, entry)
    openclaw_args = list(ctx.args)
    # Default only the empty case to `tui --local`: a leading "--flag value" is ambiguous between a
    # global and a tui option.
    if not openclaw_args:
        openclaw_args = ["tui", "--local"]
    command = ["openclaw", *openclaw_args]
    with _session_config("openclaw", launch, persist = persist) as cfg:
        config_path = cfg / "openclaw.json"
        # Resolve the project when the recipe executes, not when it is generated.
        write_openclaw_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo,
            workspace_path = "${OPENCLAW_WORKSPACE_DIR}",
            embedding_model = _studio_embedding_model(base, key),
            request_body = server_options.request_body(),
            max_tokens = max_tokens,
        )
        # Scope config and state so OpenClaw never touches ~/.openclaw. Shell env off, else it re-imports keys.
        env = {
            "OPENCLAW_CONFIG_PATH": str(config_path),
            "OPENCLAW_STATE_DIR": str(cfg),
            "OPENCLAW_LOAD_SHELL_ENV": "0",
        }
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            unset_env = _OPENCLAW_ENV_UNSET,
            cwd_env = ("OPENCLAW_WORKSPACE_DIR",),
        )


@start_app.command("opencode", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def opencode(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
    app: bool = _APP_OPTION,
):
    """Point OpenCode at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _check_app_flags(app, ctx.args, as_subagent)
    command_name, opencode_v2 = _opencode_command()
    install_hint = _npm_install_hint("@opencode-ai/cli@beta" if opencode_v2 else "opencode-ai")
    _require_agent_for_launch(command_name, install_hint, launch and not app)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    if app:
        target = _opencode_app_target(max_tokens, server_options.request_body())
        return _add_app_provider(target, base, api_key, entry)
    if opencode_v2:
        typer.echo(
            f"OpenCode V2 provider policies must allow '{_OPENCODE_PROVIDER}'.",
            err = True,
        )
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        # Append-safe: `opencode --auto run ...` would parse as the TUI.
        route_native_auto = (
            yolo and _opencode_supports_native_auto(command_name) and (launch or bool(ctx.args))
        )
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = [command_name, *opencode_args]
        with _session_config("opencode-subagent", launch, persist = persist) as cfg:
            config_path = cfg / "opencode.json"
            session_permission = write_opencode_config(
                base,
                key,
                subagent_model,
                config_path,
                yolo = yolo and not native_auto,
                as_subagent = True,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
            )
            env = {
                "OPENCODE_CONFIG": str(config_path),
                **_opencode_output_env(subagent_model, max_tokens),
            }
            inline_config = _opencode_subagent_inline_config(
                config_path,
                session_permission,
                command = command_name,
                v2 = opencode_v2,
            )
            # A project opencode.json outranks the session file; pin our agent in the inline overlay.
            inline_config.setdefault("agent", {})[_SUBAGENT_NAME] = {
                "description": _SUBAGENT_DESCRIPTION,
                "mode": "subagent",
                "model": f"{_OPENCODE_PROVIDER}/{subagent_model['id']}",
                "prompt": _SUBAGENT_INSTRUCTIONS,
            }
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps(inline_config)
            typer.echo("Unsloth is available as @unsloth and in /models.")
            _run(
                base,
                subagent_model,
                env,
                command,
                launch = launch,
                install_hint = install_hint,
            )
        return
    opencode_model = f"{_OPENCODE_PROVIDER}/{entry['id']}"
    # The inline config pins the model; --model only for an interactive bare launch, since drivers
    # append a subcommand after a printed command.
    native_auto = False
    route_native_auto = yolo and _opencode_supports_native_auto(command_name)
    if ctx.args:
        opencode_args = list(ctx.args)
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args, route_native_auto, v2 = opencode_v2
        )
        command = [command_name, *opencode_args]
    elif launch:
        opencode_args = [] if opencode_v2 else ["--model", opencode_model]
        if opencode_v2:
            opencode_args = _opencode_v2_standalone_args(opencode_args)
        opencode_args, native_auto = _opencode_native_auto_args(
            opencode_args,
            route_native_auto,
            v2 = opencode_v2,
        )
        command = [command_name, *opencode_args]
    else:
        # Append-safe base: the command is unknown here, so keep the config fallback.
        opencode_args = _opencode_v2_standalone_args([]) if opencode_v2 else []
        command = [command_name, *opencode_args]
    # Sessions live in ~/.local/share/opencode, so `opencode --continue` works.
    with _session_config("opencode", launch, persist = persist) as cfg:
        config_path = cfg / "opencode.json"
        # OPENCODE_CONFIG is an overlay between global and project configs.
        session_permission = write_opencode_config(
            base,
            key,
            entry,
            config_path,
            yolo = yolo and not native_auto,
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
        )
        # A project opencode.json outranks OPENCODE_CONFIG, so pin the model in OPENCODE_CONFIG_CONTENT.
        # V1 filters scope the session; V2 filters are policies, left intact. Pin small_model too.
        inline_config: dict = {
            "model": opencode_model,
            "small_model": opencode_model,
        }
        if not opencode_v2:
            inline_config["enabled_providers"] = [_OPENCODE_PROVIDER]
            inline_config["disabled_providers"] = []
        if session_permission:
            inline_config["permission"] = session_permission
        env = {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_CONFIG_CONTENT": json.dumps(inline_config),
            **_opencode_output_env(entry, max_tokens),
        }
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("hermes", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def hermes(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    app: bool = _APP_OPTION,
):
    """Point Hermes (Nous Research) at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _reject_as_subagent("hermes", ctx.args)
    _check_app_flags(app, ctx.args)
    native_args = [*_yolo_command_flags("hermes", yolo), *ctx.args]
    command = ["hermes", *_hermes_resume_oneshot_args(native_args)]
    install_hint = _hermes_install_hint()
    _require_agent_for_launch("hermes", install_hint, launch and not app)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    if app:
        target = _hermes_app_target(server_options.request_body())
        _add_app_provider(target, base, api_key, entry)
        window = entry.get("context_length") or entry.get("max_context_length")
        if window and int(window) < _HERMES_MIN_CONTEXT:
            # The app's compaction settings are global.
            typer.echo(
                f"Warning: {entry['id']} serves {int(window):,} tokens, below Hermes' "
                f"{_HERMES_MIN_CONTEXT:,} floor, so long chats in the app can overflow before "
                f"Hermes compacts. Load it with --max-seq-length {_HERMES_MIN_CONTEXT} to avoid this.",
                err = True,
            )
        return
    with _session_config("hermes", launch, persist = persist) as home:
        # HERMES_HOME relocates hermes' whole home, leaving ~/.hermes untouched.
        write_hermes_config(base, entry, home / "config.yaml", server_options.request_body())
        env = {_HERMES_ENV_KEY: key, "HERMES_HOME": str(home)}
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("pi", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def pi(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    max_tokens: Optional[int] = _MAX_TOKENS_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
    as_subagent: bool = _AS_SUBAGENT_OPTION,
):
    """Point Pi (coding agent) at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    install_hint = _npm_install_hint(
        "@earendil-works/pi-coding-agent",
        ignore_scripts = True,
    )
    if as_subagent and not _PI_SUBAGENT_EXTENSION.is_file():
        _fail(f"Missing Pi subagent extension: {_PI_SUBAGENT_EXTENSION}")
    _require_agent_for_launch("pi", install_hint, launch)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _ALL_REQUEST_FIELDS,
    )
    if server_options.request_body() and not _agent_version_at_least(
        "pi", _PI_SAMPLING_PARAMS_MIN_VERSION
    ):
        server_options = server_options._replace(carried = frozenset())
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    if as_subagent:
        subagent_id = _subagent_model_id(base, key, entry, model, gguf_variant)
        subagent_model = {**entry, "id": subagent_id}
        extension = _agent_config_path(_PI_SUBAGENT_EXTENSION, ["pi"])
        with _session_config("pi-subagent", launch, persist = persist) as config:
            config_path = config / "subagent.json"
            write_pi_subagent_config(
                base,
                key,
                subagent_model,
                config_path,
                approve = yolo,
                max_tokens = max_tokens,
                request_body = server_options.request_body(),
            )
            command = [
                "pi",
                "--extension",
                extension,
                *_yolo_command_flags("pi", yolo),
                *ctx.args,
            ]
            typer.echo(
                "Unsloth is available as a local agent and in /model. "
                "Ask Pi to spawn an Unsloth or local agent."
            )
            _run(
                base,
                subagent_model,
                {"UNSLOTH_PI_SUBAGENT_CONFIG": str(config_path)},
                command,
                launch = launch,
                install_hint = install_hint,
                clear_screen = True,
            )
        return
    # Pi defaults to the google provider; the endpoint itself only lives in models.json.
    command = [
        "pi",
        "--provider",
        _PI_PROVIDER,
        "--model",
        entry["id"],
        *_yolo_command_flags("pi", yolo),
        *ctx.args,
    ]
    # --ignore-scripts matches Pi's documented install recipe.
    with _session_config("pi", launch, persist = persist) as home:
        # Pi prefers PI_CODING_AGENT_DIR over $HOME/.pi/agent, so pin it or an inherited one wins.
        pi_agent_dir = home / ".pi" / "agent"
        write_pi_config(
            base,
            key,
            entry,
            pi_agent_dir / "models.json",
            max_tokens = max_tokens,
            request_body = server_options.request_body(),
        )
        write_pi_user_resources(pi_agent_dir, home)
        env = {"HOME": str(home), "PI_CODING_AGENT_DIR": str(pi_agent_dir)}
        if os.name == "nt" or os.environ.get("WSL_DISTRO_NAME"):
            # Node on Windows resolves ~ via USERPROFILE (then HOMEDRIVE + HOMEPATH), not HOME.
            env["USERPROFILE"] = str(home)
            drive, tail = os.path.splitdrive(str(home))
            if drive:
                env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
        # Pi paints inline from the cursor, so give it the clean screen it assumes.
        _run(
            base,
            entry,
            env,
            command,
            launch = launch,
            install_hint = install_hint,
            clear_screen = True,
        )


@start_app.command("dsh", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def dsh(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point DeepSeek Harness (dsh) at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _reject_as_subagent("dsh", ctx.args)
    install_hint = _npm_install_hint(_DSH_PACKAGE)
    _require_agent_for_launch("dsh", install_hint, launch)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = _REASONING_FIELDS,
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    with _session_config("dsh", launch, persist = persist) as home:
        patch = home / _DSH_PATCH_FILE
        write_dsh_patch(base, entry, patch, server_options.request_body())
        # A Windows dsh under WSL gets DSH_HOME translated through WSLENV, but not argv.
        command = _dsh_command(ctx.args, _agent_config_path(patch, ["dsh"]))
        env = {
            _DSH_ENV_KEY: key,
            "DSH_HOME": str(home),
            # dsh uploads session records once a user records /feedback.
            "DSH_TELEMETRY_DISABLED": "1",
            "DSH_PERMISSION_MODE": (
                _DSH_YOLO_PERMISSION_MODE if yolo else _DSH_SAFE_PERMISSION_MODE
            ),
        }
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)


@start_app.command("vibe", cls = _PassthroughCommand, context_settings = _PASSTHROUGH)
def vibe(
    ctx: typer.Context,
    model: Optional[str] = _MODEL_OPTION,
    api_key: Optional[str] = _KEY_OPTION,
    launch: bool = _LAUNCH_OPTION,
    gguf_variant: Optional[str] = _GGUF_VARIANT_OPTION,
    max_seq_length: int = _CONTEXT_OPTION,
    load_in_4bit: bool = _LOAD_4BIT_OPTION,
    tensor_parallel: bool = _TENSOR_PARALLEL_OPTION,
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = _GPU_MEMORY_MODE_OPTION,
    enable_tools: Optional[bool] = _ENABLE_TOOLS_OPTION,
    tool_call_healing: Optional[bool] = _TOOL_CALL_HEALING_OPTION,
    tool_call_nudging: Optional[bool] = _TOOL_CALL_NUDGING_OPTION,
    reasoning: Optional[Literal["on", "off", "auto"]] = _REASONING_OPTION,
    reasoning_effort: Optional[str] = _REASONING_EFFORT_OPTION,
    temperature: Optional[float] = _TEMPERATURE_OPTION,
    top_p: Optional[float] = _TOP_P_OPTION,
    top_k: Optional[int] = _TOP_K_OPTION,
    min_p: Optional[float] = _MIN_P_OPTION,
    repetition_penalty: Optional[float] = _REPETITION_PENALTY_OPTION,
    presence_penalty: Optional[float] = _PRESENCE_PENALTY_OPTION,
    serve: bool = _SERVE_OPTION,
    yolo: bool = _YOLO_OPTION,
    persist: bool = _PERSIST_OPTION,
):
    """Point Mistral Vibe at the running Unsloth server and start it."""
    model, ctx.args[:] = _consume_positional_model(model, ctx.args)
    _reject_as_subagent("vibe", ctx.args)
    install_hint = _vibe_install_hint()
    _require_agent_for_launch("vibe", install_hint, launch)
    server_options = ServerOptions(
        enable_tools = enable_tools,
        tool_call_healing = tool_call_healing,
        tool_call_nudging = tool_call_nudging,
        reasoning = reasoning,
        reasoning_effort = reasoning_effort,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        carried = frozenset({"temperature"}),
    )
    base, key, entry = _connect(
        api_key,
        model,
        _load_options(
            ctx, gguf_variant, max_seq_length, load_in_4bit, tensor_parallel, gpu_memory_mode
        ),
        serve = serve,
        launch = launch,
        server_options = server_options,
    )
    command = ["vibe", *_yolo_command_flags("vibe", yolo), *ctx.args]
    # Vibe keeps its own home; the env layer pins provider and model above its config files.
    request_body = server_options.request_body()
    if "temperature" not in request_body:
        # Vibe always sends a temperature, so pass the recommended one explicitly.
        status = _inference_status(base, key)
        recommended = (status.get("inference") or {}).get("temperature")
        if recommended is not None and any(
            _model_id_matches(entry["id"], status_id, allow_casefold = is_loopback_url(base))
            for status_id in (status.get("active_model"), status.get("model_identifier"))
            if status_id
        ):
            request_body = {**request_body, "temperature": recommended}
    env = {_VIBE_ENV_KEY: key, **_vibe_env(base, entry, request_body)}
    _run(base, entry, env, command, launch = launch, install_hint = install_hint)
