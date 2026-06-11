# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth connect` — launch a coding agent (Claude Code, Codex) against a running Studio server."""

import json
import os
import re
import shutil
import signal
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn, Optional

import typer

from unsloth_cli._inference import (
    _studio_token,
    ensure_studio_backend_path,
    find_studio_server,
)

connect_app = typer.Typer(
    help = "Connect a coding agent to a running Studio server.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)

_CODEX_PROFILE = "unsloth_api"
_CODEX_ENV_KEY = "UNSLOTH_STUDIO_AUTH_TOKEN"
_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


def _fail(message: str) -> NoReturn:
    typer.echo(message, err = True)
    raise typer.Exit(code = 1)


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
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method = method,
    )
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        if error is None:
            raise
        try:
            body = json.loads(exc.read().decode())
            detail = body.get("detail") or body["error"]["message"]
        except Exception:
            detail = str(exc)
        _fail(f"{error}: {detail}")


def _require_studio() -> str:
    base = find_studio_server()
    if base is None:
        expected = os.environ.get("UNSLOTH_STUDIO_URL", "http://127.0.0.1:8888")
        _fail(
            f"No running Studio server found at {expected}. Start one with "
            "`unsloth studio`, or point UNSLOTH_STUDIO_URL at a remote server."
        )
    return base


def _key_cache_path() -> Path:
    ensure_studio_backend_path()
    from utils.paths import auth_root
    return auth_root() / "agent_api_key.json"


def _agent_api_key(base: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    cache = _key_cache_path()
    try:
        key = json.loads(cache.read_text())["key"]
        _http_json("GET", f"{base}/v1/models", key)
        return key
    except Exception:
        pass

    token = _studio_token()
    if token is None:
        _fail(
            "Couldn't authenticate with the Studio server automatically (it may be "
            "remote, or running as a different OS user). Create an API key in "
            "Studio → Settings → API and pass it with --api-key."
        )
    key = _http_json(
        "POST",
        f"{base}/api/auth/api-keys",
        token,
        {"name": "Coding agents (unsloth connect)"},
        error = "Couldn't create an API key",
    )["key"]
    try:
        cache.parent.mkdir(parents = True, exist_ok = True)
        cache.write_text(json.dumps({"key": key}) + "\n")
        os.chmod(cache, 0o600)
    except OSError:
        pass  # worst case the next launch mints another key
    return key


def _loaded_models(base: str, key: str) -> list:
    return _http_json("GET", f"{base}/v1/models", key, error = "Couldn't list models").get("data", [])


def _resolve_model(base: str, key: str, requested: Optional[str]) -> dict:
    models = _loaded_models(base, key)
    match = next((m for m in models if m["id"] == requested), None)
    if requested and match is None:
        typer.echo(f"Loading {requested} on the Studio server (this can take a while)…")
        _http_json(
            "POST",
            f"{base}/api/inference/load",
            key,
            {"model_path": requested},
            timeout = 3600,
            error = "Model load failed",
        )
        models = _loaded_models(base, key)
        match = next((m for m in models if m["id"] == requested), None)
    if match is not None:
        return match
    if not models:
        _fail(
            "No model is loaded in Studio. Load one from the model dropdown in "
            "the UI, or pass --model <hf-id-or-path> to load it from here."
        )
    return models[0]


def _require_gguf_for_codex(base: str, key: str, model_id: str) -> None:
    # Codex always streams, and Studio only streams /v1/responses from llama-server.
    try:
        if _http_json("GET", f"{base}/api/inference/status", key).get("is_gguf"):
            return
    except Exception:
        return  # fail open on unknown server versions
    hint = model_id if "gguf" in model_id.lower() else f"{model_id}-GGUF"
    _fail(
        f"Codex needs a GGUF model served by llama-server, but {model_id} is on "
        f"the transformers backend. Try: unsloth connect codex --model {hint}"
    )


def claude_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def ensure_claude_attribution_header() -> None:
    # The header invalidates the llama.cpp KV cache (~90% slower) and Claude
    # Code only honors this setting from settings.json, not the env var.
    path = claude_settings_path()
    settings = {}
    if path.exists():
        try:
            settings = json.loads(path.read_text(encoding = "utf-8"))
        except (ValueError, OSError):
            settings = None
        if not isinstance(settings, dict):
            typer.echo(
                f"Warning: couldn't parse {path} — set CLAUDE_CODE_ATTRIBUTION_HEADER "
                'to "0" in its "env" section yourself, or local inference will be much slower.',
                err = True,
            )
            return
    env = settings.get("env")
    if not isinstance(env, dict):
        env = settings["env"] = {}
    if str(env.get("CLAUDE_CODE_ATTRIBUTION_HEADER")) == "0":
        return
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(settings, indent = 2) + "\n", encoding = "utf-8")
    typer.echo(f"Disabled Claude Code's attribution header in {path} (it breaks KV-cache reuse).")


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _merge_codex_config(existing: str, base: str) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{_CODEX_PROFILE}"\n'
    text = "".join(c for c in chunks if not c.startswith(_PROVIDER_HEADER))
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


def write_codex_config(base: str, model: dict) -> None:
    home = codex_home()
    home.mkdir(parents = True, exist_ok = True)

    config = home / "config.toml"
    existing = config.read_text(encoding = "utf-8") if config.exists() else ""
    merged = _merge_codex_config(existing, base)
    if merged != existing:
        config.write_text(merged, encoding = "utf-8")
        typer.echo(f"Updated {config}")

    profile_text = f'model_provider = "{_CODEX_PROFILE}"\nmodel = {json.dumps(model["id"])}\n'
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def _print_env(env: dict, command: list) -> None:
    template = '$env:{} = "{}"' if os.name == "nt" else 'export {}="{}"'
    for name, value in env.items():
        typer.echo(template.format(name, value))
    typer.echo(" ".join(command))


def _launch(command: list, env: dict, install_hint: str) -> NoReturn:
    executable = shutil.which(command[0])
    if executable is None:
        _fail(f"`{command[0]}` not found on PATH. Install it with: {install_hint}")
    # Ctrl+C cancels a turn inside the agent; don't let it kill this wrapper.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        code = subprocess.run([executable, *command[1:]], env = {**os.environ, **env}).returncode
    finally:
        signal.signal(signal.SIGINT, previous)
    raise typer.Exit(code = code)


@connect_app.command("claude", context_settings = _PASSTHROUGH)
def claude(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help = "Model for the agent; defaults to the one loaded in Studio.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar = "UNSLOTH_API_KEY",
        help = "Studio API key; minted and cached automatically when omitted.",
    ),
    launch: bool = typer.Option(
        True,
        "--launch/--no-launch",
        help = "--no-launch prints the env and command instead (remote shells, WSL).",
    ),
):
    """Point Claude Code at the running Studio server and start it."""
    base = _require_studio()
    key = _agent_api_key(base, api_key)
    model_id = _resolve_model(base, key, model)["id"]
    ensure_claude_attribution_header()

    env = {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_MODEL": model_id}
    command = ["claude", "--model", model_id, *ctx.args]
    typer.echo(f"Studio {base} · model {model_id}")
    if not launch:
        _print_env(env, command)
        return
    _launch(command, env, install_hint = "curl -fsSL https://claude.ai/install.sh | bash")


@connect_app.command("codex", context_settings = _PASSTHROUGH)
def codex(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help = "Model for the agent; defaults to the one loaded in Studio.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar = "UNSLOTH_API_KEY",
        help = "Studio API key; minted and cached automatically when omitted.",
    ),
    launch: bool = typer.Option(
        True,
        "--launch/--no-launch",
        help = "--no-launch prints the env and command instead (remote shells, WSL).",
    ),
):
    """Point OpenAI Codex at the running Studio server and start it."""
    base = _require_studio()
    key = _agent_api_key(base, api_key)
    entry = _resolve_model(base, key, model)
    _require_gguf_for_codex(base, key, entry["id"])
    write_codex_config(base, entry)

    env = {_CODEX_ENV_KEY: key}
    command = ["codex", "--oss", "--profile", _CODEX_PROFILE, *ctx.args]
    typer.echo(f"Studio {base} · model {entry['id']}")
    if not launch:
        _print_env(env, command)
        return
    _launch(command, env, install_hint = "npm install -g @openai/codex")
