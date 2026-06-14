# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth connect` — launch a coding agent (Claude Code, Codex) against a running Studio server."""

import json
import os
import re
import shlex
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
    no_args_is_help = True,
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
    except (urllib.error.URLError, TimeoutError) as exc:
        if error is None:
            raise
        _fail(f"{error}: {getattr(exc, 'reason', None) or exc}")


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


def _cached_keys(cache: Path) -> list:
    try:
        data = json.loads(cache.read_text())
    except Exception:
        return []
    keys = [k for k in data.get("keys", []) if isinstance(k, str)]
    legacy = data.get("key")  # pre-multi-key cache format
    if isinstance(legacy, str) and legacy not in keys:
        keys.append(legacy)
    return keys


def _remember_key(cache: Path, key: str) -> None:
    existing = _cached_keys(cache)
    keys = ([key] + [k for k in existing if k != key])[:8]
    if keys == existing:
        return
    try:
        cache.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
        # O_CREAT with 0o600 so the keys are never world-readable, even briefly.
        fd = os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps({"keys": keys}) + "\n")
    except OSError:
        pass  # worst case the next launch mints another key


def _agent_api_key(base: str, explicit: Optional[str]) -> str:
    cache = _key_cache_path()
    if explicit:
        _remember_key(cache, explicit)
        return explicit

    # Keys are per-server, so when switching between Studios (local one day,
    # an SSH-tunnelled remote the next) the right key is whichever validates.
    for key in _cached_keys(cache):
        try:
            _http_json("GET", f"{base}/v1/models", key)
        except Exception:
            continue
        _remember_key(cache, key)
        return key

    token = _studio_token()
    if token is None:
        _fail(
            "Couldn't authenticate with the Studio server automatically (it may be "
            "remote, or running as a different OS user). Create an API key in "
            "Studio → Settings → API and pass it once with --api-key; it is "
            "remembered for next time."
        )
    key = _http_json(
        "POST",
        f"{base}/api/auth/api-keys",
        token,
        {"name": "Coding agents (unsloth connect)"},
        error = "Couldn't create an API key",
    )["key"]
    _remember_key(cache, key)
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
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(settings, indent = 2) + "\n", encoding = "utf-8")
    except OSError:
        typer.echo(
            f"Warning: couldn't write {path} — set CLAUDE_CODE_ATTRIBUTION_HEADER "
            'to "0" in its "env" section yourself, or local inference will be much slower.',
            err = True,
        )
        return
    typer.echo(f"Disabled Claude Code's attribution header in {path} (it breaks KV-cache reuse).")


_DYNAMIC_SECTIONS_FLAG = "--exclude-dynamic-system-prompt-sections"


def _claude_cache_flags() -> list:
    # The flag moves per-machine context (cwd, env info, git status) out of
    # the system prompt, where it changes every session and defeats llama.cpp
    # prefix caching. As of 2.1.175 it only takes effect in print mode (`-p`
    # passed through ctx.args); interactive sessions accept and ignore it.
    # Claude Code < 2.1.98 aborts on the unknown flag, so check the version
    # first; no local binary means a --no-launch printout for another machine.
    executable = shutil.which("claude")
    if executable is None:
        return [_DYNAMIC_SECTIONS_FLAG]
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output = True, text = True, timeout = 10
        )
        version = tuple(int(part) for part in result.stdout.split()[0].split("."))
    except Exception:
        return []
    return [_DYNAMIC_SECTIONS_FLAG] if version >= (2, 1, 98) else []


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


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


def write_codex_config(base: str, model: dict) -> None:
    home = codex_home()
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
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def _print_env(env: dict, command: list) -> None:
    if os.name == "nt":
        for name, value in env.items():
            typer.echo(f'$env:{name} = "{value}"')
        typer.echo(" ".join(command))
        return
    for name, value in env.items():
        typer.echo(f"export {name}={shlex.quote(value)}")
    typer.echo(shlex.join(command))


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
    # Negative returncode means killed by signal N; shells expect 128+N.
    raise typer.Exit(code = code if code >= 0 else 128 - code)


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
        help = "Studio API key; minted automatically when omitted. Keys are remembered, so passing one once is enough.",
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

    env = {
        "ANTHROPIC_BASE_URL": base,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_MODEL": model_id,
        # Update checks, beta features, and other background requests either
        # stall against a local server or evict the conversation from
        # llama-server's KV-cache slots, so turn off everything nonessential.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    command = ["claude", "--model", model_id, *_claude_cache_flags(), *ctx.args]
    typer.echo(f"Studio {base} · model {model_id}")
    if not launch:
        _print_env(env, command)
        return
    install_hint = (
        "irm https://claude.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://claude.ai/install.sh | bash"
    )
    _launch(command, env, install_hint = install_hint)


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
        help = "Studio API key; minted automatically when omitted. Keys are remembered, so passing one once is enough.",
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
