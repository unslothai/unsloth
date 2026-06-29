# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` — launch a coding agent against a running Studio server."""

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple, NoReturn, Optional

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
    help = "Start a coding agent against a running Studio server.",
    no_args_is_help = True,
    context_settings = {"help_option_names": ["-h", "--help"]},
)

_CODEX_PROFILE = "unsloth_api"
_CODEX_ENV_KEY = "UNSLOTH_STUDIO_AUTH_TOKEN"
_HERMES_ENV_KEY = "UNSLOTH_API_KEY"
_HERMES_PROVIDER = "unsloth"
_PI_PROVIDER = "unsloth"
_PROVIDER_HEADER = f"[model_providers.{_CODEX_PROFILE}]"
_PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}
_CLAUDE_ENV_UNSET = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")

# Shared by every agent command; only the config/env/command differ.
_MODEL_OPTION = typer.Option(
    None, "--model", "-m", help = "Model for the agent; defaults to the one loaded in Studio."
)
_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar = "UNSLOTH_API_KEY",
    help = (
        "Studio API key. For a local Studio it is minted automatically and "
        "remembered per server. For a remote server, pass one with --api-key "
        "(or UNSLOTH_API_KEY); it is remembered for next time."
    ),
)
_LAUNCH_OPTION = typer.Option(
    True,
    "--launch/--no-launch",
    help = "--no-launch prints the env and command instead (remote shells, WSL).",
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

# Per-agent CLI flag for "run tools without prompting". opencode and openclaw have no
# such flag (config only) and are handled in their config writers, so they are absent.
_YOLO_COMMAND_FLAGS = {
    "claude": ["--dangerously-skip-permissions"],
    "codex": ["--dangerously-bypass-approvals-and-sandbox"],
    "hermes": ["--yolo"],
    # Pi never prompts per tool call; its only approval gate is project trust, so -a
    # (trust project resources) is the closest "don't ask me" equivalent.
    "pi": ["--approve"],
}


def _yolo_command_flags(agent: str, yolo: bool) -> list:
    return _YOLO_COMMAND_FLAGS[agent] if yolo else []


class LoadOptions(NamedTuple):
    """Model-load knobs forwarded to /api/inference/load when --model triggers a load."""

    gguf_variant: Optional[str] = None
    max_seq_length: int = 0
    load_in_4bit: bool = True
    tensor_parallel: bool = False


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
    try:
        _http_json("GET", f"{base}/v1/models", key)
        return True
    except Exception:
        return False


def _agent_api_key(base: str, explicit: Optional[str]) -> str:
    cache = _key_cache_path()
    if explicit:
        _remember_key(cache, base, explicit, "saved")
        return explicit

    # Replay a key the user saved for *this exact* server first (scoped per base,
    # so it only goes back there -- including a remote/SSH-tunnelled Studio whose
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
            "a local Studio. Create an API key in Studio → Settings → API and "
            "pass it with --api-key (it is remembered per server), or set "
            "UNSLOTH_API_KEY."
        )
    if not verify_studio_identity(base):
        _fail(
            f"Couldn't verify that {base} is your Studio (it may be running as a "
            "different OS user, or another process took the port). Create an API "
            "key in Studio → Settings → API and pass it with --api-key, or set "
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
            "Couldn't authenticate with the Studio server automatically. Create "
            "an API key in Studio → Settings → API and pass it with --api-key, "
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


def _resolve_model(
    base: str,
    key: str,
    requested: Optional[str],
    load: LoadOptions = LoadOptions(),
) -> dict:
    models = _loaded_models(base, key)
    match = next((m for m in models if m["id"] == requested), None)
    if requested and match is None:
        typer.echo(f"Loading {requested} on the Studio server (this can take a while)…")
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
        loaded = _http_json(
            "POST",
            f"{base}/api/inference/load",
            key,
            payload,
            timeout = 3600,
            error = "Model load failed",
        )
        # Studio registers the model under a canonical id (resolved identifier,
        # casing) that /v1/models echoes but which may differ from the path we
        # passed; match on the id the load reports so we don't silently fall
        # through to models[0] and connect to a different loaded model.
        wanted = {requested}
        if isinstance(loaded, dict):
            wanted |= {loaded.get("model"), loaded.get("display_name")} - {None}
        models = _loaded_models(base, key)
        match = next((m for m in models if m["id"] in wanted), None)
    if match is not None:
        return match
    if requested:
        # We asked Studio to load it and it didn't surface in /v1/models; don't
        # silently hand back an unrelated loaded model.
        _fail(
            f"Studio didn't report '{requested}' as loaded. Double-check the model "
            "id, or load it from the model dropdown in the UI."
        )
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
        f"the transformers backend. Try: unsloth start codex --model {hint}"
    )


_DYNAMIC_SECTIONS_FLAG = "--exclude-dynamic-system-prompt-sections"
# Session overlay applied via `claude --settings`; suppresses the attribution header
# for THIS run only (no ~/.claude write) so llama.cpp KV-cache reuse is preserved. It
# reinforces the CLAUDE_CODE_ATTRIBUTION_HEADER env var on builds that read the setting
# only from settings.json.
_CLAUDE_SETTINGS_OVERLAY = '{"env":{"CLAUDE_CODE_ATTRIBUTION_HEADER":"0"}}'


def _claude_version() -> Optional[tuple]:
    # None = no local `claude` (a --no-launch printout for another machine; assume a
    # current build). An unparseable version is treated as too old for the new flags.
    executable = shutil.which("claude")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output = True, text = True, timeout = 10
        )
        return tuple(int(part) for part in result.stdout.split()[0].split("."))
    except Exception:
        return (0,)


def _claude_flags() -> list:
    # Both knobs preserve llama.cpp KV-cache reuse: --exclude-dynamic-system-prompt-sections
    # moves per-session context out of the system prompt, and --settings suppresses the
    # attribution header for this session only (no persistent ~/.claude write; the env var
    # sets it too). Claude Code < 2.1.98 aborts on unknown flags, so gate on the version;
    # no local binary means a printout for another machine, so assume a current build.
    version = _claude_version()
    if version is not None and version < (2, 1, 98):
        return []
    return [_DYNAMIC_SECTIONS_FLAG, "--settings", _CLAUDE_SETTINGS_OVERLAY]


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
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        profile_text += f"model_context_window = {int(window)}\n"
    profile = home / f"{_CODEX_PROFILE}.config.toml"
    if not profile.exists() or profile.read_text(encoding = "utf-8") != profile_text:
        profile.write_text(profile_text, encoding = "utf-8")
        typer.echo(f"Updated {profile}")


def _wsl_windows_executable(command: list) -> Optional[str]:
    if os.name == "nt" or not os.environ.get("WSL_DISTRO_NAME"):
        return None
    executable = shutil.which(command[0])
    if executable and executable.startswith("/mnt/"):
        return executable
    return None


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
    entries = [entry for entry in current.split(":") if entry]
    existing = {entry.split("/", 1)[0] for entry in entries}
    for name in names:
        base = name.split("/", 1)[0]  # dedup on the bare name, ignoring any /p flag
        if base not in existing:
            entries.append(name)
            existing.add(base)
    return ":".join(entries)


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
    typer.echo(shlex.join(command))


def _launch(
    command: list,
    env: dict,
    install_hint: str,
    unset_env: tuple = (),
) -> NoReturn:
    executable = shutil.which(command[0])
    if executable is None:
        _fail(f"`{command[0]}` not found on PATH. Install it with: {install_hint}")
    wsl_env_bridge = _wsl_bridge_names(env, unset_env) if _wsl_windows_executable(command) else ()
    child_env = dict(os.environ)
    if wsl_env_bridge:
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
    raise typer.Exit(code = code if code >= 0 else 128 - code)


def _connect(
    api_key: Optional[str],
    model: Optional[str],
    load: LoadOptions = LoadOptions(),
) -> tuple:
    base = _require_studio()
    key = _agent_api_key(base, api_key)
    return base, key, _resolve_model(base, key, model, load)


def _run(
    base: str,
    entry: dict,
    env: dict,
    command: list,
    *,
    launch: bool,
    install_hint: str,
    unset_env: tuple = (),
) -> None:
    typer.echo(f"Studio {base} · model {entry['id']}")
    wsl_env_bridge = _wsl_bridge_names(env, unset_env) if _wsl_windows_executable(command) else ()
    if not launch:
        _print_env(env, command, unset_env = unset_env, wsl_env_bridge = wsl_env_bridge)
        return
    _launch(command, env, install_hint = install_hint, unset_env = unset_env)


def _agents_config_root() -> Path:
    ensure_studio_backend_path()
    from utils.paths import auth_root
    return auth_root() / "agents"


@contextlib.contextmanager
def _session_config(agent: str, launch: bool):
    """Yield a private directory for an agent's session config (never the user's own).

    launch: an ephemeral temp dir removed after the agent process exits, so nothing
    persists. no-launch: a stable Unsloth-owned dir (the printed recipe is run later
    on this machine), reset each run. Either way the user's real ~/.<agent> config is
    left untouched.
    """
    if launch:
        path = Path(tempfile.mkdtemp(prefix = f"unsloth-{agent}-"))
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors = True)
    else:
        path = _agents_config_root() / agent
        shutil.rmtree(path, ignore_errors = True)
        path.mkdir(parents = True, exist_ok = True, mode = 0o700)
        yield path


def write_openclaw_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
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
    # Studio is a generic OpenAI-compatible /v1 endpoint (the vLLM/LM Studio path).
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
    defaults = _subdict(_subdict(config, "agents"), "defaults")
    _subdict(defaults, "model")["primary"] = f"unsloth/{model['id']}"
    # Unauthenticated loopback gateway: without auth.mode=none the client won't open
    # the websocket. The daemon must still be started separately (`openclaw gateway`).
    gateway = _subdict(config, "gateway")
    gateway.setdefault("mode", "local")
    _subdict(gateway, "auth").setdefault("mode", "none")
    if yolo:
        # OpenClaw has no --yolo flag; the exec policy is config-driven. On a gateway
        # host, security=full + ask=off runs tools without prompting (the session's
        # fresh OPENCLAW_STATE_DIR has no stricter approvals file to override it).
        exec_policy = _subdict(_subdict(config, "tools"), "exec")
        exec_policy["host"] = "gateway"
        exec_policy["security"] = "full"
        exec_policy["ask"] = "off"
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


def write_opencode_config(
    base: str,
    key: str,
    model: dict,
    path: Path,
    yolo: bool = False,
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
    config.setdefault("$schema", "https://opencode.ai/config.json")
    model_entry = {"name": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # A custom-provider model with no limit defaults to context 0, which silently
        # disables OpenCode's auto-compaction; declare the real window (and a sane
        # output cap) so it compacts instead of overflowing the server.
        model_entry["limit"] = {"context": window, "output": min(window // 4, 8192)}
    _subdict(config, "provider")["unsloth"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Unsloth Studio",
        "options": {"baseURL": f"{base}/v1", "apiKey": key},
        "models": {model["id"]: model_entry},
    }
    # OpenCode selects a model by "<providerID>/<modelID>".
    config["model"] = f"unsloth/{model['id']}"
    if window:
        # Compact with ~10% headroom (near 90% full). The fixed 20k-token default
        # buffer over-compacts, or never settles, on a small local context.
        compaction = _subdict(config, "compaction")
        compaction["auto"] = True
        compaction["reserved"] = max(1, window // 10)
    if yolo:
        # OpenCode has no --yolo flag; auto-approve is the config `permission` block
        # (singular). Allow the prompting tools so tool calls don't block on the TUI.
        config["permission"] = {"edit": "allow", "bash": "allow", "webfetch": "allow"}
    if json.dumps(config, sort_keys = True) != before:
        _write_private_json(path, config)
        typer.echo(f"Updated {path}")


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
        # Hermes auto-detects context from GET /v1/models, but OpenAI's schema has no
        # context field, so it can fall back to a 256k default that overflows a small
        # local model. Pin the real window (top-level model.context_length is the
        # highest-priority override) and compact at 90% of it (Hermes defaults to 50%).
        _subdict(config, "model")["context_length"] = int(window)
        _subdict(config, "compression").update(enabled = True, threshold = 0.9)
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
    # session). Studio is a generic OpenAI-compatible /v1 endpoint, and the key lives
    # in the config rather than the env (matching openclaw/opencode).
    provider_model = {"id": model["id"]}
    window = model.get("context_length") or model.get("max_context_length")
    if window:
        window = int(window)
        # An unspecified model defaults to contextWindow 128000 / maxTokens 16384,
        # far larger than a small Studio context, so Pi compacts too late and overflows
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
    yolo: bool = _YOLO_OPTION,
):
    """Point Claude Code at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
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
    command = [
        "claude",
        "--model",
        model_id,
        *_claude_flags(),
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
    yolo: bool = _YOLO_OPTION,
):
    """Point OpenAI Codex at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
    )
    _require_gguf_for_codex(base, key, entry["id"])
    command = [
        "codex",
        "--oss",
        "--profile",
        _CODEX_PROFILE,
        *_yolo_command_flags("codex", yolo),
        *ctx.args,
    ]
    with _session_config("codex", launch) as home:
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
    yolo: bool = _YOLO_OPTION,
):
    """Point OpenClaw at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
    )
    command = ["openclaw", *ctx.args]
    install_hint = (
        "iwr -useb https://openclaw.ai/install.ps1 | iex"
        if os.name == "nt"
        else "curl -fsSL https://openclaw.ai/install.sh | bash"
    )
    with _session_config("openclaw", launch) as cfg:
        config_path = cfg / "openclaw.json"
        # key lives in the config, not the env; --yolo writes the exec policy here too.
        write_openclaw_config(base, key, entry, config_path, yolo = yolo)
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
    yolo: bool = _YOLO_OPTION,
):
    """Point OpenCode at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
    )
    command = ["opencode", *ctx.args]
    with _session_config("opencode", launch) as cfg:
        config_path = cfg / "opencode.json"
        # OPENCODE_CONFIG is an overlay (loaded between the user's global and project
        # configs), so this adds the Unsloth provider/model for the session without
        # changing the user's default model. Key lives in the config, not the env.
        write_opencode_config(base, key, entry, config_path, yolo = yolo)
        env = {"OPENCODE_CONFIG": str(config_path)}
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
    yolo: bool = _YOLO_OPTION,
):
    """Point Hermes (Nous Research) at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
    )
    command = ["hermes", *_yolo_command_flags("hermes", yolo), *ctx.args]
    install_hint = (
        "curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent"
        "/main/scripts/install.sh | bash"
    )
    with _session_config("hermes", launch) as home:
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
    yolo: bool = _YOLO_OPTION,
):
    """Point Pi (coding agent) at the running Studio server and start it."""
    base, key, entry = _connect(
        api_key, model, LoadOptions(gguf_variant, max_seq_length, load_in_4bit, tensor_parallel)
    )
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
    install_hint = "npm install -g @earendil-works/pi-coding-agent"
    with _session_config("pi", launch) as home:
        # Pi has no config-dir env var; it resolves ~/.pi off $HOME (Node's homedir()
        # honors it), so HOME-scope the session to leave the user's ~/.pi untouched.
        # The key rides in the config, so HOME is the only env var needed.
        write_pi_config(base, key, entry, home / ".pi" / "agent" / "models.json")
        env = {"HOME": str(home)}
        if os.name == "nt":
            # On native Windows Node resolves ~/.pi via USERPROFILE (then HOMEDRIVE +
            # HOMEPATH), not HOME, so without these it would read the user's real
            # ~/.pi. splitdrive yields no drive off a POSIX path, so HOMEDRIVE/HOMEPATH
            # are only set when there is one.
            env["USERPROFILE"] = str(home)
            drive, tail = os.path.splitdrive(str(home))
            if drive:
                env["HOMEDRIVE"], env["HOMEPATH"] = drive, tail
        _run(base, entry, env, command, launch = launch, install_hint = install_hint)
