# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Codex desktop app (Codex.app, or ChatGPT.app on builds where Codex lives inside ChatGPT)
switched to Unsloth, and back.

Shared by `unsloth start codex --app` and Studio's Settings → Agents → Codex Desktop card, so both
write the same lines into ~/.codex/config.toml, keep the same restore record (one at a time, and
either one cleans up after the other crashed) and put the file back the same way.

The switch makes Unsloth the active provider: ``model_provider`` points at an
``[model_providers.unsloth_api]`` table that carries a per-session Studio key, ``model`` at the
default model, and ``model_catalog_json`` at a catalog of the models it lists, because the app
lists only the active provider's catalog (Studio's card lists several, through a loopback gateway
that loads the one Codex asks for; the CLI lists the one it loaded). Every line it replaces is recorded, and switching back restores exactly
those lines (or the byte-for-byte backup when nothing else changed), leaving anything the user
changed in the meantime alone.

App discovery, quitting and launching follow `ollama launch codex-app` (bundle id
com.openai.codex, ChatGPT.app or Codex.app under /Applications or ~/Applications, the
codex://threads/new?mode=codex launch URL, a graceful quit that never escalates to a kill).
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import json
import os
import plistlib
import re
import secrets
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from . import claude_desktop as _cd

PROFILE = "unsloth_api"
PROVIDER_HEADER = f"[model_providers.{PROFILE}]"
PROVIDER_TABLES = (PROVIDER_HEADER, PROVIDER_HEADER[:-1] + ".")
ENV_KEY = "UNSLOTH_STUDIO_AUTH_TOKEN"
# See the long note on this value in unsloth_cli/commands/start.py: a local first turn can be
# silent for longer than Codex's 5-minute default, and each reconnect restarts prompt processing.
STREAM_IDLE_TIMEOUT_MS = 1_200_000
CATALOG_NAME = "unsloth-model-catalog.json"
BACKUP_SUFFIX = ".unsloth-backup"
STATE_NAME = "codex.json"
KEY_NAME = "Codex app"

BUNDLE_ID = "com.openai.codex"
LAUNCH_URL = "codex://threads/new?mode=codex"
DOWNLOAD_URL = "https://chatgpt.com/download"
APP_NAMES = ("ChatGPT.app", "Codex.app")
FALLBACK_PROMPT_NAME = "codex_fallback_prompt.md"


# --------------------------------------------------------------------------------------------
# config.toml
# --------------------------------------------------------------------------------------------


def provider_table(base: str, key: Optional[str] = None) -> str:
    # The desktop app gets no shell env, so it needs the key itself rather than env_key.
    auth = f"experimental_bearer_token = {json.dumps(key)}" if key else f'env_key = "{ENV_KEY}"'
    return (
        f"{PROVIDER_HEADER}\n"
        'name = "Unsloth Studio"\n'
        f"base_url = {json.dumps(base + '/v1')}\n"
        f"{auth}\n"
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
        f"stream_idle_timeout_ms = {STREAM_IDLE_TIMEOUT_MS}\n"
    )


def merge_config(
    existing: str,
    base: str,
    key: Optional[str] = None,
) -> str:
    chunks = re.split(r"(?m)^(?=\[)", existing)  # preamble, then one chunk per table
    if key is None and not re.search(r"(?m)^\s*oss_provider\s*=", chunks[0]):
        if chunks[0] and not chunks[0].endswith("\n"):
            chunks[0] += "\n"
        chunks[0] += f'oss_provider = "{PROFILE}"\n'
    text = "".join(c for c in chunks if not c.startswith(PROVIDER_TABLES))
    if not text.endswith("\n"):
        text += "\n"
    if not text.endswith("\n\n"):
        text += "\n"
    return text + provider_table(base, key)


def set_toml_key(text: str, name: str, line: str) -> tuple:
    """Set a root key, returning (text, the line it replaced or None)."""
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=[^\r\n]*", chunks[0])
    if match is None:
        return f"{line}\n{text}", None
    chunks[0] = chunks[0][: match.start()] + line + chunks[0][match.end() :]
    return "".join(chunks), match.group(0)


def unset_toml_key(text: str, name: str, line: str, previous: Optional[str]) -> Optional[str]:
    """Put back the line ``line`` replaced, or None when the key no longer holds ``line``."""
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(rf"(?m)^[ \t]*{name}[ \t]*=([^\r\n]*)(\r?\n)?", chunks[0])
    if match is None or match.group(0).strip() != line:
        return None
    restored = "" if previous is None else previous + (match.group(2) or "")
    chunks[0] = chunks[0][: match.start()] + restored + chunks[0][match.end() :]
    return "".join(chunks)


def app_lines(model_id: str) -> dict:
    return {
        "model_provider": f'model_provider = "{PROFILE}"',
        "model": f"model = {json.dumps(model_id)}",
        "model_catalog_json": f"model_catalog_json = {json.dumps(CATALOG_NAME)}",
    }


def build(text: Optional[str], base: str, key: str, model_id: str, previous) -> tuple:
    """(new config text, the lines and provider table it replaced)."""
    previous = dict(previous or {})
    tables = [c for c in re.split(r"(?m)^(?=\[)", text or "") if c.startswith(PROVIDER_TABLES)]
    previous.setdefault("table", "".join(tables) or None)
    merged = merge_config(text or "", base, key)
    for name, line in app_lines(model_id).items():
        merged, old = set_toml_key(merged, name, line)
        previous.setdefault(name, old)
    return merged, previous


def owned_models(state: dict) -> list:
    """Every model id the switch put in Codex's catalog. A model line naming any of them is ours
    (the user picked another of our models in Codex's picker), like the launch-owned slugs of
    `ollama launch codex-app`; ``previous_models`` are ones taken off the list while connected.
    Records written before the list existed name only ``model``."""
    models = []
    listed = state.get("models") or ()
    for model in [state.get("model"), *listed, *(state.get("previous_models") or ())]:
        if isinstance(model, str) and model and model not in models:
            models.append(model)
    return models


def _model_line_owned(text: str, state: dict) -> Optional[str]:
    """The owned model line config.toml holds now, or None when the line is someone else's."""
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(r"(?m)^[ \t]*model[ \t]*=[^\r\n]*", chunks[0])
    if match is None:
        return None
    current = match.group(0).strip()
    for model_id in owned_models(state):
        if current == app_lines(model_id)["model"]:
            return current
    return None


def restore(text: str, state: dict) -> tuple:
    """(text with the switch undone, the keys left alone because they changed meanwhile)."""
    skipped = []
    chunks = re.split(r"(?m)^(?=\[)", text)
    ours = provider_table(state["base"], state["key"]).strip()

    def is_ours(chunk: str) -> bool:
        # An editor on Windows may have rewritten the file with CRLF line endings.
        return chunk.replace("\r\n", "\n").strip() == ours

    if any(is_ours(chunk) for chunk in chunks):
        table = state["previous"].get("table") or ""
        text = "".join(table if is_ours(chunk) else chunk for chunk in chunks)
    else:
        skipped.append(PROVIDER_HEADER)
    owned_model_line = _model_line_owned(text, state)
    for name, line in app_lines(state["model"]).items():
        if name == "model" and owned_model_line is not None:
            line = owned_model_line
        restored = unset_toml_key(text, name, line, state["previous"].get(name))
        if restored is None:
            skipped.append(name)
        else:
            text = restored
    return text, skipped


def retarget(text: str, state: dict, model_id: str) -> Optional[str]:
    """Point a live switch at another model, keeping what it recorded to restore. None when the
    model line is no longer one the switch wrote (the user picked a model of their own)."""
    if _model_line_owned(text, state) is None:
        return None
    chunks = re.split(r"(?m)^(?=\[)", text)
    match = re.search(r"(?m)^[ \t]*model[ \t]*=[^\r\n]*", chunks[0])
    line = app_lines(model_id)["model"]
    chunks[0] = chunks[0][: match.start()] + line + chunks[0][match.end() :]
    return "".join(chunks)


def fallback_prompt_path() -> Optional[Path]:
    """Codex's own unknown-model prompt, shipped with the CLI (see unsloth_cli/start.py)."""
    candidates = []
    with contextlib.suppress(Exception):
        spec = importlib.util.find_spec("unsloth_cli")
        if spec is not None and spec.origin:
            candidates.append(Path(spec.origin).parent / FALLBACK_PROMPT_NAME)
    candidates.append(Path(__file__).resolve().parents[3] / "unsloth_cli" / FALLBACK_PROMPT_NAME)
    return next((path for path in candidates if path.is_file()), None)


def model_catalog(
    models,
    visibility: str = "none",
    prompt: Optional[str] = None,
) -> dict:
    """Conservative metadata for Unsloth models unknown to Codex's built-in catalog: one model
    entry (``{"id", "context_length"?}``) or a list of them, listed in that order."""
    if prompt is None:
        path = fallback_prompt_path()
        if path is None:
            raise FileNotFoundError(f"{FALLBACK_PROMPT_NAME} is missing from this install.")
        prompt = path.read_text(encoding = "utf-8")
    if isinstance(models, dict):
        models = [models]
    entries = []
    for index, model in enumerate(models):
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
            # Codex lists lower priorities first, so the catalog keeps the order it was given.
            "priority": 99 + index,
            "availability_nux": None,
            "upgrade": None,
            "base_instructions": prompt,
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
        entries.append(entry)
    return {"models": entries}


def load_toml(text: Optional[str]) -> Optional[dict]:
    """The parsed config ({} for an empty file), or None when no TOML parser is available.
    Raises ValueError when the text isn't valid TOML."""
    if text is None or not text.strip():
        return {}
    try:
        import tomllib  # novermin
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None
    try:
        data = tomllib.loads(text)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return data


def same_config(path: Path, text: str, other: str) -> bool:
    if text.strip() == other.strip():
        return True
    try:
        parsed = load_toml(text)
        return parsed is not None and parsed == load_toml(other)
    except Exception:
        return False


def read_file(path: Path) -> Optional[str]:
    """The file's text, None when missing. Raises ValueError for a symlink, which is never
    rewritten (the write would replace the link, or follow it somewhere unexpected)."""
    if path.is_symlink():
        raise ValueError(
            f"{path} is a symlink, so Unsloth won't rewrite it. Replace the link with a plain "
            "file and try again."
        )
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with open(fd, encoding = "utf-8", newline = "") as handle:
        return handle.read()


write_file = _cd.write_file
file_mode = _cd.file_mode


def restore_files(
    state: dict,
    *,
    read: Callable = read_file,
    write: Callable = write_file,
    same: Callable = same_config,
) -> list:
    """Undo the switch in config.toml and remove the catalog. Returns notes for the user; the
    caller revokes state["key_id"] and removes the state file."""
    path = Path(state["path"])
    text = read(path)
    skipped = []
    if text is not None:
        new_text, skipped = restore(text, state)
        backup = Path(state["backup"]) if state.get("backup") else None
        original = read(backup) if backup else None
        if original is not None and same(path, new_text, original):
            new_text = original
        if state.get("created") and same(path, new_text, ""):
            path.unlink()
        elif new_text != text:
            write(path, new_text)
        if original is not None and new_text == original:
            if state.get("mode") is not None:
                os.chmod(path, state["mode"])
            backup.unlink(missing_ok = True)
    if "model_catalog_json" not in skipped:
        path.with_name(CATALOG_NAME).unlink(missing_ok = True)
    return [
        f"Left {name} in {path} as is: it changed while Codex was on Unsloth." for name in skipped
    ]


# --------------------------------------------------------------------------------------------
# The gateway: Studio's card lists several models, and loads the one Codex asks for
# --------------------------------------------------------------------------------------------

HEALTH_PATH = "/_unsloth/codex-health"
HEALTH_HEADER = "X-Unsloth-Codex-Gateway"


def openai_error(
    status: int,
    message: str,
    kind: str = "invalid_request_error",
) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind, "code": None}}).encode()


# An SSE comment: every event-stream parser skips it, and it keeps the connection busy while a model loads.
KEEPALIVE_COMMENT = b": keep-alive\n\n"
# Codex sends ~25k tokens of instructions and tools with every turn; see CLAUDE_MIN_CONTEXT.
CODEX_MIN_CONTEXT = 32_768


def response_failed_event(message: str) -> bytes:
    """A Responses API ``response.failed`` event: how a stream that already started reports an
    error, which Codex shows in the chat."""
    event = {
        "type": "response.failed",
        "response": {
            "id": "resp_unsloth_" + secrets.token_hex(8),
            "object": "response",
            "status": "failed",
            "error": {"code": "server_error", "message": message},
        },
    }
    return f"event: response.failed\ndata: {json.dumps(event)}\n\n".encode()


class Gateway(ThreadingHTTPServer):
    """A loopback hop between the Codex app and Studio's /v1, for a card that lists several
    models: Studio's /v1 answers only for loaded models (auto-switch is off by default), so a
    request naming one of ``models()`` that isn't loaded is held while Studio loads it, like
    `ollama launch codex-app` routing every listed model to Ollama. Everything else passes through
    untouched, streaming. Codex authenticates with the session key itself, which is also what
    the gateway forwards, so config.toml and the restore record are the same as without it."""

    daemon_threads = True

    def __init__(
        self,
        upstream: str,
        key: str,
        models: Callable,
        is_loaded: Callable,
        ensure_loaded: Callable,
        listen: tuple = ("127.0.0.1", 0),
    ):
        parsed = urlparse(upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"invalid Unsloth URL {upstream!r}")
        self.upstream = parsed
        self.upstream_key = key
        self.token = key
        self.models = models
        self.is_loaded = is_loaded
        self.ensure_loaded = ensure_loaded
        self.load_lock = threading.Lock()
        self.keepalive_seconds = _cd.KEEPALIVE_SECONDS
        self._thread = None
        super().__init__(listen, _GatewayHandler)

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "Gateway":
        self._thread = threading.Thread(
            target = self.serve_forever, args = (0.05,), name = "codex-gateway", daemon = True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
        self.server_close()

    def listed(self, requested: str) -> Optional[str]:
        """The listed model a request names, or None (pass it through as is)."""
        try:
            models = list(self.models() or ())
        except Exception:
            return None
        return requested if requested in models else None

    def ready(self, model_id: str) -> None:
        """Load ``model_id`` unless Studio has it loaded. One load at a time, so two chats on
        different models take turns instead of racing."""
        with self.load_lock:
            if self.is_loaded(model_id):
                return
            self.ensure_loaded(model_id)

    connection = _cd.Gateway.connection

    def handle_error(self, request, client_address) -> None:
        _cd.quiet_handle_error(self, request, client_address)


class _GatewayHandler(_cd._GatewayHandler):
    server: Gateway
    CLIENT_NAME = "Codex"
    MIN_CONTEXT = CODEX_MIN_CONTEXT

    def _error(self, status: int, message: str) -> None:
        self._send(status, openai_error(status, message))

    def _target(self) -> Optional[str]:
        path = self.path.split("?", 1)[0]
        if path == "/v1" or path.startswith("/v1/"):
            # Studio serves /v1 under its own base path; keep the query string.
            return self.server.upstream.path.rstrip("/") + self.path
        return None

    def do_GET(self):
        if not self._guard():
            return
        if self.path.split("?", 1)[0] == HEALTH_PATH:
            self._send(204, headers = {HEALTH_HEADER: "1"})
            return
        self._proxy("GET", b"")

    def do_DELETE(self):
        if self._guard():
            self._proxy("DELETE", self._read_body() or b"")

    def do_POST(self):
        if not self._guard():
            return
        if self._target() is None:
            self._plain(404, "404 page not found")
            return
        if not self._authorized():
            self._error(401, "Invalid Unsloth key.")
            return
        body = self._read_body()
        if body is None:
            self._error(413, f"Request exceeds {_cd.MAX_REQUEST_BODY_BYTES} bytes")
            return
        requested = None
        stream = False
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            payload = json.loads(body) if body else None
            if isinstance(payload, dict) and isinstance(payload.get("model"), str):
                requested = payload["model"]
                stream = payload.get("stream") is True
        model_id = self.server.listed(requested) if requested else None
        self.served_model = model_id or requested
        if model_id is not None and not self.server.is_loaded(model_id):
            if stream:
                self._load_then_proxy(body, model_id)
                return
            try:
                self.server.ready(model_id)
            except Exception as exc:
                self._error(409, f"Unsloth Studio couldn't load {model_id}: {exc}")
                return
        self._proxy("POST", body, authorized = True)

    def _load_then_proxy(self, body: bytes, model_id: str) -> None:
        """A streaming turn on a listed model that isn't loaded: the event stream starts at once and
        carries keep-alive comments while Studio loads it, then Studio's own stream; a failure is a
        ``response.failed`` event Codex shows, not a dropped connection."""
        self._open_stream("text/event-stream")
        self._chunk(KEEPALIVE_COMMENT)

        def fail(message: str) -> None:
            self._chunk(response_failed_event(message))
            self._end_chunks()

        error = self._load_holding(lambda: self.server.ready(model_id), KEEPALIVE_COMMENT)
        if error is not None:
            fail(f"Unsloth Studio couldn't load {model_id}: {error}")
            return
        try:
            conn, response = self._open_upstream("POST", body)
        except (OSError, http.client.HTTPException):
            fail("Unsloth Studio didn't answer.")
            return
        try:
            if response.status == 200:
                self._pipe(response)
                self._end_chunks()
                return
            detail = response.read(_cd.MAX_ERROR_BODY_BYTES)
            message = _cd.upstream_error_message(response.status, detail)
            small = self._small_context_text(message)
            fail(
                small
                if small != message
                else f"Unsloth Studio couldn't answer with {model_id}: {message}"
            )
        finally:
            conn.close()

    def _open_upstream(self, method: str, body: bytes) -> tuple:
        """(connection, response) for this request sent on to Studio's /v1 with the session key."""
        headers = self._upstream_headers(body)
        if method == "GET" and not body:
            headers.pop("Content-Type", None)
            headers.pop("Content-Length", None)
        elif self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]
        conn = self.server.connection()
        try:
            conn.request(method, self._target(), body or None, headers)
            return conn, conn.getresponse()
        except BaseException:
            conn.close()
            raise

    def _proxy(
        self,
        method: str,
        body: bytes,
        authorized: bool = False,
    ) -> None:
        if self._target() is None:
            self._plain(404, "404 page not found")
            return
        if not authorized and not self._authorized():
            self._error(401, "Invalid Unsloth key.")
            return
        try:
            conn, response = self._open_upstream(method, body)
        except (OSError, http.client.HTTPException):
            self._error(502, "Unsloth Studio didn't answer.")
            return
        try:
            self._relay(response, b"")
        finally:
            # Also how Codex hanging up cancels the generation upstream.
            conn.close()


# --------------------------------------------------------------------------------------------
# The restore record, shared with `unsloth start codex --app`
# --------------------------------------------------------------------------------------------


def state_path(agents_root: Path) -> Path:
    # The CLI's `_app_state_path("codex")`: one record, so only one switch holds the app.
    return Path(agents_root) / "app" / STATE_NAME


def read_state(path: Path) -> Optional[dict]:
    try:
        state = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or not state.get("path"):
        return None
    return state


write_state = _cd.write_state
owner_record = _cd.owner_record
owner_alive = _cd.owner_alive


def config_path(home: Optional[Path] = None) -> Path:
    # The app is opened from the Dock, so it never sees CODEX_HOME, nor a redirected $HOME.
    return Path(home or _cd.user_home()) / ".codex" / "config.toml"


# --------------------------------------------------------------------------------------------
# The app itself
# --------------------------------------------------------------------------------------------


def can_launch(platform: Optional[str] = None) -> bool:
    # `ollama launch codex-app` also drives Windows; Studio's card is macOS-only for now.
    return (platform or sys.platform) == "darwin"


def app_candidates(home: Optional[Path] = None) -> list:
    home = Path(home or _cd.user_home())
    roots = (Path("/Applications"), home / "Applications")
    return [root / name for root in roots for name in APP_NAMES]


def bundle_id(app: Path) -> Optional[str]:
    try:
        with open(Path(app) / "Contents" / "Info.plist", "rb") as handle:
            value = plistlib.load(handle).get("CFBundleIdentifier")
    except Exception:
        return None
    return value if isinstance(value, str) else None


def app_path(home: Optional[Path] = None) -> Optional[Path]:
    """The installed Codex app. ChatGPT.app counts only when it is the Codex build: the older
    ChatGPT chat app (com.openai.chat) lives at the same path and has no Codex in it."""
    for candidate in app_candidates(home):
        if candidate.is_dir() and bundle_id(candidate) == BUNDLE_ID:
            return candidate
    return None


def _process_patterns(app: Optional[Path]) -> list:
    if app is not None:
        return [f"{app}/Contents/MacOS/"]
    return [f"{name}/Contents/MacOS/{name[:-4]}" for name in APP_NAMES]


def is_running(app: Optional[Path] = None) -> bool:
    if sys.platform != "darwin":
        return False
    for pattern in _process_patterns(app or app_path()):
        try:
            result = subprocess.run(["pgrep", "-f", pattern], capture_output = True, timeout = 10)
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode == 0 and result.stdout.strip():
            return True
    return False


def quit_app(timeout: float = 30.0) -> None:
    # By bundle id, so ChatGPT.app and Codex.app are both covered; a quit the app answers with a
    # dialog would otherwise block the caller.
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        subprocess.run(
            ["osascript", "-e", f'tell application id "{BUNDLE_ID}" to quit'],
            capture_output = True,
            timeout = timeout,
            env = _cd.launch_env(),
        )


def open_app(app: Optional[Path] = None) -> None:
    """Open the app on a new Codex thread, with the environment a Dock launch would give it."""
    env = _cd.launch_env()
    commands = []
    if app is not None:
        commands.append(["/usr/bin/open", "-a", str(app), LAUNCH_URL])
        commands.append(["/usr/bin/open", str(app)])
    commands.append(["/usr/bin/open", "-b", BUNDLE_ID])
    for command in commands:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            if subprocess.run(command, capture_output = True, timeout = 30, env = env).returncode == 0:
                return
    raise OSError("Couldn't open the Codex app.")


def wait_for_exit(
    timeout: float = 30.0,
    poll: float = 0.2,
    app: Optional[Path] = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while is_running(app):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True
