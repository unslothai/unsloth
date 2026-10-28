# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The Claude Desktop profile layout, the gateway protocol (model routes, catalog shape, header
# hygiene) and the unsupported-image retry are adapted from Ollama
# (https://github.com/ollama/ollama, cmd/launch/claude_desktop.go,
# internal/proxy/claude_desktop.go and internal/proxy/claude_desktop_models.go), used under
# the MIT License:
#
#   MIT License
#
#   Copyright (c) Ollama
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

"""Claude Desktop in third-party inference mode, pointed at a local Unsloth gateway.

Claude Desktop has a native "gateway" deployment mode: it reads a profile from its ``-3p``
config library and talks the Anthropic Messages API to ``inferenceGatewayBaseUrl``. It only
accepts its own model ids (``claude-…``) and expects an Anthropic-shaped ``/v1/models``, so a
small loopback gateway maps one fixed Claude route to the model Unsloth has loaded, swaps the
placeholder key Claude holds for a real Unsloth key, and retries once without images when a
text-only model rejects them. Everything else goes to Unsloth's own ``/v1/messages`` untouched.
"""

from __future__ import annotations

import contextlib
import copy
import glob
import http.client
import json
import os
import re
import secrets
import ssl
import subprocess
import tempfile
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from typing import Callable, NamedTuple, Optional
from urllib.parse import urlparse

PROFILE_NAME = "Unsloth"
# Ollama's profile is ...0114; a separate id keeps the two launchers from rewriting each other.
PROFILE_ID = "00000000-0000-4000-8000-00000000051a"
CONFIG_NAME = "claude_desktop_config.json"
HEALTH_PATH = "/_unsloth/health"
HEALTH_HEADER = "X-Unsloth-Claude-Gateway"
# Sent on every request the gateway forwards to Studio's /v1/messages: the Desktop-only prompt
# handling (dropping Claude's per-request billing line, keeping GLM's replayed reasoning) keys on it,
# so Claude Code and other Anthropic clients get exactly the upstream behaviour.
GATEWAY_CLIENT_HEADER = "X-Unsloth-Client"
GATEWAY_CLIENT_VALUE = "claude-desktop"
UNSUPPORTED_IMAGE_NOTICE = (
    "[Image omitted by Unsloth because the selected model does not support image recognition.]"
)
MAX_REQUEST_BODY_BYTES = 64 << 20
MAX_ERROR_BODY_BYTES = 64 << 10
# While a model loads for a request, the gateway answers at once and writes one of these every few
# seconds, so Claude (or Codex) holds the request open instead of giving up on a slow load. A big
# GGUF read off an external drive can take minutes; Studio's own load route pads its reply the same way.
KEEPALIVE_SECONDS = 5.0
PING_EVENT = b'event: ping\ndata: {"type": "ping"}\n\n'
# The client hung up: nothing to report, and nothing that warrants a traceback.
CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)
DEFAULT_MAX_TOKENS = 64_000
# Cowork needs unrestricted egress for user-configured plugins and MCP servers.
EGRESS_HOSTS = ["*"]

# Claude Desktop accepts only these ids. A single chosen local model takes the first slot, the
# same assignment Ollama's SelectClaudeDesktopModels makes for a user-picked local model.
MODEL_SLOTS = (
    {
        "id": "claude-fable-5",
        "display": "Fable 5",
        "family": "fable",
        "created_at": "2026-06-09T00:00:00Z",
        "family_default": True,
    },
    {
        "id": "claude-opus-5",
        "display": "Opus 5",
        "family": "opus",
        "created_at": "2026-07-24T00:00:00Z",
        "family_default": True,
    },
    {
        "id": "claude-sonnet-5",
        "display": "Sonnet 5",
        "family": "sonnet",
        "created_at": "2026-06-30T00:00:00Z",
        "family_default": True,
    },
    {
        "id": "claude-haiku-4-5-20251001",
        "display": "Haiku 4.5",
        "family": "haiku",
        "created_at": "2025-10-01T00:00:00Z",
        "family_default": True,
    },
    {
        "id": "claude-sonnet-4-6",
        "display": "Sonnet 4.6",
        "family": "sonnet",
        "created_at": "2025-11-18T00:00:00Z",
        "family_default": False,
    },
)

_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
}
_HOP_BY_HOP = frozenset(
    (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    )
)


# --------------------------------------------------------------------------------------------
# Where Claude Desktop keeps its profiles
# --------------------------------------------------------------------------------------------


class ThirdPartyPaths(NamedTuple):
    desktop_config: Path
    meta: Path
    profile: Path


class Targets(NamedTuple):
    normal_configs: list
    third_party: list

    def files(self) -> list:
        out = list(self.normal_configs)
        for target in self.third_party:
            out += [target.desktop_config, target.meta, target.profile]
        return out


def _dedupe(paths: list) -> list:
    out, seen = [], set()
    for path in paths:
        key = str(path).lower()
        if str(path).strip() and key not in seen:
            seen.add(key)
            out.append(Path(path))
    return out


def user_home() -> Path:
    """The home the OS records for this account, not $HOME.

    Claude Desktop keeps its profile, and macOS the keychain it stores its key in, under the
    account's real home. Studio can run with HOME pointed somewhere else (a sandbox, a service
    wrapper), and following that would edit, or launch Claude against, a profile it never reads."""
    if os.name == "posix":
        with contextlib.suppress(ImportError, KeyError, OSError):
            import pwd
            home = pwd.getpwuid(os.getuid()).pw_dir
            if home:
                return Path(home)
    return Path.home()


def local_app_data(home: Optional[Path] = None) -> Path:
    if os.environ.get("LOCALAPPDATA", "").strip():
        return Path(os.environ["LOCALAPPDATA"].strip())
    if os.environ.get("USERPROFILE", "").strip():
        return Path(os.environ["USERPROFILE"].strip()) / "AppData" / "Local"
    return (home or Path.home()) / "AppData" / "Local"


def profile_roots(platform: Optional[str] = None, home: Optional[Path] = None) -> tuple:
    """(normal roots, third-party roots). The 1p profile picks the mode; the 3p one holds ours."""
    platform = platform or sys.platform
    home = home or user_home()
    if platform == "darwin":
        base = home / "Library" / "Application Support"
        return [base / "Claude"], [base / "Claude-3p"]
    if platform == "win32":
        local = local_app_data(home)
        return (
            [local / "Claude", local / "Claude Nest"],
            [local / "Claude-3p", local / "Claude Nest-3p"],
        )
    raise RuntimeError("Claude Desktop is only available on macOS and Windows.")


def target_paths(platform: Optional[str] = None, home: Optional[Path] = None) -> Targets:
    normal, third_party = profile_roots(platform, home)
    return Targets(
        [root / CONFIG_NAME for root in _dedupe(normal)],
        [
            ThirdPartyPaths(
                root / CONFIG_NAME,
                root / "configLibrary" / "_meta.json",
                root / "configLibrary" / f"{PROFILE_ID}.json",
            )
            for root in _dedupe(third_party)
        ],
    )


def app_candidates(platform: Optional[str] = None, home: Optional[Path] = None) -> list:
    platform = platform or sys.platform
    home = home or user_home()
    if platform == "darwin":
        return [Path("/Applications/Claude.app"), home / "Applications" / "Claude.app"]
    if platform == "win32":
        local = local_app_data(home)
        candidates = [
            local / "Programs" / "Claude" / "Claude.exe",
            local / "Programs" / "Claude Desktop" / "Claude.exe",
            local / "Claude" / "Claude.exe",
            local / "Claude Nest" / "Claude.exe",
            local / "Claude Desktop" / "Claude.exe",
            local / "AnthropicClaude" / "Claude.exe",
        ]
        for pattern in (
            local / "AnthropicClaude" / "app-*" / "Claude.exe",
            local / "Programs" / "Claude" / "app-*" / "Claude.exe",
            local / "Programs" / "Claude Desktop" / "app-*" / "Claude.exe",
        ):
            candidates += sorted(Path(p) for p in glob.glob(str(pattern)))
        return _dedupe(candidates)
    return []


def can_launch(platform: Optional[str] = None) -> bool:
    """`ollama launch claude-desktop` only launches on macOS; switching back also covers Windows."""
    return (platform or sys.platform) == "darwin"


def app_path(platform: Optional[str] = None, home: Optional[Path] = None) -> Optional[Path]:
    return next((p for p in app_candidates(platform, home) if p.exists()), None)


# --------------------------------------------------------------------------------------------
# Switching the profile, and switching it back
# --------------------------------------------------------------------------------------------

_MISSING = {"__unsloth_missing__": True}


def profile_values(
    base_url: str,
    token: str,
    auto_mode: bool = True,
) -> dict:
    """The keys Ollama writes into its gateway profile, with Unsloth's name and gateway."""
    return {
        "inferenceProvider": "gateway",
        "inferenceGatewayBaseUrl": base_url,
        "inferenceGatewayApiKey": token,
        "inferenceGatewayAuthScheme": "bearer",
        "deploymentDisplayName": PROFILE_NAME,
        "chatTabEnabled": True,
        "disableDeploymentModeChooser": True,
        "coworkEgressAllowedHosts": list(EGRESS_HOSTS),
        "disableEssentialTelemetry": True,
        "disableNonessentialTelemetry": True,
        "autoModeEnabled": auto_mode,
    }


# Ollama also drops a pinned model list so Claude discovers the gateway's catalog.
PROFILE_DROPPED = ("inferenceModels",)


def read_json(path: Path) -> Optional[dict]:
    """The object in path, None when missing. Raises ValueError for anything else."""
    if path.is_symlink():
        raise ValueError(f"{path} is a symlink, so Unsloth won't rewrite it.")
    try:
        text = path.read_text(encoding = "utf-8")
    except FileNotFoundError:
        return None
    if not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} isn't a settings object.")
    return data


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding = "utf-8")
    except FileNotFoundError:
        return None


def write_json(path: Path, data: dict, writer: Callable) -> None:
    writer(path, json.dumps(data, indent = 2, ensure_ascii = False) + "\n")


def _remember(previous: dict, config: dict, key: str) -> None:
    previous.setdefault(key, copy.deepcopy(config[key]) if key in config else _MISSING)


def _missing(value) -> bool:
    return value == _MISSING


def _set_keys(path: Path, values: dict, dropped: tuple, writer: Callable, record: dict) -> None:
    config = read_json(path) or {}
    previous = record.setdefault("previous", {})
    for key, value in values.items():
        _remember(previous, config, key)
        config[key] = value
    for key in dropped:
        if key in config:
            _remember(previous, config, key)
            config.pop(key)
    record["wrote"] = {**record.get("wrote", {}), **values}
    write_json(path, config, writer)


def snapshot(targets: Targets) -> dict:
    """Original text of every file this touches (None when missing), for an exact restore."""
    files = {}
    for path in targets.files():
        files[str(path)] = {"original": read_text(path) if not path.is_symlink() else None}
        read_json(path)  # refuse a symlink or a file Claude could not parse, before writing any
    return files


def apply_profile(
    targets: Targets, files: dict, base_url: str, token: str, writer: Callable
) -> None:
    """Point Claude Desktop at the gateway, recording each key's previous value in files."""
    values = profile_values(base_url, token)
    for target in targets.third_party:
        _set_keys(target.profile, values, PROFILE_DROPPED, writer, files[str(target.profile)])
        meta_record = files[str(target.meta)]
        meta = read_json(target.meta) or {}
        _remember(meta_record.setdefault("previous", {}), meta, "appliedId")
        entries = [
            e
            for e in meta.get("entries") or []
            if not (isinstance(e, dict) and e.get("id") == PROFILE_ID)
        ]
        meta["appliedId"] = PROFILE_ID
        meta["entries"] = [*entries, {"id": PROFILE_ID, "name": PROFILE_NAME}]
        meta_record["wrote"] = {"appliedId": PROFILE_ID}
        write_json(target.meta, meta, writer)
        _set_keys(
            target.desktop_config,
            {"deploymentMode": "3p"},
            (),
            writer,
            files[str(target.desktop_config)],
        )
    for path in targets.normal_configs:
        _set_keys(path, {"deploymentMode": "3p"}, (), writer, files[str(path)])


def _restore_keys(config: dict, record: dict, notes: list, path: Path) -> None:
    for key, wrote in (record.get("wrote") or {}).items():
        if config.get(key, _MISSING) != wrote:
            notes.append(f"Left {key} in {path} as is: it changed while Claude was on Unsloth.")
            continue
        previous = (record.get("previous") or {}).get(key, _MISSING)
        if _missing(previous):
            config.pop(key, None)
        else:
            config[key] = previous
    for key in PROFILE_DROPPED:
        previous = (record.get("previous") or {}).get(key, _MISSING)
        if not _missing(previous) and key not in config:
            config[key] = previous


def _same(text: Optional[str], config: dict) -> bool:
    try:
        return text is not None and (json.loads(text) if text.strip() else {}) == config
    except ValueError:
        return False


def restore_profile(files: dict, writer: Callable) -> list:
    """Put back what apply_profile changed, byte for byte when nothing else moved."""
    notes = []
    for name, record in files.items():
        path = Path(name)
        if "wrote" not in record:
            continue
        try:
            config = read_json(path)
        except ValueError:
            notes.append(f"Couldn't parse {path}, so it was left as is.")
            continue
        if config is None:
            continue
        if path.name == "_meta.json":
            entries = config.get("entries")
            if isinstance(entries, list):
                config["entries"] = [
                    e for e in entries if not (isinstance(e, dict) and e.get("id") == PROFILE_ID)
                ]
        _restore_keys(config, record, notes, path)
        original = record.get("original")
        if original is None and (not config or config == {"entries": []}):
            path.unlink(missing_ok = True)
        elif _same(original, config):
            if read_text(path) != original:
                writer(path, original)
        else:
            write_json(path, config, writer)
    return notes


# --------------------------------------------------------------------------------------------
# The running app (macOS, like `ollama launch claude-desktop`)
# --------------------------------------------------------------------------------------------


def is_running() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        result = subprocess.run(
            ["pgrep", "-f", "Claude.app/Contents/MacOS/Claude"], capture_output = True, timeout = 10
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def quit_app(timeout: float = 30.0) -> None:
    # A quit Claude answers with a dialog (an unsent draft) would otherwise block the caller.
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            ["osascript", "-e", 'tell application "Claude" to quit'],
            capture_output = True,
            timeout = timeout,
            env = launch_env(),
        )


# What a Dock launch would hand Claude anyway. Everything else in Studio's environment (its venv on
# PATH, VIRTUAL_ENV, HF_HOME, UNSLOTH_*, a sandboxed HOME) must not reach Claude: `open` passes the
# caller's environment to the app it starts, and Claude passes it on to the MCP servers and tools it
# spawns. A wrong HOME also sends macOS looking for a keychain that is not there.
LAUNCH_ENV_PASSTHROUGH = (
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSH_AUTH_SOCK",
    "__CF_USER_TEXT_ENCODING",
)
LAUNCH_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


def launch_env(environ: Optional[dict] = None, home: Optional[Path] = None) -> dict:
    """A clean user environment for starting Claude Desktop, rooted at the account's real home."""
    environ = os.environ if environ is None else environ
    env = {name: environ[name] for name in LAUNCH_ENV_PASSTHROUGH if environ.get(name)}
    env["HOME"] = str(home or user_home())
    env["PATH"] = LAUNCH_PATH
    if os.name == "posix":
        with contextlib.suppress(ImportError, KeyError, OSError):
            import pwd

            entry = pwd.getpwuid(os.getuid())
            env["USER"] = env["LOGNAME"] = entry.pw_name
            if entry.pw_shell:
                env["SHELL"] = entry.pw_shell
    return env


def open_app(path: Path) -> None:
    subprocess.run(["/usr/bin/open", str(path)], capture_output = True, timeout = 30, env = launch_env())


def wait_for_exit(timeout: float = 30.0, poll: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout
    while is_running():
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True


# --------------------------------------------------------------------------------------------
# The gateway Claude Desktop talks to
# --------------------------------------------------------------------------------------------


def anthropic_error(status: int, message: str) -> bytes:
    kind = _ERROR_TYPES.get(status, "api_error" if status >= 500 else "invalid_request_error")
    return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()


def message_rejects_images(message: str) -> bool:
    """Narrow on purpose, so a generic 500 is never retried."""
    lower = message.lower()
    if "mmproj" in lower and "image" in lower:
        return True
    mentions = "image" in lower or "vision" in lower
    unsupported = "does not support" in lower or "not support" in lower or "unsupported" in lower
    return mentions and unsupported


def response_rejects_images(status: int, body: bytes) -> bool:
    try:
        data = json.loads(body)
    except ValueError:
        return False
    if not isinstance(data, dict) or not isinstance(data.get("error"), dict):
        return False
    error = data["error"]
    message = error.get("message") if isinstance(error.get("message"), str) else ""
    if status == 400:
        return (
            data.get("type") == "error"
            and error.get("type") == "invalid_request_error"
            and message_rejects_images(message)
        )
    # A llama.cpp 500 for a missing projector, either in an Anthropic api_error envelope (the
    # message is often the nested llama.cpp JSON) or forwarded raw as a server_error.
    if data.get("type") == "error" and error.get("type") in ("api_error", "invalid_request_error"):
        return message_rejects_images(message)
    return error.get("type") == "server_error" and message_rejects_images(message)


def _replace_images(content):
    if not isinstance(content, list):
        return content, False
    changed, blocks = False, []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            blocks.append({"type": "text", "text": UNSUPPORTED_IMAGE_NOTICE})
            changed = True
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            inner, inner_changed = _replace_images(block.get("content"))
            blocks.append({**block, "content": inner} if inner_changed else block)
            changed = changed or inner_changed
        else:
            blocks.append(block)
    return blocks, changed


def replace_unsupported_images(payload: dict) -> bool:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    changed = False
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content, message_changed = _replace_images(message["content"])
            if message_changed:
                message["content"] = content
                changed = True
    return changed


class GatewayModel(NamedTuple):
    route: str
    model_id: str
    max_tokens: int
    supports_vision: bool = False

    def catalog_entry(self) -> dict:
        slot = next(s for s in MODEL_SLOTS if s["id"] == self.route)
        return {
            "id": self.route,
            "type": "model",
            "display_name": self.model_id,
            "created_at": slot["created_at"],
            "max_tokens": self.max_tokens,
            "anthropic_family_tier": slot["family"],
            "is_family_default": slot["family_default"],
        }


def gateway_model(entry: dict) -> GatewayModel:
    window = entry.get("context_length") or entry.get("max_context_length")
    max_tokens = DEFAULT_MAX_TOKENS
    if isinstance(window, int) and 0 < window < max_tokens:
        max_tokens = window
    return GatewayModel(MODEL_SLOTS[0]["id"], entry["id"], max_tokens)


NO_MODEL_MESSAGE = "No model is loaded in Unsloth Studio. Load a model in Studio, then try again."

# A slot mapped to this follows whatever Studio has loaded, the behavior before slots could be
# mapped. It can't collide with a model id: no catalog id starts with a colon.
FOLLOW_LOADED = ":loaded"
# Nothing chosen yet: the first slot follows the loaded model, as `ollama launch claude-desktop`
# gives a single user-picked local model the first slot.
DEFAULT_MAPPINGS = {MODEL_SLOTS[0]["id"]: FOLLOW_LOADED}
ROUTE_IDS = tuple(slot["id"] for slot in MODEL_SLOTS)
MAX_MODEL_ID_LENGTH = 512


def normalize_mappings(mappings: object) -> dict:
    """Known slots mapped to a non-empty model id (or FOLLOW_LOADED), in Claude's slot order.
    Empty slots are dropped: like Ollama's SaveClaudeDesktopModelMappings, sparse is valid."""
    if not isinstance(mappings, dict):
        return {}
    out = {}
    for route in ROUTE_IDS:
        value = mappings.get(route)
        if isinstance(value, str) and value.strip() and len(value) <= MAX_MODEL_ID_LENGTH:
            out[route] = value.strip()
    return out


def route_names() -> list:
    return [{"id": slot["id"], "name": slot["display"]} for slot in MODEL_SLOTS]


def no_model_reply(
    requested: str,
    stream: bool,
    text: str = NO_MODEL_MESSAGE,
) -> tuple:
    """(content type, body) of an assistant turn that says no model is loaded.

    An error here reaches the user as Claude's opaque "provider rejected a test request" (its
    connection check sends one) or a failed turn, never as our message. A short reply that ends the
    turn tells them what to do, in the chat they are looking at, and nothing reaches a model."""
    message_id = "msg_unsloth_no_model_" + secrets.token_hex(8)
    usage = {"input_tokens": 0, "output_tokens": 0}
    message = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": requested,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }
    if not stream:
        return "application/json", json.dumps(message).encode()
    events = [
        (
            "message_start",
            {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return "text/event-stream", body.encode()


def _placeholder_model(route: str = MODEL_SLOTS[0]["id"]) -> GatewayModel:
    return GatewayModel(route, "Unsloth (no model loaded)", DEFAULT_MAX_TOKENS)


def load_model(
    upstream: str,
    key: str,
    model_id: str,
    timeout: float = 1800.0,
) -> None:
    """Load a downloaded model in Studio by its catalog id, with its saved launch settings: the
    same POST /models/load Open WebUI uses, so "Keep multiple models loaded" decides whether it
    loads beside the resident model or replaces it. Raises RuntimeError with Studio's reason."""
    parsed = urlparse(upstream)
    if parsed.scheme == "https":
        context = None
        if _is_loopback_host(parsed.hostname):
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(
            parsed.hostname, parsed.port, timeout = timeout, context = context
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout = timeout)
    body = json.dumps({"model": model_id}).encode()
    try:
        conn.request(
            "POST",
            parsed.path.rstrip("/") + "/models/load",
            body,
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = conn.getresponse()
        raw = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"Unsloth Studio didn't answer: {exc}") from exc
    finally:
        conn.close()
    # A long load may be padded with leading whitespace to keep a tunnel open.
    text = raw.decode("utf-8", "replace").strip()
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        data = {}
    if response.status == 200 and isinstance(data, dict) and data.get("success", True):
        return
    detail = data.get("detail") if isinstance(data, dict) else None
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error") or json.dumps(detail)
    if response.status == 404:
        detail = "it isn't downloaded on this computer"
    raise RuntimeError(str(detail or f"HTTP {response.status}"))


def loaded_matches(model_id: str, loaded: list) -> bool:
    """Whether ``model_id`` (a catalog id, maybe pinned as ``id:QUANT``) is among the loaded model
    objects (``{"id", "quant"?}``)."""
    wanted = model_id.strip().lower()
    base, _, quant = wanted.rpartition(":")
    for entry in loaded or ():
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        current = entry["id"].strip().lower()
        if current == wanted:
            return True
        if base and current == base and str(entry.get("quant") or "").lower() == quant:
            return True
    return False


class Gateway(ThreadingHTTPServer):
    """``model`` is fixed (the CLI pins one) or a callable, so Studio follows whatever it has loaded."""

    daemon_threads = True

    def __init__(
        self,
        upstream: str,
        upstream_key: str,
        model,
        token: str,
        listen: tuple = ("127.0.0.1", 0),
        *,
        mappings = None,
        is_loaded: Optional[Callable] = None,
        ensure_loaded: Optional[Callable] = None,
    ):
        parsed = urlparse(upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"invalid Unsloth URL {upstream!r}")
        self.upstream = parsed
        self.upstream_key = upstream_key
        self.model = model
        # Claude slot → Studio model id (or FOLLOW_LOADED): a dict, or a callable so Studio's
        # card can change it while connected. None keeps the one-slot "follow" behavior.
        self.mappings = mappings
        self.is_loaded = is_loaded
        self.ensure_loaded = ensure_loaded
        self.load_lock = threading.Lock()
        self.keepalive_seconds = KEEPALIVE_SECONDS
        self.token = token
        self.routed = 0
        self._thread: Optional[threading.Thread] = None
        super().__init__(listen, _GatewayHandler)

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "Gateway":
        self._thread = threading.Thread(
            target = self.serve_forever, args = (0.05,), name = "claude-gateway", daemon = True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.shutdown()
        self.server_close()

    def handle_error(self, request, client_address) -> None:
        quiet_handle_error(self, request, client_address)

    def current_model(self) -> Optional[GatewayModel]:
        if isinstance(self.model, GatewayModel):
            return self.model
        try:
            return self.model()
        except Exception:
            return None

    def current_mappings(self) -> dict:
        mappings = self.mappings
        if callable(mappings):
            try:
                mappings = mappings()
            except Exception:
                mappings = None
        return normalize_mappings(mappings) or dict(DEFAULT_MAPPINGS)

    def routes(self) -> list:
        """[(slot id, GatewayModel or None, pinned model id or None)] for every mapped slot, in
        Claude's slot order. A following slot carries the loaded model (None when nothing is)."""
        out = []
        for route, target in self.current_mappings().items():
            if target == FOLLOW_LOADED:
                current = self.current_model()
                out.append((route, current._replace(route = route) if current else None, None))
            else:
                out.append((route, GatewayModel(route, target, DEFAULT_MAX_TOKENS), target))
        return out

    def resolve(self, requested: str) -> Optional[tuple]:
        """The route a request names, by slot id or by the model id it shows as."""
        routes = self.routes()
        for route in routes:
            if route[0] == requested:
                return route
        for route in routes:
            if route[1] is not None and route[1].model_id == requested:
                return route
        return None

    def ready(self, model_id: str) -> None:
        """Load a pinned slot's model if Studio doesn't have it loaded. One load at a time, so two
        slots on different models take turns instead of racing."""
        if self.ensure_loaded is None:
            return
        with self.load_lock:
            if self.is_loaded is not None and self.is_loaded(model_id):
                return
            self.ensure_loaded(model_id)

    def connection(self) -> http.client.HTTPConnection:
        if self.upstream.scheme != "https":
            return http.client.HTTPConnection(
                self.upstream.hostname, self.upstream.port, timeout = 3600
            )
        context = None
        if _is_loopback_host(self.upstream.hostname):
            # Studio's own TLS listener uses a self-signed certificate; this hop never leaves the machine.
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return http.client.HTTPSConnection(
            self.upstream.hostname, self.upstream.port, timeout = 3600, context = context
        )


def quiet_handle_error(server: ThreadingHTTPServer, request, client_address) -> None:
    """A client that hung up mid-reply (Claude giving up on a slow load, a closed chat) is routine;
    anything else still gets the standard traceback."""
    if isinstance(sys.exc_info()[1], CLIENT_GONE):
        return
    ThreadingHTTPServer.handle_error(server, request, client_address)


def token_estimate(body_length: int) -> bytes:
    """A rough ``count_tokens`` answer, for when the real count would mean loading or waiting."""
    return json.dumps({"input_tokens": max(1, body_length // 4)}).encode()


# Claude Desktop sends ~55-62k tokens of system prompt and tools with every message. A model loaded
# with less than this can't answer it however short the chat, so its overflow gets its own message.
CLAUDE_MIN_CONTEXT = 65_536
# Every wording Studio and llama-server give a prompt that overflows the loaded window.
_OVERFLOW_PATTERNS = (
    re.compile(r"request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)"),
    re.compile(r"prompt is too long: (\d+) tokens? > (\d+) maximum", re.IGNORECASE),
    re.compile(r"(\d+) tokens? exceeds? the (\d+)-token context window"),
)


def overflow_counts(text: str) -> Optional[tuple]:
    """(prompt tokens, loaded context) from an over-context error message, or None."""
    for pattern in _OVERFLOW_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def suggested_context(prompt_tokens: int, minimum: int) -> int:
    """The next power of two that fits the prompt with room for a reply and a few turns."""
    target = max(minimum, prompt_tokens + prompt_tokens // 4)
    return 1 << (target - 1).bit_length()


def small_context_message(
    client: str, model: Optional[str], prompt_tokens: int, context: int, minimum: int
) -> str:
    name = model or "The loaded model"
    return (
        f"{name} is loaded with a context of {context:,} tokens, but {client} needs "
        f"{prompt_tokens:,} tokens for this message, so shortening the chat won't help. In "
        f"Unsloth Studio, open Run settings, set Context Length to at least "
        f'{suggested_context(prompt_tokens, minimum):,}, tick "Remember for this model", '
        f"reload the model and send the message again."
    )


def rewrite_small_context(data: object, client: str, model: Optional[str], minimum: int) -> bool:
    """Replace, in place, any error ``message``/``detail`` in ``data`` that reports an overflow of
    a window smaller than ``minimum``. True when something changed. A bigger window that overflowed
    is a long chat, and its message (which clients key on to compact) passes through untouched."""
    changed = False
    if isinstance(data, list):
        for item in data:
            changed = rewrite_small_context(item, client, model, minimum) or changed
    elif isinstance(data, dict):
        for key, value in data.items():
            if key in ("message", "detail") and isinstance(value, str):
                counts = overflow_counts(value)
                if counts and counts[1] < minimum:
                    data[key] = small_context_message(client, model, counts[0], counts[1], minimum)
                    changed = True
            else:
                changed = rewrite_small_context(value, client, model, minimum) or changed
    return changed


def rewrite_small_context_body(
    body: bytes, client: str, model: Optional[str], minimum: int
) -> Optional[bytes]:
    """A JSON error body with the small-context message, or None to send ``body`` as it is."""
    if b"token" not in body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not rewrite_small_context(data, client, model, minimum):
        return None
    return json.dumps(data).encode()


def rewrite_small_context_event(
    event: bytes, client: str, model: Optional[str], minimum: int
) -> Optional[bytes]:
    """One server-sent event (without its blank-line terminator) with the small-context message,
    or None to send it as it is."""
    if b"token" not in event:
        return None
    lines = event.decode("utf-8", "replace").splitlines()
    data_lines = [line[5:].lstrip(" ") for line in lines if line.startswith("data:")]
    if not data_lines:
        return None
    try:
        data = json.loads("\n".join(data_lines))
    except ValueError:
        return None
    if not rewrite_small_context(data, client, model, minimum):
        return None
    kept = [line for line in lines if not line.startswith("data:")]
    return ("\n".join(kept + ["data: " + json.dumps(data)])).encode()


def _event_end(buffer: bytes) -> Optional[tuple]:
    """(end of the first complete event, start of the next) in an SSE buffer, or None."""
    ends = [(i, i + len(sep)) for sep in (b"\n\n", b"\r\n\r\n") if (i := buffer.find(sep)) >= 0]
    return min(ends) if ends else None


def upstream_error_message(status: int, body: bytes) -> str:
    """The human part of an error body from Studio (FastAPI ``detail``, OpenAI or Anthropic shape)."""
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    message = None
    if isinstance(data, dict):
        error = data.get("error")
        detail = data.get("detail")
        if isinstance(error, dict):
            message = error.get("message")
        elif isinstance(error, str):
            message = error
        if message is None and isinstance(detail, dict):
            message = detail.get("message")
        elif message is None and isinstance(detail, str):
            message = detail
    if not isinstance(message, str) or not message.strip():
        message = body.decode("utf-8", "replace").strip()[:300] or f"HTTP {status}"
    return message.strip()


def _is_loopback_host(host: Optional[str]) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def probe(base_url: str, timeout: float = 2.0) -> bool:
    """True when base_url is this gateway, not some other process on the port."""
    parsed = urlparse(base_url)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout = timeout)
        conn.request("GET", HEALTH_PATH)
        response = conn.getresponse()
        response.read()
        return response.status == 204 and response.getheader(HEALTH_HEADER) == "1"
    except (OSError, http.client.HTTPException):
        return False


class _GatewayHandler(BaseHTTPRequestHandler):
    # Who the small-context message names, and the window below which it applies.
    CLIENT_NAME = "Claude Desktop"
    MIN_CONTEXT = CLAUDE_MIN_CONTEXT
    served_model: Optional[str] = None
    protocol_version = "HTTP/1.1"
    server: Gateway

    def log_message(self, format, *args):  # noqa: A002 - quiet: the CLI owns the terminal
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except CLIENT_GONE:
            # Closing our end also drops the upstream request (see _forward), so Studio stops generating.
            self.close_connection = True

    # -- plumbing ---------------------------------------------------------------------------

    def _open_stream(self, content_type: str) -> None:
        """Commit to a 200 now, body to follow in chunks: used while a model loads."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _load_holding(self, load: Callable, keepalive: bytes) -> Optional[BaseException]:
        """Run ``load`` and write ``keepalive`` every few seconds until it finishes; return its error.

        If the client hangs up, the write raises (CLIENT_GONE) and this request ends there, nothing
        sent upstream. The load itself runs on, so a retry finds the model loaded instead of starting
        over."""
        done = threading.Event()
        outcome: dict = {}

        def run() -> None:
            try:
                load()
            except BaseException as exc:  # noqa: BLE001 -- reported to the client as the turn's text
                outcome["error"] = exc
            finally:
                done.set()

        threading.Thread(target = run, name = "gateway-load", daemon = True).start()
        while not done.wait(self.server.keepalive_seconds):
            self._chunk(keepalive)
        return outcome.get("error")

    def _pipe(
        self,
        response: http.client.HTTPResponse,
        prefix: bytes = b"",
    ) -> None:
        """Copy an upstream body into an already open chunked reply, as it arrives."""
        for data in self._body(response, prefix):
            self._chunk(data)

    def _small_context_body(self, body: bytes) -> Optional[bytes]:
        return rewrite_small_context_body(
            body, self.CLIENT_NAME, self.served_model, self.MIN_CONTEXT
        )

    def _small_context_text(self, message: str) -> str:
        """``message`` (an upstream error, as a turn's text) in the small-context wording if it is one."""
        counts = overflow_counts(message)
        if counts and counts[1] < self.MIN_CONTEXT:
            return small_context_message(
                self.CLIENT_NAME, self.served_model, counts[0], counts[1], self.MIN_CONTEXT
            )
        return message

    def _body(
        self,
        response: http.client.HTTPResponse,
        prefix: bytes = b"",
    ):
        """The upstream body as it arrives. An event stream goes out event by event, so an
        over-context error inside it (Studio's 200 streams report failures in-band) can be
        reworded when the loaded window is too small for this client ever to fit."""
        if "text/event-stream" not in (response.getheader("Content-Type") or ""):
            if prefix:
                yield prefix
            while True:
                data = response.read1(65536)
                if not data:
                    return
                yield data
        buffer = prefix
        while True:
            data = response.read1(65536)
            buffer += data
            out = []
            while (bounds := _event_end(buffer)) is not None:
                event, separator = buffer[: bounds[0]], buffer[bounds[0] : bounds[1]]
                buffer = buffer[bounds[1] :]
                fixed = rewrite_small_context_event(
                    event, self.CLIENT_NAME, self.served_model, self.MIN_CONTEXT
                )
                out.append((fixed if fixed is not None else event) + separator)
            if not data or len(buffer) > MAX_ERROR_BODY_BYTES:
                # A huge event is no error: send what there is and keep going.
                out.append(buffer)
                buffer = b""
            if out:
                yield b"".join(out)
            if not data:
                return

    def _send(
        self,
        status: int,
        body: bytes = b"",
        content_type: str = "application/json",
        headers: Optional[dict] = None,
    ) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if body or status not in (204, 304):
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, anthropic_error(status, message))

    def _plain(
        self,
        status: int,
        message: str,
        headers: Optional[dict] = None,
    ) -> None:
        self._send(status, (message + "\n").encode(), "text/plain; charset=utf-8", headers)

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host", "")
        name, _, port = host.rpartition(":")
        if port != str(self.server.server_address[1]):
            return False
        name = name.strip("[]")
        if name.lower() == "localhost":
            return True
        try:
            return ip_address(name).is_loopback
        except ValueError:
            return False

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        offered = (
            auth[7:].strip() if auth[:7].lower() == "bearer " else self.headers.get("x-api-key", "")
        )
        return secrets.compare_digest(offered.encode(), self.server.token.encode())

    def _read_body(self) -> Optional[bytes]:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks, size = [], 0
            while True:
                length = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if length == 0:
                    self.rfile.readline()
                    break
                size += length
                if size > MAX_REQUEST_BODY_BYTES:
                    return None
                chunks.append(self.rfile.read(length))
                self.rfile.readline()
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_REQUEST_BODY_BYTES:
            return None
        return self.rfile.read(length)

    # -- routes -----------------------------------------------------------------------------

    def _guard(self) -> bool:
        if not self._host_allowed():
            self._plain(403, "forbidden")
            return False
        # Claude Desktop uses a native HTTP client, not a browser, so a request with an Origin
        # is a web page trying to reach this loopback port.
        if self.headers.get("Origin"):
            self._plain(403, "forbidden")
            return False
        return True

    def do_GET(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path == HEALTH_PATH:
            self._send(204, headers = {HEALTH_HEADER: "1"})
            return
        if path in ("/v1/messages", "/v1/messages/count_tokens"):
            self._plain(405, "method not allowed", {"Allow": "POST"})
            return
        if path != "/v1/models":
            self._plain(404, "404 page not found")
            return
        if not self._authorized():
            self._error(401, "Invalid gateway key.")
            return
        entries = [
            (model or _placeholder_model(route)).catalog_entry()
            for route, model, _ in self.server.routes()
        ]
        catalog = {
            "data": entries,
            "first_id": entries[0]["id"] if entries else None,
            "last_id": entries[-1]["id"] if entries else None,
            "has_more": False,
        }
        self._send(200, json.dumps(catalog).encode())

    def do_POST(self):
        if not self._guard():
            return
        path = self.path.split("?", 1)[0]
        if path == HEALTH_PATH or path == "/v1/models":
            self._plain(405, "method not allowed", {"Allow": "GET"})
            return
        if path not in ("/v1/messages", "/v1/messages/count_tokens"):
            self._plain(404, "404 page not found")
            return
        if not self._authorized():
            self._error(401, "Invalid gateway key.")
            return
        body = self._read_body()
        if body is None:
            self._error(413, f"Claude gateway request exceeds {MAX_REQUEST_BODY_BYTES} bytes")
            return
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            self._error(400, "decode Claude gateway request: invalid JSON")
            return
        requested = payload.get("model")
        if not isinstance(requested, str) or not requested.strip():
            self._error(400, "decode Claude gateway request: model is required")
            return
        resolved = self.server.resolve(requested)
        if resolved is None:
            self._error(400, f'unknown Claude model "{requested}"')
            return
        _, model, pinned = resolved
        estimate = token_estimate(len(body))
        stream = payload.get("stream") is True
        if model is None:
            if path == "/v1/messages/count_tokens":
                # A rough count keeps Claude's preflight from failing before the turn that explains.
                self._send(200, estimate)
                return
            content_type, reply = no_model_reply(requested, stream)
            self._send(200, reply, content_type, {"Cache-Control": "no-cache"})
            return
        if pinned is not None:
            loaded = self.server.is_loaded is not None and self.server.is_loaded(pinned)
            if path == "/v1/messages/count_tokens":
                if not loaded or self.server.load_lock.locked():
                    # Never load a model for a preflight count, nor wait on a load for one.
                    self._send(200, estimate)
                    return
            elif self.server.ensure_loaded is not None and not loaded:
                self._load_then_forward(path, payload, model, pinned, requested, stream)
                return
        payload["model"] = model.model_id
        self.served_model = pinned or model.model_id
        self.server.routed += 1
        self._forward(path, payload, model)

    def _load_then_forward(
        self,
        path: str,
        payload: dict,
        model: "GatewayModel",
        pinned: str,
        requested: str,
        stream: bool,
    ) -> None:
        """Load ``pinned``, then answer, without leaving Claude waiting in silence: the reply starts
        at once, with ping events (streaming) or JSON-legal leading spaces (not streaming) until the
        model is ready. Anything that goes wrong after that is a turn that says what happened."""
        if stream:
            self._open_stream("text/event-stream")
            keepalive = PING_EVENT
        else:
            self._open_stream("application/json")
            keepalive = b" "
        self._chunk(keepalive)

        def say(text: str) -> None:
            self._chunk(no_model_reply(requested, stream, text)[1])
            self._end_chunks()

        error = self._load_holding(lambda: self.server.ready(pinned), keepalive)
        if error is not None:
            say(f"Unsloth Studio couldn't load {pinned} for this chat: {error}")
            return
        payload["model"] = model.model_id
        self.served_model = pinned
        self.server.routed += 1
        try:
            conn, response, prefix = self._upstream(path, payload, model)
        except (OSError, http.client.HTTPException):
            say("Unsloth Studio didn't answer. Check that Studio is running, then try again.")
            return
        try:
            if response.status == 200:
                self._pipe(response, prefix)
                self._end_chunks()
                return
            detail = prefix + response.read(max(0, MAX_ERROR_BODY_BYTES - len(prefix)))
            message = upstream_error_message(response.status, detail)
            small = self._small_context_text(message)
            say(
                small
                if small != message
                else f"Unsloth Studio couldn't answer with {pinned}: {message}"
            )
        finally:
            conn.close()

    def _upstream_headers(self, body: bytes) -> dict:
        # Claude authenticates to this gateway with a placeholder; never forward it (or cookies).
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower()
            not in _HOP_BY_HOP
            | {"host", "authorization", "cookie", "x-api-key", "origin", "content-type"}
            # Plain bodies, so the image-rejection check can read an error.
            | {"accept-encoding"}
        }
        headers["Authorization"] = f"Bearer {self.server.upstream_key}"
        headers[GATEWAY_CLIENT_HEADER] = GATEWAY_CLIENT_VALUE
        headers["Content-Length"] = str(len(body))
        headers["Content-Type"] = "application/json"
        return headers

    def _round_trip(self, path: str, payload: dict):
        body = json.dumps(payload).encode()
        conn = self.server.connection()
        conn.request(
            "POST", self.server.upstream.path.rstrip("/") + path, body, self._upstream_headers(body)
        )
        return conn, conn.getresponse()

    def _upstream(self, path: str, payload: dict, model: GatewayModel) -> tuple:
        """(connection, response, already-read body prefix) for ``payload`` sent to Studio. A model
        without vision that rejects an image is asked again with the images replaced by a notice."""
        original = copy.deepcopy(payload)
        conn, response = self._round_trip(path, payload)
        prefix = b""
        try:
            if (
                path == "/v1/messages"
                and response.status in (400, 500)
                and not model.supports_vision
            ):
                prefix = response.read(MAX_ERROR_BODY_BYTES + 1)
                retry = copy.deepcopy(original)
                if (
                    len(prefix) <= MAX_ERROR_BODY_BYTES
                    and response_rejects_images(response.status, prefix)
                    and replace_unsupported_images(retry)
                ):
                    try:
                        retry_conn, retry_response = self._round_trip(path, retry)
                    except (OSError, http.client.HTTPException):
                        retry_conn = retry_response = None
                    if retry_response is not None:
                        conn.close()
                        conn, response, prefix = retry_conn, retry_response, b""
        except BaseException:
            conn.close()
            raise
        return conn, response, prefix

    def _forward(self, path: str, payload: dict, model: GatewayModel) -> None:
        try:
            conn, response, prefix = self._upstream(path, payload, model)
        except (OSError, http.client.HTTPException):
            self._plain(502, "Unsloth gateway unavailable")
            return
        try:
            if path == "/v1/messages/count_tokens" and response.status >= 500:
                # Studio can't count right now (a model is loading or switching): a rough count
                # keeps Claude's preflight from failing the turn.
                self._send(200, token_estimate(len(json.dumps(payload))))
                return
            self._relay(response, prefix)
        finally:
            # Also how a client that hung up cancels the generation upstream.
            conn.close()

    def _relay(self, response: http.client.HTTPResponse, prefix: bytes) -> None:
        if response.status >= 400 and len(prefix) <= MAX_ERROR_BODY_BYTES:
            prefix += response.read(MAX_ERROR_BODY_BYTES + 1 - len(prefix))
            fixed = (
                self._small_context_body(prefix) if len(prefix) <= MAX_ERROR_BODY_BYTES else None
            )
            if fixed is not None:
                self._send(
                    response.status,
                    fixed,
                    response.getheader("Content-Type") or "application/json",
                )
                return
        self.send_response(response.status, response.reason)
        length = response.getheader("Content-Length")
        for name, value in response.getheaders():
            if name.lower() not in _HOP_BY_HOP | {"content-length"}:
                self.send_header(name, value)
        # An event stream may be reworded on the way (see _body), so its length can change.
        chunked = length is None or "text/event-stream" in (
            response.getheader("Content-Type") or ""
        )
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", length)
        self.end_headers()

        def write(data: bytes) -> None:
            if not data:
                return
            if chunked:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            else:
                self.wfile.write(data)
            self.wfile.flush()

        try:
            # read1 returns what has arrived, so server-sent events stream through.
            for data in self._body(response, prefix):
                write(data)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def new_token() -> str:
    return "unsloth-claude-" + secrets.token_urlsafe(24)


# --------------------------------------------------------------------------------------------
# One switch, shared by `unsloth start claude --app` and the Studio app
# --------------------------------------------------------------------------------------------

DOWNLOAD_URL = "https://claude.ai/download"
KEY_NAME = "Claude Desktop"
BACKUP_SUFFIX = ".unsloth-backup"
STATE_NAME = "claude-desktop.json"


def state_path(agents_root: Path) -> Path:
    """Where a live switch is recorded. The CLI and Studio share it, so only one runs at a time and
    either one switches the app back after the other crashed."""
    return Path(agents_root) / "app" / STATE_NAME


def write_file(path: Path, text: str) -> None:
    """Never world-readable, even briefly; refuses a symlink instead of writing through it."""
    if path.is_symlink():
        raise ValueError(f"{path} is a symlink, so Unsloth won't rewrite it.")
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


def file_mode(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mode & 0o777
    except FileNotFoundError:
        return None


def write_state(path: Path, state: dict) -> None:
    # It holds the Unsloth key the gateway uses, so 0600 from the first byte.
    path.parent.mkdir(parents = True, exist_ok = True, mode = 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding = "utf-8") as handle:
        handle.write(json.dumps(state, indent = 2) + "\n")


def read_state(path: Path) -> Optional[dict]:
    """The recorded switch, or None when there is nothing to switch back."""
    try:
        state = json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("files"), dict):
        return None
    return state if state["files"] else None


def owner_record(kind: str) -> dict:
    """Who holds the switch; ``started`` tells a reused PID apart from the real owner."""
    try:
        import psutil
        started = psutil.Process().create_time()
    except Exception:
        started = None
    return {"pid": os.getpid(), "started": started, "kind": kind}


def owner_alive(owner: object) -> bool:
    try:
        pid = int(owner["pid"])
        started = owner.get("started")
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if started is None:
                return process.is_running()
            return abs(process.create_time() - float(started)) < 1.0
        except (psutil.Error, TypeError, ValueError):
            return False
    if os.name != "posix":
        # No way to check without psutil; assume it lives rather than undo someone's switch.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def begin(path: Path, targets: Targets, extra: dict, owner: dict) -> dict:
    """Snapshot the profile files, back them up and record the switch before anything is written.

    Raises ValueError (nothing changed) when a profile file can't be parsed."""
    files = snapshot(targets)
    for name, record in files.items():
        target = Path(name)
        record["mode"] = file_mode(target)
        if record["original"] is not None:
            write_file(target.with_name(target.name + BACKUP_SUFFIX), record["original"])
    state = {**extra, "owner": owner, "files": files}
    write_state(path, state)
    return state


def switch(path: Path, state: dict, targets: Targets, token: str) -> None:
    """Point the profile at state["gateway"]. Every write is recorded first, so a crash part way
    through is still switched back. Raises ValueError for a symlinked profile file."""

    def write(target: Path, text: str) -> None:
        write_state(path, state)
        write_file(target, text)

    apply_profile(targets, state["files"], state["gateway"], token, write)
    write_state(path, state)


def abort(path: Path, state: Optional[dict]) -> None:
    """Undo a switch that failed before Claude used it."""
    if state is not None:
        with contextlib.suppress(Exception):
            restore_profile(state["files"], write_file)
        for name in state.get("files") or {}:
            target = Path(name)
            target.with_name(target.name + BACKUP_SUFFIX).unlink(missing_ok = True)
    Path(path).unlink(missing_ok = True)


def switch_back(state: dict, quit_timeout: float = 30.0) -> list:
    """Quit Claude Desktop if it is open, then put its profile files back. Returns notes for the
    user; the caller revokes state["key_id"] and removes the state file."""
    notes = []
    # Claude writes its settings while it shuts down, so restore only after it has quit.
    if is_running():
        quit_app(timeout = quit_timeout)
        if not wait_for_exit(timeout = quit_timeout):
            notes.append(
                "Claude Desktop did not quit; quit it yourself so it keeps the switch back."
            )
        else:
            notes.append(
                "Quit Claude Desktop to switch it back; open it again to use your usual Claude."
            )
    files = state.get("files") or {}
    notes += restore_profile(files, write_file)
    for name, record in files.items():
        target = Path(name)
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        original = record.get("original")
        if original is not None and read_text(target) == original:
            if record.get("mode") is not None:
                with contextlib.suppress(OSError):
                    os.chmod(target, record["mode"])
            backup.unlink(missing_ok = True)
        elif original is not None and backup.exists():
            notes.append(f"Kept your original {target.name} at {backup}.")
    return notes
