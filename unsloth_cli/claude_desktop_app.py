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
import secrets
import subprocess
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
UNSUPPORTED_IMAGE_NOTICE = (
    "[Image omitted by Unsloth because the selected model does not support image recognition.]"
)
MAX_REQUEST_BODY_BYTES = 64 << 20
MAX_ERROR_BODY_BYTES = 64 << 10
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


def local_app_data(home: Optional[Path] = None) -> Path:
    if os.environ.get("LOCALAPPDATA", "").strip():
        return Path(os.environ["LOCALAPPDATA"].strip())
    if os.environ.get("USERPROFILE", "").strip():
        return Path(os.environ["USERPROFILE"].strip()) / "AppData" / "Local"
    return (home or Path.home()) / "AppData" / "Local"


def profile_roots(platform: Optional[str] = None, home: Optional[Path] = None) -> tuple:
    """(normal roots, third-party roots). The 1p profile picks the mode; the 3p one holds ours."""
    platform = platform or sys.platform
    home = home or Path.home()
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
    home = home or Path.home()
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


def quit_app() -> None:
    subprocess.run(
        ["osascript", "-e", 'tell application "Claude" to quit'], capture_output = True, timeout = 30
    )


def open_app(path: Path) -> None:
    subprocess.run(["/usr/bin/open", str(path)], capture_output = True, timeout = 30)


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


class Gateway(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        upstream: str,
        upstream_key: str,
        model: GatewayModel,
        token: str,
        listen: tuple = ("127.0.0.1", 0),
    ):
        parsed = urlparse(upstream)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"invalid Unsloth URL {upstream!r}")
        self.upstream = parsed
        self.upstream_key = upstream_key
        self.model = model
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

    def connection(self) -> http.client.HTTPConnection:
        cls = (
            http.client.HTTPSConnection
            if self.upstream.scheme == "https"
            else http.client.HTTPConnection
        )
        return cls(self.upstream.hostname, self.upstream.port, timeout = 3600)


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
    protocol_version = "HTTP/1.1"
    server: Gateway

    def log_message(self, format, *args):  # noqa: A002 - quiet: the CLI owns the terminal
        pass

    # -- plumbing ---------------------------------------------------------------------------

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
        entry = self.server.model.catalog_entry()
        catalog = {
            "data": [entry],
            "first_id": entry["id"],
            "last_id": entry["id"],
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
        model = self.server.model
        if requested not in (model.route, model.model_id):
            self._error(400, f'unknown Claude model "{requested}"')
            return
        payload["model"] = model.model_id
        self.server.routed += 1
        self._forward(path, payload)

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

    def _forward(self, path: str, payload: dict) -> None:
        original = copy.deepcopy(payload)
        try:
            conn, response = self._round_trip(path, payload)
        except (OSError, http.client.HTTPException):
            self._plain(502, "Unsloth gateway unavailable")
            return
        prefix = b""
        try:
            if (
                path == "/v1/messages"
                and response.status in (400, 500)
                and not self.server.model.supports_vision
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
            self._relay(response, prefix)
        finally:
            conn.close()

    def _relay(self, response: http.client.HTTPResponse, prefix: bytes) -> None:
        self.send_response(response.status, response.reason)
        length = response.getheader("Content-Length")
        for name, value in response.getheaders():
            if name.lower() not in _HOP_BY_HOP:
                self.send_header(name, value)
        chunked = length is None
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
            write(prefix)
            while True:
                # read1 returns what has arrived, so server-sent events stream through.
                data = response.read1(65536)
                if not data:
                    break
                write(data)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def new_token() -> str:
    return "unsloth-claude-" + secrets.token_urlsafe(24)
