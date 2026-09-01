# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise model download and cancellation through the packaged WebKitGTK view."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


from appimage_test_support import (
    FIXTURE_BACKEND_VERSION,
    assert_fixture_version_clears_floor,
    assert_no_loader_errors,
)


REPO_ID = "unsloth/FLUX.2-klein-4B-GGUF"
FILENAME = "FLUX.2-klein-4B-Q4_K_M.gguf"
MODEL_BYTES = 1_048_576
REPO_ROOT = Path(__file__).resolve().parents[2]
ART_DIR = Path(os.environ.get("APPIMAGE_E2E_ART_DIR", "logs/appimage-model-download"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    base: str,
    method: str,
    path: str,
    payload: object | None = None,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base}{path}",
        data = body,
        method = method,
        headers = {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout = timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors = "replace")
        raise RuntimeError(f"WebDriver {method} {path} returned {error.code}: {detail}") from error


def _execute(base: str, session_id: str, script: str) -> Any:
    response = _request(
        base,
        "POST",
        f"/session/{session_id}/execute/sync",
        {"script": script, "args": []},
    )
    return response.get("value")


def _wait_for(
    base: str,
    session_id: str,
    script: str,
    description: str,
    *,
    timeout: float = 90,
) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = _execute(base, session_id, script)
            if last:
                return last
        except Exception as error:  # The webview can reload while startup hands off.
            last = str(error)
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {description}; last result: {last!r}")


# The quant list is fetched once per expanded row, by an effect keyed on [repoId, localSource, refreshKey, hfToken].
# please relaunch it." rendered inside the FLUX.2-klein-4B row while /api/health kept answering for another 49 seconds,
# on a job that fails on roughly three runs in four across unrelated branches.
_VARIANT_RETRY_ATTEMPTS = 3
_VARIANT_ERROR_TEXT = (
    "const box=[...document.querySelectorAll('div')].find("
    "(d)=>d.className.includes('text-destructive') && "
    "[...d.querySelectorAll('button')].some((b)=>(b.innerText||'').trim()==='Retry'));"
    "return box ? (box.innerText||'').trim() : '';"
)
_VARIANT_RETRY_CLICK = (
    "const box=[...document.querySelectorAll('div')].find("
    "(d)=>d.className.includes('text-destructive') && "
    "[...d.querySelectorAll('button')].some((b)=>(b.innerText||'').trim()==='Retry'));"
    "const b=box && [...box.querySelectorAll('button')].find("
    "(e)=>(e.innerText||'').trim()==='Retry'); if(b)b.click(); return !!b;"
)


def _wait_for_quantization(
    base: str,
    session_id: str,
    script: str,
    description: str,
    *,
    timeout: float = 15,
) -> Any:
    """_wait_for, plus the row's own Retry button when the listing failed outright."""
    last_error = ""
    for attempt in range(1, _VARIANT_RETRY_ATTEMPTS + 1):
        try:
            return _wait_for(base, session_id, script, description, timeout = timeout)
        except AssertionError:
            error_text = _execute(base, session_id, _VARIANT_ERROR_TEXT) or ""
            if not error_text:
                raise  # No listing error on screen, so retrying would prove nothing.
            last_error = error_text
            print(
                f"[appimage-e2e] quant listing failed (attempt {attempt}/"
                f"{_VARIANT_RETRY_ATTEMPTS}): {error_text!r}",
                flush = True,
            )
            if attempt == _VARIANT_RETRY_ATTEMPTS:
                break
            if not _execute(base, session_id, _VARIANT_RETRY_CLICK):
                break
            time.sleep(1.0)
    raise AssertionError(
        f"Timed out waiting for {description} after {_VARIANT_RETRY_ATTEMPTS} listing "
        f"attempts. The row reported: {last_error!r}"
    )


def _write_backend_fixture(home: Path, request_log: Path) -> None:
    fixture_dir = ART_DIR / "fixture"
    fixture_dir.mkdir(parents = True, exist_ok = True)
    server = fixture_dir / "backend.py"
    server.write_text(
        textwrap.dedent(
            f"""\
            import ctypes
            import hashlib
            import json
            import os
            import signal
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from urllib.parse import parse_qs, urlparse

            # Do not leave the fixture bound after the app exits.
            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
            if os.getppid() == 1:
                raise SystemExit(0)

            REPO_ID = {REPO_ID!r}
            FILENAME = {FILENAME!r}
            MODEL_BYTES = {MODEL_BYTES}
            REQUEST_LOG = {str(request_log.resolve())!r}
            token = os.environ.get("UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN", "")
            root_path = os.path.expanduser("~/.unsloth/studio/share/studio_install_id")
            with open(root_path, encoding="utf-8") as handle:
                studio_root_id = handle.read().strip()
            owner = {{
                "kind": "tauri",
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }}
            state = {{"started": False, "cancelled": False}}

            def record(method, path, payload=None):
                with open(REQUEST_LOG, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({{"method": method, "path": path, "body": payload}}) + "\\n")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def read_json(self):
                    length = int(self.headers.get("content-length", "0"))
                    raw = self.rfile.read(length) if length else b"{{}}"
                    return json.loads(raw or b"{{}}")

                def send_json(self, payload, status=200):
                    raw = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Access-Control-Allow-Origin", "tauri://localhost")
                    self.send_header("Access-Control-Allow-Headers", self.allowed_headers())
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.end_headers()
                    self.wfile.write(raw)

                def allowed_headers(self):
                    # Echo requested headers so this fixture follows frontend changes.
                    asked = self.headers.get("Access-Control-Request-Headers")
                    return asked or "Authorization, Content-Type, X-HF-Token"

                def do_OPTIONS(self):
                    # Recorded like GET and POST. A preflight the browser rejects means the
                    # real request is never sent, so without this the request log shows
                    # nothing and the failure looks like the frontend never tried.
                    record("OPTIONS", urlparse(self.path).path)
                    self.send_response(204)
                    self.send_header("Access-Control-Allow-Origin", "tauri://localhost")
                    # Echo what was asked for. studio/backend/main.py runs CORSMiddleware
                    # with allow_headers = ["*"], so a fixed list here is not the product's
                    # behaviour but a second copy of it, and it drifted: #8879 began sending
                    # two X-Unsloth timezone headers on every authFetch, this list still
                    # named three headers, and every authed request in this test has failed
                    # its preflight since. Echoing cannot drift again.
                    requested = self.headers.get("Access-Control-Request-Headers")
                    self.send_header("Access-Control-Allow-Headers", requested or "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.end_headers()

                def do_GET(self):
                    parsed = urlparse(self.path)
                    path = parsed.path
                    query = parse_qs(parsed.query)
                    record("GET", path)
                    if path in ("/api/liveness", "/api/health"):
                        return self.send_json({{
                            "status": "alive",
                            "service": "Unsloth UI Backend",
                            "version": {FIXTURE_BACKEND_VERSION!r},
                            "desktop_protocol_version": 1,
                            "desktop_manageability_version": 2,
                            "supports_desktop_auth": True,
                            "supports_desktop_backend_ownership": True,
                            "studio_root_id": studio_root_id,
                            "desktop_owner": owner,
                            "device_type": "cuda",
                            "chat_only": False,
                            "hardware_detecting": False,
                        }})
                    if path == "/api/system":
                        device = {{
                            "index": 0, "index_kind": "physical", "name": "AppImage E2E GPU",
                            "memory_total_gb": 24, "vram_free_gb": 22,
                        }}
                        return self.send_json({{
                            "platform": "Linux", "python_version": "3.11", "device_backend": "cuda",
                            "uptime_seconds": 1,
                            "cpu": {{"logical_count": 8, "physical_count": 4, "usage_percent": 1, "frequency_mhz": 3000}},
                            "memory": {{"total_gb": 32, "available_gb": 24, "percent_used": 25, "process_used_mb": 128}},
                            "disk": {{"total_gb": 100, "free_gb": 80, "percent_used": 20}},
                            "gpu": {{"available": True, "backend": "cuda", "devices": [device]}},
                            "inference_gpu": {{"available": True, "backend": "cuda", "devices": [device]}},
                            "ml_packages": {{}},
                        }})
                    if path == "/api/inference/images/status":
                        return self.send_json({{
                            "loaded": False, "repo_id": None, "family": None, "base_repo": None,
                            "device": None, "dtype": None, "model_kind": None, "workflows": [],
                        }})
                    if path == "/api/inference/images/load-progress":
                        return self.send_json({{"phase": None, "bytes_downloaded": 0, "bytes_total": 0, "error": None}})
                    if path == "/api/inference/images/generate-progress":
                        return self.send_json({{"active": False, "step": 0, "total_steps": 0, "eta_seconds": None}})
                    if path == "/api/inference/images/info":
                        return self.send_json({{"families": []}})
                    if path == "/api/inference/images/gallery":
                        return self.send_json({{"images": [], "has_more": False}})
                    if path == "/api/inference/monitor":
                        return self.send_json({{"status": "idle", "active_model": None, "active_requests": 0, "entries": []}})
                    if path == "/api/models/diffusion-loras":
                        return self.send_json({{"loras": []}})
                    if path == "/api/models/diffusion-controlnets":
                        return self.send_json({{"controlnets": []}})
                    if path == "/api/models/list":
                        return self.send_json({{"models": [], "default_models": []}})
                    if path == "/api/models/loras":
                        return self.send_json({{"loras": [], "outputs_dir": "/tmp/outputs"}})
                    if path == "/api/hub/local":
                        return self.send_json({{"models_dir": "/tmp/models", "lmstudio_dirs": [], "models": []}})
                    if path in ("/api/hub/cached-gguf", "/api/hub/cached-models"):
                        return self.send_json({{"cached": []}})
                    if path == "/api/hub/hidden-models":
                        return self.send_json({{"patterns": []}})
                    if path in ("/api/hub/active-downloads", "/api/hub/datasets/active-downloads"):
                        return self.send_json({{"downloads": []}})
                    if path in ("/api/hub/gguf-variants", "/api/models/gguf-variants"):
                        return self.send_json({{
                            "repo_id": REPO_ID,
                            "variants": [{{
                                "filename": FILENAME, "quant": "Q4_K_M",
                                "size_bytes": MODEL_BYTES, "download_size_bytes": MODEL_BYTES,
                                "downloaded": False,
                            }}],
                            "has_vision": False, "default_variant": "Q4_K_M", "context_length": None,
                        }})
                    if path == "/api/hub/download-status":
                        job_state = (
                            "cancelled" if state["cancelled"]
                            else "running" if state["started"]
                            else "idle"
                        )
                        return self.send_json({{
                            "state": job_state, "error": None,
                            "generation": 1 if state["started"] else None,
                        }})

                    if path == "/api/studio/download-transport-capabilities":
                        return self.send_json({{
                            "http": {{"available": True, "reason": None}},
                            "xet": {{"available": False, "reason": "AppImage E2E uses HTTP"}},
                            "auto_resolves_to": "http", "auto_reason": "deterministic E2E",
                        }})
                    if path in ("/api/hub/gguf-download-progress", "/api/hub/download-progress"):
                        downloaded = 262144
                        return self.send_json({{
                            "downloaded_bytes": downloaded, "completed_bytes": downloaded,
                            "complete_on_disk": False, "expected_bytes": MODEL_BYTES,
                            "progress": downloaded / MODEL_BYTES, "cache_path": "/tmp/model-cache",
                        }})
                    if path == "/api/hub/transport-status":
                        return self.send_json({{"has_partial": False, "last_transport": None, "resumable": False}})
                    if path == "/api/settings/personalization":
                        return self.send_json({{
                            "version": 1,
                            "profile": {{"displayName": "", "nickname": "", "avatarDataUrl": None, "avatarShape": "circle", "showGreetingSloth": True}},
                            "appearance": {{"theme": "dark", "palette": "standard", "language": None, "customization": {{}}}},
                            "saved": False, "customizationSaved": False, "paletteSaved": False, "greetingSlothSaved": False,
                        }})
                    if path == "/api/export/status":
                        return self.send_json({{"current_checkpoint": None, "is_vision": False, "is_peft": False, "is_export_active": False}})
                    if path in ("/api/chat/threads", "/api/chat/projects"):
                        return self.send_json({{"threads": []}} if path.endswith("threads") else {{"projects": []}})
                    return self.send_json({{}})

                def do_POST(self):
                    path = urlparse(self.path).path
                    payload = self.read_json()
                    record("POST", path, payload)
                    if path == "/api/auth/desktop-login":
                        if payload.get("secret") == "desktop-owner-adoption-invalid-secret":
                            return self.send_json({{"detail": "invalid desktop secret"}}, 401)
                        return self.send_json({{
                            "access_token": "appimage-e2e-access", "refresh_token": "appimage-e2e-refresh",
                            "token_type": "bearer", "must_change_password": False,
                        }})
                    if path == "/api/inference/images/download-plan":
                        return self.send_json({{
                            "entries": [{{"repo_id": REPO_ID, "files": [FILENAME], "bytes": MODEL_BYTES, "gguf_filename": FILENAME}}],
                            "total_bytes": MODEL_BYTES, "required_bytes": MODEL_BYTES, "checkpoint_bytes": MODEL_BYTES,
                        }})
                    if path == "/api/hub/download":
                        state["started"] = True
                        state["cancelled"] = False
                        return self.send_json({{
                            "job_key": f"model:{{REPO_ID}}:Q4_K_M", "accepted": True,
                            "state": "running", "generation": 1, "transport": "http", "cancel_transport": None,
                        }}, 202)
                    if path == "/api/hub/download/cancel":
                        state["cancelled"] = True
                        return self.send_json({{"job_key": f"model:{{REPO_ID}}:Q4_K_M", "state": "cancelled"}}, 202)
                    return self.send_json({{}})

            requested_port = int(sys.argv[1])
            server = ThreadingHTTPServer(("127.0.0.1", requested_port), Handler)
            print(f"TAURI_PORT={{server.server_port}}", flush=True)
            server.serve_forever()
            """
        ),
        encoding = "utf-8",
    )

    managed_bin = home / ".unsloth/studio/unsloth_studio/bin/unsloth"
    managed_bin.parent.mkdir(parents = True, exist_ok = True)
    managed_bin.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${{1:-}}" == "-h" ]]; then exit 0; fi
            if [[ "$*" == *"desktop-capabilities"* ]]; then
              printf '%s\\n' '{json.dumps({
                  "desktop_protocol_version": 1,
                  "desktop_manageability_version": 2,
                  "supports_api_only": True,
                  "supports_provision_desktop_auth": True,
                  "supports_desktop_backend_ownership": True,
                  "studio_install_ok": True,
                  "version": FIXTURE_BACKEND_VERSION,
              }, separators = (",", ":"))}'
              exit 0
            fi
            if [[ "$*" == *"provision-desktop-auth"* ]]; then
              mkdir -p "$HOME/.unsloth/studio/auth"
              printf 'appimage-e2e-secret' > "$HOME/.unsloth/studio/auth/.desktop_secret"
              chmod 600 "$HOME/.unsloth/studio/auth/.desktop_secret"
              exit 0
            fi
            if [[ "$*" == *"studio"*"--api-only"* ]]; then
              port=8888
              while [[ $# -gt 0 ]]; do
                if [[ "$1" == "-p" ]]; then port="$2"; break; fi
                shift
              done
              exec /usr/bin/python3 {str(server.resolve())!r} "$port"
            fi
            exit 1
            """
        ),
        encoding = "utf-8",
    )
    managed_bin.chmod(0o755)


def _request_log_contains(request_log: Path, path: str) -> bool:
    if not request_log.is_file():
        return False
    return any(
        json.loads(line).get("method") == "POST" and json.loads(line).get("path") == path
        for line in request_log.read_text(encoding = "utf-8").splitlines()
        if line.strip()
    )


def _install_colrv1_probe_font(config_dir: Path, data_dir: Path) -> dict[str, str] | None:
    source_value = os.environ.get("APPIMAGE_COLRV1_FONT", "")
    if not source_value:
        return None

    source = Path(source_value).resolve()
    expected_sha = os.environ.get("APPIMAGE_COLRV1_FONT_SHA256", "").lower()
    if not source.is_file() or not expected_sha:
        raise RuntimeError("APPIMAGE_COLRV1_FONT and its SHA-256 are required together")
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"COLRv1 font digest mismatch: {actual_sha} != {expected_sha}")
    font_bytes = source.read_bytes()
    for table in (b"COLR", b"CPAL"):
        if table not in font_bytes:
            raise RuntimeError(f"COLRv1 regression font has no {table.decode()} table")

    font_dir = data_dir / "fonts"
    font_dir.mkdir(parents = True, exist_ok = True)
    installed = font_dir / "Noto-COLRv1.ttf"
    shutil.copy2(source, installed)
    fontconfig_dir = config_dir / "fontconfig/conf.d"
    fontconfig_dir.mkdir(parents = True, exist_ok = True)
    (fontconfig_dir / "10-unsloth-colrv1-regression.conf").write_text(
        """<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">
<fontconfig>
  <match target="scan">
    <test name="file" compare="contains"><string>Noto-COLRv1.ttf</string></test>
    <edit name="family" mode="assign"><string>Unsloth Test COLRv1</string></edit>
  </match>
  <match target="pattern">
    <edit name="family" mode="prepend" binding="strong">
      <string>Unsloth Test COLRv1</string>
    </edit>
  </match>
</fontconfig>
""",
        encoding = "utf-8",
    )
    font_env = {
        **os.environ,
        "HOME": str(data_dir.parents[1]),
        "XDG_CONFIG_HOME": str(config_dir),
        "XDG_DATA_HOME": str(data_dir),
    }
    subprocess.run(
        ["fc-cache", "-f", str(font_dir)],
        check = True,
        env = font_env,
        stdout = subprocess.DEVNULL,
    )
    selected_by_charset = {}
    for charset in ("1f680", "1faea"):
        selected = subprocess.check_output(
            ["fc-match", "-f", "%{file}\t%{family}\t%{color}\n", f"sans-serif:charset={charset}"],
            env = font_env,
            text = True,
        ).strip()
        if str(installed) not in selected:
            raise RuntimeError(
                f"Host Fontconfig did not select the COLRv1 fixture for U+{charset.upper()}: "
                f"{selected}"
            )
        selected_by_charset[charset] = selected
    result = {
        "path": str(installed),
        "sha256": actual_sha,
        "host_fc_match": selected_by_charset,
    }
    (ART_DIR / "colrv1-host-font.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    return result


def main() -> None:
    assert_fixture_version_clears_floor(REPO_ROOT)
    appimage_value = os.environ.get("APPIMAGE_PATH", "")
    if not appimage_value:
        raise SystemExit("APPIMAGE_PATH must name the AppImage under test")
    appimage = Path(appimage_value).resolve()
    if not appimage.is_file():
        raise SystemExit(f"AppImage does not exist: {appimage}")
    tauri_driver = shutil.which("tauri-driver")
    native_driver = shutil.which("WebKitWebDriver")
    if not tauri_driver or not native_driver:
        raise SystemExit("tauri-driver and WebKitWebDriver must both be on PATH")

    if ART_DIR.exists():
        shutil.rmtree(ART_DIR)
    ART_DIR.mkdir(parents = True)
    home = ART_DIR.resolve() / "home"
    runtime_dir = ART_DIR.resolve() / "runtime"
    config_dir = home / ".config"
    data_dir = home / ".local/share"
    cache_dir = home / ".cache"
    state_dir = home / ".local/state"
    runtime_dir.mkdir(parents = True)
    config_dir.mkdir(parents = True)
    data_dir.mkdir(parents = True)
    cache_dir.mkdir(parents = True)
    state_dir.mkdir(parents = True)

    colrv1_font = _install_colrv1_probe_font(config_dir, data_dir)
    request_log = ART_DIR.resolve() / "backend-requests.jsonl"
    _write_backend_fixture(home, request_log)

    driver_port = _free_port()
    native_port = _free_port()
    while native_port == driver_port:
        native_port = _free_port()
    base = f"http://127.0.0.1:{driver_port}"
    driver_log = (ART_DIR / "tauri-driver.log").open("wb")
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config_dir),
        "XDG_DATA_HOME": str(data_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "XDG_STATE_HOME": str(state_dir),
        "XDG_RUNTIME_DIR": str(runtime_dir),
        "APPIMAGE_EXTRACT_AND_RUN": "1",
        "WEBKIT_DISABLE_DMABUF_RENDERER": "1",
        "G_MESSAGES_DEBUG": "all",
    }
    process = subprocess.Popen(
        [
            tauri_driver,
            "--port",
            str(driver_port),
            "--native-port",
            str(native_port),
            "--native-driver",
            native_driver,
        ],
        stdout = driver_log,
        stderr = subprocess.STDOUT,
        env = env,
        start_new_session = True,
    )
    session_id: str | None = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"tauri-driver exited early with {process.returncode}")
            try:
                _request(base, "GET", "/status", timeout = 1)
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise RuntimeError("tauri-driver did not open its HTTP port")

        capabilities = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "wry",
                    "tauri:options": {"application": str(appimage)},
                }
            }
        }
        created = _request(base, "POST", "/session", capabilities, timeout = 60)
        session_id = str(created["value"]["sessionId"])

        _wait_for(
            base,
            session_id,
            "return document.body && document.body.innerText.includes('Starting Unsloth')",
            "the packaged frontend to render",
        )
        _wait_for(
            base,
            session_id,
            "return document.body && !document.body.innerText.includes('Starting Unsloth')",
            "the deterministic desktop backend handoff",
        )

        if colrv1_font is not None:
            clicked = _wait_for(
                base,
                session_id,
                "const b=document.querySelector('[data-testid=\"nav-row-hub\"]')||[...document.querySelectorAll('button')].find((e)=>(e.innerText||'').trim()==='Model hub');if(b){b.click();return true;}return false;",
                "the Model hub sidebar destination",
                timeout = 30,
            )
            assert clicked, "Could not click the Model hub sidebar destination"
            _wait_for(
                base,
                session_id,
                "return location.pathname==='/hub'&&document.body?.innerText.includes('Model hub')",
                "the Model hub route",
                timeout = 60,
            )
            inserted = _execute(
                base,
                session_id,
                "const d=document.createElement('div');d.id='appimage-colrv1-probe';d.textContent='🚀 🇬🇧 👨‍👩‍👧 ⭐ ✅ 🫪 🫯';d.style.cssText=\"position:fixed;z-index:2147483647;left:320px;top:90px;padding:16px;background:white;color:black;font-size:48px;font-family:'Unsloth Test COLRv1',sans-serif\";document.body.appendChild(d);return d.textContent;",
            )
            assert inserted, "Could not inject the COLRv1 renderer probe"
            evidence = {**colrv1_font, "route": "/hub", "survived_seconds": 0}
            for elapsed in range(1, 31):
                time.sleep(1)
                state = _execute(
                    base,
                    session_id,
                    "return {path:location.pathname,probe:document.getElementById('appimage-colrv1-probe')?.textContent||false,body:(document.body?.innerText||'').slice(0,12000)};",
                )
                assert state["path"] == "/hub", f"Left Model hub during COLRv1 probe: {state}"
                assert state["probe"], f"COLRv1 probe disappeared or renderer stopped: {state}"
                evidence["survived_seconds"] = elapsed
            evidence["result"] = "PASS"
            (ART_DIR / "colrv1-model-hub.json").write_text(
                json.dumps(evidence, indent = 2, ensure_ascii = False), encoding = "utf-8"
            )
            _execute(
                base,
                session_id,
                "document.getElementById('appimage-colrv1-probe')?.remove();return true;",
            )
        clicked = _wait_for(
            base,
            session_id,
            "const direct=document.querySelector('[data-testid=\"nav-row-images\"]'); if(direct){direct.click(); return 'direct';} const more=[...document.querySelectorAll('button')].find((e)=>(e.innerText||'').trim()==='More'); if(more){more.click(); return 'more';} return false;",
            "the rendered Images sidebar destination",
        )
        assert clicked, "Could not find the Images sidebar destination"
        if clicked == "more":
            clicked = _wait_for(
                base,
                session_id,
                "const e=[...document.querySelectorAll('[role=menuitem]')].find((x)=>(x.innerText||'').trim()==='Images'); if(e){e.click(); return true;} return false;",
                "the Images destination in the More menu",
            )
            assert clicked, "Could not click Images in the More menu"
        _wait_for(
            base,
            session_id,
            "return location.pathname === '/images'",
            "the Images route",
        )
        _wait_for(
            base,
            session_id,
            "return [...document.querySelectorAll('button')].some((b) => (b.innerText || b.getAttribute('aria-label') || '').includes('Select image model'))",
            "the Images model picker",
        )
        clicked = _execute(
            base,
            session_id,
            "const b=[...document.querySelectorAll('button')].find((e)=>(e.innerText||e.getAttribute('aria-label')||'').includes('Select image model')); if(b)b.click(); return !!b;",
        )
        assert clicked, "Could not click Select image model"
        _wait_for(
            base,
            session_id,
            "return document.querySelector('.unsloth-model-selector-menu')?.innerText.includes('FLUX.2')",
            "the seeded FLUX model row",
        )
        clicked = _execute(
            base,
            session_id,
            "const root=document.querySelector('.unsloth-model-selector-menu'); const rows=[...root.querySelectorAll('button')].filter((b)=>/FLUX\\.2[\\s-]klein[\\s-]4B/i.test(b.innerText||'')).sort((a,b)=>(a.innerText||'').length-(b.innerText||'').length); const row=rows[0]; if(row)row.click(); return row?.innerText||false;",
        )
        assert clicked, "Could not click the seeded FLUX model row"

        # Some catalog states show a format level before the quantization level.
        time.sleep(0.25)
        _execute(
            base,
            session_id,
            "const b=[...document.querySelectorAll('button')].find((e)=>(e.innerText||'').trim()==='GGUF'); if(b)b.click(); return !!b;",
        )
        _wait_for_quantization(
            base,
            session_id,
            "return [...document.querySelectorAll('button')].some((b)=>(b.innerText||'').includes('Q4_K_M'))",
            "the Q4_K_M quantization",
            timeout = 15,
        )
        clicked = _execute(
            base,
            session_id,
            "const b=[...document.querySelectorAll('button')].find((e)=>(e.innerText||'').includes('Q4_K_M')); if(b)b.click(); return !!b;",
        )
        assert clicked, "Could not click Q4_K_M"

        _wait_for(
            base,
            session_id,
            f"return {str(request_log)!r} && true",
            "the model download request",
            timeout = 1,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _request_log_contains(
            request_log, "/api/hub/download"
        ):
            time.sleep(0.25)
        assert _request_log_contains(
            request_log, "/api/hub/download"
        ), "The packaged webview never sent POST /api/hub/download"

        _wait_for(
            base,
            session_id,
            "return [...document.querySelectorAll('button')].some((b)=>(b.innerText||b.getAttribute('aria-label')||'').includes('Cancel download'))",
            "visible model download progress and its cancel button",
            timeout = 30,
        )
        clicked = _execute(
            base,
            session_id,
            "const b=[...document.querySelectorAll('button')].find((e)=>(e.innerText||e.getAttribute('aria-label')||'').includes('Cancel download')); if(b)b.click(); return !!b;",
        )
        assert clicked, "Could not click Cancel download"
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not _request_log_contains(
            request_log, "/api/hub/download/cancel"
        ):
            time.sleep(0.25)
        assert _request_log_contains(
            request_log, "/api/hub/download/cancel"
        ), "The packaged webview never sent POST /api/hub/download/cancel"

        _wait_for(
            base,
            session_id,
            "return document.body?.innerText.includes('Cancelled. Partial files kept.')",
            "the packaged UI to confirm cancellation",
            timeout = 15,
        )

        # Verify the packaged WebKit view exposes the required media formats.
        media = json.loads(
            _execute(
                base,
                session_id,
                "const v=document.createElement('video'); const a=document.createElement('audio');"
                "return JSON.stringify({"
                "mp4:v.canPlayType('video/mp4'), webm:v.canPlayType('video/webm'),"
                "wav:a.canPlayType('audio/wav'),"
                "capture:!!(navigator.mediaDevices&&navigator.mediaDevices.getUserMedia)});",
            )
        )
        (ART_DIR / "media-support.json").write_text(json.dumps(media, indent = 2), encoding = "utf-8")
        for media_type in ("mp4", "webm", "wav"):
            assert media[media_type], f"The packaged webview cannot play {media_type}: {media}"
        assert media["capture"], f"The packaged webview exposes no capture device API: {media}"

        screenshot = _request(base, "GET", f"/session/{session_id}/screenshot")
        (ART_DIR / "model-download-cancelled.png").write_bytes(
            base64.b64decode(screenshot["value"])
        )
        body = _execute(base, session_id, "return document.body.innerText")
        (ART_DIR / "webview-body.txt").write_text(str(body), encoding = "utf-8")
        assert_no_loader_errors(
            ART_DIR / "tauri-driver.log",
            home / ".unsloth/studio/tauri.log",
        )

        print(
            "PASS packaged AppImage survived the COLRv1 Model hub probe, "
            "then sent, rendered, and cancelled a model download"
        )
    except Exception:
        if session_id:
            try:
                screenshot = _request(base, "GET", f"/session/{session_id}/screenshot", timeout = 10)
                (ART_DIR / "failure.png").write_bytes(base64.b64decode(screenshot["value"]))
                snapshot = _execute(
                    base,
                    session_id,
                    "return {url:location.href,title:document.title,body:document.body?.innerText||'',html:document.body?.innerHTML||''};",
                )
                (ART_DIR / "failure-webview.json").write_text(
                    json.dumps(snapshot, indent = 2), encoding = "utf-8"
                )
            except Exception as evidence_error:
                (ART_DIR / "failure-evidence-error.txt").write_text(
                    str(evidence_error), encoding = "utf-8"
                )
        raise

    finally:
        if session_id:
            try:
                _request(base, "DELETE", f"/session/{session_id}", timeout = 10)
            except Exception:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout = 10)
        driver_log.close()
        tauri_log = home / ".unsloth/studio/tauri.log"
        if tauri_log.is_file():
            shutil.copy2(tauri_log, ART_DIR / "tauri.log")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Preserve the rendered text whenever a late assertion fails.
        print(f"AppImage E2E evidence: {ART_DIR.resolve()}", file = sys.stderr)
        raise
