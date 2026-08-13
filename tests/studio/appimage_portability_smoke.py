# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launch a shipped AppImage against a deterministic desktop backend fixture.

This deliberately does not install host GTK, WebKitGTK, JavaScriptCore, or an
AppIndicator implementation. Reaching desktop auth proves that the packaged
webview rendered and ran its startup JavaScript, not merely that AppRun stayed
alive.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path


ROOT_ID = "a" * 64
OWNER_SECRET = "appimage-portability-owner"
DESKTOP_SECRET = "appimage-portability-secret"


def _minimum_backend_version(repo_root: Path) -> str:
    source = (repo_root / "studio/src-tauri/src/preflight/version.rs").read_text(
        encoding = "utf-8"
    )
    marker = 'MIN_DESKTOP_BACKEND_VERSION: &str = "'
    start = source.find(marker)
    if start < 0:
        raise RuntimeError("Could not read the minimum desktop backend version")
    start += len(marker)
    return source[start : source.index('"', start)]


def _write_fixture(art_dir: Path, home: Path, version: str) -> Path:
    request_log = art_dir / "backend-requests.jsonl"
    backend = art_dir / "backend.py"
    backend.write_text(
        textwrap.dedent(
            f"""\
            import hashlib
            import json
            import os
            import sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            LOG = {str(request_log)!r}
            ROOT_ID = {ROOT_ID!r}
            VERSION = {version!r}
            token = os.environ.get("UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN", "")
            owner = {{
                "kind": "tauri",
                "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            }}

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    return

                def record(self, method):
                    length = int(self.headers.get("content-length", "0"))
                    body = self.rfile.read(length).decode(errors="replace") if length else ""
                    with open(LOG, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps({{"method": method, "path": self.path, "body": body}}) + "\\n")

                def send_json(self, payload, status=200):
                    raw = json.dumps(payload).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Access-Control-Allow-Origin", "tauri://localhost")
                    self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                    self.end_headers()
                    self.wfile.write(raw)

                def do_OPTIONS(self):
                    self.send_json({{}}, 204)

                def do_GET(self):
                    self.record("GET")
                    if self.path.startswith(("/api/liveness", "/api/health")):
                        return self.send_json({{
                            "status": "alive",
                            "service": "Unsloth UI Backend",
                            "version": VERSION,
                            "desktop_protocol_version": 1,
                            "desktop_manageability_version": 2,
                            "supports_desktop_auth": True,
                            "supports_desktop_backend_ownership": True,
                            "studio_root_id": ROOT_ID,
                            "desktop_owner": owner,
                            "chat_only": False,
                            "hardware_detecting": False,
                        }})
                    if self.path.startswith("/api/system"):
                        return self.send_json({{"device_type": "cpu", "chat_only": False}})
                    return self.send_json({{}})

                def do_POST(self):
                    self.record("POST")
                    if self.path.startswith("/api/auth/desktop-login"):
                        return self.send_json({{
                            "access_token": "portability-access",
                            "refresh_token": "portability-refresh",
                            "token_type": "bearer",
                            "must_change_password": False,
                        }})
                    return self.send_json({{}})

            port = int(sys.argv[1])
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
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
                  "version": version,
              }, separators=(",", ":"))}'
              exit 0
            fi
            if [[ "$*" == *"provision-desktop-auth"* ]]; then
              mkdir -p "$HOME/.unsloth/studio/auth"
              printf '%s' {DESKTOP_SECRET!r} > "$HOME/.unsloth/studio/auth/.desktop_secret"
              chmod 600 "$HOME/.unsloth/studio/auth/.desktop_secret"
              exit 0
            fi
            if [[ "$*" == *"studio"*"--api-only"* ]]; then
              port=8888
              while [[ $# -gt 0 ]]; do
                if [[ "$1" == "-p" ]]; then port="$2"; break; fi
                shift
              done
              exec /usr/bin/python3 {str(backend)!r} "$port"
            fi
            exit 1
            """
        ),
        encoding = "utf-8",
    )
    managed_bin.chmod(0o755)
    return request_log


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    appimage_value = os.environ.get("APPIMAGE_PATH", "")
    if not appimage_value:
        raise SystemExit("APPIMAGE_PATH must name the AppImage under test")
    appimage = Path(appimage_value).resolve()
    if not appimage.is_file():
        raise SystemExit(f"AppImage does not exist: {appimage}")
    if not shutil.which("xvfb-run"):
        raise SystemExit("xvfb-run is required")

    art_dir = Path(
        os.environ.get("APPIMAGE_SMOKE_ART_DIR", "logs/appimage-portability")
    ).resolve()
    if art_dir.exists():
        shutil.rmtree(art_dir)
    art_dir.mkdir(parents = True)
    home = art_dir / "home"
    runtime = art_dir / "runtime"
    config = art_dir / "config"
    data = art_dir / "data"
    cache = art_dir / "cache"
    state = art_dir / "state"
    for directory in (runtime, config, data, cache, state):
        directory.mkdir(parents = True)
    runtime.chmod(0o700)
    install_id = home / ".unsloth/studio/share/studio_install_id"
    install_id.parent.mkdir(parents = True)
    install_id.write_text(ROOT_ID, encoding = "utf-8")
    request_log = _write_fixture(
        art_dir, home, _minimum_backend_version(repo_root)
    )

    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "XDG_CACHE_HOME": str(cache),
        "XDG_STATE_HOME": str(state),
        "XDG_RUNTIME_DIR": str(runtime),
        "APPIMAGE_EXTRACT_AND_RUN": "1",
        "NO_AT_BRIDGE": "1",
        "WEBKIT_DISABLE_DMABUF_RENDERER": "1",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": "llvmpipe",
        "G_MESSAGES_DEBUG": "all",
    }
    stdout = (art_dir / "app-stdout.log").open("wb")
    process = subprocess.Popen(
        [
            "xvfb-run",
            "-a",
            "--server-args=-screen 0 1440x900x24",
            str(appimage),
        ],
        stdout = stdout,
        stderr = subprocess.STDOUT,
        env = env,
        start_new_session = True,
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"AppImage exited early with {process.returncode}")
            if request_log.is_file():
                requests = request_log.read_text(encoding = "utf-8")
                if '"path": "/api/auth/desktop-login"' in requests:
                    print("PASS complete AppImage rendered startup and completed desktop auth")
                    return
            time.sleep(0.25)
        raise RuntimeError("Packaged webview never completed desktop authentication")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout = 10)
        stdout.close()
        tauri_log = home / ".unsloth/studio/tauri.log"
        if tauri_log.is_file():
            shutil.copy2(tauri_log, art_dir / "tauri.log")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            f"AppImage portability evidence: {os.environ.get('APPIMAGE_SMOKE_ART_DIR', 'logs/appimage-portability')}",
            file = sys.stderr,
        )
        raise
