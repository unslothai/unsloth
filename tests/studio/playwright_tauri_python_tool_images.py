# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser regression for the desktop Python tool image boundary.

Runs as a standalone script. It serves a page with the exact Tauri CSP, then
checks that trusted code can fetch an authenticated sandbox image into a blob
URL while an allowed HTTPS image redirect cannot reach the HTTP Studio backend.
The same policy intentionally continues to allow ordinary HTTPS images,
including HTTPS loopback, which is outside this PR's HTTP-backend boundary.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
TAURI_CONFIG = REPO / "studio/src-tauri/tauri.conf.json"
REDIRECT_URL = "https://redirect.invalid/attacker.png"
HTTPS_LOOPBACK_URL = "https://127.0.0.1:9443/sensitive/direct.png"
SANDBOX_PATH = "/api/inference/sandbox/session%20id/loss%20curve.png"
SENSITIVE_PATH = "/sensitive/redirect.png"
AUTHORIZATION = "Bearer browser-test-token"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class ProbeState:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.authorization: str | None = None


class ProbeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, handler: type[BaseHTTPRequestHandler], state: ProbeState):
        super().__init__(("127.0.0.1", 0), handler)
        self.state = state


class TargetHandler(BaseHTTPRequestHandler):
    server: ProbeServer

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        self.server.state.paths.append(self.path)
        if self.path == SANDBOX_PATH:
            self.server.state.authorization = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG)))
            self._cors()
            self.end_headers()
            self.wfile.write(PNG)
            return

        self.send_response(204)
        self.end_headers()


def page_handler(csp: str, target_origin: str) -> type[BaseHTTPRequestHandler]:
    class PageHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_GET(self) -> None:
            if self.path == "/app.js":
                script = f"""
const image = document.querySelector("#sandbox");
window.sandboxResult = (async () => {{
  const response = await fetch("{target_origin}{SANDBOX_PATH}", {{
    headers: {{ Authorization: "{AUTHORIZATION}" }},
  }});
  if (!response.ok) throw new Error(`sandbox fetch failed: ${{response.status}}`);
  const blob = await response.blob();
  window.sandboxObjectUrl = URL.createObjectURL(blob);
  await new Promise((resolve, reject) => {{
    image.addEventListener("load", resolve, {{ once: true }});
    image.addEventListener("error", reject, {{ once: true }});
    image.src = window.sandboxObjectUrl;
  }});
  return {{ width: image.naturalWidth, height: image.naturalHeight }};
}})();
window.revokeSandbox = () => URL.revokeObjectURL(window.sandboxObjectUrl);
"""
                body = script.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
            else:
                body = f"""<!doctype html>
<html>
  <body>
    <img id="redirect" src="{REDIRECT_URL}" alt="remote">
    <img id="https-loopback" src="{HTTPS_LOOPBACK_URL}" alt="https loopback">
    <img id="sandbox" alt="loss curve.png">
    <script src="/app.js"></script>
  </body>
</html>
""".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Security-Policy", csp)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PageHandler


@contextmanager
def running_server(server: ThreadingHTTPServer) -> Iterator[ThreadingHTTPServer]:
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout = 5)


def main() -> None:
    config = json.loads(TAURI_CONFIG.read_text(encoding = "utf-8"))
    csp = config["app"]["security"]["csp"]
    state = ProbeState()
    target = ProbeServer(TargetHandler, state)
    target_origin = f"http://127.0.0.1:{target.server_port}"
    page_server = ProbeServer(page_handler(csp, target_origin), ProbeState())
    page_origin = f"http://127.0.0.1:{page_server.server_port}"

    with running_server(target), running_server(page_server), sync_playwright() as p:
        launch_options: dict[str, object] = {
            "headless": True,
            "args": chromium_launch_args(),
        }
        channel = os.environ.get("STUDIO_PLAYWRIGHT_CHANNEL")
        if channel:
            launch_options["channel"] = channel
        browser = p.chromium.launch(**launch_options)
        try:
            context = browser.new_context()
            redirect_requests = 0
            https_loopback_requests = 0

            def redirect(route) -> None:
                nonlocal redirect_requests
                redirect_requests += 1
                route.fulfill(
                    status = 302,
                    headers = {"Location": f"{target_origin}{SENSITIVE_PATH}"},
                )

            def https_loopback(route) -> None:
                nonlocal https_loopback_requests
                https_loopback_requests += 1
                route.fulfill(status = 200, content_type = "image/png", body = PNG)

            context.route(REDIRECT_URL, redirect)
            context.route(HTTPS_LOOPBACK_URL, https_loopback)
            page = context.new_page()
            page.goto(page_origin, wait_until = "domcontentloaded")
            dimensions = page.evaluate("window.sandboxResult")
            page.wait_for_timeout(500)

            assert dimensions == {"width": 1, "height": 1}
            assert state.authorization == AUTHORIZATION
            assert SANDBOX_PATH in state.paths
            assert redirect_requests == 1
            assert SENSITIVE_PATH not in state.paths
            assert https_loopback_requests == 1
            assert page.locator("#https-loopback").evaluate(
                "image => image.complete && image.naturalWidth === 1"
            )
            assert page.locator("#sandbox").get_attribute("src").startswith("blob:")

            object_url = page.evaluate("window.sandboxObjectUrl")
            image_loads = """url => new Promise((resolve) => {
              const image = new Image();
              image.addEventListener("load", () => resolve(true), { once: true });
              image.addEventListener("error", () => resolve(false), { once: true });
              image.src = url;
            })"""
            assert page.evaluate(image_loads, object_url)

            page.evaluate("window.revokeSandbox()")
            assert not page.evaluate(image_loads, object_url)
        finally:
            browser.close()

    print("desktop Python tool image browser regression: PASS")


if __name__ == "__main__":
    main()
