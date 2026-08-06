# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Boot Studio backend + vite dev, render tool panes on /smoke-ansi.html, assert clean text."""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args, wait_for_health  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend"
BACKEND = ROOT / "studio" / "backend"
STUDIO_PY = Path.home() / ".unsloth/studio/unsloth_studio/bin/python"
BACKEND_PORT = int(os.environ.get("STUDIO_PORT", "8888"))
VITE_PORT = int(os.environ.get("VITE_PORT", "8000"))
BACKEND_BASE = f"http://127.0.0.1:{BACKEND_PORT}"
BASE = f"http://127.0.0.1:{VITE_PORT}"
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright-ansi-studio"))
ESC = "\u001b"


def info(msg: str) -> None:
    print(f"[ansi-studio] {msg}", flush = True)


def fail(msg: str) -> None:
    raise AssertionError(f"[ansi-studio] FAIL: {msg}")


def drain_process_output(proc: subprocess.Popen[str]) -> None:
    if proc.stdout is None:
        return
    for _ in proc.stdout:
        pass


def start_logged_process(
    args: list[str], *, cwd: Path, env: dict[str, str]
) -> subprocess.Popen[str]:
    proc = subprocess.Popen(
        args,
        cwd = cwd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        env = env,
    )
    threading.Thread(target = drain_process_output, args = (proc,), daemon = True).start()
    return proc


def backend_terminal_has_ansi() -> None:
    info("backend: execute_tool terminal with forced SGR output")
    code = f"""
import sys
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.tools import execute_tool
out = execute_tool(
    "terminal",
    {{"command": "bash -c 'printf \"\\\\033[32mfile.txt\\\\033[0m\\\\n\"'"}},
    disable_sandbox=True,
)
assert chr(27) in out, f"expected ANSI in raw terminal output, got {{out!r}}"
print("backend raw terminal output has ANSI escapes")
"""
    subprocess.run([str(STUDIO_PY), "-c", code], cwd = BACKEND, check = True)


def api_json(
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    token: str | None = None,
):
    import json

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BACKEND_BASE}{path}",
        data = data,
        method = method,
        headers = headers,
    )
    with urllib.request.urlopen(req, timeout = 60) as resp:
        raw = resp.read().decode()
        return resp.status, json.loads(raw)


def bootstrap_password(auth_dir: Path) -> tuple[str, str]:
    bootstrap_path = auth_dir / ".bootstrap_password"
    deadline = time.time() + 180
    while time.time() < deadline:
        if bootstrap_path.is_file():
            break
        time.sleep(0.5)
    else:
        fail("bootstrap password file never appeared")
    old = bootstrap_path.read_text(encoding = "utf-8").strip()
    new = f"Smoke-{secrets.token_urlsafe(16)}"
    status, body = api_json(
        "/api/auth/login",
        method = "POST",
        body = {"username": "unsloth", "password": old},
    )
    if status != 200:
        fail(f"bootstrap login failed: {status} {body!r}")
    status, body = api_json(
        "/api/auth/change-password",
        method = "POST",
        body = {"current_password": old, "new_password": new},
        token = body["access_token"],
    )
    if status != 200:
        fail(f"change-password failed: {status} {body!r}")
    return new, body["access_token"]


def wait_for_vite() -> None:
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/smoke-ansi.html", timeout = 2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.5)
    fail(f"vite dev server did not become ready at {BASE}")


def main() -> None:
    ART.mkdir(parents = True, exist_ok = True)

    studio_home = Path(tempfile.mkdtemp(prefix = "unsloth-ansi-studio-"))
    auth_dir = studio_home / "auth"
    auth_dir.mkdir(parents = True, exist_ok = True)
    studio_env = {
        **os.environ,
        "UNSLOTH_STUDIO_HOME": str(studio_home),
        "UNSLOTH_API_ONLY": "1",
    }

    studio = start_logged_process(
        [str(STUDIO_PY), "run.py", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
        cwd = BACKEND,
        env = studio_env,
    )
    vite = start_logged_process(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(VITE_PORT),
            "--strictPort",
        ],
        cwd = FRONTEND,
        env = {**os.environ},
    )
    try:
        wait_for_health(BACKEND_BASE, timeout = 180.0, info = info)
        wait_for_vite()
        _pw, access = bootstrap_password(auth_dir)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless = True, args = chromium_launch_args())
            page = browser.new_page()
            page.goto(f"{BASE}/login", wait_until = "domcontentloaded")
            page.evaluate(
                """([accessToken]) => {
                    localStorage.setItem('unsloth_auth_token', accessToken);
                }""",
                [access],
            )
            page.goto(f"{BASE}/smoke-ansi.html", wait_until = "domcontentloaded")
            page.wait_for_selector('section[data-smoke="tool-result-output"] pre', timeout = 30_000)
            page.screenshot(path = str(ART / "studio-smoke-ansi.png"), full_page = True)

            for section in ("tool-result-output", "tool-fallback-result"):
                pane = page.locator(f'section[data-smoke="{section}"] pre').first
                expect(pane).to_be_visible(timeout = 15_000)
                text = pane.inner_text()
                info(f"{section} on Studio stack: {text!r}")
                assert ESC not in text
                assert "[32m" not in text
                assert "file.txt" in text and "error" in text

            info("Studio stack smoke passed (backend up, vite dev, tool panes clean)")
            browser.close()
    finally:
        for proc in (vite, studio):
            proc.terminate()
            try:
                proc.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(studio_home, ignore_errors = True)


if __name__ == "__main__":
    main()
