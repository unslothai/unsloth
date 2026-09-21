# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import socket
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

from _playwright_robust import chromium_launch_args, stop_process


def run_shared_run_config_checks():
    frontend = Path(__file__).resolve().parents[2] / "studio" / "frontend"
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "Install Node.js and npm to run the shared run settings browser check."
        )
    if not (frontend / "node_modules" / "vite" / "bin" / "vite.js").is_file():
        raise RuntimeError(
            "Missing frontend dependencies. Run `npm ci --prefix studio/frontend` "
            "from the repository root before running this browser check."
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode = "w+") as log:
        process_group = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else {"start_new_session": True}
        )
        vite = subprocess.Popen(
            [npm, "run", "dev", "--", "--config", "tests/fixtures/share-run-configs/vite.config.ts",
             "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
            cwd = frontend, stdout = log, stderr = subprocess.STDOUT,
            **process_group,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urlopen(base, timeout = 1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if vite.poll() is not None or time.monotonic() >= deadline:
                    log.seek(0)
                    raise RuntimeError(
                        "Shared run settings Vite fixture failed to start:\n" + log.read()
                    )
                time.sleep(0.1)
            with sync_playwright() as playwright:
                engine = os.environ.get("STUDIO_PLAYWRIGHT_BROWSER", "chromium")
                options = {"headless": True}
                if engine == "chromium":
                    options["args"] = chromium_launch_args()
                if os.environ.get("PW_EXECUTABLE_PATH"):
                    options["executable_path"] = os.environ["PW_EXECUTABLE_PATH"]
                browser = getattr(playwright, engine).launch(**options)
                try:
                    _checks(browser, base)
                finally:
                    browser.close()
        finally:
            stop_process(vite)
    print("[ui-shared-run-config] Shared run settings browser checks: PASS", flush = True)


def _checks(browser, base):
    context = browser.new_context(reduced_motion = "reduce")
    page = context.new_page()
    errors = []
    requests = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def respond(route):
        url = urlparse(route.request.url)
        if not route.request.url.startswith(base + "/"):
            errors.append(f"Unexpected external request: {url.netloc}{url.path}")
            route.abort()
        elif url.path.startswith("/api/hub/"):
            requests.append(url.path)
            route.fulfill(status = 503, json = {"detail": "Inventory unavailable"})
        elif url.path.startswith("/api/"):
            requests.append(url.path)
            payload = {"overrides": {}} if url.path.endswith("/overrides") else {}
            route.fulfill(status = 200, json = payload)
        else:
            route.continue_()

    context.route("**/*", respond)
    page.goto(base + "/chat")
    page.wait_for_function("window.sharingTest !== undefined")
    before = page.evaluate("window.sharingTest.snapshot()")
    page.evaluate("""async () => {
      const changed = new Promise(resolve => window.addEventListener('hashchange', resolve, {once:true}));
      location.hash = 'run?v=1&customContextLength=2048';
      await changed;
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
    }""")
    assert page.evaluate("window.sharingTest.snapshot()") == before

    link = page.evaluate("window.sharingTest.link({isGguf:true, config:{customContextLength:4096}})")
    assert urlparse(link).query == "run=1"
    page.evaluate("""url => {
      window.documentBeforeLink = true;
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.textContent = 'Open shared settings';
      document.body.append(anchor);
    }""", link)
    page.get_by_role("link", name = "Open shared settings").click()
    expect(page.get_by_text("Settings changed by link (1)", exact = True)).to_be_visible()
    assert page.evaluate("window.documentBeforeLink === undefined")
    state = page.evaluate("window.sharingTest.snapshot()")
    assert state == {"thread": None, "project": None, "incognito": False, "pending": None, "loaded": None}, state
    assert not any(path.startswith("/api/hub/") for path in requests), requests

    editor = page.locator('[role="dialog"][aria-label^="Run settings"]')
    expect(editor).to_be_visible()
    editor.get_by_role("button", name = "Share", exact = True).click()
    share = page.get_by_role("dialog", name = "Share run settings", exact = True)
    expect(share).to_be_visible()
    share.get_by_label("Shareable link", exact = True).focus()
    expect(editor).to_be_visible()
    page.keyboard.press("Escape")
    expect(share).not_to_be_visible()
    expect(editor).to_be_visible()
    editor.get_by_role("button", name = "Load model", exact = True).click()
    page.wait_for_function("window.sharingTest.snapshot().loaded !== null")
    loaded = page.evaluate("window.sharingTest.snapshot().loaded")
    assert loaded["meta"]["isGguf"] is False, loaded
    assert loaded["meta"]["config"]["customContextLength"] == 4096, loaded
    assert loaded["id"] == "owner/Native", loaded
    assert loaded["meta"]["loadId"] == "/cache/native", loaded

    page.goto(base + "/hub")
    page.wait_for_function("window.sharingTest !== undefined")
    before = page.evaluate("window.sharingTest.snapshot()")
    page.evaluate("window.sharingTest.receive({model:'owner/Unavailable',config:{customContextLength:2048}})")
    expect(page.get_by_text("Could not check local model availability. Reopen the link to try again.", exact = True)).to_be_visible()
    assert page.url == base + "/hub"
    assert page.evaluate("window.sharingTest.snapshot()") == before
    assert any(path.startswith("/api/hub/") for path in requests)
    page.get_by_role("button", name = "Share local settings", exact = True).click()
    share = page.get_by_role("dialog", name = "Share run settings", exact = True)
    expect(share.get_by_role("checkbox", name = "Model format", exact = True)).not_to_be_checked()
    expect(share.get_by_text("This model uses a local path.", exact = False)).to_be_visible()
    share.get_by_label("Open in", exact = True).click()
    page.get_by_role("option", name = "Unsloth desktop app", exact = True).click()
    expect(share.get_by_role("status")).to_contain_text("Long desktop links may not open on Windows")
    assert share.get_by_label("Shareable link", exact = True).input_value().startswith("unsloth://run?")
    expect(share.get_by_role("button", name = "Copy link", exact = True)).to_be_enabled()
    share.get_by_label("Open in", exact = True).click()
    page.get_by_role("option", name = "This Studio web address", exact = True).click()
    expect(share.get_by_role("status")).to_have_count(0)
    assert errors == [], errors
    context.close()


if __name__ == "__main__":
    run_shared_run_config_checks()
