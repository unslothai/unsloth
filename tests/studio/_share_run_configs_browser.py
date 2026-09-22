# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import socket
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse
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
    resolving = page.get_by_text("Resolving shared model…", exact = True)
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
            if url.path == "/api/hub/gguf-variants" and parse_qs(url.query).get("repo_id") == ["owner/Uncached-GGUF"]:
                expect(resolving).to_have_count(1)
                expect(resolving).to_be_visible()
                assert page.evaluate("window.sharingTest.snapshot().loaded") is None
                route.fulfill(status = 200, json = {
                    "repo_id": "owner/Uncached-GGUF",
                    "variants": [{"quant": "chosen/Q4_K_M", "filename": "chosen/model-Q4_K_M.gguf",
                                  "size_bytes": 1024, "downloaded": False}],
                    "default_variant": "chosen/Q4_K_M",
                    "has_vision": False,
                })
            else:
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
    assert urlparse(link).query == "run=1", "Startup-only link intake needs a new document from /chat"
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
    page.evaluate("window.sharingTest.receive({model:'owner/Unavailable',config:{customContextLength:2048}})")
    expect(page.get_by_text("Settings changed by link (1)", exact = True)).to_be_visible()
    expect(editor).to_be_visible()
    state = page.evaluate("window.sharingTest.snapshot()")
    assert state == {"thread": None, "project": None, "incognito": False, "pending": None, "loaded": None}, state
    assert urlparse(page.url).path == "/chat"
    assert any(path.startswith("/api/hub/") for path in requests)
    assert not any(path.startswith(("/api/hub/download", "/api/inference/load")) for path in requests), requests
    page.keyboard.press("Escape")
    expect(editor).not_to_be_visible()
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

    for variant in ["chosen/model-Q4_K_M.gguf", None]:
        page.goto(base + "/hub")
        page.wait_for_function("window.sharingTest !== undefined")
        value = {"model": "owner/Uncached-GGUF", "config": {"nParallel": 3}}
        if variant is not None:
            value["ggufVariant"] = variant
        page.evaluate("value => window.sharingTest.receive(value)", value)
        expect(page.get_by_text("Settings changed by link (1)", exact = True)).to_be_visible()
        expect(resolving).to_have_count(0)
        expect(editor).to_be_visible()
        assert page.evaluate("window.sharingTest.snapshot().loaded") is None
        editor.get_by_role("button", name = "Load model", exact = True).click()
        page.wait_for_function("window.sharingTest.snapshot().loaded !== null")
        loaded = page.evaluate("window.sharingTest.snapshot().loaded")
        assert loaded["meta"]["ggufVariant"] == "chosen/Q4_K_M", loaded
        assert loaded["meta"]["ggufFilename"] == "chosen/model-Q4_K_M.gguf", loaded
        assert loaded["meta"]["isDownloaded"] is False, loaded

    page.goto(base + "/hub")
    page.wait_for_function("window.sharingTest !== undefined")
    before = page.evaluate("window.sharingTest.snapshot()")
    page.evaluate("window.sharingTest.receive({model:'owner/Offline-GGUF',config:{nParallel:3}})")
    expect(page.get_by_text(
        "Could not look up the shared GGUF model. Check your connection and access to the Hugging Face model, then reopen the link.",
        exact = True,
    )).to_be_visible()
    expect(resolving).to_have_count(0)
    expect(editor).not_to_be_visible()
    assert page.url == base + "/hub"
    assert page.evaluate("window.sharingTest.snapshot()") == before

    page.goto(base + "/chat?run=1&choose-model=1#run?nParallel=3")
    chooser = page.get_by_role("dialog", name = "Choose a model", exact = True)
    expect(chooser).to_be_visible()
    assert page.url == base + "/chat?choose-model=1"
    page.keyboard.press("Escape")
    expect(chooser).not_to_be_visible()
    page.reload()
    page.wait_for_function("window.sharingTest !== undefined")
    expect(chooser).not_to_be_visible()
    assert page.evaluate("window.sharingTest.snapshot().pending") is None
    assert errors == [], errors
    context.close()


if __name__ == "__main__":
    run_shared_run_config_checks()
