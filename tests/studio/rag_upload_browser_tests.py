# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real browser runs of the shipped React hook and upload/SSE transport."""

import json
import os
from pathlib import Path
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(os.environ.get("RAG_SIM_ART_DIR", "./temp/rag-upload")).resolve()
ROOT.mkdir(parents = True, exist_ok = True)
BASE = "http://127.0.0.1:18948"


def request(path, body = None):
    req = urllib.request.Request(
        BASE + path,
        data = json.dumps(body).encode() if body is not None else None,
        headers = {"Content-Type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req, timeout = 15))


def wait_state(predicate):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = request("/__state")
        if predicate(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"Server condition was not reached: {state}")


def run_case(browser, mode, action):
    request("/__scenario", {"mode": mode})
    context = browser.new_context()
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda err: errors.append(str(err)))
    try:
        page.goto(BASE)
        page.wait_for_selector("#state")
        action(page)
        assert not errors, errors
        assert not page.evaluate("window.pageErrors"), "Unexpected browser error"
    finally:
        request("/__release", {})
        context.close()


def complete(page):
    # Three engines share a 2-core runner, so this waits on scheduling, not on the app.
    page.wait_for_function(
        "window.sim.documents.length>0 && window.sim.documents.every(d=>d.status==='completed') && !window.sim.uploading && !window.sim.hasIndexing",
        timeout = 30000,
    )
    assert not page.evaluate("window.sim.hasIndexing || window.sim.uploading")
    assert page.evaluate("window.errors") == []


def ordinary(ext):
    def action(page):
        page.locator("#files").set_input_files(
            {
                "name": f"Café 東京.{ext}",
                "mimeType": "application/octet-stream",
                "buffer": b"fixture document",
            }
        )
        complete(page)
        data = request("/__state")
        assert (
            len(data["uploads"]) == 1
            and "multipart/form-data; boundary=" in data["uploads"][0]["multipart"]
        )

    return action


def reselect(page):
    page.evaluate("window.sim.uploadNames(['report.pdf'])")
    complete(page)
    page.evaluate("window.sim.reselect()")
    assert page.evaluate("window.resolveCalls") == 0
    assert len(request("/__state")["uploads"]) == 1
    complete(page)


def early_end(page):
    page.evaluate("window.sim.uploadNames(['report.txt'])")
    page.wait_for_function("window.sim.documents[0]?.status==='running'")
    assert page.evaluate("window.sim.hasIndexing")
    request("/__release", {})
    complete(page)


def error(page):
    page.evaluate("window.sim.uploadNames(['report.pdf'])")
    page.wait_for_function(
        "window.errors.length===1 && !window.sim.hasIndexing && !window.sim.uploading",
        timeout = 30000,
    )
    assert not page.evaluate("window.sim.hasIndexing || window.sim.uploading")
    assert page.evaluate("window.sim.documents") == []


def failed_scope(page):
    page.evaluate("window.sim.uploadNames(['report.pdf'],true)")
    assert len(page.evaluate("window.errors")) == 1
    assert page.evaluate("window.sim.documents") == []
    assert not request("/__state")["uploads"]


def duplicate_running(page):
    page.evaluate("window.sim.uploadNames(['first.txt'])")
    page.evaluate("window.sim.uploadNames(['same-content.txt'])")
    page.wait_for_function("window.sim.documents.length===1 && !window.sim.uploading")
    assert page.evaluate("window.sim.hasIndexing")
    assert len(request("/__state")["eventRequests"]) == 1
    request("/__release", {})
    complete(page)


def navigate(page):
    page.evaluate("window.pending=window.sim.uploadNames(['old.txt']);void 0")
    page.wait_for_function("window.sim.uploading")
    wait_state(lambda s: s["waiting"] == 1)
    page.locator("#switch").click()
    page.wait_for_function("window.sim.scopeId==='other'")
    request("/__release", {})
    page.evaluate("window.pending")
    page.wait_for_timeout(150)
    assert not request("/__state")["eventRequests"], "Old upload began tracking after navigation"
    assert page.evaluate("window.errors") == []
    assert page.evaluate("window.sim.documents") == []


def unmount(page):
    page.evaluate("window.pending=window.sim.uploadNames(['old.txt']);void 0")
    page.wait_for_function("window.sim.uploading")
    wait_state(lambda s: s["waiting"] == 1)
    page.locator("#unmount").click()
    request("/__release", {})
    page.evaluate("window.pending")
    page.wait_for_timeout(150)
    assert not request("/__state")["eventRequests"], "Unmounted component began job tracking"
    assert page.evaluate("window.errors") == []


def large_batch(page):
    page.evaluate(
        "window.pending=window.sim.uploadNames(Array.from({length:12},(_,i)=>'report-'+i+'.txt'));void 0"
    )
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and len(request("/__state")["uploads"]) < 12:
        time.sleep(0.05)
    data = request("/__state")
    assert len(data["uploads"]) == 12, f"Only {len(data['uploads'])}/12 POSTs arrived with SSE open"
    request("/__release", {})
    page.evaluate("window.pending")
    complete(page)


def overlapping_uploads(page):
    page.evaluate("window.first=window.sim.uploadNames(['first.txt']);void 0")
    wait_state(lambda s: s["waiting"] == 1)
    page.evaluate("window.second=window.sim.uploadNames(['second.txt']);void 0")
    wait_state(lambda s: s["waiting"] == 2)
    request("/__release-one", {})
    page.evaluate("window.first")
    assert page.evaluate(
        "window.sim.uploading"
    ), "The first upload released the second upload's guard"
    request("/__release", {})
    page.evaluate("window.second")
    complete(page)


def concurrent_same_content(page):
    page.evaluate(
        "window.first=window.sim.uploadNames(['first.txt']);window.second=window.sim.uploadNames(['alias.txt']);void 0"
    )
    wait_state(lambda s: s["waiting"] == 2)
    request("/__release", {})
    page.evaluate("Promise.all([window.first,window.second])")
    complete(page)
    assert (
        page.evaluate("window.sim.documents.length") == 1
    ), "Concurrent deduplication left duplicate chips"


def materialize(page):
    page.goto(BASE + "/?empty")
    page.wait_for_selector("#state")
    page.evaluate("window.sim.materialize()")
    complete(page)


CASES = [
    ("file-" + ext, "normal", ordinary(ext))
    for ext in ("pdf", "txt", "md", "markdown", "docx", "html", "htm")
]
CASES += [
    ("duplicate-selection", "normal", reselect),
    ("get-only-server", "get-only", ordinary("pdf")),
    ("early-stream-end", "early-end", early_end),
    ("unavailable-sse", "sse-unavailable", early_end),
    ("indexing-error", "error", error),
    ("upload-rejection", "reject", error),
    ("chat-initialization-failure", "normal", failed_scope),
    ("inflight-duplicate", "server-duplicate", duplicate_running),
    ("navigate-during-post", "delayed-fail", navigate),
    ("unmount-during-post", "delayed-fail", unmount),
    ("twelve-files-open-streams", "hold", large_batch),
    ("overlapping-uploads", "delayed", overlapping_uploads),
    ("concurrent-same-content", "delayed-server-duplicate", concurrent_same_content),
    ("lazy-thread-materialization", "normal", materialize),
]


def main():
    records = []
    engines = [arg for arg in sys.argv[1:] if arg != "--probe"] or ["chromium", "firefox", "webkit"]
    with sync_playwright() as pw:
        for engine in engines:
            if engine == "safari":
                from rag_upload_safari import SafariBrowser
                browser = SafariBrowser(ROOT)
            elif engine == "chrome":
                browser = pw.chromium.launch(channel = "chrome", headless = True)
            elif engine == "edge":
                exe = next(
                    (ROOT / "edge-unpacked").rglob(
                        "Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
                    ),
                    None,
                )
                browser = (
                    pw.chromium.launch(executable_path = str(exe), headless = True)
                    if exe
                    else pw.chromium.launch(channel = "msedge", headless = True)
                )
            else:
                browser = getattr(pw, engine).launch(headless = True)
            version = browser.version
            if "--probe" in sys.argv:
                browser.close()
                continue
            for name, mode, action in CASES:
                start = time.monotonic()
                try:
                    run_case(browser, mode, action)
                    record = {
                        "browser": engine,
                        "version": version,
                        "case": name,
                        "status": "passed",
                    }
                except Exception as exc:
                    record = {
                        "browser": engine,
                        "version": version,
                        "case": name,
                        "status": "failed",
                        "error": str(exc),
                    }
                record["seconds"] = round(time.monotonic() - start, 3)
                print(json.dumps(record), flush = True)
                records.append(record)
            browser.close()
    (ROOT / ("browser-results-" + "-".join(engines) + ".json")).write_text(
        json.dumps(records, indent = 2), encoding = "utf-8"
    )
    raise SystemExit(any(r["status"] == "failed" for r in records))


if __name__ == "__main__":
    main()
