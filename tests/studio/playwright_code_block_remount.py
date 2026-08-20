# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does a code block collapse to the placeholder height when it is re-created on a QUIET thread?

Sibling of playwright_code_block_flicker.py, which measures stream FINALIZATION. This one never
streams at all. Its subject is the state after the thread has gone quiet and the hold has been
released -- `data-code-block-layout="settled"` on the thread root, `content-visibility: auto` live
on every block -- at which point anything that mounts a BRAND NEW block element is mounting one
with no LAST REMEMBERED SIZE, which Chromium lays out at streamdown's 200px
`contain-intrinsic-size` fallback until it renders it.

`thread.isRunning` is false through all of it, so none of these is reachable from the run state:

    edit       Leaving the edit textarea on a completed reply. thread.tsx renders an editing
               message as a <textarea> and any other one as its rendered parts, two different
               element types, so React unmounts one subtree and mounts the other.
    branch     Switching response branches. markdown-text.tsx keys <Streamdown> on the message
               id and sibling branches are distinct messages, so the markdown subtree is
               re-created wholesale.
    reasoning  Expanding a collapsed reasoning disclosure. ui/collapsible.tsx wraps Radix
               CollapsibleContent with no `forceMount`, so closed content is not in the document
               at all, and reasoning bodies render through MarkdownText like any other part.

Each is driven through the real component path -- the store field the editor reads, the runtime's
own switchToBranch, an actual click on the disclosure -- rather than by mutating the DOM, because
the question is whether React re-creates those elements and a harness that re-created them itself
would be answering its own question.

WHAT IS MEASURED

Per frame, every `[data-streamdown="code-block"]`'s rendered height, the viewport-relative top of
whatever follows it, and the thread's own scrollHeight. A COLLAPSE is a frame in which a block is
drawn in the 150-320px band while the settled thread has one taller than 400px: that is the
fallback specifically, not some other height.

VARIANTS, AND WHY THE RUN DRIVES MORE THAN ONE

A check that only ever runs against the tree cannot tell "nothing collapsed" from "the harness
reproduces nothing". So each path is driven twice:

    released   streamdown's own `content-visibility: auto` forced past the hold, i.e. the thread
               with no hold at all. Every path MUST collapse here. If one does not, the harness
               is not reproducing that path and its `tree` result means nothing.
    tree       whatever src/index.css and the signal ship right now. Nothing may collapse.

Run:
    python tests/studio/playwright_code_block_remount.py
    SMOKE_REMOUNT_PATHS=branch python tests/studio/playwright_code_block_remount.py

It starts and stops its own vite dev server; SMOKE_PORT moves the one it starts. Exits non-zero
when a path collapsed under `tree`, or when a path did NOT collapse under `released`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

ROOT = Path(__file__).resolve().parents[2]
PORT = int(os.environ.get("SMOKE_PORT", "5233"))
BASE = os.environ.get("SMOKE_BASE_URL", f"http://127.0.0.1:{PORT}")
VIEWPORT = {"width": 1180, "height": 900}
PATHS = os.environ.get("SMOKE_REMOUNT_PATHS", "edit,branch,reasoning").split(",")
MODES = os.environ.get("SMOKE_REMOUNT_MODES", "released,tree").split(",")
OUT = Path(os.environ.get("SMOKE_REMOUNT_OUT", ROOT / "logs" / "code_block_remount"))

# A block the harness seeds is ~1,760px; the fallback is 200px plus the block's own chrome, which
# measures at 226px on this tree. The band is wide enough to catch the fallback under a different
# header height and narrow enough not to catch a genuinely short block.
FALLBACK_LO = 150
FALLBACK_HI = 320
TALL_PX = 400
# How many frames after the trigger the steady height is read from. The reasoning disclosure
# animates for ~12 frames, so the tail has to start after any block has finished appearing.
STEADY_FROM = 6


def wait_settled(page, timeout_ms: int = 20_000) -> None:
    """Block until the thread root says `settled`: before that the hold is on and nothing is at risk."""
    page.wait_for_function(
        "() => document.querySelector('.aui-thread-root')"
        "?.getAttribute('data-code-block-layout') === 'settled'",
        timeout=timeout_ms,
    )


def collapses(frames: list[dict]) -> dict:
    """Frames after the trigger in which a block was drawn at the fallback height."""
    trigger = next((i for i, f in enumerate(frames) if f["mark"]), None)
    if trigger is None:
        return {"trigger": None, "frames": [], "steady": 0}
    tail = frames[trigger + STEADY_FROM :] or frames[-3:]
    steady = max((max(f["heights"], default=0) for f in tail), default=0)
    found = []
    if steady > TALL_PX:
        for f in frames[trigger:]:
            for height in f["heights"]:
                if FALLBACK_LO <= height <= FALLBACK_HI:
                    found.append([f["n"] - frames[trigger]["n"], height])
                    break
    return {"trigger": trigger, "frames": found, "steady": steady}


def run_path(page, path: str, mode: str) -> dict:
    page.goto(
        f"{BASE}/smoke-code-block-remount.html?css={mode}",
        wait_until="domcontentloaded",
    )
    page.wait_for_function("Boolean(window.__remount)", timeout=120_000)
    message_id = page.evaluate("(p) => window.__remount.seed(p)", path)
    page.wait_for_timeout(1500)
    page.evaluate("() => window.__remount.park()")
    # Highlighting settles the block heights, and the hold is only released after it.
    page.wait_for_timeout(2500)
    wait_settled(page)
    before = page.evaluate("() => window.__remount.counts()")
    computed = page.evaluate("() => window.__remount.computedFor(0)")

    if path == "edit":
        # Entering the edit textarea is not the measured half: it REMOVES blocks. Do it, let the
        # thread go quiet again, and measure the return.
        page.evaluate("(id) => window.__remount.enterEdit(id)", message_id)
        page.wait_for_timeout(1500)
        wait_settled(page)
        driver = "() => window.__remount.leaveEdit()"
    elif path == "branch":
        driver = "() => window.__remount.switchBranch()"
    elif path == "reasoning":
        driver = "() => window.__remount.expandReasoning()"
    else:
        raise SystemExit(f"unknown path {path}")

    page.evaluate("() => window.__remount.startSampling()")
    page.wait_for_timeout(120)
    detail = page.evaluate(driver)
    page.wait_for_timeout(1200)
    page.evaluate("() => window.__remount.stopSampling()")
    frames = page.evaluate("() => window.__remount.results().frames")
    return {
        "path": path,
        "mode": mode,
        "messageId": message_id,
        "detail": detail,
        "before": before,
        "computed": computed,
        "after": page.evaluate("() => window.__remount.counts()"),
        "frames": frames,
        "verdict": collapses(frames),
    }


def report(result: dict, window: int = 8) -> None:
    verdict = result["verdict"]
    print(f"\n=== path={result['path']}  css={result['mode']} ===")
    print(f"    seeded: {result['before']}")
    print(f"    computed on block 0: {result['computed']}")
    print(f"    driver returned: {result['detail']}")
    if verdict["trigger"] is None:
        print("    NO TRIGGER FRAME RECORDED")
        return
    frames = result["frames"]
    base = frames[verdict["trigger"]]["n"]
    print("    [frame, heights, top of what follows, scrollHeight, layout attr]")
    for f in frames[max(0, verdict["trigger"] - 2) : verdict["trigger"] + window]:
        tag = "  <-- trigger" if f["mark"] else ""
        print(
            f"      [{f['n'] - base:+d}, {f['heights']}, {f['nextTops']},"
            f" {f['scrollHeight']}, {f['layoutAttr']}]{tag}"
        )
    print(f"    steady tallest block: {verdict['steady']}px")
    print(
        f"    collapsed frames: {verdict['frames']}"
        if verdict["frames"]
        else "    collapsed frames: none"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = None if os.environ.get("SMOKE_BASE_URL") else start_vite(PORT)
    results: list[dict] = []
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-code-block-remount.html",
            "smoke-code-block-remount-main.tsx",
            proc=proc,
            info=print,
        )
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args=["--no-sandbox"])
            page = browser.new_page(viewport=VIEWPORT)
            page.on("pageerror", lambda e: print(f"[pageerror] {e}"))
            for mode in MODES:
                for path in PATHS:
                    result = run_path(page, path, mode)
                    report(result)
                    results.append(result)
            browser.close()
    finally:
        if proc is not None:
            stop_process(proc)

    (OUT / "frames.json").write_text(json.dumps(results, indent=1))
    failures = []
    for result in results:
        collapsed = bool(result["verdict"]["frames"])
        if result["mode"] == "tree" and collapsed:
            failures.append(
                f"{result['path']}: the tree collapsed to the fallback on"
                f" {result['verdict']['frames']}"
            )
        if result["mode"] == "released" and not collapsed:
            failures.append(
                f"{result['path']}: the released control did NOT collapse, so this path"
                " is not being reproduced and its tree result means nothing"
            )
    print(f"\nwrote {OUT / 'frames.json'}")
    if failures:
        for line in failures:
            print(f"FAIL {line}")
        return 1
    print("PASS every path collapses without the hold and none collapses with it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
