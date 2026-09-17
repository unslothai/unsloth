# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does code that merely looks like a link definition cost a reply its code-block controls?

On the document render path the whole reply lexes into one block, so no block is a fence:
FenceBlock never mounts and the Copy code / Download file buttons go with it.

    code    two fences holding `[key: string]:` and `grid[row][col]`, so two of each button
    link    a genuine `[label][ref]` + `[ref]: url` reply, which must stay on the document path
    plain   a fence and no brackets. Positive control: no buttons here means nothing was measured

Run:
    python tests/studio/playwright_link_definition_probe.py
"""

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5231"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
PAGE = "smoke-link-definition-probe.html"
LABEL = os.environ.get("SMOKE_LABEL", "tree")

CASES = ("code", "link", "plain")

# Two fences in the code reply, one in each of the others.
EXPECTED_FENCES = {"code": 2, "link": 1, "plain": 1}


def info(message: str) -> None:
    print(message, flush = True)


def warm_up(page) -> None:
    """Load the positive control first and wait for its controls to actually mount.

    Shiki and the code action bar arrive as their own lazily loaded chunk. Until it has landed
    once, every case reads zero buttons, and a quiet-interval wait cannot tell "not loaded yet"
    apart from "settled at zero" -- three quiet samples can all land inside the load. The
    `plain` case must end with controls on every branch, so waiting for them here is an
    unambiguous readiness signal, and it leaves the chunk cached for the cases that follow.
    """
    page.goto(f"{BASE}/{PAGE}?case=plain", wait_until = "domcontentloaded")
    page.wait_for_function("() => window.__probe && window.__probe.ready()", timeout = 60_000)
    page.wait_for_function(
        "() => window.__probe.counts().copyButtons >= 1",
        timeout = 120_000,
        polling = 250,
    )


def run_case(page, case: str) -> dict:
    page.goto(f"{BASE}/{PAGE}?case={case}", wait_until = "domcontentloaded")
    page.wait_for_function("() => window.__probe && window.__probe.ready()", timeout = 60_000)

    # The fences themselves mount independently of the action bar, and their count is known per
    # case, so this is a real completion signal rather than an interval. The button count is
    # deliberately NOT waited on: `code` is legitimately zero before the fix, and waiting for a
    # number that never arrives would fail the branch that is supposed to reproduce the defect.
    expected = EXPECTED_FENCES[case]
    page.wait_for_function(
        f"() => window.__probe.counts().codeBlocks === {expected}",
        timeout = 60_000,
        polling = 250,
    )
    # Then let the action bar settle, which is quick now the chunk is warm.
    page.wait_for_function(
        """() => {
            const now = JSON.stringify(window.__probe.counts());
            const seen = window.__settled;
            window.__settled = now === seen?.value
                ? {value: now, hits: seen.hits + 1}
                : {value: now, hits: 0};
            return window.__settled.hits >= 3;
        }""",
        timeout = 60_000,
        polling = 250,
    )
    return page.evaluate("() => window.__probe.counts()")


def main() -> int:
    proc = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        proc = start_vite(PORT)
    wait_for_smoke_page(
        f"{BASE}/{PAGE}", "smoke-link-definition-probe-main.tsx", proc = proc, info = info
    )
    results = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(args = chromium_launch_args())
            page = browser.new_page(viewport = {"width": 1280, "height": 900})
            warm_up(page)
            for case in CASES:
                results[case] = run_case(page, case)
                info(f"{case}: {json.dumps(results[case])}")
            browser.close()
    finally:
        if proc is not None:
            stop_process(proc)
            info("vite stopped")

    print(json.dumps({"label": LABEL, "results": results}, indent = 2))

    failures = []
    plain = results["plain"]
    if plain["copyButtons"] < 1 or plain["downloadButtons"] < 1:
        failures.append(
            "the `plain` control has no code-block controls, so the fixture measured nothing "
            "and no other row means anything"
        )
    code = results["code"]
    if code["copyButtons"] != EXPECTED_FENCES["code"]:
        failures.append(
            f"`code` mounted {code['copyButtons']} Copy code buttons, expected "
            f"{EXPECTED_FENCES['code']}: ordinary code took the document render path"
        )
    if code["downloadButtons"] != EXPECTED_FENCES["code"]:
        failures.append(
            f"`code` mounted {code['downloadButtons']} Download file buttons, expected "
            f"{EXPECTED_FENCES['code']}"
        )
    link = results["link"]
    if link["renderedLinks"] < 2:
        failures.append(
            f"`link` resolved {link['renderedLinks']} anchors, expected 2: a real reference "
            "link stopped resolving"
        )

    for line in failures:
        info("FAIL: " + line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
