# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does a code block flicker through a placeholder height when a stream finalizes?

Streamdown puts `content-visibility: auto` with `contain-intrinsic-size: auto 200px` inline on
every code-block wrapper. Such an element has no LAST REMEMBERED SIZE until it has rendered once,
so until then it lays out at the 200px fallback rather than at its content height. The re-render
at the end of a stream REPLACES the code-block node, and a replaced node is new, so it can lay out
at 200px for a frame and then snap to its real height. That one-frame change is the "reload"
flicker studio/frontend/src/index.css describes, and why the override there exists.

WHAT IS MEASURED

Not a timing. Per frame, for every `[data-streamdown="code-block"]`:

    collapses          a block at least 400px tall rendering at half that or less and coming
                       back. This is the flicker, and one is too many.
    placeholder frames how many of those landed in the 150-300px band, i.e. on the
                       `contain-intrinsic-size` fallback, so a collapse can be attributed.
    scroll height dips the thread's scrollHeight falling 300px or more and recovering: a
                       collapsing block takes the whole column with it.
    anchor shift       DOCUMENT-space movement of the last message that existed BEFORE the
                       stream. Nothing about it changes while the stream runs, so any movement is
                       settled content relaid out under the user. Document space is the point:
                       viewport-relative would count every scroll as a shift.

Then a second phase on the thread the stream left behind: scroll bottom to top and record whether
the content moved.

    sweep shift        frames in which some block's DOCUMENT-space top moved. A never-rendered
                       block is skipped at the 200px fallback, not at its real height, which
                       shows only on the way back up as each block expands when reached and
                       pushes what is below it down.
    scrollHeight grew  change in the thread's own height over the gesture. Zero means it was
                       already the right height before the sweep.

VARIANTS, AND WHY THE RUN DRIVES MORE THAN ONE

A check that only runs against the tree cannot tell "no flicker" from "the fixture reproduces
nothing". So the fixture is driven under stylesheets appended after the tree's own, and the run
asserts on the SHAPE of the whole set:

    streamdown   streamdown's inline defaults, the tree before any override. Positive control:
                 this MUST flicker, or the fixture measured nothing and the run fails whatever
                 the tree scored.
    released     the override released for every block at all times, streaming included: the
                 mistake scoping avoids. Second positive control, so it MUST flicker too.
    legacy       the override as first written: `content-visibility: visible`,
                 `contain-intrinsic-size: none`. Must not flicker.
    tree         whatever src/index.css ships now. Must not flicker.
    statusonly   held only while the part is running. The obvious CSS-only scoping, measured
                 rather than assumed: node replacement at fence close can land in the same commit
                 as the status flip, so it still flickers -- hence the tree's settle window.
    lastmessage  held only for the last message. Survives finalization but cannot give an earlier
                 message's blocks their first render, so it pays in the sweep phase instead.

Neither of the last two decides the exit code; they are here because "the simpler thing does not
work" needs a number next to it.

Run:
    python tests/studio/playwright_code_block_flicker.py
    SMOKE_FLICKER_ENGINES=chromium,webkit python tests/studio/playwright_code_block_flicker.py
    SMOKE_FLICKER_VARIANTS=tree,streamdown python tests/studio/playwright_code_block_flicker.py

It starts and stops its own vite dev server; SMOKE_BASE_URL points it at an existing one and
SMOKE_PORT moves the one it starts. Exits non-zero when a variant that must not flicker did, one
that must flicker did not, or the fixture failed to build the thread it claims to.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _code_block_flicker_analysis import (  # noqa: E402
    analyse_stream,
    analyse_sweep,
)
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5219"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
ENTRY = "smoke-code-block-flicker-main.tsx"
PAGE = "smoke-code-block-flicker.html"
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-code-block-flicker"))
OUT.mkdir(parents = True, exist_ok = True)
LABEL = os.environ.get("SMOKE_LABEL", "tree")

ENGINES = [
    e.strip() for e in os.environ.get("SMOKE_FLICKER_ENGINES", "chromium").split(",") if e.strip()
]
VARIANTS = [
    v.strip()
    for v in os.environ.get("SMOKE_FLICKER_VARIANTS", "tree,legacy,released,streamdown").split(",")
    if v.strip()
]
REPEATS = int(os.environ.get("SMOKE_FLICKER_REPEATS", "3"))
PARK = os.environ.get("SMOKE_FLICKER_PARK", "bottom")

HISTORY_MESSAGES = int(os.environ.get("SMOKE_FLICKER_HISTORY", "8"))
FENCES = int(os.environ.get("SMOKE_FLICKER_FENCES", "3"))
LINES_PER_FENCE = int(os.environ.get("SMOKE_FLICKER_FENCE_LINES", "22"))
CHUNK_CHARS = int(os.environ.get("SMOKE_FLICKER_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_FLICKER_GAP_MS", "8"))
# Sampling kept alive past `done`:
TAIL_MS = int(os.environ.get("SMOKE_FLICKER_TAIL_MS", "2500"))


MUST_FLICKER = {"streamdown", "released"}
MUST_NOT_FLICKER = {"tree", "legacy"}

# The positive control is what makes a clean run mean anything: without a variant REQUIRED to flicker, "no collapses
# anywhere" is equally consistent with a detector that measured nothing.
if not MUST_FLICKER & set(VARIANTS):
    raise SystemExit(
        "SMOKE_FLICKER_VARIANTS="
        + ",".join(VARIANTS)
        + " has no positive control. Include at least one of "
        + ", ".join(sorted(MUST_FLICKER))
        + ", or a run that reports no collapses proves only that nothing was measured."
    )

# What each variant must have computed to on a settled block, checked before anything is measured.
# Without this guard: the tree's override lives in `@layer utilities`, and for IMPORTANT declarations the cascade
# REVERSES layer order, so an unlayered `!important` variant silently loses to it.
EXPECTED_COMPUTED = {
    "streamdown": {"contentVisibility": "auto"},
    "released": {"contentVisibility": "auto"},
    "legacy": {"contentVisibility": "visible", "containIntrinsicSize": "none"},
    "statusonly": {"contentVisibility": "auto"},
    "lastmessage": {"contentVisibility": "auto"},
}

# once quiet), then reported zero flickers, reading as "there was never anything to fix".
# The same guard WHILE THE STREAM IS RUNNING, on the block being streamed.
EXPECTED_COMPUTED_RUNNING = {
    "tree": {"contentVisibility": "visible"},
    "legacy": {"contentVisibility": "visible"},
    "streamdown": {"contentVisibility": "auto"},
    "released": {"contentVisibility": "auto"},
    "statusonly": {"contentVisibility": "visible"},
    "lastmessage": {"contentVisibility": "visible"},
}

SWEEP_STEPS = int(os.environ.get("SMOKE_FLICKER_SWEEP_STEPS", "40"))
SWEEP_STEP_PX = int(os.environ.get("SMOKE_FLICKER_SWEEP_PX", "500"))


def info(message: str) -> None:
    print(message, flush = True)


def settle_highlighting(page) -> int:
    """Five stable reads a quarter second apart, the gate PR 9016 had to fix twice.

    Two adjacent reads can land in the lull between async Shiki batches and release the gate on a
    thread that is still building.
    """
    stable = 0
    last = -1
    for _ in range(200):
        count = page.evaluate("window.__flicker.counts().highlightedTokens")
        if count == last and count > 0:
            stable += 1
            if stable >= 5:
                return count
        else:
            stable = 0
            last = count
        page.wait_for_timeout(250)
    raise RuntimeError(f"highlighting never settled (last count {last})")


def run_case(page, variant: str) -> dict:
    page.goto(f"{BASE}/{PAGE}?css={variant}", wait_until = "domcontentloaded")
    page.wait_for_function("Boolean(window.__flicker)", timeout = 120_000)
    page.evaluate("(n) => window.__flicker.seed(n)", HISTORY_MESSAGES)
    page.wait_for_function(
        "(n) => window.__flicker.counts().messages >= n",
        arg = HISTORY_MESSAGES * 2,
        timeout = 120_000,
    )
    tokens = settle_highlighting(page)
    seeded = page.evaluate("window.__flicker.counts()")
    computed = page.evaluate("window.__flicker.computedFor(0)")
    for prop, want in EXPECTED_COMPUTED.get(variant, {}).items():
        if computed.get(prop) != want:
            raise RuntimeError(
                f"variant {variant}: {prop} computed to {computed.get(prop)!r}, expected {want!r}. "
                "The variant stylesheet did not win the cascade, so this run would have measured "
                "the tree under another name."
            )
    page.evaluate("(m) => window.__flicker.park(m)", PARK)
    page.wait_for_timeout(250)
    blocks_before = page.evaluate("window.__flicker.startSampling()")
    page.evaluate(
        "(o) => window.__flicker.run(o)",
        {
            "historyMessages": HISTORY_MESSAGES,
            "fences": FENCES,
            "linesPerFence": LINES_PER_FENCE,
            "chunkChars": CHUNK_CHARS,
            "gapMs": GAP_MS,
            "park": PARK,
        },
    )
    # Mid-stream cascade guard, on the block being written: the check the settled one cannot make.
    page.wait_for_function("window.__flicker.results().streamStartedAt !== null", timeout = 120_000)
    page.wait_for_timeout(400)
    running_computed = page.evaluate(
        "() => window.__flicker.computedFor(window.__flicker.counts().codeBlocks - 1)"
    )
    still_running = not page.evaluate("window.__flicker.results().done")
    if still_running:
        for prop, want in EXPECTED_COMPUTED_RUNNING.get(variant, {}).items():
            if running_computed.get(prop) != want:
                raise RuntimeError(
                    f"variant {variant}: mid-stream {prop} computed to "
                    f"{running_computed.get(prop)!r}, expected {want!r}. The rule that should "
                    "be in force while a block is streaming is not the one that is."
                )

    page.wait_for_function("window.__flicker.results().done === true", timeout = 300_000)
    page.wait_for_timeout(TAIL_MS)
    page.evaluate("window.__flicker.stopSampling()")
    results = page.evaluate("window.__flicker.results()")
    after = page.evaluate("window.__flicker.counts()")
    if results["error"]:
        raise RuntimeError(f"variant {variant}: stream failed: {results['error']}")
    stats = analyse_stream(results["frames"])

    # Phase two, on the thread the stream left behind:
    page.evaluate("window.__flicker.startSampling()")
    sweep_meta = page.evaluate(
        "(a) => window.__flicker.sweepUp(a.steps, a.px)",
        {"steps": SWEEP_STEPS, "px": SWEEP_STEP_PX},
    )
    page.evaluate("window.__flicker.stopSampling()")
    sweep_frames = page.evaluate("window.__flicker.results().frames")
    stats.update(analyse_sweep(sweep_frames))

    stats.update(
        {
            "variant": variant,
            "computed": computed,
            "runningComputed": running_computed,
            "checkedWhileRunning": still_running,
            "seededBlocks": blocks_before,
            "seededTokens": tokens,
            "seeded": seeded,
            "after": after,
            "sentChars": results["sentChars"],
            "sweepMeta": sweep_meta,
        }
    )
    return stats


def main() -> int:
    proc = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        proc = start_vite(PORT)
    failures: list[str] = []
    all_rows: list[dict] = []
    try:
        if OWNS_SERVER:
            wait_for_smoke_page(f"{BASE}/{PAGE}", ENTRY, proc = proc, info = info)
        with sync_playwright() as pw:
            for engine in ENGINES:
                info(f"engine {engine}")
                launcher = getattr(pw, engine)
                browser = (
                    launcher.launch(args = chromium_launch_args())
                    if engine == "chromium"
                    else launcher.launch()
                )
                for variant in VARIANTS:
                    for repetition in range(REPEATS):
                        page = browser.new_page(viewport = {"width": 1280, "height": 900})
                        try:
                            row = run_case(page, variant)
                        finally:
                            page.close()
                        row["engine"] = engine
                        row["repetition"] = repetition + 1
                        all_rows.append(row)
                        info(
                            f"  {engine} {variant} rep {repetition + 1}/{REPEATS}: "
                            f"collapses {row['collapses']} placeholder frames "
                            f"{row['placeholderFrames']} scrollHeight dips "
                            f"{row['scrollHeightDips']} anchor shift {row['anchorShiftPx']}px "
                            f"over {row['frames']} frames, {row['blocks']} blocks; "
                            f"sweep shift frames {row['shiftFrames']} worst "
                            f"{row['worstShiftPx']}px growth {row['scrollHeightGrowthPx']}px"
                        )
                browser.close()
    finally:
        if proc is not None:
            stop_process(proc)
            info("vite stopped")

    (OUT / f"{LABEL}.json").write_text(json.dumps(all_rows, indent = 2), encoding = "utf-8")

    info("")
    info(
        f"{'engine':10} {'variant':12} {'collapses':>10} {'placeholder':>12} {'dips':>6} "
        f"{'sweep shift':>12} {'worst px':>9} {'grew px':>9}"
    )
    for engine in ENGINES:
        for variant in VARIANTS:
            rows = [r for r in all_rows if r["engine"] == engine and r["variant"] == variant]
            if not rows:
                continue
            collapses = [r["collapses"] for r in rows]
            info(
                f"{engine:10} {variant:12} {statistics.median(collapses):>10.0f} "
                f"{statistics.median([r['placeholderFrames'] for r in rows]):>12.0f} "
                f"{statistics.median([r['scrollHeightDips'] for r in rows]):>6.0f} "
                f"{statistics.median([r['shiftFrames'] for r in rows]):>12.0f} "
                f"{statistics.median([r['worstShiftPx'] for r in rows]):>9.1f} "
                f"{statistics.median([r['scrollHeightGrowthPx'] for r in rows]):>9.0f}"
                f"   per-rep collapses {collapses}"
            )

            if variant in MUST_FLICKER and max(collapses) == 0:
                failures.append(
                    f"{engine}/{variant}: no collapse in any of {REPEATS} repetitions. This "
                    "variant is the state the override exists to prevent, so the fixture is not "
                    "reproducing the flicker and no other row here means anything."
                )
            if variant in MUST_NOT_FLICKER and max(collapses) > 0:
                failures.append(
                    f"{engine}/{variant}: {max(collapses)} collapse(s), worst drop "
                    f"{max(r['worstDropPx'] for r in rows)}px. A code block rendered at a "
                    "fraction of its height and then came back, which is the flicker."
                )
            for row in rows:
                if row["blocks"] < HISTORY_MESSAGES + FENCES:
                    failures.append(
                        f"{engine}/{variant}: only {row['blocks']} code blocks, expected at "
                        f"least {HISTORY_MESSAGES + FENCES}. The fixture did not build."
                    )

    info("")
    for row in all_rows[:4]:
        info(f"computed for block 0 under {row['variant']}: {row['computed']}")

    if failures:
        info("")
        for failure in failures:
            info(f"FAIL {failure}")
        return 1
    info("")
    info("every variant behaved as its contract says it must.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
