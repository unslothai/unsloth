# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does a code block flicker through a placeholder height when a stream finalizes?

Streamdown puts `content-visibility: auto` with `contain-intrinsic-size: auto 200px` inline on
every code-block wrapper. An element carrying those has no LAST REMEMBERED SIZE until it has been
rendered once, so until then it lays out at the 200px fallback rather than at its content height.
The re-render at the end of a stream REPLACES the code-block node, and a replaced node is a new
element with no last remembered size, so it can lay out at 200px for a frame and then snap to its
real height. That one-frame change is the "reload"-style flicker studio/frontend/src/index.css
describes, and it is why the override there exists.

WHAT IS MEASURED

Not a timing. Per frame, for every `[data-streamdown="code-block"]` in the thread:

    collapses          a block that was at least 400px tall rendered at half that or less on a
                       later frame and then came back. This is the flicker, and one is too many.
    placeholder frames how many of those frames landed in the 150-260px band, i.e. on the
                       `contain-intrinsic-size` fallback specifically rather than on some other
                       height. Reported so a collapse can be attributed rather than just counted.
    scroll height dips the thread's own scrollHeight falling by 300px or more and recovering.
                       A block collapsing takes the whole column with it.
    anchor shift       document-space movement of the last message that existed BEFORE the
                       stream started. Nothing about that message changes while the stream runs,
                       so any movement is settled content being relaid out under the user.

Then a second phase on the thread the stream left behind: scroll it from the bottom to the top
and record whether the content moved.

    sweep shift        frames in which some block's DOCUMENT-space top moved. A block that was
                       never rendered is skipped at the 200px fallback rather than at its real
                       height, and that does not show while the user sits at the bottom. It
                       shows on the way back up, as every block expands when it is reached and
                       pushes what is below it down.
    scrollHeight grew  how much the thread's own height changed over the gesture. Zero means the
                       thread was already the right height before the sweep began.

VARIANTS, AND WHY THE RUN DRIVES MORE THAN ONE

A flicker check that only ever runs against the tree cannot tell "no flicker" from "the fixture
reproduces nothing". So the same fixture is driven under stylesheets appended after the tree's
own, and the run asserts on the SHAPE of the whole set:

    streamdown   streamdown's inline defaults, i.e. the tree before any override existed. This
                 MUST flicker. If it does not, the fixture is measuring nothing and the run
                 fails, whatever the tree scored.
    released     the override released for every block at all times, streaming included: the
                 shape of the mistake that scoping the override can make. This must flicker too.
    legacy       the override as it was first written, forcing `content-visibility: visible` and
                 clobbering `contain-intrinsic-size` to `none`. Must not flicker.
    tree         whatever src/index.css ships right now. Must not flicker.
    statusonly   held only while the message part is running, released the instant it is not.
                 The obvious CSS-only scoping, and it is measured rather than assumed: the node
                 replacement at fence close can land in the same commit as the status flip, so
                 this one still flickers, which is why the tree keeps a settle window.
    lastmessage  held only for the last message in the thread. Survives finalization, and cannot
                 give an earlier message's blocks the first render they need, so it pays for the
                 stream phase in the sweep phase instead.

Neither of the last two decides the exit code. They are in the table because "the simpler thing
does not work" is a claim that needs a number next to it.

Run:
    python tests/studio/playwright_code_block_flicker.py
    SMOKE_FLICKER_ENGINES=chromium,webkit python tests/studio/playwright_code_block_flicker.py
    SMOKE_FLICKER_VARIANTS=tree,streamdown python tests/studio/playwright_code_block_flicker.py

It starts and stops its own vite dev server; SMOKE_BASE_URL points it at one you already have and
SMOKE_PORT moves the one it starts. Exits non-zero when a variant that must not flicker did, when
a variant that must flicker did not, or when the fixture failed to build the thread it claims to.
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

ENGINES = [e.strip() for e in os.environ.get("SMOKE_FLICKER_ENGINES", "chromium").split(",") if e.strip()]
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
# How long to keep sampling after the stream reports done. The flicker is AT finalization, and
# the re-render that causes it lands a frame or two after the generator returns, so a sampler
# that stops on `done` stops just before the thing it is looking for.
TAIL_MS = int(os.environ.get("SMOKE_FLICKER_TAIL_MS", "2500"))


MUST_FLICKER = {"streamdown", "released"}
MUST_NOT_FLICKER = {"tree", "legacy"}

# What each variant's stylesheet must have actually computed to on a settled block, checked
# before anything is measured.
#
# This guard is here because the absence of it produced a clean-looking run that proved nothing.
# The tree's override lives inside `@layer utilities`, and for IMPORTANT declarations the cascade
# REVERSES layer order, so a variant appended as unlayered `!important` CSS silently loses to the
# tree's rule. All four variants then computed `visible`/`none`, all four reported zero collapses,
# and the run read as "nothing flickers anywhere" while having measured one stylesheet four times.
EXPECTED_COMPUTED = {
    "streamdown": {"contentVisibility": "auto"},
    "released": {"contentVisibility": "auto"},
    "legacy": {"contentVisibility": "visible", "containIntrinsicSize": "none"},
    "statusonly": {"contentVisibility": "auto"},
    "lastmessage": {"contentVisibility": "auto"},
}

# The same guard, taken WHILE THE STREAM IS RUNNING, on the block being streamed.
#
# The settled check above is not enough on its own and that is not hypothetical: the pre-override
# variant passed it while losing the cascade during streaming, because the tree only holds while
# the thread is building and both agree once it is quiet. The variant then reported zero flickers
# and the run read as "there was never anything to fix".
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

    Two adjacent reads land inside the lull between two async Shiki batches, which releases the
    gate on a thread that is still building itself.
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
    # Mid-stream, on the block being written: the cascade guard that the settled one cannot make.
    page.wait_for_function(
        "window.__flicker.results().streamStartedAt !== null", timeout = 120_000
    )
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

    # Phase two, on the thread the stream just left behind: scroll it from bottom to top and
    # record whether the content moved.
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
