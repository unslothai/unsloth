# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Code-block flicker when a stream finalizes, from the 200px contain-intrinsic-size fallback."""

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
# The flicker re-render lands a frame or two after the generator returns, so keep sampling past done.
TAIL_MS = int(os.environ.get("SMOKE_FLICKER_TAIL_MS", "2500"))


MUST_FLICKER = {"streamdown", "released"}
MUST_NOT_FLICKER = {"tree", "legacy"}

# Without a variant required to flicker, a clean run is consistent with a detector that measured nothing.
if not MUST_FLICKER & set(VARIANTS):
    raise SystemExit(
        "SMOKE_FLICKER_VARIANTS="
        + ",".join(VARIANTS)
        + " has no positive control. Include at least one of "
        + ", ".join(sorted(MUST_FLICKER))
        + ", or a run that reports no collapses proves only that nothing was measured."
    )

# Check computed styles first: !important reverses layer order, so an unlayered variant can silently lose.
EXPECTED_COMPUTED = {
    "streamdown": {"contentVisibility": "auto"},
    "released": {"contentVisibility": "auto"},
    "legacy": {"contentVisibility": "visible", "containIntrinsicSize": "none"},
    "statusonly": {"contentVisibility": "auto"},
    "lastmessage": {"contentVisibility": "auto"},
}

# The settled check alone is not enough: a variant can lose the cascade only while streaming.
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
    """Five stable reads: two adjacent reads can fall in a lull between async Shiki batches."""
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
