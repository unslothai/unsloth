# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measures main-thread cost of real MarkdownText streaming in Chromium.

Runs a fixed reply through the production renderer and records long-task and paint budgets.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5186"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL

# Use an external server when SMOKE_BASE_URL is set; otherwise own a local server.
# Keep artifacts under gitignored logs/.

OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-stream-pacing"))
LABEL = "stream-pacing"

# Length drives renderer cost; modest chunking avoids stretching the run without signal.

TOTAL_CHARS = int(os.environ.get("SMOKE_STREAM_CHARS", "24000"))
CHUNK_CHARS = int(os.environ.get("SMOKE_STREAM_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_STREAM_GAP_MS", "2"))
THROTTLE = int(os.environ.get("SMOKE_STREAM_THROTTLE", "6"))

# Separate budgets catch the regressions observed when #8750 and #7892 were reverted.
# Values vary across machines, so tune from repeated runner measurements.

MAX_LONGEST_STALL_MS = int(os.environ.get("SMOKE_STREAM_STALL_BUDGET_MS", "2500"))
MAX_LONG_TASK_MS = int(os.environ.get("SMOKE_STREAM_LONG_TASK_BUDGET_MS", "10000"))

MIN_MODERATE_HIGHLIGHT_FPS = 50


def info(msg: str) -> None:
    print(f"[{LABEL}] {msg}", flush = True)


def _exercise_open_fence(
    page,
    marker: str,
    *,
    source_code_units: int,
    expect_highlight: bool,
    name: str,
    code_highlighting: str = "syntax",
    followed_by_prose: bool = False,
    global_scoped: bool = True,
    expect_rich_prefix: bool = True,
    source_ends_with_line_ending: bool = True,
) -> dict:
    page.goto(f"{BASE}/smoke-stream-pacing.html", wait_until = "load", timeout = 60_000)
    page.wait_for_function("() => window.__stream && window.__stream.ready", timeout = 60_000)

    page.evaluate(
        """() => {

            window.__copiedCode = null;
            Object.defineProperty(navigator, 'clipboard', {
                configurable: true,
                value: {
                    writeText: async value => { window.__copiedCode = value; },
                },
            });
            const tracking = {
                armed: false,
                node: null,
                removals: 0,
                mutations: 0,
            };
            window.__mixedPrefixTracking = tracking;
            window.__mixedPrefixObserver = new MutationObserver(records => {
                const code = document.querySelector(
                    '[data-streamdown="code-block-body"] code'
                );
                if ((code?.textContent?.length ?? 0) >= 4096) tracking.armed = true;
                if (tracking.node && !tracking.node.isConnected) {
                    if (tracking.armed) tracking.removals += 1;
                    tracking.node = null;
                }
                tracking.node ??= [...document.querySelectorAll('h2')].find(
                    heading => heading.textContent === 'Mixed terminal-fence fixture'
                ) ?? null;
                if (!tracking.armed || !tracking.node) return;
                for (const record of records) {
                    if (tracking.node.contains(record.target)) tracking.mutations += 1;
                }
            });
            window.__mixedPrefixObserver.observe(document.body, {
                attributes: true,
                childList: true,
                characterData: true,
                subtree: true,
            });
        }"""
    )
    page.evaluate(
        "(options) => window.__stream.runOpenFence(options)",
        {
            "codeHighlighting": code_highlighting,
            "marker": marker,
            "followedByProse": followed_by_prose,
            "globalScoped": global_scoped,
            "richPrefix": expect_rich_prefix,
            "sourceCodeUnits": source_code_units,
            "sourceEndsWithLineEnding": source_ends_with_line_ending,
        },
    )
    page.wait_for_function("() => window.__stream.results().paused", timeout = 60_000)
    page.wait_for_function(
        """() => {
            const code = document.querySelector('[data-streamdown="code-block-body"] code');
            return code?.textContent === window.__stream.expectedOpenCode();
        }""",
        timeout = 60_000,
    )

    open_state = page.evaluate(
        """async () => {
            const body = document.querySelector('[data-streamdown="code-block-body"]');
            const code = body?.querySelector('code');
            const container = body?.closest('[data-streamdown="code-block"]');
            const source = code?.textContent ?? '';
            const expected = window.__stream.expectedOpenCode();
            const sha256 = async value => {
                const digest = await crypto.subtle.digest(
                    'SHA-256',
                    new TextEncoder().encode(value)
                );
                return [...new Uint8Array(digest)]
                    .map(byte => byte.toString(16).padStart(2, '0'))
                    .join('');
            };
            return {
                sourceLength: source.length,
                expectedLength: expected.length,
                sourceHash: await sha256(source),
                expectedHash: await sha256(expected),
                exact: source === expected,
                spans: code?.querySelectorAll('span').length ?? -1,
                incomplete: container?.getAttribute('data-incomplete') ?? null,
                chrome: Boolean(container && body && container.querySelector(
                    '[data-streamdown="code-block-header"]'
                )),
                controls: container?.parentElement?.querySelectorAll('button').length ?? 0,

                controlStates: container
                    ? [...(container.parentElement?.querySelectorAll('button') ?? [])]
                        .map(button => button.disabled)
                    : [],
                highlightCalls: window.__stream.results().codeHighlightCalls,
                prefix: {
                    heading: [...document.querySelectorAll('h2')].some(
                        value => value.textContent === 'Mixed terminal-fence fixture'
                    ),
                    table: Boolean(document.querySelector('table')),
                    list: Boolean(document.querySelector('ul')),
                    quote: Boolean(document.querySelector('blockquote')),
                    math: Boolean(document.querySelector('.katex')),
                    removals: window.__mixedPrefixTracking?.removals ?? -1,
                    mutations: window.__mixedPrefixTracking?.mutations ?? -1,
                },
            };
        }"""
    )
    if (
        not open_state["exact"]
        or open_state["sourceLength"] != open_state["expectedLength"]
        or open_state["sourceHash"] != open_state["expectedHash"]
    ):
        raise AssertionError(f"{marker} open fence changed source bytes: {open_state}")
    if open_state["spans"] != 0:
        raise AssertionError(
            f"{marker} open fence mounted {open_state['spans']} Shiki spans; expected 0"
        )
    if open_state["incomplete"] != "true" or not open_state["chrome"]:
        raise AssertionError(f"{marker} open fence lost Streamdown chrome: {open_state}")
    if open_state["controls"] < 2:
        raise AssertionError(f"{name} open fence lost code controls: {open_state}")
    if not all(open_state["controlStates"]):
        raise AssertionError(f"{name} open fence enabled code actions early: {open_state}")
    if open_state["highlightCalls"] != 0:
        raise AssertionError(f"{name} open fence called Shiki: {open_state}")

    prefix = open_state["prefix"]
    if expect_rich_prefix and not all(
        prefix[name] for name in ("heading", "table", "list", "quote", "math")
    ):
        raise AssertionError(f"{marker} mixed prefix lost rich constructs: {prefix}")
    if prefix["removals"] != 0 or prefix["mutations"] != 0:
        raise AssertionError(f"{marker} mixed prefix churned during code growth: {prefix}")

    if marker == "`":
        screenshot = OUT / f"mixed-open-{name}.png"
        screenshot.parent.mkdir(parents = True, exist_ok = True)
        page.screenshot(path = str(screenshot), full_page = True)
    page.evaluate("() => window.__mixedPrefixObserver?.disconnect()")

    page.evaluate(
        """(expectHighlight) => {
            window.__openFenceFrames = [];
            window.__openFenceFrameActive = true;
            window.__openFenceStopAfterFrame = false;
            let previousFrame = performance.now();
            const sampleFrame = now => {
                if (!window.__openFenceFrameActive) return;
                window.__openFenceFrames.push(now - previousFrame);
                previousFrame = now;
                if (window.__openFenceStopAfterFrame) {
                    window.__openFenceFrameActive = false;
                    return;
                }
                requestAnimationFrame(sampleFrame);
            };
            requestAnimationFrame(sampleFrame);
            window.__openFenceSamples = [];
            const sample = () => {
                const body = document.querySelector('[data-streamdown="code-block-body"]');
                const code = body?.querySelector('code');
                const status = document.querySelector('[data-status]')?.getAttribute('data-status');
                const spans = code?.querySelectorAll('span').length ?? -1;
                window.__openFenceSamples.push({
                    status,
                    source: code?.textContent ?? '',
                    spans,
                });
                if (
                    status === 'complete'
                    && (expectHighlight ? spans > 0 : spans === 0)
                ) {
                    window.__openFenceStopAfterFrame = true;
                }
            };
            sample();
            window.__openFenceObserver = new MutationObserver(sample);
            window.__openFenceObserver.observe(document.body, {
                attributes: true,
                childList: true,
                subtree: true,
            });
        }""",
        expect_highlight,
    )
    page.evaluate("() => window.__stream.completeOpenFence()")
    page.wait_for_function(
        """() => document.querySelector('[data-status]')?.getAttribute('data-status')
            === 'complete'""",
        timeout = 60_000,
    )
    if expect_highlight:
        page.wait_for_function(
            """() => document.querySelector(
                '[data-streamdown="code-block-body"] code'
            )?.querySelectorAll('span').length > 0""",
            timeout = 60_000,
        )
    else:
        # Leave enough throttled frames for any incorrectly scheduled warm-up or
        # highlighted mount to fire; the extreme policy must remain quiescent.
        page.wait_for_timeout(1_000)

    copy_button = page.locator('button[title="Copy code"]')
    if copy_button.count() != 1 or copy_button.is_disabled():
        button_states = copy_button.evaluate_all(
            "buttons => buttons.map(button => ({ disabled: button.disabled, html: button.outerHTML }))"
        )
        raise AssertionError(
            f"{name} completed fence did not enable one copy action: "
            f"buttons={button_states}, open={open_state}"
        )
    copy_button.click()
    page.wait_for_function("() => window.__copiedCode !== null", timeout = 10_000)
    final_state = page.evaluate(
        """async () => {
            document.documentElement.classList.toggle('dark');
            await new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)
            ));
            window.__openFenceObserver?.disconnect();
            window.__openFenceFrameActive = false;
            const code = document.querySelector(
                '[data-streamdown="code-block-body"] code'
            );
            const samples = window.__openFenceSamples ?? [];
            const expected = window.__stream.expectedOpenCode();

            const copied = window.__copiedCode ?? '';
            const source = code?.children.length
                // Streamdown represents an empty highlighted token row with a
                // literal newline placeholder. The join owns the row separator.
                ? [...code.children].map(line => {
                    const text = line.textContent ?? '';
                    return text === '\\n' ? '' : text;
                }).join('\\n')
                : code?.textContent ?? '';
            const sha256 = async value => {
                const digest = await crypto.subtle.digest(
                    'SHA-256',
                    new TextEncoder().encode(value)
                );
                return [...new Uint8Array(digest)]
                    .map(byte => byte.toString(16).padStart(2, '0'))
                    .join('');
            };
            let previous = 0;
            let highlightTransitions = 0;
            for (const sample of samples) {
                if (sample.spans > 0 && previous === 0) highlightTransitions += 1;
                previous = sample.spans > 0 ? 1 : 0;
            }

            const frameDurations = window.__openFenceFrames ?? [];
            const frameTime = frameDurations.reduce((sum, value) => sum + value, 0);
            const completionFps = frameTime > 0
                ? frameDurations.length * 1000 / frameTime
                : 0;
            return {
                sourceLength: source.length,
                expectedLength: expected.length,
                sourceHash: await sha256(source),
                expectedHash: await sha256(expected),

                copiedLength: copied.length,
                copiedHash: await sha256(copied),
                copiedExact: copied === expected,
                exact: source === expected,
                spans: code?.querySelectorAll('span').length ?? 0,

                sourceTail: source.slice(-24),
                expectedTail: expected.slice(-24),
                rows: code?.children.length ?? 0,
                finalRows: code
                    ? [...code.children].slice(-3).map(line => line.textContent ?? '')
                    : [],
                plainCompletedPaint: samples.some(
                    sample => sample.status === 'complete'
                        && sample.spans === 0
                        && sample.source === expected
                ),
                highlightTransitions,
                completionFps,
                maxFrameMs: Math.max(0, ...frameDurations),

                highlightCalls: window.__stream.results().codeHighlightCalls,

                renderPlans: window.__stream.results().renderPlans,
                actionStates: [...document.querySelectorAll(
                    'button[title="Copy code"], button[title="Download file"]'
                )].map(button => button.disabled),
            };
        }"""
    )
    if (
        not final_state["exact"]
        or final_state["sourceLength"] != final_state["expectedLength"]
        or final_state["sourceHash"] != final_state["expectedHash"]
        or not final_state["copiedExact"]
        or final_state["copiedLength"] != final_state["expectedLength"]
        or final_state["copiedHash"] != final_state["expectedHash"]
    ):
        raise AssertionError(f"{name} completed fence changed source/actions: {final_state}")
    if any(final_state["actionStates"]):
        raise AssertionError(f"{name} completed fence left actions disabled: {final_state}")
    if not final_state["plainCompletedPaint"]:
        raise AssertionError(f"{marker} completed source never painted plain: {final_state}")
    if expect_highlight:
        if final_state["spans"] <= 0 or final_state["highlightTransitions"] != 1:
            raise AssertionError(f"{name} did not highlight exactly once: {final_state}")
        if final_state["highlightCalls"] <= 0:
            raise AssertionError(f"{name} mounted spans without a code-plugin call: {final_state}")
    elif (
        final_state["spans"] != 0
        or final_state["highlightTransitions"] != 0
        or final_state["highlightCalls"] != 0
    ):
        raise AssertionError(f"{name} invoked or mounted Shiki: {final_state}")

    if marker == "`" and final_state["completionFps"] < MIN_MODERATE_HIGHLIGHT_FPS:
        raise AssertionError(f"{name} fell below the 50-FPS floor: {final_state}")
    if final_state["maxFrameMs"] >= 100:
        raise AssertionError(f"{name} completion blocked a frame: {final_state}")

    completed_screenshot = OUT / f"mixed-complete-{name}.png"
    completed_screenshot.parent.mkdir(parents = True, exist_ok = True)
    page.screenshot(path = str(completed_screenshot), full_page = True)

    return {
        "name": name,
        "marker": marker,
        "source_code_units": source_code_units,
        "auto_highlight": expect_highlight,
        "open": open_state,
        "completed": final_state,
    }


def run() -> dict:
    headless = os.environ.get("SMOKE_HEADFUL") != "1"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = headless, args = chromium_launch_args())
        context = browser.new_context(viewport = {"width": 1200, "height": 900})
        page = context.new_page()
        cdp = context.new_cdp_session(page)
        if THROTTLE > 1:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": THROTTLE})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            open_fences = [
                _exercise_open_fence(
                    page,
                    "`",
                    source_code_units = 7_000,
                    expect_highlight = True,
                    followed_by_prose = True,
                    expect_rich_prefix = False,
                    name = "moderate-answer",
                ),
                _exercise_open_fence(
                    page,
                    "`",
                    code_highlighting = "plain",
                    source_code_units = 7_000,
                    expect_highlight = False,
                    followed_by_prose = True,
                    name = "reasoning-plain",
                ),
                _exercise_open_fence(
                    page,
                    "`",
                    source_code_units = 16_385,
                    expect_highlight = False,
                    followed_by_prose = True,
                    global_scoped = False,
                    name = "oversized-boundary",
                ),
            ]

            page.goto(f"{BASE}/smoke-stream-pacing.html", wait_until = "load", timeout = 60_000)
            page.wait_for_function("() => window.__stream && window.__stream.ready", timeout = 60_000)
            page.evaluate(
                "(o) => window.__stream.run(o)",
                {"totalChars": TOTAL_CHARS, "chunkChars": CHUNK_CHARS, "gapMs": GAP_MS},
            )

            # Poll the harness's own verdict rather than deciding out here: every round
            # trip is slowed by the throttling, so an outside "has it finished" arrives
            # late enough to hide the effect.
            deadline = time.monotonic() + 300
            results: dict = {}
            while time.monotonic() < deadline:
                results = page.evaluate("() => window.__stream.results()")
                if results.get("done"):
                    break
                time.sleep(0.25)
            if not results.get("done"):
                raise RuntimeError(
                    f"the reply never finished painting within 300s: {json.dumps(results)}"
                )
        finally:
            context.close()
            browser.close()

    results["open_fences"] = open_fences

    results["page_errors"] = errors
    results["cpu_throttle"] = THROTTLE
    results["total_chars"] = TOTAL_CHARS
    results["chunk_chars"] = CHUNK_CHARS
    results["gap_ms"] = GAP_MS
    return results


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-stream-pacing.html",
            "smoke-stream-pacing-main.tsx",
            proc = vite,
            info = info,
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    out = OUT / f"{LABEL}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    # A page that painted nothing scores a perfect zero on every budget below, so assert
    # the workload first. Not an equality: Markdown syntax (fences, list markers, math
    # delimiters) never reaches textContent, so rendered length is a few per cent under the
    # bytes sent. 90% is above that and far below "the render died early".
    floor = int(TOTAL_CHARS * 0.9)
    if results["paintedChars"] < floor:
        failures.append(
            f"only {results['paintedChars']} characters painted of {TOTAL_CHARS} sent "
            f"(floor {floor}); the budgets below measured no workload"
        )
    # paintedChars is a high-water mark and survives a completion render that truncates the
    # bubble. settledChars is the DOM a reader is actually left looking at.
    if results["settledChars"] < floor:
        failures.append(
            f"the reply settled at {results['settledChars']} characters of {TOTAL_CHARS} "
            f"sent (floor {floor}); it peaked at {results['paintedChars']} and then lost "
            "content, so the final render is incomplete"
        )
    if results["arrivals"] < TOTAL_CHARS // CHUNK_CHARS:
        failures.append(
            f"only {results['arrivals']} arrivals for {TOTAL_CHARS} characters; "
            "the stream did not run at the rate this claims to measure"
        )
    if results["longestStallMs"] > MAX_LONGEST_STALL_MS:
        failures.append(
            f"longest stall {results['longestStallMs']:.0f}ms exceeds "
            f"{MAX_LONGEST_STALL_MS}ms (the bubble stopped growing while text arrived)"
        )
    # The long-task total is the sensitive metric and the one that goes false-green most
    # quietly: an engine without the longtask entry type, or an observer that stopped
    # delivering, reports 0ms and sails under the budget without raising. Same reasoning as
    # the painted-characters floor above.
    if not results.get("longTaskSupported"):
        failures.append(
            "this engine reports no longtask entries, so the long-task budget measured "
            "nothing; run under Chromium"
        )
    elif results["longTasks"] <= 0:
        failures.append(
            "no long tasks were observed at all; the observer measured nothing, so the "
            f"{MAX_LONG_TASK_MS}ms budget below would pass on any tree"
        )
    if results["cpu_throttle"] <= 1:
        failures.append(
            f"CPU throttling was {results['cpu_throttle']}x; unthrottled, the renderer keeps "
            "up with any rate this can feed and the budgets measure nothing"
        )
    if results["longTaskMs"] > MAX_LONG_TASK_MS:
        failures.append(
            f"long tasks totalled {results['longTaskMs']:.0f}ms, over the "
            f"{MAX_LONG_TASK_MS}ms budget (the main thread is saturated by the render)"
        )
    if results["page_errors"]:
        failures.append(f"page errors: {results['page_errors']}")

    if failures:
        for f in failures:
            info(f"FAIL: {f}")
        return 1
    info(
        f"OK: longest stall {results['longestStallMs']:.0f}ms, "
        f"long tasks {results['longTaskMs']:.0f}ms, "
        f"{results['framesOver33ms']} frames over 33ms, fully painted at "
        f"{results['timeToFullyPaintedMs']:.0f}ms, {results['settledChars']} chars, "
        f"{THROTTLE}x throttle"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
