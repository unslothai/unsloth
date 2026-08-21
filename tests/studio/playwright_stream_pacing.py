# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Main-thread cost of a fast streaming reply in the chat renderer.

Four merged PRs moved this path and each rebuilt a throwaway harness to prove it:

    #7892  Streamdown's transition starvation      9.86s -> 0.34s longest freeze
    #8750  incremental Markdown parsing            O(n) per update -> tail only
    #8845  publish coalescing                      4.01s -> 0.62s longest stall
    #8935  incremental fence tokenization          21x fewer characters to Shiki

None of them left anything behind that would notice the next regression, and each had to
rediscover the same methodology. This is that harness, kept.

It drives smoke-stream-pacing.html, which mounts the real MarkdownText inside a real
assistant-ui local runtime, so nothing is a mock of the code under test and assistant-ui's
own update scheduling is inside the measurement. Runs against a vite dev server; no
backend, no auth, no GPU, no model.

The reply is a fixed string. #8845's first measurement attempts failed because a real
model gave the two sides different essays and the renderer's cost is superlinear in
length, so a comparison across different text says nothing.

CPU throttling is not decoration: on a developer machine the renderer keeps up with any
rate this can feed, so an unthrottled run measures nothing on either side.

Chromium only, deliberately. Both things that make this a measurement are Chromium-only:
`Emulation.setCPUThrottlingRate` is reached over CDP, which Playwright exposes for Chromium
alone, and `longtask` PerformanceObserver entries exist in no other engine (Gecko bug
1348405 is open; WebKit has never shipped them). Neither fails loudly on firefox or webkit,
since `observe({type: "longtask"})` is specified to abort silently on an unsupported type
rather than throw, so the budgets would read a perfect zero instead of an error. The verdict
below therefore refuses a run that saw no long tasks, and the harness records whether the
engine supported them.

Run:
    python tests/studio/playwright_stream_pacing.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port it picks with SMOKE_PORT.
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
# Unset: start and stop our own server. Set: drive that one and leave it running.
# Exported-but-empty counts as unset, else we skip the server and drive "" as the URL.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
# Under logs/ like every sibling harness. A default of "." would drop an untracked
# stream-pacing.json in the repo root every run; logs/ is gitignored, so the tree stays clean.
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-stream-pacing"))
LABEL = "stream-pacing"

# The reply and the rate it arrives at. Length is what the renderer's cost is superlinear
# in, so it is the knob that matters; the arrival count stays modest because throttling
# slows the feed's own timers too, and 1,000 arrivals at 6x stretched a one-second stream
# to 43s of wall clock for no extra signal.
TOTAL_CHARS = int(os.environ.get("SMOKE_STREAM_CHARS", "24000"))
CHUNK_CHARS = int(os.environ.get("SMOKE_STREAM_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_STREAM_GAP_MS", "2"))
THROTTLE = int(os.environ.get("SMOKE_STREAM_THROTTLE", "6"))

# Budgets, not targets, chosen against real regressions rather than by feel. Two merged
# fixes were reverted in this harness, on two machines, and they move the two numbers in
# opposite directions, which is why there are two budgets and not one headline metric.
#
#                        main   #8750 reverted   #7892 reverted
#   long tasks      5.0-8.0s      13.1s/74.4s        6.7-7.9s
#   long task count    49-71          144/598           14-23
#   longest stall   1.05-1.23s     0.97-1.40s        5.03-6.35s
#
# Reverting #8750 (incremental Markdown parsing) blows up the long-task total and leaves
# the longest stall alone. Reverting #7892 (Streamdown's `animated` config, which keeps
# block updates out of an interruptible transition) does the opposite: the stall goes 4-5x
# while the long-task total stays in the clean range. A single headline metric would have
# missed one of the two outright.
#
# Machine spread: the "main" column above is 5,029/5,901ms on one machine and 6,687-8,003ms
# on another, so clean readings vary ~60% ACROSS boxes even though a single box repeats to
# within ~15%. That is what keeps the CI step non-gating. The 10,000ms budget is NOT raised
# for that headroom, because the same two machines read the #8750 revert as 13,059ms and
# 74,353ms: a budget loose enough for the slower box would stop catching that regression on
# the faster one. Retune from observed runner numbers, not from one machine.
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
        raise AssertionError(
            f"{marker} open fence changed source bytes: {open_state}"
        )
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

    if (
        marker == "`"
        and final_state["completionFps"] < MIN_MODERATE_HIGHLIGHT_FPS
    ):
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


def _exercise_duplicate_fences(page) -> dict:
    page.goto(f"{BASE}/smoke-stream-pacing.html", wait_until = "load", timeout = 60_000)
    page.wait_for_function("() => window.__stream && window.__stream.ready", timeout = 60_000)
    page.evaluate(
        """() => {
            window.__copiedCode = null;
            window.__capturedDownloads = [];
            const blobs = new Map();
            let nextBlob = 0;
            Object.defineProperty(navigator, 'clipboard', {
                configurable: true,
                value: {
                    writeText: async value => { window.__copiedCode = value; },
                },
            });
            URL.createObjectURL = blob => {
                const url = `https://fixture.invalid/blob/${nextBlob++}`;
                blobs.set(url, blob);
                return url;
            };
            URL.revokeObjectURL = () => {};
            HTMLAnchorElement.prototype.click = function() {
                const blob = blobs.get(this.href);
                if (!blob) throw new Error(`uncaptured download URL: ${this.href}`);
                void blob.text().then(text => {
                    window.__capturedDownloads.push({ filename: this.download, text });
                });
            };
        }"""
    )
    page.evaluate("() => window.__stream.runDuplicateFences()")
    page.wait_for_function("() => window.__stream.results().paused", timeout = 60_000)
    page.wait_for_function(
        """() => {
            const bodies = [...document.querySelectorAll(
                '[data-streamdown="code-block-body"]'
            )];
            const expected = window.__stream.expectedCodeSources();
            return bodies.length === 2
                && bodies.every((body, index) => body.textContent === expected[index]);
        }""",
        timeout = 60_000,
    )
    open_state = page.evaluate(
        """() => {
            const bodies = [...document.querySelectorAll(
                '[data-streamdown="code-block-body"]'
            )];
            const actionStates = bodies.map(body => {
                const container = body.closest('[data-streamdown="code-block"]');
                return [...(container?.parentElement?.querySelectorAll('button') ?? [])]
                    .map(button => button.disabled);
            });
            return {
                actionStates,
                highlightCalls: window.__stream.results().codeHighlightCalls,
                labels: bodies.map(body => body.getAttribute('data-language')),
                sources: bodies.map(body => body.textContent ?? ''),
                spans: bodies.map(body => body.querySelectorAll('span').length),
            };
        }"""
    )
    if open_state["labels"] != ["typescript", "javascript"]:
        raise AssertionError(f"duplicate open fences crossed labels: {open_state}")
    if open_state["spans"] != [0, 0] or open_state["highlightCalls"] != 0:
        raise AssertionError(f"duplicate open fences invoked Shiki: {open_state}")
    if open_state["actionStates"] != [[False, False], [True, True]]:
        raise AssertionError(f"duplicate open fence action ownership changed: {open_state}")

    page.evaluate("() => window.__stream.completeOpenFence()")
    page.wait_for_function(
        """() => document.querySelector('[data-status]')?.getAttribute('data-status')
            === 'complete'""",
        timeout = 60_000,
    )
    page.wait_for_function(
        """() => {
            const bodies = [...document.querySelectorAll(
                '[data-streamdown="code-block-body"]'
            )];
            const expected = window.__stream.expectedCodeSources();
            return bodies.length === 2
                && bodies.every((body, index) => body.textContent === expected[index]);
        }""",
        timeout = 60_000,
    )
    page.evaluate(
        """async () => {
            const bodies = [...document.querySelectorAll(
                '[data-streamdown="code-block-body"]'
            )];
            window.__copiedCodes = [];
            for (const body of bodies) {
                const container = body.closest('[data-streamdown="code-block"]');
                const wrapper = container?.parentElement;
                window.__copiedCode = null;
                wrapper?.querySelector('button[title="Copy code"]')?.click();
                await new Promise(resolve => setTimeout(resolve, 0));
                window.__copiedCodes.push(window.__copiedCode);
                wrapper?.querySelector('button[title="Download file"]')?.click();
            }
        }"""
    )
    page.wait_for_function(
        "() => window.__capturedDownloads?.length === 2",
        timeout = 10_000,
    )
    final_state = page.evaluate(
        """async () => {
            document.documentElement.classList.add('dark');
            await new Promise(resolve => requestAnimationFrame(
                () => requestAnimationFrame(resolve)
            ));
            const bodies = [...document.querySelectorAll(
                '[data-streamdown="code-block-body"]'
            )];
            const expected = window.__stream.expectedCodeSources();
            const sha256 = async value => {
                const digest = await crypto.subtle.digest(
                    'SHA-256',
                    new TextEncoder().encode(value)
                );
                return [...new Uint8Array(digest)]
                    .map(byte => byte.toString(16).padStart(2, '0'))
                    .join('');
            };
            const containers = bodies.map(body => body.closest(
                '[data-streamdown="code-block"]'
            ));
            return {
                actionStates: containers.map(container =>
                    [...(container?.parentElement?.querySelectorAll('button') ?? [])]
                        .map(button => button.disabled)
                ),
                copiedExact: window.__copiedCodes.map(
                    (source, index) => source === expected[index]
                ),
                copiedHashes: await Promise.all(window.__copiedCodes.map(sha256)),
                downloadExact: window.__capturedDownloads.map(
                    (download, index) => download.text === expected[index]
                ),
                downloadFilenames: window.__capturedDownloads.map(
                    download => download.filename
                ),
                downloadHashes: await Promise.all(
                    window.__capturedDownloads.map(download => sha256(download.text))
                ),
                expectedHashes: await Promise.all(expected.map(sha256)),
                highlightCalls: window.__stream.results().codeHighlightCalls,
                labels: bodies.map(body => body.getAttribute('data-language')),
                metadataLeaked: document.body.textContent.includes('title="first.ts"')
                    || document.body.textContent.includes('title="second.js"')
                    || document.body.textContent.includes('unsloth-fence:'),
                prefixPresent: [...document.querySelectorAll('h2')].some(
                    heading => heading.textContent === 'Mixed terminal-fence fixture'
                ),
                sourceExact: bodies.map(
                    (body, index) => body.textContent === expected[index]
                ),
                sourceHashes: await Promise.all(
                    bodies.map(body => sha256(body.textContent ?? ''))
                ),
                sourceLengths: bodies.map(body => (body.textContent ?? '').length),
                spans: bodies.map(body => body.querySelectorAll('span').length),
            };
        }"""
    )
    if (
        final_state["labels"] != ["typescript", "javascript"]
        or final_state["sourceExact"] != [True, True]
        or final_state["copiedExact"] != [True, True]
        or final_state["downloadExact"] != [True, True]
        or final_state["sourceHashes"] != final_state["expectedHashes"]
        or final_state["copiedHashes"] != final_state["expectedHashes"]
        or final_state["downloadHashes"] != final_state["expectedHashes"]
        or final_state["downloadFilenames"] != ["snippet.ts", "snippet.js"]
        or final_state["metadataLeaked"]
        or not final_state["prefixPresent"]
    ):
        raise AssertionError(f"duplicate fences crossed presentation/actions: {final_state}")
    if final_state["actionStates"] != [[False, False], [False, False]]:
        raise AssertionError(f"duplicate completed actions remained disabled: {final_state}")
    if final_state["spans"] != [0, 0] or final_state["highlightCalls"] != 0:
        raise AssertionError(f"duplicate extreme fences invoked Shiki: {final_state}")

    screenshot = OUT / "mixed-complete-duplicate-identities.png"
    screenshot.parent.mkdir(parents = True, exist_ok = True)
    page.screenshot(path = str(screenshot), full_page = True)
    return {"name": "duplicate-identities", "open": open_state, "completed": final_state}



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
                    name = "moderate-backtick-followed-by-prose",
                ),
                _exercise_open_fence(
                    page,
                    "`",
                    code_highlighting = "plain",
                    source_code_units = 7_000,
                    expect_highlight = False,
                    followed_by_prose = True,
                    name = "reasoning-plain-backtick-followed-by-prose",
                ),
                _exercise_open_fence(
                    page,
                    "~",
                    source_code_units = 7_000,
                    expect_highlight = True,
                    name = "moderate-tilde-no-final-newline",
                    source_ends_with_line_ending = False,
                ),
                _exercise_open_fence(
                    page,
                    "`",
                    source_code_units = 16_385,
                    expect_highlight = False,
                    followed_by_prose = True,

                    global_scoped = False,
                    name = "boundary-lf-followed-by-prose",
                ),
                _exercise_open_fence(
                    page,
                    "~",
                    source_code_units = 16_385,
                    expect_highlight = False,
                    followed_by_prose = True,

                    global_scoped = False,
                    name = "boundary-crlf-followed-by-prose",
                ),
            ]
            duplicate_fences = _exercise_duplicate_fences(page)

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

    results["duplicate_fences"] = duplicate_fences
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
