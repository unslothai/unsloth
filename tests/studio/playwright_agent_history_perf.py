# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Compare composer typing cost in normal vs opt-in virtualized chat at 2k/4k messages.

Launch ONLY against a disposable Studio instance. This script creates synthetic
threads and deletes them afterward. It does not send prompts to an LLM.

This is an observational Chromium benchmark, NOT a CPU instruction profiler:
CDP TaskDuration / ScriptDuration / LayoutDuration / RecalcStyleDuration are
browser-main-thread busy-time proxies, not exact per-key CPU cycles. Double-rAF
latency is an event-to-frame proxy, not INP or guaranteed paint completion.
Report both modes; do not assert an unmeasured speedup or hardcode a ratio.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Support execution as a standalone CI driver from the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.studio.playwright_agent_history_scroll import create_fixture
from tests.studio.studiobench.runtime.lifecycle import (
    authenticate,
    auth_request_json,
    seed_init_script,
)

METRICS = ("TaskDuration", "ScriptDuration", "LayoutDuration", "RecalcStyleDuration")
BATCHES = 3
KEYS_PER_BATCH = 80
KEY_DELAY_MS = 20


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo), 3)


def snapshot(cdp) -> dict[str, float]:
    return {
        item["name"]: float(item["value"]) for item in cdp.send("Performance.getMetrics")["metrics"]
    }


def measure_mode(browser, base: str, auth, thread_id: str, count: int, opt_in: bool) -> dict:
    storage = (
        {
            "unsloth_agent_chat_history_v1": {
                "state": {"enabledThreads": {thread_id: True}},
                "version": 0,
            }
        }
        if opt_in
        else {}
    )
    context = browser.new_context(viewport = {"width": 1440, "height": 960})
    context.add_init_script(seed_init_script(auth, [], storage))
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    cdp = context.new_cdp_session(page)
    cdp.send("Performance.enable")
    try:
        started = time.perf_counter()
        page.goto(
            f"{base}/chat?thread={thread_id}",
            wait_until = "domcontentloaded",
            timeout = 120_000,
        )
        editor = page.locator(".aui-composer-input[aria-label='Message input']")
        editor.wait_for(state = "visible", timeout = 120_000)
        # Both modes must load the actual *last* persisted message, not just an
        # empty composer or the initial 32 rows of a partially hydrated thread.
        last_text = f"agent-history-e2e-row-{count - 1:05d}"
        page.wait_for_function(
            "(marker) => document.body?.textContent?.includes(marker) ?? false",
            arg = last_text,
            timeout = 120_000,
        )
        if opt_in:
            page.locator('[data-agent-history-scroll-list="true"]').wait_for(
                state = "attached", timeout = 30_000
            )
        loaded_ms = round((time.perf_counter() - started) * 1000, 3)
        # The upstream renderer progressively mounts 32 more messages per
        # animation frame. Seeing the last message does NOT prove that mounting
        # has finished. Wait for every expected role row, then require eight
        # consecutive animation frames with unchanged DOM size and scroll height.
        # A timeout is a failed measurement, never a partial success.
        page.wait_for_function(
            """expected => {
                const viewport = document.querySelector('.aui-thread-viewport');
                const roleRows = viewport?.querySelectorAll('[data-role]').length ?? 0;
                const virtual = document.querySelector('[data-agent-history-scroll-list="true"]');
                const complete = virtual
                    ? roleRows > 0 && roleRows <= 96
                    : roleRows === expected;
                const state = window.__agentPerfSettling ??= {
                    frames: 0, count: -1, height: -1, lastFrame: -1,
                };
                const frame = Math.floor(performance.now() / 16);
                if (frame !== state.lastFrame) {
                    const height = viewport?.scrollHeight ?? 0;
                    state.frames = complete && state.count === roleRows && state.height === height
                        ? state.frames + 1 : 0;
                    state.count = roleRows;
                    state.height = height;
                    state.lastFrame = frame;
                }
                return complete && state.frames >= 8;
            }""",
            arg = count,
            polling = "raf",
            timeout = 120_000,
        )
        dom = page.evaluate(
            """() => {
                const viewport = document.querySelector('.aui-thread-viewport');
                const rows = document.querySelectorAll('[data-agent-history-row]');
                return {
                    viewportNodes: viewport ? viewport.querySelectorAll('*').length : null,
                    virtualRows: rows.length,
                    messageElements: document.querySelectorAll('.aui-thread-message').length,
                    renderedRoleRows: viewport?.querySelectorAll('[data-role]').length ?? null,
                    settledFrames: window.__agentPerfSettling?.frames ?? null,
                };
            }"""
        )
        if opt_in and not (1 <= dom["virtualRows"] <= 96):
            raise AssertionError(f"opt-in DOM mounting is not bounded: {dom}")
        if not opt_in and dom["virtualRows"] != 0:
            raise AssertionError(f"default mode unexpectedly enabled virtual rows: {dom}")
        if not opt_in and dom["renderedRoleRows"] != count:
            raise AssertionError(f"default renderer not fully mounted before typing: {dom}")

        # The observer only samples keydowns targeted at the actual composer.
        # No network/inference is involved, and only printable characters count.
        page.evaluate(
            """() => {
                window.__agentPerf = {samples: [], total: 0};
                document.addEventListener('keydown', (event) => {
                    if (!(event.target instanceof Element) ||
                        !event.target.closest('.aui-composer-input') ||
                        event.key.length !== 1 || event.ctrlKey || event.metaKey) return;
                    const started = performance.now();
                    window.__agentPerf.total += 1;
                    requestAnimationFrame(() => requestAnimationFrame(() => {
                        window.__agentPerf.samples.push(performance.now() - started);
                    }));
                }, true);
            }"""
        )
        editor.click()
        samples = []
        for batch in range(BATCHES):
            # Reset outside the timed segment; deleting previous text is not
            # confused with printable keystrokes.
            editor.press("ControlOrMeta+A")
            editor.press("Backspace")
            page.evaluate(
                "() => { window.__agentPerf.samples = []; window.__agentPerf.total = 0; }"
            )
            payload = ("v" if batch % 2 == 0 else "w") * KEYS_PER_BATCH
            before = snapshot(cdp)
            batch_start = time.perf_counter()
            page.keyboard.type(payload, delay = KEY_DELAY_MS)
            page.wait_for_function(
                "(n) => window.__agentPerf.samples.length >= n",
                arg = KEYS_PER_BATCH,
                timeout = 60_000,
            )
            elapsed_ms = (time.perf_counter() - batch_start) * 1000
            after = snapshot(cdp)
            response = page.evaluate("() => window.__agentPerf")
            if response["total"] != KEYS_PER_BATCH:
                raise AssertionError(
                    f"Expected {KEYS_PER_BATCH} actual printable keydowns: {response['total']}"
                )
            busy = {
                key + "_ms_per_key": round(
                    max(0.0, after.get(key, 0.0) - before.get(key, 0.0)) * 1000 / KEYS_PER_BATCH,
                    4,
                )
                for key in METRICS
            }
            samples.append(
                {
                    "keys": KEYS_PER_BATCH,
                    "wall_ms_per_key": round(elapsed_ms / KEYS_PER_BATCH, 3),
                    "event_to_two_frames_p50_ms": percentile(response["samples"], 0.5),
                    "event_to_two_frames_p95_ms": percentile(response["samples"], 0.95),
                    **busy,
                }
            )
        if page_errors:
            raise AssertionError(f"Uncaught browser exceptions: {page_errors[:5]}")
        medians = {
            key: round(statistics.median(float(s[key]) for s in samples), 4)
            for key in (
                "wall_ms_per_key",
                "event_to_two_frames_p50_ms",
                "event_to_two_frames_p95_ms",
                *(metric + "_ms_per_key" for metric in METRICS),
            )
        }
        return {
            "messages": count,
            "mode": "virtual_opt_in" if opt_in else "default",
            "hydration_to_visible_tail_ms": loaded_ms,
            "dom": dom,
            "batches": samples,
            "medians": medians,
            "total_key_events": BATCHES * KEYS_PER_BATCH,
        }
    finally:
        context.close()


def run(base: str, username: str, password: str, output: Path) -> None:
    from playwright.sync_api import sync_playwright

    auth = authenticate(base, username, password)
    cases = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless = True)
            try:
                for count in (2000, 4000):
                    thread_id = create_fixture(base, auth, count)
                    original = auth_request_json(
                        auth, f"{base}/api/chat/threads/{thread_id}/messages"
                    )
                    try:
                        # Alternate mode order to reduce warm-cache/order bias.
                        modes = (False, True) if count == 2000 else (True, False)
                        for opt_in in modes:
                            record = measure_mode(browser, base, auth, thread_id, count, opt_in)
                            cases.append(record)
                            print(json.dumps(record, sort_keys = True), flush = True)
                        stored = auth_request_json(
                            auth, f"{base}/api/chat/threads/{thread_id}/messages"
                        )
                        if original["messages"] != stored["messages"]:
                            raise AssertionError(
                                f"Stored messages changed during {count}-message typing benchmark"
                            )
                    finally:
                        auth_request_json(
                            auth,
                            f"{base}/api/chat/threads",
                            method = "DELETE",
                            body = {"ids": [thread_id]},
                        )
            finally:
                browser.close()
    finally:
        output.parent.mkdir(parents = True, exist_ok = True)
        output.write_text(
            json.dumps(
                {
                    "status": "PASS" if len(cases) == 4 else "INCOMPLETE",
                    "methodology": {
                        "browser": "Chromium/Playwright/CDP",
                        "counts": [2000, 4000],
                        "modes": ["default", "virtual_opt_in"],
                        "sample_batches": BATCHES,
                        "printable_keys_per_batch": KEYS_PER_BATCH,
                        "key_delay_ms": KEY_DELAY_MS,
                        "metrics_are_browser_main_thread_busy_time_proxies": True,
                        "inference_is_not_measured": True,
                        "relative_speedup_is_not_asserted": True,
                        "fully_mounted_default_required": True,
                        "stable_animation_frames_required": 8,
                    },
                    "cases": cases,
                },
                sort_keys = True,
                indent = 2,
            )
            + "\n",
            encoding = "utf-8",
        )
    if len(cases) != 4:
        raise AssertionError(f"Expected four complete benchmark conditions, got {len(cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--url", required = True)
    parser.add_argument("--username", default = "unsloth")
    parser.add_argument("--password", required = True)
    parser.add_argument("--output", default = "logs/agent-history/perf.json")
    args = parser.parse_args()
    run(args.url.rstrip("/"), args.username, args.password, Path(args.output))


if __name__ == "__main__":
    main()
