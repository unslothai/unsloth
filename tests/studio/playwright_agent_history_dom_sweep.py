# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fast, focused 4,000-message / >=100,000 live-DOM agent-history stress probe.

Run only against an ISOLATED, disposable Studio installation:
    python tests/studio/playwright_agent_history_dom_sweep.py \
        --url http://127.0.0.1:18899 --password "$AGENT_HISTORY_PW" \
        --output logs/agent-history/dom-sweep.json

One real 4,000-message persisted fixture; actual Studio/assistant-ui virtualizer,
header menu and composer. DOM-heavy repeated tool-like markup is attached ONLY
to the currently mounted test rows in the disposable browser, never persisted
to the chat data or passed to the LLM. This is a synthetic style/DOM stressor,
NOT a reproduction of the real user's tool-call components or a loaded model.

Hard assertions cover DOM size/bounding, preference switching/persistence,
message count and browser errors. Input latency, CDP main-thread metrics and
window-size scaling are OBSERVATIONS, not runner-dependent millisecond gates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.studio.playwright_agent_history_scroll import create_fixture
from tests.studio.playwright_agent_history_perf import percentile, snapshot
from tests.studio.studiobench.runtime.lifecycle import (
    authenticate,
    auth_request_json,
    seed_init_script,
)

LIST = '[data-agent-history-scroll-list="true"]'
ROW = "[data-agent-history-row]"
EDITOR = ".aui-composer-input[aria-label='Message input']"
SIZES = (32, 5, 2, 8, 16, 32)
METRICS = ("TaskDuration", "ScriptDuration", "LayoutDuration", "RecalcStyleDuration")
KEYS = 16
CARDS_PER_ROW = 700

# No user data. Five descendants/card: wrapper, button, svg, path, span.
# The DOM is visually similar to repeated collapsed tool-call controls.
INJECT = """({cards}) => {
  const rows = Array.from(document.querySelectorAll('[data-agent-history-row]'));
  const template = document.createElement('template');
  template.innerHTML = '<div class="agent-perf-tool-card">' +
    '<button type="button" class="group/tool-group text-muted-foreground">' +
    '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">' +
    '<path d="M1 6h10"/></svg><span>Used tool</span></button></div>';
  const item = template.innerHTML;
  // Always reset synthetic material after a menu switch. React owns the actual
  // messages; this diagnostic intentionally owns only its marked children.
  for (const row of rows) {
    row.querySelectorAll('[data-agent-perf-fixture]').forEach(node => node.remove());
    const wrapper = document.createElement('div');
    wrapper.dataset.agentPerfFixture = 'true';
    wrapper.style.cssText = 'contain:layout;max-width:100%;overflow:hidden';
    wrapper.innerHTML = item.repeat(cards);
    // Within the message root so the row ResizeObserver sees a real height.
    const owner = row.querySelector('[data-role]');
    if (!owner) throw Error('Virtual row has no [data-role] child');
    owner.appendChild(wrapper);
  }
  return {
    rows: rows.length,
    descendants: document.querySelector('[data-agent-history-scroll-list="true"]')?.querySelectorAll('*').length ?? 0,
    syntheticCards: document.querySelectorAll('[data-agent-perf-fixture] .agent-perf-tool-card').length,
    totalDom: document.querySelectorAll('*').length,
  };
}"""

INSTALL_INPUT_PROBE = """() => {
  window.__agentDomProbe = {samples: [], received: 0};
  document.addEventListener('keydown', (event) => {
    if (!(event.target instanceof Element) ||
        !event.target.closest('.aui-composer-input') ||
        event.key.length !== 1 || event.metaKey || event.ctrlKey) return;
    const begin = performance.now();
    window.__agentDomProbe.received += 1;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      window.__agentDomProbe.samples.push(performance.now() - begin);
    }));
  }, true);
}"""


def select_rows(page, rows: int) -> None:
    """Load a legacy numeric preference through persisted production hydration.

    UI has only three modes; old row windows remain supported for backwards
    compatibility and are exercised without resurrecting duplicate controls.
    """
    if page.evaluate("() => document.querySelectorAll('[data-agent-history-row]').length") != rows:
        page.evaluate(
            """rows => {
              const key = 'unsloth_agent_chat_history_v1';
              const stored = JSON.parse(localStorage.getItem(key) || '{}');
              const state = stored.state || {};
              const threadId = new URL(location.href).searchParams.get('thread');
              if (!threadId) throw Error('No synthetic fixture thread');
              state.enabledThreads = { ...(state.enabledThreads || {}), [threadId]: true };
              state.domLevelByThreadId = { ...(state.domLevelByThreadId || {}) };
              delete state.domLevelByThreadId[threadId];
              state.windowRowsByThreadId = {
                ...(state.windowRowsByThreadId || {}), [threadId]: rows
              };
              localStorage.setItem(key, JSON.stringify({ ...stored, state }));
            }""",
            rows,
        )
        page.reload(wait_until = "domcontentloaded", timeout = 120_000)
    page.wait_for_function(
        """rows => document.querySelectorAll('[data-agent-history-row]').length === rows""",
        arg = rows,
        timeout = 30_000,
    )
    if page.locator(LIST).get_attribute("data-agent-history-dom-level") != "legacy":
        raise AssertionError("Numeric compatibility profile did not hydrate legacy mode")


def condition(page, cdp, rows: int, cards_per_row: int, keys: int) -> dict:
    select_rows(page, rows)
    before_dom = page.evaluate(INJECT, {"cards": cards_per_row})
    if before_dom["rows"] != rows or before_dom["syntheticCards"] != rows * cards_per_row:
        raise AssertionError(f"DOM fixture not attached to all {rows} rows: {before_dom}")
    if rows == 32 and before_dom["descendants"] < 100_000:
        raise AssertionError(f"Expected at least 100k *live* list DOM elements: {before_dom}")

    # Wait for stable row bounds outside the timed typing interval.
    page.wait_for_timeout(150)
    editor = page.locator(EDITOR)
    editor.click()
    editor.fill("")
    page.evaluate(
        "() => { window.__agentDomProbe.samples = []; window.__agentDomProbe.received = 0; }"
    )
    before = snapshot(cdp)
    start = time.perf_counter()
    page.keyboard.type("a" * keys, delay = 6)
    page.wait_for_function(
        "count => window.__agentDomProbe.samples.length >= count",
        arg = keys,
        timeout = 90_000,
    )
    wall_ms = 1000 * (time.perf_counter() - start)
    after = snapshot(cdp)
    probe = page.evaluate("() => window.__agentDomProbe")
    if probe["received"] != keys or editor.input_value() != "a" * keys:
        raise AssertionError(f"Composer did not receive {keys} unmodified inputs: {probe}")
    # A React update can remount rows; do not silently measure an empty fixture.
    post = page.evaluate("""() => ({
      rows: document.querySelectorAll('[data-agent-history-row]').length,
      syntheticCards: document.querySelectorAll('.agent-perf-tool-card').length,
      listNodes: document.querySelector('[data-agent-history-scroll-list="true"]')
        ?.querySelectorAll('*').length ?? 0,
    })""")
    if post["rows"] != rows or post["syntheticCards"] != rows * cards_per_row:
        raise AssertionError(f"Unstable virtual window/synthetic DOM during typing: {post}")

    measurements = {
        key + "_ms_per_key": round(
            max(0.0, after.get(key, 0.0) - before.get(key, 0.0)) * 1000 / keys, 3
        )
        for key in METRICS
    }
    samples = probe["samples"]
    return {
        "window_rows": rows,
        "dom": before_dom,
        "after_typing": post,
        "key_events": keys,
        "wall_ms_per_key": round(wall_ms / keys, 3),
        "event_to_two_frames_p50_ms": percentile(samples, 0.5),
        "event_to_two_frames_p95_ms": percentile(samples, 0.95),
        **measurements,
    }


def run(
    base: str, username: str, password: str, output: Path, cards_per_row: int, keys: int
) -> dict:
    from playwright.sync_api import sync_playwright

    auth = authenticate(base, username, password)
    thread_id = create_fixture(base, auth, 4000)
    results = []
    succeeded = False
    started = time.perf_counter()
    try:
        persisted = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
        if len(persisted["messages"]) != 4000:
            raise AssertionError("4,000-message precondition not met")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless = True)
            try:
                context = browser.new_context(viewport = {"width": 1440, "height": 960})
                context.add_init_script(seed_init_script(auth, []))
                # The page reloads for each numeric history window. Install on
                # every document, not only on the initial page before reload.
                context.add_init_script(f"({INSTALL_INPUT_PROBE})();")
                history_state = {
                    "state": {
                        "enabledThreads": {thread_id: True},
                        "windowRowsByThreadId": {thread_id: 32},
                    },
                    "version": 0,
                }
                context.add_init_script(
                    "(() => { const key = 'unsloth_agent_chat_history_v1';"
                    " if (localStorage.getItem(key) !== null) return;"
                    " localStorage.setItem(key, "
                    + json.dumps(json.dumps(history_state))
                    + "); })();"
                )
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda err: errors.append(str(err)))
                cdp = context.new_cdp_session(page)
                cdp.send("Performance.enable")
                page.goto(
                    f"{base}/chat?thread={thread_id}",
                    wait_until = "domcontentloaded",
                    timeout = 120_000,
                )
                page.locator(LIST).wait_for(state = "attached", timeout = 60_000)
                page.wait_for_function(
                    "() => document.querySelectorAll('[data-agent-history-row]').length === 32",
                    timeout = 60_000,
                )
                for rows in SIZES:
                    record = condition(page, cdp, rows, cards_per_row, keys)
                    results.append(record)
                    print(json.dumps(record, sort_keys = True), flush = True)

                dom32 = results[0]["dom"]["descendants"]
                dom5 = results[1]["dom"]["descendants"]
                if not (dom32 > 3 * dom5):
                    raise AssertionError(f"Reduced window failed DOM scaling: {dom32} vs {dom5}")
                # 32 at the end catches retained test-only nodes and window state drift.
                if results[-1]["dom"]["syntheticCards"] != 32 * cards_per_row:
                    raise AssertionError("Final 32-row control did not restore full DOM")

                # This test has real persisted messages but does not send any user message.
                persisted_after = auth_request_json(
                    auth, f"{base}/api/chat/threads/{thread_id}/messages"
                )
                if persisted_after["messages"] != persisted["messages"]:
                    raise AssertionError("Synthetic DOM benchmark changed persisted chat content")
                if errors:
                    raise AssertionError(f"Unhandled browser exceptions: {errors[:5]}")
            finally:
                browser.close()
        report = {
            "status": "PASS",
            "case": "live-studio-4k-history-synthetic-100k-dom",
            "message_count": 4000,
            "model_loaded": False,
            "synthetic_fixture_not_persisted": True,
            "cards_per_row": cards_per_row,
            "keys_per_window": keys,
            "window_sequence": SIZES,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "metrics_are_observational_not_timing_gates": True,
            "cases": results,
            "p50_wall_ms_by_window": {
                str(row): round(
                    statistics.median(
                        r["wall_ms_per_key"] for r in results if r["window_rows"] == row
                    ),
                    3,
                )
                for row in (2, 5, 8, 16, 32)
            },
        }
        succeeded = True
        return report
    finally:
        auth_request_json(
            auth, f"{base}/api/chat/threads", method = "DELETE", body = {"ids": [thread_id]}
        )
        output.parent.mkdir(parents = True, exist_ok = True)
        # Persist even partial results to make CI failures inspectable.
        output.write_text(
            json.dumps(
                {
                    "case": "live-studio-4k-history-synthetic-100k-dom",
                    "message_count": 4000,
                    "status": "PASS" if succeeded else "INCOMPLETE",
                    "cases": results,
                },
                indent = 2,
                sort_keys = True,
            )
            + "\n",
            encoding = "utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--url", required = True)
    parser.add_argument("--username", default = "unsloth")
    parser.add_argument("--password", required = True)
    parser.add_argument("--output", default = "logs/agent-history/dom-sweep.json")
    parser.add_argument("--cards-per-row", type = int, default = CARDS_PER_ROW)
    parser.add_argument("--keys", type = int, default = KEYS)
    args = parser.parse_args()
    if not (700 <= args.cards_per_row <= 1200) or not (6 <= args.keys <= 40):
        parser.error("The synthetic fixture needs 700..1200 cards/row and 6..40 keys")
    report = run(
        args.url.rstrip("/"),
        args.username,
        args.password,
        Path(args.output),
        args.cards_per_row,
        args.keys,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_s": report["elapsed_s"],
                "p50_wall_ms_by_window": report["p50_wall_ms_by_window"],
            },
            sort_keys = True,
        ),
        flush = True,
    )


if __name__ == "__main__":
    main()
