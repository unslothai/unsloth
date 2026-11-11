# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Measure heavy post-agent DOM remount churn on an isolated, disposable Studio.

Synthetic 1,000-message conversation: each *potential* row can represent
~10,000 DOM elements, but only the mounted virtual window exists in the DOM.
Extra-large bands make 200k/400k live-element windows possible in isolation.
These synthetic decorations are NOT a claim about real React tool components,
Windows WebView2, GPU-driver behavior or actual screen locking. Never use
production installations or private conversations for this test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.studio.playwright_agent_history_scroll import create_fixture
from tests.studio.studiobench.runtime.lifecycle import (
    auth_request_json,
    authenticate,
    seed_init_script,
)

LIST = '[data-agent-history-scroll-list="true"]'
VIEWPORT = ".aui-thread-viewport"
METRICS = ("TaskDuration", "ScriptDuration", "LayoutDuration", "RecalcStyleDuration")

# SIX descendants: div, button, svg, path, span, pre.
# Deliberately different from the low-DOM smoke test: multiple complex tool
# response blocks are mounted and discarded as the virtual window changes.
INJECT = """({regularCards, heavyCards, total}) => {
  const markup = '<div class="qa-tool-result">' +
    '<button type="button"><svg viewBox="0 0 12 12"><path d="M1 6h10"/></svg>' +
    '<span>Tool call</span></button><pre>synthetic stdout: ok</pre></div>';
  const heavyBand = index => index < 28 ||
    (index >= total - 30) ||
    [0.25, 0.50, 0.75].some(f => Math.abs(index - Math.floor(total * f)) <= 18);
  const seen = new Set();
  const stats = { mounts: 0, createdElements: 0, injectMs: 0, maxInjectMs: 0,
    mutationCalls: 0, longTasks: 0, maxLongTaskMs: 0, frameLagMs: 0,
    violation: false };
  let scheduled = false, previousFrame = 0;
  // Synthetic-only numerical trace: detect repeated oscillation between
  // logical row windows without ever recording message or tool content.
  const recentWindows = [];
  const observedWindows = new Set();
  let previousWindow = null;
  let windowTransitions = 0, windowRevisits = 0;
  const viewport = document.querySelector('.aui-thread-viewport');
  function frame(t) {
    if (previousFrame) stats.frameLagMs = Math.max(stats.frameLagMs, t - previousFrame);
    previousFrame = t;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  function mount() {
    scheduled = false;
    const rows = document.querySelectorAll('[data-agent-history-row]');
    if (rows.length > 5) { stats.violation = true; return; }
    for (const row of rows) {
      if (row.querySelector('[data-qa-heavy-fixture]')) continue;
      const owner = row.querySelector('[data-role]');
      if (!owner) continue;
      const index = Number(row.dataset.agentHistoryRow);
      const cards = heavyBand(index) ? heavyCards : regularCards;
      const before = performance.now();
      const fragment = document.createElement('div');
      fragment.dataset.qaHeavyFixture = 'true';
      // Bound the synthetic markup's height; DOM count remains real.
      fragment.style.cssText = 'contain:layout;max-height:2400px;overflow:hidden';
      fragment.innerHTML = markup.repeat(cards);
      owner.appendChild(fragment);
      const delta = performance.now() - before;
      stats.injectMs += delta;
      stats.maxInjectMs = Math.max(stats.maxInjectMs, delta);
      stats.mounts++;
      stats.createdElements += cards * 6 + 1;
      seen.add(index);
    }
  }
  function schedule() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(mount);
  }
  const root = document.querySelector('[data-agent-history-scroll-list="true"]');
  if (!root) throw Error('Virtualized list missing');
  const observer = new MutationObserver(() => {
    stats.mutationCalls++;
    const rows = [...root.querySelectorAll('[data-agent-history-row]')];
    const signature = rows.map(row => row.dataset.agentHistoryRow).join(',');
    if (signature !== previousWindow) {
      windowTransitions++;
      if (observedWindows.has(signature)) windowRevisits++;
      observedWindows.add(signature);
      previousWindow = signature;
      recentWindows.push({
        from: Number(rows[0]?.dataset.agentHistoryRow ?? -1),
        to: Number(rows[rows.length - 1]?.dataset.agentHistoryRow ?? -1),
        count: rows.length,
        filled: rows.filter(row => row.querySelector('[data-qa-heavy-fixture]')).length,
        scrollTop: Math.round(viewport?.scrollTop ?? -1),
      });
      if (recentWindows.length > 20) recentWindows.shift();
    }
    for (const row of rows) {
      if (!row.querySelector('[data-qa-heavy-fixture]')) { schedule(); break; }
    }
  });
  observer.observe(root, {subtree:true, childList:true});
  let longObserver;
  try {
    longObserver = new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        stats.longTasks++;
        stats.maxLongTaskMs = Math.max(stats.maxLongTaskMs, entry.duration);
      }
    });
    longObserver.observe({type:'longtask', buffered:true});
  } catch {}
  window.__qaDomChurn = {
    read: () => {
      const rows = [...root.querySelectorAll('[data-agent-history-row]')];
      return {
        indices: rows.map(row => Number(row.dataset.agentHistoryRow)),
        mounted: rows.length,
        filled: rows.filter(row => row.querySelector('[data-qa-heavy-fixture]')).length,
        listNodes: root.querySelectorAll('*').length,
        totalNodes: document.querySelectorAll('*').length,
        distinctRows: seen.size,
        windowTransitions, windowRevisits, distinctWindows: observedWindows.size,
        recentWindows,
        heapBytes: performance.memory?.usedJSHeapSize ?? null,
        ...stats,
      };
    },
    stop: () => { observer.disconnect(); longObserver?.disconnect(); },
  };
  schedule();
}"""


def metrics(cdp) -> dict:
    current = cdp.send("Performance.getMetrics")["metrics"]
    return {item["name"]: item["value"] for item in current}


def deltas(before: dict, after: dict) -> dict:
    return {
        name + "_ms": round(1000 * (after.get(name, 0) - before.get(name, 0)), 2)
        for name in METRICS
    }


def gc_retention_sample(cdp, page, messages: int, cap: int, label: str) -> dict:
    """Only numeric telemetry: never save heap snapshots or private DOM content."""
    before = metrics(cdp)
    # HeapProfiler GC is test-only CDP instrumentation, not application behavior.
    cdp.send("HeapProfiler.collectGarbage")
    page.wait_for_timeout(100)
    after = metrics(cdp)
    counters = cdp.send("Memory.getDOMCounters")
    current = state(page, messages, cap, label)
    used = after.get("JSHeapUsedSize")
    total = after.get("JSHeapTotalSize")
    if used is None or not isinstance(counters.get("nodes"), int):
        raise AssertionError("Required CDP heap/DOM counters unavailable")
    return {
        "checkpoint": label,
        "visibleIndices": current["indices"],
        "liveDocumentNodes": current["totalNodes"],
        "chromiumDOMNodesAfterGC": counters["nodes"],
        "chromiumDocumentsAfterGC": counters["documents"],
        "chromiumListenersAfterGC": counters["jsEventListeners"],
        "jsHeapUsedBytesAfterGC": int(used),
        "jsHeapTotalBytesAfterGC": int(total) if total is not None else None,
        "jsHeapUsedBeforeGC": int(before.get("JSHeapUsedSize", 0)),
        "gcObservationMs": round(
            1000 * (after.get("TaskDuration", 0) - before.get("TaskDuration", 0)),
            2,
        ),
        "totalMounts": current["mounts"],
        "cumulativeElementsCreated": current["createdElements"],
    }


def state(page, messages: int, limit: int, label: str) -> dict:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    try:
        page.wait_for_function(
            """() => {
              const rows = [...document.querySelectorAll('[data-agent-history-row]')];
              return rows.length === 5 && rows.every(row =>
                row.querySelector('[data-qa-heavy-fixture]'));
            }""",
            timeout = 90000,
        )
    except PlaywrightTimeoutError as exc:
        # Record numeric-only, synthetic-only state when the renderer stalls.
        # Never dump DOM HTML, transcript text, network payloads or screenshots.
        debug = page.evaluate("""() => {
          const rows = [...document.querySelectorAll('[data-agent-history-row]')];
          const viewport = document.querySelector('.aui-thread-viewport');
          return {
            mounted: rows.length,
            indices: rows.map(r => Number(r.dataset.agentHistoryRow)),
            filled: rows.filter(r => r.querySelector('[data-qa-heavy-fixture]')).length,
            liveNodes: document.querySelectorAll('*').length,
            scrollTop: Math.round(viewport?.scrollTop ?? -1),
            scrollHeight: Math.round(viewport?.scrollHeight ?? -1),
            viewportHeight: Math.round(viewport?.clientHeight ?? -1),
            createdElements: window.__qaDomChurn?.read()?.createdElements ?? null,
            windowTransitions: window.__qaDomChurn?.read()?.windowTransitions ?? null,
            windowRevisits: window.__qaDomChurn?.read()?.windowRevisits ?? null,
            recentWindows: window.__qaDomChurn?.read()?.recentWindows ?? [],
          };
        }""")
        raise AssertionError(f"{label}: destination window failed to settle: {debug}") from exc
    value = page.evaluate("() => window.__qaDomChurn.read()")
    positions = value["indices"]
    if (
        not 1 <= value["mounted"] <= 5
        or value["mounted"] != value["filled"]
        or positions != list(range(positions[0], positions[-1] + 1))
        or min(positions) < 0
        or max(positions) >= messages
        or value["totalNodes"] > limit
        or value["violation"]
    ):
        raise AssertionError(f"{label}: unbounded/broken virtual window: {value}")
    return value


def scroll(page, fraction: float) -> None:
    page.locator(VIEWPORT).hover()
    page.mouse.wheel(0, -1600 if fraction == 0 else 1600)
    for _ in range(3):
        page.evaluate(
            """fraction => {
              const viewport = document.querySelector('.aui-thread-viewport');
              viewport.scrollTop =
                (viewport.scrollHeight - viewport.clientHeight) * fraction;
              viewport.dispatchEvent(new Event('scroll'));
            }""",
            fraction,
        )
        page.wait_for_timeout(125)


def run(
    url: str,
    username: str,
    password: str,
    output: Path,
    messages: int,
    elements: int,
    rounds: int,
    retention_cycles: int,
) -> dict:
    from playwright.sync_api import sync_playwright

    base = url.rstrip("/")
    regular_cards = 1600
    heavy_cards = max(regular_cards, elements // (5 * 6))
    # Numeric, source-free evidence. Never serialize the transcript or HTML.
    result = {
        "scenario": "post-completion-virtual-window-DOM-churn",
        "status": "INCOMPLETE",
        "messageCount": messages,
        "approximateAllRowsDomIfMounted": messages * regular_cards * 6,
        "targetHeavyWindowNodes": elements,
        "normalRowElementsApprox": regular_cards * 6,
        "heavyRowElementsApprox": heavy_cards * 6,
        "browser": "headless-chromium",
        "modelLoaded": False,
        "windowsUnlockTested": False,
        "measurements": [],
        "retention": [],
        "retentionCycles": retention_cycles,
        "retentionCountersNote": "CDP process-wide DOM counters include objects beyond the live document; they are not an exact detached-node count.",
        "browserErrors": 0,
    }
    thread_id = None
    errors = []
    auth = authenticate(base, username, password)
    try:
        thread_id = create_fixture(base, auth, messages)
        original = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless = True)
            try:
                context = browser.new_context(viewport = {"width": 1440, "height": 960})
                context.add_init_script(
                    seed_init_script(
                        auth,
                        [],
                        {
                            "unsloth_agent_chat_history_v1": {
                                "state": {"globalEnabled": True, "globalWindowRows": 5},
                                "version": 0,
                            }
                        },
                    )
                )
                page = context.new_page()
                page.on("pageerror", lambda _: errors.append("pageerror"))
                page.on("crash", lambda: errors.append("renderer-crash"))
                cdp = context.new_cdp_session(page)
                cdp.send("Performance.enable")
                page.goto(
                    f"{base}/chat?thread={thread_id}",
                    wait_until = "domcontentloaded",
                    timeout = 120000,
                )
                page.locator(LIST).wait_for(state = "attached", timeout = 60000)
                page.wait_for_function(
                    """count => document.querySelectorAll('[data-agent-history-row]').length === 5 &&
                      document.querySelector('[data-agent-history-row]')?.getAttribute('aria-setsize') === String(count)""",
                    arg = messages,
                    timeout = 60000,
                )
                page.evaluate(
                    INJECT,
                    {"regularCards": regular_cards, "heavyCards": heavy_cards, "total": messages},
                )
                cap = elements + 30000
                initial = state(page, messages, cap, "initial-tail")
                result["measurements"].append(
                    {"phase": "initial", "targetFraction": 1.0, **initial}
                )

                # Browser tab backgrounding, not a real Windows lock/unlock.
                other = context.new_page()
                other.goto("about:blank")
                other.bring_to_front()
                page.wait_for_timeout(2000)
                page.bring_to_front()
                other.close()

                fractions = (0.75, 0.5, 0.25, 0.0, 1.0, 0.5, 0.9, 0.1, 1.0)
                for cycle in range(rounds):
                    for step, fraction in enumerate(fractions):
                        before = metrics(cdp)
                        started = time.perf_counter()
                        prior = state(page, messages, cap, "before-scroll")
                        scroll(page, fraction)
                        current = state(page, messages, cap, f"{cycle}-{step}")
                        after = metrics(cdp)
                        record = {
                            "phase": "scroll",
                            "cycle": cycle,
                            "step": step,
                            "fraction": fraction,
                            "elapsedMs": round(1000 * (time.perf_counter() - started), 2),
                            "liveNodeDelta": current["totalNodes"] - prior["totalNodes"],
                            "heapDeltaBytes": (
                                current["heapBytes"] - prior["heapBytes"]
                                if current["heapBytes"] is not None
                                and prior["heapBytes"] is not None
                                else None
                            ),
                            "createdElementDelta": (
                                current["createdElements"] - prior["createdElements"]
                            ),
                            "mountDelta": current["mounts"] - prior["mounts"],
                            "longTaskDelta": current["longTasks"] - prior["longTasks"],
                            **deltas(before, after),
                            **current,
                        }
                        result["measurements"].append(record)
                        if errors:
                            raise AssertionError("Browser error during heavy DOM remount")
                measurements = result["measurements"]
                high = max(value["listNodes"] for value in measurements)
                if high < elements * 0.8:
                    raise AssertionError(f"Stress intensity insufficient: {high} < 0.8*{elements}")
                if max(value["distinctRows"] for value in measurements) < 15:
                    raise AssertionError("Too few unmounted messages traversed")
                if not any(value["indices"][0] == 0 for value in measurements):
                    raise AssertionError("Beginning of transcript unreachable")
                if not any(value["indices"][-1] == messages - 1 for value in measurements):
                    raise AssertionError("End of transcript unreachable")
                if retention_cycles:
                    # Always compare the same mounted five-row tail after GC.
                    # A changing viewport would conflate legit live DOM and leaks.
                    baseline = gc_retention_sample(cdp, page, messages, cap, "baseline-tail")
                    result["retention"].append(baseline)
                    checkpoints = {10, 20, retention_cycles}
                    for iteration in range(1, retention_cycles + 1):
                        scroll(page, 0.0)
                        state(page, messages, cap, f"retention-top-{iteration}")
                        scroll(page, 1.0)
                        state(page, messages, cap, f"retention-tail-{iteration}")
                        if iteration in checkpoints:
                            observation = gc_retention_sample(
                                cdp, page, messages, cap, f"after-{iteration}"
                            )
                            if observation["visibleIndices"] != baseline["visibleIndices"]:
                                raise AssertionError(
                                    "Retention snapshots must use identical mounted rows"
                                )
                            observation["heapDeltaFromBaselineBytes"] = (
                                observation["jsHeapUsedBytesAfterGC"]
                                - baseline["jsHeapUsedBytesAfterGC"]
                            )
                            observation["domCounterDeltaFromBaseline"] = (
                                observation["chromiumDOMNodesAfterGC"]
                                - baseline["chromiumDOMNodesAfterGC"]
                            )
                            result["retention"].append(observation)
                        if errors:
                            raise AssertionError("Browser error during GC retention cycles")
                persisted = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
                if persisted["messages"] != original["messages"]:
                    raise AssertionError("Synthetic transcript persistence changed")
                if errors:
                    raise AssertionError("Unhandled browser error")
                result["status"] = "PASS"
            finally:
                browser.close()
    finally:
        result["browserErrors"] = len(errors)
        if thread_id is not None:
            auth_request_json(
                auth,
                f"{base}/api/chat/threads",
                method = "DELETE",
                body = {"ids": [thread_id]},
            )
        output.parent.mkdir(parents = True, exist_ok = True)
        output.write_text(json.dumps(result, indent = 2, sort_keys = True) + "\n", encoding = "utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--url", required = True)
    parser.add_argument("--username", default = "unsloth")
    parser.add_argument("--password", required = True)
    parser.add_argument("--output", default = "logs/agent-history/dom-churn.json")
    parser.add_argument("--messages", type = int, default = 1000)
    parser.add_argument("--elements", type = int, choices = (50000, 200000, 400000), default = 200000)
    parser.add_argument("--rounds", type = int, default = 1)
    parser.add_argument("--retention-cycles", type = int, default = 0)
    args = parser.parse_args()
    if not (
        1000 <= args.messages <= 2000
        and 1 <= args.rounds <= 2
        and 0 <= args.retention_cycles <= 40
        and (args.retention_cycles == 0 or args.elements == 50000)
    ):
        parser.error("Stress scenario outside bounded range")
    outcome = run(
        args.url,
        args.username,
        args.password,
        Path(args.output),
        args.messages,
        args.elements,
        args.rounds,
        args.retention_cycles,
    )
    samples = outcome["measurements"]
    print(
        json.dumps(
            {
                "status": outcome["status"],
                "samples": len(samples),
                "maxLiveDOM": max((x["totalNodes"] for x in samples), default = 0),
                "maxListDOM": max((x["listNodes"] for x in samples), default = 0),
                "errors": outcome["browserErrors"],
                "retentionSamples": len(outcome["retention"]),
            }
        ),
        flush = True,
    )


if __name__ == "__main__":
    main()
