# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Synthetic completed-agent scroll/remount stress. Disposable Studio ONLY.

No private user data, dumps, traces, model, screenshots or message text exported.
Browser tab backgrounding is NOT an OS screen lock; Windows WebView2 requires
separate controlled hardware QA. This test intentionally stresses DOM churn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.studio.playwright_agent_history_scroll import create_fixture
from tests.studio.studiobench.runtime.lifecycle import (
    authenticate,
    auth_request_json,
    seed_init_script,
)

LIST = '[data-agent-history-scroll-list="true"]'
ROW = "[data-agent-history-row]"
VIEWPORT = ".aui-thread-viewport"
INJECT = """({cards,large}) => {
  const template = '<div class="agent-stress-tool"><button>Tool<svg><path d="M1 5h8"/></svg></button>' +
    '<pre>synthetic tool output success</pre></div>';
  const stats = { mountedTimes: 0, mutationCalls: 0, longTasks: 0, longestMs: 0, limit: false };
  const distinct = new Set();
  let pending = false;
  function inject() {
    pending = false;
    const rows = document.querySelectorAll('[data-agent-history-row]');
    if (rows.length > 5) { stats.limit = true; return; }
    for (const row of rows) {
      if (row.querySelector('[data-agent-idle-fixture]')) continue;
      const target = row.querySelector('[data-role]');
      if (!target) continue;
      const id = Number(row.dataset.agentHistoryRow);
      const n = id === large ? 2000 : cards;
      if (document.querySelectorAll('*').length + n * 7 > 120000) {
        stats.limit = true;
        return;
      }
      const el = document.createElement('div');
      el.dataset.agentIdleFixture = 'true';
      el.style.cssText = 'contain:layout;max-width:100%;overflow:hidden';
      el.innerHTML = template.repeat(n);
      target.appendChild(el);
      distinct.add(id); stats.mountedTimes++;
    }
  }
  function schedule() {
    if (!pending) { pending = true; requestAnimationFrame(inject); }
  }
  const observer = new MutationObserver(() => {
    stats.mutationCalls++;
    if (document.querySelector('[data-agent-history-row]:not(:has([data-agent-idle-fixture]))'))
      schedule();
  });
  observer.observe(document, {childList:true, subtree:true});
  try {
    new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        stats.longTasks++;
        stats.longestMs = Math.max(stats.longestMs, entry.duration);
      }
    }).observe({type:'longtask', buffered:true});
  } catch (_) {}
  window.__agentIdleQA = () => ({
    rows: document.querySelectorAll('[data-agent-history-row]').length,
    injected: document.querySelectorAll('[data-agent-idle-fixture]').length,
    indices: [...document.querySelectorAll('[data-agent-history-row]')].map(
      el => Number(el.dataset.agentHistoryRow)),
    listDom: document.querySelector('[data-agent-history-scroll-list="true"]')
      ?.querySelectorAll('*').length ?? 0,
    allDom: document.querySelectorAll('*').length,
    uniqueRows: distinct.size, ...stats,
    heapBytes: performance.memory?.usedJSHeapSize ?? null,
  });
  schedule();
}"""


def sample(page, count: int, label: str) -> dict:
    # An independent evaluate after wait_for_function can observe a *different*
    # staged window. Return the validated numeric snapshot atomically and require
    # the same completely filled rows over three animation frames.
    handle = page.wait_for_function(
        """() => {
      const x = window.__agentIdleQA?.();
      if (!x || x.rows !== 5 || x.injected !== 5 ||
          x.indices.length !== 5 || x.limit) {
        window.__agentIdleStable = null;
        return false;
      }
      const signature = x.indices.join(',') + ':' + x.mountedTimes;
      const last = window.__agentIdleStable;
      if (!last || last.signature !== signature) {
        window.__agentIdleStable = {signature, frames: 1};
        return false;
      }
      last.frames += 1;
      return last.frames >= 3 ? x : false;
    }""",
        polling = "raf",
        timeout = 30000,
    )
    x = handle.json_value()
    indices = x["indices"]
    if (
        not 1 <= x["rows"] <= 5
        or x["injected"] != x["rows"]
        or indices != list(range(indices[0], indices[-1] + 1))
        or min(indices) < 0
        or max(indices) >= count
        or x["allDom"] > 120000
        or x["limit"]
    ):
        raise AssertionError(f"{label}: DOM/window safety invariant failed: {x}")
    return {"step": label, **x}


def scroll(page, fraction: float) -> None:
    page.locator(VIEWPORT).hover()
    page.mouse.wheel(0, -1600 if fraction == 0 else 1600)
    for _ in range(3):
        page.evaluate(
            """fraction => {
          const el = document.querySelector('.aui-thread-viewport');
          el.scrollTop = (el.scrollHeight-el.clientHeight)*fraction;
          el.dispatchEvent(new Event('scroll'));
        }""",
            fraction,
        )
        page.wait_for_timeout(130)


def run(
    url: str, username: str, password: str, output: Path, messages: int, cards: int, rounds: int
) -> None:
    from playwright.sync_api import sync_playwright

    base = url.rstrip("/")
    auth = authenticate(base, username, password)
    thread_id = None
    data = []
    errors = []
    status = "INCOMPLETE"
    try:
        thread_id = create_fixture(base, auth, messages)
        initial = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
        if len(initial["messages"]) != messages:
            raise AssertionError("Synthetic transcript persistence failed")
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
                page.on("pageerror", lambda err: errors.append("pageerror"))
                page.on("crash", lambda: errors.append("renderer-crashed"))
                page.goto(
                    f"{base}/chat?thread={thread_id}", wait_until = "domcontentloaded", timeout = 120000
                )
                page.locator(LIST).wait_for(state = "attached", timeout = 60000)
                page.wait_for_function(
                    "n => document.querySelector('[data-agent-history-row]')?.getAttribute('aria-setsize') === String(n) && document.querySelectorAll('[data-agent-history-row]').length === 5",
                    arg = messages,
                    timeout = 60000,
                )
                page.evaluate(INJECT, {"cards": cards, "large": messages - 1})
                data.append(sample(page, messages, "initial-huge-tail"))
                # Page background/foreground approximates resume but not Windows unlock.
                tab = context.new_page()
                tab.goto("about:blank")
                tab.bring_to_front()
                page.wait_for_timeout(2500)
                page.bring_to_front()
                tab.close()
                data.append(sample(page, messages, "after-background"))
                fractions = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.55, 0.1, 0.9)
                for cycle in range(rounds):
                    for j, fraction in enumerate(fractions):
                        scroll(page, fraction)
                        data.append(sample(page, messages, f"cycle-{cycle}-step-{j}"))
                        if errors:
                            raise AssertionError("Browser emitted errors or crashed")
                if max(x["uniqueRows"] for x in data) < 12:
                    raise AssertionError("Too few previously unmounted rows visited")
                if max(x["listDom"] for x in data) < 6000:
                    raise AssertionError("DOM load below required threshold")
                if not any(x["indices"][0] == 0 for x in data):
                    raise AssertionError("Start of transcript unreachable")
                if not any(x["indices"][-1] == messages - 1 for x in data):
                    raise AssertionError("Tail of transcript unreachable")
                latest = auth_request_json(auth, f"{base}/api/chat/threads/{thread_id}/messages")
                if latest.get("messages") != initial["messages"]:
                    raise AssertionError("Synthetic transcript was modified by scrolling")
                status = "PASS"
            finally:
                browser.close()
    finally:
        if thread_id:
            auth_request_json(
                auth, f"{base}/api/chat/threads", method = "DELETE", body = {"ids": [thread_id]}
            )
        output.parent.mkdir(parents = True, exist_ok = True)
        output.write_text(
            json.dumps(
                {
                    "status": status,
                    "scenario": "synthetic-completed-long-run-heavy-scroll",
                    "messages": messages,
                    "cardsPerNormalRow": cards,
                    "oversizedRowCards": 2000,
                    "globalWindow": 5,
                    "browser": "headless-chromium",
                    "modelLoaded": False,
                    "osUnlockSimulated": False,
                    "numericSamples": data,
                    "browserErrorCount": len(errors),
                },
                indent = 2,
            )
            + "\n",
            encoding = "utf-8",
        )
    print(
        json.dumps(
            {
                "status": status,
                "samples": len(data),
                "maxDOM": max(x["allDom"] for x in data),
                "browserErrors": len(errors),
            }
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument("--url", required = True)
    p.add_argument("--username", default = "unsloth")
    p.add_argument("--password", required = True)
    p.add_argument("--output", default = "logs/agent-history/idle-scroll-stress.json")
    p.add_argument("--messages", type = int, default = 800)
    p.add_argument("--cards", type = int, default = 400)
    p.add_argument("--rounds", type = int, default = 3)
    a = p.parse_args()
    if not (400 <= a.messages <= 4000 and 200 <= a.cards <= 700 and 1 <= a.rounds <= 5):
        p.error("Test fixture outside bounded safe limits")
    run(a.url, a.username, a.password, Path(a.output), a.messages, a.cards, a.rounds)


if __name__ == "__main__":
    main()
