# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioural probes for use-hover-quiet-during-scroll.ts (PR #9068).

Only the USER action bar is hover-gated (`autohide="always"` in thread.tsx); the assistant bar is
always rendered. So every case below is stated in terms of `.aui-user-action-bar-root` VISIBILITY,
which is what a person sees.

Each case is written so that the ANSWER DIFFERS between a tree with the hook and a tree without
it, when the hypothesis is real. Run the same file against both trees and diff the JSON.

Cases:
    h1_stale_pointer     hover a message, move the cursor OUT of the viewport, scroll
                         programmatically. Does a bar appear with the cursor elsewhere?
    h2_remount_no_move   hover a message, remount the viewport (closeThread/openThread, which is
                         what a thread switch does), do NOT move the mouse, then wheel. The hook's
                         `pointerSeen` is per-effect, so it is false again.
    h2_wheel_pointermove does a stationary wheel gesture emit `pointermove` on the viewport at all?
    h3_continuous        wheel deltas closer together than QUIET_MS for ~3s, then a decaying
                         momentum tail, then an abrupt stop. Bar count sampled throughout.
    h4_stream_scroll     scrollTop driven repeatedly (stand-in for stream auto-scroll) while the
                         user moves the pointer onto a message. Does the bar ever appear?
    h5_keyboard          PageDown with focus in the viewport, cursor parked outside it.
    h6_reseed_midscroll  re-import the thread mid-scroll (re-keys every message) and check for a
                         stuck bar or a console error.
    h8_nested_scroller   scroll a nested `overflow-y-auto` box; confirm the hook does not see it
                         (scroll does not bubble) by hovering immediately afterwards.

Run:
    SMOKE_PORT=5231 PROBE_CHARS=120000 PROBE_ENGINES=chromium,webkit,firefox \\
      PROBE_LABEL=head python -u tests/studio/hover_semantics_probe.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args, start_vite, stop_process  # noqa: E402
import playwright_heavy_thread as hv  # noqa: E402

PORT = int(os.environ.get("SMOKE_PORT", "5231"))
BASE = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/") or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not os.environ.get("SMOKE_BASE_URL", "").strip()
CHARS = int(os.environ.get("PROBE_CHARS", "120000"))
ENGINES = [e.strip() for e in os.environ.get("PROBE_ENGINES", "chromium").split(",") if e.strip()]
LABEL = os.environ.get("PROBE_LABEL", "tree")
REPS = int(os.environ.get("PROBE_REPS", "3"))
OUT = Path(os.environ.get("PW_ART_DIR", "logs/hover_semantics"))
OUT.mkdir(parents = True, exist_ok = True)

# Counts what a person sees, plus which message owns each visible bar, so "a bar appeared" can be
# told apart from "a bar appeared on the wrong message".
CENSUS_JS = """
() => {
  const vis = (b) => {
    const cs = getComputedStyle(b);
    if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0") return false;
    const r = b.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const bars = [...document.querySelectorAll(".aui-user-action-bar-root")].filter(vis);
  return {
    shown: bars.length,
    owners: bars.map((b) => {
      const m = b.closest("[data-message-id]");
      return m ? m.getAttribute("data-message-id") : null;
    }),
  };
}
"""

# Which message is under a given viewport point right now, independent of any hover bookkeeping.
UNDER_JS = """
([x, y]) => {
  const el = document.elementFromPoint(x, y);
  const m = el ? el.closest("[data-message-id]") : null;
  return m ? { id: m.getAttribute("data-message-id"), role: m.getAttribute("data-role") } : null;
}
"""

# Installed before any app code. Counts the events the hook depends on, and records console
# errors, so "nothing broke" is a measurement rather than an absence of screenshots.
INSTRUMENT_INIT = """
window.__probe = { pointermove: 0, scroll: 0, scrollCapture: 0, wheel: 0, errors: [] };
window.addEventListener("error", (e) => {
  window.__probe.errors.push(String(e.message));
});
window.__probeArm = () => {
  const v = document.querySelector(".aui-thread-viewport");
  if (!v) return false;
  if (v.__probeArmed) return true;
  v.__probeArmed = true;
  // BUBBLE phase, exactly like the hook. This matters: `scroll` does not bubble, but it does
  // have a capture phase, so a CAPTURE listener here would also count scrolls of nested boxes
  // and the H8 answer would be an instrumentation artefact. Both are counted so the difference
  // is visible rather than assumed.
  v.addEventListener("pointermove", () => { window.__probe.pointermove += 1; });
  v.addEventListener("scroll", () => { window.__probe.scroll += 1; });
  v.addEventListener("wheel", () => { window.__probe.wheel += 1; });
  v.addEventListener("scroll", () => { window.__probe.scrollCapture += 1; }, true);
  return true;
};
window.__probeReset = () => {
  window.__probe.pointermove = 0;
  window.__probe.scroll = 0;
  window.__probe.scrollCapture = 0;
  window.__probe.wheel = 0;
};
"""


def info(m: str) -> None:
    print(f"[hover-probe] {m}", flush = True)


def census(page):
    return page.evaluate(CENSUS_JS)


def under(page, x, y):
    return page.evaluate(UNDER_JS, [x, y])


def viewport_box(page):
    return page.evaluate(
        """() => { const v = window.__heavyThread.viewport();
             const r = v.getBoundingClientRect();
             return { x: r.x, y: r.y, w: r.width, h: r.height }; }"""
    )


def pick_user_message(page):
    """A user message roughly in the middle of the viewport, and its centre point."""
    return page.evaluate(
        """() => {
          const v = window.__heavyThread.viewport();
          const vr = v.getBoundingClientRect();
          const mid = vr.y + vr.height / 2;
          const users = [...document.querySelectorAll('[data-role="user"]')];
          let best = null, bestD = Infinity;
          for (const u of users) {
            const r = u.getBoundingClientRect();
            if (r.height < 20 || r.width < 20) continue;
            if (r.bottom < vr.top + 40 || r.top > vr.bottom - 40) continue;
            const d = Math.abs((r.top + r.bottom) / 2 - mid);
            if (d < bestD) { bestD = d; best = u; }
          }
          if (!best) return null;
          const r = best.getBoundingClientRect();
          return {
            id: best.getAttribute("data-message-id"),
            x: Math.round(r.x + r.width / 2),
            y: Math.round(Math.max(r.top + 8, Math.min(r.bottom - 8, (r.top + r.bottom) / 2))),
          };
        }"""
    )


def scroll_by(page, dy):
    page.evaluate(
        """(dy) => { const v = window.__heavyThread.viewport();
             v.scrollTo({ top: Math.max(0, v.scrollTop + dy), behavior: "instant" }); }""",
        dy,
    )


def plan_scroll_to_put_user_message_under(page, x, y, exclude_id):
    """A scroll delta that lands a DIFFERENT user message under the point (x, y).

    Without this the settle step usually resolves an assistant message, whose action bar is not
    hover-gated at all, and the case would report "no phantom bar" for the wrong reason.
    """
    return page.evaluate(
        """([x, y, exclude]) => {
          const v = window.__heavyThread.viewport();
          const max = v.scrollHeight - v.clientHeight;
          const users = [...document.querySelectorAll('[data-role="user"]')];
          let best = null;
          for (const u of users) {
            if (u.getAttribute("data-message-id") === exclude) continue;
            const r = u.getBoundingClientRect();
            if (r.height < 24 || r.width < 24) continue;
            if (r.left > x || r.right < x) continue;
            // Scrolling by dy moves content up by dy, so the element covers y when
            // r.top - dy <= y <= r.bottom - dy.
            const lo = Math.ceil(r.top - y + 6);
            const hi = Math.floor(r.bottom - y - 6);
            if (hi < lo) continue;
            const candidates = [lo, hi, Math.round((lo + hi) / 2)];
            for (const dy of candidates) {
              if (Math.abs(dy) < 80 || Math.abs(dy) > 1200) continue;
              if (v.scrollTop + dy < 0 || v.scrollTop + dy > max) continue;
              if (!best || Math.abs(dy) < Math.abs(best.dy)) {
                best = { dy, id: u.getAttribute("data-message-id") };
              }
            }
          }
          return best;
        }""",
        [x, y, exclude_id],
    )


def park_outside_viewport(page):
    """A point that is really OUTSIDE the scroller, found by hit test rather than assumed.

    This matters more than it looks. The smoke page has no sidebar and the scroller reaches the
    window edges, so the obvious "corner of the window" point is still INSIDE the viewport
    element: moving there fires a `pointermove` on the viewport, the hook updates
    pointerX/pointerY, and the coordinates are no longer stale. A first pass reported H1 clean for
    exactly that reason, and a second pass reported it clean again with a point 25px below the
    scroller. Only a point the viewport does not contain leaves the recorded coordinates
    untouched, so the point is searched for and the search result is reported.
    """
    return page.evaluate(
        """() => {
          const v = window.__heavyThread.viewport();
          const r = v.getBoundingClientRect();
          const W = window.innerWidth, H = window.innerHeight;
          const composer = window.__heavyThread.composer();
          const tries = [];
          if (composer) {
            const cr = composer.getBoundingClientRect();
            if (cr.width > 0 && cr.height > 0) {
              tries.push([Math.round(cr.x + cr.width / 2), Math.round(cr.y + cr.height / 2)]);
            }
          }
          for (let y = 2; y < H - 2; y += 12) {
            for (let x = 2; x < W - 2; x += 60) tries.push([x, y]);
          }
          for (const [x, y] of tries) {
            const el = document.elementFromPoint(x, y);
            if (!el) continue;
            if (el === v || v.contains(el)) continue;
            return {
              x, y, ok: true, tag: el.tagName,
              cls: String(el.className || "").slice(0, 60),
              viewport_rect: { x: r.x, y: r.y, w: r.width, h: r.height },
            };
          }
          return { x: 2, y: 2, ok: false, tag: null,
                   viewport_rect: { x: r.x, y: r.y, w: r.width, h: r.height } };
        }"""
    )


def seed(page):
    page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
    page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 180_000)
    plan = page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
    page.wait_for_function(
        "(n) => window.__heavyThread.messageCount() >= n",
        arg = plan["messages"],
        timeout = 600_000,
    )
    hv.wait_for_highlighting_settled(page, 600_000)
    page.wait_for_function("() => window.__probeArm()", timeout = 60_000)
    return plan


# --------------------------------------------------------------------------------------------
# Cases. Each returns a dict; each is given a freshly seeded page.
# --------------------------------------------------------------------------------------------


def case_h1_stale_pointer(page) -> dict:
    """Hover a message, take the cursor OUT of the viewport, then scroll. Any bar left showing is
    a bar on a message the cursor is not over."""
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    # steps=1 so the only pointermove the viewport sees is the one at the target.
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(500)
    hovered = census(page)
    # Out of the viewport entirely, verified by hit test rather than assumed.
    away = park_outside_viewport(page)
    away_x, away_y = away["x"], away["y"]
    page.mouse.move(away_x, away_y)
    page.wait_for_timeout(500)
    after_leave = census(page)
    page.evaluate("() => window.__probeReset()")
    # Aim the scroll so a DIFFERENT user message ends up under the stale point. Otherwise the
    # settle step usually resolves an assistant message, whose bar is not hover-gated, and the
    # case would come back clean without having tested anything.
    plan = plan_scroll_to_put_user_message_under(page, target["x"], target["y"], target["id"])
    scroll_by(page, plan["dy"] if plan else -900)
    page.wait_for_timeout(1200)
    settled = census(page)
    return {
        "target": target,
        "hovered_shown": hovered["shown"],
        "after_leave_shown": after_leave["shown"],
        "cursor_at": [away_x, away_y],
        "cursor_left_viewport": away["ok"],
        "cursor_over_message": under(page, away_x, away_y),
        "scroll_plan": plan,
        "settled_shown": settled["shown"],
        "settled_owners": settled["owners"],
        "message_at_stale_point": under(page, target["x"], target["y"]),
        "events": page.evaluate("() => ({...window.__probe})"),
        # The break: a bar is visible while the cursor is not over any message.
        "phantom_bar": settled["shown"] > 0,
    }


def case_h2_remount_no_move(page) -> dict:
    """A thread switch remounts the viewport, so the hook's effect re-runs and `pointerSeen` is
    false again. If the cursor never moves after that, every boundary event during a scroll is
    swallowed and nothing is ever delivered."""
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(500)
    before = census(page)
    page.evaluate("() => window.__heavyThread.closeThread()")
    page.wait_for_timeout(400)
    page.evaluate("() => window.__heavyThread.openThread()")
    page.wait_for_function("() => window.__heavyThread.messageCount() > 0", timeout = 120_000)
    page.wait_for_timeout(1500)
    page.wait_for_function("() => window.__probeArm()", timeout = 60_000)
    page.evaluate("() => window.__probeReset()")
    after_remount = census(page)
    top_before = page.evaluate("() => window.__heavyThread.viewportMetrics().scrollTop")
    # Wheel WITHOUT any mouse.move: the cursor is where it was, over the conversation. Upward,
    # because a remount can leave the viewport pinned at the bottom where a downward wheel is a
    # no-op and the case would silently test nothing.
    plan = plan_scroll_to_put_user_message_under(page, target["x"], target["y"], target["id"])
    total = plan["dy"] if plan else -1200
    for _ in range(5):
        page.mouse.wheel(0, total / 5)
        page.wait_for_timeout(40)
    page.wait_for_timeout(1200)
    settled = census(page)
    top_after = page.evaluate("() => window.__heavyThread.viewportMetrics().scrollTop")
    now_under = under(page, target["x"], target["y"])
    owner = settled["owners"][0] if settled["owners"] else None
    return {
        "target": target,
        "before_remount_shown": before["shown"],
        "after_remount_shown": after_remount["shown"],
        "scrolled_px": top_before - top_after,
        "scroll_plan": plan,
        "settled_shown": settled["shown"],
        "settled_owners": settled["owners"],
        "message_under_cursor": now_under,
        "events": page.evaluate("() => ({...window.__probe})"),
        # Only meaningful if the scroll actually moved a different message under the cursor.
        "inconclusive": abs(top_before - top_after) < 50
        or (now_under or {}).get("id") == target["id"],
        # The break: the cursor is over a USER message and either no bar is shown or the bar is
        # on some other message.
        "wrong_or_missing_bar": bool(
            (now_under or {}).get("role") == "user" and owner != (now_under or {}).get("id")
        ),
    }


def case_h2_wheel_pointermove(page) -> dict:
    """Does a stationary wheel gesture emit `pointermove` on the viewport? If not, the hook's
    `pointerSeen` can never be set by scrolling alone."""
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(400)
    page.evaluate("() => window.__probeReset()")
    for _ in range(10):
        page.mouse.wheel(0, 120)
        page.wait_for_timeout(60)
    page.wait_for_timeout(600)
    return {"events_during_wheel_only": page.evaluate("() => ({...window.__probe})")}


def case_h3_continuous(page) -> dict:
    """A gesture that never goes quiet for QUIET_MS, then a momentum tail, then a stop.

    Playwright cannot generate real OS trackpad momentum; this approximates it with wheel deltas
    at a fixed 60ms cadence (all gaps under QUIET_MS) followed by a geometric decay in delta size
    at the same cadence. Real momentum is generated by the OS compositor and continues to deliver
    wheel events after the fingers lift; the cadence and the decay are the parts reproduced here,
    the compositor-side delivery is not.
    """
    scroll_by(page, 6000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(500)
    x, y = target["x"], target["y"]

    def sample():
        c = census(page)
        u = under(page, x, y)
        owner = c["owners"][0] if c["owners"] else None
        want = (u or {}).get("id") if (u or {}).get("role") == "user" else None
        return {
            "shown": c["shown"],
            "owner": owner,
            "under": (u or {}).get("id"),
            "under_role": (u or {}).get("role"),
            # What a person would call correct: a bar on the user message under the cursor, and
            # no bar when the cursor is on an assistant message.
            "correct": (owner == want) and c["shown"] == (1 if want else 0),
        }

    start = sample()
    during = []
    for i in range(30):
        page.mouse.wheel(0, -40)
        page.wait_for_timeout(60)
        if i % 6 == 0:
            during.append(sample())
    delta = 40.0
    tail = []
    for _ in range(12):
        page.mouse.wheel(0, -max(1, int(delta)))
        page.wait_for_timeout(60)
        delta *= 0.75
        tail.append(sample())
    recovery = []
    t0 = time.perf_counter()
    for _ in range(20):
        page.wait_for_timeout(50)
        r = sample()
        r["ms"] = round((time.perf_counter() - t0) * 1000)
        recovery.append(r)
    end = sample()
    first_correct = next((r["ms"] for r in recovery if r["correct"]), None)
    return {
        "start": start,
        "during_gesture": during,
        "tail": tail,
        "recovery": recovery,
        "end": end,
        "ms_to_correct_after_stop": first_correct,
        "correct_at_end": end["correct"],
        "never_settled_during_gesture": all(not d["correct"] for d in during),
    }


def case_h4_stream_scroll(page) -> dict:
    """Stand-in for stream auto-scroll: scrollTop written every ~40ms for 3s. The user moves the
    pointer onto a message part-way through. Does the bar appear before the stream ends?"""
    scroll_by(page, 6000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(3, 3)
    page.wait_for_timeout(300)
    # Room to grow downward, otherwise the driver writes the same scrollTop every tick, the
    # viewport fires no `scroll` at all, and the case would report "not suppressed" for the wrong
    # reason. This bit me once already.
    # Centre a USER message before the driver starts. Picking one afterwards fails outright: a
    # single assistant message in this fixture can be taller than the viewport, so a position
    # chosen by pixel offset routinely has no user message on screen at all and the case skips.
    page.evaluate(
        """() => {
          const users = [...document.querySelectorAll('[data-role="user"]')];
          const pick = users[Math.floor(users.length * 0.6)] || users[users.length - 1];
          if (pick) pick.scrollIntoView({ block: "center", behavior: "instant" });
        }"""
    )
    page.wait_for_timeout(600)
    page.evaluate("() => window.__probeReset()")
    # Auto-scroll driver in the page, so the Python round trip does not set the cadence. 40ms is
    # comfortably under QUIET_MS, which is the point: a stream keeps `scrolling` true forever.
    page.evaluate(
        """() => {
          const v = window.__heavyThread.viewport();
          window.__streamStop = false;
          const tick = () => {
            if (window.__streamStop) return;
            v.scrollTop = v.scrollTop + 2;
            window.setTimeout(tick, 40);
          };
          tick();
        }"""
    )
    page.wait_for_timeout(500)
    # The driver keeps moving the content, so the first look can land between user messages.
    tgt = None
    for _ in range(20):
        tgt = pick_user_message(page)
        if tgt:
            break
        page.wait_for_timeout(200)
    if not tgt:
        page.evaluate("() => { window.__streamStop = true; }")
        return {"skipped": "no user message in view while the stream driver ran"}
    page.mouse.move(tgt["x"], tgt["y"])
    samples = []
    for _ in range(14):
        page.wait_for_timeout(150)
        samples.append(census(page)["shown"])
    during_stream_max = max(samples) if samples else 0
    events = page.evaluate("() => ({...window.__probe})")
    page.evaluate("() => { window.__streamStop = true; }")
    page.wait_for_timeout(900)
    after = census(page)
    now_under = under(page, tgt["x"], tgt["y"])
    return {
        "samples_during_stream": samples,
        "during_stream_max_bars": during_stream_max,
        "after_stream_shown": after["shown"],
        "after_stream_owners": after["owners"],
        "message_under_cursor": now_under,
        "events": events,
        # If the driver never actually scrolled, nothing below means anything.
        "inconclusive": events["scroll"] < 10,
        # The break: the pointer sat on a user message for ~2s of streaming and no bar appeared.
        "suppressed_for_whole_stream": during_stream_max == 0,
    }


def case_h5_keyboard(page) -> dict:
    """PageDown with focus in the viewport, cursor parked OUTSIDE it.

    Same settle path as H1, reached without any pointer input at all, which is the point: a
    keyboard scroll can move a message under a pointer position that was recorded minutes ago.
    """
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(500)
    away = park_outside_viewport(page)
    away_x, away_y = away["x"], away["y"]
    page.mouse.move(away_x, away_y)
    page.wait_for_timeout(400)
    before = census(page)
    page.evaluate("() => window.__probeReset()")
    page.evaluate(
        "() => { const v = window.__heavyThread.viewport(); v.tabIndex = -1; v.focus(); }"
    )
    top_before = page.evaluate("() => window.__heavyThread.viewportMetrics().scrollTop")
    # Several presses, because one PageDown may land an assistant message under the stale point
    # and the case would look clean without having reached the interesting state.
    rows = []
    for i in range(6):
        page.keyboard.press("PageUp" if i % 2 == 0 else "PageDown")
        page.wait_for_timeout(700)
        c = census(page)
        u = under(page, target["x"], target["y"])
        rows.append(
            {
                "press": "PageUp" if i % 2 == 0 else "PageDown",
                "shown": c["shown"],
                "owners": c["owners"],
                "stale_point_message": u,
            }
        )
    top_after = page.evaluate("() => window.__heavyThread.viewportMetrics().scrollTop")
    events = page.evaluate("() => ({...window.__probe})")
    return {
        "before_shown": before["shown"],
        "rows": rows,
        "scrolled_px": top_after - top_before,
        "events": events,
        "cursor_over_message": under(page, away_x, away_y),
        "cursor_left_viewport": away["ok"],
        "inconclusive": events["scroll"] == 0,
        # The break: a bar visible while the cursor is not over the viewport at all.
        "phantom_bar": any(r["shown"] > 0 for r in rows),
    }


def case_h6_reseed_midscroll(page) -> dict:
    """Re-import the thread while a scroll is in flight, which re-keys every message, and switch
    the thread view out and back. Look for a stuck bar or a console error."""
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    page.mouse.move(target["x"], target["y"])
    page.wait_for_timeout(500)
    hovered = census(page)
    scroll_by(page, -300)
    page.wait_for_timeout(40)
    # Inside the quiet window, so the settle timer is still pending when the tree is replaced.
    page.evaluate("(n) => window.__heavyThread.seed(n)", max(20000, CHARS // 4))
    page.wait_for_timeout(2000)
    after_reseed = census(page)
    errors_a = page.evaluate("() => window.__probe.errors.slice()")
    # Now the same, but with a viewport remount landing inside the quiet window.
    tgt = pick_user_message(page)
    if tgt:
        page.mouse.move(tgt["x"], tgt["y"])
        page.wait_for_timeout(400)
        scroll_by(page, -200)
        page.wait_for_timeout(40)
        page.evaluate("() => window.__heavyThread.closeThread()")
        page.wait_for_timeout(60)
        page.evaluate("() => window.__heavyThread.openThread()")
        page.wait_for_timeout(2000)
    after_remount = census(page)
    page.mouse.move(3, 3)
    page.wait_for_timeout(700)
    at_rest = census(page)
    return {
        "hovered_shown": hovered["shown"],
        "after_reseed_shown": after_reseed["shown"],
        "after_remount_shown": after_remount["shown"],
        "at_rest_after_shown": at_rest["shown"],
        "errors": page.evaluate("() => window.__probe.errors.slice()"),
        "errors_after_reseed": errors_a,
        "stuck_bar_at_rest": at_rest["shown"] > 0,
    }


def case_h8_nested_scroller(page) -> dict:
    """Scroll a nested `overflow-y-auto` box. `scroll` does not bubble from elements, so the
    hook's viewport listener should not see it and hover should keep working immediately."""
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    nested = page.evaluate(
        """() => {
          const v = window.__heavyThread.viewport();
          const els = [...v.querySelectorAll("*")].filter((e) => {
            if (e === v) return false;
            const cs = getComputedStyle(e);
            const oy = cs.overflowY, ox = cs.overflowX;
            const scrolls = (oy === "auto" || oy === "scroll") && e.scrollHeight > e.clientHeight + 8;
            const scrollsX = (ox === "auto" || ox === "scroll") && e.scrollWidth > e.clientWidth + 8;
            return scrolls || scrollsX;
          });
          return els.map((e) => ({
            cls: e.className && e.className.baseVal === undefined ? String(e.className).slice(0, 80) : "",
            tag: e.tagName,
            sh: e.scrollHeight, ch: e.clientHeight, sw: e.scrollWidth, cw: e.clientWidth,
            inMessage: Boolean(e.closest("[data-message-id]")),
          })).slice(0, 20);
        }"""
    )
    # Does scrolling a nested box raise the viewport's scroll counter?
    page.evaluate("() => window.__probeReset()")
    bubbled = page.evaluate(
        """() => {
          const v = window.__heavyThread.viewport();
          const els = [...v.querySelectorAll("*")].filter((e) => {
            const cs = getComputedStyle(e);
            const oy = cs.overflowY, ox = cs.overflowX;
            return ((oy === "auto" || oy === "scroll") && e.scrollHeight > e.clientHeight + 8) ||
                   ((ox === "auto" || ox === "scroll") && e.scrollWidth > e.clientWidth + 8);
          });
          if (!els.length) return { scrolled: 0 };
          for (const e of els.slice(0, 5)) { e.scrollTop += 20; e.scrollLeft += 20; }
          return { scrolled: Math.min(5, els.length) };
        }"""
    )
    page.wait_for_timeout(120)
    counters = page.evaluate("() => ({...window.__probe})")
    # Hover immediately after a nested scroll: must work, since the hook never went scrolling.
    target = pick_user_message(page)
    hovered = None
    if target:
        page.mouse.move(target["x"], target["y"])
        page.wait_for_timeout(300)
        hovered = census(page)["shown"]
    return {
        "nested_scrollers": nested,
        "nested_scroller_count": len(nested),
        "scrolled": bubbled,
        "viewport_scroll_events_seen": counters["scroll"],
        "hover_after_nested_scroll_shown": hovered,
    }


def case_h1t_touch_scroll(page) -> dict:
    """The touch variant of H1.

    A touch drag emits `pointermove` before the browser claims the gesture, so `pointerX/pointerY`
    and `pointerSeen` are set from a FINGER position. The hook does not look at `pointerType`.
    Touch scrolling produces no hover at all on any engine, so any action bar visible after this
    is a bar the user cannot explain.

    On Chromium the drag is a REAL touch sequence through CDP `Input.dispatchTouchEvent`, so the
    events are trusted and the engine's own gesture handling runs. On WebKit and Firefox the CDP
    endpoint does not exist and the drag is synthesised with `PointerEvent(pointerType: "touch")`
    dispatched from the page, which exercises the hook's code path but is NOT a real gesture;
    that difference is recorded in `drag` below.
    """
    scroll_by(page, 4000)
    page.wait_for_timeout(600)
    target = pick_user_message(page)
    if not target:
        return {"skipped": "no user message in view"}
    x, y = target["x"], target["y"]
    before = census(page)
    drag = "synthetic"
    try:
        cdp = page.context.new_cdp_session(page)
        for i in range(8):
            points = [{"x": x, "y": y - i * 40}]
            cdp.send(
                "Input.dispatchTouchEvent",
                {
                    "type": "touchStart" if i == 0 else "touchMove",
                    "touchPoints": points,
                },
            )
            page.wait_for_timeout(30)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        drag = "cdp"
    except Exception:  # noqa: BLE001
        page.evaluate(
            """([x, y]) => {
              const v = window.__heavyThread.viewport();
              for (let i = 0; i < 8; i += 1) {
                v.dispatchEvent(new PointerEvent("pointermove", {
                  bubbles: true, clientX: x, clientY: y - i * 40, pointerType: "touch",
                  pointerId: 1, isPrimary: true,
                }));
              }
            }""",
            [x, y],
        )
    page.wait_for_timeout(300)
    # The scroll a touch drag would have produced. Kept separate from the drag so the engines
    # that could not take the CDP path still reach settle().
    plan = plan_scroll_to_put_user_message_under(page, x, y, target["id"])
    scroll_by(page, plan["dy"] if plan else -600)
    page.wait_for_timeout(1200)
    settled = census(page)
    return {
        "drag": drag,
        "target": target,
        "before_shown": before["shown"],
        "settled_shown": settled["shown"],
        "settled_owners": settled["owners"],
        "scroll_plan": plan,
        "message_at_finger_point": under(page, x, y),
        "events": page.evaluate("() => ({...window.__probe})"),
        # The break: a bar is visible after a touch-only interaction, where there is no hover.
        "phantom_bar": settled["shown"] > 0,
    }


CASES = {
    "h1t_touch_scroll": case_h1t_touch_scroll,
    "h1_stale_pointer": case_h1_stale_pointer,
    "h2_remount_no_move": case_h2_remount_no_move,
    "h2_wheel_pointermove": case_h2_wheel_pointermove,
    "h3_continuous": case_h3_continuous,
    "h4_stream_scroll": case_h4_stream_scroll,
    "h5_keyboard": case_h5_keyboard,
    "h6_reseed_midscroll": case_h6_reseed_midscroll,
    "h8_nested_scroller": case_h8_nested_scroller,
}

ONLY = [c.strip() for c in os.environ.get("PROBE_CASES", "").split(",") if c.strip()]
STRICT = os.environ.get("PROBE_STRICT", "").strip() not in ("", "0")

# The key each case sets when it has seen the break it exists to look for. Under PROBE_STRICT the
# file exits non-zero if any repetition of any case sets one, so it can gate CI. Verified to fail
# on the tree WITHOUT the fix and pass on the tree with it, rather than assumed.
BREAK_KEYS = {
    "h1_stale_pointer": ["phantom_bar"],
    "h1t_touch_scroll": ["phantom_bar"],
    "h2_remount_no_move": ["wrong_or_missing_bar"],
    "h4_stream_scroll": ["suppressed_for_whole_stream"],
    "h5_keyboard": ["phantom_bar"],
    "h6_reseed_midscroll": ["stuck_bar_at_rest"],
}


def failures_in(engine: str, name: str, rows: list) -> list[str]:
    out: list[str] = []
    for i, row in enumerate(rows):
        if row.get("inconclusive"):
            continue
        # A case that threw carries no break key, so without this it scored as a pass and the
        # gate went green precisely when the probe stopped working.
        if row.get("failed"):
            out.append(f"{engine} {name} rep{i}: case raised -- {row['failed'][:300]}")
            continue
        for key in BREAK_KEYS.get(name, []):
            if row.get(key):
                out.append(f"{engine} {name} rep{i}: {key} -- {json.dumps(row)[:300]}")
    return out


def main() -> int:
    results = {"label": LABEL, "chars": CHARS, "reps": REPS, "engines": {}}
    vite = None
    try:
        if OWNS_SERVER:
            vite = start_vite(PORT)
            deadline = time.time() + 300
            while time.time() < deadline:
                with socket.socket() as s:
                    s.settimeout(1)
                    if s.connect_ex(("127.0.0.1", PORT)) == 0:
                        break
                time.sleep(1)
            info("vite ready")
        with sync_playwright() as pw:
            for engine in ENGINES:
                launcher = getattr(pw, engine)
                kwargs = {"args": chromium_launch_args()} if engine == "chromium" else {}
                browser = launcher.launch(**kwargs)
                per_case: dict[str, list] = {}
                for name, fn in CASES.items():
                    if ONLY and name not in ONLY:
                        continue
                    for rep in range(REPS):
                        # Fresh context per repetition: these cases mutate hover state and the
                        # thread, and a carried-over `active` would make the next one a lie.
                        ctx = browser.new_context(viewport = {"width": 1280, "height": 900})
                        ctx.add_init_script(INSTRUMENT_INIT)
                        page = ctx.new_page()
                        try:
                            seed(page)
                            row = fn(page)
                        except Exception as exc:  # noqa: BLE001
                            row = {"failed": repr(exc)}
                        finally:
                            ctx.close()
                        per_case.setdefault(name, []).append(row)
                        info(f"{engine} {name} rep{rep}: {json.dumps(row)[:400]}")
                results["engines"][engine] = per_case
                browser.close()
        (OUT / f"{LABEL}.json").write_text(json.dumps(results, indent = 2), encoding = "utf-8")
        info(f"wrote {OUT / f'{LABEL}.json'}")
    finally:
        if vite is not None:
            stop_process(vite)
    broken: list[str] = []
    for engine, cases in results["engines"].items():
        for name, rows in cases.items():
            broken.extend(failures_in(engine, name, rows))
    if broken:
        info("")
        for line in broken:
            info(f"FAIL {line}")
    if STRICT:
        return 1 if broken else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
