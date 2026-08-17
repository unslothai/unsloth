# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which action BEFORE the scroll is what makes the scroll slow.

Re-running PR 9016's harness on this host reproduces its 300K scroll column exactly:

    scroll per repetition   666.9 / 1049.7 / 949.9 / 916.7 / 950.1 ms
    scroll longtasks        4, totalling 312 ms, worst 94 ms
    scroll task ms          558.2, of which layout 3.7 and style recalc 9.6

Repetition ONE is at the 667 ms floor of the gesture and every later repetition is at ~950 ms.
Whatever costs 280 ms is therefore not a property of scrolling a large thread -- repetition one
scrolls the same thread -- it is something a PREVIOUS action left behind. And a Long Tasks entry
is a main-thread task, so it is script or paint on the main thread, not raster and not
compositing.

`one_repetition` runs, in order:

    settle gate, expandTools, keystroke, scroll, jump, menu, delete, reopen

so the scroll in repetition N follows repetition N-1's jump, menu, delete and reopen, and its
own repetition's keystroke. This file runs each of those alone before an otherwise identical
scroll and reports what the scroll then costs. Every arm uses the harness's OWN action scripts,
imported rather than reimplemented, so an arm cannot differ from the harness by a detail of this
file.

Run:
    SMOKE_PORT=5271 PROBE_CHARS=300000 PROBE_REPS=4 python tests/studio/scroll_predecessor_probe.py
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args, start_vite, stop_process  # noqa: E402
import playwright_heavy_thread as hv  # noqa: E402

# The HARNESS's recorder, not paint_probe.py's. The harness's action scripts call
# `__hv.quiet()` and read `__longTasks`, and paint_probe.py's cut-down recorder has neither, so
# importing the wrong one fails every arm at the first action.
RECORDER_INIT = hv.RECORDER_INIT

PORT = int(os.environ.get("SMOKE_PORT", "5271"))
BASE = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/") or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not os.environ.get("SMOKE_BASE_URL", "").strip()
CHARS = int(os.environ.get("PROBE_CHARS", "300000"))
REPS = int(os.environ.get("PROBE_REPS", "4"))
LABEL = os.environ.get("PROBE_LABEL", "predecessor")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/paint-probe"))
OUT.mkdir(parents = True, exist_ok = True)

SETTLE = 120_000


def info(m: str) -> None:
    print(f"[pred] {m}", flush = True)


def settled(page) -> None:
    hv.wait_for_highlighting_settled(page, 900_000)


def expand(page) -> None:
    n = page.evaluate("() => window.__heavyThread.expandTools()")
    if n:
        page.wait_for_function(
            "(k) => window.__heavyThread.counts().collapsibleOutputs >= k",
            arg = n, timeout = 600_000,
        )


def hover_last(page) -> None:
    page.evaluate(
        """() => { const m = window.__heavyThread.lastAssistantMessage();
            if (m) m.scrollIntoView({ block: "center", behavior: "instant" }); }"""
    )
    page.wait_for_function(
        """() => {
            const top = window.__heavyThread.viewportMetrics().scrollTop;
            const s = window.__hvTop === top;
            window.__hvTop = top;
            return s;
        }""",
        timeout = 600_000,
    )
    page.locator('[data-role="assistant"]').last.hover(timeout = 600_000)


# Each entry runs `before(page)` and then the harness's own SCROLL_JS. `nothing` is the control:
# it is repetition one of the harness, which is the repetition that comes in at the floor.
def before_nothing(page):
    return None


def before_keystroke(page):
    return page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES)


def before_jump(page):
    return page.evaluate(hv.JUMP_JS, SETTLE)



def before_hover_only(page):
    """Hover the last assistant message and DO NOT open the menu.

    This separates a confound the first version of this file left in: the `menu` arm hovers the
    message to reveal the action bar before it can click "More", so "menu" was really
    "hover plus menu" and hover was never tested on its own.
    """
    hover_last(page)
    return {"hovered": True}



def park_pointer_in_gutter(page) -> dict:
    """Park the REAL pointer where the element under it does not change as content scrolls.

    NAMED FOR WHAT IT DOES, after the first version of it was named `park_pointer_outside` and
    measured, honestly, `pointer_outside: False` at every candidate. The thread scroller fills the
    viewport in this fixture, so there is no point on the page that is outside it and the premise
    of that name was false. What the top-left corner actually gives is a position over the
    scroller's own gutter, where the hit-test target is the same element on every step. That, not
    leaving the scroller, is the variable under test.

    The gesture dispatches its wheel events on the viewport element directly, so the pointer's
    position does not steer the scroll. What it does steer is whether the engine re-hit-tests
    content moving under a stationary cursor and fires pointerover / pointerout / mouseover /
    mouseout at it, which is the mechanism under test.

    The verification is not decoration. A corner that happens to sit inside the scroller makes
    this arm silently identical to the arm it is supposed to contrast with, and the run would
    then report "cost persists" for a reason that is an error in this function.
    """
    rect = page.evaluate(
        """() => { const v = window.__heavyThread.viewport();
            const r = v.getBoundingClientRect();
            return { x: r.x, y: r.y, w: r.width, h: r.height,
                     iw: window.innerWidth, ih: window.innerHeight }; }"""
    )
    # Prefer a point above the scroller, then left of it, then the top-right corner.
    candidates = []
    if rect["y"] > 12:
        candidates.append((rect["x"] + rect["w"] / 2, max(2, rect["y"] - 6)))
    if rect["x"] > 12:
        candidates.append((max(2, rect["x"] - 6), rect["y"] + rect["h"] / 2))
    candidates.append((rect["iw"] - 3, 3))
    candidates.append((3, 3))
    for x, y in candidates:
        page.mouse.move(x, y)
        page.wait_for_timeout(120)
        ok = page.evaluate(
            """([x, y]) => {
                const v = window.__heavyThread.viewport();
                const el = document.elementFromPoint(x, y);
                return { outside: Boolean(el) && !v.contains(el) && el !== v,
                         tag: el ? el.tagName.toLowerCase() : null };
            }""",
            [round(x), round(y)],
        )
        if ok["outside"]:
            return {"parked_at": [round(x), round(y)], "pointer_outside": True, "under": ok["tag"]}
    return {"parked_at": None, "pointer_outside": False}


def before_hover_then_gutter(page):
    """Arm the hover state, then take the pointer off the thread before scrolling.

    Separates "the first hover armed something that stays armed" from "content transiting under
    a stationary cursor keeps re-rendering the list".
    """
    hover_last(page)
    return {"hovered": True, **park_pointer_in_gutter(page)}


def before_gutter_only(page):
    """Control for the arm above: never hover, and park the pointer off the thread anyway."""
    return park_pointer_in_gutter(page)


def before_menu(page):
    hover_last(page)
    return page.evaluate(hv.MENU_JS, SETTLE)


def before_delete(page):
    hover_last(page)
    return page.evaluate(hv.DELETE_JS, SETTLE)


def before_reopen(page):
    out = page.evaluate(hv.REOPEN_JS, [SETTLE, SETTLE])
    settled(page)
    expand(page)
    return out


def before_delete_reopen_keystroke(page):
    """The exact predecessor set of the harness's repetition two."""
    hover_last(page)
    page.evaluate(hv.DELETE_JS, SETTLE)
    page.evaluate(hv.REOPEN_JS, [SETTLE, SETTLE])
    settled(page)
    expand(page)
    return page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES)


def before_menu_keystroke(page):
    hover_last(page)
    page.evaluate(hv.MENU_JS, SETTLE)
    return page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES)


ARMS = [
    ("nothing", before_nothing, "control: the harness's repetition one"),
    ("keystroke", before_keystroke, "5 characters into the composer, then scroll"),
    ("hover_only", before_hover_only,
     "hover the last assistant message to reveal its action bar, then scroll"),
    ("gutter_only", before_gutter_only,
     "no hover, pointer parked on the scroller gutter, then scroll"),
    ("hover_then_gutter", before_hover_then_gutter,
     "hover a message, then park the pointer on the gutter, then scroll"),
    ("menu", before_menu, "open and close the message action menu, then scroll"),
    ("reopen", before_reopen, "leave and re-enter the thread, then scroll"),
    (
        "delete_reopen_keystroke",
        before_delete_reopen_keystroke,
        "the harness's real repetition-two predecessor set",
    ),
]


def main() -> int:
    results = {"label": LABEL, "chars": CHARS, "reps": REPS, "arms": {}}
    # PROBE_ARMS trims the list without reordering it, so a two-arm A/B against another branch
    # runs the same arms in the same order as the full sweep did.
    wanted = os.environ.get("PROBE_ARMS", "").strip()
    arms = ARMS
    if wanted:
        keep = {a.strip() for a in wanted.split(",")}
        arms = [a for a in ARMS if a[0] in keep]
        results["arms_requested"] = sorted(keep)
    vite = None
    try:
        if OWNS_SERVER:
            info(f"starting vite on {PORT}")
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
            browser = pw.chromium.launch(args = chromium_launch_args())
            for name, before, why in arms:
                info(f"--- {name}")
                # A FRESH CONTEXT PER ARM, and that is the point. Every arm has to start from a
                # page that has never run any other action, or arm N inherits arm N-1's residue
                # and the whole file measures the order it happened to pick.
                ctx = browser.new_context(viewport = {"width": 1280, "height": 900})
                ctx.add_init_script(RECORDER_INIT)
                page = ctx.new_page()
                cdp = ctx.new_cdp_session(page)
                try:
                    page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
                    page.wait_for_function(
                        "() => Boolean(window.__heavyThread)", timeout = 180_000
                    )
                    plan = page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
                    page.wait_for_function(
                        "(n) => window.__heavyThread.messageCount() >= n",
                        arg = plan["messages"], timeout = 1_200_000,
                    )
                    expand(page)
                    settled(page)
                    rows = []
                    applied = None
                    # The mechanism claim is that the cost tracks how often the element under a
                    # stationary cursor CHANGES. That is directly countable, so count it rather
                    # than infer it from the timings.
                    install_boundary_counter = """() => {
                        window.__pb = { over: 0, out: 0, targets: new Set() };
                        window.__pbOver = (e) => {
                            window.__pb.over += 1;
                            window.__pb.targets.add(e.target);
                        };
                        window.__pbOut = () => { window.__pb.out += 1; };
                        document.addEventListener("pointerover", window.__pbOver, true);
                        document.addEventListener("pointerout", window.__pbOut, true);
                    }"""
                    read_boundary_counter = """() => {
                        document.removeEventListener("pointerover", window.__pbOver, true);
                        document.removeEventListener("pointerout", window.__pbOut, true);
                        return { over: window.__pb.over, out: window.__pb.out,
                                 distinctTargets: window.__pb.targets.size };
                    }"""
                    for i in range(REPS):
                        # Kept, not discarded: `park_pointer_outside` verifies through
                        # elementFromPoint that the cursor really left the scroller, and an arm
                        # whose whole meaning is "the pointer is elsewhere" must publish that
                        # verification rather than leave a reader to trust it.
                        applied = before(page)
                        page.evaluate(install_boundary_counter)
                        rows.append(
                            hv.run_action(
                                page, cdp, "scroll", hv.SCROLL_JS,
                                [hv.SCROLL_STEPS, hv.SCROLL_STEP_PX, SETTLE],
                            )
                        )
                        rows[-1]["boundary"] = page.evaluate(read_boundary_counter)
                        r = rows[-1]
                        info(
                            f"  rep {i + 1}/{REPS} gesture {r.get('gestureMs')} "
                            f"f>33 {r.get('frames_over_33')} worst {r.get('worst_frame_ms')} "
                            f"longtasks {r.get('long_tasks')}/{r.get('long_task_ms')}ms "
                            f"task {r.get('task_ms')} layout {r.get('layout_ms')} "
                            f"style {r.get('recalc_style_ms')}"
                        )
                        if name in ("delete", "delete_reopen_keystroke", "reopen"):
                            # These mutate the thread, so re-settle before the next repetition
                            # rather than letting arm state drift silently.
                            settled(page)
                            expand(page)

                    def med(k):
                        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                        return round(statistics.median(vals), 1) if vals else None

                    results["arms"][name] = {
                        "why": why,
                        "applied": applied,
                        "gesture_ms": med("gestureMs"),
                        "gesture_ms_all": [r.get("gestureMs") for r in rows],
                        "frames_over_33": med("frames_over_33"),
                        "frames_over_33_all": [r.get("frames_over_33") for r in rows],
                        "worst_frame_ms": med("worst_frame_ms"),
                        "longest_stall_ms": med("longest_stall_ms"),
                        "long_tasks": med("long_tasks"),
                        "long_task_ms": med("long_task_ms"),
                        "worst_long_task_ms": med("worst_long_task_ms"),
                        "task_ms": med("task_ms"),
                        "layout_ms": med("layout_ms"),
                        "recalc_style_ms": med("recalc_style_ms"),
                        "settle_ms": med("settleMs"),
                        "boundary_events": [r.get("boundary") for r in rows],
                    }
                except Exception as exc:  # noqa: BLE001
                    results["arms"][name] = {"why": why, "failed": repr(exc)}
                    info(f"  FAILED {exc!r}")
                finally:
                    ctx.close()
            browser.close()
        out = OUT / f"{LABEL}.json"
        out.write_text(json.dumps(results, indent = 2))
        info(f"wrote {out}")
        info("")
        info(
            f"{'predecessor':<26}{'gesture ms':>12}{'f>33':>7}{'worst ms':>10}"
            f"{'longtasks':>11}{'lt ms':>8}{'task ms':>9}{'layout':>8}{'style':>8}"
        )
        for n, r in results["arms"].items():
            if "failed" in r:
                info(f"{n:<26} FAILED {r['failed'][:60]}")
                continue
            info(
                f"{n:<26}{(r['gesture_ms'] or 0):>12.1f}{(r['frames_over_33'] or 0):>7.0f}"
                f"{(r['worst_frame_ms'] or 0):>10.1f}{(r['long_tasks'] or 0):>11.0f}"
                f"{(r['long_task_ms'] or 0):>8.0f}{(r['task_ms'] or 0):>9.1f}"
                f"{(r['layout_ms'] or 0):>8.1f}{(r['recalc_style_ms'] or 0):>8.1f}"
            )
    finally:
        if vite is not None:
            stop_process(vite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
