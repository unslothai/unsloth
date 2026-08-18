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
            arg = n,
            timeout = 600_000,
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


# What each predecessor's OWN return value has to say before the scroll after it may be published
# under that predecessor's name. Every one of the harness's action scripts encodes "I did not
# happen" as a value rather than by throwing -- `return null` when the element it needs is not in
# the DOM, and a null FIELD when the DOM never reached the state it was waiting for inside
# SETTLE. MENU_JS returns normally with openMs None when the menu never opened, DELETE_JS with ms
# None when the message never left, REOPEN_JS with ms and closedMs None. The loop in main() marks
# an arm failed only when something RAISED, so without these an arm whose predecessor silently did
# nothing measures a scroll with no predecessor at all, reports it in the `menu` or `delete` row,
# and the process exits 0.
def _keystroke_proof(out: dict) -> list[str]:
    """The DOM value is what this file wrote; only the runtime's copy shows React received it."""
    if out.get("median_sample_ms") is None:
        return ["no keystroke was timed"]
    if out.get("runtimeText") != out.get("domText"):
        return [f"the composer holds {out.get('runtimeText')!r}, not {out.get('domText')!r}"]
    return []


def _jump_proof(out: dict) -> list[str]:
    landed = out.get("landedAt")
    travelled = out.get("travelledPx") or 0
    problems = []
    if landed is None or landed > 1 or travelled <= 0:
        problems.append(f"landed at {landed}px after {travelled}px of travel")
    # travelledPx is observed now, so a jump that began part-way up the thread reports a smaller
    # number rather than the full height, and the arm would quietly average unequal gestures.
    # hv owns this too, so the probe and the harness cannot disagree about what counts as a jump
    # that began at the bottom.
    shortfall = hv.jump_anchor_shortfall(out.get("startedFrom"), out.get("bottom"))
    if shortfall:
        problems.append(
            f"{shortfall}, so this repetition is a shorter gesture than the ones it is "
            "aggregated with"
        )
    return problems


def _menu_proof(out: dict) -> list[str]:
    problems = []
    if out.get("openMs") is None:
        problems.append(f"the menu never opened within {SETTLE}ms")
    if out.get("closeMs") is None:
        problems.append(f"the menu never closed within {SETTLE}ms")
    if not out.get("itemsWhileOpen"):
        problems.append("the menu opened with no items in it")
    return problems


def _delete_proof(out: dict) -> list[str]:
    problems = []
    if out.get("ms") is None:
        problems.append(f"the message never left the DOM within {SETTLE}ms")
    before, after = out.get("before"), out.get("after")
    if before is None or after is None or after >= before:
        problems.append(f"the message count did not drop ({before} -> {after})")
    return problems


def _reopen_proof(out: dict) -> list[str]:
    problems = []
    if out.get("ms") is None:
        problems.append(f"the thread never came back within {SETTLE}ms")
    if out.get("closedMs") is None:
        problems.append(f"the thread was never seen to unmount within {SETTLE}ms")
    before, after = out.get("before"), out.get("after")
    if before is None or after is None or after < before:
        problems.append(f"it came back with {after} of {before} messages")
    return problems


PREDECESSOR_PROOFS = {
    "keystroke": _keystroke_proof,
    "jump": _jump_proof,
    "menu": _menu_proof,
    "delete": _delete_proof,
    "reopen": _reopen_proof,
}


def checked(action: str, out):
    """The predecessor's own return value, or an exception naming what it failed to prove.

    Raised rather than recorded, because `main()` already treats a raise as a failed arm, writes
    the JSON without it and exits 1. A predecessor that did not happen leaves the arm measuring
    the control under another arm's label, which is worse than measuring nothing.
    """
    if out is None:
        raise RuntimeError(
            f"the {action} predecessor returned null: the element it needs was not in the DOM"
        )
    problems = PREDECESSOR_PROOFS[action](out)
    if problems:
        raise RuntimeError(f"the {action} predecessor did not complete: {'; '.join(problems)}")
    return out


# The measured scroll itself is the one thing every arm has in common, so it is the one thing no
# arm may be allowed to skip. `checked()` above proves the PREDECESSOR happened; this proves the
# gesture the predecessor is supposed to be affecting happened too. Without it a row where
# SCROLL_JS returned null, or never went quiet, or never moved, is appended like any other, `med()`
# drops the missing fields as if they were merely absent, and the arm publishes a predecessor
# comparison built on a scroll that did not occur.
def scroll_row_problems(row: dict) -> list[str]:
    problems = []
    if not row.get("ran"):
        problems.append("SCROLL_JS returned null, so the viewport was not in the DOM")
        return problems
    gesture = row.get("gestureMs")
    if not isinstance(gesture, (int, float)):
        problems.append(f"it reported no gesture duration (gestureMs={gesture!r})")
    if row.get("settleMs") is None:
        problems.append(
            "the page never went quiet within the settle timeout, so the post-gesture work is "
            "unbounded and settle_ms is not a measurement of anything"
        )
    travelled = row.get("scrolledPx")
    if not isinstance(travelled, (int, float)):
        problems.append(f"it reported no travel (scrolledPx={travelled!r})")
    else:
        # Against the FULL requested distance, not merely against zero. A viewport clamped or
        # snapped back for part of a repetition still reports a positive number, and skewed_arms
        # compares medians, so one short repetition among complete ones is hidden by the median
        # while its timing stays in the experiment. hv owns the threshold so this cannot drift
        # away from the check the main harness applies to the same gesture.
        shortfall = hv.scroll_travel_shortfall(travelled)
        if shortfall:
            problems.append(shortfall)
    return problems


# The tolerance on "the same gesture". The scroll is 20 steps of 400px from a fixed anchor, so a
# healthy arm travels exactly as far as the control; 5% is slack for a fixture whose scroll height
# leaves the last step clamped at a boundary, not room for an arm to scroll a different distance.
TRAVEL_TOLERANCE = 0.05


def skewed_arms(arms: dict) -> list[str]:
    """Arms whose measured scroll covered a different distance from the control's.

    Every arm is a comparison against `nothing`, and that comparison only means anything if the
    gesture being compared is the same gesture. `scroll_row_problems` rejects a scroll that did
    not happen; this rejects a set of scrolls that happened DIFFERENTLY, which reads as a
    predecessor cost and is not one.
    """
    control = arms.get("nothing", {}).get("scrolled_px")
    if not control:
        return []
    return [
        f"{n} travelled {r['scrolled_px']}px against the control's {control}px"
        for n, r in arms.items()
        if r.get("scrolled_px") and abs(r["scrolled_px"] - control) > TRAVEL_TOLERANCE * control
    ]


def checked_scroll(name: str, row: dict) -> dict:
    """Raised, not recorded, for the same reason `checked()` raises: `main()` marks the arm
    failed, keeps it out of the table and exits non-zero."""
    problems = scroll_row_problems(row)
    if problems:
        raise RuntimeError(
            f"the measured scroll under the {name} predecessor did not complete: "
            f"{'; '.join(problems)}"
        )
    return row


# Each entry runs `before(page)` and then the harness's own SCROLL_JS. `nothing` is the control:
# it is repetition one of the harness, which is the repetition that comes in at the floor.
def before_nothing(page):
    return None


def before_keystroke(page):
    return checked("keystroke", page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES))


def before_jump(page):
    # The measured scroll leaves the viewport thousands of px above the bottom, so from
    # repetition 2 on a raw JUMP_JS would begin from a different place than repetition 1 and the
    # arm would take a median across gestures of different lengths. ACTION_SETUPS is what the
    # harness applies to its own jump for exactly this reason; applying it here keeps the
    # predecessor the same gesture as the harness's, repetition after repetition.
    page.evaluate(hv.ACTION_SETUPS["jump"])
    return checked("jump", page.evaluate(hv.JUMP_JS, SETTLE))


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
                if (!el) return { stable: false, outside: false, tag: null };
                // The scroller ITSELF is the stable hit, and is the one this function is named
                // for: a point that lands on the viewport element rather than on any message
                // inside it keeps the same hit-test target while content moves under the cursor,
                // which is the variable under test. Demanding a point OUTSIDE the scroller can
                // never be satisfied in this fixture -- the docstring above says so -- so the
                // old predicate rejected every candidate, returned `pointer_outside: False` with
                // the mouse left on an unverified corner, and both control arms published
                // timings anyway.
                return { stable: el === v || !v.contains(el),
                         outside: !v.contains(el),
                         tag: el.tagName.toLowerCase() };
            }""",
            [round(x), round(y)],
        )
        if ok["stable"]:
            return {
                "parked_at": [round(x), round(y)],
                "pointer_stable": True,
                "pointer_outside": ok["outside"],
                "under": ok["tag"],
            }
    # No verified point means the arm's premise does not hold on this page, and a timing published
    # under the label "pointer parked on the gutter" would then be a second copy of the arm it is
    # supposed to contrast with. Fail the arm instead; `main` records it as failed and exits 1.
    raise RuntimeError(
        "no stable pointer position found: every candidate hit content inside the scroller"
    )


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
    return checked("menu", page.evaluate(hv.MENU_JS, SETTLE))


def before_delete(page):
    hover_last(page)
    return checked("delete", page.evaluate(hv.DELETE_JS, SETTLE))


# REOPEN_JS destructures [timeoutMs, settleMs, graceMs, probeEveryMs]. Passing only the two
# timeouts left graceMs undefined, so quietUntilIdle's `now - lastActivity >= graceMs` compared
# against NaN, never became true, and every reopen arm sat out the full 120s timeout instead of
# returning when Shiki went idle. That is ~16 minutes added to a four-repetition sweep, and it
# put a two-minute idle gap before each affected scroll, so those arms were not measuring the
# immediate-predecessor sequence they claim to.
def before_reopen(page):
    out = checked(
        "reopen",
        page.evaluate(
            hv.REOPEN_JS,
            [SETTLE, SETTLE, hv.HIGHLIGHT_GRACE_MS, hv.HIGHLIGHT_PROBE_MS],
        ),
    )
    settled(page)
    expand(page)
    return out


def before_delete_reopen_keystroke(page):
    """The exact predecessor set of the harness's repetition two.

    Every step is checked, not just the last one. A composite arm that publishes only its final
    keystroke's proof is an arm whose delete and reopen can both have silently failed.
    """
    hover_last(page)
    deleted = checked("delete", page.evaluate(hv.DELETE_JS, SETTLE))
    reopened = checked(
        "reopen",
        page.evaluate(
            hv.REOPEN_JS,
            [SETTLE, SETTLE, hv.HIGHLIGHT_GRACE_MS, hv.HIGHLIGHT_PROBE_MS],
        ),
    )
    settled(page)
    expand(page)
    typed = checked("keystroke", page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES))
    return {"delete": deleted, "reopen": reopened, "keystroke": typed}


def before_menu_keystroke(page):
    hover_last(page)
    opened = checked("menu", page.evaluate(hv.MENU_JS, SETTLE))
    typed = checked("keystroke", page.evaluate(hv.KEYSTROKE_JS, hv.KEYSTROKES))
    return {"menu": opened, "keystroke": typed}


ARMS = [
    ("nothing", before_nothing, "control: the harness's repetition one"),
    ("keystroke", before_keystroke, "5 characters into the composer, then scroll"),
    # The docstring's list of predecessors is the harness's own action order, so every one of them
    # has to be reachable from the default sweep. `before_jump` and `before_delete` were written
    # and then never registered, which left the file measuring five of the seven it claims.
    ("jump", before_jump, "jump to the top of the thread, then scroll"),
    (
        "hover_only",
        before_hover_only,
        "hover the last assistant message to reveal its action bar, then scroll",
    ),
    (
        "gutter_only",
        before_gutter_only,
        "no hover, pointer parked on the scroller gutter, then scroll",
    ),
    (
        "hover_then_gutter",
        before_hover_then_gutter,
        "hover a message, then park the pointer on the gutter, then scroll",
    ),
    ("menu", before_menu, "open and close the message action menu, then scroll"),
    ("delete", before_delete, "delete the last assistant message, then scroll"),
    ("reopen", before_reopen, "leave and re-enter the thread, then scroll"),
    (
        "delete_reopen_keystroke",
        before_delete_reopen_keystroke,
        "the harness's real repetition-two predecessor set",
    ),
    (
        "menu_keystroke",
        before_menu_keystroke,
        "open and close the menu, then type into the composer, then scroll",
    ),
]


# Arms whose predecessor PERMANENTLY removes a message from the runtime's repository. Re-opening
# the thread does not undo a delete -- reopen deliberately preserves the runtime -- so these are
# the arms whose fixture shrinks by one message per repetition unless it is put back.
MUTATING_ARMS = ("delete", "delete_reopen_keystroke")
# Arms after which the thread has to be re-highlighted and its tool cards re-opened: the two above
# because restoring re-imports the thread, and `reopen` because it tears the view down itself.
SETTLE_AFTER_ARMS = MUTATING_ARMS + ("reopen",)


def fixture_drift(counts: list[int]) -> str | None:
    """The message count at the start of every repetition, or a complaint if they differ.

    The claim this whole file rests on is that every arm scrolled the SAME thread with a different
    thing in front of it. A count that falls between repetitions says the later repetitions of
    that arm scrolled a smaller thread than the `nothing` control did, and the arm's median is
    then part predecessor and part missing content.
    """
    if len(set(counts)) <= 1:
        return None
    return (
        f"the thread changed size between repetitions ({counts}); the fixture was not restored, "
        "so the later repetitions scrolled a smaller thread than the control did"
    )


def main() -> int:
    results = {"label": LABEL, "chars": CHARS, "reps": REPS, "arms": {}}
    # PROBE_ARMS trims the list without reordering it, so a two-arm A/B against another branch
    # runs the same arms in the same order as the full sweep did.
    wanted = os.environ.get("PROBE_ARMS", "").strip()
    arms = ARMS
    if wanted:
        keep = {a.strip() for a in wanted.split(",")}
        # A name that is not an arm used to filter the sweep down to nothing and then report a
        # successful run with an empty table, which is the worst answer a probe can give.
        unknown = sorted(keep - {a[0] for a in ARMS})
        if unknown:
            info(f"PROBE_ARMS names no such arm: {', '.join(unknown)}")
            info(f"known arms: {', '.join(a[0] for a in ARMS)}")
            return 2
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
                # The main harness does this too, and it is not optional: Performance.getMetrics
                # on a session whose Performance domain was never enabled returns an EMPTY metric
                # list rather than an error. `hv.run_action` then hands two empty dicts to
                # `cdp_counters`, every counter comes back None, and the table below prints
                # `(r["task_ms"] or 0)` -- a 0.0 in the task, layout and style columns that reads
                # as "this arm did no main-thread work". Measured on this tree, Chromium 1234:
                # getMetrics without enable returns 0 metrics, with enable 36.
                cdp.send("Performance.enable")
                try:
                    page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
                    page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 180_000)
                    plan = page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
                    page.wait_for_function(
                        "(n) => window.__heavyThread.messageCount() >= n",
                        arg = plan["messages"],
                        timeout = 1_200_000,
                    )
                    expand(page)
                    settled(page)
                    rows = []
                    applied = None
                    fixture = []
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
                        # The thread this repetition is about to be measured against, read before
                        # the predecessor runs. Printed and published, because the whole claim of
                        # this file is that N arms scrolled the SAME thread with different things
                        # in front of them, and a drifting count says they did not.
                        fixture.append(page.evaluate("() => window.__heavyThread.messageCount()"))
                        # Kept, not discarded: `park_pointer_outside` verifies through
                        # elementFromPoint that the cursor really left the scroller, and an arm
                        # whose whole meaning is "the pointer is elsewhere" must publish that
                        # verification rather than leave a reader to trust it.
                        applied = before(page)
                        # Armed from run_action's after_setup hook, NOT here. ACTION_SETUPS
                        # anchors the viewport to the bottom first, and after the predecessor or
                        # the previous repetition that is a full-height reposition, which fires
                        # its own pointer boundary events. Counting from here folded those into
                        # the measured gesture's total by an amount that depended on where each
                        # arm's predecessor had left the viewport -- so the counter would appear
                        # to support a predecessor effect that was generated outside the gesture
                        # being compared.
                        rows.append(
                            hv.run_action(
                                page,
                                cdp,
                                "scroll",
                                hv.SCROLL_JS,
                                [hv.SCROLL_STEPS, hv.SCROLL_STEP_PX, SETTLE],
                                after_setup = lambda p: p.evaluate(install_boundary_counter),
                            )
                        )
                        rows[-1]["boundary"] = page.evaluate(read_boundary_counter)
                        checked_scroll(name, rows[-1])
                        r = rows[-1]
                        info(
                            f"  rep {i + 1}/{REPS} fixture {fixture[-1]} msgs "
                            f"gesture {r.get('gestureMs')} "
                            f"f>33 {r.get('frames_over_33')} worst {r.get('worst_frame_ms')} "
                            f"longtasks {r.get('long_tasks')}/{r.get('long_task_ms')}ms "
                            f"task {r.get('task_ms')} layout {r.get('layout_ms')} "
                            f"style {r.get('recalc_style_ms')}"
                        )
                        if name in MUTATING_ARMS:
                            # `delete` is destructive to the runtime's REPOSITORY, not to the
                            # view, so neither re-opening the thread nor re-expanding its tool
                            # cards puts the message back. Left alone, repetition 2 of these arms
                            # scrolls a thread one message shorter than repetition 1 and
                            # repetition 3 one shorter again -- and the fixture is whole cycles of
                            # one message per kind, so each pass takes a DIFFERENT content kind
                            # off the end. The arms would then be compared against a `nothing`
                            # control that still holds the whole fixture. Untimed, and after every
                            # snapshot for this repetition has been taken.
                            restored = page.evaluate("() => window.__heavyThread.restore()")
                            page.wait_for_function(
                                "(n) => window.__heavyThread.messageCount() >= n",
                                arg = restored,
                                timeout = 600_000,
                            )
                        if name in SETTLE_AFTER_ARMS:
                            # A restore and a re-open both throw away every highlighted fence and
                            # collapse every tool card, so re-settle and re-expand before the next
                            # repetition rather than letting arm state drift silently.
                            settled(page)
                            expand(page)

                    drift = fixture_drift(fixture)
                    if drift:
                        raise RuntimeError(drift)

                    def med(k):
                        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
                        return round(statistics.median(vals), 1) if vals else None

                    results["arms"][name] = {
                        "why": why,
                        "applied": applied,
                        # Flat across repetitions on a healthy arm. A descending list is the
                        # fixture drifting under the arm, which makes its later repetitions
                        # incomparable with the `nothing` control.
                        "fixture_messages": fixture,
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
                        # Published so the cross-arm check below has something to compare, and so
                        # a reader can see that every arm scrolled the same distance rather than
                        # take it on trust.
                        "scrolled_px": med("scrolledPx"),
                        "scrolled_px_all": [r.get("scrolledPx") for r in rows],
                        "boundary_events": [r.get("boundary") for r in rows],
                    }
                except Exception as exc:  # noqa: BLE001
                    results["arms"][name] = {"why": why, "failed": repr(exc)}
                    info(f"  FAILED {exc!r}")
                finally:
                    ctx.close()
            browser.close()
        out = OUT / f"{LABEL}.json"
        out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
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
        # The JSON above is written either way, because a partial experiment is still evidence.
        # The EXIT CODE is not: an arm that threw produced no comparison, and a run where every
        # arm threw produced nothing at all. Reporting that as success is how a broken experiment
        # gets quoted as a result.
        failed = sorted(n for n, r in results["arms"].items() if "failed" in r)
        if failed:
            info(f"FAILED arms: {', '.join(failed)}")
            return 1
        if not results["arms"]:
            info("no arms ran")
            return 1
        # Every arm is a comparison against `nothing`, and that comparison only means anything if
        # the gesture being compared is the same gesture. Per-row validation above rejects a
        # scroll that did not happen; this rejects a set of scrolls that happened differently,
        # which reads as a predecessor cost and is not one.
        skewed = skewed_arms(results["arms"])
        if skewed:
            info("arms did not scroll a comparable distance: " + "; ".join(skewed))
            return 1
    finally:
        if vite is not None:
            stop_process(vite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
