# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The action bar is WAITED for, not sampled for once.

WHAT THIS PROTECTS, and it is a liveness property rather than a timing one. Studio mounts the
assistant action bar with `hideWhenRunning` (studio/frontend/src/components/assistant-ui/
thread.tsx), so while a turn is generating there is no Copy, no Delete and no More anywhere in the
tree. Four scene actions need one of those controls, and every film schedules them after a
`send_turn` on the arithmetic that the follow-up drains in FOLLOW_UP_CHARS over the field cadence.

That arithmetic is a FLOOR, not an estimate. It assumes the pacer's cadence is the binding
constraint, and at the 100K rung the renderer is: measured over six 100K cells, the follow-up kept
streaming for 4.4 to 4.7 s after the send window closed, against a 4.59 s nominal drain and a
5.3 s gap to the slot. The reply therefore settles within a few hundred milliseconds of the slot
opening -- sometimes before it, sometimes after. The payload of the studiobench CI run that failed
the liveness gate has the "after" case: one more SSE chunk arrived INSIDE the `message_menu`
window and the reply stopped growing 71 characters later, in that same window.

Sampled once, that third of a second reads as `NOT RUN -- no More button on the last assistant
message`, the liveness gate exits 1, and the report names a missing control when what happened is
a clock read too early. Both branches of the pull request that first hit it were red for it, and
so was the branch it was cut from, which is what says this is the film's packing rather than
anything either branch changed.

WHY NODE AND THE REAL SOURCES. `MENU_JS` is the string that ships inside `scene/actions.py` and
`waitForActionButton` is the function that ships inside `scene/dom.js`; a Python re-implementation
of either would pass forever while the shipped pair drifted. So node runs both, against a shim of
the handful of DOM globals they touch, with the ACTION BAR ARRIVING LATE -- which is the one thing
that cannot be shimmed away, because it is the thing under test. No browser, no Studio, and if
node is missing the test SKIPS rather than passing on a substitute.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.scene.actions import ACTION_BAR_WAIT_MS, MENU_JS  # noqa: E402

DOM_JS = Path(__file__).resolve().parents[1] / "dom.js"

#: The shim mounts the bar this long after the run starts, standing in for a follow-up turn whose
#: last chunks land just after the slot opens. Comfortably inside `ACTION_BAR_WAIT_MS` and
#: comfortably outside anything a single sample could catch.
LATE_MOUNT_MS = 400

#: The mount deadline is armed by the action's own first look (see `HARNESS_JS`), so it can only
#: start level with or after `waitForActionButton`'s clock and `waitedMs` cannot undershoot
#: `LATE_MOUNT_MS` by any elapsed time at all. What is left is dom.js rounding `waitedMs` to a
#: tenth. This is an allowance for clock granularity, not a tolerance for load: the wait ends at
#: the first sample taken at or after the bar mounts, and a slow machine can only push that sample
#: LATER. Nothing here scales with `LATE_MOUNT_MS`.
MOUNT_CLOCK_SLACK_MS = 1

HARNESS_JS = r"""
const fs = require("fs");
const domSrc = fs.readFileSync(process.argv[2], "utf8");
const menuSrc = fs.readFileSync(process.argv[3], "utf8");
const mountAfterMs = Number(process.argv[4]);   // Infinity => the bar never mounts
const waitMs = Number(process.argv[5]);

// ── the smallest DOM these two files actually touch ────────────────────────────────────────────
//
// A node matches a selector when the selector is in its `sel` list. That is enough: dom.js reads
// the app through a fixed, short list of selectors, and a real matcher here would only be a second
// implementation to get wrong.
const node = (sel, attrs, kids) => {
  const self = {
    sel: sel,
    kids: kids || [],
    getAttribute: (k) => (attrs && k in attrs ? attrs[k] : null),
    textContent: (attrs && attrs.text) || "",
    dispatchEvent: (ev) => { if (attrs && attrs.on) attrs.on(ev); return true; },
  };
  const all = (s) => {
    const out = [];
    for (const k of self.kids) {
      if (k.sel.indexOf(s) >= 0) out.push(k);
      for (const g of k.querySelectorAll(s)) out.push(g);
    }
    return out;
  };
  self.querySelectorAll = all;
  self.querySelector = (s) => all(s)[0] || null;
  return self;
};

// ── one clock, one origin ──────────────────────────────────────────────────────────────────────
//
// The bar mounts `mountAfterMs` after THE ACTION FIRST LOOKS FOR IT, and the deadline is armed by
// that first look rather than by this script loading. The two instants are not the same: between
// them sit two readFileSync calls, the dom.js eval, two Function compiles and V8's lazy compile of
// the action body on its first call, all of it charged to whatever else the machine is doing.
// Armed at load, the deadline would already be part-spent by the time `waitForActionButton` set
// its own `started`, and the `waitedMs` measured from THAT origin would come back short by the
// setup cost -- a number about the runner's load rather than about the wait.
//
// Arming here instead puts both clocks on one origin, and puts them there in the only order that
// is safe. `waitForActionButton` stamps `started`, hovers, and only then reads a control; that
// read is the first call below, so the mount deadline can start level with the action's clock or
// a hair after it, never before. `waitedMs` therefore cannot come in under `mountAfterMs`, and a
// loaded machine can only push the sample that ends the wait LATER.
let started = null;
const mounted = () => {
  if (started === null) started = performance.now();
  return performance.now() - started >= mountAfterMs;
};

const state = { menuOpen: false, hovers: 0 };

const moreButton = node(["button"], {
  "aria-label": "More",
  on: (ev) => {
    // Radix opens on pointerdown and this is what the action dispatches; anything else must not
    // open it, because an element.click() that "worked" here would hide a real regression.
    if (ev && ev.type === "pointerdown") { state.menuOpen = true; notify(); }
  },
});
const bar = node(["button", ".aui-assistant-action-bar-root"], {}, [moreButton]);
bar.sel = [".aui-assistant-action-bar-root"];

const message = {
  sel: ['[data-role="assistant"]'],
  getAttribute: () => null,
  textContent: "the reply",
  dispatchEvent: (ev) => { if (ev && ev.type === "pointerover") state.hovers += 1; return true; },
  querySelectorAll: (s) => (mounted() ? bar.querySelectorAll(s).concat(
    bar.sel.indexOf(s) >= 0 ? [bar] : []) : []),
  querySelector: (s) => (mounted() ? (bar.sel.indexOf(s) >= 0 ? bar
    : bar.querySelector(s)) : null),
};

const menuItems = [node([".aui-action-bar-more-item"], {}), node([".aui-action-bar-more-item"], {})];
const menu = node([".aui-action-bar-more-content"], {}, menuItems);

const body = { sel: ["body"], querySelector: () => null, querySelectorAll: () => [] };
const document = {
  body: body,
  // `isRunning()` is read off the composer's Stop button. It stays up until the bar mounts,
  // exactly as it does in the app: both are driven by the same "the thread is generating" state.
  querySelector: (s) => document.querySelectorAll(s)[0] || null,
  querySelectorAll: (s) => {
    if (s === '[data-role="assistant"]') return [message];
    if (s === 'button[aria-label="Stop generating"]') return mounted() ? [] : [{}];
    if (s === ".aui-action-bar-more-content") return state.menuOpen ? [menu] : [];
    if (s === ".aui-action-bar-more-item") return state.menuOpen ? menuItems : [];
    return [];
  },
  dispatchEvent: (ev) => {
    if (ev && ev.key === "Escape") { state.menuOpen = false; notify(); }
    return true;
  },
};

// MutationObserver delivers on a microtask checkpoint after the mutation; the action relies on the
// flag being set by the time its next paint resolves, so notifying the callbacks when the shimmed
// DOM changes is the same contract.
const observers = [];
class MutationObserver {
  constructor(cb) { this.cb = cb; }
  observe() { observers.push(this); }
  disconnect() { const i = observers.indexOf(this); if (i >= 0) observers.splice(i, 1); }
}
const notify = () => { for (const o of observers.slice()) o.cb(); };

class PointerEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }
class KeyboardEvent { constructor(type, init) { this.type = type; Object.assign(this, init || {}); } }
const getComputedStyle = () => ({ pointerEvents: "auto" });

const window = {};
// The real one is frames.js's rAF promise. A timer is the same shape and does not need a compositor.
window.__sbNextPaint = () => new Promise((r) => setTimeout(r, 16));
// dom.js's follow sampler survives a navigation by writing itself to sessionStorage on `pagehide`,
// so installing it REGISTERS a listener. Nothing here ever fires it -- the registration only has to
// not throw -- and the read side is already inside its own try/catch, which is why sessionStorage
// itself does not need shimming.
window.addEventListener = () => {};

(new Function("window", "document", "PointerEvent", domSrc))(window, document, PointerEvent);

const menuFn = (new Function(
  "window", "document", "PointerEvent", "KeyboardEvent", "MutationObserver", "getComputedStyle",
  "return (" + menuSrc + ")",
))(window, document, PointerEvent, KeyboardEvent, MutationObserver, getComputedStyle);

menuFn({ timeoutMs: 4000, waitForButtonMs: waitMs }).then((out) => {
  out.hovers = state.hovers;
  console.log(JSON.stringify(out));
  process.exit(0);
}, (err) => { console.error(String((err && err.stack) || err)); process.exit(1); });
"""


def _node() -> str:
    exe = shutil.which("node") or shutil.which("nodejs")
    if exe is None:
        pytest.skip(
            "node is not installed, so the shipped dom.js and MENU_JS could not be evaluated; "
            "this is NOT MEASURED rather than passing"
        )
    return exe


def run_menu(mount_after_ms: float, wait_ms: int = ACTION_BAR_WAIT_MS) -> dict:
    """Run the shipped `MENU_JS` against the shipped `dom.js` with the bar mounting late."""
    exe = _node()
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.js"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        menu = Path(tmp) / "menu.js"
        menu.write_text(MENU_JS, encoding = "utf-8")
        got = subprocess.run(
            [exe, str(harness), str(DOM_JS), str(menu), str(mount_after_ms), str(wait_ms)],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    if got.returncode != 0:
        raise AssertionError(f"the MENU_JS harness failed: {got.stderr.strip()[-1200:]}")
    return json.loads(got.stdout)


def test_a_control_that_arrives_late_is_waited_for_rather_than_reported_missing():
    """THE REGRESSION. A bar that mounts 400 ms into the slot must produce a run, not a NOT RUN.

    Sampled once, this is the studiobench CI failure exactly: `no More button on the last
    assistant message`, from a follow-up turn that was still arriving when the slot opened.
    """
    out = run_menu(LATE_MOUNT_MS)
    assert out["ran"] is True, out
    # Still a liveness assertion, not a timing one: the bar was NOT there when the wait began, and
    # the action stayed until it arrived. The bound is tight because the harness arms the mount
    # deadline off the same clock and the same instant the action starts looking, so anything short
    # of `LATE_MOUNT_MS` means the wait ended early -- a control that was there all along.
    assert out["waitedMs"] >= LATE_MOUNT_MS - MOUNT_CLOCK_SLACK_MS, out
    assert out["waitedMs"] < ACTION_BAR_WAIT_MS, out
    # The menu really opened and closed on the control that was waited for, so the wait produced a
    # measurement rather than merely a truthy `ran`.
    assert out["openMs"] is not None and out["closeMs"] is not None, out
    assert out["items"] == 2, out


def test_the_wait_is_bounded_and_a_control_that_never_appears_still_reports_not_run():
    """The gate keeps its teeth. Waiting must not turn a genuinely absent control into a pass."""
    out = run_menu(float("inf"), wait_ms = 300)
    assert out["ran"] is False, out
    assert out["waitedMs"] >= 250, out
    # The reason has to separate the two cases, because they are different bugs: a reply that had
    # not settled, and a control that is not there at all.
    assert "STILL GENERATING" in out["reason"], out
    assert out["running"] is True, out


def test_a_settled_reply_costs_the_action_nothing():
    """The normal case pays no wait at all, so the fix cannot show up in the measured latency."""
    out = run_menu(0)
    assert out["ran"] is True, out
    assert out["waitedMs"] < 50, out
    # Hovered before the control is read: the bar unmounts on every message that is not hovered,
    # so a read without the hover is a read of a tree the control was never in.
    assert out["hovers"] >= 1, out
