# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The settle loop inside `REASONING_JS`, executed rather than described.

WHY THIS FILE EXISTS. The settling fix is a change to a piece of JAVASCRIPT, and the Python tests
beside it drive `reasoning_toggle` through a stubbed `evaluate` that returns a canned reply. Those
tests pin how the Python wrapper reads a reply; they cannot see the loop that produces one. So the
loop shipped with a defect that inverted the entire fix, and every test still passed:

    `quiet` counted consecutive frames on which the span census had not moved, starting from the
    FIRST frame of the wait. Whenever the panes needed `quietFrames` or more frames to reach the
    open state -- with the census necessarily static through them, because the content it would
    count has not been revealed yet -- the streak was already satisfied on the very frame the state
    flipped, and the census was read exactly where the unfixed code read it.

That is not a corner. It is the normal shape at the rungs this was written for: the catalogue's own
500K reading is "the open count reached 16 after 10440ms", which is hundreds of static frames. Run
against a page whose state flips at frame 6 and whose spans keep mounting to frame 40, the shipped
loop returned 44,075 with `censored: false` -- the withdrawn number, reported confidently, out of
the code whose purpose is to stop it.

WHAT IS AND IS NOT COVERED HERE. `reasoningTriggers` and `reasoningOpenCount` are shimmed, so this
does not test that `dom.js` finds the right elements. What it tests is the part that decides WHEN
to read, which is the whole of the fix and the whole of the defect. The clock and the paint pump
are shimmed too, so a frame here is a step rather than 16 ms of wall time; the loop's own budget
arithmetic runs unmodified against them.

If node is missing the tests SKIP rather than passing on a Python re-implementation, which would be
a second copy of the instrument to get wrong.
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

from studiobench.scene.actions import REASONING_JS, SETTLE_QUIET_FRAMES  # noqa: E402

PANES = 16
SPANS_BEFORE = 44075  # what the grid arm read at the state flip, and had to stop reading
SPANS_SETTLED = 74250  # what both arms read once the mount finished

HARNESS_JS = r"""
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const cfg = JSON.parse(process.argv[3]);

// ── a page whose state flip and whose content mount are SEPARATE events ────────────────────────
//
// That separation is the entire subject. `data-state` flips when the Collapsible's state changes;
// the spans it reveals mount later, and how much later depends on the collapse mechanism, which is
// the thing an A/B across a collapse change is comparing.
let frame = 0;
let now = 0;
const PAINT_MS = 16;

// Frames are counted per settle() call: the close phase gets its own clock, as it does in the app.
let phaseStart = 0;
const f = () => frame - phaseStart;

let closing = false;
let triggerReads = 0;

// THE CLOSE PHASE HAS THE SAME SEPARATION, RUNNING THE OTHER WAY. `data-state` flips on the click;
// the children stay in the document until the exit animation ends, so the census does not move at
// all for `closeUnmountFrame` frames and then drops in one step. `closeUnmountFrame = 0` is the
// page the shim used to model, where the teardown is instantaneous.
const spanCount = () => {
  if (closing) {
    return f() >= cfg.closeUnmountFrame ? cfg.spansBefore : cfg.spansSettled;
  }
  if (cfg.spansStatic) return cfg.spansBefore;
  if (f() < cfg.flipFrame) return cfg.spansBefore;           // STATIC before the flip
  if (f() >= cfg.mountDoneFrame) return cfg.spansSettled;
  const t = (f() - cfg.flipFrame) / (cfg.mountDoneFrame - cfg.flipFrame);
  return Math.round(cfg.spansBefore + t * (cfg.spansSettled - cfg.spansBefore));
};

// How many reasoning panes still have their content in the document. Open panes are mounted; a
// closed one stays mounted until its exit animation finishes.
const mountedCount = () => {
  if (closing) return f() >= cfg.closeUnmountFrame ? 0 : cfg.panes;
  return f() >= cfg.flipFrame ? cfg.panes : 0;
};

const openCount = () => {
  if (closing) return 0;
  if (f() < cfg.flipFrame) return 0;
  // `loseStateAfter` drops the count back below `want` once it has been reached, so a run that
  // oscillates around the target cannot bank quiet frames it never held.
  if (cfg.loseStateAfter && f() >= cfg.flipFrame + cfg.loseStateAfter
      && f() < cfg.flipFrame + cfg.loseStateAfter + cfg.loseStateFor) {
    return cfg.panes - 1;
  }
  return cfg.panes;
};

globalThis.performance = { now: () => now };
globalThis.document = {
  querySelectorAll: (sel) => ({ length: sel === "pre span" ? spanCount() : 0 }),
};
globalThis.window = {
  __sbNextPaint: async () => { frame += 1; now += PAINT_MS; },
  __sb: { dom: {
    // The click handler costs real time, because it is the app's own handler running
    // synchronously and it is the first half of what "opening the panes" means.
    // THE SECOND CALL IS THE CLOSE. `REASONING_JS` reads the triggers once to open and once more
    // to collapse, so asking for them again is the shim's signal to enter its close phase and
    // restart the per-phase frame clock. It used to be a flag that nothing ever set, which left
    // every close settle waiting for an open count that never fell and censoring itself -- so the
    // close direction was, in effect, not exercised at all.
    reasoningTriggers: () => {
      triggerReads += 1;
      if (triggerReads === 2) { closing = true; phaseStart = frame; }
      return Array.from({ length: cfg.panes }, () => ({
        click: () => { now += cfg.clickMs; },
      }));
    },
    reasoningOpenCount: () => openCount(),
    reasoningContentMounted: () => mountedCount(),
  } },
};

const fn = eval("(" + src.trim() + ")");

fn([cfg.timeoutMs, cfg.quietFrames]).then((out) => {
  console.log(JSON.stringify(out));
  process.exit(0);
}, (err) => { console.error(String((err && err.stack) || err)); process.exit(1); });
"""


def _node() -> str:
    exe = shutil.which("node") or shutil.which("nodejs")
    if exe is None:
        pytest.skip(
            "node is not installed, so the shipped REASONING_JS could not be evaluated; "
            "this is NOT MEASURED rather than passing"
        )
    return exe


def run_settle(
    flip_frame: int,
    mount_done_frame: int,
    *,
    timeout_ms: int = 8000,
    quiet_frames: int = SETTLE_QUIET_FRAMES,
    spans_static: bool = False,
    lose_state_after: int = 0,
    lose_state_for: int = 0,
    click_ms: float = 0.0,
    close_unmount_frame: int = 0,
) -> dict:
    """Run the SHIPPED `REASONING_JS` against a page with a late flip and a later mount."""
    exe = _node()
    cfg = {
        "panes": PANES,
        "clickMs": click_ms,
        "flipFrame": flip_frame,
        "mountDoneFrame": mount_done_frame,
        "spansBefore": SPANS_BEFORE,
        "spansSettled": SPANS_SETTLED,
        "spansStatic": spans_static,
        "loseStateAfter": lose_state_after,
        "loseStateFor": lose_state_for,
        "closeUnmountFrame": close_unmount_frame,
        "timeoutMs": timeout_ms,
        "quietFrames": quiet_frames,
    }
    with tempfile.TemporaryDirectory() as tmp:
        harness = Path(tmp) / "harness.js"
        harness.write_text(HARNESS_JS, encoding = "utf-8")
        js = Path(tmp) / "reasoning.js"
        js.write_text(REASONING_JS, encoding = "utf-8")
        got = subprocess.run(
            [exe, str(harness), str(js), json.dumps(cfg)],
            capture_output = True,
            text = True,
            timeout = 120,
        )
    assert got.returncode == 0, f"node failed:\n{got.stderr}"
    return json.loads(got.stdout.strip().splitlines()[-1])


def test_a_slow_state_flip_does_not_bank_quiet_frames_before_it():
    """THE REGRESSION. The census must not be read on the frame the state flips.

    The panes take six frames to reach the open state and the census is static through all of
    them, so a streak counted from frame one is already satisfied when the state arrives. Reading
    there gives 44,075 -- the number this whole change exists to retract -- and gives it with
    `censored: false`, which is worse than giving nothing.
    """
    out = run_settle(flip_frame = 6, mount_done_frame = 40)
    assert out["spansOpen"] != SPANS_BEFORE, (
        "the span census was read on the frame the state flipped, before the content it counts "
        "had mounted. That is the defect the settling fix exists to remove, reproduced inside "
        "the fix itself."
    )
    assert out["spansOpen"] == SPANS_SETTLED
    assert out["openCensored"] is False
    # The flip and the settled read are different moments, and both are reported so a reader can
    # see how much of open_ms was spent after the state arrived.
    assert out["openStateReachedMs"] < out["openMs"]


def test_the_streak_requires_that_many_frames_after_the_flip():
    """The quiet window is measured from the flip, so the read lands after the mount finishes."""
    out = run_settle(flip_frame = 6, mount_done_frame = 40)
    # The census stops moving at frame 40; the streak then needs SETTLE_QUIET_FRAMES more.
    assert out["openFrames"] >= 40 + SETTLE_QUIET_FRAMES
    assert out["quietFramesRequired"] == SETTLE_QUIET_FRAMES


def test_a_census_that_never_goes_quiet_is_withheld_with_a_reason():
    """Silence beats a confident wrong answer: no number, and a reason naming the budget."""
    # Spans still climbing when the budget runs out: 8000ms / 16ms = 500 frames.
    out = run_settle(flip_frame = 6, mount_done_frame = 100_000)
    assert out["spansOpen"] is None
    assert out["openMs"] is None
    assert out["openCensored"] is True
    assert "still changing" in out["openCensoredReason"]
    # It reached the state, so the reason must say THAT rather than blaming the open count.
    assert out["openStateReachedMs"] is not None


def test_a_state_that_is_never_reached_says_so_instead():
    """The other censoring reason, so the two failures are not reported as one."""
    out = run_settle(flip_frame = 100_000, mount_done_frame = 100_001)
    assert out["openCensored"] is True
    assert "never reached" in out["openCensoredReason"]
    assert out["openStateReachedMs"] is None


def test_a_page_that_is_already_settled_still_returns_promptly():
    """The refusal must not swallow the easy case, or every fast reading disappears.

    A page whose panes flip on the first frame and whose census never moves is the cheapest
    possible reading. It has to come back quickly and uncensored, or the fix has traded a wrong
    number for no number at all.
    """
    out = run_settle(flip_frame = 1, mount_done_frame = 2, spans_static = True)
    assert out["openCensored"] is False
    assert out["spansOpen"] == SPANS_BEFORE
    assert out["openFrames"] <= 1 + SETTLE_QUIET_FRAMES + 1


def test_losing_the_state_restarts_the_streak():
    """A count that oscillates around the target must not bank the frames it did not hold.

    Without this the loop could reach `want`, drop below it while the DOM churned, and still
    return on a streak accumulated across the gap -- reading a document that was mid-change.
    """
    out = run_settle(
        flip_frame = 2,
        mount_done_frame = 3,
        spans_static = True,
        lose_state_after = 1,
        lose_state_for = 6,
    )
    assert out["openCensored"] is False
    # It cannot have returned during the window where the count was below `want`.
    assert out["openFrames"] >= 2 + 1 + 6


def test_the_timing_includes_the_click_dispatch_it_names():
    """`open_ms` must cover the clicks, not just the wait that follows them.

    `t.click()` runs the app's own handler synchronously, and on a long thread that is React state
    plus layout, once per pane. It is the first half of opening the panes. A timing that starts
    after the whole loop has returned is a smaller number for the same action, and it moves for a
    reason a reader cannot see: an arm that makes the handler slower and the settle faster would
    look unchanged, or better.

    With a 40 ms handler on 16 panes the settle alone reads 96 ms against 640 ms of dispatch it had
    just performed.
    """
    out = run_settle(flip_frame = 2, mount_done_frame = 3, spans_static = True, click_ms = 40.0)
    dispatch = PANES * 40.0
    assert out["openDispatchMs"] == dispatch
    assert out["openMs"] >= dispatch, (
        "open_ms excluded the click dispatch, so it names an operation larger than the one it "
        "measures. That is this branch's own defect, committed by the fix for it."
    )
    # And the two halves are reported apart, because they answer different questions.
    assert out["openMs"] == pytest.approx(out["openDispatchMs"] + out["openSettleMs"], abs = 0.2)
    assert out["openSettleMs"] < out["openMs"]


def test_the_state_reached_mark_shares_the_timing_origin():
    """`open_state_reached_ms` is quoted against `open_ms`, so it cannot start from a later zero."""
    out = run_settle(flip_frame = 3, mount_done_frame = 20, click_ms = 25.0)
    dispatch = PANES * 25.0
    assert out["openStateReachedMs"] >= dispatch, (
        "the state-reached mark was measured from the settle's start while open_ms was measured "
        "from before the clicks, so subtracting one from the other yields a phantom interval"
    )
    assert out["openStateReachedMs"] <= out["openMs"]


# ── the close direction ─────────────────────────────────────────────────────────────────────────
#
# The same defect the quiet streak fixes for the open direction, arriving from the other side.
# `data-state` flips on the click, and BOTH collapse mechanisms keep the children in the document
# until the exit animation ends: Radix's `Presence` until `animationend`, the grid arm's
# `UnmeasuredCollapsibleContent` until the `grid-template-rows` `transitionend` or its 250 ms
# backstop. So `pre span` is frozen at its open value for that whole window, and a streak that only
# asks whether the census has stopped moving is satisfied by a census that has not started.


def test_a_collapse_is_not_settled_while_its_panes_are_still_mounted():
    """THE REGRESSION. `close_ms` must not name the state flip plus four frames.

    Twelve frames of exit animation is 192 ms at this shim's 16 ms paint, which is the duration
    both arms actually run. Four quiet frames fit inside it with room to spare, so the unfixed loop
    returns while every span it would have counted is still in the document -- a pre-settled point,
    reported as a measurement of the collapse.
    """
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 12)
    assert out["closeCensored"] is False
    assert out["closeFrames"] >= 12 + SETTLE_QUIET_FRAMES, (
        f"the close settle returned after {out['closeFrames']} frames, before the panes it had "
        f"just collapsed unmounted at frame 12. close_ms then measures the state flip plus a "
        f"quiet streak the teardown had not begun to disturb."
    )


def test_the_close_bias_does_not_depend_on_the_paint_interval():
    """WHY IT IS A COMPARISON PROBLEM RATHER THAN AN OFFSET.

    Whether the old streak ended before or after the teardown depended on the paint interval
    against the animation duration -- and the paint interval is exactly what differs between the
    arms and the rungs this instrument compares. A slow page whose teardown lands inside the streak
    was measured to the unmount; a fast one was not. Both must now be measured to the unmount, so
    the two readings mean the same thing.
    """
    fast_page = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 20)
    slow_page = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 2)
    assert fast_page["closeFrames"] >= 20 + SETTLE_QUIET_FRAMES
    assert slow_page["closeFrames"] >= 2 + SETTLE_QUIET_FRAMES
    # Both readings are the same quantity: the teardown, plus the streak that proves it is over.
    assert fast_page["closeFrames"] - 20 == slow_page["closeFrames"] - 2


def test_a_collapse_that_never_tears_down_is_censored_and_says_which_half_failed():
    """Silence beats a confident wrong answer here too, and the reason has to be usable.

    A pane that reports closed and never unmounts is a different finding from one that never
    closed. Reported as "the open count never reached 0" it would send a reader to the wrong half
    of the app.
    """
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 100_000)
    assert out["closeCensored"] is True
    assert out["closeMs"] is None
    assert "still mounted" in out["closeCensoredReason"]


def test_an_instant_teardown_still_returns_promptly():
    """The control. The fix must not turn the cheap case into a censored one."""
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 0)
    assert out["closeCensored"] is False
    assert out["closeFrames"] <= 1 + SETTLE_QUIET_FRAMES + 1
