# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs the REASONING_JS settle loop in node, because it decides when the span census is read."""

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
SPANS_BEFORE = 44075
SPANS_SETTLED = 74250

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
    """The span census must not be read on the frame the open state flips, before its spans have mounted."""
    out = run_settle(flip_frame = 6, mount_done_frame = 40)
    assert out["spansOpen"] != SPANS_BEFORE, (
        "the span census was read on the frame the state flipped, before the content it counts "
        "had mounted. That is the defect the settling fix exists to remove, reproduced inside "
        "the fix itself."
    )
    assert out["spansOpen"] == SPANS_SETTLED
    assert out["openCensored"] is False
    assert out["openStateReachedMs"] < out["openMs"]


def test_the_streak_requires_that_many_frames_after_the_flip():
    """The quiet window is measured from the flip, so the read lands after the mount finishes."""
    out = run_settle(flip_frame = 6, mount_done_frame = 40)
    assert out["openFrames"] >= 40 + SETTLE_QUIET_FRAMES
    assert out["quietFramesRequired"] == SETTLE_QUIET_FRAMES


def test_a_census_that_never_goes_quiet_is_withheld_with_a_reason():
    """Silence beats a confident wrong answer: no number, and a reason naming the budget."""
    # Budget is 8000ms / 16ms = 500 frames.
    out = run_settle(flip_frame = 6, mount_done_frame = 100_000)
    assert out["spansOpen"] is None
    assert out["openMs"] is None
    assert out["openCensored"] is True
    assert "still changing" in out["openCensoredReason"]
    assert out["openStateReachedMs"] is not None


def test_a_state_that_is_never_reached_says_so_instead():
    """The other censoring reason, so the two failures are not reported as one."""
    out = run_settle(flip_frame = 100_000, mount_done_frame = 100_001)
    assert out["openCensored"] is True
    assert "never reached" in out["openCensoredReason"]
    assert out["openStateReachedMs"] is None


def test_a_page_that_is_already_settled_still_returns_promptly():
    """An already-settled page must still return promptly and uncensored, not be refused."""
    out = run_settle(flip_frame = 1, mount_done_frame = 2, spans_static = True)
    assert out["openCensored"] is False
    assert out["spansOpen"] == SPANS_BEFORE
    assert out["openFrames"] <= 1 + SETTLE_QUIET_FRAMES + 1


def test_losing_the_state_restarts_the_streak():
    """Dropping below the target restarts the quiet streak, so frames it did not hold are not banked."""
    out = run_settle(
        flip_frame = 2,
        mount_done_frame = 3,
        spans_static = True,
        lose_state_after = 1,
        lose_state_for = 6,
    )
    assert out["openCensored"] is False
    assert out["openFrames"] >= 2 + 1 + 6


def test_the_timing_includes_the_click_dispatch_it_names():
    """open_ms must include the click dispatch as well as the settle wait, or slower handlers look
    faster."""
    out = run_settle(flip_frame = 2, mount_done_frame = 3, spans_static = True, click_ms = 40.0)
    dispatch = PANES * 40.0
    assert out["openDispatchMs"] == dispatch
    assert out["openMs"] >= dispatch, (
        "open_ms excluded the click dispatch, so it names an operation larger than the one it "
        "measures. That is this branch's own defect, committed by the fix for it."
    )
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


# Collapse keeps children mounted until the exit animation ends (Radix Presence, or the grid arm
# until transitionend / 250 ms backstop), so a frozen span count is not a finished close.


def test_a_collapse_is_not_settled_while_its_panes_are_still_mounted():
    """A collapse is not settled while its panes are still mounted, so close_ms waits for the unmount."""
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 12)
    assert out["closeCensored"] is False
    assert out["closeFrames"] >= 12 + SETTLE_QUIET_FRAMES, (
        f"the close settle returned after {out['closeFrames']} frames, before the panes it had "
        f"just collapsed unmounted at frame 12. close_ms then measures the state flip plus a "
        f"quiet streak the teardown had not begun to disturb."
    )


def test_the_close_bias_does_not_depend_on_the_paint_interval():
    """Close timing reaches the unmount at any paint interval, so slow and fast pages are comparable."""
    fast_page = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 20)
    slow_page = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 2)
    assert fast_page["closeFrames"] >= 20 + SETTLE_QUIET_FRAMES
    assert slow_page["closeFrames"] >= 2 + SETTLE_QUIET_FRAMES
    assert fast_page["closeFrames"] - 20 == slow_page["closeFrames"] - 2


def test_a_collapse_that_never_tears_down_is_censored_and_says_which_half_failed():
    """A collapse that never tears down is censored, and the reason names the still-mounted panes."""
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 100_000)
    assert out["closeCensored"] is True
    assert out["closeMs"] is None
    assert "still mounted" in out["closeCensoredReason"]


def test_an_instant_teardown_still_returns_promptly():
    """The control. The fix must not turn the cheap case into a censored one."""
    out = run_settle(flip_frame = 1, mount_done_frame = 2, close_unmount_frame = 0)
    assert out["closeCensored"] is False
    assert out["closeFrames"] <= 1 + SETTLE_QUIET_FRAMES + 1
