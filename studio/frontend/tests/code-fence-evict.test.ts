// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DWELL_MS,
  EVICT_BUDGET,
  HOLD_BAND,
  PASS_INTERVAL_MS,
  REACH_BAND,
  SHIP_DEFAULT,
  bandDistance,
  evictionCannotImmediatelyRelatch,
  fenceIsEvictable,
  nextPassDelayMs,
  passIsDue,
  planEviction,
  relatchGapPx,
  resolveFenceEvictMode,
  withinBand,
  type EvictCandidate,
} from "../src/components/assistant-ui/code-fence-evict.ts";

/**
 * The eviction decision table, RUN rather than described.
 *
 * This feature can unmount a highlighted subtree the reader has already been shown, so what has to
 * hold is that it is OFF unless somebody turned it on and that its geometry cannot produce churn.
 * Neither is evidence unless the rows execute.
 */

/* ---------------------------------------------------------------- the flag */

test("an install that has never set the flag gets nothing", () => {
  assert.equal(SHIP_DEFAULT, "off", "PRECONDITION: this feature ships off");
  assert.equal(resolveFenceEvictMode(undefined, ""), "off");
  assert.equal(resolveFenceEvictMode(null, ""), "off");
});

test("the build flag turns it on, and only on the values that mean on", () => {
  assert.equal(resolveFenceEvictMode(undefined, "evict"), "evict");
  assert.equal(resolveFenceEvictMode(undefined, "1"), "evict");
  assert.equal(resolveFenceEvictMode(undefined, "off"), "off");
  assert.equal(resolveFenceEvictMode(undefined, "0"), "off");
  // A mistyped value must not land on "evict".
  assert.equal(resolveFenceEvictMode(undefined, "evcit"), "off");
  assert.equal(resolveFenceEvictMode(undefined, "true"), "off");
});

test("the runtime global overrides the build flag in BOTH directions", () => {
  // PRECONDITION: these two builds disagree, so the rows below are about the override itself.
  assert.equal(resolveFenceEvictMode(undefined, "evict"), "evict");
  assert.equal(resolveFenceEvictMode(undefined, ""), "off");

  assert.equal(resolveFenceEvictMode("off", "evict"), "off");
  assert.equal(resolveFenceEvictMode(false, "evict"), "off");
  assert.equal(resolveFenceEvictMode("evict", ""), "evict");
  assert.equal(resolveFenceEvictMode(true, ""), "evict");
  assert.equal(resolveFenceEvictMode("1", ""), "evict");
});

test("a non-string, non-boolean runtime value falls through to the build flag", () => {
  assert.equal(resolveFenceEvictMode({}, "evict"), "evict");
  assert.equal(resolveFenceEvictMode(0, ""), "off");
});

/* ------------------------------------------------------------ the geometry */

/*
 * `withinBand(rect, band, 1)` has to be the SAME predicate `inBand` in `code-fence-defer.tsx`
 * computes, or the anti-churn invariant means nothing. That file's arithmetic is reproduced here
 * literally and the two are run against each other over a grid.
 */
const inBandAsShipped = (
  rect: { top: number; bottom: number },
  band: { top: number; height: number },
): boolean =>
  rect.bottom > band.top - band.height
  && rect.top < band.top + band.height * 2;

test("the reach band is byte-for-byte the predicate code-fence-defer.tsx already uses", () => {
  const band = { top: 120, height: 800 };
  let agreed = 0;
  let sawTrue = false;
  let sawFalse = false;
  for (let top = -3000; top <= 3000; top += 37) {
    for (const height of [0, 1, 18, 400, 2400]) {
      const rect = { top, bottom: top + height };
      const mine = withinBand(rect, band, REACH_BAND);
      assert.equal(mine, inBandAsShipped(rect, band), `top=${top} h=${height}`);
      agreed += 1;
      if (mine) sawTrue = true;
      else sawFalse = true;
    }
  }
  // CONTROL: a grid that only ever produced one answer would agree trivially.
  assert.ok(sawTrue && sawFalse, "the grid must exercise both answers");
  assert.ok(agreed > 500, "the grid must actually have rows");
});

test("the hold band is strictly wider than the reach band", () => {
  assert.ok(HOLD_BAND > REACH_BAND, `${HOLD_BAND} must exceed ${REACH_BAND}`);
  const band = { top: 0, height: 500 };
  // A rect that the reach band accepts must be accepted by the hold band too, for every rect.
  for (let top = -6000; top <= 6000; top += 41) {
    const rect = { top, bottom: top + 200 };
    if (withinBand(rect, band, REACH_BAND)) {
      assert.ok(withinBand(rect, band, HOLD_BAND), `reach implies hold at ${top}`);
    }
  }
});

test("nothing this file evicts can be re-latched on the next frame", () => {
  const band = { top: 64, height: 720 };
  let evicted = 0;
  for (let top = -8000; top <= 8000; top += 29) {
    for (const height of [0, 20, 300, 1500]) {
      const rect = { top, bottom: top + height };
      assert.ok(
        evictionCannotImmediatelyRelatch(rect, band),
        `churn window at top=${top} h=${height}`,
      );
      if (!withinBand(rect, band, HOLD_BAND)) evicted += 1;
    }
  }
  // CONTROL: the invariant is vacuously true if nothing is ever outside the hold band.
  assert.ok(evicted > 100, `the grid must reach outside the hold band, saw ${evicted}`);
});

test("CONTROL: the invariant is not a tautology, it fails for a hold band inside the reach band", () => {
  // Without this row the test above proves nothing: `withinBand(hold) || !withinBand(reach)` is
  // true for every rect whenever hold >= reach. Driving it the other way shows it can report false.
  const band = { top: 0, height: 100 };
  const rect = { top: 250, bottom: 270 };
  assert.equal(evictionCannotImmediatelyRelatch(rect, band, 0, 5), false);
  assert.equal(evictionCannotImmediatelyRelatch(rect, band), true);
});

test("there is a real gap to scroll back across, and the metric can report there is not", () => {
  const band = { top: 0, height: 800 };
  assert.equal(relatchGapPx(band), (HOLD_BAND - REACH_BAND) * 800);
  assert.ok(relatchGapPx(band) > 0, "a zero gap is a single boundary, which is churn");
  // CONTROL: 0 when the two bands coincide, so the answer above is a property of the constants.
  assert.equal(relatchGapPx(band, REACH_BAND, REACH_BAND), 0);
  assert.equal(relatchGapPx({ top: 0, height: 0 }), 0);
});

test("a fence between the two bands is neither evicted nor re-latched", () => {
  // The gap as behaviour rather than arithmetic: at two root heights out, neither band acts.
  const band = { top: 0, height: 100 };
  const between = { top: 250, bottom: 270 };
  assert.equal(withinBand(between, band, REACH_BAND), false, "reach would not take it");
  assert.equal(withinBand(between, band, HOLD_BAND), true, "hold will not give it back");
});

test("distance is zero inside the box and grows in root heights outside it", () => {
  const band = { top: 100, height: 200 };
  assert.equal(bandDistance({ top: 150, bottom: 160 }, band), 0);
  assert.equal(bandDistance({ top: 500, bottom: 520 }, band), 1);
  assert.equal(bandDistance({ top: -300, bottom: -100 }, band), 1);
  assert.ok(
    bandDistance({ top: 900, bottom: 920 }, band)
      > bandDistance({ top: 500, bottom: 520 }, band),
    "further is further",
  );
});

test("a zero-height band cannot divide by zero", () => {
  assert.equal(bandDistance({ top: 10, bottom: 20 }, { top: 0, height: 0 }), 0);
});

/* --------------------------------------------------------------- the brakes */

const far = { top: 100_000, bottom: 100_100 };
const near = { top: 10, bottom: 110 };
const band = { top: 0, height: 800 };

const candidate = (over: Partial<EvictCandidate> = {}): EvictCandidate => ({
  id: 1,
  rect: far,
  band,
  latchedAt: 0,
  streaming: false,
  ...over,
});

test("a fence far outside the hold band, past its dwell, is evictable", () => {
  assert.equal(fenceIsEvictable(candidate(), DWELL_MS), true);
});

test("a fence inside the hold band is not, however long it has been there", () => {
  assert.equal(
    fenceIsEvictable(candidate({ rect: near }), DWELL_MS * 1000),
    false,
  );
});

test("a fence inside its dwell is not, however far away it is", () => {
  assert.equal(fenceIsEvictable(candidate(), DWELL_MS - 1), false);
  // The boundary is inclusive, and the row above is what makes that assertion mean something.
  assert.equal(fenceIsEvictable(candidate(), DWELL_MS), true);
});

test("a streaming fence is never evicted", () => {
  assert.equal(
    fenceIsEvictable(candidate({ streaming: true }), DWELL_MS * 100),
    false,
  );
});

test("a live selection stops the whole pass", () => {
  const rows = [candidate({ id: 1 }), candidate({ id: 2 })];
  // PRECONDITION: without the selection these two would be evicted.
  assert.deepEqual(
    planEviction(rows, DWELL_MS, { selectionLive: false, printing: false }),
    [1, 2],
  );
  assert.deepEqual(
    planEviction(rows, DWELL_MS, { selectionLive: true, printing: false }),
    [],
  );
});

test("a print stops the whole pass", () => {
  const rows = [candidate({ id: 7 })];
  assert.deepEqual(
    planEviction(rows, DWELL_MS, { selectionLive: false, printing: false }),
    [7],
  );
  assert.deepEqual(
    planEviction(rows, DWELL_MS, { selectionLive: false, printing: true }),
    [],
  );
});

test("one pass gives back at most the budget, furthest first", () => {
  const rows: EvictCandidate[] = [];
  for (let i = 0; i < EVICT_BUDGET * 3; i += 1) {
    rows.push(
      candidate({
        id: i,
        // Ascending distance, so the LAST ids are the furthest and must come out first.
        rect: { top: 10_000 + i * 1_000, bottom: 10_100 + i * 1_000 },
      }),
    );
  }
  const plan = planEviction(rows, DWELL_MS, {
    selectionLive: false,
    printing: false,
  });
  assert.equal(plan.length, EVICT_BUDGET);
  assert.deepEqual(
    plan,
    Array.from({ length: EVICT_BUDGET }, (_, i) => rows.length - 1 - i),
  );
});

test("a pass over nothing eligible returns nothing", () => {
  assert.deepEqual(
    planEviction([candidate({ rect: near })], DWELL_MS, {
      selectionLive: false,
      printing: false,
    }),
    [],
  );
  assert.deepEqual(
    planEviction([], DWELL_MS, { selectionLive: false, printing: false }),
    [],
  );
});

/* -------------------------------------------------- draining without a gesture */

test("a pass that evicted something asks for another, so the budget does not strand the rest", () => {
  const rows = [candidate({ id: 1 })];
  assert.equal(nextPassDelayMs(rows, DWELL_MS, 1), PASS_INTERVAL_MS);
  assert.equal(nextPassDelayMs(rows, DWELL_MS, EVICT_BUDGET), PASS_INTERVAL_MS);
});

test("a fence that is far away but still inside its dwell sets the next pass from its own clock", () => {
  // Latched at 0, so at t = 500 it has 1000 ms of dwell left.
  assert.equal(nextPassDelayMs([candidate({ latchedAt: 0 })], 500, 0), DWELL_MS - 500);
  // Floored at the pass interval, so a fence that matures in one millisecond cannot spin.
  assert.equal(
    nextPassDelayMs([candidate({ latchedAt: 0 })], DWELL_MS - 1, 0),
    PASS_INTERVAL_MS,
  );
  // The soonest of several is the one that decides.
  assert.equal(
    nextPassDelayMs(
      [candidate({ id: 1, latchedAt: 0 }), candidate({ id: 2, latchedAt: 600 })],
      600,
      0,
    ),
    DWELL_MS - 600,
  );
});

test("nothing to do asks for no timer at all", () => {
  // Nothing latched.
  assert.equal(nextPassDelayMs([], 0, 0), null);
  // Everything still inside the hold band, so no amount of waiting makes it eligible.
  assert.equal(nextPassDelayMs([candidate({ rect: near, latchedAt: 0 })], 10, 0), null);
  // Eligible and not taken means the pass was refused, so rescheduling would spin against it.
  assert.equal(nextPassDelayMs([candidate({ latchedAt: 0 })], DWELL_MS * 10, 0), null);
  // A streaming fence is never a reason to schedule anything.
  assert.equal(
    nextPassDelayMs([candidate({ streaming: true, latchedAt: 0 })], 10, 0),
    null,
  );
});

test("the drain terminates: every scheduled pass either evicts or waits out a dwell", () => {
  // The loop the wiring runs, 40 fences with the budget capping every pass, must stop: an
  // unbounded reschedule is a timer that never sleeps.
  let remaining: EvictCandidate[] = [];
  for (let i = 0; i < 40; i += 1) {
    remaining.push(candidate({ id: i, latchedAt: 0 }));
  }
  let at = DWELL_MS;
  let passes = 0;
  for (;;) {
    passes += 1;
    assert.ok(passes < 100, "the drain must terminate");
    const plan = planEviction(remaining, at, { selectionLive: false, printing: false });
    const gone = new Set(plan);
    const next = nextPassDelayMs(remaining, at, plan.length);
    remaining = remaining.filter((row) => !gone.has(row.id));
    if (next === null) break;
    at += next;
  }
  assert.equal(remaining.length, 0, "and it must actually drain the thread");
  assert.equal(passes, Math.ceil(40 / EVICT_BUDGET) + 1, "one pass per budget, plus the last");
});

test("the first pass never waits for an interval that has not started", () => {
  assert.equal(passIsDue(0, null), true);
});

test("passes are spaced, so a flick that fires many scroll events runs one", () => {
  assert.equal(passIsDue(1000, 1000), false);
  assert.equal(passIsDue(1000 + PASS_INTERVAL_MS - 1, 1000), false);
  assert.equal(passIsDue(1000 + PASS_INTERVAL_MS, 1000), true);
});

/** JSX in the mode module makes this file unloadable and every assertion above a comment. */
test("the mode module is plain TypeScript", () => {
  const source = readFileSync(
    new URL(
      "../src/components/assistant-ui/code-fence-evict.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // Comments stripped first: the module's header says it has neither JSX nor `import.meta`, and a
  // check that read its prose would fail on the explanation.
  const code = source.replace(/^\s*[/*].*$/gm, "");
  assert.ok(!/<[A-Za-z]/.test(code), "no JSX in the mode module");
  assert.ok(
    !/\bfrom\s+["']react["']/.test(code),
    "and no react import: this module has to load under --experimental-strip-types",
  );
  assert.ok(
    !/import\.meta/.test(code),
    "and no import.meta: the runner cannot evaluate it, so the flag read stays in the wiring",
  );
  // PRECONDITION: the strip must not have eaten the whole file, or the three rows above pass on "".
  assert.ok(code.includes("resolveFenceEvictMode"), "the stripped source must still be code");
});

test("the constants are the ones the design note argues for", () => {
  // Pinned so that widening a brake is a deliberate edit with a test to change, not a tweak.
  assert.equal(REACH_BAND, 1);
  assert.equal(HOLD_BAND, 3);
  assert.equal(DWELL_MS, 1500);
  assert.equal(EVICT_BUDGET, 8);
  assert.equal(PASS_INTERVAL_MS, 400);
});
