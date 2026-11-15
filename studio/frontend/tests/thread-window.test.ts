// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sliding mount window: which rows a reader's position asks for, how tall the rows outside
// are taken to be, and how a height surprise is kept from moving what the reader sees.
// The .tsx glue is pinned in progressive-mount-glue.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type MountedRow,
  type RowMeta,
  type RowWindow,
  FILL_BUDGET_PX,
  RowHeights,
  STREAM_QUIET_MS,
  TailPulse,
  WINDOW_KEEP_PX,
  WINDOW_MIN_ROWS,
  WINDOW_MIN_TOTAL_PX,
  WINDOW_REACH_PX,
  TAIL_TURN_MAX_ROWS,
  absorbShift,
  holdTailTurn,
  followCount,
  initialRowWindow,
  messageWeight,
  numberedEquations,
  planEngage,
  planFill,
  planSlide,
  sameWindow,
  spacerHeights,
  spacerRows,
  threadSlots,
  withPins,
} from "../src/components/assistant-ui/thread-window.ts";

const VIEW = 800;

const meta = (index: number, weight = 20, role = index % 2 ? "assistant" : "user"): RowMeta => ({
  id: `m${index}`,
  role,
  weight,
  renders: true,
});

/** Rows of one height stacked from `firstTop`, as the glue would measure them. */
function stack(win: RowWindow, firstTop: number, height: number): MountedRow[] {
  const rows: MountedRow[] = [];
  for (let index = win.start; index < win.end; index += 1) {
    const top = firstTop + (index - win.start) * height;
    rows.push({ index, top, bottom: top + height });
  }
  return rows;
}

function slide(
  win: RowWindow,
  rows: MountedRow[],
  overrides: Partial<Parameters<typeof planSlide>[0]> = {},
): RowWindow {
  return planSlide({
    win,
    count: 1000,
    rows,
    viewHeight: VIEW,
    threadTop: (rows[0]?.top ?? 0) - win.start * 500,
    slack: 0,
    heightOf: () => 500,
    maxStep: 4,
    trim: true,
    ...overrides,
  });
}

test("a part's weight is the lines it renders: text by line, collapsed parts by a fixed share", () => {
  assert.equal(messageWeight(undefined), 0);
  assert.equal(messageWeight([{ type: "text", text: "abcd" }]), 1);
  assert.equal(messageWeight([{ type: "text", text: "a\nb\nc" }]), 3);
  // A long line wraps; forty short lines of code are forty lines, not one paragraph.
  assert.equal(messageWeight([{ type: "text", text: "x".repeat(450) }]), 6);
  const code = Array.from({ length: 40 }, () => "let a = 1;").join("\n");
  assert.equal(messageWeight([{ type: "text", text: code }]), 40);
  const tool = messageWeight([{ type: "tool-call" }]);
  assert.ok(tool > 0);
  // A reasoning block renders collapsed however long it ran, so its text is not counted.
  assert.equal(messageWeight([{ type: "reasoning", text: "x".repeat(50_000) }]), tool);
  assert.ok(messageWeight([{ type: "image" }]) > tool);
  // Sources are inline badges on lines the text already counted.
  assert.equal(messageWeight([{ type: "source" }]), 0);
});

test("an unmeasured row is estimated from its lines, and a measured one is exact", () => {
  const heights = new RowHeights();
  const short = heights.heightOf(meta(1, 2));
  const long = heights.heightOf(meta(3, 400));
  assert.ok(long > short * 10, "a long reply must not be given a short reply's room");
  heights.measure(meta(3, 400), 1234.5);
  assert.equal(heights.heightOf(meta(3, 400)), 1234.5);
  assert.equal(heights.has("m3"), true);
  assert.equal(heights.has("m1"), false);
});

test("a message that renders no row takes no room", () => {
  const heights = new RowHeights();
  assert.equal(heights.heightOf({ ...meta(0), renders: false }), 0);
});

test("the estimate learns what a line takes in each role from the rows measured so far", () => {
  const heights = new RowHeights();
  // Replies here run 40px per line (code, tables), well off the prior.
  for (let index = 1; index < 41; index += 2) {
    const lines = 40 + index;
    heights.measure(meta(index, lines), 72 + lines * 40);
  }
  const guess = heights.heightOf(meta(999, 200));
  assert.ok(Math.abs(guess - (72 + 8000)) < 250, `expected about 8072, got ${guess}`);
  // Prompts were not measured, so they keep their own prior rather than the replies' density.
  const prompt = heights.heightOf(meta(998, 2, "user"));
  assert.ok(prompt < 300, `a short prompt was estimated at ${prompt}`);
});

test("three alike rows at the tail are enough to size the rest", () => {
  // What a thread opens with: the last few rows measured, every other row a guess.
  const heights = new RowHeights();
  for (const index of [395, 397, 399]) heights.measure(meta(index, 60), 1700);
  const guess = heights.heightOf(meta(1, 60));
  assert.ok(Math.abs(guess - 1700) < 1700 * 0.1, `expected within 10% of 1700, got ${guess}`);
});

test("rows of one length converge on their measured height, whatever the prior said", () => {
  const heights = new RowHeights();
  for (let index = 0; index < 60; index += 2) heights.measure(meta(index, 4, "user"), 213);
  const guess = heights.heightOf(meta(500, 4, "user"));
  assert.ok(Math.abs(guess - 213) < 15, `expected about 213, got ${guess}`);
});

test("measuring a row again replaces what it taught the estimate", () => {
  const once = new RowHeights();
  once.measure(meta(1, 80), 2000);
  const twice = new RowHeights();
  // A row is measured while it is still rendering, then again at its settled height.
  twice.measure(meta(1, 80), 90);
  twice.measure(meta(1, 80), 2000);
  assert.equal(twice.heightOf(meta(7, 180)), once.heightOf(meta(7, 180)));
});

test("the spacers are the heights of the rows on either side of the window", () => {
  const heights = new RowHeights();
  const metas = Array.from({ length: 10 }, (_, index) => meta(index));
  metas.forEach((row, index) => heights.measure(row, 100 + index));
  assert.deepEqual(spacerHeights(heights, metas, null), { top: 0, bottom: 0 });
  assert.deepEqual(spacerHeights(heights, metas, { start: 2, end: 7 }), {
    top: 100 + 101,
    bottom: 107 + 108 + 109,
  });
  assert.deepEqual(spacerHeights(heights, metas, { start: 0, end: 10 }), { top: 0, bottom: 0 });
  // A window past a thread that shrank counts nothing twice.
  assert.deepEqual(spacerHeights(heights, metas.slice(0, 3), { start: 5, end: 9 }), {
    top: 100 + 101 + 102,
    bottom: 0,
  });
});

test("a thread opens on the tail, sized by height and capped in rows", () => {
  const cover = VIEW + WINDOW_REACH_PX;
  // Tall replies: two rows already cover the reach.
  assert.deepEqual(initialRowWindow(600, () => 2000, cover), { start: 598, end: 600 });
  // Short rows: the row cap bounds the first commit.
  assert.deepEqual(initialRowWindow(6000, () => 20, cover), { start: 5984, end: 6000 });
});

test("a thread is windowed by its height, not its message count", () => {
  const cover = VIEW + WINDOW_REACH_PX;
  // A few turns of very long replies (an agent run) is exactly the thread that needs it.
  assert.deepEqual(initialRowWindow(10, () => 12_000, cover), { start: 9, end: 10 });
  // Many short messages are cheap to mount whole.
  assert.equal(initialRowWindow(200, () => 100, cover), null);
  const rows = Math.ceil(WINDOW_MIN_TOTAL_PX / 500);
  assert.equal(initialRowWindow(rows - 1, () => 500, cover), null, "just under the height floor");
  assert.notEqual(initialRowWindow(rows, () => 500, cover), null, "at the height floor");
  // Too few messages to leave any out, however tall.
  assert.equal(initialRowWindow(WINDOW_MIN_ROWS - 1, () => 50_000, cover), null);
  // The tail that fits reaches the first row: nothing would be withheld.
  assert.equal(initialRowWindow(WINDOW_MIN_ROWS, () => 5000, 1_000_000, 64), null);
});

test("messages that render nothing do not use up the opening rows", () => {
  const cover = VIEW + WINDOW_REACH_PX;
  const win = initialRowWindow(6000, (index) => (index % 2 ? 0 : 20), cover, 16);
  assert.deepEqual(win, { start: 6000 - 32, end: 6000 });
});

test("a reader at rest inside the reach keeps the window as it is", () => {
  // Fourteen 500px rows from 2500px above the viewport to 4500px below its top.
  const win = { start: 100, end: 114 };
  const rows = stack(win, -WINDOW_REACH_PX - 500, 500);
  assert.ok(sameWindow(slide(win, rows), win));
});

test("scrolling up mounts rows above, a step at a time, and never past the first", () => {
  const win = { start: 100, end: 110 };
  // The first mounted row starts 300px above the viewport: 1700px of the reach are uncovered.
  const next = slide(win, stack(win, -300, 500));
  assert.equal(next.start, 96, "four 500px rows cover the 1700px gap");
  assert.equal(next.end, 110);
  const capped = slide(win, stack(win, -300, 500), { maxStep: 2 });
  assert.equal(capped.start, 98, "a commit mounts maxStep rows at most");
  const atHead = slide({ start: 1, end: 10 }, stack({ start: 1, end: 10 }, 0, 500));
  assert.equal(atHead.start, 0);
});

test("scrolling down mounts rows below, up to the end of the thread", () => {
  const win = { start: 100, end: 107 };
  // Seven rows from above the upper reach end 1000px below the viewport top: 1800px of the
  // lower reach are uncovered.
  const next = slide(win, stack(win, -2500, 500));
  assert.equal(next.start, 100);
  assert.equal(next.end, 111);
  const tail = { start: 992, end: 999 };
  assert.equal(slide(tail, stack(tail, -2500, 500)).end, 1000);
});

test("rows past the keep distance unmount; rows between reach and keep stay", () => {
  const win = { start: 100, end: 140 };
  // Row 100 ends at -9500; row 119 starts at 0.
  const rows = stack(win, -10_000, 500);
  const next = slide(win, rows);
  const firstKept = rows.find((row) => row.bottom >= -WINDOW_KEEP_PX);
  const lastKept = [...rows].reverse().find((row) => row.top <= VIEW + WINDOW_KEEP_PX);
  assert.equal(next.start, firstKept?.index);
  assert.equal(next.end, (lastKept?.index ?? 0) + 1);
  assert.ok(next.start > win.start && next.end < win.end);
  // Hysteresis: a row just past the reach is not dropped the moment it leaves it.
  const near = { start: 100, end: 112 };
  assert.ok(sameWindow(slide(near, stack(near, -WINDOW_REACH_PX - 700, 500)), near));
});

test("nothing unmounts while trimming is held", () => {
  const win = { start: 100, end: 140 };
  const next = slide(win, stack(win, -10_000, 500), { trim: false });
  assert.ok(sameWindow(next, win), "a selection or a held pointer keeps every mounted row");
});

test("a pinned row is held apart from the window, not by stretching the window to it", () => {
  // Rows the next window leaves out but the reader is working in: before it, inside it, after it.
  const win = withPins({ start: 300, end: 310 }, [104, 305, 599, 600], 600);
  assert.deepEqual(win, { start: 300, end: 310, pins: [104, 599] });
  // No pin leaves a plain window, so it compares equal to one that never had any.
  assert.deepEqual(withPins({ start: 300, end: 310, pins: [104] }, [], 600), { start: 300, end: 310 });
  assert.ok(!sameWindow(win, { start: 300, end: 310 }));
  assert.ok(!sameWindow(win, { start: 300, end: 310, pins: [104, 598] }));
  assert.ok(sameWindow(win, { start: 300, end: 310, pins: [104, 599] }));
});

test("the thread is laid out as rows and the spacers standing in for the rest", () => {
  const names = (win: RowWindow | null, count: number) =>
    threadSlots(win, count).map((slot) =>
      slot.kind === "row" ? slot.index : `${slot.name}[${slot.from},${slot.to})`,
    );
  assert.deepEqual(names({ start: 3, end: 5 }, 8), ["top[0,3)", 3, 4, "bottom[5,8)"]);
  // A full mount keeps its two spacers, so the nodes survive the window closing and re-opening.
  assert.deepEqual(names(null, 3), ["top[0,0)", 0, 1, 2, "bottom[3,3)"]);
  // Pins on both sides: each is a row of its own, and the spacers against the window stop at
  // the nearest one.
  assert.deepEqual(names({ start: 10, end: 12, pins: [2, 6, 15, 19] }, 20), [
    "gap[0,2)", 2, "gap[3,6)", 6, "top[7,10)", 10, 11, "bottom[12,15)", 15, "gap[16,19)", 19,
    "gap[20,20)",
  ]);
  // Every row of the thread is in exactly one slot, in order.
  const covered: number[] = [];
  for (const slot of threadSlots({ start: 10, end: 12, pins: [2, 6, 15, 19] }, 20)) {
    if (slot.kind === "row") covered.push(slot.index);
    else for (let index = slot.from; index < slot.to; index += 1) covered.push(index);
  }
  assert.deepEqual(covered, Array.from({ length: 20 }, (_, index) => index));
  // A window clamped to a thread that shrank, and a pin past its end.
  assert.deepEqual(names({ start: 2, end: 9, pins: [7] }, 4), ["top[0,2)", 2, 3, "bottom[4,4)"]);
});

test("the spacers against the window stop at a pinned row", () => {
  const heights = new RowHeights();
  const metas = Array.from({ length: 20 }, (_, index) => meta(index));
  metas.forEach((row, index) => heights.measure(row, 100 + index));
  const win = { start: 10, end: 12, pins: [2, 6, 15, 19] };
  assert.deepEqual(spacerRows(win, 20), { topFrom: 7, bottomTo: 15 });
  assert.deepEqual(spacerHeights(heights, metas, win), {
    top: 107 + 108 + 109,
    bottom: 112 + 113 + 114,
  });
  assert.deepEqual(spacerRows({ start: 10, end: 12 }, 20), { topFrom: 0, bottomTo: 20 });
  // A window that starts right after a pin has a leading spacer standing for no rows.
  assert.deepEqual(spacerRows({ start: 7, end: 12, pins: [6] }, 20), { topFrom: 7, bottomTo: 20 });
});

test("a window follows a thread that grows at its tail and is clamped to one that shrinks", () => {
  const tail = { start: 590, end: 600 };
  assert.deepEqual(followCount(tail, 600, 602), { start: 590, end: 602 });
  // A reader further up is left where they are, on the same object.
  const middle = { start: 300, end: 310 };
  assert.equal(followCount(middle, 600, 602), middle);
  assert.deepEqual(followCount(tail, 600, 595), { start: 590, end: 595 });
  assert.equal(followCount(tail, 600, 590), null);
  // Pins come along: one the tail grew over is the window's now, one past the end is gone.
  assert.deepEqual(followCount({ start: 300, end: 310, pins: [100, 599] }, 600, 598), {
    start: 300,
    end: 310,
    pins: [100],
  });
  assert.deepEqual(followCount({ start: 590, end: 600, pins: [100] }, 600, 601), {
    start: 590,
    end: 601,
    pins: [100],
  });
});

test("a jump above the window re-opens it on the row the offset lands on", () => {
  const win = { start: 500, end: 510 };
  // The viewport now sits 20 rows (10,000px) above the first mounted row.
  const firstTop = 10_000 + VIEW;
  const rows = stack(win, firstTop, 500);
  const next = slide(win, rows, { threadTop: firstTop - 500 * 500 });
  // Offset into the spacer: (500 rows * 500px) - 10,800px = row 478.4.
  assert.equal(next.start, 478);
  assert.ok(next.end > next.start && next.end <= next.start + 5);
  assert.ok(next.end <= win.start, "the old rows are dropped, not kept across the gap");
});

test("a jump below the window walks the rows after it", () => {
  const win = { start: 100, end: 110 };
  const rows = stack(win, -5000 - 20_000, 500);
  const next = slide(win, rows);
  // The viewport top is 20,000px into the trailing spacer: 40 rows on.
  assert.equal(next.start, 150);
});

test("a jump drops every row of the old window: holding one is the caller's pin, not a stretch", () => {
  const win = { start: 100, end: 110, pins: [40] };
  const next = slide(win, stack(win, -5000 - 20_000, 500));
  assert.equal(next.start, 150);
  assert.ok(next.end <= 150 + 5, "the window does not reach back for the rows it left");
  assert.equal(next.pins, undefined);
});

test("a jump accounts for what the leading spacer carries beyond its rows", () => {
  const win = { start: 500, end: 510 };
  const firstTop = 10_000 + VIEW;
  const plain = slide(win, stack(win, firstTop, 500), { threadTop: firstTop - 500 * 500 });
  // 1000px of slack: the same viewport offset is two rows earlier in the thread.
  const slack = slide(win, stack(win, firstTop, 500), {
    threadTop: firstTop - 500 * 500 - 1000,
    slack: 1000,
  });
  assert.equal(slack.start, plain.start);
});

test("a short hop past the mounted rows extends the window instead of dropping it", () => {
  const win = { start: 100, end: 110 };
  // The viewport is 1000px above the first row: inside the keep distance.
  const next = slide(win, stack(win, 1000 + VIEW, 500));
  assert.ok(next.start < win.start);
  assert.ok(next.end > next.start);
  assert.ok(next.end >= 100, "rows the reader just left are still within the keep distance");
});

test("a window with nothing mounted yet still makes progress", () => {
  const next = slide({ start: 100, end: 101 }, [], { heightOf: () => 0 });
  assert.ok(next.end > next.start);
  const grown = slide({ start: 100, end: 100 }, []);
  assert.ok(grown.end > grown.start, "an empty window never stays empty");
});

test("a window past a thread that shrank is clamped before it is planned", () => {
  const next = slide({ start: 100, end: 400 }, stack({ start: 100, end: 110 }, 0, 500), { count: 105 });
  assert.ok(next.end <= 105);
  assert.ok(next.start < next.end);
});

test("sliding converges: the planned window is a fixed point once the reach is covered", () => {
  let win: RowWindow = { start: 500, end: 501 };
  for (let step = 0; step < 50; step += 1) {
    // The reader holds row 500's top at the viewport top; rows are 500px.
    const next = slide(win, stack(win, (win.start - 500) * 500, 500));
    if (sameWindow(next, win)) {
      assert.ok(win.start <= 496, "the upper reach is covered");
      assert.ok(win.end >= 506, "the lower reach is covered");
      return;
    }
    win = next;
  }
  assert.fail("the window never settled");
});

test("a full mount widens both ways and ends unrestricted", () => {
  assert.deepEqual(planFill({ start: 100, end: 110 }, 1000, 32), { start: 68, end: 142 });
  assert.deepEqual(planFill({ start: 10, end: 990 }, 1000, 32), null);
  assert.deepEqual(planFill({ start: 0, end: 500 }, 1000, 32), { start: 0, end: 532 });
  let win: RowWindow | null = { start: 598, end: 600 };
  let steps = 0;
  while (win != null) {
    const next: RowWindow | null = planFill(win, 600, 16);
    assert.ok(next == null || (next.start <= win.start && next.end >= win.end), "never narrows");
    win = next;
    steps += 1;
    assert.ok(steps < 100, "a full mount must finish");
  }
});

test("a full mount pass is bounded by estimated height, not only by rows", () => {
  const win = { start: 100, end: 110 };
  // Short messages: the row cap is what bounds the pass, as before.
  assert.deepEqual(planFill(win, 1000, 32, () => 200), { start: 68, end: 142 });
  // Very long replies: the budget stops the pass after a few rows, split between both sides.
  const tall = planFill(win, 1000, 32, () => 6000);
  assert.deepEqual(tall, { start: 98, end: 111 });
  assert.ok((win.start - 98 + (111 - win.end)) * 6000 <= FILL_BUDGET_PX);
  // A row taller than the whole budget still mounts, one a pass: a full mount must finish.
  assert.deepEqual(planFill(win, 1000, 32, () => FILL_BUDGET_PX * 3), { start: 99, end: 110 });
  // Messages that render nothing cost nothing and do not count as the row a pass must add.
  const sparse = planFill(win, 1000, 32, (index) => (index % 2 ? 0 : FILL_BUDGET_PX * 3));
  assert.deepEqual(sparse, { start: 99, end: 111 }, "row 99 renders nothing; row 110 is the pass");
  // One side exhausted: the other takes the whole budget.
  assert.deepEqual(planFill({ start: 0, end: 10 }, 1000, 32, () => 5000), { start: 0, end: 14 });
  let current: RowWindow | null = { start: 598, end: 600 };
  let passes = 0;
  while (current != null) {
    const before: RowWindow = current;
    current = planFill(before, 600, 32, (index) => 500 + (index % 7) * 4000);
    if (current != null) {
      let added = 0;
      for (let index = current.start; index < before.start; index += 1) added += 500 + (index % 7) * 4000;
      for (let index = before.end; index < current.end; index += 1) added += 500 + (index % 7) * 4000;
      const rows = before.start - current.start + (current.end - before.end);
      assert.ok(rows >= 1, "every pass adds a row");
      assert.ok(added <= FILL_BUDGET_PX || rows === 1, `a pass added ${added}px over ${rows} rows`);
    }
    passes += 1;
    assert.ok(passes < 1000, "a full mount must finish");
  }
});

test("a fully mounted thread re-opens its window only when enough would unmount", () => {
  const all = { start: 0, end: 600 };
  // The reader is at row 300; every row is 500px.
  const rows = stack(all, -300 * 500, 500);
  const win = planEngage(rows, 600, VIEW);
  assert.ok(win != null);
  assert.ok(win.start > 280 && win.start < 300);
  assert.ok(win.end > 300 && win.end < 320);
  // Short thread: under the height floor, so it stays whole.
  const small = { start: 0, end: 40 };
  assert.equal(planEngage(stack(small, -2000, 200), 40, VIEW), null);
  // Tall enough, but it is one long reply the reader is in: the few rows after it are not worth
  // a commit.
  const giant: MountedRow[] = [{ index: 0, top: -20_000, bottom: 22_000 }];
  for (let index = 1; index < 9; index += 1) {
    giant.push({ index, top: 22_000 + (index - 1) * 300, bottom: 22_000 + index * 300 });
  }
  assert.equal(planEngage(giant, 9, VIEW), null);
  assert.equal(planEngage([], 600, VIEW), null);
});

test("re-opening counts the rows on both sides of the reader", () => {
  const all = { start: 0, end: 200 };
  // The reader is at the top: everything to unmount is below.
  const below = planEngage(stack(all, 0, 500), 200, VIEW);
  assert.deepEqual(below?.start, 0);
  assert.ok(below != null && below.end < 20);
  // At the tail: everything to unmount is above.
  const above = planEngage(stack(all, -199 * 500, 500), 200, VIEW);
  assert.ok(above != null && above.start > 180);
  assert.equal(above?.end, 200);
});

test("the leading spacer takes a height surprise, so the reader's row does not move", () => {
  // Rows mounted above came out 120px taller than estimated: the spacer gives that back.
  assert.deepEqual(absorbShift(5000, 120, false), { spacer: 4880, scroll: 0 });
  // Shorter than estimated: the spacer grows.
  assert.deepEqual(absorbShift(5000, -80, false), { spacer: 5080, scroll: 0 });
});

test("only what the spacer cannot take is left for a scroll correction", () => {
  // The spacer runs out: the rest is a scroll write.
  assert.deepEqual(absorbShift(100, 250, false), { spacer: 0, scroll: 150 });
  // At the first row there is no spacer; whatever it still held is corrected by scrolling.
  assert.deepEqual(absorbShift(0, 40, true), { spacer: 0, scroll: 40 });
  assert.deepEqual(absorbShift(300, 40, true), { spacer: 0, scroll: -260 });
});

test("the reach and keep distances leave hysteresis between mounting and unmounting", () => {
  assert.ok(WINDOW_KEEP_PX >= WINDOW_REACH_PX * 1.5);
});

test("a window holding the tail also holds the prompt its reply answers", () => {
  assert.deepEqual(holdTailTurn({ start: 9, end: 10 }, 10, 8), { start: 8, end: 10 });
  assert.deepEqual(holdTailTurn({ start: 5, end: 10 }, 10, 8), { start: 5, end: 10 });
  assert.deepEqual(holdTailTurn({ start: 9, end: 10 }, 10, -1), { start: 9, end: 10 });
  // A window away from the tail is the reader's, not the tail's.
  assert.deepEqual(holdTailTurn({ start: 2, end: 6 }, 10, 1), { start: 2, end: 6 });
  assert.equal(holdTailTurn(null, 10, 8), null);
  // A prompt many rows up (a long agent turn) is not worth mounting everything between.
  const far = { start: 50, end: 60 };
  assert.deepEqual(holdTailTurn(far, 60, 50 - TAIL_TURN_MAX_ROWS - 1), far);
  assert.deepEqual(holdTailTurn(far, 60, 50 - TAIL_TURN_MAX_ROWS), {
    start: 50 - TAIL_TURN_MAX_ROWS,
    end: 60,
  });
  // A pin the widened window now covers is no longer a pin.
  assert.deepEqual(holdTailTurn({ start: 9, end: 10, pins: [2, 8] }, 10, 7), {
    start: 7,
    end: 10,
    pins: [2],
  });
  const held = holdTailTurn({ start: 9, end: 10 }, 10, 8);
  assert.deepEqual(holdTailTurn(held, 10, 8), held);
});

test("a new row layout forgets every height, since every row re-wraps", () => {
  const heights = new RowHeights();
  const row = meta(1);
  assert.equal(heights.relayout("800 Inter 15px"), false, "the first layout forgets nothing");
  heights.measure(row, 900);
  assert.equal(heights.relayout("800 Inter 15px"), false);
  assert.equal(heights.heightOf(row), 900, "the same layout keeps them");
  assert.equal(heights.relayout("800 Inter 18px"), true);
  assert.equal(heights.has(row.id), false);
  assert.notEqual(heights.heightOf(row), 900);
});

test("numbered equations are counted where KaTeX's page-wide counters would count them", () => {
  const text = (value: string) => [{ type: "text", text: value }];
  assert.equal(numberedEquations(text("$$x$$ and \\begin{equation*}y\\end{equation*}")), 0);
  assert.equal(numberedEquations(text("$$\\begin{equation}a=b\\end{equation}$$")), 1);
  assert.equal(numberedEquations(text("\\[\\begin{align}a\\\\b \\notag\\\\c\\end{align}\\]")), 2);
  assert.equal(numberedEquations(text("$$\\begin{equation}a\\tag{*}\\end{equation}$$")), 0);
  // A comment is not a tag: KaTeX drops it to the end of the line.
  assert.equal(numberedEquations(text("$$\\begin{equation}a % \\notag\n\\end{equation}$$")), 1);
  assert.equal(numberedEquations(text("$$\\begin{equation}a = 5\\%\\notag\\end{equation}$$")), 0);
  // A nested environment's or a group's rows are not the outer one's.
  assert.equal(
    numberedEquations(text("$$\\begin{align}f &= \\begin{cases}1\\\\0\\end{cases}\\\\ g &= 2\\end{align}$$")),
    2,
  );
  assert.equal(numberedEquations(text("$$\\begin{gather}\\substack{a\\\\b}\\\\c\\end{gather}$$")), 2);
  // Only display math renders them: bare, inline-coded or fenced, they are text.
  assert.equal(numberedEquations(text("\\begin{equation}a=b\\end{equation}")), 0);
  assert.equal(numberedEquations(text("Use `$$\\begin{equation}a\\end{equation}$$` here")), 0);
  assert.equal(
    numberedEquations(
      text("```latex\n$$\\begin{equation}a\\end{equation}$$\n```\n\n$$\\begin{equation}b\\end{equation}$$"),
    ),
    1,
  );
  assert.equal(numberedEquations(text("~~~\n$$\\begin{equation}a\\end{equation}$$")), 0);
  assert.equal(numberedEquations(text("Example:\n\n    $$\\begin{equation}a\\end{equation}$$\n")), 0);
  assert.equal(numberedEquations(text("> ~~~\n> $$\\begin{equation}a\\end{equation}$$\n> ~~~")), 0);
  assert.equal(numberedEquations([{ type: "reasoning", text: "\\begin{equation}a\\end{equation}" }]), 0);
});

test("a spacer carries the equation counts of the rows it stands for", () => {
  const heights = new RowHeights();
  const metas = [0, 1, 2].map((index) => ({ ...meta(index), equations: index }));
  // Estimated from the text until a row is counted in the DOM.
  assert.deepEqual(heights.equationsIn(metas, 0, 3), { html: 3, mathml: 3 });
  heights.countEquations(metas[2], { html: 5, mathml: 0 });
  assert.deepEqual(heights.equationsIn(metas, 1, 3), { html: 6, mathml: 1 });
  assert.deepEqual(heights.equationsIn(metas, 2, 2), { html: 0, mathml: 0 });
  assert.equal(heights.measure(metas[0], 100), true);
  assert.equal(heights.measure(metas[0], 100), false, "an unchanged height is not a change");
});

test("always expanded thinking and tool cards count their text", () => {
  const trace = "step\n".repeat(40);
  const content = [
    { type: "reasoning", text: trace },
    { type: "tool-call", argsText: '{"q": 1}', result: "row\n".repeat(20) },
    { type: "text", text: "answer" },
  ];
  const collapsed = messageWeight(content);
  assert.equal(collapsed, messageWeight(content, { reasoning: false, tools: false }));
  assert.ok(messageWeight(content, { reasoning: true, tools: false }) >= collapsed + 40);
  assert.ok(messageWeight(content, { reasoning: false, tools: true }) >= collapsed + 20);
});

test("an open trace's equations count, a closed one's do not", () => {
  const content = [
    { type: "reasoning", text: "$$\\begin{equation}a\\end{equation}$$" },
    { type: "text", text: "done" },
  ];
  assert.equal(numberedEquations(content), 0);
  assert.equal(numberedEquations(content, { reasoning: true, tools: false }), 1);
});

test("a relayout forgets the equation counts read off the page", () => {
  const heights = new RowHeights();
  const row = { ...meta(1), equations: 0 };
  heights.relayout("thinking expanded");
  heights.countEquations(row, { html: 2, mathml: 2 });
  assert.deepEqual(heights.equationsIn([row], 0, 1), { html: 2, mathml: 2 });
  heights.relayout("thinking collapsed");
  assert.deepEqual(heights.equationsIn([row], 0, 1), { html: 0, mathml: 0 });
});

test("an edited message's page readings no longer stand for it", () => {
  const heights = new RowHeights();
  const before = { ...meta(1), equations: 0, revision: ["old"] };
  heights.measure(before, 300);
  heights.countEquations(before, { html: 2, mathml: 2 });
  assert.equal(heights.heightOf(before), 300);
  const after = { ...before, weight: 200, equations: 1, revision: ["new"] };
  assert.notEqual(heights.heightOf(after), 300, "the old height stood for new content");
  assert.deepEqual(heights.equationsIn([after], 0, 1), { html: 1, mathml: 1 });
  assert.equal(heights.measure(after, 300), true, "a new revision is a new reading at any height");
  assert.equal(heights.heightOf(after), 300);
});

test("a reply counts as streaming while its content keeps changing", () => {
  const first = ["a"];
  const pulse = new TailPulse(first);
  assert.equal(pulse.streaming(0), false, "the content a thread opened with is not a chunk");
  pulse.note(first, 10);
  assert.equal(pulse.changes, 0, "a store write that left the tail alone is not a chunk");
  assert.equal(pulse.streaming(10), false);
  pulse.note(["a", "b"], 100);
  assert.equal(pulse.changes, 1);
  assert.equal(pulse.streaming(100 + STREAM_QUIET_MS - 1), true);
  assert.equal(pulse.streaming(100 + STREAM_QUIET_MS), false, "the reply paused");
  pulse.note(["a", "b", "c"], 5000);
  assert.equal(pulse.changes, 2);
  assert.equal(pulse.streaming(5001), true);
  // A token a second is still one stream: a transition issued between two would be restarted.
  assert.ok(STREAM_QUIET_MS >= 1000);
});
