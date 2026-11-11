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
  RowHeights,
  WINDOW_KEEP_PX,
  WINDOW_MIN_ROWS,
  WINDOW_MIN_TOTAL_PX,
  WINDOW_REACH_PX,
  absorbShift,
  initialRowWindow,
  messageWeight,
  planEngage,
  planFill,
  planSlide,
  sameWindow,
  spacerHeights,
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
    topSpacerTop: (rows[0]?.top ?? 0) - win.start * 500,
    bottomSpacerTop: rows[rows.length - 1]?.bottom ?? 0,
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

test("a pinned row stops the trim on its side", () => {
  const win = { start: 100, end: 140 };
  const rows = stack(win, -10_000, 500);
  const free = slide(win, rows);
  const pinnedAbove = slide(win, rows, { pinned: (index) => index === 104 });
  assert.equal(pinnedAbove.start, 104, "rows before the pinned one still go");
  assert.equal(pinnedAbove.end, free.end);
  const pinnedBelow = slide(win, rows, { pinned: (index) => index === 139 });
  assert.equal(pinnedBelow.end, 140);
  assert.equal(pinnedBelow.start, free.start);
});

test("a jump above the window re-opens it on the row the offset lands on", () => {
  const win = { start: 500, end: 510 };
  // The viewport now sits 20 rows (10,000px) above the first mounted row.
  const firstTop = 10_000 + VIEW;
  const rows = stack(win, firstTop, 500);
  const next = slide(win, rows, { topSpacerTop: firstTop - 500 * 500 });
  // Offset into the spacer: (500 rows * 500px) - 10,800px = row 478.4.
  assert.equal(next.start, 478);
  assert.ok(next.end > next.start && next.end <= next.start + 5);
  assert.ok(next.end <= win.start, "the old rows are dropped, not kept across the gap");
});

test("a jump below the window walks the rows after it", () => {
  const win = { start: 100, end: 110 };
  const rows = stack(win, -5000 - 20_000, 500);
  const bottomSpacerTop = rows[rows.length - 1].bottom;
  const next = slide(win, rows, { bottomSpacerTop });
  // The viewport top is 20,000px into the trailing spacer: 40 rows on.
  assert.equal(next.start, 150);
});

test("a jump accounts for what the leading spacer carries beyond its rows", () => {
  const win = { start: 500, end: 510 };
  const firstTop = 10_000 + VIEW;
  const plain = slide(win, stack(win, firstTop, 500), { topSpacerTop: firstTop - 500 * 500 });
  // 1000px of slack: the same viewport offset is two rows earlier in the thread.
  const slack = slide(win, stack(win, firstTop, 500), {
    topSpacerTop: firstTop - 500 * 500 - 1000,
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
