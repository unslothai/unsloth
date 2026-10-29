// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const {
  MIN_RANGE_S,
  addRange,
  clampRange,
  defaultRange,
  dragToRange,
  formatRange,
  formatRangeTime,
  mergeRanges,
  moveRange,
  nextFreeRange,
  normalizeRanges,
  nudgeStep,
  pxToSeconds,
  rangeLabel,
  removeRange,
  secondsBeyondEnd,
  secondsToFraction,
  secondsToPx,
  timelineEnd,
} = await import("../src/features/audio/components/waveform-range.ts");

const inside = { durationS: 10, maxRanges: 3 };
const one = { durationS: 10, maxRanges: 1 };
const tail = { durationS: 10, maxRanges: 1, beyondEndS: 5 };

test("pixels and seconds map across the timeline, tail included", () => {
  assert.equal(timelineEnd(inside), 10);
  assert.equal(timelineEnd(tail), 15);
  assert.equal(pxToSeconds(50, 200, 10), 2.5);
  assert.equal(pxToSeconds(-20, 200, 10), 0);
  assert.equal(pxToSeconds(400, 200, 10), 10);
  assert.equal(pxToSeconds(10, 0, 10), 0);
  assert.equal(secondsToPx(2.5, 200, 10), 50);
  assert.equal(secondsToFraction(15, 15), 1);
  assert.equal(secondsToFraction(20, 15), 1);
  assert.equal(secondsToFraction(1, 0), 0);
});

test("a range is ordered, clamped and at least 0.25 s", () => {
  assert.deepEqual(clampRange({ start_s: 4, end_s: 2 }, inside), {
    start_s: 2,
    end_s: 4,
  });
  assert.deepEqual(clampRange({ start_s: -1, end_s: 12 }, inside), {
    start_s: 0,
    end_s: 10,
  });
  assert.deepEqual(clampRange({ start_s: 3, end_s: 3.1 }, inside), {
    start_s: 3,
    end_s: 3 + MIN_RANGE_S,
  });
  assert.deepEqual(clampRange({ start_s: 9.95, end_s: 10 }, inside), {
    start_s: 9.75,
    end_s: 10,
  });
  assert.equal(
    clampRange({ start_s: 0, end_s: 1 }, { durationS: 0.1, maxRanges: 1 }),
    null,
  );
  assert.equal(clampRange({ start_s: Number.NaN, end_s: 1 }, inside), null);
});

test("overlapping and touching ranges merge, sorted", () => {
  assert.deepEqual(
    mergeRanges(
      [
        { start_s: 6, end_s: 8 },
        { start_s: 1, end_s: 3 },
        { start_s: 2, end_s: 4 },
        { start_s: 8, end_s: 9 },
      ],
      inside,
    ),
    [
      { start_s: 1, end_s: 4 },
      { start_s: 6, end_s: 9 },
    ],
  );
});

test("max ranges caps the list; one allowed replaces, past the cap is refused", () => {
  const three = [
    { start_s: 0, end_s: 1 },
    { start_s: 2, end_s: 3 },
    { start_s: 4, end_s: 5 },
  ];
  assert.equal(
    normalizeRanges([...three, { start_s: 7, end_s: 8 }], inside).length,
    3,
  );
  assert.deepEqual(addRange(three, { start_s: 7, end_s: 8 }, inside), three);
  assert.deepEqual(addRange(three, { start_s: 4.5, end_s: 6 }, inside).at(-1), {
    start_s: 4,
    end_s: 6,
  });
  assert.deepEqual(
    addRange([{ start_s: 0, end_s: 1 }], { start_s: 5, end_s: 6 }, one),
    [{ start_s: 5, end_s: 6 }],
  );
  assert.deepEqual(
    addRange(three, { start_s: 7, end_s: 8 }, { durationS: 10, maxRanges: 0 }),
    [],
  );
  assert.deepEqual(removeRange(three, 1), [three[0], three[2]]);
});

test("arrow keys nudge by 0.1 s, Shift+arrow by 1 s", () => {
  assert.equal(nudgeStep({ key: "ArrowRight" }), 0.1);
  assert.equal(nudgeStep({ key: "ArrowLeft" }), -0.1);
  assert.equal(nudgeStep({ key: "ArrowUp", shiftKey: true }), 1);
  assert.equal(nudgeStep({ key: "ArrowDown", shiftKey: true }), -1);
  assert.equal(nudgeStep({ key: "Enter" }), null);
  const ranges = [{ start_s: 2, end_s: 4 }];
  assert.deepEqual(moveRange(ranges, 0, "both", 0.1, inside), [
    { start_s: 2.1, end_s: 4.1 },
  ]);
  assert.deepEqual(moveRange(ranges, 0, "start", -1, inside), [
    { start_s: 1, end_s: 4 },
  ]);
  assert.deepEqual(moveRange(ranges, 0, "end", 1, inside), [
    { start_s: 2, end_s: 5 },
  ]);
  assert.deepEqual(moveRange(ranges, 0, "start", 5, inside), [
    { start_s: 3.75, end_s: 4 },
  ]);
  assert.deepEqual(moveRange(ranges, 0, "both", -5, inside), [
    { start_s: 0, end_s: 2 },
  ]);
  assert.deepEqual(moveRange(ranges, 0, "both", 50, inside), [
    { start_s: 8, end_s: 10 },
  ]);
  assert.deepEqual(
    moveRange(
      [
        { start_s: 1, end_s: 2 },
        { start_s: 3, end_s: 4 },
      ],
      0,
      "end",
      1.5,
      inside,
    ),
    [{ start_s: 1, end_s: 4 }],
  );
});

test("past the end only when the tail is allowed", () => {
  assert.deepEqual(moveRange([{ start_s: 8, end_s: 9 }], 0, "end", 3, inside), [
    { start_s: 8, end_s: 10 },
  ]);
  assert.deepEqual(moveRange([{ start_s: 8, end_s: 9 }], 0, "end", 3, tail), [
    { start_s: 8, end_s: 12 },
  ]);
  assert.deepEqual(moveRange([{ start_s: 8, end_s: 9 }], 0, "end", 30, tail), [
    { start_s: 8, end_s: 15 },
  ]);
  assert.equal(secondsBeyondEnd({ start_s: 8, end_s: 12 }, 10), 2);
  assert.equal(secondsBeyondEnd({ start_s: 1, end_s: 2 }, 10), 0);
  assert.deepEqual(dragToRange(9, 13, inside), { start_s: 9, end_s: 10 });
  assert.deepEqual(dragToRange(9, 13, tail), { start_s: 9, end_s: 13 });
});

test("a tiny drag is a click; a drag either way makes a range", () => {
  assert.equal(dragToRange(3, 3.05, inside), null);
  assert.deepEqual(dragToRange(5, 2, inside), { start_s: 2, end_s: 5 });
});

test("keyboard users get a first range and the next free gap", () => {
  assert.deepEqual(defaultRange(inside), { start_s: 4, end_s: 6 });
  const next = nextFreeRange([{ start_s: 4, end_s: 6 }], inside);
  assert.ok(next);
  assert.ok(
    next.end_s < 4 || next.start_s > 6,
    "lands in a gap, not on the existing part",
  );
  assert.equal(
    normalizeRanges([{ start_s: 4, end_s: 6 }, next], inside).length,
    2,
  );
  assert.equal(nextFreeRange([{ start_s: 0, end_s: 10 }], inside), null);
  assert.equal(nextFreeRange([{ start_s: 1, end_s: 2 }], one), null);
});

test("times read as 0:01.2 and ranges have spoken labels", () => {
  assert.equal(formatRangeTime(1.24), "0:01.2");
  assert.equal(formatRangeTime(62.5), "1:02.5");
  assert.equal(formatRangeTime(-3), "0:00.0");
  assert.equal(formatRange({ start_s: 1.2, end_s: 2.5 }), "0:01.2–0:02.5");
  assert.equal(
    rangeLabel(0, { start_s: 1.2, end_s: 2.5 }),
    "Range 1, 1.2 to 2.5 seconds",
  );
});

test("the range select is neutral, static and keyboard reachable", () => {
  const source = readSrc("features/audio/components/waveform-range-select.tsx");
  assert.match(
    source,
    /var\(--foreground\)_calc\(10%\*var\(--contrast-wash-gain,1\)\)/,
  );
  assert.match(source, /ring-1/);
  assert.match(source, /font-mono/);
  assert.match(
    source,
    /event\.key === "Delete" \|\| event\.key === "Backspace"/,
  );
  assert.match(source, /rangeLabel\(/);
  assert.match(source, /setPointerCapture/);
  assert.doesNotMatch(source, /animate-|transition-/);
  const waveform = readSrc("features/audio/components/waveform.tsx");
  assert.match(waveform, /border-dashed/);
  assert.doesNotMatch(waveform, /animate-|transition-/);
});
