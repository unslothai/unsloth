// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatCompactNumber,
  formatDayCount,
  formatDuration,
  formatMilliseconds,
  formatProfileCount,
  heatLevel,
  parseDayKey,
  seriesForMode,
  windowBaseline,
} from "../src/features/profile/utils/stats-format.ts";

test("compact numbers match the tile format", () => {
  assert.equal(formatCompactNumber(0), "0");
  assert.equal(formatCompactNumber(999), "999");
  assert.equal(formatCompactNumber(1000), "1K");
  assert.equal(formatCompactNumber(12_340), "12.3K");
  assert.equal(formatCompactNumber(1_900_000_000), "1.9B");
  assert.equal(formatCompactNumber(19_800_000_000), "19.8B");
  // Past 100 of a unit the decimal is noise.
  assert.equal(formatCompactNumber(123_400), "123K");
  assert.equal(formatCompactNumber(Number.NaN), "0");
});

test("rounding up a unit steps to the next suffix", () => {
  // Rounding 999.5K to "1000K" is four digits, which is not compact.
  assert.equal(formatCompactNumber(999_999), "1M");
  assert.equal(formatCompactNumber(999_500), "1M");
  assert.equal(formatCompactNumber(999_999_999), "1B");
  assert.equal(formatCompactNumber(999_999_999_999), "1T");
  assert.equal(formatCompactNumber(-999_999), "-1M");
  // Just below the rounding boundary the unit is unchanged.
  assert.equal(formatCompactNumber(999_499), "999K");
});

test("durations read the way the header does", () => {
  assert.equal(formatDuration(0), "0m");
  assert.equal(formatDuration(45), "45s");
  assert.equal(formatDuration(90), "1m 30s");
  assert.equal(formatDuration(14_880), "4h 8m");
  assert.equal(formatDuration(180_000), "2d 2h");
  assert.equal(formatMilliseconds(420), "420ms");
  assert.equal(formatMilliseconds(2500), "2.5s");
  assert.equal(formatMilliseconds(0), "—");
});

test("day counts follow each locale's plural rules", () => {
  assert.equal(formatDayCount(1, "en"), "1 day");
  assert.equal(formatDayCount(2, "en"), "2 days");
  assert.equal(formatDayCount(1, "it"), "1 giorno");
  assert.equal(formatDayCount(2, "it"), "2 giorni");
  assert.equal(formatDayCount(1, "ar"), "يوم");
  assert.equal(formatDayCount(2, "ar"), "يومان");
  assert.equal(formatDayCount(3, "ar"), "3 أيام");
  assert.equal(formatDayCount(11, "ar"), "11 يومًا");
  assert.equal(formatDayCount(100, "ar"), "100 يوم");
});

test("profile counts follow the selected locale's plural rules", () => {
  assert.equal(formatProfileCount(1, "step", "en"), "1 step");
  assert.equal(formatProfileCount(2, "step", "en"), "2 steps");
  assert.equal(formatProfileCount(1, "week", "es"), "1 semana");
  assert.equal(formatProfileCount(2, "week", "es"), "2 semanas");

  assert.equal(formatProfileCount(1, "week", "it"), "1 settimana");
  assert.equal(formatProfileCount(2, "week", "it"), "2 settimane");
  assert.equal(formatProfileCount(1, "message", "it"), "1 messaggio");
  assert.equal(formatProfileCount(2, "message", "it"), "2 messaggi");
  assert.equal(formatProfileCount(2, "token", "it"), "2 token");
  assert.equal(formatProfileCount(2, "step", "it"), "2 step");

  assert.equal(formatProfileCount(1, "step", "ru"), "1 шаг");
  assert.equal(formatProfileCount(2, "step", "ru"), "2 шага");
  assert.equal(formatProfileCount(5, "step", "ru"), "5 шагов");
  assert.equal(formatProfileCount(21, "step", "ru"), "21 шаг");

  assert.equal(formatProfileCount(1, "message", "ar"), "رسالة واحدة");
  assert.equal(formatProfileCount(2, "message", "ar"), "رسالتان");
  assert.equal(formatProfileCount(3, "message", "ar"), "3 رسائل");
  assert.equal(formatProfileCount(11, "message", "ar"), "11 رسالة");
  assert.equal(formatProfileCount(100, "message", "ar"), "100 رسالة");
  assert.equal(formatProfileCount(1200, "token", "ar", "1.2K"), "1.2K توكن");
});

test("heat levels are relative to the busiest day", () => {
  assert.equal(heatLevel(0, 1000), 0);
  assert.equal(heatLevel(50, 1000), 1);
  assert.equal(heatLevel(200, 1000), 2);
  assert.equal(heatLevel(400, 1000), 3);
  assert.equal(heatLevel(1000, 1000), 4);
  // A single active day with no other history still shows up.
  assert.equal(heatLevel(5, 0), 1);
});

test("day keys parse as local dates, not UTC", () => {
  const parsed = parseDayKey("2026-03-09");
  assert.equal(parsed.getFullYear(), 2026);
  assert.equal(parsed.getMonth(), 2);
  assert.equal(parsed.getDate(), 9);
});

test("series modes reshape the same daily data", () => {
  // 2026-03-02 is a Monday, so this spans exactly two calendar weeks.
  const daily = [
    { date: "2026-03-02", tokens: 10 },
    { date: "2026-03-03", tokens: 20 },
    { date: "2026-03-08", tokens: 5 },
    { date: "2026-03-09", tokens: 100 },
  ];

  assert.deepEqual(seriesForMode(daily, "daily"), [10, 20, 5, 100]);
  assert.deepEqual(seriesForMode(daily, "cumulative"), [10, 30, 35, 135]);
  // First three days are in the week of Mar 2 (35), Mar 9 starts a new week.
  assert.deepEqual(seriesForMode(daily, "weekly"), [35, 35, 35, 100]);
  assert.deepEqual(seriesForMode([], "weekly"), []);
});

test("a trimmed cumulative window rebases off the last hidden day", () => {
  const daily = [
    { date: "2026-01-01", tokens: 1000 },
    { date: "2026-01-02", tokens: 2000 },
    { date: "2026-01-03", tokens: 5 },
    { date: "2026-01-04", tokens: 10 },
  ];
  const values = seriesForMode(daily, "cumulative");
  assert.deepEqual(values, [1000, 3000, 3005, 3015]);

  // Showing only the last two days: without rebasing, both bars sit at ~3000
  // and the 5 vs 10 difference is invisible.
  const baseline = windowBaseline(values, 2, "cumulative");
  assert.equal(baseline, 3000);
  assert.deepEqual(
    values.slice(2).map((value) => value - baseline),
    [5, 15],
  );
});

test("nothing is rebased when the window shows everything", () => {
  const values = [1, 3, 6];
  assert.equal(windowBaseline(values, 0, "cumulative"), 0);
});

test("only cumulative rebases: daily and weekly are already per-window", () => {
  const values = [10, 20, 5];
  assert.equal(windowBaseline(values, 2, "daily"), 0);
  assert.equal(windowBaseline(values, 2, "weekly"), 0);
});
