// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatCompactNumber,
  formatDayCount,
  formatDuration,
  formatFullNumber,
  formatMilliseconds,
  formatProfileCount,
  heatLevel,
  parseDayKey,
  seriesForMode,
  windowBaseline,
} from "../src/features/profile/utils/stats-format.ts";

test("compact numbers match the tile format", () => {
  assert.equal(formatCompactNumber(0, "en"), "0");
  assert.equal(formatCompactNumber(999, "en"), "999");
  assert.equal(formatCompactNumber(1000, "en"), "1K");
  assert.equal(formatCompactNumber(12_340, "en"), "12.3K");
  assert.equal(formatCompactNumber(1_900_000_000, "en"), "1.9B");
  assert.equal(formatCompactNumber(19_800_000_000, "en"), "19.8B");
  // Past 100 of a unit the decimal is noise.
  assert.equal(formatCompactNumber(123_400, "en"), "123K");
  assert.equal(formatCompactNumber(Number.NaN, "en"), "0");
});

test("sub-unit counts stay whole, in every locale", () => {
  // averageTokensPerChat is the one fractional caller and these are whole tokens:
  // 25 across 2 chats is 13, not 12.5. The pre-localization code rounded anything
  // under 1000; the unit boundary replaces that hardcoded 1000 because it is
  // per-locale (ja and de do not compact until \u4e07 / Mio.).
  assert.equal(formatCompactNumber(12.5, "en"), "13");
  assert.equal(formatCompactNumber(12.4, "en"), "12");
  assert.equal(formatCompactNumber(0.5, "en"), "1");
  // Intl rounds half away from zero, Math.round rounds half UP, so an exact negative
  // half differs from the pre-localization code (-13 vs -12). Every caller here is a
  // count, so this is unreachable in practice; pinned so the difference is deliberate.
  assert.equal(formatCompactNumber(-12.5, "en"), "-13");
  // ja does not compact below \u4e07, so a four-digit value is still whole there.
  assert.equal(formatCompactNumber(5000.4, "ja"), "5000");
  // ...and past the unit the decimal comes back.
  assert.equal(formatCompactNumber(12_340, "en"), "12.3K");
  assert.equal(formatCompactNumber(12_340, "ja"), "1.2\u4e07");
});

test("a non-Latin numbering system keeps its decimal", () => {
  // The threshold used to be read off the DISPLAY string, so a locale whose
  // digits are not ASCII gave Number("\u0661") -> NaN, NaN < 100 -> false, and every
  // Arabic value silently lost its decimal. Which locales default to `arab`
  // varies by ICU build, so pin an explicit one rather than bare "ar".
  const arab = "ar-EG" as Parameters<typeof formatCompactNumber>[1];
  const onePointNine = formatCompactNumber(1_900_000, arab);
  // The decimal separator is the Arabic one; what matters is that a fraction survived.
  assert.ok(
    /\u0661[\u066b.,]\u0669/.test(onePointNine),
    `expected a 1.9-style value, got ${onePointNine}`,
  );
  // ...and past 100 of a unit it is still dropped, exactly as in en.
  assert.ok(
    !/[\u066b.,]/.test(formatCompactNumber(190_000_000, arab)),
    "expected no decimal past 100 of a unit",
  );
});

test("the latn probe does not change which unit Intl picked", () => {
  // Unit grouping is a locale property, not a numbering-system one, so probing
  // in latn must leave ja/zh (\u4e07) and hi (\u0932\u093e\u0916) exactly as they were.
  assert.equal(formatCompactNumber(1_900_000, "ja"), "190\u4e07");
  assert.equal(formatCompactNumber(190_000_000, "ja"), "1.9\u5104");
  assert.equal(formatCompactNumber(1_900_000, "zh-CN"), "190\u4e07");
  assert.equal(formatCompactNumber(1_234, "en"), "1.2K");
});

test("rounding up a unit steps to the next suffix", () => {
  // Rounding 999.5K to "1000K" is four digits, which is not compact.
  assert.equal(formatCompactNumber(999_999, "en"), "1M");
  assert.equal(formatCompactNumber(999_500, "en"), "1M");
  assert.equal(formatCompactNumber(999_999_999, "en"), "1B");
  assert.equal(formatCompactNumber(999_999_999_999, "en"), "1T");
  assert.equal(formatCompactNumber(-999_999, "en"), "-1M");
  // Just below the rounding boundary the unit is unchanged.
  assert.equal(formatCompactNumber(999_499, "en"), "999K");
});

test("compact numbers use each locale's own magnitude units", () => {
  // K/M/B is an English convention. ja and ko group in 万/억, zh in 万/亿,
  // and hi in लाख, so a hardcoded ladder is wrong in half the locales.
  assert.equal(formatCompactNumber(12_340, "ja"), "1.2万");
  assert.equal(formatCompactNumber(1_900_000_000, "ja"), "19億");
  assert.equal(formatCompactNumber(12_340, "zh-CN"), "1.2万");
  assert.equal(formatCompactNumber(1_900_000_000, "zh-CN"), "19亿");
  // U+00A0, not a plain space: CLDR keeps the unit from wrapping away from
  // its number, so an assertion with a normal space silently fails.
  assert.equal(formatCompactNumber(1_900_000, "hi"), "19\u00a0लाख");
  // Decimal separator and unit word follow the locale too.
  assert.equal(formatCompactNumber(1_900_000, "de"), "1,9\u00a0Mio.");
  assert.equal(formatCompactNumber(1_900_000, "ru"), "1,9\u00a0млн");
  // German CLDR has no short form below a million, so thousands stay written
  // out. That is the locale's rule, not a fallback.
  assert.equal(formatCompactNumber(12_340, "de"), "12.340");
});

test("full numbers group the way the chosen locale groups", () => {
  assert.equal(formatFullNumber(1_234_567, "en"), "1,234,567");
  assert.equal(formatFullNumber(1_234_567, "de"), "1.234.567");
  // Indian grouping is 2-2-3, not 3-3-3.
  assert.equal(formatFullNumber(1_234_567, "hi"), "12,34,567");
  assert.equal(formatFullNumber(Number.NaN, "en"), "0");
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
  assert.equal(formatProfileCount(5, "token", "en", "$&"), "$& tokens");
  assert.equal(formatProfileCount(5, "token", "en", "$'"), "$' tokens");

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
