// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatRelativeTime } from "../src/i18n/relative-time.ts";

// Hand-listed like check-parity.ts: messages.ts has extensionless imports node cannot resolve.
const LOCALE_LIST = [
  "en",
  "zh-CN",
  "ja",
  "ko",
  "es",
  "pt-BR",
  "fr",
  "de",
  "it",
  "ru",
  "sv",
  "hi",
  "ar",
] as const;

const AR_PAST_MARKER = /^قبل/;

const UNITS: Intl.RelativeTimeFormatUnit[] = [
  "minute",
  "hour",
  "day",
  "month",
  "year",
];

test("a past time never reads as a future time", () => {
  for (const locale of LOCALE_LIST) {
    for (const unit of UNITS) {
      for (let value = 1; value <= 60; value++) {
        assert.notEqual(
          formatRelativeTime(locale, -value, unit),
          formatRelativeTime(locale, value, unit),
          `${locale} ${value} ${unit} reads the same in both directions`,
        );
      }
    }
  }
});

test("Arabic past months keep the past marker", () => {
  // CLDR ar month-short past for "few" uses "in", so formatRelativeTime falls back to long style.
  for (const value of [1, 2, 3, 5, 10, 11, 12]) {
    assert.match(
      formatRelativeTime("ar", -value, "month"),
      AR_PAST_MARKER,
      `ar -${value} month should read as past`,
    );
  }
});

test("non-finite values return empty text instead of throwing", () => {
  // format() throws RangeError on NaN, which would unmount the tree.
  for (const value of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ]) {
    for (const locale of LOCALE_LIST) {
      assert.equal(formatRelativeTime(locale, value, "day"), "");
    }
  }
});

test("Arabic dual and plural relative times stay distinct", () => {
  assert.notEqual(
    formatRelativeTime("ar", -2, "day"),
    formatRelativeTime("ar", -3, "day"),
  );
});

test("cached formatters do not leak across locales", () => {
  const perLocale = LOCALE_LIST.map((locale) =>
    formatRelativeTime(locale, -2, "day"),
  );
  assert.equal(new Set(perLocale).size > 1, true);
  assert.notEqual(
    formatRelativeTime("ja", -2, "day"),
    formatRelativeTime("de", -2, "day"),
  );
});
