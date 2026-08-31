// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatRelativeTime } from "../src/i18n/relative-time.ts";

// Listed by hand rather than imported from messages.ts, which reaches
// locale-store.ts and its extensionless imports that node cannot resolve.
// check-parity.ts enumerates the same set for the same reason.
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
  "hi",
  "ar",
] as const;

// Callers produce minutes up to 59, hours up to 23, months up to 12, and
// unbounded day and year counts. The sweep below checks values 1 through 60
// for every unit, covering the bounded ranges and representative larger counts.
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
  // CLDR's ar month-short past pattern for the "few" plural category (3-10)
  // carries خلال ("in"); formatRelativeTime falls back to the long style so
  // every magnitude keeps قبل ("ago").
  for (const value of [1, 2, 3, 5, 10, 11, 12]) {
    assert.match(
      formatRelativeTime("ar", -value, "month"),
      AR_PAST_MARKER,
      `ar -${value} month should read as past`,
    );
  }
});

test("non-finite values return empty text instead of throwing", () => {
  // An unparseable timestamp reaches the callers as NaN. format() answers
  // non-finite input with a RangeError, which would unmount the tree.
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
