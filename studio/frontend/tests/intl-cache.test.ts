// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { formatCardTime, formatSize } from "../src/features/library/format.ts";
import { formatDate, formatNumber } from "../src/lib/intl-cache.ts";
import {
  type TimeFormat,
  setTimeFormat,
  withTimeFormat,
} from "../src/lib/time-format.ts";

const store = new Map<string, string>();
(globalThis as { localStorage?: unknown }).localStorage = {
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => {
    store.set(key, value);
  },
  removeItem: (key: string) => {
    store.delete(key);
  },
};

test.afterEach(() => store.clear());

const browserLanguage = new Intl.DateTimeFormat(undefined, { hour: "numeric" })
  .resolvedOptions()
  .locale.split("-")[0];
const LOCALES = [
  undefined,
  "en",
  "en-US",
  "en-GB",
  "de-DE",
  "ja-JP",
  "ar-EG",
  "hi-IN",
  browserLanguage,
];
const FORMATS: TimeFormat[] = ["system", "12h", "24h"];
const NOW = new Date(2026, 8, 25, 12, 0).getTime();
const STAMPS = Array.from(
  { length: 400 },
  (_, i) => NOW - i * 3_600_000 * 7.3 + i * 61_000,
);

// The uncached calls format.ts made before the cache.
function referenceCardTime(
  ts: number,
  locale: string | undefined,
  now: number,
): string {
  const then = new Date(ts);
  const today = new Date(now);
  const start = (d: Date) =>
    new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((start(today) - start(then)) / 86_400_000);
  if (days === 0) {
    return then.toLocaleTimeString(
      locale,
      withTimeFormat({ hour: "numeric", minute: "2-digit" }, locale),
    );
  }
  if (days > 1 && days < 7)
    return then.toLocaleDateString(locale, { weekday: "long" });
  return then.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === today.getFullYear() ? undefined : "numeric",
  });
}

test("cached card times match the uncached toLocale calls in every locale and clock", () => {
  for (const format of FORMATS) {
    setTimeFormat(format);
    for (const locale of LOCALES) {
      for (const ts of STAMPS) {
        const then = new Date(ts);
        const days = Math.round(
          (new Date(2026, 8, 25).getTime() -
            new Date(
              then.getFullYear(),
              then.getMonth(),
              then.getDate(),
            ).getTime()) /
            86_400_000,
        );
        if (days === 1) continue; // "Yesterday" comes from RelativeTimeFormat, unchanged.
        assert.equal(
          formatCardTime(ts, locale as never, NOW),
          referenceCardTime(ts, locale, NOW),
          `${format} ${locale} ${ts}`,
        );
      }
    }
  }
});

test("cached sizes match Number.toLocaleString", () => {
  const t = (_key: string, values?: { value?: unknown }) =>
    String(values?.value);
  for (const locale of LOCALES) {
    for (let bytes = 0; bytes < 2 ** 44; bytes = bytes * 3 + 7) {
      const value =
        bytes < 1024
          ? bytes
          : bytes / 1024 ** Math.min(4, Math.floor(Math.log2(bytes) / 10));
      const digits = value >= 10 ? 0 : 1;
      const expected =
        bytes < 1024
          ? bytes.toLocaleString(locale, { useGrouping: false })
          : value.toLocaleString(locale, {
              minimumFractionDigits: digits,
              maximumFractionDigits: digits,
            });
      assert.equal(
        formatSize(bytes, locale as never, t as never),
        expected,
        `${locale} ${bytes}`,
      );
      assert.equal(
        formatNumber(value, locale, { maximumFractionDigits: 1 }),
        value.toLocaleString(locale, { maximumFractionDigits: 1 }),
      );
    }
  }
});

test("an invalid date formats as the toLocale calls do instead of throwing", () => {
  const invalid = new Date(Number.NaN);
  assert.equal(
    formatDate(invalid, "en", { weekday: "long" }),
    invalid.toLocaleDateString("en", { weekday: "long" }),
  );
  assert.equal(
    formatDate(invalid, "de-DE", { hour: "numeric" }),
    invalid.toLocaleTimeString("de-DE", { hour: "numeric" }),
  );
  // Finite but past the Date range: formatCardTime's own isFinite guard does not catch it.
  assert.equal(
    formatCardTime(1e17, "en" as never, NOW),
    new Date(1e17).toLocaleDateString("en", { month: "short", day: "numeric" }),
  );
});

test("the system clock still follows the locale's language on every call", () => {
  setTimeFormat("system");
  const browser = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
  }).resolvedOptions();
  const expected =
    browser.hourCycle === "h11" || browser.hourCycle === "h12" ? "h12" : "h23";
  for (let i = 0; i < 3; i++) {
    assert.equal(
      withTimeFormat({ hour: "numeric" }, browserLanguage).hourCycle,
      expected,
    );
    assert.equal(
      withTimeFormat(
        { hour: "numeric" },
        browserLanguage === "de" ? "fr" : "de",
      ).hourCycle,
      undefined,
    );
  }
  setTimeFormat("24h");
  assert.equal(
    withTimeFormat({ hour: "numeric" }, browserLanguage).hourCycle,
    "h23",
  );
});

test("a time zone change while the page is open formats in the new zone", () => {
  const previous = process.env.TZ;
  const when = new Date(Date.UTC(2026, 3, 10, 22, 10));
  const options: Intl.DateTimeFormatOptions = { hour: "numeric", minute: "2-digit" };
  try {
    process.env.TZ = "America/New_York";
    assert.equal(formatDate(when, "en-US", options), when.toLocaleString("en-US", options));
    process.env.TZ = "Asia/Tokyo";
    assert.equal(formatDate(when, "en-US", options), when.toLocaleString("en-US", options));
  } finally {
    if (previous === undefined) delete process.env.TZ;
    else process.env.TZ = previous;
  }
});
