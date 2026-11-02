// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  TIME_FORMAT_STORAGE_KEY,
  getTimeFormat,
  setTimeFormat,
  withTimeFormat,
} from "../src/lib/time-format.ts";
import { formatMessageDate } from "../src/lib/format-message-date.ts";

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

const at = new Date(2026, 8, 25, 21, 5);
const labels = {
  today: (time: string) => time,
  yesterday: (time: string) => time,
};
const browser = new Intl.DateTimeFormat(undefined, {
  hour: "numeric",
}).resolvedOptions();
const browserLanguage = browser.locale.split("-")[0];
const otherLanguage = browserLanguage === "de" ? "fr" : "de";

test.afterEach(() => store.clear());

test("an explicit 12- or 24-hour choice applies in every language", () => {
  setTimeFormat("24h");
  assert.equal(
    formatMessageDate(at.getTime(), at.getTime(), "en", labels),
    "21:05",
  );
  setTimeFormat("12h");
  assert.equal(
    formatMessageDate(at.getTime(), at.getTime(), "de", labels),
    "9:05 PM",
  );
});

test("system default follows the browser's clock when the app speaks its language", () => {
  const expected =
    browser.hourCycle === "h11" || browser.hourCycle === "h12" ? "h12" : "h23";
  assert.equal(
    withTimeFormat({ hour: "numeric" }, browserLanguage).hourCycle,
    expected,
  );
});

test("system default leaves another language's own clock alone", () => {
  const options = { hour: "numeric", minute: "2-digit" } as const;
  assert.deepEqual(withTimeFormat(options, otherLanguage), options);
  assert.deepEqual(withTimeFormat(options), options);
});

test("hour12 cannot outrank the chosen clock", () => {
  setTimeFormat("24h");
  assert.deepEqual(withTimeFormat({ hour: "numeric", hour12: true }), {
    hour: "numeric",
    hourCycle: "h23",
  });
});

test("system default is stored as no value, and junk reads as system", () => {
  setTimeFormat("24h");
  assert.equal(store.get(TIME_FORMAT_STORAGE_KEY), "24h");
  setTimeFormat("system");
  assert.equal(store.has(TIME_FORMAT_STORAGE_KEY), false);
  store.set(TIME_FORMAT_STORAGE_KEY, "36h");
  assert.equal(getTimeFormat(), "system");
});
