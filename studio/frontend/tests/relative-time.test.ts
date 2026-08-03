// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatRelativeTime } from "../src/i18n/relative-time.ts";

const OPTIONS: Intl.RelativeTimeFormatOptions = {
  numeric: "always",
  style: "short",
};

test("relative times preserve locale, direction, value, and unit", () => {
  const cases = [
    ["en", -2, "day"],
    ["ar", -2, "day"],
    ["ar", -3, "day"],
    ["ru", -2, "hour"],
    ["ja", 2, "month"],
  ] as const;

  for (const [locale, value, unit] of cases) {
    const expected = new Intl.RelativeTimeFormat(locale, OPTIONS).format(
      value,
      unit,
    );
    assert.equal(formatRelativeTime(locale, value, unit), expected);
  }
});

test("Arabic dual and plural relative times stay distinct", () => {
  assert.notEqual(
    formatRelativeTime("ar", -2, "day"),
    formatRelativeTime("ar", -3, "day"),
  );
});
