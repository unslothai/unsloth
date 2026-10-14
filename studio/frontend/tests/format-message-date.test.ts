// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { formatMessageDate } = await import("../src/lib/format-message-date.ts");

const now = new Date(2026, 8, 25, 12, 0).getTime();
const labels = {
  today: (time: string) => `Today, ${time}`,
  yesterday: (time: string) => `Yesterday, ${time}`,
};
const at = (...parts: [number, number, number, number, number]) =>
  formatMessageDate(new Date(...parts).getTime(), now, "en", labels);

test("today and yesterday go by calendar day, not elapsed time", () => {
  assert.match(at(2026, 8, 25, 4, 45), /^Today, 4:45\s?AM$/);
  assert.match(at(2026, 8, 25, 0, 0), /^Today, 12:00\s?AM$/);
  assert.match(at(2026, 8, 24, 23, 59), /^Yesterday, 11:59\s?PM$/);
  assert.match(at(2026, 8, 24, 0, 1), /^Yesterday, 12:01\s?AM$/);
});

test("older messages show the date, with the year only when it differs", () => {
  assert.match(at(2026, 8, 23, 21, 12), /^Sep 23, 9:12\s?PM$/);
  assert.match(at(2025, 8, 13, 21, 12), /2025/);
  // Across New Year, the day before is still yesterday.
  const jan1 = new Date(2027, 0, 1, 9, 0).getTime();
  assert.match(
    formatMessageDate(new Date(2026, 11, 31, 22, 0).getTime(), jan1, "en", labels),
    /^Yesterday, 10:00\s?PM$/,
  );
});
