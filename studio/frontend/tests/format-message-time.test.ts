// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { formatMessageTime } = await import("../src/lib/format-message-time.ts");

const now = new Date(2026, 8, 25, 12, 0).getTime();
const ago = (ms: number) => formatMessageTime(now - ms, now, "en", "just now");
const MIN = 60_000;

test("recent messages read as elapsed time", () => {
  assert.equal(ago(0), "just now");
  assert.equal(ago(59 * 1000), "just now");
  assert.equal(ago(3 * MIN), "3 min. ago");
  assert.equal(ago(59 * MIN), "59 min. ago");
  assert.equal(ago(3 * 60 * MIN), "3 hr. ago");
  assert.equal(ago(2 * 24 * 60 * MIN), "2 days ago");
  // A clock that runs behind the message never reads as the future.
  assert.equal(formatMessageTime(now + 5 * MIN, now, "en", "just now"), "just now");
});

test("a week or more shows the date, with the year only when it differs", () => {
  const sameYear = formatMessageTime(new Date(2026, 8, 13, 21, 12).getTime(), now, "en", "just now");
  assert.match(sameYear, /^Sep 13, 9:12\s?PM$/);
  const lastYear = formatMessageTime(new Date(2025, 8, 13, 21, 12).getTime(), now, "en", "just now");
  assert.match(lastYear, /2025/);
});
