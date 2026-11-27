// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");

test("the disabled-control lookup on pointermove skips message rows instead of walking the thread", () => {
  const src = read("components/ui/title-tooltip.tsx");
  const fn = src.slice(src.indexOf("function disabledUnder"), src.indexOf("\n}\n", src.indexOf("function disabledUnder")));
  assert.match(fn, /hasAttribute\("data-role"\) \? NodeFilter\.FILTER_REJECT/);
  assert.match(fn, /if \(!hit\.querySelector\("\[data-role\]"\)\)/);
  // The skip relies on message rows (and only them) carrying data-role.
  const thread = read("components/assistant-ui/thread.tsx");
  assert.match(thread, /data-role="assistant"/);
  assert.match(thread, /data-role="user"/);
});
