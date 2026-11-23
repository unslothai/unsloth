// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const selector = readSrc("features/model-picker/components/model-selector.tsx");

test("the suffix shrinks first, then the description, and the name last", () => {
  assert.match(selector, /"min-w-0 shrink-\[1000000\] truncate whitespace-nowrap text-xs leading-tight text-muted-foreground"/);
  assert.match(selector, /"min-w-0 shrink-\[1000\] truncate text-xs leading-tight text-muted-foreground"/);
  assert.match(selector, /"flex max-w-full shrink-0 items-baseline whitespace-nowrap font-heading/);
});

test("no truncating span pairs truncate with leading-none, which clips descenders", () => {
  // At text-xs a 12px line box clips descenders under truncate.
  for (const literal of selector.match(/"[^"\n]*"/g) ?? []) {
    const tokens = literal.slice(1, -1).split(/\s+/);
    assert.ok(
      !(tokens.includes("truncate") && tokens.includes("leading-none")),
      `truncate with leading-none clips descenders: ${literal}`,
    );
  }
});

test("the gap before the description or suffix truncates away with it", () => {
  assert.match(selector, /const GAP_BEFORE = "before:inline-block before:w-2 before:content-\[''\]";/);
  assert.doesNotMatch(selector, /"ml-2"/);
});
