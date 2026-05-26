// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { clampLines } from "../src/components/assistant-ui/model-selector/chat-template-preview.ts";

test("chat template preview clamps regular text by lines", () => {
  assert.equal(clampLines("abcdefg", 2, 3), "abc\nde…");
  assert.equal(clampLines("abc\ndef", 3, 3), "abc\ndef");
});

test("chat template preview preserves emoji graphemes", () => {
  const family = "👨‍👩‍👧‍👦";

  assert.equal(clampLines(`${family}ab`, 1, 2), `${family}…`);
});

test("chat template preview preserves combining-mark graphemes", () => {
  const eAcute = "e\u0301";

  assert.equal(clampLines(`${eAcute}bc`, 1, 2), `${eAcute}…`);
});
