// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearQuotePillPosition,
  getQuotePillPosition,
  setQuotePillPosition,
} from "../src/components/assistant-ui/message-quote-position-store.ts";

test("pill positions default to missing and round-trip per thread", () => {
  const thread = "quote-pill-position-a";

  assert.equal(getQuotePillPosition(thread), null);
  setQuotePillPosition(thread, { left: 10, top: 20 });
  assert.deepEqual(getQuotePillPosition(thread), { left: 10, top: 20 });

  setQuotePillPosition(thread, { left: 30, top: 40 });
  assert.deepEqual(getQuotePillPosition(thread), { left: 30, top: 40 });
  assert.equal(clearQuotePillPosition(thread), true);
  assert.equal(getQuotePillPosition(thread), null);
  assert.equal(clearQuotePillPosition(thread), false);
});

test("pill positions are isolated by thread", () => {
  const left = "quote-pill-position-left";
  const right = "quote-pill-position-right";

  setQuotePillPosition(left, { left: 1, top: 2 });
  setQuotePillPosition(right, { left: 3, top: 4 });

  assert.deepEqual(getQuotePillPosition(left), { left: 1, top: 2 });
  assert.deepEqual(getQuotePillPosition(right), { left: 3, top: 4 });

  clearQuotePillPosition(left);
  assert.equal(getQuotePillPosition(left), null);
  assert.deepEqual(getQuotePillPosition(right), { left: 3, top: 4 });
});
