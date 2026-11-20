// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  quotePillExpandedAfterCountChange,
  quotePillMode,
} from "../src/components/assistant-ui/message-quote-presentation.ts";

test("the quote pill is hidden, single, or multiple", () => {
  assert.equal(quotePillMode(0), "hidden");
  assert.equal(quotePillMode(1), "single");
  assert.equal(quotePillMode(2), "multiple");
  assert.equal(quotePillMode(5), "multiple");
});

test("the dropdown state survives count changes until the collection empties", () => {
  assert.equal(quotePillExpandedAfterCountChange(0, 1, true), false);
  assert.equal(quotePillExpandedAfterCountChange(1, 2, true), true);
  assert.equal(quotePillExpandedAfterCountChange(1, 2, false), false);
  assert.equal(quotePillExpandedAfterCountChange(2, 3, true), true);
  assert.equal(quotePillExpandedAfterCountChange(3, 4, false), false);
  assert.equal(quotePillExpandedAfterCountChange(2, 1, true), true);
  assert.equal(quotePillExpandedAfterCountChange(1, 0, true), false);
  assert.equal(quotePillExpandedAfterCountChange(0, 0, true), false);
});
