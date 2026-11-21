// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  quotePillExpandedAfterCountChange,
  quotePillMode,
  quoteSourceLabel,
  shouldApplyQuotePreviewOpen,
  toggledPinnedQuoteId,
} from "../src/components/assistant-ui/message-quote-presentation.ts";

test("the quote pill is hidden, single, or multiple", () => {
  assert.equal(quotePillMode(0), "hidden");
  assert.equal(quotePillMode(1), "single");
  assert.equal(quotePillMode(2), "multiple");
  assert.equal(quotePillMode(5), "multiple");
});

test("visible quote source attribution is only the turn label", () => {
  assert.equal(quoteSourceLabel(4, (turn) => `Turn ${turn}`), "Turn 4");
  assert.equal(quoteSourceLabel(1, (turn) => `Turn ${turn}`), "Turn 1");
  assert.equal(quoteSourceLabel(0, (turn) => `Turn ${turn}`), "");
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

test("a touch pin survives Radix's temporary close and unpins on the second tap", () => {
  assert.equal(toggledPinnedQuoteId(null, "quote-a"), "quote-a");
  assert.equal(
    shouldApplyQuotePreviewOpen("quote-a", "quote-a", true),
    true,
    "opening still applies",
  );
  assert.equal(
    shouldApplyQuotePreviewOpen("quote-a", "quote-a", false),
    false,
    "Radix's pointerdown/click close must not clear the active pin",
  );
  assert.equal(
    toggledPinnedQuoteId("quote-a", "quote-a"),
    null,
    "second tap unpins",
  );
  assert.equal(
    shouldApplyQuotePreviewOpen(null, "quote-a", false),
    true,
    "after unpinning, the close applies",
  );
  assert.equal(
    shouldApplyQuotePreviewOpen("quote-a", "quote-b", false),
    true,
    "a stale pin for another quote cannot keep this preview open",
  );
  assert.equal(
    shouldApplyQuotePreviewOpen(null, null, false),
    true,
    "a close with no active quote still clears state",
  );
});
