// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  mergeJsonValue,
  shouldPreferIncomingTerminalScalars,
} from "../src/features/recipe-studio/data/execution-reconciliation.ts";

test("equal terminal events enrich without replacing persisted scalars", () => {
  const preferIncoming = shouldPreferIncomingTerminalScalars(12, 12);

  assert.equal(preferIncoming, false);
  assert.deepEqual(
    mergeJsonValue(
      { metrics: { score: 0.9, rows: 10 } },
      { metrics: { score: 0.1, duration: 5 } },
      preferIncoming,
    ),
    { metrics: { score: 0.9, rows: 10, duration: 5 } },
  );
});

test("newer terminal events replace stale scalars while retaining enrichment", () => {
  const preferIncoming = shouldPreferIncomingTerminalScalars(13, 12);

  assert.equal(preferIncoming, true);
  assert.deepEqual(
    mergeJsonValue(
      { metrics: { score: 0.1, rows: 10 } },
      { metrics: { score: 0.9, duration: 5 } },
      preferIncoming,
    ),
    { metrics: { score: 0.9, rows: 10, duration: 5 } },
  );
});
