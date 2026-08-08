// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The network catalog check promises that an unreachable Hub produces warnings, never a red run.
// Bounding each request is not enough to keep that promise: the batches are serial, so a peer
// that stalls every one of them can still outlive the workflow's own timeout and be killed --
// which is the red run, arriving by a different route. This pins the arithmetic.

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const CHECK = fileURLToPath(
  new URL(
    "../src/features/model-picker/components/model-selector/model-catalog.check.ts",
    import.meta.url,
  ),
);
const WORKFLOW = fileURLToPath(
  new URL("../../../.github/workflows/model-catalog-network-check.yml", import.meta.url),
);

function constant(name: string): number {
  const source = readFileSync(CHECK, "utf8");
  const match = source.match(new RegExp(`const ${name} = ([^;]+);`));
  assert.ok(match, `model-catalog.check.ts no longer defines ${name}`);
  // The literals are written for readers (7 * 60 * 1000, 20_000), so evaluate rather than parse.
  return Number(new Function(`return (${match[1].replaceAll("_", "")})`)());
}

test("the whole network pass is bounded, not just each request", () => {
  const source = readFileSync(CHECK, "utf8");
  assert.match(
    source,
    /Date\.now\(\) >= networkDeadlineAt/,
    "fetchWithRetry must short-circuit once the overall budget is spent",
  );
  assert.match(
    source,
    /networkDeadlineAt = Date\.now\(\) \+ NETWORK_DEADLINE_MS/,
    "the deadline has to be armed when the network pass starts",
  );
});

test("the overall budget leaves room inside the workflow timeout", () => {
  const workflow = readFileSync(WORKFLOW, "utf8");
  const timeout = workflow.match(/timeout-minutes:\s*(\d+)/);
  assert.ok(timeout, "the network-check workflow no longer declares timeout-minutes");
  const jobBudgetMs = Number(timeout[1]) * 60 * 1000;

  // Worst case after the deadline fires: the batch already in flight still has to unwind.
  const perBatchMs =
    constant("NETWORK_ATTEMPTS") * constant("NETWORK_TIMEOUT_MS") + 500 + 1000;
  const worstCaseMs = constant("NETWORK_DEADLINE_MS") + perBatchMs;

  assert.ok(
    worstCaseMs < jobBudgetMs,
    `a fully stalled Hub takes ${(worstCaseMs / 60000).toFixed(1)} min to fail open, and the ` +
      `job is killed at ${Number(timeout[1])} min -- the red run this retry logic exists to avoid`,
  );
});
