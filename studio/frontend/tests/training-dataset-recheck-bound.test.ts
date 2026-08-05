// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { DatasetCacheRejectionTracker } = await import(
  "../src/features/training/lib/dataset-cache-rejection.ts"
);
const {
  claimDatasetCacheRecheck,
  datasetCacheRecheckKey,
  resetDatasetCacheRecheckBudget,
} = await import("../src/features/training/lib/dataset-recheck-budget.ts");

function usabilityIdentity(overrides: Record<string, unknown> = {}) {
  return {
    dataset: "Org/Data",
    cachePath: "/cache/datasets--Org--Data",
    subset: "default",
    split: "train",
    streaming: false,
    ...overrides,
  };
}

/**
 * Replays the decision sequence in training-config-store.ts runDatasetCheck: begin a
 * validation, let an inventory poll land while it is in flight, then either settle or
 * re-fire. Drives the real tracker and the real retry budget.
 */
function driveStoreRetryLoop(
  churn: boolean,
  maxIterations = 500,
): { requests: number; terminated: boolean } {
  resetDatasetCacheRecheckBudget();
  const tracker = new DatasetCacheRejectionTracker();
  const identity = usabilityIdentity();
  const key = datasetCacheRecheckKey("Org/Data", "train");
  let sizeBytes = 128;
  let requests = 0;

  const poll = () => ({
    cachePath: "/cache/datasets--Org--Data",
    sizeBytes: churn ? (sizeBytes += 64) : sizeBytes,
    partial: churn,
    partialTransport: null,
  });

  // The generation only advances once a prior inventory row has been seen, which is
  // the steady state by the time a user has a dataset selected.
  tracker.observe(identity, poll());

  for (let i = 0; i < maxIterations; i += 1) {
    requests += 1;
    const token = tracker.beginValidation(identity);
    tracker.observe(identity, poll());

    if (tracker.rejectValidation(token)) {
      return { requests, terminated: true };
    }
    // Stale generation: the store re-fires only while the budget allows it, otherwise
    // it falls through to the uncached check, which ends the loop.
    if (!claimDatasetCacheRecheck(key)) {
      return { requests, terminated: true };
    }
  }
  return { requests, terminated: false };
}

test("a settled cache inventory terminates the dataset re-check promptly", () => {
  const { requests, terminated } = driveStoreRetryLoop(false);
  assert.equal(terminated, true);
  assert.ok(
    requests <= 2,
    `a stable inventory must settle within 2 checks, took ${requests}`,
  );
});

test("a churning cache inventory cannot drive unbounded dataset re-checks", () => {
  // Regression guard for unslothai/unsloth#7853: while a dataset downloads, sizeBytes
  // changes on every poll, so every in-flight validation is invalidated. Without a
  // bound the store re-fires forever (measured at 480 requests in 60s).
  const { requests, terminated } = driveStoreRetryLoop(true);
  assert.ok(
    terminated,
    `re-check never settled under inventory churn: ${requests} requests and still going`,
  );
  assert.ok(
    requests <= 8,
    `inventory churn must not drive unbounded re-checks, saw ${requests}`,
  );
});

test("switching dataset selection starts a fresh re-check budget", () => {
  resetDatasetCacheRecheckBudget();
  const first = datasetCacheRecheckKey("Org/Data", "train");
  const second = datasetCacheRecheckKey("Org/Data", "validation");

  let drained = 0;
  // Bounded so an unbounded budget fails the assertion instead of hanging the suite.
  while (drained < 50 && claimDatasetCacheRecheck(first)) {
    drained += 1;
  }
  assert.ok(drained < 50, `budget never exhausted after ${drained} claims`);
  assert.equal(claimDatasetCacheRecheck(first), false);
  assert.equal(
    claimDatasetCacheRecheck(second),
    true,
    "a new selection must not inherit the exhausted budget",
  );
});
