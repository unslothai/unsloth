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
  const key = datasetCacheRecheckKey({
    dataset: "Org/Data",
    subset: "default",
    split: "train",
    streaming: false,
  });
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

function selection(overrides: Record<string, unknown> = {}) {
  return {
    dataset: "Org/Data",
    subset: "default",
    split: "train",
    streaming: false,
    ...overrides,
  } as Parameters<typeof datasetCacheRecheckKey>[0];
}

/** Drain the budget for `key`, returning how many claims it granted. */
function drain(key: string): number {
  let drained = 0;
  // Bounded so an unbounded budget fails the assertion instead of hanging the suite.
  while (drained < 50 && claimDatasetCacheRecheck(key)) {
    drained += 1;
  }
  assert.ok(drained < 50, `budget never exhausted after ${drained} claims`);
  assert.equal(claimDatasetCacheRecheck(key), false);
  return drained;
}

test("switching dataset selection starts a fresh re-check budget", () => {
  resetDatasetCacheRecheckBudget();
  drain(datasetCacheRecheckKey(selection()));
  assert.equal(
    claimDatasetCacheRecheck(
      datasetCacheRecheckKey(selection({ split: "validation" })),
    ),
    true,
    "a new selection must not inherit the exhausted budget",
  );
});

// Regression guard: the budget key originally used only dataset + split, so changing
// either of the other two user-chosen dimensions silently inherited an exhausted budget
// and dropped the local-cache preference for a genuinely different selection.
for (const [label, override] of [
  ["subset", { subset: "fr" }],
  ["streaming mode", { streaming: true }],
] as const) {
  test(`changing ${label} starts a fresh re-check budget`, () => {
    resetDatasetCacheRecheckBudget();
    drain(datasetCacheRecheckKey(selection()));
    assert.equal(
      claimDatasetCacheRecheck(datasetCacheRecheckKey(selection(override))),
      true,
      `changing ${label} must not inherit the exhausted budget`,
    );
  });
}

test("a moving cache path does NOT refresh the budget", () => {
  // The inverse guard. cachePath is derived state that advances while a dataset
  // downloads; if it fed the key, every poll would mint a fresh budget and re-arm the
  // non-terminating loop of #7853. Keying on the selection alone keeps the bound.
  resetDatasetCacheRecheckBudget();
  const key = datasetCacheRecheckKey(selection());
  drain(key);
  assert.equal(
    datasetCacheRecheckKey(selection()),
    key,
    "the key must not vary with anything outside the selection",
  );
  assert.equal(
    claimDatasetCacheRecheck(key),
    false,
    "the budget must stay exhausted regardless of cache-path churn",
  );
});
