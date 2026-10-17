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

/** Replays runDatasetCheck's decision sequence in training-config-store.ts. */
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

  tracker.observe(identity, poll());

  for (let i = 0; i < maxIterations; i += 1) {
    requests += 1;
    const token = tracker.beginValidation(identity);
    tracker.observe(identity, poll());

    if (tracker.rejectValidation(token)) {
      return { requests, terminated: true };
    }
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
  // sizeBytes changes on every poll while downloading, so without a bound the store re-fires forever.
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

function drain(key: string): number {
  let drained = 0;
  // Bounded so an unbounded budget fails instead of hanging the suite.
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

// The budget key must include every user-chosen dimension, not just dataset + split.
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
  // cachePath advances during a download; keying on it would mint a fresh budget every poll.
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
