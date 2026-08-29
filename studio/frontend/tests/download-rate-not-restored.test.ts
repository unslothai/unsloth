// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A persisted rate or ETA must never come back: the wall-clock gap while the
// app was closed is not transfer time. The estimator restarts cold instead.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const PERSIST_KEY = "unsloth.studio.downloads";
Object.assign(globalThis.window, {
  addEventListener: () => {},
});

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        "model:org/mid-transfer": {
          key: "model:org/mid-transfer",
          kind: "model",
          repoId: "org/mid-transfer",
          variant: null,
          state: "running",
          downloadedBytes: 4_000_000_000,
          completedBytes: 0,
          expectedBytes: 8_000_000_000,
          fraction: 0.5,
          error: null,
          startedAt: 1,
          // Whatever a previous session happened to leave behind.
          bytesPerSec: 50_000_000,
          etaSeconds: 900,
        },
      },
      conflicts: {},
    },
    version: 1,
  }),
);

const { getState, jobKeyOf } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

test("a persisted rate and ETA are both dropped on hydration", () => {
  const job = getState().jobs[jobKeyOf("model", "org/mid-transfer", null)];
  assert.ok(job, "the job itself should still hydrate");
  assert.equal(job.bytesPerSec, 0);
  assert.equal(job.etaSeconds, 0);
  // The progress itself is still worth restoring; only the timing is not.
  assert.equal(job.downloadedBytes, 4_000_000_000);
  assert.equal(job.fraction, 0.5);
});
