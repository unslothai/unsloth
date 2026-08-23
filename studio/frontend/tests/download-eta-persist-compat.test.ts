// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `etaSeconds` is a new required field on ManagedDownload, so an install that
// upgrades into it has records without one, and one that downgrades away must
// still read what this version wrote. It is injected on the way in and omitted
// on the way out, leaving the persisted shape unchanged in both directions.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const PERSIST_KEY = "unsloth.studio.downloads";
let flushPersistedState: (() => void) | undefined;
Object.assign(globalThis.window, {
  addEventListener: (type: string, listener: () => void) => {
    if (type === "pagehide") flushPersistedState = listener;
  },
});

/** Exactly what a Studio from before this change wrote: no `etaSeconds` key. */
function legacyJob(repoId: string, extra: Record<string, unknown> = {}) {
  return {
    key: `model:${repoId}`,
    kind: "model",
    repoId,
    variant: null,
    state: "running",
    downloadedBytes: 4_000_000_000,
    completedBytes: 0,
    expectedBytes: 8_000_000_000,
    fraction: 0.5,
    error: null,
    startedAt: 1,
    ...extra,
  };
}

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        legacy: legacyJob("org/legacy"),
        // A record whose eta survived some other route, or was hand-edited.
        poisoned: legacyJob("org/poisoned", { etaSeconds: "not a number" }),
        infinite: legacyJob("org/infinite", {
          etaSeconds: Number.POSITIVE_INFINITY,
        }),
        negative: legacyJob("org/negative", { etaSeconds: -1 }),
      },
      conflicts: {},
    },
    version: 1,
  }),
);

const { getState, jobKeyOf, patchJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

const keyOf = (repoId: string) => jobKeyOf("model", repoId, null);

test("a record written before etaSeconds existed still hydrates", () => {
  const job = getState().jobs[keyOf("org/legacy")];
  assert.ok(job, "the legacy job should hydrate");
  assert.equal(job.etaSeconds, 0);
  assert.equal(job.downloadedBytes, 4_000_000_000);
  assert.equal(job.fraction, 0.5);
  assert.equal(job.state, "running");
});

test("a hostile persisted etaSeconds cannot reach the UI", () => {
  for (const repoId of ["org/poisoned", "org/infinite", "org/negative"]) {
    const job = getState().jobs[keyOf(repoId)];
    assert.ok(job, `${repoId} should hydrate`);
    assert.equal(job.etaSeconds, 0, `${repoId} kept ${job.etaSeconds}`);
    assert.ok(Number.isFinite(job.etaSeconds));
  }
});

test("what this version writes back carries no etaSeconds, so an older Studio reads it unchanged", () => {
  patchJob(keyOf("org/legacy"), { etaSeconds: 1234, bytesPerSec: 5_000_000 });
  assert.equal(getState().jobs[keyOf("org/legacy")].etaSeconds, 1234);

  assert.ok(flushPersistedState, "the store should register a pagehide flush");
  flushPersistedState?.();

  const written = JSON.parse(store.get(PERSIST_KEY) as string);
  const job = written.state.jobs[keyOf("org/legacy")];
  assert.ok(job, "the job should be persisted");
  assert.ok(
    !("etaSeconds" in job),
    `persisted record leaked etaSeconds: ${JSON.stringify(job)}`,
  );
  // The rate has never been persisted either; the ETA follows the same rule.
  assert.ok(!("bytesPerSec" in job), "persisted record leaked bytesPerSec");
  // Everything an older Studio does read is still there and unchanged.
  assert.equal(job.downloadedBytes, 4_000_000_000);
  assert.equal(job.expectedBytes, 8_000_000_000);
  assert.equal(job.state, "running");
});
