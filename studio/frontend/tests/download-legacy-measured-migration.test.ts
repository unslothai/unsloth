// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A record written before measuredTransfer existed cannot say whether its byte
// counters were measured, so an absent marker there is not the "never polled"
// it means in a current record. The migration reads a legacy record that already
// carries counters as held, so an upgrade landing mid-reclaim does not restore
// the dead run's bytes as measured.

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

function legacyJob(repoId: string, downloadedBytes: number) {
  return {
    key: `model:${repoId}`,
    kind: "model",
    repoId,
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes,
    completedBytes: 0,
    expectedBytes: 500_000_000,
    fraction: 0.1,
    error: null,
    startedAt: 1,
  };
}

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        carried: legacyJob("org/carried-model", 3_000_000_000),
        fresh: legacyJob("org/fresh-model", 0),
      },
      conflicts: {},
    },
    // The version written before the marker existed.
    version: 1,
  }),
);

const { getState, jobKeyOf } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

test("a legacy record carrying counters is restored as held", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/carried-model", "Q4_K_M")]?.measuredTransfer,
    false,
  );
});

test("a legacy record with nothing counted stays unknown", () => {
  // No held figure to distrust, so undefined is the honest answer and the
  // first poll writes the real one.
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/fresh-model", "Q4_K_M")]?.measuredTransfer,
    undefined,
  );
});
