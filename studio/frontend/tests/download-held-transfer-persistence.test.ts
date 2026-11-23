// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ManagedDownload } from "../src/features/hub/download-manager/download-manager-types.ts";
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

function persistedJob(repoId: string, extra: Record<string, unknown> = {}) {
  return {
    key: `model:${repoId}`,
    kind: "model",
    repoId,
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes: 3_000_000_000,
    completedBytes: 0,
    expectedBytes: 500_000_000,
    fraction: 0.1,
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
        held: persistedJob("org/held-model", { measuredTransfer: false }),
        measured: persistedJob("org/measured-model", {
          measuredTransfer: true,
        }),
        unpolled: persistedJob("org/unpolled-model"),
      },
      conflicts: {},
    },
    version: 2,
  }),
);

const { getState, jobKeyOf, putJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

test("a held reading is still held after the reload that carried it", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/held-model", "Q4_K_M")]?.measuredTransfer,
    false,
  );
});

test("a measured reading restores as measured", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/measured-model", "Q4_K_M")]?.measuredTransfer,
    true,
  );
});

test("a current record with no marker still means never polled", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/unpolled-model", "Q4_K_M")]?.measuredTransfer,
    undefined,
  );
});

test("the held marker is written out with the job", () => {
  const key = jobKeyOf("model", "org/writeback-model", "Q4_K_M");
  const job: ManagedDownload = {
    key,
    kind: "model",
    repoId: "org/writeback-model",
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes: 3_000_000_000,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 500_000_000,
    fraction: 0.1,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 2,
    measuredTransfer: false,
  };

  putJob(job);
  assert.ok(flushPersistedState);
  flushPersistedState();

  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[key].measuredTransfer, false);
});
