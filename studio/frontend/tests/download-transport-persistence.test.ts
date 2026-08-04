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

function persistedJob(repoId: string, transport: unknown) {
  return {
    key: `model:${repoId}`,
    kind: "model",
    repoId,
    variant: null,
    state: "running",
    downloadedBytes: 25,
    completedBytes: 0,
    expectedBytes: 100,
    fraction: 0.25,
    error: null,
    startedAt: 1,
    transport,
  };
}

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        http: persistedJob("org/http-model", "http"),
        invalid: persistedJob("org/auto-model", "auto"),
      },
      conflicts: {},
    },
    version: 1,
  }),
);

const { getState, jobKeyOf, putJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

test("reload hydration keeps only a resolved active transport", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/http-model", null)]?.transport,
    "http",
  );
  assert.equal(
    jobs[jobKeyOf("model", "org/auto-model", null)]?.transport,
    undefined,
  );
});

test("the active transport is written with the persisted job", () => {
  const key = jobKeyOf("dataset", "org/dataset", null);
  const job: ManagedDownload = {
    key,
    kind: "dataset",
    repoId: "org/dataset",
    variant: null,
    state: "running",
    downloadedBytes: 50,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 100,
    fraction: 0.5,
    bytesPerSec: 10,
    error: null,
    startedAt: 2,
    transport: "xet",
  };

  putJob(job);
  assert.ok(flushPersistedState);
  flushPersistedState();

  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[key].transport, "xet");
});
