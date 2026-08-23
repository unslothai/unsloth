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

function persistedJob(
  repoId: string,
  transport: unknown,
  cancelTransport?: unknown,
) {
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
    ...(cancelTransport === undefined ? {} : { cancelTransport }),
  };
}

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        http: persistedJob("org/http-model", "http"),
        invalid: persistedJob("org/auto-model", "auto"),
        fallback: persistedJob("org/fallback-model", "http", "xet"),
        badMarker: persistedJob("org/bad-marker-model", "http", "auto"),
      },
      conflicts: {},
    },
    version: 1,
  }),
);

const { getState, hasActiveDownloadJob, jobKeyOf, putJob } = await import(
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
    etaSeconds: 0,
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

test("a fallback run's cancel marker survives the reload too", () => {
  // Without it the restored job reads as plain HTTP and offers Pause for a
  // stop that leaves a restart-only partial.
  const jobs = getState().jobs;
  const job = jobs[jobKeyOf("model", "org/fallback-model", null)];
  assert.equal(job?.transport, "http");
  assert.equal(job?.cancelTransport, "xet");
});

test("an unresolved persisted marker is dropped, not trusted", () => {
  const jobs = getState().jobs;
  assert.equal(
    jobs[jobKeyOf("model", "org/bad-marker-model", null)]?.cancelTransport,
    undefined,
  );
});

test("the cancel marker is written with the persisted job", () => {
  const key = jobKeyOf("model", "org/retry-model", null);
  putJob({
    key,
    kind: "model",
    repoId: "org/retry-model",
    variant: null,
    state: "running",
    downloadedBytes: 10,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 100,
    fraction: 0.1,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 3,
    transport: "http",
    cancelTransport: "xet",
  });
  assert.ok(flushPersistedState);
  flushPersistedState();

  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[key].transport, "http");
  assert.equal(persisted.state.jobs[key].cancelTransport, "xet");
});

test("a running job is the activity the desktop quit path asks about", () => {
  // What set_renderer_activity mirrors into Rust, so the close button warns first.
  assert.equal(hasActiveDownloadJob(getState().jobs), true);
});

test("an external job counts too, since a quit kills its transfer as well", () => {
  assert.equal(hasActiveDownloadJob({}), false);
  assert.equal(
    hasActiveDownloadJob({
      external: {
        key: "model:org/external",
        kind: "model",
        repoId: "org/external",
        variant: null,
        state: "running",
        downloadedBytes: 0,
        completedBytes: 0,
        completeOnDisk: false,
        expectedBytes: 0,
        fraction: 0,
        bytesPerSec: 0,
        etaSeconds: 0,
        error: null,
        startedAt: 4,
        external: true,
      },
    }),
    true,
  );
  // A settled job is not activity, whoever owns it.
  assert.equal(
    hasActiveDownloadJob({
      done: {
        key: "model:org/done",
        kind: "model",
        repoId: "org/done",
        variant: null,
        state: "complete",
        downloadedBytes: 0,
        completedBytes: 0,
        completeOnDisk: true,
        expectedBytes: 0,
        fraction: 1,
        bytesPerSec: 0,
        etaSeconds: 0,
        error: null,
        startedAt: 4,
      },
    }),
    false,
  );
});
