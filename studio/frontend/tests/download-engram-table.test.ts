// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A GGUF download that fetches a standalone Engram table keeps that choice when it is retried
// and across a page reload, so recovery never fetches the quant's own table instead.

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
const REPO = "unsloth/Qwen3.8-Flash-Next-GGUF";
let flushPersistedState: (() => void) | undefined;
Object.assign(globalThis.window, {
  addEventListener: (type: string, listener: () => void) => {
    if (type === "pagehide") flushPersistedState = listener;
  },
});

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        swapped: {
          key: `model:${REPO}`,
          kind: "model",
          repoId: REPO,
          variant: "UD-IQ1_S",
          state: "running",
          downloadedBytes: 1,
          completedBytes: 0,
          expectedBytes: 100,
          fraction: 0.01,
          error: null,
          startedAt: 1,
          engramPrecision: "Q8_0",
        },
      },
      conflicts: {},
    },
    version: 2,
  }),
);

const { getState, jobKeyOf, putJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);
const { retryRequestFor } = await import(
  "../src/features/hub/download-manager/retry-download.ts"
);

function job(over: Partial<ManagedDownload> = {}): ManagedDownload {
  return {
    key: jobKeyOf("model", REPO, "UD-IQ1_S"),
    kind: "model",
    repoId: REPO,
    variant: "UD-IQ1_S",
    state: "error",
    downloadedBytes: 1,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 100,
    fraction: 0.01,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: "failed",
    startedAt: 1,
    ...over,
  };
}

test("a retry fetches the same table the failed job was fetching", () => {
  assert.equal(retryRequestFor(job({ engramPrecision: "Q8_0" })).engramPrecision, "Q8_0");
  assert.equal(retryRequestFor(job()).engramPrecision, undefined);
});

test("a reload restores the job with its table", () => {
  assert.equal(
    getState().jobs[jobKeyOf("model", REPO, "UD-IQ1_S")]?.engramPrecision,
    "Q8_0",
  );
});

test("the table is written out with the job", () => {
  const written = job({
    key: jobKeyOf("model", REPO, "UD-Q4_K_XL"),
    variant: "UD-Q4_K_XL",
    state: "running",
    engramPrecision: "BF16",
  });
  putJob(written);
  assert.ok(flushPersistedState);
  flushPersistedState();
  const persisted = JSON.parse(store.get(PERSIST_KEY) ?? "null");
  assert.equal(persisted.state.jobs[written.key].engramPrecision, "BF16");
});
