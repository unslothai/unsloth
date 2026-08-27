// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// claimedThreadIds is what the composer reads to take deep research away. Only a finished run
// spends the chat's research: one still going keeps the toggle lit, and a stopped one is
// re-pointed at the next question rather than refused, so neither may claim the thread.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();
Object.assign(globalThis.window as object, { addEventListener: () => {} });

const { ingestResearchUpdate, resetResearchRunState, useResearchRunStore } =
  await import("../src/features/chat/stores/research-run-store.ts");

type AnyRecord = Record<string, unknown>;

function run(overrides: AnyRecord = {}): AnyRecord {
  return {
    id: "run-1",
    threadId: "thread-1",
    userMessageId: "user-1",
    status: "planning",
    plan: null,
    planRevision: 0,
    planHash: null,
    steps: [],
    sources: [],
    documentSources: [],
    lastEventSeq: 0,
    createdAt: 1,
    updatedAt: 1,
    retryCount: 0,
    ...overrides,
  };
}

// biome-ignore lint/suspicious/noExplicitAny: the store's run shape is exercised structurally
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ingest = ingestResearchUpdate as any;

const claimed = (threadId = "thread-1") =>
  Boolean(useResearchRunStore.getState().claimedThreadIds[threadId]);

test("a run still going keeps the toggle, a finished one takes it", () => {
  resetResearchRunState();
  for (const status of ["planning", "queued", "running", "cancelling"]) {
    ingest(run({ status }));
    assert.equal(claimed(), false, status);
  }
  ingest(run({ status: "completed" }));
  assert.equal(claimed(), true);
});

test("a failed run spends the research too, retry is its way back", () => {
  resetResearchRunState();
  ingest(run({ status: "failed" }));
  assert.equal(claimed(), true);
});

test("stopping the run leaves the thread unclaimed", () => {
  resetResearchRunState();
  ingest(run());
  ingest(run({ status: "cancelled" }));
  assert.equal(claimed(), false);
});

test("an older stopped run does not unclaim a thread a newer finished run spent", () => {
  resetResearchRunState();
  ingest(run({ id: "run-2", createdAt: 5, status: "completed" }));
  ingest(run({ id: "run-1", createdAt: 1, status: "cancelled" }));
  assert.equal(claimed(), true);
});

test("re-pointing the stopped run keeps the toggle until it finishes", () => {
  resetResearchRunState();
  ingest(run({ status: "cancelled" }));
  assert.equal(claimed(), false);
  ingest(run({ status: "planning", userMessageId: "user-2", planRevision: 1 }));
  assert.equal(claimed(), false);
  ingest(run({ status: "completed", userMessageId: "user-2", planRevision: 1 }));
  assert.equal(claimed(), true);
});
