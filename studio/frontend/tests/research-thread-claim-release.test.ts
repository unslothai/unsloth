// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// claimedThreadIds is what the composer reads to grey out deep research. A stopped run is
// re-pointed at the next question rather than refused, so it must not leave the thread claimed.

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

test("a live run claims its thread", () => {
  resetResearchRunState();
  ingest(run());
  assert.equal(claimed(), true);
  ingest(run({ status: "completed" }));
  assert.equal(claimed(), true);
});

test("stopping the run releases the thread", () => {
  resetResearchRunState();
  ingest(run());
  ingest(run({ status: "cancelled" }));
  assert.equal(claimed(), false);
});

test("an older stopped run does not release a thread a newer run holds", () => {
  resetResearchRunState();
  ingest(run({ id: "run-2", createdAt: 5 }));
  ingest(run({ id: "run-1", createdAt: 1, status: "cancelled" }));
  assert.equal(claimed(), true);
});

test("re-pointing the stopped run claims the thread again", () => {
  resetResearchRunState();
  ingest(run({ status: "cancelled" }));
  assert.equal(claimed(), false);
  ingest(run({ status: "planning", userMessageId: "user-2", planRevision: 1 }));
  assert.equal(claimed(), true);
});
