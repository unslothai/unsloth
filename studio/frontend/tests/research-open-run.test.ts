// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import type { ResearchRun } from "../src/features/chat/types/research.ts";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
register("./helpers/vite-env-loader.mjs", import.meta.url);
registerBundlerResolver();
installLocalStorageFake();
const {
  ingestResearchUpdate,
  openResearchRun,
  resetResearchRunState,
  useResearchRunStore,
} = await import("../src/features/chat/stores/research-run-store.ts");

const run: ResearchRun = {
  id: "run",
  threadId: "thread",
  userMessageId: "user",
  status: "awaiting_approval",
  plan: { title: "Plan", steps: [{ title: "Search", query: "topic" }] },
  planRevision: 1,
  planHash: "hash",
  steps: [],
  sources: [],
  createdAt: 1,
  updatedAt: 1,
  lastEventSeq: 0,
};

test("metadata-only awaiting-approval run is hydrated before its panel opens, without fetching", () => {
  resetResearchRunState();
  const originalFetch = globalThis.fetch;
  let requests = 0;
  globalThis.fetch = async () => {
    requests++;
    throw new Error("Unexpected request");
  };
  const observed: Array<string | undefined> = [];
  const unsubscribe = useResearchRunStore.subscribe((state) => {
    if (state.openRunId)
      observed.push(state.sessions[state.openRunId]?.run.threadId);
  });
  try {
    openResearchRun(run);
    assert.deepEqual(observed, ["thread"]);
    assert.equal(
      useResearchRunStore.getState().planReviewByRunId.run.open,
      true,
    );
    assert.equal(useResearchRunStore.getState().sessions.run.following, false);
    assert.equal(requests, 0);
  } finally {
    unsubscribe();
    globalThis.fetch = originalFetch;
    resetResearchRunState();
  }
});

test("reopening a dismissed plan preserves edits, session identity, error and follower state", () => {
  resetResearchRunState();
  ingestResearchUpdate(run);
  const state = useResearchRunStore.getState();
  state.setPlanReviewOpen(run.id, false);
  state.setPlanReviewEditing(run.id, true);
  state.setPlanReviewDraft(run.id, { title: "Unsaved", steps: [] });
  state.setConnectionError(run.id, "Authentication failed");
  const before = useResearchRunStore.getState();
  openResearchRun(run);
  const after = useResearchRunStore.getState();
  assert.equal(after.sessions, before.sessions);
  assert.equal(after.latestRunByThreadId, before.latestRunByThreadId);
  assert.equal(after.claimedThreadIds, before.claimedThreadIds);
  assert.deepEqual(after.planReviewByRunId.run, {
    ...before.planReviewByRunId.run,
    open: true,
  });
  assert.equal(after.openRunId, run.id);
});

for (const status of [
  "planning",
  "queued",
  "running",
  "paused",
  "cancelling",
  "cancelled",
  "completed",
  "failed",
] as const) {
  test(`opening ${status} activity preserves existing session and review state`, () => {
    resetResearchRunState();
    ingestResearchUpdate(run);
    useResearchRunStore.getState().setPlanReviewOpen(run.id, false);
    ingestResearchUpdate({ ...run, status, updatedAt: 2 });
    const before = useResearchRunStore.getState();
    // Stale message metadata must not replace the current session or reopen its old review.
    openResearchRun(run);
    const after = useResearchRunStore.getState();
    assert.equal(after.sessions, before.sessions);
    assert.equal(after.planReviewByRunId, before.planReviewByRunId);
    assert.equal(after.openRunId, run.id);
  });
}
