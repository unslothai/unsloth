// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { sandboxSessionIdFor } from "../src/components/assistant-ui/sandbox-files.ts";
import {
  bindCompareContextSnapshot,
  compareContextSnapshotForPair,
  releaseCompareContextSnapshot,
} from "../src/features/chat/utils/compare-context-snapshot.ts";

test("compare panes keep distinct threads while sharing the project workspace session", () => {
  const projectId = "workspace-project";
  const panes = [
    { threadId: "compare-left", model: "local-mlx" },
    { threadId: "compare-right", model: "external-gemini" },
  ];

  const sessions = panes.map((pane) => ({
    threadId: pane.threadId,
    model: pane.model,
    sessionId: sandboxSessionIdFor(pane.threadId, projectId),
  }));

  assert.notEqual(sessions[0]?.threadId, sessions[1]?.threadId);
  assert.notEqual(sessions[0]?.model, sessions[1]?.model);
  assert.deepEqual(
    sessions.map((pane) => pane.sessionId),
    ["project-workspace-project", "project-workspace-project"],
  );
});

test("switching models cannot change project workspace identity", () => {
  const before = sandboxSessionIdFor("thread-a", "workspace-project");
  const after = sandboxSessionIdFor("thread-a", "workspace-project");

  assert.equal(before, "project-workspace-project");
  assert.equal(after, before);
  assert.equal(sandboxSessionIdFor("thread-a", null), "thread-a");
});

test("both compare panes read one pair-scoped context snapshot", () => {
  const pairId = "compare-pair";
  const snapshotId = "opaque-server-context-snapshot";
  bindCompareContextSnapshot(pairId, snapshotId);

  const left = compareContextSnapshotForPair(pairId);
  const right = compareContextSnapshotForPair(pairId);

  assert.equal(left, snapshotId);
  assert.equal(right, snapshotId);
  assert.equal(compareContextSnapshotForPair("other-pair"), undefined);

  releaseCompareContextSnapshot(pairId, "stale-snapshot");
  assert.equal(compareContextSnapshotForPair(pairId), snapshotId);
  releaseCompareContextSnapshot(pairId, snapshotId);
  assert.equal(compareContextSnapshotForPair(pairId), undefined);
});
