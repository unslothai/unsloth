// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The durable run relays a pause as an `_admissionStatus` chunk so a reload or another tab
 * can show it. Only the live adapter read that chunk; the follower in runtime-provider
 * dropped it and showed a bare spinner for the whole pause, the wedged look the relay exists
 * to prevent. The follower is pinned to the same label the live adapter uses, and to
 * clearing it with the run.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/chat/runtime-provider.tsx", import.meta.url)),
  "utf8",
);

// The chunk branch, from the status read to the `continue` that closes it: located
// structurally, since a fixed-length slice stopped short as the branch grew.
function chunkBranch(): string {
  const start = source.indexOf("chunk._admissionStatus !== undefined");
  assert.notEqual(start, -1, "the chunk branch reads _admissionStatus");
  const call = source.indexOf("setToolStatus(", start);
  assert.notEqual(call, -1, "the chunk branch sets the tool status");
  const end = source.indexOf("continue;", call);
  assert.notEqual(end, -1, "the chunk branch ends in a continue");
  return source.slice(start, end + "continue;".length);
}

test("the follower renders a relayed pause through the live adapter's label", () => {
  const branch = chunkBranch();
  assert.match(branch, /setToolStatus\(\s*threadId,\s*admissionStatusLabel\(chunk\._admissionStatus\),\s*serverCancel,?\s*\)/);
  assert.match(branch, /continue;/);
  assert.match(source, /import \{\s*type AdmissionStatus,\s*admissionStatusLabel,\s*\} from "\.\/utils\/admission-status"/);
});

test("a recompute marks the thread instead of rewriting the status line", () => {
  // It arrives after the resume it qualifies, so the run is generating again: showing a
  // pause label for it would say the opposite of what happened.
  const branch = chunkBranch();
  assert.match(branch, /chunk\._admissionStatus === "recomputed"/);
  assert.match(branch, /notePreemptRecompute\(threadId\)/);
  assert.ok(
    branch.indexOf("notePreemptRecompute") < branch.indexOf("setToolStatus"),
    "the recompute is taken before the label branch it must not reach",
  );
});

test("the label goes with the run", () => {
  const cleanup = source.indexOf('store.setThreadRunning(threadId, false, { owner: serverCancel });');
  assert.notEqual(cleanup, -1);
  const before = source.slice(cleanup - 200, cleanup);
  assert.match(before, /store\.setToolStatus\(threadId, null, serverCancel\);/);
});
