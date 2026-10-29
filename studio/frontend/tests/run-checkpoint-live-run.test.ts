// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

register("./helpers/settings-api-resolver.mjs", import.meta.url);
registerBundlerResolver();

const {
  claimLiveGenerationRun,
  releaseLiveGenerationRun,
  resetServerActiveGenerationRuns,
  syncServerActiveGenerationRuns,
  threadHasDurableGenerationRun,
  threadHasLiveGenerationRun,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

test("only a run this tab streams counts as live, not one the server last named", () => {
  resetServerActiveGenerationRuns();
  syncServerActiveGenerationRuns("thread-1", ["followed-run"]);
  assert.equal(threadHasDurableGenerationRun("thread-1"), true);
  assert.equal(
    threadHasLiveGenerationRun("thread-1"),
    false,
    "a follower that stalls leaves the server's answer behind, and a Continue after it streams through checkpoints alone",
  );

  claimLiveGenerationRun("run-1", "thread-1", { provisional: true });
  assert.equal(threadHasLiveGenerationRun("thread-1"), false, "not admitted yet, so nothing on the server holds it");
  claimLiveGenerationRun("run-1", "thread-1");
  assert.equal(threadHasLiveGenerationRun("thread-1"), true);
  assert.equal(threadHasLiveGenerationRun("thread-2"), false);
  releaseLiveGenerationRun("run-1");
  assert.equal(threadHasLiveGenerationRun("thread-1"), false);
  resetServerActiveGenerationRuns();
});

test("checkpoints are skipped only while this tab streams the thread's durable run", () => {
  const provider = readSrc("features/chat/runtime-provider.tsx");
  assert.match(
    provider,
    /\(threadId\) =>\s*threadHasLiveGenerationRun\(threadId\)\s*\?\s*Promise\.resolve\(\)\s*:\s*queueSaveRef\.current\(threadId\),/,
  );
  assert.match(
    provider,
    /isBounded: \(threadId\) => threadHasDurableGenerationRun\(threadId\),/,
  );
});
