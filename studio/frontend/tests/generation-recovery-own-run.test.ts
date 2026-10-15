// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  claimLiveGenerationRun,
  isLiveGenerationRun,
  releaseLiveGenerationRun,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

test("a claimed run is reported as this tab's, and a released one is not", () => {
  assert.equal(isLiveGenerationRun("run-a"), false);
  claimLiveGenerationRun("run-a");
  assert.equal(isLiveGenerationRun("run-a"), true);
  releaseLiveGenerationRun("run-a");
  assert.equal(
    isLiveGenerationRun("run-a"),
    false,
    "a released run must be recoverable again, or a dead stream strands it forever",
  );
});

test("releasing a run that was never claimed is not an error", () => {
  releaseLiveGenerationRun("never-claimed");
  assert.equal(isLiveGenerationRun("never-claimed"), false);
});

test("runs are tracked independently", () => {
  claimLiveGenerationRun("run-b");
  claimLiveGenerationRun("run-c");
  releaseLiveGenerationRun("run-b");
  assert.equal(isLiveGenerationRun("run-b"), false);
  assert.equal(isLiveGenerationRun("run-c"), true, "releasing one run must not free another");
  releaseLiveGenerationRun("run-c");
});

test("the recovery scheduler refuses a run this tab is streaming", () => {
  // The guard must come BEFORE the scheduler registers itself, or the early return is unreachable.
  const provider = readText("../src/features/chat/runtime-provider.tsx");
  const guard = provider.indexOf("if (isLiveGenerationRun(runId)) return;");
  const register = provider.indexOf("runtime.registerThreadServerCancel(threadId, serverCancel)");

  assert.ok(guard > 0, "the ownership guard is gone; the follower will race the adapter again");
  assert.ok(register > 0);
  assert.ok(guard < register, "the guard must precede the scheduler taking ownership");
});

test("the adapter claims the run BEFORE admission, not after the response", () => {
  // Claim before awaiting the create POST: a recovery could start inside that round trip.
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  // Matched without the closing paren: the call carries an options object and wraps.
  const claim = adapter.indexOf("claimLiveGenerationRun(cancelId, resolvedThreadId!,");
  const admission = adapter.indexOf("generationRun = await createChatGenerationRunUntilAbort(");

  assert.ok(claim > 0, "nothing claims the run before admission");
  assert.ok(admission > 0);
  assert.ok(
    claim < admission,
    "the claim must precede the create POST, or the round trip is an open window",
  );
  assert.ok(
    adapter.slice(claim, admission).includes("provisional: true"),
    "the pre-admission claim must not make the thread bounded",
  );
  assert.ok(
    adapter.includes("releaseLiveGenerationRun(cancelId)"),
    "the pre-admission claim is never released; a failed admission would strand the run",
  );
});

test("the adapter claims the run and releases it in a finally", () => {
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");

  assert.ok(
    adapter.includes("claimLiveGenerationRun(generationRunId, resolvedThreadId!)"),
    "nothing claims the run, so the guard above can never fire",
  );
  // The release must be in a finally, or a failed stream leaves the run claimed forever.
  const release = adapter.indexOf("releaseLiveGenerationRun(generationRunId)");
  assert.ok(release > 0);
  const before = adapter.slice(0, release);
  assert.ok(
    before.lastIndexOf("} finally {") > before.lastIndexOf("} catch ("),
    "the release must sit in a finally, not on the success path",
  );
});
