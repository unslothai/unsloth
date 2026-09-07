// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A durable run gets TWO readers in the tab that started it: the adapter streaming it, and
// `scheduleGenerationRecovery` replaying the same run from storage. Its only gate is
// `generationNeedsRecovery(metadata)`, and `history.load()` force-writes
// `generationSettled: false` onto any message matching an active run, so nothing ever asks
// whether this tab is already the producer.
//
// The visible cost is the reasoning pane flicker: the follower is always behind, so it imports
// a lagging prefix over the live reply. The unseen cost is larger. The follower publishes on
// EVERY chunk event and each publish re-parses the whole reply and awaits a PUT of the entire
// message, so a reply of N chunks costs N round trips whose payload grows with the reply.
//
// These pin the ownership registry that stops it, and the leak that would be its own failure
// mode: a run left claimed is a run this tab can never recover.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  claimLiveGenerationRun,
  isLiveGenerationRun,
  releaseLiveGenerationRun,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

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
  // The adapter releases in a finally that also runs on paths where no durable run was ever
  // created, so this has to be a no-op rather than a throw.
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
  // Source-pinned: scheduleGenerationRecovery is not reachable from a test (it needs a live
  // aui view and the store), so the guard is asserted where it sits. It must come BEFORE the
  // scheduler registers itself as running, or the early return is unreachable.
  const provider = read("../src/features/chat/runtime-provider.tsx");
  const guard = provider.indexOf("if (isLiveGenerationRun(runId)) return;");
  const register = provider.indexOf("runtime.registerThreadServerCancel(threadId, serverCancel)");

  assert.ok(guard > 0, "the ownership guard is gone; the follower will race the adapter again");
  assert.ok(register > 0);
  assert.ok(guard < register, "the guard must precede the scheduler taking ownership");
});

test("the adapter claims the run BEFORE admission, not after the response", () => {
  // The create POST is awaited, and the run is visible through /active as soon as it lands.
  // A claim that waits for the response leaves that whole round trip open: a visibility,
  // pageshow, online or history-load trigger can start a recovery inside it, and the later
  // claim does not stop one already running, because the scheduler only tests ownership at
  // startup. The run id is the client's own (`cancelId` is passed as `runId`), so there is no
  // reason to wait for the server to hand it back.
  const adapter = read("../src/features/chat/api/chat-adapter.ts");
  // Matched without the closing paren: the call carries an options object and wraps.
  const claim = adapter.indexOf("claimLiveGenerationRun(cancelId, resolvedThreadId!,");
  const admission = adapter.indexOf("generationRun = await createChatGenerationRunUntilAbort(");

  assert.ok(claim > 0, "nothing claims the run before admission");
  assert.ok(admission > 0);
  assert.ok(
    claim < admission,
    "the claim must precede the create POST, or the round trip is an open window",
  );
  // Provisional, so it owns recovery without marking the thread bounded: the await below
  // can outlast the checkpoint cap, and a capped thread leaves the fallback stream with no
  // persistence at all.
  assert.ok(
    adapter.slice(claim, admission).includes("provisional: true"),
    "the pre-admission claim must not make the thread bounded",
  );
  // And the pre-admission id has to be released too, or a failed admission strands it.
  assert.ok(
    adapter.includes("releaseLiveGenerationRun(cancelId)"),
    "the pre-admission claim is never released; a failed admission would strand the run",
  );
});

test("the adapter claims the run and releases it in a finally", () => {
  const adapter = read("../src/features/chat/api/chat-adapter.ts");

  assert.ok(
    adapter.includes("claimLiveGenerationRun(generationRunId, resolvedThreadId!)"),
    "nothing claims the run, so the guard above can never fire",
  );
  // The release has to be inside a finally. Without that, an aborted or failed stream leaves
  // the run claimed and this tab stops being able to recover it at all, which is worse than
  // the defect being fixed.
  const release = adapter.indexOf("releaseLiveGenerationRun(generationRunId)");
  assert.ok(release > 0);
  const before = adapter.slice(0, release);
  assert.ok(
    before.lastIndexOf("} finally {") > before.lastIndexOf("} catch ("),
    "the release must sit in a finally, not on the success path",
  );
});
