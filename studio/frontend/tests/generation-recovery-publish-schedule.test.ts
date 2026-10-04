// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Full scheduler catch-up and settlement are tested in generation-tool-recovery.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  createRecoveryPublishSchedule,
  generationIsSettled,
  registerRecoveredRunStop,
  stopRecoveredRun,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

const INTERVAL_MS = 8_000;

test("a live follow renders every event but saves at checkpoint pace", () => {
  // The live trace: attached with 1,595 events of history, then followed at ~35 events/s for 55 s.
  let clock = 0;
  const schedule = createRecoveryPublishSchedule(INTERVAL_MS, () => clock);
  schedule.attach(1595);
  const events: Array<{
    seq: number;
    status: "running" | "completed";
    lastEventSeq: number;
    at: number;
  }> = [];
  for (let seq = 1; seq <= 1595; seq += 1) {
    events.push({ seq, status: "running", lastEventSeq: 1595, at: 0 });
  }
  for (let seq = 1596; seq <= 3507; seq += 1) {
    const at = Math.round(((seq - 1595) / 1912) * 55_000);
    events.push({ seq, status: "running", lastEventSeq: seq, at });
  }
  events.push({ seq: 3508, status: "completed", lastEventSeq: 3508, at: 55_100 });

  let published = 0;
  const saved: string[] = [];
  for (const event of events) {
    clock = event.at;
    const settled = generationIsSettled(event.status, event.seq, event.lastEventSeq);
    if (!schedule.shouldPublish(event.seq, settled)) continue;
    published += 1;
    if (schedule.takeSave(settled)) saved.push(event.status);
  }

  assert.equal(published, 3508 - 1595 + 1, "history folds, live events all render");
  // One at the live edge, one per elapsed interval, and the settled end.
  assert.ok(saved.length <= Math.ceil(55_100 / INTERVAL_MS) + 2, `saved ${saved.length} times`);
  assert.equal(saved.at(-1), "completed", "the settled end is always saved");
});

test("Stop reaches the run a recovery is replaying, and only while it is", () => {
  const stopped: string[] = [];
  const unregister = registerRecoveredRunStop("thread-a", () => stopped.push("run-1"));
  assert.equal(stopRecoveredRun("thread-b"), false, "another thread's Stop must not cancel it");
  assert.equal(stopRecoveredRun(undefined), false);
  assert.equal(stopRecoveredRun("thread-a"), true);
  assert.deepEqual(stopped, ["run-1"]);

  // Old cleanup must preserve the newer registration.
  const unregisterNewer = registerRecoveredRunStop("thread-a", () => stopped.push("run-2"));
  unregister();
  assert.equal(stopRecoveredRun("thread-a"), true);
  assert.deepEqual(stopped, ["run-1", "run-2"]);
  unregisterNewer();
  assert.equal(stopRecoveredRun("thread-a"), false, "a finished recovery leaves nothing to stop");
});

test("the composer Stop and the recovery are wired to that handle", () => {
  // Inspect wiring directly because this harness cannot mount the composer.
  const provider = readText("../src/features/chat/runtime-provider.tsx");
  const recovery = provider.slice(
    provider.indexOf("function scheduleGenerationRecovery("),
    provider.indexOf("const temporaryThreadCreation"),
  );
  assert.match(recovery, /registerRecoveredRunStop\(threadId, serverCancel\)/);
  assert.match(recovery, /finally \{[\s\S]*?unregisterStop\(\);/);

  const thread = readText("../src/components/assistant-ui/thread.tsx");
  const stop = thread.slice(thread.indexOf("  const stop = () => {"));
  assert.match(stop.slice(0, stop.indexOf("\n  };")), /stopRecoveredRun\(threadRemoteId\)/);
});
