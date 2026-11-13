// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  BATCH_OPEN_TIMEOUT_MS,
  BATCH_POLL_MS,
  BATCH_SEND_TIMEOUT_MS,
  batchStepOutcome,
  batchTargetKey,
  batchThreadReader,
  claimBatchStep,
  orderBatchFiles,
  startBatchNewChats,
  stopBatchNewChats,
  useBatchNewChatsStore,
} = await import("../src/features/chat/batch/batch-new-chats.ts");
type BatchFinish = import("../src/features/chat/batch/batch-new-chats.ts").BatchFinish;
type BatchThreadSnapshot =
  import("../src/features/chat/batch/batch-new-chats.ts").BatchThreadSnapshot;

const user = { role: "user", status: null };
const reply = (type: string, reason?: string) => ({ role: "assistant", status: { type, reason } });

function file(name: string): File {
  return new File(["x"], name, { type: "video/mp4" });
}

function harness() {
  const opened: string[] = [];
  const progress: number[] = [];
  const finished: BatchFinish[] = [];
  let n = 0;
  return {
    opened,
    progress,
    finished,
    deps: {
      newNonce: () => `n${n++}`,
      openNewChat: (nonce: string) => opened.push(nonce),
      notify: {
        progress: (index: number) => progress.push(index),
        finish: (result: BatchFinish) => finished.push(result),
      },
    },
  };
}

test("a step settles only on a finished reply; errors fail it and a user stop stops it", () => {
  assert.equal(batchStepOutcome(null), "pending");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [] }), "pending");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user] }), "pending");
  assert.equal(batchStepOutcome({ isRunning: true, messages: [user, reply("complete")] }), "pending");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user, reply("running")] }), "pending");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user, reply("complete")] }), "done");
  // A reply cut off by the length limit is still that file's answer.
  assert.equal(
    batchStepOutcome({ isRunning: false, messages: [user, reply("incomplete", "length")] }),
    "done",
  );
  assert.equal(
    batchStepOutcome({ isRunning: false, messages: [user, reply("incomplete", "error")] }),
    "failed",
  );
  assert.equal(
    batchStepOutcome({ isRunning: false, messages: [user, reply("incomplete", "cancelled")] }),
    "stopped",
  );
  const research = (status: string) => ({
    ...reply("complete"),
    metadata: { custom: { researchRun: { status } } },
  });
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user, research("completed")] }), "done");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user, research("failed")] }), "failed");
  assert.equal(batchStepOutcome({ isRunning: false, messages: [user, research("cancelled")] }), "stopped");
});

test("files run in natural name order, so clip-2 comes before clip-10", () => {
  const ordered = orderBatchFiles([file("clip-10.mp4"), file("Clip-2.mp4"), file("clip-1.mp4")]);
  assert.deepEqual(
    ordered.map((f) => f.name),
    ["clip-1.mp4", "Clip-2.mp4", "clip-10.mp4"],
  );
});

test("the reader follows a chat from its local id to its saved id", () => {
  const snapshot: BatchThreadSnapshot = { isRunning: false, messages: [user] };
  let saved = false;
  const threads = {
    getItemById: (id: string) => ({
      getState: () => {
        if (id !== "local" && id !== "remote") throw new Error("gone");
        return { id: "local", remoteId: saved ? "remote" : null };
      },
    }),
    getById: (id: string) => ({
      getState: () => {
        // Once saved, the runtime only answers to the saved id.
        if (id !== (saved ? "remote" : "local")) throw new Error("no thread");
        return snapshot;
      },
    }),
  };
  const read = batchThreadReader(threads, "local");
  assert.equal(read(), snapshot);
  saved = true;
  assert.equal(read(), snapshot);
});

test("each file gets its own new chat, the next opening only after the reply finishes", () => {
  mock.timers.enable({ apis: ["setInterval", "setTimeout", "Date"] });
  try {
    const h = harness();
    assert.equal(startBatchNewChats("  Describe this clip  ", [file("b.mp4"), file("a.mp4")], h.deps), true);
    // A second batch can't start while one runs.
    assert.equal(startBatchNewChats("again", [file("c.mp4")], h.deps), false);
    assert.deepEqual(h.opened, ["n0"]);
    // Only the composer of that new chat can claim the step, and only once.
    assert.equal(claimBatchStep("single:other"), null);
    const first = claimBatchStep(batchTargetKey("n0"));
    assert.ok(first);
    assert.equal(claimBatchStep(batchTargetKey("n0")), null);
    assert.equal(first.file.name, "a.mp4");
    assert.equal(first.prompt, "Describe this clip");
    assert.equal(first.total, 2);

    let state: BatchThreadSnapshot = { isRunning: true, messages: [user, reply("running")] };
    first.sent({ read: () => state, unsent: () => false });
    mock.timers.tick(BATCH_POLL_MS * 4);
    assert.deepEqual(h.opened, ["n0"], "still answering: no second chat yet");

    state = { isRunning: false, messages: [user, reply("complete")] };
    mock.timers.tick(BATCH_POLL_MS);
    assert.deepEqual(h.opened, ["n0", "n1"]);
    assert.deepEqual(h.progress, [0, 1]);

    const second = claimBatchStep(batchTargetKey("n1"));
    assert.ok(second);
    assert.equal(second.file.name, "b.mp4");
    second.sent({
      read: () => ({ isRunning: false, messages: [user, reply("complete")] }),
      unsent: () => false,
    });
    mock.timers.tick(BATCH_POLL_MS);
    assert.deepEqual(h.finished, [{ kind: "done", total: 2 }]);
    assert.equal(useBatchNewChatsStore.getState().run, null);
  } finally {
    mock.timers.reset();
  }
});

test("a reply that errors stops the batch at that file", () => {
  mock.timers.enable({ apis: ["setInterval", "setTimeout", "Date"] });
  try {
    const h = harness();
    startBatchNewChats("p", [file("a.mp4"), file("b.mp4")], h.deps);
    const claim = claimBatchStep(batchTargetKey("n0"));
    assert.ok(claim);
    claim.sent({
      read: () => ({ isRunning: false, messages: [user, reply("incomplete", "error")] }),
      unsent: () => false,
    });
    mock.timers.tick(BATCH_POLL_MS);
    assert.equal(h.finished.length, 1);
    const result = h.finished[0];
    assert.equal(result.kind, "failed");
    assert.equal(result.kind === "failed" && result.fileName, "a.mp4");
    assert.deepEqual(h.opened, ["n0"]);
  } finally {
    mock.timers.reset();
  }
});

test("a send Studio refused, or a chat nobody opened, ends the batch instead of hanging", () => {
  mock.timers.enable({ apis: ["setInterval", "setTimeout", "Date"] });
  try {
    const refused = harness();
    startBatchNewChats("p", [file("a.mp4")], refused.deps);
    const claim = claimBatchStep(batchTargetKey("n0"));
    assert.ok(claim);
    // The prompt is still sitting in the composer and the chat is empty.
    claim.sent({ read: () => ({ isRunning: false, messages: [] }), unsent: () => true });
    mock.timers.tick(BATCH_SEND_TIMEOUT_MS - BATCH_POLL_MS);
    assert.equal(refused.finished.length, 0, "a parked send gets time to go out");
    mock.timers.tick(BATCH_POLL_MS * 2);
    assert.equal(refused.finished[0]?.kind, "failed");

    const unopened = harness();
    startBatchNewChats("p", [file("a.mp4")], unopened.deps);
    mock.timers.tick(BATCH_OPEN_TIMEOUT_MS + 1);
    assert.equal(unopened.finished[0]?.kind, "failed");
    assert.equal(claimBatchStep(batchTargetKey("n0")), null);
  } finally {
    mock.timers.reset();
  }
});

test("Stop ends the batch before the next chat, counting chats already sending", () => {
  mock.timers.enable({ apis: ["setInterval", "setTimeout", "Date"] });
  try {
    const h = harness();
    startBatchNewChats("p", [file("a.mp4"), file("b.mp4"), file("c.mp4")], h.deps);
    const claim = claimBatchStep(batchTargetKey("n0"));
    assert.ok(claim);
    claim.sent({
      read: () => ({ isRunning: true, messages: [user, reply("running")] }),
      unsent: () => false,
    });
    stopBatchNewChats();
    assert.deepEqual(h.finished, [{ kind: "stopped", sent: 1, total: 3 }]);
    mock.timers.tick(BATCH_POLL_MS * 10);
    assert.deepEqual(h.opened, ["n0"]);
    // A late failure from the stopped step is ignored.
    claim.failed("late");
    assert.equal(h.finished.length, 1);

    // Stopped while the next chat is still opening: that file was never sent.
    const again = harness();
    startBatchNewChats("p", [file("a.mp4"), file("b.mp4")], again.deps);
    stopBatchNewChats();
    assert.deepEqual(again.finished, [{ kind: "stopped", sent: 0, total: 2 }]);
    assert.equal(claimBatchStep(batchTargetKey("n0")), null);
  } finally {
    mock.timers.reset();
  }
});

test("the main composer claims batch steps and sends them through its own form", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /useBatchNewChatsComposer\(aui, nativeAttachmentTargetKey, formRef\)/);
  assert.match(thread, /batchNewChats: \(\s*<DropdownMenuItem/);
  const hook = readSrc("features/chat/batch/use-batch-new-chats.ts");
  assert.match(hook, /form\.requestSubmit\(\)/);
  // Compare chat sends to several models at once, so it offers no batch item.
  assert.match(readSrc("features/chat/shared-composer.tsx"), /batchNewChats: null,/);
});
