// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

// Exercise the real release effect with controlled runtime/store transitions.
const source = readSrc("components/assistant-ui/thread.tsx");
const marker = source.indexOf("// Fire the parked send once");
assert.ok(marker >= 0);
const start = source.indexOf("useEffect(() => {", marker) + "useEffect(() => {".length;
const end = source.indexOf("\n  }, [", start);
assert.ok(end > start);
const body = ts.transpileModule(source.slice(start, end), {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
}).outputText;

function fixture() {
  const calls: string[] = [];
  const runtime = { isRunning: false, queueActive: false, preStream: false };
  const draft = { text: "", attachments: [{ id: "image-1", type: "image" }] };
  const pending = { current: true };
  const env = {
    pendingSend: true,
    pendingSendRef: pending,
    pendingSendForceQueueRef: { current: false },
    indexingActive: false,
    threadScopedSettingsPending: false,
    threadIsRunning: false,
    promptQueueThreadIds: ["t1"],
    preStreamThreadIds: ["t1"],
    hasAttachments: true,
    attachmentsAreAllPastedText: false,
    hasPendingAudio: false,
    hasMaterializingImageAttachments: false,
    hasMaterializingAudioAttachments: false,
    hasMaterializingVideoAttachments: false,
    isResearchActive: false,
    disableQueue: false,
    overlay: false,
    canQueueCurrentPrompt: false,
    canQueuePastedTextPrompt: false,
    aui: {
      thread: () => ({ getState: () => runtime }),
      composer: () => ({ getState: () => draft }),
    },
    usePromptQueueUI: { getState: () => runtime },
    findPromptQueueEntry: () => runtime.queueActive,
    hasPreStreamRunReservation: () => runtime.preStream,
    setPendingSend: (value: boolean) => { env.pendingSend = value; },
    dismissWaitToast: () => calls.push("dismiss"),
    clearStoredDraft: () => calls.push("clear"),
    sendReservedComposer: () => calls.push("send"),
    queueComposerText: () => calls.push("queue-text"),
    queuePastedTextPrompt: () => { calls.push("queue-paste"); return true; },
    toast: { error: () => calls.push("error") },
  };
  const release = new Function(...Object.keys(env), body);
  return { env, runtime, draft, pending, calls, run: () => release(...Object.values(env)) };
}

for (const blocker of ["isRunning", "queueActive", "preStream"] as const) {
  test(`an attachment stays parked during ${blocker} and sends once afterwards`, () => {
    const f = fixture();
    const attachments = f.draft.attachments;
    f.runtime[blocker] = true;
    f.run();
    assert.deepEqual(f.calls, []);
    assert.equal(f.pending.current, true);
    assert.equal(f.draft.attachments, attachments);
    f.runtime[blocker] = false;
    f.run();
    f.run();
    assert.deepEqual(f.calls, ["dismiss", "clear", "send"]);
    assert.equal(f.draft.attachments, attachments);
  });
}

test("video materialization and same-commit cancellation keep the draft unsent", () => {
  const f = fixture();
  f.env.hasMaterializingVideoAttachments = true;
  f.run();
  assert.deepEqual(f.calls, []);
  f.pending.current = false;
  f.env.hasMaterializingVideoAttachments = false;
  f.run();
  assert.deepEqual(f.calls, []);
});

test("a text submit released after hydration still joins the running prompt queue", () => {
  const f = fixture();
  f.env.hasAttachments = false;
  f.env.canQueueCurrentPrompt = true;
  f.draft.text = "follow-up";
  f.draft.attachments = [];
  f.runtime.isRunning = true;
  f.run();
  assert.deepEqual(f.calls, ["dismiss", "queue-text"]);
});
