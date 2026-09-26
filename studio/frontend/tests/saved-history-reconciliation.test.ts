// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  beginSavedHistoryReconciliation,
  isSavedHistoryReconciliationSuperseded,
  reconcileOrdinarySavedMessagesInExport,
  reconcileOrdinarySavedMessagesInView,
  shouldReconcileOrdinaryStoredAssistant,
} = await import(
  "../src/features/chat/utils/saved-history-reconciliation.ts"
);
const { subscribeGenerationRecoveryTriggers } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

const assistantRuntime = (id: string, text: string) =>
  ({
    id,
    role: "assistant" as const,
    createdAt: new Date(0),
    content: [{ type: "text" as const, text }],
    status: { type: "complete" as const, reason: "unknown" as const },
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
  });

const storedAssistant = (id: string, text: string) => ({
  id,
  threadId: "thread-1",
  parentId: "user-1",
  role: "assistant" as const,
  content: [{ type: "text" as const, text }],
  createdAt: 0,
});

test("ordinary saved assistant text reconciles into the open export", () => {
  const exported = {
    headId: "assistant-1",
    messages: [
      {
        parentId: "user-1",
        message: assistantRuntime("assistant-1", "Old checkpoint"),
      },
    ],
  };
  const { messages, changed } = reconcileOrdinarySavedMessagesInExport(
    exported,
    [storedAssistant("assistant-1", "New saved checkpoint")],
    { editingMessageId: null },
  );
  assert.equal(changed, true);
  const text = (messages[0].message.content as { text: string }[])[0].text;
  assert.equal(text, "New saved checkpoint");
});

test("a running assistant message is not reconciled", () => {
  const runtime = assistantRuntime("assistant-1", "Old checkpoint");
  runtime.status = { type: "running" };
  assert.equal(
    shouldReconcileOrdinaryStoredAssistant({
      stored: storedAssistant("assistant-1", "New saved checkpoint"),
      runtime,
      editingMessageId: null,
    }),
    false,
  );
});

test("the message being edited is not reconciled", () => {
  assert.equal(
    shouldReconcileOrdinaryStoredAssistant({
      stored: storedAssistant("assistant-1", "New saved checkpoint"),
      runtime: assistantRuntime("assistant-1", "Old checkpoint"),
      editingMessageId: "assistant-1",
    }),
    false,
  );
});

test("a superseded reconciliation generation is dropped", () => {
  const threadId = "thread-1";
  const first = beginSavedHistoryReconciliation(threadId);
  const second = beginSavedHistoryReconciliation(threadId);
  assert.equal(isSavedHistoryReconciliationSuperseded(threadId, first), true);
  assert.equal(isSavedHistoryReconciliationSuperseded(threadId, second), false);
});

test("reconcileOrdinarySavedMessagesInView imports without writing storage", () => {
  let imported = false;
  const view = {
    threadListItem: () => ({
      getState: () => ({ remoteId: "thread-1" }),
    }),
    thread: () => ({
      export: () => ({
        headId: "assistant-1",
        messages: [
          {
            parentId: "user-1",
            message: assistantRuntime("assistant-1", "Old checkpoint"),
          },
        ],
      }),
      import: () => {
        imported = true;
      },
    }),
  };
  const changed = reconcileOrdinarySavedMessagesInView(
    view,
    "thread-1",
    [storedAssistant("assistant-1", "New saved checkpoint")],
    { editingMessageId: null },
  );
  assert.equal(changed, true);
  assert.equal(imported, true);
});

test("window focus triggers recovery when the document is visible", () => {
  const windowTarget = new EventTarget();
  const documentTarget = Object.assign(new EventTarget(), {
    visibilityState: "visible",
  });
  let recoveries = 0;
  const unsubscribe = subscribeGenerationRecoveryTriggers(
    windowTarget,
    documentTarget,
    () => {
      recoveries += 1;
    },
  );
  windowTarget.dispatchEvent(new Event("focus"));
  unsubscribe();
  assert.equal(recoveries, 1);
});
