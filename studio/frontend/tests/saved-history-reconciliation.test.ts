// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "@typescript/typescript6";

import type { MessageStatus } from "@assistant-ui/react";

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
    status: { type: "complete", reason: "unknown" } as MessageStatus,
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
  const text = (messages[0].message.content as unknown as { text: string }[])[0].text;
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
    { editingMessageId: null, toThreadMessage: () => assert.fail("nothing to append") },
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

// The wake handler is the only caller; removing it leaves every other test green.
test("recoverCurrentThread reconciles ordinary saved messages", () => {
  const source = fileURLToPath(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  );
  const parsed = ts.createSourceFile(
    "runtime-provider.tsx",
    readFileSync(source, "utf8"),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  let handler: ts.Node | null = null;
  const walk = (node: ts.Node): void => {
    if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.name.text === "recoverCurrentThread"
    ) {
      handler = node;
      return;
    }
    ts.forEachChild(node, walk);
  };
  walk(parsed);
  assert.ok(handler, "recoverCurrentThread not found");
  const calls = new Set<string>();
  let reconcileGuarded = false;
  const collect = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
      calls.add(node.expression.text);
      if (node.expression.text === "reconcileOrdinarySavedMessagesInView") {
        assert.match(node.getText(), /toThreadMessage/, "reconcile must be able to append external turns");
        for (let p = node.parent; p; p = p.parent) {
          if (ts.isIfStatement(p) && p.expression.getText() === "fromBackend") {
            reconcileGuarded = true;
          }
        }
      }
    }
    ts.forEachChild(node, collect);
  };
  collect(handler);
  for (const name of [
    "beginSavedHistoryReconciliation",
    "isSavedHistoryReconciliationSuperseded",
    "reconcileOrdinarySavedMessagesInView",
  ]) {
    assert.ok(calls.has(name), `${name} not called from recoverCurrentThread`);
  }
  assert.ok(reconcileGuarded, "reconciliation must skip legacy fallback reads");
  assert.match(
    readFileSync(source, "utf8"),
    /subscribeThreadSwitchRecovery\([\s\S]*?recoverCurrentThread,\s*\)/,
    "a thread switch must reconcile the view",
  );
});

const userRuntime = (id: string, text: string) =>
  ({
    id,
    role: "user" as const,
    createdAt: new Date(0),
    content: [{ type: "text" as const, text }],
    attachments: [],
    metadata: { custom: {} },
  });

const stored = (id: string, role: "user" | "assistant", parentId: string | null, createdAt: number) => ({
  id,
  threadId: "thread-1",
  parentId,
  role,
  content: [{ type: "text" as const, text: id }],
  createdAt,
});

const toRuntime = (record: { id: string; role: string }) =>
  (record.role === "user" ? userRuntime(record.id, record.id) : assistantRuntime(record.id, record.id)) as never;

const { appendExternalStoredTurns, subscribeThreadSwitchRecovery } = await import(
  "../src/features/chat/utils/saved-history-reconciliation.ts"
);

const openView = () => ({
  headId: "a1",
  messages: [
    { parentId: null, message: userRuntime("u1", "u1") },
    { parentId: "u1", message: assistantRuntime("a1", "a1") },
  ],
});

test("turns added by another client are appended and become the head", () => {
  const next = appendExternalStoredTurns(
    openView() as never,
    [stored("u1", "user", null, 1), stored("a1", "assistant", "u1", 2), stored("a2", "assistant", "u2", 4), stored("u2", "user", "a1", 3)],
    toRuntime,
  );
  assert.ok(next);
  assert.deepEqual(next.messages.map((item) => item.message.id), ["u1", "a1", "u2", "a2"]);
  assert.equal(next.messages[2].parentId, "a1");
  assert.equal(next.headId, "a2");
});

test("nothing is appended when the view already has every stored turn", () => {
  assert.equal(
    appendExternalStoredTurns(openView() as never, [stored("u1", "user", null, 1), stored("a1", "assistant", "u1", 2)], toRuntime),
    null,
  );
});

test("a running view is left alone", () => {
  const view = openView();
  (view.messages[1].message as { status: unknown }).status = { type: "running" };
  assert.equal(
    appendExternalStoredTurns(view as never, [stored("u2", "user", "a1", 3)], toRuntime),
    null,
  );
});

test("a turn on another branch is added without moving the head", () => {
  const next = appendExternalStoredTurns(
    openView() as never,
    [stored("a1b", "assistant", "u1", 3)],
    toRuntime,
  );
  assert.ok(next);
  assert.equal(next.headId, "a1");
  assert.equal(next.messages.length, 3);
});

test("the open view imports external turns on reconcile", () => {
  let imported: { headId?: string | null; messages: unknown[] } | null = null;
  const view = {
    threadListItem: () => ({ getState: () => ({ remoteId: "thread-1" }) }),
    thread: () => ({ export: () => openView() as never, import: (data: { headId?: string | null; messages: unknown[] }) => { imported = data; } }),
  };
  const changed = reconcileOrdinarySavedMessagesInView(
    view,
    "thread-1",
    [stored("u1", "user", null, 1), stored("a1", "assistant", "u1", 2), stored("u2", "user", "a1", 3)] as never,
    { editingMessageId: null, toThreadMessage: toRuntime },
  );
  assert.equal(changed, true);
  assert.equal(imported!.headId, "u2");
});

test("an assistant reply still streaming elsewhere is added so recovery can follow it", () => {
  const streaming = {
    ...stored("a2", "assistant", "u2", 4),
    metadata: { generationRunId: "run-phone", generationStatus: "running" },
  };
  const next = appendExternalStoredTurns(
    openView() as never,
    [stored("u2", "user", "a1", 3), streaming],
    toRuntime,
  );
  assert.deepEqual(next?.messages.map((item) => item.message.id), ["u1", "a1", "u2", "a2"]);
  assert.equal(next?.headId, "a2");
});

test("an open edit is not interrupted by external turns", () => {
  let imported = false;
  const view = {
    threadListItem: () => ({ getState: () => ({ remoteId: "thread-1" }) }),
    thread: () => ({ export: () => openView() as never, import: () => { imported = true; } }),
  };
  const changed = reconcileOrdinarySavedMessagesInView(
    view,
    "thread-1",
    [stored("u2", "user", "a1", 3)] as never,
    { editingMessageId: "u1", toThreadMessage: toRuntime },
  );
  assert.equal(changed, false);
  assert.equal(imported, false);
});

test("switching threads reconciles only the view that owns the new active thread", async () => {
  type State = { activeThreadId: string | null };
  const listeners = new Set<(state: State, prev: State) => void>();
  let state: State = { activeThreadId: "thread-1" };
  const store = {
    subscribe: (listener: (state: State, prev: State) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
  const set = (activeThreadId: string | null) => {
    const prev = state;
    state = { activeThreadId };
    for (const listener of listeners) listener(state, prev);
  };
  const recovered: string[] = [];
  const stops = ["thread-1", "thread-2"].map((own) =>
    subscribeThreadSwitchRecovery(store, (id) => id === own, () => recovered.push(own)),
  );
  set("thread-2");
  set(null);
  set("thread-1");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(recovered, ["thread-2", "thread-1"]);
  for (const stop of stops) stop();
  set("thread-2");
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(recovered, ["thread-2", "thread-1"]);
});
