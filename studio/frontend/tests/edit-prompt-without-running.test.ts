// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { MessageRepository } from "@assistant-ui/core/internal";
import * as researchSync from "../src/features/chat/utils/research-message-sync.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Exported = {
  headId: string | null;
  messages: { parentId: string | null; message: Record<string, unknown> }[];
};

type Module = {
  updateThreadMessage: (args: {
    thread: { export: () => Exported; import: (data: Exported) => void };
    messageId: string;
    remoteId: string | undefined;
    newText: string;
    isIncognito: boolean;
    plainText?: boolean;
  }) => Promise<unknown>;
};

const RECORD_MODULE = loadWithStubs(
  new URL("../src/features/chat/utils/delete-thread-message.ts", import.meta.url),
  {
    "@assistant-ui/core/internal": { MessageRepository },
    "../api/chat-api": { listChatMessages: async () => [] },
    "./chat-history-storage": {
      ensureStoredChatThread: async () => {},
      syncStoredChatMessages: async (_threadId: string, records: unknown[]) => records,
    },
    "./research-message-sync": researchSync,
  },
);

const ATTACHMENT = {
  id: "att-1",
  type: "image",
  name: "cat.png",
  contentType: "image/png",
  status: { type: "complete" },
  content: [{ type: "image", image: "data:image/png;base64,AAAA" }],
};

function conversation(): Exported {
  return {
    headId: "a0",
    messages: [
      {
        parentId: null,
        message: {
          id: "u0",
          role: "user",
          content: [{ type: "text", text: "summarise the chat" }],
          attachments: [ATTACHMENT],
          createdAt: new Date(1000),
        },
      },
      {
        parentId: "u0",
        message: {
          id: "a0",
          role: "assistant",
          content: [{ type: "text", text: "the summary" }],
          createdAt: new Date(2000),
        },
      },
    ],
  };
}

async function savePrompt(newText: string) {
  const saved: Record<string, unknown>[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/utils/update-thread-message.ts", import.meta.url),
    {
      "../api/chat-api": {
        saveChatMessage: async (record: Record<string, unknown>, options: unknown) => {
          saved.push({ record, options });
          return record;
        },
      },
      "./delete-thread-message": RECORD_MODULE,
      "./research-message-sync": researchSync,
    },
  );
  let current = conversation();
  const before = structuredClone(current);
  await module.updateThreadMessage({
    thread: { export: () => current, import: (data) => (current = data) },
    messageId: "u0",
    remoteId: "remote-1",
    newText,
    isIncognito: false,
    plainText: true,
  });
  return { saved, before, after: current };
}

test("a prompt saved in place keeps its id, parent, attachments and reply", async () => {
  const { saved, before, after } = await savePrompt("summarise only the last turn");

  assert.equal(saved.length, 1);
  const { record, options } = saved[0] as {
    record: Record<string, unknown>;
    options: Record<string, unknown>;
  };
  assert.deepEqual(options, { allowGenerationEdit: true });
  assert.equal(record.id, "u0");
  assert.equal(record.role, "user");
  assert.equal(record.parentId, null);
  assert.equal(record.createdAt, 1000);
  assert.deepEqual(record.content, [{ type: "text", text: "summarise only the last turn" }]);
  assert.deepEqual(record.attachments, [ATTACHMENT]);
  // Same tree, same head: no new branch and the reply under the prompt is untouched.
  assert.equal(after.headId, before.headId);
  assert.deepEqual(
    after.messages.map((m) => [m.parentId, m.message.id]),
    before.messages.map((m) => [m.parentId, m.message.id]),
  );
  assert.deepEqual(after.messages[1], before.messages[1]);
});

test("a prompt's own <THINK> and tool marker text stays prose", async () => {
  const text = "explain <THINK>tags</THINK> and <TOOL 1: search> markers";
  const { saved } = await savePrompt(text);
  const { record } = saved[0] as { record: Record<string, unknown> };
  assert.deepEqual(record.content, [{ type: "text", text }]);
});

test("saving the same prompt text twice writes the same record", async () => {
  const first = await savePrompt("same text");
  const second = await savePrompt("same text");
  assert.deepEqual(first.saved, second.saved);
  assert.deepEqual(first.after, second.after);
});

const THREAD = readSrc("components/assistant-ui/thread.tsx");

function component(name: string): string {
  const start = THREAD.indexOf(`const ${name}: FC = () => {`);
  assert.notEqual(start, -1, `${name} is gone; this test needs rewriting`);
  return THREAD.slice(start, THREAD.indexOf("\n};", start));
}

test("a prompt's bar ends in Delete then a More menu holding Edit without running", () => {
  const bar = component("UserActionBar");
  assert.match(bar, /<DeleteMessageButton \/>\s*\{editable && <UserMoreMenu \/>\}\s*<\/UserMessageActionBar>/);
  assert.match(component("UserMoreMenu"), /<EditWithoutRunMenuItem \/>/);
  const item = component("EditWithoutRunMenuItem");
  assert.match(item, /Edit without running/);
  assert.match(item, /disabled=\{isRunning\}/);
  assert.match(item, /setEditWithoutRun\(messageId, true\);\s*aui\.composer\(\)\.beginEdit\(\);/);
});

test("the Edit button keeps sending: it clears save-only mode first", () => {
  assert.match(
    component("UserActionBar"),
    /tooltip="Edit"[\s\S]*?onClick=\{\(\) => setEditWithoutRun\(messageId, false\)\}/,
  );
});

test("save-only edits write in place and never start a run", () => {
  const editor = component("EditComposer");
  assert.match(editor, /if \(saveOnly\) void saveEdit\(\);\s*else submitEdit\(\);/);
  const save = editor.slice(
    editor.indexOf("const saveEdit = useCallback("),
    editor.indexOf("}, [aui, incognito, isComposingRef, messageId, researchActive]);"),
  );
  assert.match(save, /plainText: true/);
  assert.doesNotMatch(save, /\.send\(|startRun/);
  assert.match(editor, /\{saveOnly \? \(\s*<Button type="submit" size="sm" disabled=\{researchActive \|\| isRunning \|\| editEmpty\}>\s*Save/);
  // Enter skips the button: the same guards run inside the save.
  assert.match(save, /researchActive \|\| aui\.thread\(\)\.getState\(\)\.isRunning\) return;/);
  assert.match(save, /if \(aui\.composer\(\)\.getState\(\)\.isEmpty\) return;/);
});

test("each open editor keeps its own save-only mode", () => {
  // A set of prompt ids, not one slot: opening a second editor must not turn the first into Send.
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(store, /editWithoutRunMessageIds: readonly string\[\];/);
  assert.match(
    store,
    /setEditWithoutRun: \(id, on\) =>\s*set\(\(s\) => \{\s*const ids = s\.editWithoutRunMessageIds\.filter\(\(x\) => x !== id\);\s*return \{ editWithoutRunMessageIds: on \? \[\.\.\.ids, id\] : ids \};/,
  );
  // Captured at mount, so the cleanup StrictMode replays on mount cannot flip Save to Send.
  assert.match(
    component("EditComposer"),
    /const \[saveOnly\] = useState\(\(\) =>\s*useChatRuntimeStore\.getState\(\)\.editWithoutRunMessageIds\.includes\(messageId\),\s*\);/,
  );
});
