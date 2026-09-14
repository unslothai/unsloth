// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import {
  exportFormatIncludesSiblings,
  ndjsonBody,
} from "../src/features/chat/utils/ndjson.ts";
import { readSrc } from "./helpers/kit.ts";

type StoredMessage = {
  id: string;
  parentId: string | null;
  createdAt: number;
  role: string;
  content: string;
};

type Exporters = {
  exportConversationShareGPT: (threadId: string) => Promise<void>;
  buildThreadContent: (
    threadId: string,
    format: string,
  ) => Promise<string | null>;
};

const SOURCE = readSrc(
  "features/chat/prompt-storage/prompt-storage-dialog.tsx",
);

function sliceSource(startMarker: string, endMarker: string): string {
  const start = SOURCE.indexOf(startMarker);
  const end = SOURCE.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return SOURCE.slice(start, end);
}

function storedMessages(
  rows: [id: string, parentId: string | null, role: string, content: string][],
): StoredMessage[] {
  return rows.map(([id, parentId, role, content], index) => ({
    id,
    parentId,
    createdAt: index + 1,
    role,
    content,
  }));
}

function loadExporters(
  stored: StoredMessage[],
  downloads: string[] = [],
  headId: string | null = null,
) {
  const javascript = ts.transpileModule(
    [
      sliceSource(
        "async function loadConversationMessages(",
        "function exportTs(",
      ),
      sliceSource("// ShareGPT training JSONL", "// OpenAI/ChatML JSONL"),
      sliceSource("async function buildThreadContent(", "function csvHeader("),
      "globalThis.__exporters = { exportConversationShareGPT, buildThreadContent };",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const context = {
    exports: {},
    toast: { info: () => {} },
    listStoredChatMessages: async () => stored,
    liveThreadHeadId: () => headId,
    orderByParentChain,
    exportFormatIncludesSiblings,
    ndjsonBody,
    messageToText: (message: StoredMessage) => message.content,
    csvEscape: (value: string) => value,
    exportTs: () => "ts",
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__exporters as Exporters;
}

async function shareGptConversations(
  stored: StoredMessage[],
  headId: string | null = null,
) {
  const downloads: string[] = [];
  const exporters = loadExporters(stored, downloads, headId);
  await exporters.exportConversationShareGPT("thread");
  const bulk = await exporters.buildThreadContent("thread", "sharegpt");
  assert.ok(bulk);
  assert.deepEqual(downloads, [`${bulk}\n`]);
  return JSON.parse(bulk).conversations;
}

const regenerated = storedMessages([
  ["u1", null, "user", "Name one fruit."],
  ["a1", "u1", "assistant", "Apples."],
  ["a1-retry", "u1", "assistant", "Apples!"],
]);

test("ShareGPT exports only the regenerated reply", async () => {
  assert.deepEqual(await shareGptConversations(regenerated), [
    { from: "human", value: "Name one fruit." },
    { from: "gpt", value: "Apples!" },
  ]);
});

test("ShareGPT exports only the edited prompt and its reply", async () => {
  const edited = storedMessages([
    ["u1", null, "user", "Name one color."],
    ["a1", "u1", "assistant", "Blue."],
    ["u1-edit", null, "user", "Name one animal."],
    ["a1-edit", "u1-edit", "assistant", "Cat."],
  ]);
  assert.deepEqual(await shareGptConversations(edited), [
    { from: "human", value: "Name one animal." },
    { from: "gpt", value: "Cat." },
  ]);
});

test("ShareGPT exports the branch picked in the branch picker", async () => {
  assert.deepEqual(await shareGptConversations(regenerated, "a1"), [
    { from: "human", value: "Name one fruit." },
    { from: "gpt", value: "Apples." },
  ]);
  const edited = storedMessages([
    ["u1", null, "user", "Name one color."],
    ["a1", "u1", "assistant", "Blue."],
    ["u1-edit", null, "user", "Name one animal."],
    ["a1-edit", "u1-edit", "assistant", "Cat."],
  ]);
  assert.deepEqual(await shareGptConversations(edited, "a1"), [
    { from: "human", value: "Name one color." },
    { from: "gpt", value: "Blue." },
  ]);
});

test("ShareGPT falls back to the newest branch when the head is not stored", async () => {
  assert.deepEqual(await shareGptConversations(regenerated, "unsaved"), [
    { from: "human", value: "Name one fruit." },
    { from: "gpt", value: "Apples!" },
  ]);
});

test("CSV still exports every branch", async () => {
  const csv = await loadExporters(regenerated).buildThreadContent(
    "thread",
    "csv",
  );
  assert.equal(
    csv,
    "user,Name one fruit.\nassistant,Apples!\nassistant,Apples.",
  );
});
