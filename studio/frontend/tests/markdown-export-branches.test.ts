// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import { stripSearchImageTokens } from "../src/features/chat/search-images/search-images.ts";
import {
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";
import { buildConversationMarkdown } from "../src/features/chat/utils/conversation-markdown.ts";
import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import { readSrc } from "./helpers/kit.ts";

type StoredMessage = {
  id: string;
  parentId: string | null;
  createdAt: number;
  role: string;
  content: string;
};

type Exporters = {
  buildConversationMarkdownForThread: (
    threadId: string,
  ) => Promise<string | null>;
  exportConversationMarkdown: (threadId: string) => Promise<void>;
  saveConversationAsProjectSource: (
    threadId: string,
    projectId: string,
    title: string,
  ) => Promise<string>;
  exportConversationCsv: (threadId: string) => Promise<void>;
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
  downloads: string[],
  sources: string[],
) {
  const javascript = ts.transpileModule(
    [
      sliceSource(
        "async function loadConversationMessages(",
        "function exportTs(",
      ),
      sliceSource(
        "export async function exportConversationCsv(",
        "export async function saveChatItemAsProjectSource(",
      ),
      "globalThis.__exporters = { buildConversationMarkdownForThread, exportConversationMarkdown, saveConversationAsProjectSource, exportConversationCsv };",
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
    ...liveThreadHead,
    orderByParentChain,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    buildConversationMarkdown,
    stripSearchImageTokens,
    messageToMarkdown: (message: StoredMessage) => message.content,
    messageToText: (message: StoredMessage) => message.content,
    csvEscape: (value: string) => value,
    exportTs: () => "ts",
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
    saveMarkdownAsProjectSource: async (_projectId: string, body: string) => {
      sources.push(body);
      return true;
    },
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__exporters as Exporters;
}

async function markdownOutputs(
  stored: StoredMessage[],
  liveBranch: string[] | null = null,
) {
  const downloads: string[] = [];
  const sources: string[] = [];
  const exporters = loadExporters(stored, downloads, sources);
  const unregister = liveBranch
    ? liveThreadHead.registerLiveThreadView({
        threadListItem: () => ({ getState: () => ({ remoteId: "thread" }) }),
        thread: () => ({
          getState: () => ({ messages: liveBranch.map((id) => ({ id })) }),
        }),
      })
    : () => {};
  try {
    const copied = await exporters.buildConversationMarkdownForThread("thread");
    await exporters.exportConversationMarkdown("thread");
    await exporters.saveConversationAsProjectSource("thread", "project", "t");
    return { copied, downloads, sources };
  } finally {
    unregister();
  }
}

const regenerated = storedMessages([
  ["u1", null, "user", "Name one fruit."],
  ["a1", "u1", "assistant", "Apples."],
  ["a1-retry", "u1", "assistant", "Pears."],
]);

const edited = storedMessages([
  ["u1", null, "user", "Name one color."],
  ["a1", "u1", "assistant", "Blue."],
  ["u1-edit", null, "user", "Name one animal."],
  ["a1-edit", "u1-edit", "assistant", "Cat."],
]);

test("Markdown follows the older reply picked in the branch picker", async () => {
  const expected = "## User\n\nName one fruit.\n\n## Assistant\n\nApples.\n";
  assert.deepEqual(await markdownOutputs(regenerated, ["u1", "a1"]), {
    copied: expected,
    downloads: [expected],
    sources: [expected],
  });
});

test("Markdown leaves out the reply a regeneration replaced", async () => {
  const expected = "## User\n\nName one fruit.\n\n## Assistant\n\nPears.\n";
  assert.deepEqual(await markdownOutputs(regenerated), {
    copied: expected,
    downloads: [expected],
    sources: [expected],
  });
});

// The markdown paths moving to the displayed branch must not drag CSV along: a spreadsheet of
// every version is the one export people use to compare them.
test("CSV still writes both replies while markdown writes one", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(regenerated, downloads, []);
  await exporters.exportConversationCsv("thread");
  assert.deepEqual(downloads, [
    "role,content\nuser,Name one fruit.\nassistant,Pears.\nassistant,Apples.",
  ]);
});

test("Markdown follows the prompt version on screen", async () => {
  const older = "## User\n\nName one color.\n\n## Assistant\n\nBlue.\n";
  assert.deepEqual(await markdownOutputs(edited, ["u1", "a1"]), {
    copied: older,
    downloads: [older],
    sources: [older],
  });
  const newer = "## User\n\nName one animal.\n\n## Assistant\n\nCat.\n";
  assert.deepEqual(await markdownOutputs(edited), {
    copied: newer,
    downloads: [newer],
    sources: [newer],
  });
});
