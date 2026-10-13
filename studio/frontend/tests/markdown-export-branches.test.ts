// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import { stripSearchImageTokens } from "../src/features/chat/search-images/search-images.ts";
import {
  buildNamedConversationsMarkdown,
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";
import {
  buildConversationMarkdown,
  CONVERSATION_MARKDOWN_MIME_TYPE,
} from "../src/features/chat/utils/conversation-markdown.ts";
import { csvDocument, CSV_MIME } from "../src/features/chat/utils/csv-export.ts";
import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import { parseConversationMarkdownDocument } from "../src/features/chat/utils/conversation-markdown-import.ts";
import { canMergeConversationExport } from "../src/features/chat/utils/ndjson.ts";
import { planChatItemSources } from "../src/features/chat/utils/project-source-plan.ts";
import type { ThreadRecord } from "../src/features/chat/types.ts";
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
  exportBulkConversationsMerged: (
    threadIds: string[],
    format: "markdown",
    basename: string,
  ) => Promise<void>;
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
  threads: ThreadRecord[] = [],
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
      sliceSource(
        "export async function exportBulkConversationsMerged(",
        "export async function exportBulkConversationsSeparate(",
      ),
      "globalThis.__exporters = { buildConversationMarkdownForThread, exportConversationMarkdown, saveConversationAsProjectSource, exportConversationCsv, exportBulkConversationsMerged };",
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
    getStoredChatThread: async (id: string) =>
      threads.find((thread) => thread.id === id),
    resolveChatInstructions: async () => "",
    threadScopedDefault: () => undefined,
    settleThreadScopedSettingsForCopy: async () => {},
    buildNamedConversationsMarkdown,
    CONVERSATION_MARKDOWN_MIME_TYPE,
    canMergeConversationExport,
    planChatItemSources,
    ...liveThreadHead,
    savedBranchHead: () => undefined,
    orderByParentChain,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    buildConversationMarkdown,
    stripSearchImageTokens,
    messageToMarkdown: (message: StoredMessage) => message.content,
    messageToText: (message: StoredMessage) => message.content,
    csvEscape: (value: string) => value,
    csvDocument,
    CSV_MIME,
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
        threads: () => ({ getState: () => ({ mainThreadId: "thread" }) }),
        threadListItem: () => ({
          getState: () => ({ id: "thread", remoteId: "thread" }),
        }),
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
  const expectedExport = `<!-- unsloth-chat-v1:[24,21] -->\n\n${expected}`;
  assert.deepEqual(await markdownOutputs(regenerated, ["u1", "a1"]), {
    copied: expectedExport,
    downloads: [expectedExport],
    sources: [expected],
  });
});

test("Markdown leaves out the reply a regeneration replaced", async () => {
  const expected = "## User\n\nName one fruit.\n\n## Assistant\n\nPears.\n";
  const expectedExport = `<!-- unsloth-chat-v1:[24,20] -->\n\n${expected}`;
  assert.deepEqual(await markdownOutputs(regenerated), {
    copied: expectedExport,
    downloads: [expectedExport],
    sources: [expected],
  });
});

// Mid-switch the branch on screen is briefly an empty list, which is no opinion about which reply is showing.
test("Markdown still exports while the switched-to chat is loading", async () => {
  const expected = "## User\n\nName one fruit.\n\n## Assistant\n\nPears.\n";
  const expectedExport = `<!-- unsloth-chat-v1:[24,20] -->\n\n${expected}`;
  assert.deepEqual(await markdownOutputs(regenerated, []), {
    copied: expectedExport,
    downloads: [expectedExport],
    sources: [expected],
  });
});

// CSV must not follow markdown onto one branch: it is the export used to compare versions.
test("CSV still writes both replies while markdown writes one", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(regenerated, downloads, []);
  await exporters.exportConversationCsv("thread");
  assert.deepEqual(downloads, [
    "\ufeffrole,content\nuser,Name one fruit.\nassistant,Pears.\nassistant,Apples.",
  ]);
});

test("Markdown follows the prompt version on screen", async () => {
  const older = "## User\n\nName one color.\n\n## Assistant\n\nBlue.\n";
  const olderExport = `<!-- unsloth-chat-v1:[24,19] -->\n\n${older}`;
  assert.deepEqual(await markdownOutputs(edited, ["u1", "a1"]), {
    copied: olderExport,
    downloads: [olderExport],
    sources: [older],
  });
  const newer = "## User\n\nName one animal.\n\n## Assistant\n\nCat.\n";
  const newerExport = `<!-- unsloth-chat-v1:[25,18] -->\n\n${newer}`;
  assert.deepEqual(await markdownOutputs(edited), {
    copied: newerExport,
    downloads: [newerExport],
    sources: [newer],
  });
});

for (const { models, panes, titles } of [
  {
    models: ["org/Alpha", "org/Beta"],
    panes: ["model1", "model2"],
    titles: ["Compare - Beta", "Standalone", "Compare - Alpha"],
  },
  {
    models: ["org/Alpha", "org/Alpha"],
    panes: ["base", "lora"],
    titles: [
      "Compare - Alpha - fine-tuned",
      "Standalone",
      "Compare - Alpha - base",
    ],
  },
]) {
  test(`combined Markdown names comparison halves: ${panes.join("/")}`, async () => {
    const threads = [
      ...models.map((modelId, index) => ({
        id: `half-${index}`,
        title: "Compare",
        pairId: "pair",
        modelId,
        modelType: panes[index],
        createdAt: 1,
      })),
      { id: "single", title: "Standalone", modelType: "base", createdAt: 1 },
    ] as ThreadRecord[];
    const downloads: string[] = [];
    const exporters = loadExporters(regenerated, downloads, [], threads);
    await exporters.exportBulkConversationsMerged(
      ["half-1", "single", "half-0"],
      "markdown",
      "chats",
    );
    assert.equal(downloads.length, 1);
    assert.deepEqual(
      parseConversationMarkdownDocument(downloads[0], "chats").map(
        ({ title }) => title,
      ),
      titles,
    );
  });
}
