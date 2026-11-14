// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";
import { type ViteDevServer, createServer } from "vite";
import type { ParsedConversation } from "../src/features/chat/types.ts";
import {
  COMBINED_CSV_HEADER,
  CSV_MIME,
  csvDocument,
  csvEscape,
} from "../src/features/chat/utils/csv-export.ts";
import { canMergeConversationExport } from "../src/features/chat/utils/ndjson.ts";
import { planChatItemSources } from "../src/features/chat/utils/project-source-plan.ts";

import { readSrc } from "./helpers/kit.ts";

let vite: ViteDevServer;
let parseImportText: (text: string, filename: string) => ParsedConversation[];

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule("/src/features/chat/utils/chat-import.ts");
  parseImportText = loaded.parseImportText as typeof parseImportText;
});

after(async () => {
  await vite.close();
});

const row = (...cells: string[]) => cells.map(csvEscape).join(",");

const summary = (conversations: ParsedConversation[]) =>
  conversations.map(({ title, messages }) => ({
    title,
    messages: messages.map(({ role, content, parentId }, index) => ({
      role,
      text: (content as readonly { text?: string }[])[0].text,
      chained: index === 0 ? parentId === null : parentId === messages[index - 1].id,
    })),
  }));

test("a combined CSV of several chats imports as one chat per chat, with titles", () => {
  const doc = csvDocument([
    COMBINED_CSV_HEADER,
    row("1", "Python list question", "user", "How do I reverse a list?"),
    row("1", "Python list question", "assistant", "Use my_list[::-1], or a, b"),
    row("2", "Paris trip plan", "user", "Plan a trip.\nTwo days."),
    row("2", "Paris trip plan", "assistant", "Day 1: Louvre."),
  ]);

  const conversations = parseImportText(doc, "all-chats.csv");

  assert.deepEqual(summary(conversations), [
    {
      title: "Python list question",
      messages: [
        { role: "user", text: "How do I reverse a list?", chained: true },
        { role: "assistant", text: "Use my_list[::-1], or a, b", chained: true },
      ],
    },
    {
      title: "Paris trip plan",
      messages: [
        { role: "user", text: "Plan a trip.\nTwo days.", chained: true },
        { role: "assistant", text: "Day 1: Louvre.", chained: true },
      ],
    },
  ]);
  assert.notEqual(conversations[0].threadId, conversations[1].threadId);
});

test("chats that share a title stay apart, and an empty title falls back to the file name", () => {
  const doc = csvDocument([
    COMBINED_CSV_HEADER,
    row("1", "New Chat", "user", "first"),
    row("2", "New Chat", "user", "second"),
    row("3", "", "user", "third"),
  ]);

  assert.deepEqual(
    parseImportText(doc, "chats.csv").map(({ title, messages }) => [title, messages.length]),
    [
      ["New Chat", 1],
      ["New Chat", 1],
      ["chats 3", 1],
    ],
  );
});

test("a role,content CSV still imports as one chat named after the file", () => {
  const doc = csvDocument([
    "role,content",
    row("user", "Hello"),
    row("assistant", "Hi there"),
    "user,unquoted, with a comma",
  ]);

  assert.deepEqual(summary(parseImportText(doc, "my-chat.csv")), [
    {
      title: "my-chat",
      messages: [
        { role: "user", text: "Hello", chained: true },
        { role: "assistant", text: "Hi there", chained: true },
        { role: "user", text: "unquoted, with a comma", chained: true },
      ],
    },
  ]);
});

const SOURCE = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");

function exportMerged(
  chats: Record<string, { title: string; messages: [string, string][] }>,
): (threadIds: string[]) => Promise<string[]> {
  const start = SOURCE.indexOf("async function buildThreadContent(");
  const end = SOURCE.indexOf("export async function exportBulkConversationsSeparate(");
  assert.ok(start !== -1 && end > start);
  const javascript = ts.transpileModule(
    `${SOURCE.slice(start, end)}\nglobalThis.__merged = exportBulkConversationsMerged;`,
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  const downloads: string[] = [];
  const context: Record<string, unknown> = {
    exports: {},
    toast: { info: () => {} },
    canMergeConversationExport,
    exportFormatIncludesSiblings: () => false,
    getStoredChatThread: async (id: string) => ({ id, title: chats[id].title }),
    planChatItemSources,
    loadConversationMessages: async (id: string) =>
      chats[id].messages.map(([role, content]) => ({ role, content })),
    messageToText: (message: { content: string }) => message.content,
    csvEscape,
    csvDocument,
    COMBINED_CSV_HEADER,
    CSV_MIME,
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
  };
  vm.runInNewContext(javascript, context);
  const merged = context.__merged as (ids: string[], format: string, basename: string) => Promise<void>;
  return async (threadIds) => {
    downloads.length = 0;
    await merged(threadIds, "csv", "all-chats");
    return [...downloads];
  };
}

test("a Combined CSV export of several chats imports back as the same chats", async () => {
  const run = exportMerged({
    a: { title: "Python list question", messages: [["user", "Reverse a list?"], ["assistant", 'Use "a[::-1]", or a, b\nok']] },
    b: { title: "Paris trip plan", messages: [["user", "Plan a trip."], ["assistant", ""], ["assistant", "Day 1."]] },
    empty: { title: "Nothing", messages: [["user", "  "]] },
  });

  const [body] = await run(["a", "empty", "b"]);
  assert.ok(body.startsWith(`\ufeff${COMBINED_CSV_HEADER}\n`));
  assert.deepEqual(summary(parseImportText(body, "all-chats.csv")), [
    {
      title: "Python list question",
      messages: [
        { role: "user", text: "Reverse a list?", chained: true },
        { role: "assistant", text: 'Use "a[::-1]", or a, b\nok', chained: true },
      ],
    },
    {
      title: "Paris trip plan",
      messages: [
        { role: "user", text: "Plan a trip.", chained: true },
        { role: "assistant", text: "Day 1.", chained: true },
      ],
    },
  ]);

  const [single] = await run(["a"]);
  assert.ok(single.startsWith("\ufeffrole,content\n"));
  assert.deepEqual(parseImportText(single, "one.csv").map(({ title }) => title), ["one"]);
});
