// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";
import type { ParsedConversation } from "../src/features/chat/types.ts";
import {
  COMBINED_CSV_HEADER,
  csvDocument,
  csvEscape,
} from "../src/features/chat/utils/csv-export.ts";

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

test("combined CSV exports of several chats write the chat column header", () => {
  const source = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
  const merged = source.slice(source.indexOf("export async function exportBulkConversationsMerged"));
  assert.match(merged, /format === "csv" && threadIds\.length > 1[\s\S]*?combinedConversationsCsv\(threadIds\)/);
  assert.match(source, /csvDocument\(\[COMBINED_CSV_HEADER, \.\.\.rows\]\)/);
});
