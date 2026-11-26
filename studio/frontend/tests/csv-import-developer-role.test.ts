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

const turns = (conversations: ParsedConversation[]) =>
  conversations.map(({ messages }) =>
    messages.map(({ role, content }) => [
      role,
      (content as readonly { text?: string }[])[0].text,
    ]),
  );

const chat = [
  ["system", "Answer in one word."],
  ["user", "Capital of France?"],
  ["assistant", "Paris."],
];

test("a CSV developer row imports as a system message", () => {
  const doc = csvDocument([
    "role,content",
    row("developer", "Answer in one word."),
    row("user", "Capital of France?"),
    row("assistant", "Paris."),
  ]);
  assert.deepEqual(turns(parseImportText(doc, "chat.csv")), [chat]);
});

test("a headerless CSV that starts with a developer row keeps that row", () => {
  const doc = [
    row("Developer", "Answer in one word."),
    row("user", "Capital of France?"),
    row("assistant", "Paris."),
  ].join("\n");
  assert.deepEqual(turns(parseImportText(doc, "chat.csv")), [chat]);
});

test("CSV and JSONL give the same roles for a developer turn", () => {
  const csv = parseImportText(
    ["role,content", row("developer", "Answer in one word."), row("user", "Capital of France?"), row("assistant", "Paris.")].join("\n"),
    "chat.csv",
  );
  const jsonl = parseImportText(
    JSON.stringify({
      messages: [
        { role: "developer", content: "Answer in one word." },
        { role: "user", content: "Capital of France?" },
        { role: "assistant", content: "Paris." },
      ],
    }),
    "chat.jsonl",
  );
  assert.deepEqual(turns(csv), turns(jsonl));
});

test("a combined CSV keeps developer rows as system messages in each chat", () => {
  const doc = csvDocument([
    COMBINED_CSV_HEADER,
    row("1", "Quiz", "developer", "Answer in one word."),
    row("1", "Quiz", "user", "Capital of France?"),
    row("1", "Quiz", "assistant", "Paris."),
    row("2", "Other", "user", "Hi"),
  ]);
  assert.deepEqual(turns(parseImportText(doc, "all.csv")), [chat, [["user", "Hi"]]]);
});

test("unknown CSV roles still import as user messages", () => {
  const doc = ["role,content", row("narrator", "Once upon a time.")].join("\n");
  assert.deepEqual(turns(parseImportText(doc, "chat.csv")), [[["user", "Once upon a time."]]]);
});
