// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { after, before, test } from "node:test";
import ts from "typescript";
import { type ViteDevServer, createServer } from "vite";
import type { ParsedConversation } from "../src/features/chat/types.ts";

let vite: ViteDevServer;
let parseImportText: (text: string, filename: string) => ParsedConversation[];
let messageToOpenAI: (message: {
  role: unknown;
  content: unknown;
  attachments?: unknown;
}) => unknown[];

function loadMessageToOpenAI(): typeof messageToOpenAI {
  const source = readFileSync(
    new URL(
      "../src/features/chat/prompt-storage/prompt-storage-dialog.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const start = source.indexOf("type OAIContentPart =");
  const end = source.indexOf("// ShareGPT training JSONL", start);
  assert.notEqual(start, -1, "message serializer start marker must exist");
  assert.notEqual(end, -1, "message serializer end marker must exist");
  const exactSerializer =
    source.slice(start, end) +
    "\nglobalThis.__messageToOpenAI = messageToOpenAI;\n";
  const javascript = ts.transpileModule(exactSerializer, {
    compilerOptions: {
      module: ts.ModuleKind.None,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const context = {
    unwrapPastedTextContent: (text: string) => text,
    toolResultModelText: (result: unknown) => result,
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__messageToOpenAI as typeof messageToOpenAI;
}

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/utils/chat-import.ts",
  );
  parseImportText = loaded.parseImportText as typeof parseImportText;
  messageToOpenAI = loadMessageToOpenAI();
});

after(async () => {
  await vite.close();
});

test("message JSONL imports as one conversation", () => {
  const conversations = parseImportText(
    '{"role":"user","content":"Hello"}\n' +
      '{"role":"assistant","content":"Hi"}',
    "conversation-messages.jsonl",
  );

  assert.equal(conversations.length, 1);
  assert.equal(conversations[0].title, "conversation-messages");
  assert.deepEqual(
    conversations[0].messages.map(({ role }) => role),
    ["user", "assistant"],
  );
});

test("developer and assistant array content survive message JSONL import", () => {
  const image = "data:image/png;base64,QUFBQQ==";
  const [conversation] = parseImportText(
    '{"role":"developer","content":"Follow policy"}\n' +
      '{"role":"assistant","content":[{"type":"text","text":"Done"},{"type":"image_url","image_url":{"url":"' +
      image +
      '"}}]}',
    "conversation-messages.jsonl",
  );

  assert.deepEqual(
    conversation.messages.map(({ role }) => role),
    ["system", "assistant"],
  );
  assert.deepEqual(conversation.messages[1].content, [
    { type: "text", text: "Done" },
    { type: "image", image },
  ]);
});

test("assistant images are represented explicitly in JSONL exports", () => {
  const exported = structuredClone(
    messageToOpenAI({
      role: "assistant",
      content: [
        { type: "text", text: "Chart" },
        { type: "image", image: "data:image/png;base64,QUFBQQ==" },
      ],
    }),
  );
  assert.deepEqual(
    exported,
    [{ role: "assistant", content: "Chart\n\n[image attachment]" }],
  );
});
