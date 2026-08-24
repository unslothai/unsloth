// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";
import type { ParsedConversation } from "../src/features/chat/types.ts";

let vite: ViteDevServer;
let parseImportText: (text: string, filename: string) => ParsedConversation[];

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { middlewareMode: true },
  });
  const loaded = await vite.ssrLoadModule(
    "/src/features/chat/utils/chat-import.ts",
  );
  parseImportText = loaded.parseImportText as typeof parseImportText;
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
