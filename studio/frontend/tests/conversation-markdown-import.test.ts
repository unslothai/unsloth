// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { buildConversationMarkdown } from "../src/features/chat/utils/conversation-markdown.ts";
import {
  parseConversationMarkdownDocument,
  parseConversationMarkdownMessages,
} from "../src/features/chat/utils/conversation-markdown-import.ts";

test("round-trips a single exported conversation", () => {
  const exported = buildConversationMarkdown([
    { role: "system", content: "Be concise." },
    { role: "user", content: "Explain `RED → GREEN`." },
    { role: "assistant", content: "1. Write a failing test.\n2. Fix it." },
  ]);
  const [parsed] = parseConversationMarkdownDocument(exported, "chat");
  assert.equal(parsed?.title, "chat");
  assert.deepEqual(
    parsed?.messages.map(({ role, content }) => ({ role, content })),
    [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Explain `RED → GREEN`." },
      { role: "assistant", content: "1. Write a failing test.\n2. Fix it." },
    ],
  );
});

test("parses combined bulk exports with titles and separators", () => {
  const first = `# First chat\n\n${buildConversationMarkdown([
    { role: "user", content: "Hi" },
  ])}`;
  const second = `# Second chat\n\n${buildConversationMarkdown([
    { role: "assistant", content: "Hello" },
  ])}`;
  const combined = `${first}---\n\n${second}`;
  const parsed = parseConversationMarkdownDocument(combined, "export");
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0]?.title, "First chat");
  assert.equal(parsed[1]?.title, "Second chat");
  assert.equal(parsed[0]?.messages[0]?.content, "Hi");
  assert.equal(parsed[1]?.messages[0]?.content, "Hello");
});

test("does not treat inline headings as role breaks", () => {
  const body = buildConversationMarkdown([
    { role: "assistant", content: "# Existing heading\n\n> quote" },
  ]);
  const messages = parseConversationMarkdownMessages(body);
  assert.equal(messages.length, 1);
  assert.equal(messages[0]?.content, "# Existing heading\n\n> quote");
});
