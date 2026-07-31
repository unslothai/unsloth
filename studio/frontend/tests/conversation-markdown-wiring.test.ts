// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const CHAT_SOURCE_ROOT = new URL("../src/features/chat/", import.meta.url);
const THREAD_SOURCE = new URL(
  "../src/components/assistant-ui/thread.tsx",
  import.meta.url,
);

async function chatSource(relativePath: string): Promise<string> {
  return readFile(new URL(relativePath, CHAT_SOURCE_ROOT), "utf8");
}

test("the single-chat export menu wires Markdown to the active thread", async () => {
  const source = await readFile(THREAD_SOURCE, "utf8");
  assert.match(source, /\{CONVERSATION_MARKDOWN_LABEL\}/);
  assert.match(source, /exportConversationMarkdown\(activeThreadId\)/);
});

test("the compare-chat export menu wires Markdown to every selected thread", async () => {
  const source = await chatSource("shared-composer.tsx");
  assert.match(source, /label:\s*CONVERSATION_MARKDOWN_LABEL/);
  assert.match(source, /fn:\s*exportConversationMarkdown/);
});

test("the sidebar chat menu exposes the Markdown exporter", async () => {
  const source = await chatSource("thread-sidebar.tsx");
  assert.match(
    source,
    /\{\s*label:\s*CONVERSATION_MARKDOWN_LABEL,\s*fn:\s*exportConversationMarkdown\s*\}/,
  );
});

test("project chat menus dispatch Markdown through the conversation exporter", async () => {
  const source = await chatSource("chat-page.tsx");
  assert.match(source, /label:\s*CONVERSATION_MARKDOWN_LABEL/);
  assert.match(
    source,
    /format === CONVERSATION_MARKDOWN_FORMAT[\s\S]*exportConversationMarkdown\(threadId\)/,
  );
});
