// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildNamedConversationsMarkdown,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";

type TestMessage = {
  readonly role: unknown;
  readonly text: string;
};

function exporterFor(
  messages: readonly TestMessage[] | null,
  downloads: Array<{
    content: string;
    filename: string;
    mimeType: string;
  }>,
  notifications: string[],
) {
  return createConversationMarkdownExporter<TestMessage>({
    loadMessages: async (threadId) => {
      assert.equal(threadId, "thread-1");
      return messages;
    },
    renderMessage: (message) => message.text,
    download: async (content, filename, mimeType) => {
      downloads.push({ content, filename, mimeType });
    },
    exportTimestamp: () => "2026-07-31T00-00-00",
    notifyNoContent: () => notifications.push("empty"),
  });
}

test("downloads a timestamped Markdown conversation", async () => {
  const downloads: Array<{
    content: string;
    filename: string;
    mimeType: string;
  }> = [];
  const notifications: string[] = [];
  await exporterFor(
    [{ role: "user", text: "Hello" }],
    downloads,
    notifications,
  )("thread-1");

  assert.deepEqual(downloads, [
    {
      content: "## User\n\nHello\n",
      filename: "conversation-2026-07-31T00-00-00.md",
      mimeType: "text/markdown",
    },
  ]);
  assert.deepEqual(notifications, []);
});

test("does nothing when the conversation does not exist", async () => {
  const downloads: Array<{
    content: string;
    filename: string;
    mimeType: string;
  }> = [];
  const notifications: string[] = [];
  await exporterFor(null, downloads, notifications)("thread-1");
  assert.deepEqual(downloads, []);
  assert.deepEqual(notifications, []);
});

test("notifies when every message is empty", async () => {
  const downloads: Array<{
    content: string;
    filename: string;
    mimeType: string;
  }> = [];
  const notifications: string[] = [];
  await exporterFor(
    [{ role: null, text: " " }],
    downloads,
    notifications,
  )("thread-1");
  assert.deepEqual(downloads, []);
  assert.deepEqual(notifications, ["empty"]);
});

test("a compare pair copies with each half under its model", async () => {
  const build = async (id: string) => `## User\n\nhi\n\n## Assistant\n\nfrom ${id}`;
  const markdown = await buildNamedConversationsMarkdown(
    [
      { id: "thread-1", title: "Chat - Qwen3-8B" },
      { id: "thread-2", title: "Chat - gpt-oss-20b" },
    ],
    build,
  );
  // Named, so the reader can tell which model wrote which answer.
  assert.match(markdown, /^# Chat - Qwen3-8B\n\n## User/);
  assert.ok(markdown.includes("\n---\n\n# Chat - gpt-oss-20b\n\n## User"));
});

test("a single chat copies exactly as the download writes it", async () => {
  const body = "## User\n\nhi";
  const markdown = await buildNamedConversationsMarkdown(
    [{ id: "thread-1", title: "Chat" }],
    async () => body,
  );
  assert.equal(markdown, body);
});

test("a half that loads nothing is dropped, not left as a bare heading", async () => {
  const markdown = await buildNamedConversationsMarkdown(
    [
      { id: "thread-1", title: "Chat - base" },
      { id: "thread-2", title: "Chat - fine-tuned" },
    ],
    async (id) => (id === "thread-1" ? "## User\n\nhi" : ""),
  );
  assert.equal(markdown, "# Chat - base\n\n## User\n\nhi");
});

test("a title carrying a line break stays on its heading", async () => {
  const markdown = await buildNamedConversationsMarkdown(
    [
      { id: "thread-1", title: "Two\nlines - base" },
      { id: "thread-2", title: "Two lines - lora" },
    ],
    async () => "## User\n\nhi",
  );
  assert.match(markdown, /^# Two lines - base\n\n## User/);
});
