// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { createConversationMarkdownExporter } from "../src/features/chat/utils/conversation-markdown-export.ts";

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
