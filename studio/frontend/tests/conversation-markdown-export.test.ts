// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildNamedConversationsMarkdown,
  conversationExportBasename,
  createConversationMarkdownBuilder,
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
    exportBasename: async (threadId) => `named-${threadId}`,
    notifyNoContent: () => notifications.push("empty"),
  });
}

test("downloads a Markdown conversation under the thread's name", async () => {
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
      content: "<!-- unsloth-chat-v1:[14] -->\n\n## User\n\nHello\n",
      filename: "named-thread-1.md",
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

// Search results render their images from tokens the answer text carries. Both
// ways out of a thread go through the one builder, so neither can start
// shipping them as prose.
test("renderer tokens never leave a thread, downloaded or copied", async () => {
  const answer = "Golden Retriever\n\n[[img:0123456789ab]]\n\nDone.";
  const stripped = "Golden Retriever\n\nDone.";
  const messages: TestMessage[] = [{ role: "assistant", text: answer }];

  const downloads: Array<{
    content: string;
    filename: string;
    mimeType: string;
  }> = [];
  await exporterFor(messages, downloads, [])("thread-1");
  assert.equal(downloads.length, 1);
  assert.ok(!downloads[0].content.includes("[[img:"));
  assert.ok(downloads[0].content.includes(stripped));

  const build = createConversationMarkdownBuilder<TestMessage>({
    loadMessages: async () => messages,
    renderMessage: (message) => message.text,
  });
  const copied = await buildNamedConversationsMarkdown(
    [{ id: "thread-1", title: "Dogs" }],
    build,
  );
  assert.ok(!copied.includes("[[img:"));
  assert.ok(copied.includes(stripped));
});

const EXPORT_DAY = new Date(2026, 8, 1, 23, 30);

test("an export is named after its model, title and local day", () => {
  assert.equal(
    conversationExportBasename(
      { modelId: "unsloth/Gemma-4-26B-A4B", title: "Biology of Stag Beetles Explained" },
      EXPORT_DAY,
    ),
    "Gemma-4-26B-A4B - Biology of Stag Beetles Explained (2026-09-01)",
  );
});

test("an export drops whichever of model and title is missing", () => {
  assert.equal(
    conversationExportBasename({ title: "Untitled run" }, EXPORT_DAY),
    "Untitled run (2026-09-01)",
  );
  assert.equal(
    conversationExportBasename({ modelId: "qwen3", title: "  " }, EXPORT_DAY),
    "qwen3 (2026-09-01)",
  );
  assert.equal(
    conversationExportBasename(null, EXPORT_DAY),
    "conversation (2026-09-01)",
  );
});

test("an export name never carries a path separator or a trailing dot", () => {
  assert.equal(
    conversationExportBasename(
      { modelId: "org/model", title: 'a/b\\c: "d"?\nnext...' },
      EXPORT_DAY,
    ),
    "model - a_b_c_ _d__ next (2026-09-01)",
  );
});

test("an export name is cut by code point and drops bidi controls", () => {
  const name = conversationExportBasename(
    { title: `${"a".repeat(119)}\u{1F600}tail \u202Egpj.exe` },
    EXPORT_DAY,
  );
  assert.equal(name, `${"a".repeat(119)}\u{1F600} (2026-09-01)`);
  assert.equal(
    conversationExportBasename({ title: "x\u202Egpj.exe" }, EXPORT_DAY),
    "x_gpj.exe (2026-09-01)",
  );
});
