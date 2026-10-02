// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { buildNamedConversationsMarkdown } from "../src/features/chat/utils/conversation-markdown-export.ts";
import {
  parseConversationMarkdownDocument,
  parseConversationMarkdownMessages,
} from "../src/features/chat/utils/conversation-markdown-import.ts";
import { buildConversationMarkdown } from "../src/features/chat/utils/conversation-markdown.ts";

test("round-trips a single exported conversation", () => {
  const messages = [
    { role: "system", content: "Be concise." },
    { role: "user", content: "Explain `RED → GREEN`." },
    { role: "assistant", content: "1. Write a failing test.\n2. Fix it." },
  ];
  assert.deepEqual(
    parseConversationMarkdownDocument(
      buildConversationMarkdown(messages),
      "chat",
    ),
    [{ title: "chat", messages }],
  );
});

test("parses real combined bulk exports with titles and separators", async () => {
  const conversations = [
    { id: "first", title: "First chat" },
    { id: "second", title: "Second chat" },
  ];
  const messages = [
    [{ role: "user", content: "Hi" }],
    [{ role: "assistant", content: "Hello" }],
  ];
  const combined = await buildNamedConversationsMarkdown(
    conversations,
    async (id) => buildConversationMarkdown(messages[id === "first" ? 0 : 1]),
  );
  assert.deepEqual(parseConversationMarkdownDocument(combined, "export"), [
    { title: "First chat", messages: messages[0] },
    { title: "Second chat", messages: messages[1] },
  ]);
});

for (const content of [
  "# Existing heading\n\n> quote",
  "Intro\n\n## Summary\n\nThe useful answer.",
  "Intro\n\n## Summary\nThe useful answer.",
  "First part\n\n---\n\nSecond part",
  "Example:\n\n```markdown\n## User\n\nexample prompt\n\n## Assistant\n\nexample reply\n```\n\nConclusion",
  "Example:\n\n~~~~markdown\n## Assistant\n\nexample\n~~~\n\n---\n\n# Example chat\n\n## User\n\nliteral\n~~~~\n\nConclusion",
  "Quoted:\n\n> ## User\n>\n> literal\n\n- Example\n\n  ## Assistant\n\n  literal",
  "Indented code:\n\n    ## User\n\n    literal\n\nConclusion",
  "First part\n\n---\n\n# Report\n\n## Summary\n\nSecond part",
  "Intro\n\n## constructor\n\nA JavaScript method.",
  "  leading spaces\n\ninternal blank lines\n\ntrailing spaces  ",
]) {
  test(`preserves message markdown: ${JSON.stringify(content)}`, async () => {
    const messages = [
      { role: "user", content: "Explain this" },
      { role: "assistant", content },
      { role: "user", content: "Thanks" },
    ];
    const exported = buildConversationMarkdown(messages);
    assert.deepEqual(parseConversationMarkdownMessages(exported), messages);
    assert.deepEqual(parseConversationMarkdownDocument(exported, "chat"), [
      { title: "chat", messages },
    ]);
    const combined = await buildNamedConversationsMarkdown(
      [
        { id: "first", title: "First" },
        { id: "second", title: "Second" },
      ],
      async () => exported,
    );
    assert.deepEqual(parseConversationMarkdownDocument(combined, "chat"), [
      { title: "First", messages },
      { title: "Second", messages },
    ]);
    assert.deepEqual(
      parseConversationMarkdownDocument(
        combined.replaceAll("\n", "\r\n"),
        "chat",
      ),
      [
        { title: "First", messages },
        { title: "Second", messages },
      ],
    );
  });
}

test("does not import an ordinary markdown document as a conversation", () => {
  assert.deepEqual(
    parseConversationMarkdownDocument("## Summary\n\nA report.", "report"),
    [],
  );
});
