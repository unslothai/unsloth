// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { fromMarkdown } from "mdast-util-from-markdown";
import { ReasoningPageSelector } from "../src/components/assistant-ui/reasoning-pagination.ts";
import { openWebUIRecordToConversation } from "../src/features/chat/utils/openwebui-import.ts";

test("imported consecutive reasoning documents retain their Markdown boundaries", () => {
  const first = "Earlier analysis.\n\n".repeat(1200) + "BOUNDARY PARAGRAPH";
  const second = "---\n\nINDEPENDENT DOCUMENT";
  const conversation = openWebUIRecordToConversation(
    {
      chat: {
        messages: [
          {
            id: "assistant",
            role: "assistant",
            content: "",
            output: [
              {
                type: "reasoning",
                summary: [{ type: "summary_text", text: first }],
              },
              {
                type: "reasoning",
                summary: [{ type: "summary_text", text: second }],
              },
            ],
          },
        ],
      },
    },
    "reasoning boundaries",
  );
  assert.ok(conversation);
  const parts = conversation.messages[0].content as {
    type: string;
    text: string;
  }[];
  const documents = parts
    .filter((part) => part.type === "reasoning")
    .map((part) => part.text);
  assert.deepEqual(documents, [first, second]);

  const selector = new ReasoningPageSelector();
  const latest = selector.selectDocument(documents);
  assert.equal(latest.markdown, second);
  assert.equal(latest.documentIndex, 1);
  assert.equal(fromMarkdown(latest.markdown).children[0].type, "thematicBreak");
  const earlier = selector.selectDocument(documents, { end: latest.start });
  assert.equal(earlier.documentIndex, 0);
  assert.equal(
    fromMarkdown(earlier.markdown).children.at(-1)?.type,
    "paragraph",
  );
  assert.ok(earlier.markdown.endsWith("BOUNDARY PARAGRAPH"));
});

test("document pagination preserves all source bytes and isolates open fences", () => {
  const documents = [
    "```text\n" + "code row\n".repeat(3000),
    "# A separate thought\n",
    "",
    "More analysis.\n\n".repeat(1400),
  ];
  const selector = new ReasoningPageSelector();
  const pages = [selector.selectDocument(documents)];
  while (pages.at(-1)!.hasEarlier) {
    pages.push(
      selector.selectDocument(documents, { end: pages.at(-1)!.start }),
    );
  }
  assert.equal(
    pages
      .reverse()
      .map((page) => page.markdown)
      .join(""),
    documents.join(""),
  );
  assert.ok(pages.every((page) => page.markdown.length <= 8192));
  const heading = pages.find((page) => page.documentIndex === 1)!;
  assert.equal(heading.oversizedCode, false);
  assert.equal(fromMarkdown(heading.markdown).children[0].type, "heading");
});

test("streaming a new document resets the prior document's live page start", () => {
  const documents = ["first line\n".repeat(3000)];
  const selector = new ReasoningPageSelector();
  selector.selectDocument(documents, { streaming: true });
  documents.push("A new independent thought.");
  const page = selector.selectDocument(documents, { streaming: true });
  assert.equal(page.markdown, documents[1]);
  assert.equal(page.start, documents[0].length);
  assert.equal(page.hasNewer, false);
  assert.equal(page.hasEarlier, true);
});
