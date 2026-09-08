// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "Save to project sources" on one reply used to upload getCopyText(), which is
// the text parts of the message joined and nothing else. A reply that searched,
// ran a tool or reasoned would be saved with that work missing, and a reply that
// is only a tool call has no text part at all, so it saved nothing.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

// The module under test imports a sibling without its extension, the way vite
// resolves it.
register("./bundler-resolver.mjs", import.meta.url);
const { replySourceMarkdown } = await import(
  "../src/features/chat/utils/reply-source-markdown.ts"
);

/** An assistant reply as assistant-ui holds it: parts, not a string. */
const reply = [
  { type: "reasoning", text: "the docs pin it to 3.1" },
  {
    type: "tool-call",
    toolCallId: "call_1",
    toolName: "web_search",
    argsText: '{"query":"pin version"}',
    args: { query: "pin version" },
    result: { answer: "3.1" },
  },
  { type: "text", text: "Pin it to 3.1." },
  {
    type: "source",
    sourceType: "url",
    id: "s1",
    url: "https://example.com/changelog",
    title: "Changelog",
  },
];

test("a saved reply keeps the work behind it, not just its prose", () => {
  const markdown = replySourceMarkdown(reply);
  assert.match(markdown, /Pin it to 3\.1\./);
  assert.match(markdown, /web_search/, "the tool call is missing");
  assert.match(markdown, /pin version/, "the tool arguments are missing");
  assert.match(markdown, /3\.1/, "the tool result is missing");
  assert.match(markdown, /the docs pin it to 3\.1/, "the reasoning is missing");
  assert.match(markdown, /example\.com\/changelog/, "the citation is missing");
});

test("a reply that is only a tool call still has something to save", () => {
  const markdown = replySourceMarkdown([
    {
      type: "tool-call",
      toolCallId: "call_2",
      toolName: "generate_image",
      argsText: '{"prompt":"a red bus"}',
      args: { prompt: "a red bus" },
      result: { url: "sandbox:/bus.png" },
    },
  ]);
  assert.ok(
    markdown.trim().length > 0,
    "a tool-only reply is still reported as having no content to save",
  );
  assert.match(markdown, /generate_image/);
});

test("a tool result is normalised the way the whole-chat save normalises it", () => {
  const markdown = replySourceMarkdown(
    [
      {
        type: "tool-call",
        toolCallId: "call_3",
        toolName: "read_file",
        argsText: "{}",
        args: {},
        result: { raw: "unreadable" },
      },
    ],
    () => "the model saw this",
  );
  assert.match(markdown, /the model saw this/);
});

test("the reply action saves through this conversion", async () => {
  // thread.tsx is 6k lines of TSX that node cannot load, so this reads the
  // source, the way project-source-reply-destination.test.ts does.
  const src = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const marker = src.indexOf("Save to project sources");
  assert.ok(marker > 0, "the reply action is gone or was renamed");
  const handler = src.slice(
    src.lastIndexOf("<ActionBarMorePrimitive.Item", marker),
    marker,
  );
  assert.match(
    handler,
    /replySourceMarkdown\(\n\s*aui\.message\(\)\.getState\(\)\.content,/,
    "the reply is uploaded without rendering its non-text parts",
  );
  assert.ok(
    !/aui\.message\(\)\.getCopyText\(\)/.test(handler),
    "the reply is still saved as text parts alone",
  );
});
