// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";

test("exports a readable markdown transcript in conversation order", () => {
  assert.equal(
    buildConversationMarkdown([
      { role: "system", content: "Be concise." },
      { role: "user", content: "Explain `RED → GREEN`." },
      { role: "assistant", content: "1. Write a failing test.\n2. Fix it." },
    ]),
    [
      "## System",
      "",
      "Be concise.",
      "",
      "## User",
      "",
      "Explain `RED → GREEN`.",
      "",
      "## Assistant",
      "",
      "1. Write a failing test.\n2. Fix it.",
      "",
    ].join("\n"),
  );
});

test("omits empty messages without rewriting markdown content", () => {
  assert.equal(
    buildConversationMarkdown([
      { role: "user", content: "  " },
      { role: "assistant", content: "# Existing heading\n\n> quote" },
    ]),
    "## Assistant\n\n# Existing heading\n\n> quote\n",
  );
});

test("keeps an unknown role label and returns empty output for empty content", () => {
  assert.equal(
    buildConversationMarkdown([{ role: "tool", content: "result" }]),
    "## Tool\n\nresult\n",
  );
  assert.equal(
    buildConversationMarkdown([{ role: "user", content: "\n\t" }]),
    "",
  );
});

test("labels a missing role as a generic message", () => {
  assert.equal(
    buildConversationMarkdown([{ role: "", content: "orphaned content" }]),
    "## Message\n\norphaned content\n",
  );
});

test("renders a multi-line arg as code instead of an escaped json string", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "render_html",
        args: {
          code: "<!DOCTYPE html>\n<html>\n  <body>hi</body>\n</html>",
          title: "Canvas",
        },
        result: "Rendered HTML canvas: Canvas.",
      },
    ]),
    [
      "**tool call:** `render_html`",
      "",
      "**code:**",
      "",
      "```",
      "<!DOCTYPE html>",
      "<html>",
      "  <body>hi</body>",
      "</html>",
      "```",
      "",
      "**title:** Canvas",
      "",
      "**result:** Rendered HTML canvas: Canvas.",
    ].join("\n"),
  );
});

test("keeps single line markup inert without fencing prose", () => {
  assert.equal(
    renderConversationBlocks([
      {
        kind: "tool-call",
        name: "render_html",
        args: { code: "<script>alert(1)</script>" },
        result: "ok",
      },
    ]),
    [
      "**tool call:** `render_html`",
      "",
      "**code:** `<script>alert(1)</script>`",
      "",
      "**result:** ok",
    ].join("\n"),
  );
});

test("widens code spans and fences past backticks in the payload", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "tool-call", name: "run", args: { cmd: "echo ```x```" } },
    ]),
    ["**tool call:** `run`", "", "**cmd:** ```` echo ```x``` ````"].join("\n"),
  );

  const multiline = renderConversationBlocks([
    { kind: "tool-call", name: "run", args: { script: "```\nx\n```" } },
  ]);
  assert.ok(multiline.includes("````\n```\nx\n```\n````"));
});

test("keeps whitespace that markdown depends on", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "text", text: "    indented_code_block()\n" },
      { kind: "thinking", text: "  padded reasoning  " },
      { kind: "text", text: "   " },
    ]),
    [
      "    indented_code_block()\n",
      "",
      "<details>",
      "<summary>thinking</summary>",
      "",
      "  padded reasoning  ",
      "",
      "</details>",
    ].join("\n"),
  );
});

test("collapses thinking and leaves prose untouched", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "thinking", text: "weighing options" },
      { kind: "text", text: "Here is the answer." },
      { kind: "attachment", label: "[image attachment]" },
    ]),
    [
      "<details>",
      "<summary>thinking</summary>",
      "",
      "weighing options",
      "",
      "</details>",
      "",
      "Here is the answer.",
      "",
      "[image attachment]",
    ].join("\n"),
  );
});

test("omits generated image bytes while retaining useful result metadata", () => {
  const blocks = contentBlocksToMarkdownBlocks([
    {
      type: "tool-call",
      toolName: "image_generation",
      result: {
        image_b64: "very-large-base64-payload",
        image_mime: "image/png",
        size: "1024x1024",
      },
    },
  ]);

  const markdown = renderConversationBlocks(blocks);
  assert.doesNotMatch(markdown, /very-large-base64-payload/);
  assert.match(markdown, /generated image omitted/);
  assert.match(markdown, /image\/png/);
  assert.match(markdown, /1024x1024/);
});

test("preserves assistant citation sources in markdown exports", () => {
  assert.equal(
    renderConversationBlocks(
      contentBlocksToMarkdownBlocks([
        {
          type: "source",
          title: "Unsloth documentation",
          url: "https://docs.unsloth.ai/",
        },
      ]),
    ),
    "**source:** [Unsloth documentation](<https://docs.unsloth.ai/>)",
  );
});

test("does not turn unsafe citation schemes into markdown links", () => {
  assert.equal(
    renderConversationBlocks([
      { kind: "source", title: "Untrusted source", url: "javascript:alert(1)" },
    ]),
    "**source:** Untrusted source",
  );
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Injected source",
        url: "https://safe.test/\n[evil](https://evil.test)",
      },
    ]),
    "**source:** Injected source",
  );
  assert.equal(
    renderConversationBlocks([
      {
        kind: "source",
        title: "Parenthesized source",
        url: "https://en.wikipedia.org/wiki/Foo_(bar)",
      },
      {
        kind: "source",
        title: "Attempted link injection",
        url: "https://safe.test/) [evil](https://evil.test",
      },
    ]),
    [
      "**source:** [Parenthesized source](<https://en.wikipedia.org/wiki/Foo_(bar)>)",
      "",
      "**source:** [Attempted link injection](<https://safe.test/)%20[evil](https://evil.test>)",
    ].join("\n"),
  );
});
