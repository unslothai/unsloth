// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildConversationMarkdown,
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
