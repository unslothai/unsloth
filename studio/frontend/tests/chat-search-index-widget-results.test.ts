// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { splitMcpImages } from "../src/features/chat/api/mcp-images.ts";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/chat/hooks/use-chat-search-index.ts", import.meta.url)),
  "utf8",
);

function slice(signature: string): string {
  const start = source.indexOf(signature);
  assert.ok(start >= 0, `${signature} is no longer in use-chat-search-index.ts`);
  const end = source.indexOf("\n}\n", start);
  assert.ok(end > start, `${signature} has no top-level closing brace`);
  return source.slice(start, end + 3);
}

const binaryKey = /^const BINARY_KEY = .*$/m.exec(source)?.[0];
assert.ok(binaryKey, "BINARY_KEY is no longer defined in use-chat-search-index.ts");

const searchableText = new Function(
  "splitMcpImages",
  `${
    ts.transpileModule(
      [
        binaryKey,
        slice("function mcpWidgetText("),
        slice("function searchableText("),
      ].join("\n"),
      { compilerOptions: { target: ts.ScriptTarget.ES2020 } },
    ).outputText
  }; return searchableText;`,
)(splitMcpImages) as (value: unknown, depth?: number, toolName?: string) => string;

const WIDGET_RESULT = {
  text: "San Francisco: 18C, humidity 72%.",
  ui: {
    resourceUri: "ui://weather-server/dashboard",
    content: [{ type: "text", text: "San Francisco: 18C, humidity 72%." }],
    structuredContent: { tempC: 18, station: "KSFO-INTERNAL", raw: "x".repeat(5000) },
    _meta: { cursor: "opaque-paging-token" },
  },
};

test("a widget result is indexed by what was on screen, not its seed data", () => {
  const indexed = searchableText(WIDGET_RESULT, 0, "mcp__a3f9__get_weather");
  assert.equal(indexed, "San Francisco: 18C, humidity 72%.");
  for (const hidden of ["ui://", "KSFO-INTERNAL", "opaque-paging-token"]) {
    assert.ok(!indexed.includes(hidden), `${hidden} must not be searchable`);
  }
  assert.ok(indexed.length < 200, "the 5000-char seed blob must not reach the index");
});

test("someone else's result in that shape is still indexed whole", () => {
  // Imports keep arbitrary objects under any tool name, so shape alone must not decide.
  const imported = {
    text: "Q3 summary",
    ui: { resourceUri: "ui://reports/q3" },
    owner: "finance-team",
  };
  const indexed = searchableText(imported, 0, "get_report");
  assert.ok(indexed.includes("Q3 summary"));
  assert.ok(
    indexed.includes("finance-team"),
    "the rest of an unrelated result must stay searchable",
  );
});

test("ordinary results are untouched", () => {
  assert.equal(searchableText("plain tool output", 0, "terminal"), "plain tool output");
  assert.equal(
    searchableText('shot\n__MCP_IMAGES__:[{"data":"AAAA","mimeType":"image/png"}]', 0, "mcp__a__b"),
    "shot",
  );
  assert.equal(searchableText({ text: "hi", images: ["AAAA"] }, 0, "python"), "hi");
});
