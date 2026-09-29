// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { splitMcpImages } from "../src/features/chat/api/mcp-images.ts";
import {
  extractMcpUiEnvelope,
  isMcpUiToolResult,
  mcpUiReplayImages,
  toolApprovalScope,
  toolResultParams,
} from "../src/features/chat/mcp-apps/mcp-ui.ts";
import { splitMcpToolName } from "../src/features/chat/utils/mcp-tool-name.ts";

const read = (path: string) =>
  readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
const MARKER = "\n__MCP_UI__:"; // mcp_client.py::_ui_envelope
const MCP = "mcp__srv__get_status";

test("the envelope comes off an MCP result and leaves the model text", () => {
  const images = '\n__MCP_IMAGES__:[{"data":"AAAA","mimeType":"image/png"}]';
  for (const [raw, text] of [
    [
      `cpu 12%${MARKER}{"resourceUri":"ui://a/b","structuredContent":{"cpu":12}}`,
      "cpu 12%",
    ],
    // Stops at its own line, so a trailing image envelope survives.
    [`shot${MARKER}{"resourceUri":"ui://a/b"}${images}`, `shot${images}`],
    // The last marker wins over an earlier literal mention.
    [
      `see __MCP_UI__: docs${MARKER}{"resourceUri":"ui://a/b"}`,
      "see __MCP_UI__: docs",
    ],
  ]) {
    const got = extractMcpUiEnvelope(raw, MCP);
    assert.equal(got.text, text);
    assert.equal(got.ui?.resourceUri, "ui://a/b");
  }
});

test("output that merely mentions the marker, or is not MCP, is untouched", () => {
  const cases: [string, string][] = [
    ["log\n__MCP_UI__: documented here", MCP],
    ['x\n__MCP_UI__:{"resourceUri":5}', MCP],
    ["x\n__MCP_UI__:[1,2,3]", MCP],
    ["plain output\nwith lines", MCP],
    [`docs${MARKER}{"resourceUri":"ui://a/b"}`, "terminal"],
    [`docs${MARKER}{"resourceUri":"ui://a/b"}`, ""],
  ];
  for (const [raw, tool] of cases) {
    assert.deepEqual(extractMcpUiEnvelope(raw, tool), { text: raw, ui: null });
  }
});

test("only an MCP tool's result is treated as a widget", () => {
  const widget = {
    text: "Q3",
    ui: { resourceUri: "ui://r/q3" },
    owner: "finance",
  };
  assert.ok(isMcpUiToolResult(widget, MCP));
  for (const tool of ["get_report", ""])
    assert.ok(!isMcpUiToolResult(widget, tool));
  for (const bad of [
    { text: "x", ui: {} },
    { text: "x" },
    { ui: widget.ui },
    "s",
    null,
  ]) {
    assert.ok(!isMcpUiToolResult(bad, MCP));
  }
});

test("the model and exports never see the envelope or the wrapper", () => {
  const adapter = read("features/chat/api/chat-adapter.ts");
  const modelText = adapter.slice(
    adapter.indexOf("export function toolResultModelText("),
  );
  assert.match(
    modelText.slice(0, 400),
    /isMcpUiToolResult\(result, toolName\)[\s\S]*return result\.text;/,
  );
  assert.match(
    adapter,
    /isMcpUiToolResult\(result, tc\.toolName \?\? ""\) \|\|/,
  );
  // The image guard must refuse a widget result, or the widget would be dropped.
  assert.match(
    adapter,
    /v\.sessionId === undefined &&\s*v\.ui === undefined &&/,
  );
});

test("Always allow shares the chat adapter's approval scope", () => {
  assert.match(
    read("features/chat/api/chat-adapter.ts"),
    /toolConfirmationScopeId = resolvedThreadId\s*\? `\$\{sandboxSessionId \|\| "_default"\}:\$\{resolvedThreadId\}`\s*: sandboxSessionId \|\| "_default"/,
  );
  assert.equal(toolApprovalScope("project-p", "t-1"), "project-p:t-1");
  assert.equal(toolApprovalScope(undefined, "t-1"), "_default:t-1");
  assert.equal(toolApprovalScope("", undefined), "_default");
});

test("the seed refills image blocks in order and drops one with no image", () => {
  const png = { data: "AAAA", mimeType: "image/png" };
  const ui = {
    resourceUri: "ui://a/b",
    structuredContent: { n: 1 },
    content: [
      { type: "text", text: "a" },
      { type: "image", mimeType: "image/png" },
      { type: "image", data: "inline", mimeType: "image/gif" },
      { type: "image", mimeType: "image/png" },
    ],
  };
  assert.deepEqual(toolResultParams(ui, [png]), {
    content: [
      { type: "text", text: "a" },
      { type: "image", ...png },
      { type: "image", data: "inline", mimeType: "image/gif" },
    ],
    structuredContent: { n: 1 },
  });
});

test("the tool name carries the server the widget is scoped to", () => {
  assert.deepEqual(splitMcpToolName("mcp__a3f9__get__thing"), {
    serverId: "a3f9",
    tool: "get__thing",
  });
  assert.equal(splitMcpToolName("python"), null);
});

test("the search index keeps a widget's shown text, not its seed", () => {
  const source = read("features/chat/hooks/use-chat-search-index.ts");
  const start = source.indexOf("function searchableText(");
  const body = source.slice(start, source.indexOf("\n}\n", start) + 3);
  const binaryKey = /^const BINARY_KEY = .*$/m.exec(source)?.[0] ?? "";
  const js = ts.transpileModule(`${binaryKey}\n${body}`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const searchableText = new Function(
    "splitMcpImages",
    "isMcpUiToolResult",
    `${js}; return searchableText;`,
  )(splitMcpImages, isMcpUiToolResult) as (
    value: unknown,
    depth?: number,
    toolName?: string,
  ) => string;
  const widget = {
    text: "SF: 18C",
    ui: {
      resourceUri: "ui://w/d",
      structuredContent: { station: "KSFO", raw: "x".repeat(5000) },
    },
  };
  assert.equal(searchableText(widget, 0, "mcp__a__weather"), "SF: 18C");
  assert.match(searchableText(widget, 0, "get_report"), /KSFO/);
  assert.equal(
    searchableText({ text: "hi", images: ["AAAA"] }, 0, "python"),
    "hi",
  );
});

test("a widget result replays the images it carried to the model", () => {
  const images = [{ data: "QUJD", mimeType: "image/png" }];
  const wrapped = { text: "chart", images, ui: { resourceUri: "ui://s/v" } };
  assert.deepEqual(mcpUiReplayImages(wrapped, "mcp__s__chart"), images);
  assert.deepEqual(mcpUiReplayImages(wrapped, "render_chart"), []);
  assert.deepEqual(
    mcpUiReplayImages({ ...wrapped, images: [{ data: 1 }] }, "mcp__s__chart"),
    [],
  );
  const adapter = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    adapter,
    /mcpUiReplayImages\(result, tc\.toolName \?\? ""\);\n\s*if \(uiImages\.length > 0\) content \+= mcpImagesEnvelope\(uiImages\);/,
  );
});
