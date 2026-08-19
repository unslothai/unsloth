// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  mcpBareToolName,
  mcpServerIdFromToolName,
} from "../src/features/chat/utils/mcp-tool-name.ts";

// Lifted out of chat-adapter.ts rather than copied, like the other adapter
// tests: importing the module would drag in the stores and the toast layer for
// two pure functions. Both must stay in step with the backend's __MCP_UI__
// writer (studio/backend/core/inference/mcp_client.py::_ui_envelope).
const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");

// The marker the extractor closes over, lifted too so the test cannot drift
// from the string the shipped code actually looks for.
const markerLine = /^const MCP_UI_MARKER = .*$/m.exec(source)?.[0];
if (!markerLine) {
  throw new Error("MCP_UI_MARKER is no longer defined in chat-adapter.ts");
}

function lift<T>(signature: string, name: string): T {
  const start = source.indexOf(signature);
  assert.ok(start >= 0, `${name} is no longer defined in chat-adapter.ts`);
  // "\n}\n", not "\n}": a multi-line return type closes with "} {" and would
  // otherwise cut the declaration off before its body.
  const end = source.indexOf("\n}\n", start);
  assert.ok(end > start, `${name} has no top-level closing brace`);
  // Drop the `export` keyword: with it the transpile emits an ES module, which
  // `new Function` cannot evaluate.
  const declaration = source.slice(start, end + 3).replace(/^export /, "");
  return new Function(
    `${
      ts.transpileModule(`${markerLine}\n${declaration}`, {
        compilerOptions: { target: ts.ScriptTarget.ES2020 },
      }).outputText
    }; return ${name};`,
  )() as T;
}

const MARKER = "\n__MCP_UI__:";
const extractMcpUiEnvelope = lift<
  (raw: string) => {
    text: string;
    ui: { resourceUri: string; structuredContent?: unknown } | null;
  }
>("export function extractMcpUiEnvelope(", "extractMcpUiEnvelope");
const isMcpUiToolResult = lift<(val: unknown) => boolean>(
  "export function isMcpUiToolResult(",
  "isMcpUiToolResult",
);

test("pulls the envelope off and leaves the model text untouched", () => {
  const raw = `cpu 12%${MARKER}{"resourceUri":"ui://sys/dash","structuredContent":{"cpu":12}}`;
  const { text, ui } = extractMcpUiEnvelope(raw);
  assert.equal(text, "cpu 12%");
  assert.equal(ui?.resourceUri, "ui://sys/dash");
  assert.deepEqual(ui?.structuredContent, { cpu: 12 });
});

test("stops at the line end so a trailing image envelope survives", () => {
  // The backend writes __MCP_UI__ before __MCP_IMAGES__, and the image parse
  // reads to the end of the string; a UI scan that ran past its own line would
  // swallow the images with it.
  const images = '\n__MCP_IMAGES__:[{"data":"AAAA","mimeType":"image/png"}]';
  const raw = `shot${MARKER}{"resourceUri":"ui://a/b"}${images}`;
  const { text, ui } = extractMcpUiEnvelope(raw);
  assert.equal(text, `shot${images}`);
  assert.equal(ui?.resourceUri, "ui://a/b");
});

test("a tool that merely prints the marker keeps its whole output", () => {
  for (const raw of [
    "log line\n__MCP_UI__: documented here, not an envelope",
    '{"resourceUri": 5} was the shape\n__MCP_UI__:{"resourceUri":5}',
    "trailing\n__MCP_UI__:[1,2,3]",
  ]) {
    const { text, ui } = extractMcpUiEnvelope(raw);
    assert.equal(ui, null);
    assert.equal(text, raw);
  }
});

test("an earlier literal mention is not mistaken for the envelope", () => {
  const raw = `see __MCP_UI__: in the docs${MARKER}{"resourceUri":"ui://a/b"}`;
  const { text, ui } = extractMcpUiEnvelope(raw);
  assert.equal(text, "see __MCP_UI__: in the docs");
  assert.equal(ui?.resourceUri, "ui://a/b");
});

test("a result with no envelope round-trips byte for byte", () => {
  const raw = "plain output\nwith lines";
  const { text, ui } = extractMcpUiEnvelope(raw);
  assert.equal(text, raw);
  assert.equal(ui, null);
});

test("the widget guard needs both the text and a named resource", () => {
  assert.ok(isMcpUiToolResult({ text: "x", ui: { resourceUri: "ui://a/b" } }));
  assert.ok(!isMcpUiToolResult({ text: "x", ui: {} }));
  assert.ok(!isMcpUiToolResult({ text: "x" }));
  assert.ok(!isMcpUiToolResult({ ui: { resourceUri: "ui://a/b" } }));
  assert.ok(!isMcpUiToolResult("a string"));
  assert.ok(!isMcpUiToolResult(null));
});

test("the image guard refuses a widget result that also carries images", () => {
  // Both shapes have `text` and `images`; if the image guard claimed a widget
  // result the card would draw the pictures and drop the widget.
  const isMcpImageToolResult = lift<(val: unknown) => boolean>(
    "export function isMcpImageToolResult(",
    "isMcpImageToolResult",
  );
  const images = [{ data: "AAAA", mimeType: "image/png" }];
  assert.ok(isMcpImageToolResult({ text: "x", images }));
  assert.ok(
    !isMcpImageToolResult({ text: "x", images, ui: { resourceUri: "ui://a" } }),
  );
});

test("the tool name carries the server the widget is scoped to", () => {
  assert.equal(
    mcpServerIdFromToolName("mcp__a3f9c1d2e4b6f807__get_status"),
    "a3f9c1d2e4b6f807",
  );
  assert.equal(
    mcpBareToolName("mcp__a3f9c1d2e4b6f807__get_status"),
    "get_status",
  );
  // Double underscores inside the tool name stay with the tool.
  assert.equal(mcpBareToolName("mcp__srv__get__thing"), "get__thing");
  assert.equal(mcpServerIdFromToolName("python"), null);
  assert.equal(mcpBareToolName("python"), null);
});
