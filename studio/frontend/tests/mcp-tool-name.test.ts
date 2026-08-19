// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatMcpToolName,
  mcpServerFromProvenance,
} from "../src/features/chat/utils/mcp-tool-name.ts";

test("formats an MCP name with the stamped server name", () => {
  assert.equal(
    formatMcpToolName("mcp__a3f9c1d2e4b6f807__create_issue", "GitHub"),
    "GitHub · create_issue",
  );
});

test("falls back to the raw server id without a stamp", () => {
  assert.equal(
    formatMcpToolName("mcp__a3f9c1d2e4b6f807__create_issue"),
    "a3f9c1d2e4b6f807 · create_issue",
  );
});

test("keeps double underscores inside the tool name", () => {
  assert.equal(
    formatMcpToolName("mcp__srv__get__thing", "S"),
    "S · get__thing",
  );
});

test("returns null for non-MCP and malformed names", () => {
  assert.equal(formatMcpToolName("python"), null);
  assert.equal(formatMcpToolName("mcp__missing_separator"), null);
  assert.equal(formatMcpToolName("mcp____tool"), null);
});

test("reads mcp_server from provenance, string-only", () => {
  assert.equal(mcpServerFromProvenance({ mcp_server: "GitHub" }), "GitHub");
  assert.equal(mcpServerFromProvenance({ mcp_server: "" }), undefined);
  assert.equal(mcpServerFromProvenance({ mcp_server: 3 }), undefined);
  assert.equal(mcpServerFromProvenance(undefined), undefined);
  assert.equal(mcpServerFromProvenance("GitHub"), undefined);
});

test("old history with no stamp still renders, showing the raw id", () => {
  assert.equal(
    formatMcpToolName("mcp__a3f9c1d2e4b6f807__create_issue", undefined),
    "a3f9c1d2e4b6f807 · create_issue",
  );
});

test("a stamped name survives the server being renamed or deleted", () => {
  assert.equal(formatMcpToolName("mcp__gone__run", "Old Name"), "Old Name · run");
});

test("adversarial display names are passed through verbatim for React to escape", () => {
  for (const name of [
    "line\nbreak",
    "**bold**",
    "<script>alert(1)</script>",
    "‮gnitseT",
    "x".repeat(1000),
  ]) {
    const out = formatMcpToolName("mcp__srv__tool", name);
    assert.ok(out?.startsWith(name), `display name mangled: ${JSON.stringify(out)}`);
    assert.ok(out?.endsWith(" · tool"));
  }
});
