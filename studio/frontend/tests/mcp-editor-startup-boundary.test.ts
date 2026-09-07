// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
const source = (name: string) => readFileSync(new URL("../src/features/chat/" + name, import.meta.url), "utf8");

test("MCP editor is not an eager composer dependency", () => {
  assert.ok(source("mcp-composer-button.tsx").includes('from "./chat-mcp-servers-dialog-mount"'));
  const wrapper = source("chat-mcp-servers-dialog-mount.tsx");
  assert.ok(wrapper.includes('import type { ChatMcpServersDialogProps }'));
  assert.ok(wrapper.includes('const McpEditor = lazy(() => import("./chat-mcp-servers-dialog")'));
  assert.ok(wrapper.indexOf("const McpEditor = lazy") < wrapper.indexOf("export function ChatMcpServersDialog"));
});

test("closed MCP editor is cold before first use and retained after activation", () => {
  const wrapper = source("chat-mcp-servers-dialog-mount.tsx");
  assert.ok(wrapper.includes("if (!activated && !props.open) return null"));
  assert.ok(wrapper.includes("<McpEditor {...props} />"));
  assert.ok(!wrapper.includes("props.open ? <McpEditor"));
  assert.ok(source("mcp-composer-button.tsx").includes('useShortcut("openMcpServers"'));
});

test("loading and failed MCP editor both remain dismissible", () => {
  const wrapper = source("chat-mcp-servers-dialog-mount.tsx");
  assert.ok(wrapper.includes("<LazyImportBoundary"));
  assert.ok(wrapper.includes("<Suspense"));
  assert.ok(wrapper.includes("<DialogTitle>MCP Servers</DialogTitle>"));
  assert.ok(wrapper.includes('role="status"'));
  assert.ok(wrapper.includes('dismissLabel="Cancel" onDismiss={() => onOpenChange(false)}'));
});
