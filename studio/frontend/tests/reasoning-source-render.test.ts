// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import test from "node:test";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// Vite's Markdown plugin graph retains close hooks after SSR. Run the real
// renderer in an isolated process so the assertion owns its complete lifetime
// and can terminate those hooks after the markup has been produced.
const RENDER_SOURCE_WITHOUT_PART_SCOPE = String.raw`
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
} from "@assistant-ui/react";
import { createServer } from "vite";

const vite = await createServer({
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true },
  server: { middlewareMode: true, watch: null },
});
const loaded = await vite.ssrLoadModule(
  "/src/components/assistant-ui/markdown-text.tsx",
);
const MarkdownTextSource = loaded.MarkdownTextSource;
const adapter = {
  async *run() {
    yield { content: [{ type: "text", text: "unused" }] };
  },
};
function GroupScopeHarness() {
  const runtime = useLocalRuntime(adapter);
  return createElement(
    AssistantRuntimeProvider,
    { runtime },
    createElement(MarkdownTextSource, {
      messageId: "reasoning-group-message",
      sourceText: "Group-scoped **reasoning page**",
      streaming: false,
    }),
  );
}
const html = renderToStaticMarkup(createElement(GroupScopeHarness));
process.stdout.write(html, () => process.exit(0));
`;

const RENDER_SOURCE_WITH_RENDER_HTML_SIBLING = String.raw`
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
} from "@assistant-ui/react";
import { createServer } from "vite";

const virtualId = "virtual:reasoning-source-render";
const resolvedId = "\0" + virtualId;
const vite = await createServer({
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true },
  server: { middlewareMode: true, watch: null },
  plugins: [{
    name: "reasoning-source-render",
    resolveId(id) {
      return id === virtualId ? resolvedId : undefined;
    },
    load(id) {
      if (id !== resolvedId) return undefined;
      return 'export { MarkdownTextSource } from "/src/components/assistant-ui/markdown-text.tsx"; export { useChatRuntimeStore } from "/src/features/chat/index.ts";';
    },
  }],
});
const loaded = await vite.ssrLoadModule(virtualId);
Object.assign(loaded.useChatRuntimeStore.getInitialState(), {
  artifactsEnabled: true,
  collapseHtmlArtifacts: true,
  loadedIsDiffusion: false,
});
const adapter = { async *run() {} };
const sourceText = [
  "\`\`\`html",
  "<!DOCTYPE html>",
  "<html><body>reasoning HTML fence</body></html>",
  "\`\`\`",
].join(String.fromCharCode(10));
function GroupScopeHarness() {
  const runtime = useLocalRuntime(adapter);
  return createElement(
    AssistantRuntimeProvider,
    { runtime },
    createElement(loaded.MarkdownTextSource, {
      messageHasRenderableRenderHtmlTool: true,
      messageId: "reasoning-group-message",
      sourceText,
      streaming: false,
    }),
  );
}
const html = renderToStaticMarkup(createElement(GroupScopeHarness));
process.stdout.write(html, () => process.exit(0));
`;

test("source Markdown renders in an assistant root scope without a message part", async () => {
  const { stdout } = await execFileAsync(
    process.execPath,
    ["--input-type=module", "--eval", RENDER_SOURCE_WITHOUT_PART_SCOPE],
    {
      cwd: process.cwd(),
      maxBuffer: 1_000_000,
      timeout: 30_000,
    },
  );

  assert.match(stdout, /Group-scoped/);
  assert.match(
    stdout,
    /<span[^>]+data-streamdown="strong"[^>]*>reasoning page<\/span>/,
  );
});

test("source Markdown suppresses a duplicate HTML card when render_html is present", async () => {
  const { stdout } = await execFileAsync(
    process.execPath,
    ["--input-type=module", "--eval", RENDER_SOURCE_WITH_RENDER_HTML_SIBLING],
    {
      cwd: process.cwd(),
      maxBuffer: 1_000_000,
      timeout: 30_000,
    },
  );

  assert.match(stdout, /data-streamdown="code-block"/);
  assert.match(stdout, /reasoning HTML fence/);
  assert.doesNotMatch(stdout, /Open HTML preview preview/);
});
