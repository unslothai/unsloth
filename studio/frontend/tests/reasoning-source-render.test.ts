// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import test from "node:test";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// Each render cold-starts Vite with no prebundle and runs beside thousands of tests on Windows, so
// a 30 s cap failed unrelated PRs; the cap only stops a hung child.
const RENDER_TIMEOUT_MS = 120_000;

// Vite's Markdown plugin graph retains close hooks after SSR, so the real renderer runs in an
// isolated process that can terminate them once the markup is produced.
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
      timeout: RENDER_TIMEOUT_MS,
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
      timeout: RENDER_TIMEOUT_MS,
    },
  );

  assert.match(stdout, /data-streamdown="code-block"/);
  assert.match(stdout, /reasoning HTML fence/);
  assert.doesNotMatch(stdout, /Open HTML preview preview/);
});

const RENDER_BOUNDED_BOLD_TRACE = String.raw`
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
const { MarkdownTextSource } = await vite.ssrLoadModule(
  "/src/components/assistant-ui/markdown-text.tsx",
);
const { ReasoningTranscriptIndex } = await vite.ssrLoadModule(
  "/src/components/assistant-ui/reasoning-transcript-index.ts",
);
const text = "**" + "long thought ".repeat(3000).trimEnd() + "**";
const rows = new ReasoningTranscriptIndex().update([text]);
const adapter = { async *run() {} };
function Harness({ row }) {
  const runtime = useLocalRuntime(adapter);
  return createElement(
    AssistantRuntimeProvider,
    { runtime },
    createElement(MarkdownTextSource, {
      messageId: "bounded-trace",
      sourceText: row.renderText ?? row.text,
      streaming: false,
    }),
  );
}
const html = rows.map((row) => renderToStaticMarkup(createElement(Harness, { row })));
process.stdout.write(JSON.stringify(html), () => process.exit(0));
`;

test("a finished long trace keeps bold across its bounded fragments", async () => {
  const { stdout } = await execFileAsync(
    process.execPath,
    ["--input-type=module", "--eval", RENDER_BOUNDED_BOLD_TRACE],
    {
      cwd: process.cwd(),
      maxBuffer: 10_000_000,
      timeout: RENDER_TIMEOUT_MS,
    },
  );
  const fragments = JSON.parse(stdout) as string[];
  assert.ok(fragments.length > 1);
  for (const html of fragments) {
    assert.match(html, /data-streamdown="strong"/);
    assert.doesNotMatch(html, /\*\*/);
  }
});
