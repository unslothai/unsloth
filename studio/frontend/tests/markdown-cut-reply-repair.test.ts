// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import test from "node:test";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const RENDER_TIMEOUT_MS = 120_000;

const RENDER_MESSAGES = String.raw`
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  AssistantRuntimeProvider,
  MessagePrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { createServer } from "vite";

const vite = await createServer({
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true },
  server: { middlewareMode: true, watch: null },
});
const { MarkdownText } = await vite.ssrLoadModule(
  "/src/components/assistant-ui/markdown-text.tsx",
);
const Assistant = () =>
  createElement(MessagePrimitive.Parts, {
    components: { Text: MarkdownText, Reasoning: MarkdownText },
  });
function Harness({ message }) {
  const runtime = useExternalStoreRuntime({
    messages: [message],
    convertMessage: (m) => m,
    isRunning: false,
    onNew: async () => {},
  });
  return createElement(
    AssistantRuntimeProvider,
    { runtime },
    createElement(ThreadPrimitive.Messages, {
      components: { AssistantMessage: Assistant },
    }),
  );
}
const messages = JSON.parse(process.env.MESSAGES);
const html = messages.map((message) =>
  renderToStaticMarkup(createElement(Harness, { message })),
);
process.stdout.write(JSON.stringify(html), () => process.exit(0));
`;

type Part = { type: string; [key: string]: unknown };

const COMPLETE = { type: "complete", reason: "stop" };
const SOURCE: Part = {
  type: "source",
  sourceType: "url",
  id: "s1",
  url: "https://example.com",
  title: "Example",
};

function message(content: Part[], status: object, incomplete?: string): object {
  return {
    id: "m1",
    role: "assistant",
    content,
    status,
    metadata: incomplete
      ? { custom: { incomplete: { reason: incomplete } } }
      : {},
  };
}

async function render(messages: object[]): Promise<string[]> {
  const { stdout } = await execFileAsync(
    process.execPath,
    ["--input-type=module", "--eval", RENDER_MESSAGES],
    {
      cwd: process.cwd(),
      env: { ...process.env, MESSAGES: JSON.stringify(messages) },
      maxBuffer: 10_000_000,
      timeout: RENDER_TIMEOUT_MS,
    },
  );
  return JSON.parse(stdout) as string[];
}

test("finished replies render as written; cut replies are still repaired", async () => {
  const text = (value: string): Part => ({ type: "text", text: value });
  const reasoning = (value: string): Part => ({
    type: "reasoning",
    text: value,
  });
  const [
    finished,
    finishedBackTick,
    finishedThinking,
    lengthLive,
    lengthReloaded,
    lengthWithSource,
    stopped,
    stoppedThinking,
    finishedComparison,
    finishedRanges,
  ] = await render([
    message(
      [text("Fields marked with an asterisk (*) are required.")],
      COMPLETE,
    ),
    message([text("Press the ` key to open the console.")], COMPLETE),
    message([reasoning("Match *.pyc files."), text("Done.")], COMPLETE),
    message([text("Then **bol")], COMPLETE, "length"),
    message(
      [text("Then **bol")],
      { type: "incomplete", reason: "length" },
      "length",
    ),
    message([text("Then **bol"), SOURCE], COMPLETE, "length"),
    message(
      [text("call `foo(")],
      { type: "incomplete", reason: "cancelled" },
      "cancelled",
    ),
    message(
      [reasoning("so **maybe")],
      { type: "incomplete", reason: "cancelled" },
      "cancelled",
    ),
    message([text("- >= 16 GB of RAM")], COMPLETE),
    message([text("It takes 5~10 minutes and uses 20~30 GB.")], COMPLETE),
  ]);

  assert.match(finished, /asterisk \(\*\) are required\./);
  assert.doesNotMatch(finished, /<em/);
  assert.match(finishedBackTick, /Press the ` key to open the console\./);
  assert.doesNotMatch(finishedBackTick, /inline-code/);
  assert.match(finishedThinking, /Match \*\.pyc files\./);
  assert.doesNotMatch(finishedThinking, /<em/);

  for (const cut of [lengthLive, lengthReloaded, lengthWithSource]) {
    assert.match(cut, /data-streamdown="strong"[^>]*>bol</);
  }
  assert.match(stopped, /data-streamdown="inline-code"/);
  assert.match(stoppedThinking, /data-streamdown="strong"[^>]*>maybe</);
  assert.match(finishedComparison, /&gt;= 16 GB of RAM/);
  assert.doesNotMatch(finishedComparison, /blockquote/);
  assert.match(finishedRanges, /5~10 minutes and uses 20~30 GB\./);
  assert.doesNotMatch(finishedRanges, /<del|strikethrough/);
});
