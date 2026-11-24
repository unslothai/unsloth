// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The code an approval card shows is the code that runs: Python ends a line at a lone carriage
// return, so the card shows it as ␍ and starts a new row there instead of leaving the next
// statement on the row of a comment. Copy and Download keep the code as is.
import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import * as jsxRuntime from "react/jsx-runtime";

import type * as ToolCodeCellModule from "../src/components/assistant-ui/tool-code-cell.tsx";
import * as escapeBidi from "../src/lib/escape-bidi-controls.ts";
import * as markdownPlugins from "../src/lib/markdown-plugins.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const copied: string[] = [];
const downloaded: string[] = [];
const clicks = new Map<string, () => void>();
const { ToolCodeCell } = loadWithStubs<typeof ToolCodeCellModule>(
  new URL("../src/components/assistant-ui/tool-code-cell.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@/lib/copy-to-clipboard": {
      copyToClipboard: async (text: string) => {
        copied.push(text);
        return false;
      },
    },
    "@/lib/escape-bidi-controls": escapeBidi,
    "@/lib/markdown-plugins": markdownPlugins,
    "@/lib/native-files": {
      downloadFile: async (text: string) => {
        downloaded.push(text);
      },
      isDownloadCancelled: () => false,
    },
    "@/lib/tick-icon": { Tick02Icon: {} },
    "@/lib/copy-icon": { Copy01Icon: {} },
    "@/lib/toast": { toast: { error() {} } },
    "@hugeicons/core-free-icons": { Copy01Icon: {}, Download01Icon: {} },
    "@/lib/folder-icons": { Copy01Icon: {}, Download01Icon: {} },
    "@hugeicons/react": { HugeiconsIcon: () => null },
    "./icon-action-button": {
      IconActionButton: ({ label, onClick }: { label: string; onClick: () => void }) => {
        clicks.set(label, onClick);
        return React.createElement("button", { "aria-label": label });
      },
    },
    streamdown: {
      Streamdown: ({ children }: { children: string }) => React.createElement("pre", null, children),
    },
    "./shared-code-plugin": { codePlugin: {} },
  },
);

// Reads as one line ending in a comment; Python runs `send(rows)` as the next line.
const SCRIPT = "print(len(rows))  # one row per sale\rsend(rows)\n";

function render(code: string, streaming: boolean): string {
  return renderToStaticMarkup(
    React.createElement(ToolCodeCell, { label: "script", code, language: "python", downloadName: "script.py", streaming }),
  );
}

for (const streaming of [true, false]) {
  const state = streaming ? "streaming" : "settled";

  test(`a statement after a lone carriage return gets its own row (${state})`, () => {
    const html = render(SCRIPT, streaming);
    assert.equal(html.includes("\r"), false, "a lone carriage return reached the card");
    assert.ok(html.includes("print(len(rows))  # one row per sale␍\nsend(rows)\n"));
  });

  test(`every lone carriage return is a row break, also one before CRLF (${state})`, () => {
    const html = render("a = 1\rb = 2\r\r\nc = 3\r", streaming);
    assert.ok(html.includes("a = 1␍\nb = 2␍\n\r\nc = 3␍\n"));
  });

  test(`CRLF line ends are shown unchanged (${state})`, () => {
    const html = render("a = 1\r\nb = 2\r\n", streaming);
    assert.ok(html.includes("a = 1\r\nb = 2\r\n"));
    assert.equal(html.includes("␍"), false);
  });
}

test("Copy and Download keep the script exactly as it runs", async () => {
  render(SCRIPT, false);
  clicks.get("Copy to clipboard")?.();
  clicks.get("Download")?.();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(copied, [SCRIPT]);
  assert.deepEqual(downloaded, [SCRIPT]);
});

test("a bash card marks a lone carriage return without starting a row bash would not start", () => {
  const html = renderToStaticMarkup(
    React.createElement(ToolCodeCell, { label: "command", code: "echo hi\r# note", language: "bash", downloadName: "cmd.sh", streaming: true }),
  );
  assert.equal(html.includes("\r"), false);
  assert.ok(html.includes("echo hi␍# note"));
});
