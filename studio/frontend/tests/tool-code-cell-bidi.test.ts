// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The code an approval card shows is the code that runs: a bidi control in it is shown as
// `\uXXXX` instead of reordering what the user reads, while Copy and Download keep it as is.
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

// Reads as `echo safe # ...` with the override; runs `rm -rf ~` hidden behind it.
const COMMAND = "echo safe \u202e# ~ fr- mr; \u2066x\u2069";

for (const streaming of [true, false]) {
  test(`bidi controls are shown escaped in the card (${streaming ? "streaming" : "settled"})`, () => {
    const html = renderToStaticMarkup(
      React.createElement(ToolCodeCell, { label: "Command", code: COMMAND, language: "bash", downloadName: "cmd.sh", streaming }),
    );
    assert.equal(/[\u202a-\u202e\u2066-\u2069]/.test(html), false, "a raw bidi control reached the card");
    assert.ok(html.includes("echo safe \\u202e# ~ fr- mr; \\u2066x\\u2069"));
  });
}

test("Copy and Download keep the command exactly as it runs", async () => {
  renderToStaticMarkup(
    React.createElement(ToolCodeCell, { label: "Command", code: COMMAND, language: "bash", downloadName: "cmd.sh" }),
  );
  clicks.get("Copy to clipboard")?.();
  clicks.get("Download")?.();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(copied, [COMMAND]);
  assert.deepEqual(downloaded, [COMMAND]);
});

test("an ordinary command is shown unchanged", () => {
  const html = renderToStaticMarkup(
    React.createElement(ToolCodeCell, { label: "Command", code: "ls -la /tmp", language: "bash", downloadName: "cmd.sh", streaming: true }),
  );
  assert.ok(html.includes("ls -la /tmp"));
});
