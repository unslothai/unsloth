// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";

import * as sandboxFiles from "../src/components/assistant-ui/sandbox-files.ts";
import { stripSearchImageTokens } from "../src/features/chat/search-images/search-images.ts";
import * as conversationHtml from "../src/features/chat/utils/conversation-html.ts";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";
import {
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import { unwrapPastedTextContent } from "../src/features/chat/utils/pasted-text.ts";
import { readSrc } from "./helpers/kit.ts";

const DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");

const PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

const pythonCall = (id: string, image: string) => ({
  type: "tool-call",
  toolCallId: id,
  toolName: "python",
  args: { code: "import matplotlib.pyplot as plt" },
  result: { text: "ok", images: [image], sessionId: "t1", files: [] },
});

const TURNS = [
  {
    id: "u1",
    parentId: null,
    role: "user",
    content: [{ type: "text", text: "plot y = x^2" }],
  },
  {
    id: "a1",
    parentId: "u1",
    role: "assistant",
    content: [
      pythonCall("c1", "plot.png"),
      { type: "text", text: "Here is the chart." },
    ],
  },
  {
    id: "u2",
    parentId: "a1",
    role: "user",
    content: [{ type: "text", text: "and the old one?" }],
  },
  {
    id: "a2",
    parentId: "u2",
    role: "assistant",
    content: [pythonCall("c2", "gone.png"), { type: "text", text: "That file is gone." }],
  },
].map((turn, index) => ({ ...turn, threadId: "t1", createdAt: index + 1 }));

class FileReader {
  result: string | null = null;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readAsDataURL(blob: Blob) {
    void blob.arrayBuffer().then((bytes) => {
      this.result = `data:${blob.type};base64,${Buffer.from(bytes).toString("base64")}`;
      this.onload?.();
    });
  }
}

function slice(source: string, startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return source.slice(start, end);
}

async function exportedBodies(turns: unknown[] = TURNS) {
  const javascript = ts.transpileModule(
    [
      slice(DIALOG, "function contentBlocksToText(", "/** A sidebar row as one markdown document."),
      "globalThis.__exporters = { exportConversationHtml, exportConversationPdf };",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const html: string[] = [];
  const pdf: string[] = [];
  const context = {
    exports: {},
    toast: { info: () => {}, loading: () => 0, dismiss: () => {} },
    listStoredChatMessages: async () => turns,
    getStoredChatThread: async () => ({ id: "t1", title: "Plot", createdAt: 0, settings: {} }),
    settleThreadScopedSettingsForCopy: async () => {},
    threadScopedDefault: () => "",
    resolveChatInstructions: async () => "",
    liveThreadBranch: () => null,
    savedBranchHead: () => undefined,
    conversationExportBasename: () => "plot",
    orderByParentChain,
    unwrapPastedTextContent,
    toolResultModelText: (result: { text?: unknown }) => result?.text ?? result,
    contentBlocksToMarkdownBlocks,
    renderConversationBlocks,
    buildConversationMarkdown,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    stripSearchImageTokens,
    ...sandboxFiles,
    ...conversationHtml,
    FileReader,
    authFetch: async (url: string) =>
      url === "/api/inference/sandbox/t1/plot.png"
        ? new Response(Buffer.from(PNG_BASE64, "base64"), {
            headers: { "content-type": "image/png" },
          })
        : url === "/api/inference/sandbox/t1/photo.png"
          ? new Response(new Uint8Array(7 * 1024 * 1024), {
              headers: { "content-type": "image/png" },
            })
          : new Response("Not found", { status: 404 }),
    downloadBlob: async (body: string) => {
      html.push(body);
    },
    renderConversationPdf: async (_title: string, body: string) => {
      pdf.push(body);
      return new Uint8Array();
    },
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  const exporters = context.__exporters as {
    exportConversationHtml: (threadId: string) => Promise<void>;
    exportConversationPdf: (threadId: string) => Promise<void>;
  };
  await exporters.exportConversationHtml("t1");
  await exporters.exportConversationPdf("t1");
  return { html: html[0] as string, pdf: pdf[0] as string };
}

test("HTML and PDF chat exports keep the pictures the python tool drew", async () => {
  const { html, pdf } = await exportedBodies();
  for (const document of [html, pdf]) {
    const picture = document.indexOf(`<img src="data:image/png;base64,${PNG_BASE64}" alt="plot.png">`);
    assert.notEqual(picture, -1);
    assert.ok(picture > document.indexOf("import matplotlib.pyplot as plt"));
    assert.ok(picture < document.indexOf("Here is the chart."));
  }
});

test("a python picture that is no longer on disk is named in the export", async () => {
  const { html, pdf } = await exportedBodies();
  for (const document of [html, pdf]) {
    assert.ok(document.includes("[image: gone.png]"));
    assert.ok(document.includes("That file is gone."));
    assert.equal(document.split("<img ").length - 1, 1);
  }
});

test("a python picture too large to inline is named and the export still renders", async () => {
  const turns = [
    TURNS[0],
    {
      ...TURNS[1],
      content: [pythonCall("c1", "photo.png"), { type: "text", text: "Here is the photo." }],
    },
  ];
  const { html, pdf } = await exportedBodies(turns);
  for (const document of [html, pdf]) {
    assert.ok(document.includes("[image: photo.png]"));
    assert.ok(document.includes("Here is the photo."));
    assert.equal(document.split("<img ").length - 1, 0);
  }
});
