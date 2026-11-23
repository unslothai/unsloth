// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";

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
import { conversationModelLabel } from "../src/features/chat/utils/conversation-markdown-export.ts";
import { readSrc } from "./helpers/kit.ts";

const DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");

const PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
const JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==";

const TURNS = [
  {
    id: "u1",
    parentId: null,
    role: "user",
    content: [
      { type: "text", text: "What is wrong in this chart?" },
      { type: "image", image: PNG },
    ],
  },
  {
    id: "a1",
    parentId: "u1",
    role: "assistant",
    content: [{ type: "text", text: "The y axis starts at 50." }],
  },
  {
    id: "u2",
    parentId: "a1",
    role: "user",
    content: [{ type: "text", text: "And this photo?" }],
    attachments: [{ id: "att", type: "image", content: [{ type: "image", image: JPEG }] }],
  },
  {
    id: "a2",
    parentId: "u2",
    role: "assistant",
    content: [
      { type: "text", text: "Here it is again:" },
      { type: "image", image: PNG },
      { type: "image", image: "https://example.com/remote.png" },
    ],
  },
].map((turn, index) => ({ ...turn, threadId: "t", createdAt: index + 1 }));

function slice(source: string, startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return source.slice(start, end);
}

async function exportedBodies() {
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
    listStoredChatMessages: async () => TURNS,
    getStoredChatThread: async () => ({ id: "t", title: "Chart", createdAt: 0, settings: {} }),
    settleThreadScopedSettingsForCopy: async () => {},
    threadScopedDefault: () => "",
    resolveChatInstructions: async () => "",
    liveThreadBranch: () => null,
    savedBranchHead: () => undefined,
    conversationExportBasename: () => "chart",
    conversationModelLabel,
    orderByParentChain,
    unwrapPastedTextContent,
    toolResultModelText: (result: unknown) => result,
    contentBlocksToMarkdownBlocks,
    renderConversationBlocks,
    buildConversationMarkdown,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    stripSearchImageTokens,
    ...conversationHtml,
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
  await exporters.exportConversationHtml("t");
  await exporters.exportConversationPdf("t");
  return { html: html[0] as string, pdf: pdf[0] as string };
}

test("HTML and PDF chat exports keep the pictures in the chat", async () => {
  const { html, pdf } = await exportedBodies();
  for (const document of [html, pdf]) {
    assert.ok(document.includes(`<img src="${PNG}"`));
    assert.ok(document.includes(`<img src="${JPEG}"`));
    assert.equal(document.split(`<img src="${PNG}"`).length - 1, 2);
    assert.ok(document.includes("What is wrong in this chart?"));
    assert.ok(document.includes("And this photo?"));
    assert.ok(document.includes("Here it is again:"));
    assert.doesNotMatch(document, /<img src="https/);
    assert.equal(document.split("[image attachment]").length - 1, 1);
  }
});
