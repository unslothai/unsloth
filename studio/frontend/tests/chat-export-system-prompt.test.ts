// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import {
  codexLocalToolRoundId,
  startsNewCodexToolRound,
} from "../src/features/chat/codex-reasoning.ts";
import { stripSearchImageTokens } from "../src/features/chat/search-images/search-images.ts";
import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import type { ThreadRecord } from "../src/features/chat/types.ts";
import {
  buildConversationMarkdown,
  contentBlocksToMarkdownBlocks,
  CONVERSATION_MARKDOWN_MIME_TYPE,
  renderConversationBlocks,
} from "../src/features/chat/utils/conversation-markdown.ts";
import {
  buildNamedConversationsMarkdown,
  createConversationMarkdownBuilder,
  createConversationMarkdownExporter,
} from "../src/features/chat/utils/conversation-markdown-export.ts";
import { csvDocument, csvEscape, CSV_MIME } from "../src/features/chat/utils/csv-export.ts";
import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { orderByParentChain } from "../src/features/chat/utils/message-order.ts";
import {
  conversationJsonlBody,
  exportFormatIncludesSiblings,
  ndjsonBody,
} from "../src/features/chat/utils/ndjson.ts";
import { unwrapPastedTextContent } from "../src/features/chat/utils/pasted-text.ts";
import { readSrc } from "./helpers/kit.ts";

type Exporters = {
  buildFineTuneJsonl: (format: string) => Promise<{ lines: string[] }>;
  exportConversationRawJsonl: (threadId: string) => Promise<void>;
  exportConversationMessagesJsonl: (threadId: string) => Promise<void>;
  exportConversationShareGPT: (threadId: string) => Promise<void>;
  exportConversationCsv: (threadId: string) => Promise<void>;
  exportConversationMarkdown: (threadId: string) => Promise<void>;
  buildThreadContent: (threadId: string, format: string) => Promise<string | null>;
  saveConversationAsProjectSource: (
    threadId: string,
    projectId: string,
    title: string,
  ) => Promise<string>;
};

const DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

function slice(source: string, startMarker: string, endMarker: string): string {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start);
  assert.notEqual(start, -1, `${startMarker} must exist`);
  assert.notEqual(end, -1, `${endMarker} must exist`);
  return source.slice(start, end);
}

const THREADS: ThreadRecord[] = [
  {
    id: "support",
    title: "Support",
    modelType: "base",
    projectId: "billing",
    archived: false,
    createdAt: 1,
    settings: {
      systemPrompt: "You answer for the {{team}} team. End with Ticket closed.",
      systemVariables: '{"team":"Billing"}',
    },
  },
  {
    id: "plain",
    title: "Plain",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 2,
    settings: { systemPrompt: "  " },
  },
  {
    id: "inherits",
    title: "Inherits",
    modelType: "base",
    projectId: null,
    archived: false,
    createdAt: 3,
    settings: { temperature: 0.2 },
  },
];

const INSTALLATION_DEFAULTS: Record<string, string> = {
  systemPrompt: "Sign off as {{name}}.",
  systemVariables: '{"name":"Ada"}',
};

const PROJECTS: Record<string, { instructions: string; archived: boolean }> = {
  billing: { instructions: "Cite the refund policy.", archived: false },
  openInComposer: { instructions: "Reply in French.", archived: false },
};

const SUPPORT_INSTRUCTIONS =
  "<project_instructions>\nCite the refund policy.\n</project_instructions>\n\n" +
  "You answer for the Billing team. End with Ticket closed.";

function turns(threadId: string) {
  return [
    ["u1", null, "user", "Where is my refund?"],
    ["a1", "u1", "assistant", "It went out today. Ticket closed."],
  ].map(([id, parentId, role, text], index) => ({
    id: `${threadId}-${id}`,
    threadId,
    parentId: parentId ? `${threadId}-${parentId}` : null,
    createdAt: index + 10,
    role,
    content: [{ type: "text", text }],
  }));
}

function loadExporters(
  threadIds: string[],
  downloads: string[],
  sources: string[] = [],
) {
  const javascript = ts.transpileModule(
    [
      slice(ADAPTER, "function parseSystemVariablesMap(", "export const ThreadAutosaveHandle"),
      slice(ADAPTER, "async function resolveProjectInstructions(", "// Answered once per thread"),
      slice(ADAPTER, "export async function resolveProjectId(", "async function resolveSandboxSessionId("),
      slice(DIALOG, "function contentBlocksToText(", "/** A sidebar row as one markdown document."),
      slice(DIALOG, "async function buildThreadContent(", "function csvHeader("),
      slice(DIALOG, "// One JSONL line per conversation", "/** Download the fine-tuning JSONL"),
      "globalThis.__exporters = { buildFineTuneJsonl, exportConversationRawJsonl, exportConversationMessagesJsonl, exportConversationShareGPT, exportConversationCsv, exportConversationMarkdown, buildThreadContent, saveConversationAsProjectSource };",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const context = {
    exports: {},
    toast: { info: () => {}, success: () => {} },
    listStoredChatThreads: async () => threadIds.map((id) => ({ id })),
    listStoredChatMessages: async (id: string) => turns(id),
    getStoredChatThread: async (id: string) =>
      THREADS.find((thread) => thread.id === id),
    getStoredChatProject: async (id: string) => PROJECTS[id] ?? null,
    useChatRuntimeStore: { getState: () => ({ activeProjectId: "openInComposer" }) },
    isThreadIncognito: () => false,
    threadScopedDefault: (key: string) => INSTALLATION_DEFAULTS[key],
    composerProjectByPendingThread: new Map(),
    ...liveThreadHead,
    orderByParentChain,
    unwrapPastedTextContent,
    toolResultModelText: (result: unknown) => result,
    toolCallReplayArguments,
    codexLocalToolRoundId,
    startsNewCodexToolRound,
    contentBlocksToMarkdownBlocks,
    renderConversationBlocks,
    buildConversationMarkdown,
    buildNamedConversationsMarkdown,
    createConversationMarkdownBuilder,
    createConversationMarkdownExporter,
    CONVERSATION_MARKDOWN_MIME_TYPE,
    stripSearchImageTokens,
    conversationJsonlBody,
    exportFormatIncludesSiblings,
    ndjsonBody,
    csvDocument,
    csvEscape,
    CSV_MIME,
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
    saveMarkdownAsProjectSource: async (_projectId: string, body: string) => {
      sources.push(body);
      return true;
    },
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__exporters as Exporters;
}

test("chat training data starts with the system prompt the chat ran with", async () => {
  const exporters = loadExporters(["support", "plain"], []);

  const openai = (await exporters.buildFineTuneJsonl("openai")).lines.map(
    (line) => JSON.parse(line).messages,
  );
  assert.deepEqual(openai[0], [
    { role: "system", content: SUPPORT_INSTRUCTIONS },
    { role: "user", content: "Where is my refund?" },
    { role: "assistant", content: "It went out today. Ticket closed." },
  ]);
  assert.deepEqual(openai[1], [
    { role: "user", content: "Where is my refund?" },
    { role: "assistant", content: "It went out today. Ticket closed." },
  ]);

  const sharegpt = (await exporters.buildFineTuneJsonl("sharegpt")).lines.map(
    (line) => JSON.parse(line).conversations,
  );
  assert.deepEqual(sharegpt[0][0], { from: "system", value: SUPPORT_INSTRUCTIONS });
  assert.equal(sharegpt[1][0].from, "human");

  const alpaca = (await exporters.buildFineTuneJsonl("alpaca")).lines.map((line) =>
    JSON.parse(line),
  );
  assert.equal(alpaca[0].input, SUPPORT_INSTRUCTIONS);
  assert.equal(alpaca[1].input, "");
});

test("every chat export format starts with the chat's system prompt", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["support"], downloads);

  await exporters.exportConversationRawJsonl("support");
  assert.deepEqual(JSON.parse(downloads[0]).messages[0], {
    role: "system",
    content: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationMessagesJsonl("support");
  assert.deepEqual(JSON.parse(downloads[1].split("\n")[0]), {
    role: "system",
    content: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationShareGPT("support");
  assert.deepEqual(JSON.parse(downloads[2]).conversations[0], {
    from: "system",
    value: SUPPORT_INSTRUCTIONS,
  });

  await exporters.exportConversationCsv("support");
  assert.match(downloads[3], /^\W*role,content\r?\n"system","<project_instructions>/);

  await exporters.exportConversationMarkdown("support");
  const markdown = downloads[4];
  assert.ok(markdown.includes(`## System\n\n${SUPPORT_INSTRUCTIONS}`));
  assert.ok(markdown.indexOf("## System") < markdown.indexOf("## User"));

  const bulk = await exporters.buildThreadContent("support", "jsonl-raw");
  assert.equal(JSON.parse(bulk ?? "").messages[0].role, "system");
});

test("a chat with no system prompt exports only its own turns", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["plain"], downloads);

  await exporters.exportConversationRawJsonl("plain");
  await exporters.exportConversationShareGPT("plain");
  await exporters.exportConversationCsv("plain");
  await exporters.exportConversationMarkdown("plain");

  assert.equal(JSON.parse(downloads[0]).messages[0].role, "user");
  assert.equal(JSON.parse(downloads[1]).conversations[0].from, "human");
  assert.ok(!downloads[2].includes('"system"'));
  assert.ok(!downloads[3].includes("## System"));
  assert.ok(!downloads.join("").includes("French"));
});

test("a chat whose snapshot omits the system prompt exports the default it runs with", async () => {
  const downloads: string[] = [];
  const exporters = loadExporters(["inherits"], downloads);

  await exporters.exportConversationRawJsonl("inherits");
  assert.deepEqual(JSON.parse(downloads[0]).messages[0], {
    role: "system",
    content: "Sign off as Ada.",
  });
});

test("saving a chat to project sources leaves its system prompt out", async () => {
  const sources: string[] = [];
  const exporters = loadExporters(["support"], [], sources);

  await exporters.saveConversationAsProjectSource("support", "billing", "Support");

  assert.equal(sources.length, 1);
  assert.ok(!sources[0].includes("## System"));
  assert.ok(sources[0].includes("Where is my refund?"));
});
