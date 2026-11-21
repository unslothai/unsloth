// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";

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
  conversationExportBasename,
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
import {
  EMPTY_AGENTS_MD,
  composeChatInstructions,
} from "../src/features/chat/utils/agents-md.ts";
import { formatApiErrorBody } from "../src/lib/format-fastapi-error.ts";
import { readSrc } from "./helpers/kit.ts";

type Exporters = {
  resolveChatInstructions: (
    threadId: string,
    systemPrompt: string,
    systemVariables: string,
    readThreadRecord?: () => Promise<ThreadRecord | undefined>,
    opts?: { agentsMd?: boolean },
  ) => Promise<string>;
  buildFineTuneJsonl: (format: string) => Promise<{ lines: string[] }>;
  exportConversationRawJsonl: (threadId: string) => Promise<void>;
  exportConversationMessagesJsonl: (threadId: string) => Promise<void>;
  exportConversationShareGPT: (threadId: string) => Promise<void>;
  exportConversationCsv: (threadId: string) => Promise<void>;
  exportConversationMarkdown: (threadId: string) => Promise<void>;
  buildConversationMarkdownForThread: (threadId: string) => Promise<string | null>;
  buildThreadContent: (threadId: string, format: string) => Promise<string | null>;
  saveConversationAsProjectSource: (
    threadId: string,
    projectId: string,
    title: string,
  ) => Promise<string>;
};

const DIALOG = readSrc("features/chat/prompt-storage/prompt-storage-dialog.tsx");
const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const CHAT_API = readSrc("features/chat/api/chat-api.ts");
const STORAGE = readSrc("features/chat/utils/chat-history-storage.ts");

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
  overrides: Record<string, unknown> = {},
) {
  const javascript = ts.transpileModule(
    [
      slice(CHAT_API, "function parseErrorText(", "export async function listModels("),
      slice(CHAT_API, "export async function getChatProject(", "export async function getChatAgentsMd("),
      slice(STORAGE, "export async function getStoredChatProject(", "export async function createStoredChatProject("),
      slice(ADAPTER, "function parseSystemVariablesMap(", "export const ThreadAutosaveHandle"),
      slice(ADAPTER, "async function resolveProjectInstructions(", "// Answered once per thread"),
      slice(ADAPTER, "export async function resolveProjectId(", "async function resolveSandboxSessionId("),
      slice(DIALOG, "function contentBlocksToText(", "/** A sidebar row as one markdown document."),
      slice(DIALOG, "async function buildThreadContent(", "function csvHeader("),
      slice(DIALOG, "// One JSONL line per conversation", "/** Download the fine-tuning JSONL"),
      "globalThis.__exporters = { resolveChatInstructions, buildFineTuneJsonl, exportConversationRawJsonl, exportConversationMessagesJsonl, exportConversationShareGPT, exportConversationCsv, exportConversationMarkdown, buildConversationMarkdownForThread, buildThreadContent, saveConversationAsProjectSource };",
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
    formatApiErrorBody,
    authFetch: async (url: string) => {
      const path = new URL(url, "http://localhost").pathname;
      const id = decodeURIComponent(path.slice("/api/chat/projects/".length));
      const project = PROJECTS[id];
      return new Response(JSON.stringify(project ?? { detail: "Project not found" }), {
        status: project ? 200 : 404,
      });
    },
    composeChatInstructions,
    agentsMdEditCount: () => 0,
    getChatAgentsMd: async () => EMPTY_AGENTS_MD,
    EMPTY_AGENTS_MD,
    useChatRuntimeStore: { getState: () => ({ activeProjectId: "openInComposer" }) },
    isThreadIncognito: () => false,
    settleThreadScopedSettingsForCopy: async () => {},
    threadScopedDefault: (key: string) => INSTALLATION_DEFAULTS[key],
    composerProjectByPendingThread: new Map(),
    ...liveThreadHead,
    savedBranchHead: () => undefined,
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
    conversationExportBasename,
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
    ...overrides,
  } as Record<string, unknown>;
  vm.runInNewContext(javascript, context);
  return context.__exporters as Exporters;
}

test("an unavailable project workspace stops a model-bound instruction read with the server's recovery message", async () => {
  const detail = "The project folder /Volumes/Projects/Billing is unavailable. Reconnect its drive or restore the folder, then try again.";
  const requests: string[] = [];
  const exporters = loadExporters(["support"], [], [], {
    authFetch: async (url: string) => {
      requests.push(url);
      return new Response(JSON.stringify({ detail }), { status: 503 });
    },
  });

  await assert.rejects(
    exporters.resolveChatInstructions(
      "support", "Keep this system prompt.", "", undefined, { agentsMd: true },
    ),
    { status: 503, message: detail },
  );
  assert.deepEqual(requests, ["/api/chat/projects/billing"]);
});

test("model-bound instructions keep project instructions and AGENTS.md when the workspace is available", async () => {
  const requests: string[] = [];
  const exporters = loadExporters(["support"], [], [], {
    authFetch: async (url: string) => {
      requests.push(url);
      return new Response(JSON.stringify(PROJECTS.billing));
    },
    getChatAgentsMd: async () => ({ text: "Use local tools." }),
  });

  assert.equal(
    await exporters.resolveChatInstructions(
      "support", "Keep this system prompt.", "", undefined, { agentsMd: true },
    ),
    "<agents_md>\nUse local tools.\n</agents_md>\n\n" +
      "<project_instructions>\nCite the refund policy.\n</project_instructions>\n\n" +
      "Keep this system prompt.",
  );
  assert.deepEqual(requests, ["/api/chat/projects/billing"]);
});

test("exports and copy keep saved project instructions when its workspace is unavailable", async () => {
  const downloads: string[] = [];
  const requests: string[] = [];
  const exporters = loadExporters(["support", "plain"], downloads, [], {
    authFetch: async (url: string) => {
      requests.push(url);
      const metadataOnly =
        new URL(url, "http://localhost").searchParams.get("require_workspace") === "false";
      const body = metadataOnly ? PROJECTS.billing : {
        detail: "The project folder is unavailable. Reconnect its drive or restore the folder, then try again.",
      };
      return new Response(JSON.stringify(body), { status: metadataOnly ? 200 : 503 });
    },
  });

  const training = (await exporters.buildFineTuneJsonl("openai")).lines.map(
    (line) => JSON.parse(line),
  );
  assert.equal(training[0].messages[0].content, SUPPORT_INSTRUCTIONS);
  assert.equal(training[1].messages[0].role, "user");

  const bulk = await exporters.buildThreadContent("support", "jsonl-raw");
  assert.equal(JSON.parse(bulk ?? "").messages[0].content, SUPPORT_INSTRUCTIONS);

  await exporters.exportConversationRawJsonl("support");
  await exporters.exportConversationMessagesJsonl("support");
  await exporters.exportConversationShareGPT("support");
  await exporters.exportConversationCsv("support");
  await exporters.exportConversationMarkdown("support");
  assert.equal(JSON.parse(downloads[0]).messages[0].content, SUPPORT_INSTRUCTIONS);
  assert.equal(JSON.parse(downloads[1].split("\n")[0]).content, SUPPORT_INSTRUCTIONS);
  assert.equal(JSON.parse(downloads[2]).conversations[0].value, SUPPORT_INSTRUCTIONS);
  assert.ok(downloads[3].includes(SUPPORT_INSTRUCTIONS));
  assert.ok(downloads[4].includes(`## System\n\n${SUPPORT_INSTRUCTIONS}`));

  // The copy chord uses this same Markdown builder without downloading a file.
  const copied = await exporters.buildConversationMarkdownForThread("support");
  assert.equal(copied, downloads[4]);
  assert.equal(
    await exporters.resolveChatInstructions(
      "support", "", "", undefined, { agentsMd: false },
    ),
    "<project_instructions>\nCite the refund policy.\n</project_instructions>",
  );
  assert.ok(requests.length > 0);
  assert.ok(requests.every(
    (url) => url === "/api/chat/projects/billing?require_workspace=false",
  ));
});

for (const failure of ["missing project", "other HTTP error", "network error"] as const) {
  test(`project instruction lookup preserves the existing fallback for ${failure}`, async () => {
    const exporters = loadExporters(["support"], [], [], {
      authFetch: async () => {
        if (failure === "network error") throw new TypeError("Failed to fetch");
        return new Response(JSON.stringify({ detail: "Unavailable" }), {
          status: failure === "missing project" ? 404 : 500,
        });
      },
    });

    for (const agentsMd of [false, true]) {
      assert.equal(
        await exporters.resolveChatInstructions(
          "support", "Keep this system prompt.", "", undefined, { agentsMd },
        ),
        "Keep this system prompt.",
      );
    }
  });
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

test("an export waits for the chat's settings edit still being saved", async () => {
  const downloads: string[] = [];
  let saved = false;
  const exporters = loadExporters(["plain"], downloads, [], {
    settleThreadScopedSettingsForCopy: async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      saved = true;
      return true;
    },
    getStoredChatThread: async (id: string) => ({
      ...THREADS.find((thread) => thread.id === id),
      settings: { systemPrompt: saved ? "Edited prompt." : "" },
    }),
  });

  await exporters.exportConversationRawJsonl("plain");
  assert.deepEqual(JSON.parse(downloads[0]).messages[0], {
    role: "system",
    content: "Edited prompt.",
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
