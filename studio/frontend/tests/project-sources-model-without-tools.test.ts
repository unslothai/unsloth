// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as React from "react";
import * as jsxRuntime from "react/jsx-runtime";
import { renderToStaticMarkup } from "react-dom/server";

import type * as BarModule from "../src/features/rag/components/thread-documents-bar.tsx";
import type * as GateModule from "../src/features/chat/hooks/use-rag-tool-disabled.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const PROJECT_DOC = { id: "doc-1", filename: "handbook.pdf", status: "completed" };

let runtime: Record<string, unknown> = {};
const useChatRuntimeStore = (select: (s: Record<string, unknown>) => unknown) =>
  select(runtime);

function selectorStore(state: Record<string, unknown>) {
  return (select: (s: Record<string, unknown>) => unknown) => select(state);
}

const gate = loadWithStubs<typeof GateModule>(
  new URL("../src/features/chat/hooks/use-rag-tool-disabled.ts", import.meta.url),
  {
    "../external-providers": {
      parseExternalModelId: () => null,
      providerModelSupportsStudioTools: () => null,
    },
    "../stores/external-providers-store": {
      useExternalProvidersStore: selectorStore({ providers: [] }),
    },
    "../stores/chat-runtime-store": { useChatRuntimeStore },
  },
);

const Nothing = () => null;
const Passthrough = ({ children }: { children?: React.ReactNode }) =>
  React.createElement(React.Fragment, null, children);

const { ThreadDocumentsBar } = loadWithStubs<typeof BarModule>(
  new URL("../src/features/rag/components/thread-documents-bar.tsx", import.meta.url),
  {
    react: React,
    "react/jsx-runtime": jsxRuntime,
    "@hugeicons/react": { HugeiconsIcon: Nothing },
    "@hugeicons/core-free-icons": {},
    "@/lib/tick-icon": {},
    "@/lib/chevron-icons": {},
    "@assistant-ui/react": { useAui: () => ({}) },
    "@/lib/utils": {
      cn: (...classes: unknown[]) => classes.filter(Boolean).join(" "),
    },
    "@/features/chat/stores/chat-runtime-store": {
      PENDING_CHAT_ATTACHMENT_KEY: "pending",
      readPendingAttachmentTargetClaim: () => null,
      useChatRuntimeStore,
    },
    "@/features/chat/hooks/use-rag-tool-disabled": gate,
    "@/features/chat": {
      isThreadIncognito: () => false,
      chatHistoryClearBoundary: { capture: () => 0 },
    },
    "@/features/native-intents": {
      useNativeAttachmentTargetKey: () => null,
      useNativeIntentStore: selectorStore({ pendingAttachments: {} }),
    },
    "@/lib/toast": { toast: () => undefined },
    "@/components/ui/dropdown-menu": new Proxy({}, { get: () => Passthrough }),
    "../api/rag-api": {
      listKnowledgeBases: async () => [],
      subscribeKnowledgeBasesChanged: () => () => undefined,
      listProjectDocuments: async () => [PROJECT_DOC],
      listThreadDocuments: async () => [],
    },
    "../api/rag-availability": {
      useRagAvailabilityStore: selectorStore({ isUnavailable: () => false }),
    },
    "../types/rag": { RAG_UPLOAD_ACCEPT: "", isLinkedFolderManaged: () => false },
    "@/components/ui/alert-dialog": new Proxy({}, { get: () => Passthrough }),
    "./document-status-chip": {
      DocumentStatusChip: ({ filename }: { filename: string }) =>
        React.createElement("span", null, filename),
    },
    "./knowledge-base-dialog": { KnowledgeBaseDialog: Nothing },
    "./staged-source": { EXPIRY_GRACE_MS: 0 },
    "./use-rag-documents": {
      uploadItemFromIntent: () => null,
      useRagDocuments: (scope: { type: string } | null) => ({
        documents: scope?.type === "project" ? [PROJECT_DOC] : [],
        uploading: false,
        hasIndexing: false,
        loading: false,
        upload: async () => undefined,
        remove: async () => undefined,
      }),
    },
  },
);

function renderProjectChat(
  model: { checkpoint: string; supportsTools: boolean },
  ragEnabled = false,
) {
  runtime = {
    ragEnabled,
    ragSource: { type: "thread" },
    activeProjectId: "project-1",
    projectAttachmentTarget: "project",
    projectAttachmentTargetByThread: {},
    params: { checkpoint: model.checkpoint },
    modelLoading: false,
    supportsTools: model.supportsTools,
  };
  return renderToStaticMarkup(
    React.createElement(
      Passthrough,
      null,
      React.createElement(ThreadDocumentsBar, { threadId: null }),
    ),
  );
}

test("a model without tool calling does not claim to use the project's sources", () => {
  const html = renderProjectChat({
    checkpoint: "unsloth/gemma-3-4b-it-GGUF",
    supportsTools: false,
  });
  assert.match(html, /handbook\.pdf/);
  assert.doesNotMatch(
    html,
    /This chat retrieves from its project/,
    "the chat sends no search_knowledge_base or rag_scope without tool support",
  );
  assert.match(html, /not used/);
});

test("a model with tool calling still shows the project's sources as in effect", () => {
  const html = renderProjectChat({
    checkpoint: "unsloth/Qwen3-4B-GGUF",
    supportsTools: true,
  });
  assert.match(html, /handbook\.pdf/);
  assert.match(html, /This chat retrieves from its project/);
  assert.doesNotMatch(html, /not used/);
});

test("with Docs on, a model without tool calling dims the files it will not search", () => {
  const html = renderProjectChat(
    { checkpoint: "unsloth/gemma-3-4b-it-GGUF", supportsTools: false },
    true,
  );
  assert.match(html, /handbook\.pdf/);
  assert.match(html, /these files aren&#x27;t used/);
  assert.match(html, /opacity-50/);
});

test("with Docs on, a model with tool calling keeps the files in effect", () => {
  const html = renderProjectChat(
    { checkpoint: "unsloth/Qwen3-4B-GGUF", supportsTools: true },
    true,
  );
  assert.match(html, /handbook\.pdf/);
  assert.doesNotMatch(html, /aren&#x27;t used/);
  assert.doesNotMatch(html, /opacity-50/);
});
