// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { loadWithStubs } from "./helpers/module-stubs.ts";

// Every reader of the knowledge base list keeps its own copy: the composer's source menu,
// whose fallback drops a deleted KB from the chat, and the source chip, which shows the
// name. They stay right only if every create, rename and delete announces itself.

type RagApi = {
  createKnowledgeBase: (payload: { name: string }) => Promise<unknown>;
  updateKnowledgeBase: (kbId: string, payload: { name: string }) => Promise<unknown>;
  deleteKnowledgeBase: (kbId: string) => Promise<unknown>;
  listKnowledgeBases: () => Promise<unknown>;
  subscribeKnowledgeBasesChanged: (onChanged: () => void) => () => void;
};

function load(responses: Response[]) {
  const requests: string[] = [];
  const api = loadWithStubs<RagApi>(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: { method?: string }) => {
          requests.push(`${init?.method ?? "GET"} ${url}`);
          return responses.shift()!;
        },
      },
      "@/lib/api-base": { apiUrl: (path: string) => path },
      "@/lib/format-fastapi-error": { formatFastApiDetail: () => null },
      "@/lib/open-stream-response": {},
      "@/lib/sse-json-events": {},
      "./rag-availability": {
        noteRagAvailability() {},
        noteRagResponse() {},
      },
    },
  );
  return { api, requests };
}

globalThis.window = new EventTarget() as unknown as Window & typeof globalThis;

test("create, rename and delete announce the change; listing does not", async () => {
  const { api } = load([
    Response.json({ id: "kb-1", name: "Product docs" }),
    Response.json({ ok: true }),
    Response.json({ ok: true }),
    Response.json({ knowledgeBases: [], ragAvailable: true }),
  ]);
  let changes = 0;
  const unsubscribe = api.subscribeKnowledgeBasesChanged(() => changes++);

  await api.createKnowledgeBase({ name: "Product docs" });
  assert.equal(changes, 1);
  await api.updateKnowledgeBase("kb-1", { name: "Handbook" });
  assert.equal(changes, 2);
  await api.deleteKnowledgeBase("kb-1");
  assert.equal(changes, 3);
  await api.listKnowledgeBases();
  assert.equal(changes, 3);

  unsubscribe();
});

test("a delete that fails still announces, so readers refetch and see what is there", async () => {
  const { api } = load([Response.json({ detail: "Not found" }, { status: 404 })]);
  let changes = 0;
  const unsubscribe = api.subscribeKnowledgeBasesChanged(() => changes++);
  await assert.rejects(api.deleteKnowledgeBase("kb-gone"));
  assert.equal(changes, 1);
  unsubscribe();
});

test("an unsubscribed reader hears nothing more", async () => {
  const { api } = load([Response.json({ id: "kb-2", name: "Notes" })]);
  let changes = 0;
  api.subscribeKnowledgeBasesChanged(() => changes++)();
  await api.createKnowledgeBase({ name: "Notes" });
  assert.equal(changes, 0);
});
