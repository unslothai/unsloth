// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `/chat?q=%s` is what a browser custom search engine opens: a new chat with the text typed in,
// never sent, in its own draft slot.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

function memoryStorage(): Storage & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    get length() {
      return data.size;
    },
    key: (index: number) => [...data.keys()][index] ?? null,
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    removeItem: (key: string) => void data.delete(key),
    clear: () => data.clear(),
  };
}
const local = memoryStorage();
const session = memoryStorage();
(globalThis as { window?: unknown }).window = { localStorage: local, sessionStorage: session };

const { queryParam } = await import("../src/features/chat/chat-search.ts");
const drafts = await import("../src/features/chat/utils/composer-draft.ts");
const { seedQueryDraft, temporaryQueryDraft, clearTemporaryQueryDraft } = await import(
  "../src/features/chat/query-draft.ts"
);

test("a blank or non-text q is no query", () => {
  assert.equal(queryParam({ q: "what is lora" }), "what is lora");
  for (const q of [undefined, "", "   ", 42, null, { a: 1 }]) {
    assert.equal(queryParam({ q }), undefined);
  }
});

test("a link's text lands in its own session slot, not the shared new-chat draft", () => {
  drafts.writeComposerDraft(drafts.composerDraftKey(drafts.NEW_CHAT_DRAFT_ID), "unsent work");
  const first = seedQueryDraft("first", false);
  const second = seedQueryDraft("second", false);
  assert.ok(drafts.isQueryDraftId(first) && first !== second);
  assert.equal(drafts.latestDraft(drafts.composerDraftKey(first)), "first");
  assert.equal(drafts.latestDraft(drafts.composerDraftKey(second)), "second");
  assert.equal(session.data.get(drafts.composerDraftKey(first)), "first");
  assert.equal(local.data.get(drafts.composerDraftKey(drafts.NEW_CHAT_DRAFT_ID)), "unsent work");
  assert.ok(![...local.data.keys()].some((key) => key.includes("query-")));
  drafts.writePasteDraft(drafts.composerPasteDraftKey(first), ["paste"]);
  assert.equal(session.data.get(drafts.composerPasteDraftKey(first)), '["paste"]');
  drafts.writeComposerDraft(drafts.composerDraftKey(first), "");
  assert.equal(session.data.has(drafts.composerDraftKey(first)), false);
});

test("a temporary chat's link text never enters browser storage", () => {
  const before = [new Map(local.data), new Map(session.data)];
  const nonce = seedQueryDraft("private", true);
  const key = drafts.composerDraftKey(nonce);
  assert.equal(temporaryQueryDraft(key), "private");
  assert.deepEqual([local.data, session.data], before);
  clearTemporaryQueryDraft(key);
  assert.equal(temporaryQueryDraft(key), "");
});

test("a LAN address, which has no crypto.randomUUID, still opens the link", () => {
  const crypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  Object.defineProperty(globalThis, "crypto", { value: {}, configurable: true });
  try {
    assert.ok(drafts.isQueryDraftId(seedQueryDraft("lan", false)));
  } finally {
    Object.defineProperty(globalThis, "crypto", crypto!);
  }
});

test("the chat route seeds the draft and drops q without sending", () => {
  const route = readText("../src/app/routes/chat.tsx");
  assert.match(route, /if \(!search\.q \|\| preload \|\| abortController\.signal\.aborted\) return;/);
  assert.match(route, /seedQueryDraft\(search\.q, store\.incognito\)/);
  assert.match(route, /search: \{ new: nonce \}, replace: true/);
  assert.doesNotMatch(route, /\.send\(|append\(|runPrompt/);
  const index = readText("../src/app/routes/index.tsx");
  assert.match(index, /to === "\/chat" \? \{ to, search: \{ q \} \} : \{ to \}/);
});
