// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `/chat?q=...` is what a browser's custom search engine opens (e.g. `http://localhost:8888/chat?q=%s`).
// It starts a new chat with the query in the composer and stops there: any page can link to the
// local Studio, so the query is never sent on its own.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const store = new Map<string, string>();
(globalThis as { window?: unknown }).window = {
  localStorage: {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  },
};

const { chatQueryParam } = await import(
  "../src/features/chat/utils/chat-query-param.ts"
);
const {
  composerDraftKey,
  latestDraft,
  readComposerDraft,
  seedComposerDraft,
  subscribeDraft,
} = await import("../src/features/chat/utils/composer-draft.ts");

const page = readText("../src/features/chat/chat-page.tsx");
const root = readText("../src/app/routes/__root.tsx");
const index = readText("../src/app/routes/index.tsx");

test("a query is read back as the text that was typed", () => {
  assert.equal(chatQueryParam("what is lora"), "what is lora");
  // The router parses search values as JSON, so these arrive typed.
  assert.equal(chatQueryParam(42), "42");
  assert.equal(chatQueryParam(true), "true");
});

test("a blank or non-text query is no query", () => {
  assert.equal(chatQueryParam(undefined), undefined);
  assert.equal(chatQueryParam(""), undefined);
  assert.equal(chatQueryParam("   "), undefined);
  assert.equal(chatQueryParam(Number.NaN), undefined);
  assert.equal(chatQueryParam({ a: 1 }), undefined);
  assert.equal(chatQueryParam(["a"]), undefined);
});

test("a seeded draft reaches the composer on screen and the next one to mount", () => {
  const composer = Symbol("composer");
  const seen: string[] = [];
  const unsubscribe = subscribeDraft(composer, (text) => seen.push(text));
  seedComposerDraft("explain qlora");
  unsubscribe();
  assert.deepEqual(seen, ["explain qlora"]);
  assert.equal(readComposerDraft(composerDraftKey()), "explain qlora");
  assert.equal(latestDraft(), "explain qlora");
});

test("the chat route accepts q", () => {
  assert.match(page, /q: chatQueryParam\(search\.q\),/);
  assert.match(root, /const rawQuery = chatQueryParam\(rawSearch\?\.q\);/);
  assert.match(root, /q: rawQuery,/);
});

test("a query opens a new chat with the query in the composer, unsent", () => {
  const effect = page.slice(
    page.indexOf("// `/chat?q=...`"),
    page.indexOf("const [modelSelectorOpen, setModelSelectorOpen]"),
  );
  assert.match(effect, /if \(!\(active && search\.q\)\) return;/);
  assert.match(effect, /seedComposerDraft\(query\);/);
  // A fresh chat, and the query leaves the URL so a reload does not seed it again.
  assert.match(
    effect,
    /search: \{ new: crypto\.randomUUID\(\) \},\s*replace: true,/,
  );
  // Never sent from here.
  assert.doesNotMatch(effect, /\.send\(|append\(|runPrompt/);
});

test("the root URL carries the query on to the chat", () => {
  assert.match(
    index,
    /if \(to === "\/chat" && q\) \{\s*throw redirect\(\{ to, search: \{ q \} \}\);/,
  );
});
