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
const page = readText("../src/features/chat/chat-page.tsx");
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

test("the chat route consumes a query before the composer mounts, without sending", () => {
  const route = readText("../src/app/routes/chat.tsx");
  assert.match(page, /q: chatQueryParam\(search\.q\),/);
  assert.match(
    route,
    /if \(preload \|\| abortController.signal.aborted\) return;/,
  );
  assert.match(route, /createQueryDraft\(search\.q, store\.incognito\)/);
  assert.match(route, /search: \{ new: nonce \}, replace: true/);
  assert.doesNotMatch(route, /\.send\(|append\(|runPrompt/);
});

test("the root URL carries the query on to the chat", () => {
  assert.match(
    index,
    /if \(to === "\/chat" && q\) \{\s*throw redirect\(\{ to, search: \{ q \} \}\);/,
  );
});
