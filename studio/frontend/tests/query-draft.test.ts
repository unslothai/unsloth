// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  createQueryDraft,
  loadQueryDraft,
} from "../src/features/chat/utils/query-draft.ts";

const saved = new Map<string, string>();
Object.assign(globalThis, {
  window: {
    sessionStorage: {
      getItem: (key: string) => saved.get(key) ?? null,
      setItem: (key: string, value: string) => saved.set(key, value),
      removeItem: (key: string) => saved.delete(key),
    },
  },
});

test("query drafts are isolated from each other and survive a normal tab reload", async () => {
  const first = createQueryDraft("first", false);
  const second = createQueryDraft("second", false);
  const draft = first;
  draft.text = "edited";
  draft.pastes = ["query's own paste"];
  draft.save();
  assert.equal(second.text, "second");
  const modulePath = "../src/features/chat/utils/query-draft.ts?reload";
  const reloaded = await import(modulePath);
  assert.equal(reloaded.loadQueryDraft(first.nonce)?.text, "edited");
  assert.deepEqual(reloaded.loadQueryDraft(first.nonce)?.pastes, [
    "query's own paste",
  ]);
});

test("temporary query drafts never enter browser storage", () => {
  const before = new Map(saved);
  const draft = createQueryDraft("private", true);
  draft.text = "private edit";
  draft.save();
  assert.equal(draft.text, "private edit");
  assert.equal(loadQueryDraft(draft.nonce), null);
  assert.deepEqual(saved, before);
});

test("blocked storage keeps a query usable in memory", () => {
  const write = window.sessionStorage.setItem;
  window.sessionStorage.setItem = () => {
    throw new Error("blocked");
  };
  try {
    const draft = createQueryDraft("available", false);
    assert.equal(draft.text, "available");
  } finally {
    window.sessionStorage.setItem = write;
  }
});
