// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const store = new Map<string, string>();
let failWrites = false;

// The draft helpers read window at call time, so a stub is enough here.
(globalThis as { window?: unknown }).window = {
  localStorage: {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      if (failWrites) throw new Error("QuotaExceededError");
      store.set(key, value);
    },
    removeItem: (key: string) => {
      store.delete(key);
    },
  },
};

const {
  adoptLegacyDraft,
  clearComposerDraft,
  composerDraftKey,
  composerPasteDraftKey,
  latestDraft,
  publishDraft,
  readComposerDraft,
  readPasteDraft,
  subscribeDraft,
  writeComposerDraft,
  writePasteDraft,
} = await import("../src/features/chat/utils/composer-draft.ts");

test.beforeEach(() => {
  store.clear();
  failWrites = false;
});

test("a per-chat draft from before the upgrade becomes the shared draft when its chat opens", () => {
  store.set("chat-draft:t1", "unsent");
  store.set("chat-draft-pastes:t1", '["pasted"]');
  adoptLegacyDraft("t1");
  assert.equal(readComposerDraft(composerDraftKey()), "unsent");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey()), ["pasted"]);
  assert.equal(store.has("chat-draft:t1"), false);
  assert.equal(store.has("chat-draft-pastes:t1"), false);
});

test("an old new-chat draft is adopted when a new chat opens", () => {
  store.set("chat-draft:__new__", "unsent");
  adoptLegacyDraft(null);
  assert.equal(readComposerDraft(composerDraftKey()), "unsent");
});

test("a newer shared draft wins over an old per-chat one", () => {
  writeComposerDraft(composerDraftKey(), "newer");
  store.set("chat-draft:t1", "older");
  adoptLegacyDraft("t1");
  assert.equal(readComposerDraft(composerDraftKey()), "newer");
  assert.equal(store.has("chat-draft:t1"), false);
});

test("a paste too big to adopt rolls back the text, so the next try still carries both", () => {
  store.set("chat-draft:t1", "unsent");
  store.set("chat-draft-pastes:t1", '["pasted"]');
  const setItem = window.localStorage.setItem;
  window.localStorage.setItem = (key: string, value: string) => {
    if (key === composerPasteDraftKey()) throw new Error("QuotaExceededError");
    setItem(key, value);
  };
  assert.equal(adoptLegacyDraft("t1"), false);
  window.localStorage.setItem = setItem;
  assert.equal(readComposerDraft(composerDraftKey()), null);
  assert.equal(adoptLegacyDraft("t1"), true);
  assert.equal(readComposerDraft(composerDraftKey()), "unsent");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey()), ["pasted"]);
});

test("composers hear each other's text, never their own", () => {
  const chat = Symbol("chat");
  const compare = Symbol("compare");
  const heard: string[] = [];
  const stop = subscribeDraft(chat, (text) => heard.push(text));
  publishDraft("from compare", compare);
  publishDraft("from chat", chat);
  stop();
  publishDraft("after stop", compare);
  assert.deepEqual(heard, ["from compare"]);
  assert.equal(latestDraft(), "after stop");
});

test("a failed adoption keeps the old draft rather than losing it", () => {
  store.set("chat-draft:t1", "unsent");
  failWrites = true;
  adoptLegacyDraft("t1");
  assert.equal(store.get("chat-draft:t1"), "unsent");
});

test("an unsent paste survives a reload", () => {
  const key = composerPasteDraftKey();
  const paste = `Deploy log\n${"line\n".repeat(500)}`;

  writePasteDraft(key, [paste]);
  assert.deepEqual(readPasteDraft(key), [paste]);

  // Several pastes keep their order, and clearing removes the slot.
  writePasteDraft(key, [paste, "second"]);
  assert.deepEqual(readPasteDraft(key), [paste, "second"]);
  writePasteDraft(key, []);
  assert.deepEqual(readPasteDraft(key), []);
});

test("the paste slot is separate from the text draft", () => {
  // Typing must never rewrite a paste that can run to megabytes, so the two
  // live under different keys.
  assert.notEqual(composerDraftKey(), composerPasteDraftKey());
});

test("deleting a chat drops its old per-chat draft but keeps the shared one", () => {
  writeComposerDraft(composerDraftKey(), "typed");
  writePasteDraft(composerPasteDraftKey(), ["pasted"]);
  store.set("chat-draft:t1", "old");
  store.set("chat-draft-pastes:t1", '["old"]');
  clearComposerDraft("t1");
  assert.equal(store.has("chat-draft:t1"), false);
  assert.equal(store.has("chat-draft-pastes:t1"), false);
  assert.equal(readComposerDraft(composerDraftKey()), "typed");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey()), ["pasted"]);
});

test("the shared draft keeps the prefix the account switch clears", () => {
  assert.ok(composerDraftKey().startsWith("chat-draft"));
  assert.ok(composerPasteDraftKey().startsWith("chat-draft"));
});

test("a paste too big for the quota does not cost the typed draft", () => {
  writeComposerDraft(composerDraftKey(), "typed");
  failWrites = true;

  writePasteDraft(composerPasteDraftKey(), ["x".repeat(64)]);
  assert.equal(readComposerDraft(composerDraftKey()), "typed");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey()), []);
});

test("a corrupt paste slot reads as empty rather than throwing", () => {
  const key = composerPasteDraftKey();
  for (const raw of ["not json", '{"text":"a"}', "[1,2]", ""]) {
    store.set(key, raw);
    assert.deepEqual(readPasteDraft(key), []);
  }
  // Non-string entries are dropped, the rest survive.
  store.set(key, JSON.stringify(["keep", 5, null, "also"]));
  assert.deepEqual(readPasteDraft(key), ["keep", "also"]);
});
