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
  clearComposerDraft,
  composerDraftKey,
  composerPasteDraftKey,
  readComposerDraft,
  readPasteDraft,
  writeComposerDraft,
  writePasteDraft,
} = await import("../src/features/chat/utils/composer-draft.ts");

test.beforeEach(() => {
  store.clear();
  failWrites = false;
});

test("an unsent paste survives a reload", () => {
  const key = composerPasteDraftKey("t1");
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
  assert.notEqual(composerDraftKey("t1"), composerPasteDraftKey("t1"));
  assert.notEqual(composerPasteDraftKey("t1"), composerPasteDraftKey("t2"));

  writeComposerDraft(composerDraftKey("t1"), "typed");
  writePasteDraft(composerPasteDraftKey("t1"), ["pasted"]);
  clearComposerDraft("t1");
  assert.equal(readComposerDraft(composerDraftKey("t1")), null);
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("t1")), []);
});

test("a paste too big for the quota does not cost the typed draft", () => {
  writeComposerDraft(composerDraftKey("t1"), "typed");
  failWrites = true;

  writePasteDraft(composerPasteDraftKey("t1"), ["x".repeat(64)]);
  assert.equal(readComposerDraft(composerDraftKey("t1")), "typed");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("t1")), []);
});

test("a corrupt paste slot reads as empty rather than throwing", () => {
  const key = composerPasteDraftKey("t1");
  for (const raw of ["not json", '{"text":"a"}', "[1,2]", ""]) {
    store.set(key, raw);
    assert.deepEqual(readPasteDraft(key), []);
  }
  // Non-string entries are dropped, the rest survive.
  store.set(key, JSON.stringify(["keep", 5, null, "also"]));
  assert.deepEqual(readPasteDraft(key), ["keep", "also"]);
});
