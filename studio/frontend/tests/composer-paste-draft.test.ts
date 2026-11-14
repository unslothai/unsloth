// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const store = new Map<string, string>();
let failWrites = false;
const windowListeners = new Map<string, ((event: unknown) => void)[]>();

(globalThis as { window?: unknown }).window = {
  addEventListener: (type: string, listener: (event: unknown) => void) => {
    windowListeners.set(type, [...(windowListeners.get(type) ?? []), listener]);
  },
  removeEventListener: () => {},
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
  NEW_CHAT_DRAFT_ID,
  NEW_COMPARE_DRAFT_ID,
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
  publishDraft("", Symbol("reset"));
});

test("unsaved text typed this session beats an old per-chat draft", () => {
  publishDraft("typed just now", Symbol("chat"));
  store.set("chat-draft:t1", "older");
  assert.equal(adoptLegacyDraft("t1"), false);
  assert.equal(latestDraft(), "typed just now");
  assert.equal(store.get("chat-draft:t1"), "older", "kept for a later, empty-handed open");
});

test("a per-chat draft from before the upgrade becomes the shared draft when its chat opens", () => {
  store.set("chat-draft:t1", "unsent");
  store.set("chat-draft-pastes:t1", '["pasted"]');
  adoptLegacyDraft("t1");
  assert.equal(readComposerDraft(composerDraftKey("", true)), "unsent");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("", true)), ["pasted"]);
  assert.equal(store.has("chat-draft:t1"), false);
  assert.equal(store.has("chat-draft-pastes:t1"), false);
});

test("an old new-chat draft is adopted when a new chat opens", () => {
  store.set("chat-draft:__new__", "unsent");
  adoptLegacyDraft(null);
  assert.equal(readComposerDraft(composerDraftKey("", true)), "unsent");
});

test("a newer shared draft wins over an old per-chat one", () => {
  writeComposerDraft(composerDraftKey("", true), "newer");
  store.set("chat-draft:t1", "older");
  adoptLegacyDraft("t1");
  assert.equal(readComposerDraft(composerDraftKey("", true)), "newer");
  assert.equal(store.has("chat-draft:t1"), true);
});

test("a paste too big to adopt rolls back the text, so the next try still carries both", () => {
  store.set("chat-draft:t1", "unsent");
  store.set("chat-draft-pastes:t1", '["pasted"]');
  const setItem = window.localStorage.setItem;
  window.localStorage.setItem = (key: string, value: string) => {
    if (key === composerPasteDraftKey("", true)) throw new Error("QuotaExceededError");
    setItem(key, value);
  };
  assert.equal(adoptLegacyDraft("t1"), false);
  window.localStorage.setItem = setItem;
  assert.equal(readComposerDraft(composerDraftKey("", true)), null);
  assert.equal(adoptLegacyDraft("t1"), true);
  assert.equal(readComposerDraft(composerDraftKey("", true)), "unsent");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("", true)), ["pasted"]);
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

// A fresh module stands in for a newly opened tab or a reload.
const openTab = () =>
  import(`../src/features/chat/utils/composer-draft.ts?tab=${Math.random()}`);

test("a new tab or a reload starts from the saved draft", async () => {
  writeComposerDraft(composerDraftKey("", true), "saved before the reload");
  const tab = await openTab();
  assert.equal(tab.latestDraft(), "saved before the reload");
});

test("typing or sending in another tab never rewrites this tab's draft", async () => {
  const tab = await openTab();
  assert.equal(tab.latestDraft(), "", "nothing saved when the tab opened");
  const heard: string[] = [];
  const stop = tab.subscribeDraft(Symbol("chat"), (text: string) => heard.push(text));
  // The other tab's debounced save, then its send clearing the slot; the browser
  // tells this tab about each with a storage event.
  const otherTabWrites = (text: string) => {
    writeComposerDraft(composerDraftKey("", true), text);
    for (const listener of windowListeners.get("storage") ?? []) {
      listener({ key: composerDraftKey("", true), newValue: text === "" ? null : text });
    }
  };
  otherTabWrites("typed in the other tab");
  assert.equal(tab.latestDraft(), "", "a chat switch here does not pick it up");
  otherTabWrites("");
  stop();
  assert.deepEqual(heard, []);
});

test("this tab keeps its own text after another tab saves", async () => {
  const tab = await openTab();
  tab.publishDraft("mine", Symbol("chat"));
  writeComposerDraft(composerDraftKey("", true), "theirs");
  assert.equal(tab.latestDraft(), "mine");
});

test("a new tab starts from the saved pastes, then keeps its own", async () => {
  writePasteDraft(composerPasteDraftKey("", true), ["saved paste"]);
  const tab = await openTab();
  assert.deepEqual(tab.latestPasteDraft(), ["saved paste"]);
  tab.writePasteDraft(tab.composerPasteDraftKey("", true), []);
  writePasteDraft(composerPasteDraftKey("", true), ["pasted in the other tab"]);
  assert.deepEqual(tab.latestPasteDraft(), [], "a chat switch here does not pick it up");
});

test("an adopted per-chat paste reaches the restore", async () => {
  const tab = await openTab();
  tab.writePasteDraft(tab.composerPasteDraftKey("", true), []);
  store.set("chat-draft-pastes:t1", '["old paste"]');
  assert.equal(tab.adoptLegacyDraft("t1"), true);
  assert.deepEqual(tab.latestPasteDraft(), ["old paste"]);
});

test("a failed adoption keeps the old draft rather than losing it", () => {
  store.set("chat-draft:t1", "unsent");
  failWrites = true;
  adoptLegacyDraft("t1");
  assert.equal(store.get("chat-draft:t1"), "unsent");
});

test("an unsent paste survives a reload", () => {
  const key = composerPasteDraftKey("", true);
  const paste = `Deploy log\n${"line\n".repeat(500)}`;

  writePasteDraft(key, [paste]);
  assert.deepEqual(readPasteDraft(key), [paste]);

  writePasteDraft(key, [paste, "second"]);
  assert.deepEqual(readPasteDraft(key), [paste, "second"]);
  writePasteDraft(key, []);
  assert.deepEqual(readPasteDraft(key), []);
});

test("the paste slot is separate from the text draft", () => {
  // Typing must never rewrite a paste that can run to megabytes, so the two
  // live under different keys.
  assert.notEqual(composerDraftKey("", true), composerPasteDraftKey("", true));
});

test("deleting a chat drops its old per-chat draft but keeps the shared one", () => {
  writeComposerDraft(composerDraftKey("", true), "typed");
  writePasteDraft(composerPasteDraftKey("", true), ["pasted"]);
  store.set("chat-draft:t1", "old");
  store.set("chat-draft-pastes:t1", '["old"]');
  clearComposerDraft("t1");
  assert.equal(store.has("chat-draft:t1"), false);
  assert.equal(store.has("chat-draft-pastes:t1"), false);
  assert.equal(readComposerDraft(composerDraftKey("", true)), "typed");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("", true)), ["pasted"]);
});

test("the shared draft keeps the prefix the account switch clears", () => {
  assert.ok(composerDraftKey("", true).startsWith("chat-draft"));
  assert.ok(composerPasteDraftKey("", true).startsWith("chat-draft"));
});

test("a paste too big for the quota does not cost the typed draft", () => {
  writeComposerDraft(composerDraftKey("", true), "typed");
  failWrites = true;

  writePasteDraft(composerPasteDraftKey("", true), ["x".repeat(64)]);
  assert.equal(readComposerDraft(composerDraftKey("", true)), "typed");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("", true)), []);
});

test("a corrupt paste slot reads as empty rather than throwing", () => {
  const key = composerPasteDraftKey("", true);
  for (const raw of ["not json", '{"text":"a"}', "[1,2]", ""]) {
    store.set(key, raw);
    assert.deepEqual(readPasteDraft(key), []);
  }
  store.set(key, JSON.stringify(["keep", 5, null, "also"]));
  assert.deepEqual(readPasteDraft(key), ["keep", "also"]);
});

test("draft text and pastes belong to their chat unless sharing is explicitly enabled", () => {
  const first = composerDraftKey("chat-a");
  const second = composerDraftKey("chat-b");
  writeComposerDraft(first, "unfinished A");
  writePasteDraft(composerPasteDraftKey("chat-a"), ["large paste A"]);
  assert.equal(readComposerDraft(second), null);
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("chat-b")), []);
  writeComposerDraft(second, "unfinished B");
  assert.equal(readComposerDraft(first), "unfinished A");
  clearComposerDraft("chat-b");
  assert.equal(readComposerDraft(first), "unfinished A");
  assert.deepEqual(readPasteDraft(composerPasteDraftKey("chat-a")), ["large paste A"]);
  assert.equal(composerDraftKey("chat-a", true), composerDraftKey("chat-b", true));
  assert.equal(composerPasteDraftKey("chat-a", true), composerPasteDraftKey("chat-b", true));
});

test("every unsent New chat and Compare pair shares one slot, apart from saved chats and the shared draft", () => {
  // Their ids are minted per navigation; per-id slots would never be read or removed again.
  const newChat = composerDraftKey(NEW_CHAT_DRAFT_ID);
  const newPair = composerDraftKey(NEW_COMPARE_DRAFT_ID);
  assert.equal(newChat, "chat-draft:__new__", "the pre-sharing New chat key, so old drafts still load");
  assert.notEqual(newPair, newChat);
  writeComposerDraft(newChat, "left in a New chat");
  writeComposerDraft(composerDraftKey("", true), "shared before upgrade");
  assert.equal(readComposerDraft(newChat), "left in a New chat");
  assert.equal(readComposerDraft(composerDraftKey("saved-chat")), null);
  assert.equal(readComposerDraft(newPair), null);
  assert.equal(readComposerDraft(composerDraftKey("", true)), "shared before upgrade");
});

test("shared publications cannot replace a per-chat draft", () => {
  writeComposerDraft(composerDraftKey("chat-a"), "my local draft");
  publishDraft("from Compare", Symbol("compare"));
  assert.equal(readComposerDraft(composerDraftKey("chat-a")), "my local draft");
  assert.equal(latestDraft(), "from Compare");
});

test("per-chat text and pastes stay isolated from other open tabs", async () => {
  const tab = await openTab();
  const textKey = tab.composerDraftKey("isolated-chat");
  const pasteKey = tab.composerPasteDraftKey("isolated-chat");
  assert.equal(tab.latestDraft(textKey), "");
  assert.deepEqual(tab.latestPasteDraft(pasteKey), []);
  writeComposerDraft(textKey, "other tab's draft");
  writePasteDraft(pasteKey, ["other tab's paste"]);
  assert.equal(tab.latestDraft(textKey), "");
  assert.deepEqual(tab.latestPasteDraft(pasteKey), []);
  tab.writeComposerDraft(textKey, "my draft");
  tab.writePasteDraft(pasteKey, ["my paste"]);
  writeComposerDraft(textKey, "another external edit");
  writePasteDraft(pasteKey, ["another external paste"]);
  assert.equal(tab.latestDraft(textKey), "my draft");
  assert.deepEqual(tab.latestPasteDraft(pasteKey), ["my paste"]);
  const reloaded = await openTab();
  assert.equal(reloaded.latestDraft(textKey), "another external edit");
  assert.deepEqual(reloaded.latestPasteDraft(pasteKey), ["another external paste"]);
});

test("a draft adopted into the shared slot and sent there does not return when sharing is turned off", async () => {
  const tab = await openTab();
  const textKey = tab.composerDraftKey("adopted-chat");
  const pasteKey = tab.composerPasteDraftKey("adopted-chat");
  tab.writeComposerDraft(textKey, "sent from the shared slot");
  tab.writePasteDraft(pasteKey, ["sent paste"]);
  assert.equal(tab.adoptLegacyDraft("adopted-chat"), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "sent from the shared slot");
  tab.publishDraft("", Symbol("chat"));
  tab.writeComposerDraft(tab.composerDraftKey("", true), "");
  tab.writePasteDraft(tab.composerPasteDraftKey("", true), []);
  assert.equal(tab.latestDraft(textKey), "");
  assert.deepEqual(tab.latestPasteDraft(pasteKey), []);
  const reloaded = await openTab();
  assert.equal(reloaded.latestDraft(textKey), "");
  assert.deepEqual(reloaded.latestPasteDraft(pasteKey), []);
});

test("enabling sharing adopts this window's per-chat draft, not another window's later write", async () => {
  const tab = await openTab();
  const textKey = tab.composerDraftKey("two-windows");
  const pasteKey = tab.composerPasteDraftKey("two-windows");
  tab.writeComposerDraft(textKey, "mine");
  tab.writePasteDraft(pasteKey, ["my paste"]);
  store.set(textKey, "theirs");
  store.set(pasteKey, '["their paste"]');
  tab.publishDraft("", Symbol("chat"));
  assert.equal(tab.adoptLegacyDraft("two-windows"), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "mine");
  assert.deepEqual(tab.latestPasteDraft(tab.composerPasteDraftKey("", true)), ["my paste"]);
});

test("an unsent Compare pair's draft is adopted from its own slot", async () => {
  const tab = await openTab();
  tab.writeComposerDraft(tab.composerDraftKey(tab.NEW_COMPARE_DRAFT_ID), "compare prompt");
  assert.equal(tab.adoptLegacyDraft(tab.NEW_COMPARE_DRAFT_ID), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "compare prompt");
  assert.equal(tab.latestDraft(tab.composerDraftKey(tab.NEW_COMPARE_DRAFT_ID)), "");
});

test("switching sharing on adopts the slot this tab last typed into, even with its composer unmounted", async () => {
  const tab = await openTab();
  tab.stageComposerDraft(tab.composerDraftKey("older-chat"), "older");
  tab.writeComposerDraft(tab.composerDraftKey("older-chat"), "older");
  tab.stageComposerDraft(tab.composerDraftKey(tab.NEW_COMPARE_DRAFT_ID), "typed in Compare");
  assert.equal(tab.adoptCurrentDraft(), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "typed in Compare");
  assert.equal(tab.latestDraft(tab.composerDraftKey("older-chat")), "older");
});

test("a paste-only draft is what switching sharing on adopts", async () => {
  const tab = await openTab();
  tab.writeComposerDraft(tab.composerDraftKey("typed-earlier"), "older text");
  tab.writePasteDraft(tab.composerPasteDraftKey("paste-only"), ["a long document"]);
  assert.equal(tab.adoptCurrentDraft(), true);
  assert.deepEqual(tab.latestPasteDraft(tab.composerPasteDraftKey("", true)), ["a long document"]);
  assert.equal(tab.latestDraft(tab.composerDraftKey("typed-earlier")), "older text");
});

test("deleting a comparison clears its pair-scoped draft once the delete holds, and a failed delete reopens it", async () => {
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../src/features/chat/hooks/use-chat-sidebar-items.ts", import.meta.url), "utf8");
  const del = src.slice(src.indexOf("export async function deleteChatItems("));
  assert.match(del, /items\.filter\(\(i\) => i\.type !== "single"\)\.map\(\(i\) => i\.id\)/);
  assert.match(del, /for \(const id of draftIds\) clearComposerDraft\(id\)/);
  assert.match(del, /for \(const id of draftIds\) reopenComposerDraft\(id\)/);
});

test("a pending autosave landing after its chat was deleted is dropped", async () => {
  const tab = await openTab();
  const key = tab.composerDraftKey("deleted-chat");
  tab.stageComposerDraft(key, "typed");
  tab.writeComposerDraft(key, "typed");
  tab.clearComposerDraft("deleted-chat");
  tab.writeComposerDraft(key, "late flush");
  tab.writePasteDraft(tab.composerPasteDraftKey("deleted-chat"), ["late paste"]);
  assert.equal(store.has(key), false);
  assert.equal(store.has(tab.composerPasteDraftKey("deleted-chat")), false);
  assert.equal(tab.adoptCurrentDraft(), false);
});

test("what this tab just typed replaces a shared draft another window left", async () => {
  const tab = await openTab();
  tab.stageComposerDraft(tab.composerDraftKey("mine-only"), "this window");
  store.set(composerDraftKey("", true), "other window");
  assert.equal(tab.adoptCurrentDraft(), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "this window");
  assert.equal(store.get(composerDraftKey("", true)), "this window");
});

test("a shared copy that does not fit keeps the saved per-chat draft and the old shared one", async () => {
  const tab = await openTab();
  const key = tab.composerDraftKey("quota-chat");
  tab.stageComposerDraft(key, "saved prompt");
  tab.writeComposerDraft(key, "saved prompt");
  store.set(composerDraftKey("", true), "older shared");
  failWrites = true;
  assert.equal(tab.adoptCurrentDraft(), false);
  failWrites = false;
  assert.equal(store.get(key), "saved prompt");
  assert.equal(store.get(composerDraftKey("", true)), "older shared");
});

test("a delete in flight drops late writes, and a failed delete reopens the slot with its draft", async () => {
  const tab = await openTab();
  const key = tab.composerDraftKey("restored-chat");
  tab.writeComposerDraft(key, "keep me");
  tab.closeComposerDraft("restored-chat");
  tab.writeComposerDraft(key, "late flush");
  assert.equal(store.get(key), "keep me");
  tab.reopenComposerDraft("restored-chat");
  tab.writeComposerDraft(key, "typed after restore");
  assert.equal(store.get(key), "typed after restore");
});

test("typing still inside the save debounce is what sharing adopts, and its late flush is dropped", async () => {
  const tab = await openTab();
  const key = tab.composerDraftKey("debounced-chat");
  tab.writeComposerDraft(key, "saved earlier");
  tab.stageComposerDraft(key, "saved earlier, then typed more");
  assert.equal(tab.adoptCurrentDraft(), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "saved earlier, then typed more");
  tab.flushComposerDraft(key, "saved earlier, then typed more");
  assert.equal(store.has(key), false);
});

test("an adoption keeps its own text when another window adopts right after", async () => {
  const tab = await openTab();
  tab.stageComposerDraft(tab.composerDraftKey("window-a"), "window A draft");
  tab.writePasteDraft(tab.composerPasteDraftKey("window-a"), ["A paste"]);
  assert.equal(tab.adoptCurrentDraft(), true);
  store.set(composerDraftKey("", true), "window B draft");
  store.set(composerPasteDraftKey("", true), '["B paste"]');
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "window A draft");
  assert.deepEqual(tab.latestPasteDraft(tab.composerPasteDraftKey("", true)), ["A paste"]);
});

test("a delayed save of another slot does not take adoption away from the latest edit", async () => {
  const tab = await openTab();
  tab.stageComposerDraft(tab.composerDraftKey("hidden-chat"), "chat text");
  tab.stageComposerDraft(tab.composerDraftKey(tab.NEW_COMPARE_DRAFT_ID), "compare text");
  tab.writeComposerDraft(tab.composerDraftKey("hidden-chat"), "chat text");
  assert.equal(tab.adoptCurrentDraft(), true);
  assert.equal(tab.latestDraft(tab.composerDraftKey("", true)), "compare text");
});
