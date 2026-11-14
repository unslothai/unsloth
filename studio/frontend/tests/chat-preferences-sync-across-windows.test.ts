// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

const key = "unsloth_chat_preferences";
const data = new Map<string, string>();
const listeners: ((event: { key: string | null }) => void)[] = [];
const localStorage = {
  getItem: (name: string) => data.get(name) ?? null,
  setItem: (name: string, value: string) => void data.set(name, value),
  removeItem: (name: string) => void data.delete(name),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    addEventListener: (type: string, listener: (event: { key: string | null }) => void) => {
      if (type === "storage") listeners.push(listener);
    },
  },
});

const { useChatPreferencesStore } = await import(
  "../src/features/chat/stores/chat-preferences-store.ts"
);

function stored(): Record<string, unknown> {
  return JSON.parse(data.get(key) ?? "{}").state ?? {};
}

function otherWindowWrites(change: Record<string, unknown> | null) {
  if (change === null) data.delete(key);
  else {
    const saved = JSON.parse(data.get(key) ?? "{}");
    data.set(key, JSON.stringify({ ...saved, state: { ...saved.state, ...change } }));
  }
  for (const listener of listeners) listener({ key });
}

test("chat settings chosen in another window survive a change in this one", () => {
  useChatPreferencesStore.getState().setShowReplyTimes(true);
  otherWindowWrites({
    sendShortcut: "mod-enter",
    notifyWhenReplyFinishes: true,
    confirmDeleteChats: false,
  });
  useChatPreferencesStore.getState().setFilesPanelCollapsed(true);
  assert.equal(stored().sendShortcut, "mod-enter");
  assert.equal(stored().notifyWhenReplyFinishes, true);
  assert.equal(stored().confirmDeleteChats, false);
  assert.equal(stored().filesPanelCollapsed, true);
  assert.equal(stored().showReplyTimes, true);
  assert.equal(useChatPreferencesStore.getState().sendShortcut, "mod-enter");
});

test("a setting from another window is still checked before use", () => {
  otherWindowWrites({ followUpBehavior: "steer", pastedTextMinChars: 123 });
  assert.equal(useChatPreferencesStore.getState().followUpBehavior, "steer");
  assert.equal(useChatPreferencesStore.getState().pastedTextMinChars, 4000);
});

test("chat settings reset in another window reset here too", () => {
  otherWindowWrites(null);
  assert.equal(useChatPreferencesStore.getState().sendShortcut, "enter");
  assert.equal(useChatPreferencesStore.getState().filesPanelCollapsed, false);
});
