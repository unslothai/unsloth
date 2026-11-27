// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

const data = new Map<string, string>();
const listeners = new Map<string, ((event: { key: string | null }) => void)[]>();
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    addEventListener: (type: string, listener: (event: { key: string | null }) => void) =>
      listeners.set(type, [...(listeners.get(type) ?? []), listener]),
  },
});

const { usePinnedProjectsStore } = await import("../src/features/chat/stores/pinned-projects-store.ts");
const { useChatFavoritesStore } = await import("../src/features/library/chats/favorites-store.ts");
const { useSidebarOrganizationStore } = await import(
  "../src/features/chat/stores/sidebar-organization-store.ts"
);

function stored(key: string): Record<string, unknown> {
  return JSON.parse(data.get(key) ?? "{}").state ?? {};
}

function otherWindowWrites(key: string, change: (state: Record<string, unknown>) => Record<string, unknown>) {
  const saved = JSON.parse(data.get(key) ?? "{}");
  data.set(key, JSON.stringify({ ...saved, state: change(saved.state ?? {}) }));
  for (const listener of listeners.get("storage") ?? []) listener({ key });
}

test("projects pinned in two windows are all kept", () => {
  usePinnedProjectsStore.getState().togglePin("project-1");
  otherWindowWrites("unsloth_pinned_projects", () => ({ pinnedIds: ["project-2", "project-1"] }));
  usePinnedProjectsStore.getState().togglePin("project-3");
  assert.deepEqual(stored("unsloth_pinned_projects").pinnedIds, ["project-3", "project-2", "project-1"]);
});

test("a chat starred in another window stays starred after this window stars a project", () => {
  otherWindowWrites("unsloth_library_chat_favorites", () => ({
    chatIds: ["chat-1"],
    projectIds: [],
    sectionIds: [],
  }));
  useChatFavoritesStore.getState().setProjects(["project-1"], true);
  assert.deepEqual(stored("unsloth_library_chat_favorites").chatIds, ["chat-1"]);
  assert.deepEqual(stored("unsloth_library_chat_favorites").projectIds, ["project-1"]);
});

test("a sidebar section made in another window keeps its chats when this window changes a sort", () => {
  const key = "unsloth_sidebar_organization";
  useSidebarOrganizationStore.getState().setChatSort("updated");
  otherWindowWrites(key, (state) => ({
    ...state,
    customSections: [{ id: "work", name: "Work", sort: "manual" }],
    sectionByChatId: { "chat-1": "work" },
    sectionOrder: ["pinned", "work", "projects"],
  }));
  useSidebarOrganizationStore.getState().setPinnedSort("updated");
  assert.deepEqual(
    (stored(key).customSections as { id: string }[]).map((section) => section.id),
    ["work"],
  );
  assert.deepEqual(stored(key).sectionByChatId, { "chat-1": "work" });
  assert.equal(stored(key).pinnedSort, "updated");
  assert.equal(useSidebarOrganizationStore.getState().sectionByChatId["chat-1"], "work");
});
