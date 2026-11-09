// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

async function setup(
  deletedThreadIds: string[],
  failedThreadIds: string[] = [],
  reject = false,
  createDuringClear = false,
  listFails = false,
) {
  installLocalStorageFake();
  const drafts = await import(
    `../src/features/chat/utils/composer-draft.ts?clear=${Math.random()}`
  );
  const rows = [
    { id: "single" },
    { id: "left", pairId: "pair" },
    { id: "right", pairId: "pair" },
  ];
  const seed = (id: string) => {
    drafts.stageComposerDraft(drafts.composerDraftKey(id), `text ${id}`);
    drafts.flushComposerDraft(drafts.composerDraftKey(id), `text ${id}`);
    drafts.writePasteDraft(drafts.composerPasteDraftKey(id), [`paste ${id}`]);
  };
  const { clearAllChats } = loadWithStubs<{
    clearAllChats: () => Promise<unknown>;
  }>(
    new URL("../src/features/chat/utils/clear-all-chats.ts", import.meta.url),
    {
      "../stores/bookmarked-turns-store": {
        useBookmarkedTurnsStore: { getState: () => ({ forgetThreads() {} }) },
      },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: {
          getState: () => ({
            runningByThreadId: {},
            cancelByThreadId: {},
            serverCancelByThreadId: {},
          }),
        },
      },
      "../stores/pinned-chats-store": {
        usePinnedChatsStore: {
          getState: () => ({ pinnedIds: [], setPinned() {} }),
        },
      },
      "./chat-history-clear-boundary": {
        chatHistoryClearBoundary: { advance() {} },
      },
      "./chat-history-storage": {
        listStoredChatThreads: async () => {
          if (listFails) throw new Error("backend unreachable");
          return rows;
        },
        countStoredChats: async () => rows.length,
        clearStoredChats: async () => {
          if (createDuringClear) seed("created-during-clear");
          if (reject) throw new Error("clear failed");
          return {
            backend: "cleared",
            legacy: failedThreadIds.length ? "failed" : "cleared",
            deletedThreadIds,
            failedThreadIds,
            sandboxesKept: [],
          };
        },
      },
      "./composer-draft": drafts,
      "./prompt-queue-boundary": { requestPromptQueueStop() {} },
      "./stop-chat-thread": { stopChatThread() {} },
    },
  );
  const assertDraft = (id: string, exists: boolean) => {
    const textKey = drafts.composerDraftKey(id),
      pasteKey = drafts.composerPasteDraftKey(id);
    assert.equal(drafts.latestDraft(textKey), exists ? `text ${id}` : "");
    assert.equal(
      drafts.readComposerDraft(textKey),
      exists ? `text ${id}` : null,
    );
    assert.deepEqual(
      drafts.latestPasteDraft(pasteKey),
      exists ? [`paste ${id}`] : [],
    );
    assert.deepEqual(
      drafts.readPasteDraft(pasteKey),
      exists ? [`paste ${id}`] : [],
    );
  };
  return { drafts, seed, assertDraft, clearAllChats };
}

test("Clear all chats removes deleted drafts and cannot adopt or autosave them afterward", async () => {
  const h = await setup(["single", "left", "right"]);
  h.seed(h.drafts.NEW_CHAT_DRAFT_ID);
  h.seed("pair");
  h.seed("single");
  await h.clearAllChats();
  h.assertDraft("single", false);
  h.assertDraft("pair", false);
  h.assertDraft(h.drafts.NEW_CHAT_DRAFT_ID, true);
  h.seed("single");
  h.assertDraft("single", false);
  assert.equal(h.drafts.adoptLegacyDraft("single", true), false);
  assert.equal(h.drafts.adoptCurrentDraft(), false);
});

test("bulk deletion preserves a new draft created while the clear is pending", async () => {
  const h = await setup(["single"], [], false, true);
  h.seed("single");
  await h.clearAllChats();
  h.assertDraft("single", false);
  h.assertDraft("created-during-clear", true);
  assert.equal(h.drafts.adoptCurrentDraft(), true);
  assert.equal(h.drafts.latestDraft(), "text created-during-clear");
});

test("bulk deletion preserves a failed chat and a Compare pair with a surviving side", async () => {
  const h = await setup(["left"], ["single", "right"]);
  for (const id of ["left", "single", "pair"]) h.seed(id);
  await h.clearAllChats();
  h.assertDraft("left", false);
  h.assertDraft("single", true);
  h.assertDraft("pair", true);
});

test("a rejected bulk deletion preserves drafts", async () => {
  const h = await setup([], [], true);
  h.seed("single");
  h.seed("pair");
  await assert.rejects(h.clearAllChats(), /clear failed/);
  h.assertDraft("single", true);
  h.assertDraft("pair", true);
});

test("an unreachable chat listing still clears the deleted chats' drafts", async () => {
  const h = await setup(["single"], [], false, false, true);
  h.seed("single");
  await h.clearAllChats();
  h.assertDraft("single", false);
});
