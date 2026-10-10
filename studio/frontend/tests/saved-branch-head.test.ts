// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type View = { remoteId?: string; messages: { id: string }[]; incognito: boolean };

// rerun the mocked effect only when dependencies change, matching React
function recorder(temporary: Set<string>) {
  const writes: [string, string][] = [];
  let view: View = { messages: [], incognito: false };
  let previousDeps: unknown[] | null = null;
  const runRecorder = loadWithStubs<{ useBranchHeadRecorder: () => void }>(
    new URL("../src/features/chat/hooks/use-branch-head-recorder.ts", import.meta.url),
    {
      react: {
        useEffect: (effect: () => void, deps: unknown[]) => {
          if (previousDeps && deps.every((dep, i) => dep === previousDeps?.[i])) return;
          previousDeps = deps;
          effect();
        },
      },
      "@assistant-ui/react": {
        useAuiState: (select: (state: unknown) => unknown) =>
          select({
            threadListItem: { remoteId: view.remoteId },
            thread: { messages: view.messages },
          }),
      },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: (select: (state: unknown) => unknown) =>
          select({ incognito: view.incognito }),
      },
      "../utils/branch-head": {
        writeBranchHead: (threadId: string, messageId: string) =>
          writes.push([threadId, messageId]),
      },
      "../utils/chat-history-storage": {
        isThreadIncognito: (threadId: string) => temporary.has(threadId),
      },
    },
  ).useBranchHeadRecorder;
  return {
    writes,
    render(next: View) {
      view = next;
      runRecorder();
    },
  };
}

const branch = (...ids: string[]) => ids.map((id) => ({ id }));

test("the recorder saves the head of the branch on screen as it moves", () => {
  const { writes, render } = recorder(new Set());
  render({ remoteId: "t", messages: [], incognito: false });
  render({ remoteId: "t", messages: branch("u1", "a1", "u2", "a2"), incognito: false });
  render({ remoteId: "t", messages: branch("u1", "a1-retry"), incognito: false });
  render({ remoteId: "t", messages: branch("u1", "a1", "u2", "a2"), incognito: false });
  assert.deepEqual(writes, [
    ["t", "a2"],
    ["t", "a1-retry"],
    ["t", "a2"],
  ]);
});

test("a temporary chat saved after a branch switch records the branch on screen", () => {
  const temporary = new Set(["t"]);
  const { writes, render } = recorder(temporary);
  render({ remoteId: "t", messages: branch("u1", "a1", "u2", "a2"), incognito: true });
  render({ remoteId: "t", messages: branch("u1", "a1-retry"), incognito: true });
  render({ remoteId: "t", messages: branch("u1", "a1", "u2", "a2"), incognito: true });
  assert.deepEqual(writes, []);
  // saving unmarks the thread and disables incognito without moving the head
  temporary.delete("t");
  render({ remoteId: "t", messages: branch("u1", "a1", "u2", "a2"), incognito: false });
  assert.deepEqual(writes, [["t", "a2"]]);
});

type DeleteModule = {
  deleteChatItems: (
    items: { type: "single"; id: string }[],
    activeId: string | undefined,
    onSelect: () => void,
  ) => Promise<void>;
};

function sidebar(deleteResult: () => Promise<string[]>) {
  const cleared: string[] = [];
  const restored: string[][] = [];
  const module = loadWithStubs<DeleteModule>(
    new URL("../src/features/chat/hooks/use-chat-sidebar-items.ts", import.meta.url),
    {
      react: { useEffect: () => {}, useMemo: () => {}, useState: () => [] },
      "../api/chat-api": {
        CHAT_HISTORY_UPDATED_EVENT: "chat-history-updated",
        notifyChatHistoryUpdated: () => {},
      },
      "../stores/chat-runtime-store": {
        useChatRuntimeStore: { getState: () => ({ setActiveThreadId: () => {} }) },
      },
      "../utils/chat-history-storage": { deleteStoredChatThreads: deleteResult },
      "../utils/branch-head": { clearBranchHead: (id: string) => cleared.push(id) },
      "../utils/composer-draft": { clearComposerDraft: () => {} },
      "../utils/offer-kept-sandbox-files": { offerToDeleteKeptSandboxes: () => {} },
      "../utils/stop-chat-thread": { stopChatThread: () => {} },
      "../utils/chat-thread-tombstones": {
        markChatThreadsDeleted: () => {},
        removeChatThreadTombstones: (ids: string[]) => restored.push(ids),
      },
      "../utils/prompt-queue-boundary": { requestPromptQueueStop: () => {} },
      "../utils/repair-legacy-chat-titles": { repairLegacyChatTitles: () => {} },
    },
  );
  return { ...module, cleared, restored };
}

test("deleting a chat clears its saved branch once the delete holds", async () => {
  const { deleteChatItems, cleared } = sidebar(async () => []);
  await deleteChatItems([{ type: "single", id: "t" }], undefined, () => {});
  assert.deepEqual(cleared, ["t"]);
});

test("a rejected delete keeps the saved branch of the chat it brings back", async () => {
  const { deleteChatItems, cleared, restored } = sidebar(async () => {
    throw new Error("delete failed");
  });
  await assert.rejects(
    deleteChatItems([{ type: "single", id: "t" }], undefined, () => {}),
    /delete failed/,
  );
  assert.deepEqual(restored, [["t"]]);
  assert.deepEqual(cleared, []);
});
