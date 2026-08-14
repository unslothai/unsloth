// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The transitive weight of use-chat-sidebar-items: the chat API, Dexie-backed storage and
// the zustand stores. Only the calls archiveAllChatItems makes are modelled; everything
// else is an inert stub so the real batching and notification bookkeeping runs.

export interface StubThread {
  id: string;
  archived?: boolean;
  pairId?: string;
  title?: string;
  updatedAt?: number;
}

interface Recorder {
  notifications: number;
  patched: string[];
  threads: StubThread[];
  failOn: Set<string>;
}

export const recorder: Recorder = {
  notifications: 0,
  patched: [],
  threads: [],
  failOn: new Set(),
};

export function resetRecorder(threads: StubThread[], failOn: string[] = []) {
  recorder.notifications = 0;
  recorder.patched = [];
  recorder.threads = threads;
  recorder.failOn = new Set(failOn);
}

export const CHAT_HISTORY_UPDATED_EVENT = "unsloth:chat-history-updated";

export function notifyChatHistoryUpdated(): void {
  recorder.notifications += 1;
}

export async function listStoredChatThreads(): Promise<StubThread[]> {
  return recorder.threads;
}

export async function updateStoredChatThread(
  threadId: string,
  _patch: unknown,
  options: { notify?: boolean } = {},
): Promise<StubThread> {
  if (recorder.failOn.has(threadId)) {
    throw new Error(`PATCH failed for ${threadId}`);
  }
  recorder.patched.push(threadId);
  if (options.notify !== false) notifyChatHistoryUpdated();
  return { id: threadId, archived: true };
}

export async function listStoredChatThreadsWithMessages(): Promise<StubThread[]> {
  return [];
}
export async function deleteStoredChatThreads(): Promise<void> {}
export function isExpectedBackgroundChatStorageError(): boolean {
  return false;
}
export function clearComposerDraft(): void {}
export async function offerToDeleteKeptSandboxes(): Promise<void> {}
export async function stopChatThread(): Promise<void> {}
export function markChatThreadsDeleted(): void {}
export function removeChatThreadTombstones(): void {}
export function requestPromptQueueStop(): void {}
export async function repairLegacyChatTitles(): Promise<void> {}

const inertStore = {
  getState: () => ({
    setActiveThreadId: () => {},
    cancelIfRunning: () => {},
  }),
};
export const useChatRuntimeStore = inertStore;
export const useChatArtifactsStore = inertStore;
