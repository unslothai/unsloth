// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

let threads: unknown[] = [];
let messagesByThread = new Map<string, unknown[]>();
let batchFails = false;
let messageReadsFail = false;
let messageReadFailures = new Set<string>();

export function listStoredChatThreads(): Promise<unknown[]> {
  return Promise.resolve(threads);
}

export function listStoredChatMessages(threadId: string): Promise<unknown[]> {
  if (messageReadsFail || messageReadFailures.has(threadId)) {
    return Promise.reject(new Error("message read failed"));
  }
  return Promise.resolve(messagesByThread.get(threadId) ?? []);
}

export function batchListChatMessages(): Promise<Map<string, unknown[]>> {
  if (batchFails) {
    return Promise.reject(new Error("batch read failed"));
  }
  return Promise.resolve(new Map(messagesByThread));
}

export function configureChatSearchHistoryStub(options: {
  threads?: unknown[];
  messagesByThread?: Map<string, unknown[]>;
  batchFails?: boolean;
  messageReadsFail?: boolean;
  messageReadFailures?: Set<string>;
}): void {
  threads = options.threads ?? [];
  messagesByThread = options.messagesByThread ?? new Map();
  batchFails = options.batchFails ?? false;
  messageReadsFail = options.messageReadsFail ?? false;
  messageReadFailures = options.messageReadFailures ?? new Set();
}

export const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";
// Must match the real chat-api constant: the cross-tab listener keys off it.
export const CHAT_HISTORY_REVISION_KEY = "unsloth_chat_history_revision";
