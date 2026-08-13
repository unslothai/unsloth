// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The index build itself is not under test here, only the cache and hint bookkeeping around
// it, so the Dexie-backed history is a pair of empty readers.

export async function listStoredChatThreads(): Promise<unknown[]> {
  return [];
}

export async function listStoredChatMessages(): Promise<unknown[]> {
  return [];
}

export async function batchListChatMessages(): Promise<unknown[]> {
  return [];
}

export const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";
// Must match the real chat-api constant: the cross-tab listener keys off it.
export const CHAT_HISTORY_REVISION_KEY = "unsloth_chat_history_revision";
