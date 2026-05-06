// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const deletedThreadIds = new Set<string>();

export function markChatThreadDeleted(threadId: string): void {
  deletedThreadIds.add(threadId);
}

export function isChatThreadDeleted(threadId: string): boolean {
  return deletedThreadIds.has(threadId);
}
