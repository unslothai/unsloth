// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChatThreadDeletedError } from "@/features/chat/api/chat-api";
import { isAssistantLocalThreadId } from "@/features/chat/utils/thread-ids";

export type ThreadScopeMaterialization = {
  threadId: string | null;
  readCurrentThreadItem: () => { id: string; remoteId: string | undefined };
  isThreadDeleted: (threadId: string) => boolean;
  requireStoredThread: (threadId: string) => Promise<void>;
  initialize: () => Promise<string>;
};

export async function materializeThreadScope(
  m: ThreadScopeMaterialization,
): Promise<string> {
  if (!m.threadId) {
    return m.initialize();
  }
  try {
    await m.requireStoredThread(m.threadId);
    return m.threadId;
  } catch (error) {
    if (error instanceof ChatThreadDeletedError) {
      throw error;
    }
    const state = m.readCurrentThreadItem();
    if (
      m.isThreadDeleted(m.threadId) ||
      state.id !== m.threadId ||
      state.remoteId ||
      !isAssistantLocalThreadId(m.threadId)
    ) {
      throw error;
    }
    return m.initialize();
  }
}
