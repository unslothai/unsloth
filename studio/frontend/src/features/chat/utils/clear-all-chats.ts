// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearBackendChats,
  countBackendChats,
  listChatThreads,
} from "../api/chat-api";
import { db } from "../db";
import {
  isChatThreadDeleted,
  markChatThreadsDeleted,
} from "./chat-thread-tombstones";

export async function countAllChats(): Promise<number> {
  const [backendCount, legacyThreads] = await Promise.all([
    countBackendChats().catch(() => 0),
    db.threads.toArray(),
  ]);
  const backendIds = new Set(
    (await listChatThreads().catch(() => [])).map((thread) => thread.id),
  );
  return (
    backendCount +
    legacyThreads.filter(
      (thread) => !backendIds.has(thread.id) && !isChatThreadDeleted(thread.id),
    ).length
  );
}

export async function clearAllChats(): Promise<void> {
  const [backendThreads, legacyThreads] = await Promise.all([
    listChatThreads().catch(() => []),
    db.threads.toArray(),
  ]);
  markChatThreadsDeleted(
    [...backendThreads, ...legacyThreads].map((thread) => thread.id),
  );
  await Promise.all([
    clearBackendChats(),
    db.transaction("rw", db.threads, db.messages, async () => {
      await db.messages.clear();
      await db.threads.clear();
    }),
  ]);
}
