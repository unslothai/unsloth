// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { buildBackendChatExport } from "../api/chat-api";
import { db } from "../db";
import type { MessageRecord, ThreadRecord } from "../types";

interface ExportedChat {
  exportedAt: string;
  version: 1;
  threadCount: number;
  threads: unknown[];
  messages: unknown[];
}

export async function buildChatExport(): Promise<ExportedChat> {
  const [backend, legacyThreads, legacyMessages] = await Promise.all([
    buildBackendChatExport().catch(() => null),
    db.threads.toArray(),
    db.messages.toArray(),
  ]);
  const threadsById = new Map<string, unknown>();
  const messagesById = new Map<string, unknown>();
  for (const thread of legacyThreads as ThreadRecord[]) {
    threadsById.set(thread.id, thread);
  }
  for (const message of legacyMessages as MessageRecord[]) {
    messagesById.set(message.id, message);
  }
  for (const thread of backend?.threads ?? []) {
    threadsById.set(thread.id, thread);
  }
  for (const message of backend?.messages ?? []) {
    messagesById.set(message.id, message);
  }
  const threads = Array.from(threadsById.values());
  const messages = Array.from(messagesById.values());
  return {
    exportedAt: new Date().toISOString(),
    version: 1,
    threadCount: threads.length,
    threads,
    messages,
  };
}

export async function downloadChatExport(): Promise<void> {
  const data = await buildChatExport();
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `unsloth-chats-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
