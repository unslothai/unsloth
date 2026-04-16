// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { db } from "../db";

interface ExportedChat {
  exportedAt: string;
  version: 1;
  threadCount: number;
  threads: unknown[];
  messages: unknown[];
}

export async function buildChatExport(): Promise<ExportedChat> {
  const [threads, messages] = await Promise.all([
    db.threads.toArray(),
    db.messages.toArray(),
  ]);
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
