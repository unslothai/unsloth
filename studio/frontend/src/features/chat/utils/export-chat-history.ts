// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadBlobStreaming } from "@/lib/native-files";

import {
  type ChatExportData,
  filterArchivedChatExport,
} from "./archived-chat-export";
import { buildStoredChatExport } from "./chat-history-storage";

export const buildChatExport = buildStoredChatExport;

function dateStamp(): string {
  return new Date().toISOString().slice(0, 10);
}

// Per message: a backup with videos can outgrow the browser's longest string.
function chatExportBlob({ messages, ...rest }: ChatExportData): Blob {
  const parts = [
    JSON.stringify(rest, null, 2).slice(0, -2),
    ',\n  "messages": [',
  ];
  messages.forEach((message, i) => {
    parts.push(i ? ",\n    " : "\n    ", JSON.stringify(message));
  });
  parts.push("\n  ]\n}");
  return new Blob(parts, { type: "application/json" });
}

export async function downloadChatExport(): Promise<void> {
  const data = await buildChatExport();
  await downloadBlobStreaming(
    chatExportBlob(data),
    `unsloth-chats-${dateStamp()}.json`,
  );
}

export async function buildArchivedChatExport() {
  return filterArchivedChatExport(await buildChatExport(true));
}

export async function downloadArchivedChatExport(): Promise<number> {
  const { data, archivedCount } = await buildArchivedChatExport();
  if (archivedCount === 0) {
    return 0;
  }
  await downloadBlobStreaming(
    chatExportBlob(data),
    `unsloth-archived-chats-${dateStamp()}.json`,
  );
  return archivedCount;
}
