// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { downloadFile } from "@/lib/native-files";

import { filterArchivedChatExport } from "./archived-chat-export";
import { buildStoredChatExport } from "./chat-history-storage";

export const buildChatExport = buildStoredChatExport;

function dateStamp(): string {
  // Date only (no colons) so the filename is valid on every OS.
  return new Date().toISOString().slice(0, 10);
}

export async function downloadChatExport(): Promise<void> {
  const data = await buildChatExport();
  await downloadFile(
    JSON.stringify(data, null, 2),
    `unsloth-chats-${dateStamp()}.json`,
    "application/json",
  );
}

// Full backup restricted to archived chats. Returns the archived thread count.
export async function buildArchivedChatExport() {
  return filterArchivedChatExport(await buildChatExport());
}

// Download only the archived chats. Returns how many were exported; skips the
// download entirely when there are none.
export async function downloadArchivedChatExport(): Promise<number> {
  const { data, archivedCount } = await buildArchivedChatExport();
  if (archivedCount === 0) {
    return 0;
  }
  await downloadFile(
    JSON.stringify(data, null, 2),
    `unsloth-archived-chats-${dateStamp()}.json`,
    "application/json",
  );
  return archivedCount;
}
