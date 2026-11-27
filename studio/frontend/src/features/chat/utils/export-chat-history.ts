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

const STRING_SLICE = 1 << 24;

// Firefox's JSON.stringify throws past ~2^28 chars, so one video can overflow it; long strings go in slices.
function pushJson(parts: string[], value: unknown): void {
  const long: string[] = [];
  const mark = `__unsloth_slice_${Math.random().toString(36).slice(2)}_`;
  const text = JSON.stringify(value, (_key, v: unknown) =>
    typeof v === "string" && v.length > STRING_SLICE
      ? `${mark}${long.push(v) - 1}`
      : v,
  );
  let at = 0;
  long.forEach((s, i) => {
    const token = `"${mark}${i}"`;
    const hit = text.indexOf(token, at);
    parts.push(text.slice(at, hit), '"');
    for (let j = 0; j < s.length; j += STRING_SLICE) {
      parts.push(JSON.stringify(s.slice(j, j + STRING_SLICE)).slice(1, -1));
    }
    parts.push('"');
    at = hit + token.length;
  });
  parts.push(text.slice(at));
}

// Per message: a backup with videos can outgrow the browser's longest string.
export function chatExportBlob({ messages, ...rest }: ChatExportData): Blob {
  const parts = [
    JSON.stringify(rest, null, 2).slice(0, -2),
    ',\n  "messages": [',
  ];
  messages.forEach((message, i) => {
    parts.push(i ? ",\n    " : "\n    ");
    pushJson(parts, message);
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
