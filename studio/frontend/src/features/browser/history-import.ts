// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Safari's History.json and Chrome's Takeout export: JSON, times in microseconds since 1970.

import { isWebUrl } from "./address";
import { MAX_TITLE_CHARS, MAX_URL_CHARS, useBrowserHistoryStore } from "./history-store";

const MAX_FILE_BYTES = 50 * 1024 * 1024;
const EARLIEST_VISIT_MS = Date.UTC(1995, 0, 1);

export class HistoryFileError extends Error {}

function isParsableUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export function parseHistoryExport(text: string): { url: string; title: string; visitedAt: number }[] {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new HistoryFileError("not a history file");
  }
  const root = data !== null && typeof data === "object" ? (data as Record<string, unknown>) : {};
  const entries = Array.isArray(root.history) ? root.history : root["Browser History"];
  if (!Array.isArray(entries)) throw new HistoryFileError("not a history file");
  const now = Date.now();
  // Takeout lists a notice string in place of visits when the synced history is encrypted.
  return entries.flatMap((entry: unknown) => {
    if (entry === null || typeof entry !== "object") return [];
    const { url, title, time_usec: usec } = entry as Record<string, unknown>;
    if (typeof url !== "string" || url.length > MAX_URL_CHARS || !isWebUrl(url) || !isParsableUrl(url)) return [];
    if (typeof usec !== "number" || !Number.isFinite(usec)) return [];
    const visitedAt = Math.floor(usec / 1000);
    if (visitedAt < EARLIEST_VISIT_MS || visitedAt > now) return [];
    return [{ url, title: typeof title === "string" ? title.trim().slice(0, MAX_TITLE_CHARS) : "", visitedAt }];
  });
}

export async function importHistoryFile(file: File): Promise<{ added: number; leftOut: number }> {
  if (file.size > MAX_FILE_BYTES) throw new HistoryFileError("file too large");
  const items = parseHistoryExport(await file.text());
  return useBrowserHistoryStore.getState().importVisits(items);
}
