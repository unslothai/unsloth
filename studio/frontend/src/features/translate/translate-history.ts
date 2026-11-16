// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TranslateHistoryEntry = {
  id: string;
  /** Epoch milliseconds. */
  at: number;
  source: string;
  output: string;
  target: string;
  /** Label of the translation model; null when the chat model translated. */
  model: string | null;
};

export const TRANSLATE_HISTORY_MAX_ENTRIES = 100;
/** A longer translation is not kept: the list lives in localStorage, a few megabytes in all. */
export const TRANSLATE_HISTORY_MAX_CHARS = 20_000;

export function parseHistory(value: unknown): TranslateHistoryEntry[] {
  if (!Array.isArray(value)) return [];
  const entries: TranslateHistoryEntry[] = [];
  for (const item of value) {
    if (!item || typeof item !== "object") continue;
    const entry = item as Partial<TranslateHistoryEntry>;
    if (
      typeof entry.id !== "string" ||
      typeof entry.at !== "number" ||
      typeof entry.source !== "string" ||
      typeof entry.output !== "string" ||
      typeof entry.target !== "string"
    ) {
      continue;
    }
    entries.push({
      id: entry.id,
      at: entry.at,
      source: entry.source,
      output: entry.output,
      target: entry.target,
      model: typeof entry.model === "string" ? entry.model : null,
    });
  }
  return entries.slice(0, TRANSLATE_HISTORY_MAX_ENTRIES);
}

/** Newest first; translating the same text into the same language again replaces the older entry. */
export function addHistoryEntry(
  history: readonly TranslateHistoryEntry[],
  entry: Omit<TranslateHistoryEntry, "id" | "at">,
  now: number = Date.now(),
): TranslateHistoryEntry[] {
  if (
    entry.output.trim() === "" ||
    entry.source.length + entry.output.length > TRANSLATE_HISTORY_MAX_CHARS
  ) {
    return [...history];
  }
  const rest = history.filter(
    (item) => !(item.source === entry.source && item.target === entry.target),
  );
  return [
    {
      ...entry,
      id: `${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      at: now,
    },
    ...rest,
  ].slice(0, TRANSLATE_HISTORY_MAX_ENTRIES);
}

export function searchHistory(
  history: readonly TranslateHistoryEntry[],
  query: string,
): TranslateHistoryEntry[] {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return [...history];
  return history.filter(
    (entry) =>
      entry.source.toLocaleLowerCase().includes(needle) ||
      entry.output.toLocaleLowerCase().includes(needle),
  );
}
