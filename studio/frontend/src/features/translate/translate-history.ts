// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TRANSLATE_LANGUAGES } from "./languages.ts";

export type TranslateHistoryEntry = {
  id: string;
  at: number;
  source: string;
  output: string;
  target: string;
  /** Language code the user set; null when the model detected it. */
  from: string | null;
  model: string | null;
};

export const TRANSLATE_HISTORY_MAX_ENTRIES = 100;
/** A longer translation is not kept: the list lives in localStorage, a few megabytes in all. */
export const TRANSLATE_HISTORY_MAX_CHARS = 20_000;
export const TRANSLATE_HISTORY_TOTAL_CHARS = 1_000_000;
export const TRANSLATE_HISTORY_FILE_NAME = "UnslothTranslateFullHistory.txt";

export function capHistory(
  history: readonly TranslateHistoryEntry[],
): TranslateHistoryEntry[] {
  let chars = 0;
  let keep = 0;
  for (const entry of history.slice(0, TRANSLATE_HISTORY_MAX_ENTRIES)) {
    chars += entry.source.length + entry.output.length;
    if (chars > TRANSLATE_HISTORY_TOTAL_CHARS && keep > 0) break;
    keep += 1;
  }
  return history.slice(0, keep);
}

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
      from: typeof entry.from === "string" ? entry.from : null,
      model: typeof entry.model === "string" ? entry.model : null,
    });
  }
  return capHistory(entries);
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
  return capHistory([
    {
      ...entry,
      id: `${now.toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      at: now,
    },
    ...rest,
  ]);
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

const englishName = (code: string | null): string =>
  code === null
    ? "Detected language"
    : (TRANSLATE_LANGUAGES.find((language) => language.code === code)?.name ??
      code);

export function buildChatHandoff(
  entries: readonly Pick<
    TranslateHistoryEntry,
    "at" | "source" | "output" | "target" | "from"
  >[],
): string {
  const blocks = [...entries]
    .sort((a, b) => a.at - b.at)
    .map(
      (entry) =>
        `Original (${englishName(entry.from)}):\n${entry.source}\n\nTranslation (${englishName(entry.target)}):\n${entry.output}`,
    );
  if (blocks.length === 0) return "";
  return `Here ${blocks.length === 1 ? "is a translation" : "are my recent translations"}:\n\n${blocks.join("\n\n---\n\n")}`;
}

export function exportHistory(
  history: readonly TranslateHistoryEntry[],
): string {
  if (history.length === 0) return "";
  return `${[...history]
    .sort((a, b) => a.at - b.at)
    .map((entry) =>
      [
        `[${new Date(entry.at).toISOString()}] ${englishName(entry.from)} -> ${englishName(entry.target)}${entry.model ? ` (${entry.model})` : ""}`,
        entry.source,
        "",
        entry.output,
      ].join("\n"),
    )
    .join("\n\n---\n\n")}\n`;
}
