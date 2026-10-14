// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Search tabs, in order. "all" mixes every kind. */
export const CHAT_SEARCH_TABS = [
  "all",
  "chats",
  "projects",
  "files",
  "models",
] as const;
export type ChatSearchTab = (typeof CHAT_SEARCH_TABS)[number];
export type ChatSearchKind = Exclude<ChatSearchTab, "all">;

/** A search result of any kind. */
export interface ChatSearchRow {
  /** Unique across kinds; the cmdk value. */
  key: string;
  kind: ChatSearchKind;
  title: string;
  /** Sort key, newest first. */
  time: number;
  /** Lowercased text to match against. */
  haystack: string;
}

/** Rows per kind on the All tab while typing. */
export const ALL_TAB_GROUP_LIMIT = 5;
/** Rows under Recents on an empty All tab. */
export const RECENT_ROW_LIMIT = 5;

// Lowercased whitespace tokens of the query.
export function queryTokens(search: string): string[] {
  return search.trim().toLowerCase().split(/\s+/).filter(Boolean);
}

export function haystackMatches(haystack: string, tokens: string[]): boolean {
  return tokens.every((token) => haystack.includes(token));
}

/** Every token must be a substring. */
export function filterRows<T extends ChatSearchRow>(rows: T[], search: string): T[] {
  const tokens = queryTokens(search);
  if (tokens.length === 0) return rows;
  return rows.filter((row) => haystackMatches(row.haystack, tokens));
}

/** The next or previous tab, without wrapping. */
export function stepTab(tab: ChatSearchTab, step: -1 | 1): ChatSearchTab {
  const index = CHAT_SEARCH_TABS.indexOf(tab) + step;
  return CHAT_SEARCH_TABS[Math.min(Math.max(index, 0), CHAT_SEARCH_TABS.length - 1)];
}

/** Newest rows across kinds, for the empty All tab. */
export function recentRows<T extends ChatSearchRow>(
  byKind: Record<ChatSearchKind, T[]>,
  limit = RECENT_ROW_LIMIT,
): T[] {
  // Kinds need not arrive sorted by `time` (chats are listed by creation, ranked by activity).
  return Object.values(byKind)
    .flat()
    .sort((a, b) => b.time - a.time)
    .slice(0, limit);
}

/** Arrows switch tabs only at the ends of the query, so they still move the caret. */
export function tabStepForKey(
  key: string,
  input: { value: string; selectionStart: number | null; selectionEnd: number | null },
): -1 | 1 | null {
  const { value, selectionStart, selectionEnd } = input;
  if (selectionStart !== selectionEnd) return null;
  if (key === "ArrowLeft" && selectionStart === 0) return -1;
  if (key === "ArrowRight" && selectionEnd === value.length) return 1;
  return null;
}
