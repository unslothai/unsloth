// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Character } from "./types";

export type SortKey = "recent" | "name" | "newest";

const SORT_KEY = "unsloth_characters_sort";

export function readSort(): SortKey {
  try {
    const value = window.localStorage.getItem(SORT_KEY);
    return value === "name" || value === "newest" ? value : "recent";
  } catch {
    return "recent";
  }
}

export function writeSort(sort: SortKey): void {
  try {
    window.localStorage.setItem(SORT_KEY, sort);
  } catch {
    // the order is a convenience
  }
}

export interface TagGroup {
  key: string;
  count: number;
  label: string;
}

/** Tags group case-insensitively; each group shows its most common spelling. */
export function groupTags(characters: readonly Character[]): TagGroup[] {
  const groups = new Map<string, { count: number; spellings: Map<string, number> }>();
  for (const character of characters) {
    for (const value of character.tags) {
      const key = value.toLowerCase();
      const group = groups.get(key) ?? { count: 0, spellings: new Map<string, number>() };
      group.count += 1;
      group.spellings.set(value, (group.spellings.get(value) ?? 0) + 1);
      groups.set(key, group);
    }
  }
  return [...groups.entries()]
    .map(([key, group]) => ({
      key,
      count: group.count,
      label: [...group.spellings.entries()].sort((a, b) => b[1] - a[1])[0][0],
    }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
}

export interface LibraryFilter {
  query: string;
  favoritesOnly: boolean;
  /** A lowercase tag key from groupTags, or null for every tag. */
  tag: string | null;
}

export function filterCharacters(characters: readonly Character[], filter: LibraryFilter): Character[] {
  const needle = filter.query.trim().toLowerCase();
  return characters.filter(
    (character) =>
      (!filter.favoritesOnly || character.favorite) &&
      (!filter.tag || character.tags.some((value) => value.toLowerCase() === filter.tag)) &&
      (!needle ||
        character.name.toLowerCase().includes(needle) ||
        character.tags.some((value) => value.toLowerCase().includes(needle)) ||
        character.creator.toLowerCase().includes(needle)),
  );
}

/** "recent" keeps the store's order, which already sorts by last chat. */
export function sortCharacters(characters: Character[], sort: SortKey): Character[] {
  if (sort === "name") return [...characters].sort((a, b) => a.name.localeCompare(b.name));
  if (sort === "newest") return [...characters].sort((a, b) => b.createdAt - a.createdAt);
  return characters;
}

/** The selection minus characters deleted since they were picked. */
export function liveSelection(selected: readonly string[], characters: Record<string, Character>): string[] {
  return selected.filter((id) => characters[id] !== undefined);
}
