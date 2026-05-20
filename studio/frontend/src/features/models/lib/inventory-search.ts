// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export function tokenizeQuery(query: string): string[] {
  if (!query) return [];
  const tokens: string[] = [];
  for (const piece of query.toLowerCase().split(/\s+/)) {
    if (piece) tokens.push(piece);
  }
  return tokens;
}

function inventoryRowHaystack(
  row: CachedInventoryRow | LocalInventoryRow,
): string {
  if (row.kind === "cache") return row.repoId.toLowerCase();
  return [row.title, row.owner, row.sourceLabel, row.path, row.repoId ?? ""]
    .join(" ")
    .toLowerCase();
}

export function inventoryRowMatches(
  row: CachedInventoryRow | LocalInventoryRow,
  tokens: readonly string[],
): boolean {
  if (tokens.length === 0) return true;
  const haystack = inventoryRowHaystack(row);
  for (const token of tokens) {
    if (!haystack.includes(token)) return false;
  }
  return true;
}
