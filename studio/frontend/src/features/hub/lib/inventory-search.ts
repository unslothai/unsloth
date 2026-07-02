// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  matchTokens,
  normalizeForSearch,
  tokenizeQuery,
} from "@/features/hub/lib/search-text";
import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export { tokenizeQuery };

function inventoryRowName(row: CachedInventoryRow | LocalInventoryRow): string {
  return row.kind === "cache" ? row.repoId : row.title;
}

function inventoryRowHaystack(
  row: CachedInventoryRow | LocalInventoryRow,
): string {
  if (row.kind === "cache")
    return [
      row.repoId,
      row.modelFormat,
      row.pipelineTag ?? "",
      row.libraryName ?? "",
      ...(row.tags ?? []),
    ].join(" ");
  return [
    row.title,
    row.owner,
    row.sourceLabel,
    row.path,
    row.loadId,
    row.modelFormat,
    row.repoId ?? "",
  ].join(" ");
}

export function inventoryRowMatches(
  row: CachedInventoryRow | LocalInventoryRow,
  tokens: readonly string[],
): boolean {
  if (tokens.length === 0) return true;
  return matchTokens(inventoryRowHaystack(row), tokens);
}

export function scoreInventoryRow(
  row: CachedInventoryRow | LocalInventoryRow,
  tokens: readonly string[],
): number {
  if (tokens.length === 0) return 0;
  const name = normalizeForSearch(inventoryRowName(row));
  const haystack = normalizeForSearch(inventoryRowHaystack(row));
  let score = 0;
  for (const token of tokens) {
    if (name.startsWith(token)) score += 3;
    else if (name.includes(token)) score += 2;
    else if (haystack.includes(token)) score += 1;
    else return 0;
  }
  return score;
}
