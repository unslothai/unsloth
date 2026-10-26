// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  matchTokens,
  normalizeForSearch,
  tokenizeQuery,
} from "@/features/hub/lib/search-text";
import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export { tokenizeQuery };

/** Rank personal names as visible names while the haystack retains original identifiers. */
function inventoryRowName(row: CachedInventoryRow | LocalInventoryRow): string {
  return row.annotation?.display_name ||
    (row.kind === "cache" ? row.repoId : row.title);
}

/** Include personal details in search text without treating labels as Hub capability tags. */
function inventoryRowHaystack(
  row: CachedInventoryRow | LocalInventoryRow,
): string {
  const personal = row.annotation
    ? [row.annotation.display_name, ...row.annotation.labels, row.annotation.notes]
    : [];
  if (row.kind === "cache")
    return [
      row.repoId,
      row.modelFormat,
      row.pipelineTag ?? "",
      row.libraryName ?? "",
      ...(row.tags ?? []),
      ...personal,
    ].join(" ");
  return [
    row.title,
    row.owner,
    row.sourceLabel,
    row.path,
    row.loadId,
    row.modelFormat,
    row.repoId ?? "",
    ...personal,
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
