// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { matchTokens, tokenizeQuery } from "@/lib/search-text";
import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export { tokenizeQuery };

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
  return matchTokens(inventoryRowHaystack(row), tokens);
}
