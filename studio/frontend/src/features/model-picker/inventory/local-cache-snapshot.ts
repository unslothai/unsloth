// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "../../../lib/local-path.ts";
import type { CachedInventoryRow, LocalInventoryRow } from "../../hub/inventory/types";

// Runs after Hub dedupe; an active-cache row loads by repo id, not by this snapshot path.
export function isLocalCacheSnapshot(row: LocalInventoryRow): boolean {
  return (
    row.source === "hf_cache" &&
    row.activeCache === false &&
    looksLikeLocalPath(row.loadId) &&
    !row.partial &&
    !row.companionPrefetch &&
    row.capabilities.canChat
  );
}

// A directory load picks its GGUF before Transformers, so a mixed snapshot is offered as GGUF only.
export function pickerLocalCacheSnapshots(
  localRows: readonly LocalInventoryRow[],
  cachedRows: readonly CachedInventoryRow[] = [],
): Set<LocalInventoryRow> {
  const ggufLoadIds = new Set(
    [...localRows, ...cachedRows]
      // A partial GGUF has no file linked into the snapshot yet, so the directory load skips it.
      .filter((row) => row.modelFormat === "gguf" && !row.partial)
      .map((row) => row.loadId),
  );
  return new Set(
    localRows.filter(
      (row) =>
        isLocalCacheSnapshot(row) &&
        (row.modelFormat === "gguf" || !ggufLoadIds.has(row.loadId)),
    ),
  );
}
