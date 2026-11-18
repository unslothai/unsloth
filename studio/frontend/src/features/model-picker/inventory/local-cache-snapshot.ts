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

// Auto-load reads raw /api/hub/local rows, which this picker rule covers too.
export function isOtherCacheSnapshot(row: {
  id: string;
  load_id?: string | null;
  active_cache?: boolean | null;
  companion_prefetch?: boolean;
}): boolean {
  return (
    row.active_cache === false &&
    row.companion_prefetch !== true &&
    looksLikeLocalPath(row.load_id ?? row.id)
  );
}

// A directory load picks its GGUF before Transformers, so a mixed snapshot is offered as GGUF only.
export function pickerLocalCacheSnapshots(
  localRows: readonly LocalInventoryRow[],
  cachedRows: readonly CachedInventoryRow[] = [],
  ggufLoadIds: Iterable<string> = [],
): Set<LocalInventoryRow> {
  const gguf = new Set(ggufLoadIds);
  for (const row of [...localRows, ...cachedRows]) {
    // A partial GGUF has no file linked into the snapshot yet, so the directory load skips it.
    if (row.modelFormat === "gguf" && !row.partial) gguf.add(row.loadId);
  }
  return new Set(
    localRows.filter(
      (row) => isLocalCacheSnapshot(row) && (row.modelFormat === "gguf" || !gguf.has(row.loadId)),
    ),
  );
}
