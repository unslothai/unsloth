// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "../types";

/** The key an On Device row pins under: a cached repo by its repo id, a filesystem row (custom
 *  folder, LM Studio, oMLX, ./models) by its load id. */
export function inventoryPinId(
  row: CachedInventoryRow | LocalInventoryRow,
): string | null {
  if (row.kind === "cache" || row.source === "hf_cache") {
    return row.repoId;
  }
  // A pin key reads "::" as its repo/quant separator, so such an id would come back as a
  // phantom quant pin.
  return row.loadId.includes("::") ? null : row.loadId;
}
