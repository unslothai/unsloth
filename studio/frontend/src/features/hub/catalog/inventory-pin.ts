// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export function inventoryPinId(
  row: CachedInventoryRow | LocalInventoryRow,
): string | null {
  if (row.kind === "cache" || row.source === "hf_cache") {
    return row.repoId;
  }
  // "::" is the repo/quant separator: such an id would read back as a phantom quant pin.
  return row.loadId.includes("::") ? null : row.loadId;
}
