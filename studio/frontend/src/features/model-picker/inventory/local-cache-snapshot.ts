// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "../../../lib/local-path.ts";
import type { LocalInventoryRow } from "../../hub/inventory/types";

// Hub deduplication already removed snapshots represented in the cached repo lists.
// An active-cache repo id would resolve via the Hub instead of this snapshot.
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
