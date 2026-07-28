// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How an inventory row maps onto the identity its saved settings are keyed by.

import type { CachedInventoryRow, LocalInventoryRow } from "./types";

/**
 * The GGUF variant a settings page should key this row's config by, before any
 * per-repo quant lookup.
 *
 * A standalone `.gguf` has no quant to choose between, but the backend inventory
 * still labels it from its filename (hub/services/models/common.py sets
 * `format_variant` only when the scanned path is a single file). Adopting that
 * label would key its settings to `<path>:Q4_K_M` while the Chat model picker, the
 * detail view's on-device card and the one-time backfill all use the bare path,
 * leaving two surfaces editing two different configs for one file.
 */
export function settingsGgufVariantForRow(
  row: CachedInventoryRow | LocalInventoryRow,
): string | null {
  if (row.kind === "local" && row.path.toLowerCase().endsWith(".gguf")) {
    return null;
  }
  return row.formatVariant?.trim() || null;
}
