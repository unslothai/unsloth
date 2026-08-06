


// How an inventory row maps onto the identity its saved settings are keyed by.

import type { CachedInventoryRow, LocalInventoryRow } from "./types";

/**
 * The GGUF variant a settings page keys this row's config by, before any per-repo lookup. A
 * standalone `.gguf` has no quant to choose between, but the inventory still labels it from its
 * filename, and adopting that label would key its settings to `<path>:Q4_K_M` while every other
 * surface uses the bare path, leaving two configs for one file.
 */
export function settingsGgufVariantForRow(
  row: CachedInventoryRow | LocalInventoryRow,
): string | null {
  if (row.kind === "local" && row.path.toLowerCase().endsWith(".gguf")) {
    return null;
  }
  return row.formatVariant?.trim() || null;
}
