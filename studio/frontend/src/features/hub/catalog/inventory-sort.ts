// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormatFilterModelFormat } from "../lib/format-filters";
import type { CachedInventoryRow, LocalInventoryRow } from "../types";

export type InventoryItem =
  | { variant: "cached"; row: CachedInventoryRow }
  | { variant: "local"; row: LocalInventoryRow };

export function inventoryItemTitle(item: InventoryItem): string {
  return item.variant === "cached" ? item.row.repo : item.row.title;
}

export function inventoryItemSize(item: InventoryItem): number {
  return item.variant === "cached" ? item.row.bytes : (item.row.bytes ?? 0);
}

/** The scan never reports "mlx"; an MLX folder says so in its library, tags or oMLX source. */
export function inventoryRowFormat(
  row: CachedInventoryRow | LocalInventoryRow,
): FormatFilterModelFormat {
  if (row.modelFormat === "gguf") {
    return "gguf";
  }
  const mlx =
    row.libraryName?.toLowerCase() === "mlx" ||
    (row.tags ?? []).some((tag) => tag.toLowerCase() === "mlx") ||
    (row.kind === "local" && row.source === "omlx");
  return mlx ? "mlx" : row.modelFormat;
}

const FORMAT_RANK: Record<FormatFilterModelFormat, number> = {
  gguf: 0,
  safetensors: 1,
  checkpoint: 1,
  mlx: 2,
  adapter: 3,
  unknown: 4,
};

/** The format dropdown's order: GGUF, Safetensors, MLX, then adapters and the rest. */
export function inventoryItemFormatRank(item: InventoryItem): number {
  return FORMAT_RANK[inventoryRowFormat(item.row)];
}

export function inventoryItemUpdatedAt(item: InventoryItem): number | null {
  return item.variant === "cached"
    ? (item.row.lastModified ?? null)
    : item.row.updatedAt;
}

/** Newest first; ties and unknowns keep source order. */
export function compareInventoryItemsByRecent(
  a: InventoryItem,
  b: InventoryItem,
): number {
  const aUpdatedAt = inventoryItemUpdatedAt(a);
  const bUpdatedAt = inventoryItemUpdatedAt(b);
  if (aUpdatedAt === bUpdatedAt) {
    return 0;
  }
  if (aUpdatedAt == null) {
    return 1;
  }
  if (bUpdatedAt == null) {
    return -1;
  }
  return bUpdatedAt - aUpdatedAt;
}
