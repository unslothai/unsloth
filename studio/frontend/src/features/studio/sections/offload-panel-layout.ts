// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Placement of each layer while the window sits at `pos` in the offloaded list. */
export function layerPlacement(
  total: number,
  swapped: number[],
  depth: number,
  pos: number,
): ("resident" | "running" | "copying" | "host")[] {
  const order = new Map(swapped.map((layer, k) => [layer, k]));
  return Array.from({ length: total }, (_, layer) => {
    const k = order.get(layer);
    if (k === undefined) return "resident";
    if (k === pos) return "running";
    if (k > pos && k <= pos + depth) return "copying";
    return "host";
  });
}

/** One card in `GET /api/train/offload`'s `vram_devices`. */
export interface OffloadCard {
  index: number;
  gpu_id?: number;
  name?: string;
  allocated_bytes?: number;
  peak_bytes?: number;
  total_bytes?: number;
  fraction?: number;
}

/** What a VRAM bar shows: peak use, the budget line (null when uncapped) and the card's size. */
export function vramUsage(
  total: number | undefined,
  peak: number | undefined,
  allocated: number | undefined,
  fraction: number | undefined,
): { used: number; budget: number | null; total: number } {
  const size = total ?? 0;
  return {
    used: peak ?? allocated ?? 0,
    budget: fraction && fraction < 1 ? fraction * size : null,
    total: size,
  };
}

/** Decoder layers split by the card they run on, in card order; layers on no known card come last. */
export function groupLayersByCard<T extends { index: number }>(
  total: number,
  layerDevice: Record<string, number> | undefined,
  cards: readonly T[],
): { card: T | null; layers: number[] }[] {
  const groups = cards.map((card) => ({ card: card as T | null, layers: [] as number[] }));
  const byIndex = new Map(cards.map((card, k) => [card.index, k]));
  const unplaced: number[] = [];
  for (let layer = 0; layer < total; layer++) {
    const k = byIndex.get(layerDevice?.[String(layer)] ?? -1);
    if (k === undefined) unplaced.push(layer);
    else groups[k].layers.push(layer);
  }
  if (unplaced.length) groups.push({ card: null, layers: unplaced });
  return groups;
}

/** The layer count a Count box edit sets, or null to keep the current one (empty, zero, not a number). */
export function offloadCountFromInput(text: string): number | null {
  const count = Math.floor(Number(text));
  return count > 0 ? count : null;
}
