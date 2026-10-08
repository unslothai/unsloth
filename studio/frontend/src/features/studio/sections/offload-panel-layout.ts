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

/** The layer count a Count box edit sets, or null to keep the current one (empty, zero, not a number). */
export function offloadCountFromInput(text: string): number | null {
  const count = Math.floor(Number(text));
  return count > 0 ? count : null;
}
