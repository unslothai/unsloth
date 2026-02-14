import type { XYPosition } from "@xyflow/react";

export function syncPositionsRecord(
  prev: Record<string, XYPosition>,
  activeIds: string[],
  defaults: Record<string, XYPosition>,
): Record<string, XYPosition> {
  const next: Record<string, XYPosition> = {};
  for (const id of activeIds) {
    const existing = prev[id];
    if (existing) {
      next[id] = existing;
      continue;
    }
    const fallback = defaults[id];
    if (fallback) {
      next[id] = fallback;
    }
  }

  const prevIds = Object.keys(prev);
  const nextIds = Object.keys(next);
  if (prevIds.length !== nextIds.length) {
    return next;
  }
  for (const id of nextIds) {
    const a = prev[id];
    const b = next[id];
    if (!(a && b && a.x === b.x && a.y === b.y)) {
      return next;
    }
  }
  return prev;
}

export function syncSizesRecord(
  prev: Record<string, { width: number; height: number }>,
  activeIds: string[],
): Record<string, { width: number; height: number }> {
  const active = new Set(activeIds);
  const next: Record<string, { width: number; height: number }> = {};
  for (const [id, size] of Object.entries(prev)) {
    if (active.has(id)) {
      next[id] = size;
    }
  }

  const prevIds = Object.keys(prev);
  const nextIds = Object.keys(next);
  if (prevIds.length !== nextIds.length) {
    return next;
  }
  for (const id of nextIds) {
    const a = prev[id];
    const b = next[id];
    if (!(a && b && a.width === b.width && a.height === b.height)) {
      return next;
    }
  }
  return prev;
}

