// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DisplayGraph } from "./derive-display-graph";

function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function sameEntries(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
  same: (x: unknown, y: unknown) => boolean,
): boolean {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) {
    return false;
  }
  return keys.every((key) => Object.hasOwn(b, key) && same(a[key], b[key]));
}

/** Equal own keys, each value identical or a plain object whose own values are identical. */
function sameDisplayItem(a: object, b: object): boolean {
  return sameEntries(
    a as Record<string, unknown>,
    b as Record<string, unknown>,
    (x, y) =>
      Object.is(x, y) ||
      (isPlainObject(x) && isPlainObject(y) && sameEntries(x, y, Object.is)),
  );
}

function reuseItems<T extends { id: string }>(prev: T[], next: T[]): T[] {
  if (prev.length === 0) {
    return next;
  }
  const byId = new Map<string, T>();
  for (const item of prev) {
    byId.set(item.id, item);
  }
  let changed = prev.length !== next.length;
  const out = next.map((item, index) => {
    const old = byId.get(item.id);
    const kept = old && sameDisplayItem(old, item) ? old : item;
    if (kept !== prev[index]) {
      changed = true;
    }
    return kept;
  });
  return changed ? out : prev;
}

export function reuseDisplayGraph(
  prev: DisplayGraph | null,
  next: DisplayGraph,
): DisplayGraph {
  if (!prev) {
    return next;
  }
  const nodes = reuseItems(prev.nodes, next.nodes);
  const edges = reuseItems(prev.edges, next.edges);
  if (nodes === prev.nodes && edges === prev.edges) {
    return prev;
  }
  return { nodes, edges };
}

// React Flow re-renders every node whose object identity changed, so a drag step would re-render the whole graph.
export function createDisplayGraphReuser(): (
  next: DisplayGraph,
) => DisplayGraph {
  let prev: DisplayGraph | null = null;
  return (next) => {
    prev = reuseDisplayGraph(prev, next);
    return prev;
  };
}
