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

function shallowEqual(
  a: Record<string, unknown>,
  b: Record<string, unknown>,
): boolean {
  const aKeys = Object.keys(a);
  if (aKeys.length !== Object.keys(b).length) {
    return false;
  }
  for (const key of aKeys) {
    if (!(Object.hasOwn(b, key) && Object.is(a[key], b[key]))) {
      return false;
    }
  }
  return true;
}

/** Equal own keys, each value identical or a plain object whose own values are identical. */
function sameDisplayItem(a: object, b: object): boolean {
  const left = a as Record<string, unknown>;
  const right = b as Record<string, unknown>;
  const keys = Object.keys(left);
  if (keys.length !== Object.keys(right).length) {
    return false;
  }
  for (const key of keys) {
    if (!Object.hasOwn(right, key)) {
      return false;
    }
    const x = left[key];
    const y = right[key];
    if (Object.is(x, y)) {
      continue;
    }
    if (!(isPlainObject(x) && isPlainObject(y) && shallowEqual(x, y))) {
      return false;
    }
  }
  return true;
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

/**
 * Hands back the previous display node / edge object wherever the new one has the same content,
 * so React Flow keeps its internal node and skips re-rendering nodes and edges that did not change
 * (a drag otherwise re-renders every node of the graph per pointer move).
 */
export function createDisplayGraphReuser(): (
  next: DisplayGraph,
) => DisplayGraph {
  let prev: DisplayGraph | null = null;
  return (next) => {
    prev = reuseDisplayGraph(prev, next);
    return prev;
  };
}
