// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FlowEdge } from "../types";

const NONE: readonly FlowEdge[] = Object.freeze([]);
const indexes = new WeakMap<object, Map<string, FlowEdge[]>>();

/** Index builds, for the perf tests. */
export const wireIndexStats = { builds: 0 };

/**
 * The wires into a node. Card selectors run on every doc store update; the index is built once per
 * edge set, so each lookup costs the node's own wires rather than a scan of every edge.
 */
export function wiresInto(
  graph: { edges: Record<string, FlowEdge> },
  nodeId: string,
): readonly FlowEdge[] {
  let index = indexes.get(graph.edges);
  if (!index) {
    wireIndexStats.builds += 1;
    index = new Map();
    for (const edge of Object.values(graph.edges)) {
      const list = index.get(edge.target);
      if (list) list.push(edge);
      else index.set(edge.target, [edge]);
    }
    indexes.set(graph.edges, index);
  }
  return index.get(nodeId) ?? NONE;
}
