// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dagre from "@dagrejs/dagre";

export interface LayoutRequest {
  nodes: { id: string; width: number; height: number }[];
  edges: { source: string; target: string }[];
  direction: "LR" | "TB";
  nodesep?: number;
  ranksep?: number;
}

export type LayoutResult = Record<string, [number, number]>;

export function computeLayout(request: LayoutRequest): LayoutResult {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({
    rankdir: request.direction,
    nodesep: request.nodesep ?? 40,
    ranksep: request.ranksep ?? 90,
    marginx: 20,
    marginy: 20,
  });
  graph.setDefaultEdgeLabel(() => ({}));
  for (const node of request.nodes)
    graph.setNode(node.id, { width: node.width, height: node.height });
  for (const edge of request.edges) {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target))
      graph.setEdge(edge.source, edge.target);
  }
  dagre.layout(graph);
  const result: LayoutResult = {};
  for (const node of request.nodes) {
    const laid = graph.node(node.id) as { x: number; y: number } | undefined;
    if (laid)
      result[node.id] = [
        Math.round(laid.x - node.width / 2),
        Math.round(laid.y - node.height / 2),
      ];
  }
  return result;
}
