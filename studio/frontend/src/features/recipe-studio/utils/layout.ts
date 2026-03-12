// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../constants";
import type { LayoutDirection } from "../types";
import { readNodeHeight, readNodeWidth } from "./rf-node-dimensions";

type LayoutOptions = {
  direction?: LayoutDirection;
  nodesep?: number;
  ranksep?: number;
  edgesep?: number;
  nodeWidth?: number;
  nodeHeight?: number;
};

export function getLayoutedElements<TNode extends Node>(
  nodes: TNode[],
  edges: Edge[],
  options: LayoutOptions = {},
): { nodes: TNode[]; edges: Edge[] } {
  const {
    direction = "LR",
    nodesep = 80,
    ranksep = 80,
    edgesep = 28,
    nodeWidth = DEFAULT_NODE_WIDTH,
    nodeHeight = DEFAULT_NODE_HEIGHT,
  } = options;

  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: direction,
    nodesep,
    ranksep,
    edgesep,
    ranker: "network-simplex",
  });

  nodes.forEach((node) => {
    const width = readNodeWidth(node) ?? nodeWidth;
    const height = readNodeHeight(node) ?? nodeHeight;
    graph.setNode(node.id, { width, height });
  });

  edges.forEach((edge) => {
    const semantic = edge.type === "semantic";
    const aux = edge.source.startsWith("aux-") || edge.target.startsWith("aux-");
    graph.setEdge(edge.source, edge.target, {
      minlen: semantic ? 1 : 1,
      weight: semantic ? 10 : aux ? 1 : 3,
    });
  });

  dagre.layout(graph);

  const layoutedNodes = nodes.map((node) => {
    const pos = graph.node(node.id);
    const width = readNodeWidth(node) ?? nodeWidth;
    const height = readNodeHeight(node) ?? nodeHeight;
    return {
      ...node,
      position: {
        x: pos.x - width / 2,
        y: pos.y - height / 2,
      },
    };
  });

  return { nodes: layoutedNodes, edges };
}
