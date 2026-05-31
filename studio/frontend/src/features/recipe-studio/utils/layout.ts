// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../constants";
import { INFRA_NODE_KINDS, type LayoutDirection, type NodeConfig } from "../types";
import { readNodeHeight, readNodeWidth } from "./rf-node-dimensions";

type LayoutOptions = {
  direction?: LayoutDirection;
  nodesep?: number;
  ranksep?: number;
  edgesep?: number;
  nodeWidth?: number;
  nodeHeight?: number;
  configs?: Record<string, NodeConfig>;
};

/**
 * Pipeline rank order used to enforce a logical flow even for disconnected nodes.
 * Lower rank = earlier in the pipeline.
 */
function getPipelineRank(config: NodeConfig | undefined): number {
  if (!config) {
    return 2;
  }
  switch (config.kind) {
    case "seed":
      return 0;
    case "sampler":
      return 1;
    case "expression":
      return 2;
    case "llm":
      return 3;
    case "validator":
      return 4;
    default:
      return 2;
  }
}

function isInfraNode(
  nodeId: string,
  configs: Record<string, NodeConfig>,
): boolean {
  const config = configs[nodeId];
  return config ? INFRA_NODE_KINDS.has(config.kind) : false;
}

function isAuxNode(nodeId: string): boolean {
  return nodeId.startsWith("aux-");
}

function getEdgeWeight(edgeType: string | undefined): number {
  if (edgeType === "phantom") {
    return 0;
  }
  if (edgeType === "semantic") {
    return 10;
  }
  return 3;
}

/**
 * Build phantom edges between disconnected data-pipeline nodes so dagre
 * respects the pipeline rank order even when blocks aren't wired together.
 *
 * Groups nodes by rank, then inserts invisible edges from the last node of
 * rank N to the first node of rank N+1 when no real edge already connects them.
 */
function buildPhantomEdges(
  nodes: Node[],
  edges: Edge[],
  configs: Record<string, NodeConfig>,
): Edge[] {
  // Group nodes by rank
  const byRank = new Map<number, string[]>();
  for (const node of nodes) {
    const rank = getPipelineRank(configs[node.id]);
    const list = byRank.get(rank) ?? [];
    list.push(node.id);
    byRank.set(rank, list);
  }

  const ranks = Array.from(byRank.keys()).sort((a, b) => a - b);
  const phantoms: Edge[] = [];

  for (let i = 0; i < ranks.length - 1; i++) {
    const currentIds = byRank.get(ranks[i]) ?? [];
    const nextIds = byRank.get(ranks[i + 1]) ?? [];
    if (currentIds.length === 0 || nextIds.length === 0) {
      continue;
    }

    // Check if any real edge already connects these rank groups
    const hasRealEdge = edges.some(
      (e) => currentIds.includes(e.source) && nextIds.includes(e.target),
    );
    if (hasRealEdge) {
      continue;
    }

    // Insert one phantom edge from last node in current rank to first in next
    phantoms.push({
      id: `phantom-${ranks[i]}-${ranks[i + 1]}`,
      source: currentIds[currentIds.length - 1],
      target: nextIds[0],
      type: "phantom",
    });
  }

  return phantoms;
}

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
    configs,
  } = options;

  // When configs are provided, filter out infra and aux nodes from dagre
  const hasConfigs = configs && Object.keys(configs).length > 0;
  const dataNodes = hasConfigs
    ? nodes.filter((n) => !(isInfraNode(n.id, configs) || isAuxNode(n.id)))
    : nodes;
  const dataEdges = hasConfigs
    ? edges.filter(
        (e) =>
          !(
            isInfraNode(e.source, configs) ||
            isInfraNode(e.target, configs) ||
            isAuxNode(e.source) ||
            isAuxNode(e.target)
          ),
      )
    : edges;

  // Build phantom edges to enforce pipeline rank ordering for disconnected nodes
  const phantomEdges = hasConfigs
    ? buildPhantomEdges(dataNodes, dataEdges, configs)
    : [];

  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({
    rankdir: direction,
    nodesep,
    ranksep,
    edgesep,
    ranker: "network-simplex",
  });

  for (const node of dataNodes) {
    const width = readNodeWidth(node) ?? nodeWidth;
    const height = readNodeHeight(node) ?? nodeHeight;
    graph.setNode(node.id, { width, height });
  }

  const allDagreEdges = [...dataEdges, ...phantomEdges];
  for (const edge of allDagreEdges) {
    const weight = getEdgeWeight(edge.type);
    graph.setEdge(edge.source, edge.target, { minlen: 1, weight });
  }

  dagre.layout(graph);

  // Build position map from dagre results (data nodes only)
  const layoutedPositions = new Map<string, { x: number; y: number }>();
  for (const node of dataNodes) {
    const pos = graph.node(node.id);
    const width = readNodeWidth(node) ?? nodeWidth;
    const height = readNodeHeight(node) ?? nodeHeight;
    layoutedPositions.set(node.id, {
      x: pos.x - width / 2,
      y: pos.y - height / 2,
    });
  }

  // Apply positions: data nodes get dagre positions, infra/aux keep original
  const layoutedNodes = nodes.map((node) => {
    const position = layoutedPositions.get(node.id);
    if (!position) {
      return node;
    }
    return {
      ...node,
      position,
    };
  });

  return { nodes: layoutedNodes, edges };
}
