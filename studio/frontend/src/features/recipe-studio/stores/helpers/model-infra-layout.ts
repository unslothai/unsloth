// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, XYPosition } from "@xyflow/react";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../../constants";
import type { LayoutDirection, NodeConfig, RecipeNode } from "../../types";
import { HANDLE_IDS, normalizeRecipeHandleId } from "../../utils/handles";
import { readNodeHeight, readNodeWidth } from "../../utils/rf-node-dimensions";

type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type Bounds = {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
};

function toRect(node: RecipeNode): Rect {
  return {
    x: node.position.x,
    y: node.position.y,
    width: readNodeWidth(node) ?? DEFAULT_NODE_WIDTH,
    height: readNodeHeight(node) ?? DEFAULT_NODE_HEIGHT,
  };
}

function intersects(a: Rect, b: Rect, pad = 18): boolean {
  return !(
    a.x + a.width + pad <= b.x ||
    b.x + b.width + pad <= a.x ||
    a.y + a.height + pad <= b.y ||
    b.y + b.height + pad <= a.y
  );
}

function findNonOverlappingPosition(
  preferred: XYPosition,
  width: number,
  height: number,
  occupied: Rect[],
): XYPosition {
  const step = 24;
  for (let ring = 0; ring <= 16; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        if (ring > 0 && Math.max(Math.abs(dx), Math.abs(dy)) !== ring) {
          continue;
        }
        const candidate = {
          x: preferred.x + dx * step,
          y: preferred.y + dy * step,
        };
        const rect = {
          x: candidate.x,
          y: candidate.y,
          width,
          height,
        };
        if (!occupied.some((item) => intersects(rect, item))) {
          return candidate;
        }
      }
    }
  }
  return preferred;
}

function isProviderToConfigEdge(
  edge: Edge,
  configs: Record<string, NodeConfig>,
): boolean {
  const source = configs[edge.source];
  const target = configs[edge.target];
  return source?.kind === "model_provider" && target?.kind === "model_config";
}

function isConfigToLlmEdge(
  edge: Edge,
  configs: Record<string, NodeConfig>,
): boolean {
  const source = configs[edge.source];
  const target = configs[edge.target];
  return source?.kind === "model_config" && target?.kind === "llm";
}

function isToolConfigToLlmEdge(
  edge: Edge,
  configs: Record<string, NodeConfig>,
): boolean {
  const source = configs[edge.source];
  const target = configs[edge.target];
  return source?.kind === "tool_config" && target?.kind === "llm";
}

function usageKey(nodeId: string, handleId: string): string {
  return `${nodeId}::${handleId}`;
}

function incrementUsage(
  map: Map<string, number>,
  nodeId: string,
  handleId: string,
): void {
  const key = usageKey(nodeId, handleId);
  map.set(key, (map.get(key) ?? 0) + 1);
}

function decrementUsage(
  map: Map<string, number>,
  nodeId: string,
  handleId: string,
): void {
  const key = usageKey(nodeId, handleId);
  map.set(key, Math.max(0, (map.get(key) ?? 0) - 1));
}

function getUsage(
  map: Map<string, number>,
  nodeId: string,
  handleId: string,
): number {
  return map.get(usageKey(nodeId, handleId)) ?? 0;
}

function pickHandleByUsage(
  candidates: string[],
  nodeId: string,
  usageMap: Map<string, number>,
): string {
  const free = candidates.filter(
    (handleId) => getUsage(usageMap, nodeId, handleId) === 0,
  );
  if (free.length > 0) {
    return free[0];
  }
  let bestHandle = candidates[0];
  let bestCount = Number.POSITIVE_INFINITY;
  for (const handleId of candidates) {
    const count = getUsage(usageMap, nodeId, handleId);
    if (count < bestCount) {
      bestHandle = handleId;
      bestCount = count;
    }
  }
  return bestHandle;
}

function applyEdgeWithHandles(
  edge: Edge,
  sourceHandle: string,
  targetHandle: string,
  sourceUsage: Map<string, number>,
  targetUsage: Map<string, number>,
): Edge {
  incrementUsage(sourceUsage, edge.source, sourceHandle);
  incrementUsage(targetUsage, edge.target, targetHandle);
  return { ...edge, sourceHandle, targetHandle, type: "semantic" };
}

function getNodeCenter(node: RecipeNode): { x: number; y: number } {
  const width = readNodeWidth(node) ?? DEFAULT_NODE_WIDTH;
  const height = readNodeHeight(node) ?? DEFAULT_NODE_HEIGHT;
  return {
    x: node.position.x + width / 2,
    y: node.position.y + height / 2,
  };
}

function collectBounds(
  ids: string[],
  nodesById: Map<string, RecipeNode>,
): Bounds | null {
  const rects = ids
    .map((id) => nodesById.get(id))
    .flatMap((node) => (node ? [toRect(node)] : []));
  if (rects.length === 0) {
    return null;
  }
  return rects.reduce<Bounds>(
    (acc, rect) => ({
      minX: Math.min(acc.minX, rect.x),
      maxX: Math.max(acc.maxX, rect.x + rect.width),
      minY: Math.min(acc.minY, rect.y),
      maxY: Math.max(acc.maxY, rect.y + rect.height),
    }),
    {
      minX: rects[0].x,
      maxX: rects[0].x + rects[0].width,
      minY: rects[0].y,
      maxY: rects[0].y + rects[0].height,
    },
  );
}

function sortPreferredLlmTargetHandles(
  direction: LayoutDirection,
  sourceNode: RecipeNode | undefined,
  targetNode: RecipeNode | undefined,
): string[] {
  const sourceCenter = sourceNode ? getNodeCenter(sourceNode) : { x: 0, y: 0 };
  const targetCenter = targetNode ? getNodeCenter(targetNode) : { x: 0, y: 0 };

  if (direction === "TB") {
    const horizontalFirst =
      sourceCenter.x <= targetCenter.x
        ? [HANDLE_IDS.dataIn, HANDLE_IDS.dataInRight]
        : [HANDLE_IDS.dataInRight, HANDLE_IDS.dataIn];
    return [...horizontalFirst, HANDLE_IDS.dataInTop, HANDLE_IDS.dataInBottom];
  }

  const verticalFirst =
    sourceCenter.y <= targetCenter.y
      ? [HANDLE_IDS.dataInTop, HANDLE_IDS.dataInBottom]
      : [HANDLE_IDS.dataInBottom, HANDLE_IDS.dataInTop];
  return [...verticalFirst, HANDLE_IDS.dataIn, HANDLE_IDS.dataInRight];
}

function getProviderSourceHandleCandidates(
  direction: LayoutDirection,
): string[] {
  return direction === "TB"
    ? [HANDLE_IDS.semanticOut, HANDLE_IDS.semanticOutBottom]
    : [HANDLE_IDS.semanticOutBottom, HANDLE_IDS.semanticOut];
}

function getProviderTargetHandleCandidates(
  direction: LayoutDirection,
): string[] {
  return direction === "TB"
    ? [HANDLE_IDS.semanticIn, HANDLE_IDS.semanticInTop]
    : [HANDLE_IDS.semanticInTop, HANDLE_IDS.semanticIn];
}

function getConfigSourceHandleCandidates(direction: LayoutDirection): string[] {
  return direction === "TB"
    ? [HANDLE_IDS.semanticOut]
    : [HANDLE_IDS.semanticOutBottom];
}

export function optimizeModelInfraEdgeHandles(
  edges: Edge[],
  nodes: RecipeNode[],
  configs: Record<string, NodeConfig>,
  direction: LayoutDirection,
): Edge[] {
  const nodesById = new Map(nodes.map((node) => [node.id, node] as const));
  const sourceUsage = new Map<string, number>();
  const targetUsage = new Map<string, number>();

  for (const edge of edges) {
    const sourceHandle = normalizeRecipeHandleId(edge.sourceHandle);
    const targetHandle = normalizeRecipeHandleId(edge.targetHandle);
    if (sourceHandle) {
      incrementUsage(sourceUsage, edge.source, sourceHandle);
    }
    if (targetHandle) {
      incrementUsage(targetUsage, edge.target, targetHandle);
    }
  }

  const nextEdges: Edge[] = [];
  for (const edge of edges) {
    const source = configs[edge.source];
    const target = configs[edge.target];
    if (!(source && target)) {
      nextEdges.push(edge);
      continue;
    }

    const sourceHandleBefore = normalizeRecipeHandleId(edge.sourceHandle);
    const targetHandleBefore = normalizeRecipeHandleId(edge.targetHandle);
    const isSemanticInfra =
      isProviderToConfigEdge(edge, configs) ||
      isConfigToLlmEdge(edge, configs) ||
      isToolConfigToLlmEdge(edge, configs);
    if (!isSemanticInfra) {
      nextEdges.push(edge);
      continue;
    }

    if (sourceHandleBefore) {
      decrementUsage(sourceUsage, edge.source, sourceHandleBefore);
    }
    if (targetHandleBefore) {
      decrementUsage(targetUsage, edge.target, targetHandleBefore);
    }

    if (isProviderToConfigEdge(edge, configs)) {
      const sourceCandidates = getProviderSourceHandleCandidates(direction);
      const targetCandidates = getProviderTargetHandleCandidates(direction);
      const sourceHandle = pickHandleByUsage(
        sourceCandidates,
        edge.source,
        sourceUsage,
      );
      const targetHandle = pickHandleByUsage(
        targetCandidates,
        edge.target,
        targetUsage,
      );
      nextEdges.push(
        applyEdgeWithHandles(
          edge,
          sourceHandle,
          targetHandle,
          sourceUsage,
          targetUsage,
        ),
      );
      continue;
    }

    const targetCandidates = sortPreferredLlmTargetHandles(
      direction,
      nodesById.get(edge.source),
      nodesById.get(edge.target),
    );
    const sourceCandidates = getConfigSourceHandleCandidates(direction);
    const sourceHandle = pickHandleByUsage(
      sourceCandidates,
      edge.source,
      sourceUsage,
    );
    const targetHandle = pickHandleByUsage(
      targetCandidates,
      edge.target,
      targetUsage,
    );
    nextEdges.push(
      applyEdgeWithHandles(
        edge,
        sourceHandle,
        targetHandle,
        sourceUsage,
        targetUsage,
      ),
    );
  }

  return nextEdges;
}

export function centerModelInfraNodes(
  nodes: RecipeNode[],
  edges: Edge[],
  configs: Record<string, NodeConfig>,
  direction: LayoutDirection,
): RecipeNode[] {
  const nodesById = new Map(nodes.map((node) => [node.id, node] as const));
  const configToLlmIds = new Map<string, string[]>();
  const toolConfigToLlmIds = new Map<string, string[]>();
  const providerToConfigIds = new Map<string, string[]>();

  for (const edge of edges) {
    if (isProviderToConfigEdge(edge, configs)) {
      const entries = providerToConfigIds.get(edge.source) ?? [];
      if (!entries.includes(edge.target)) {
        entries.push(edge.target);
      }
      providerToConfigIds.set(edge.source, entries);
      continue;
    }
    if (isConfigToLlmEdge(edge, configs)) {
      const entries = configToLlmIds.get(edge.source) ?? [];
      if (!entries.includes(edge.target)) {
        entries.push(edge.target);
      }
      configToLlmIds.set(edge.source, entries);
      continue;
    }
    if (isToolConfigToLlmEdge(edge, configs)) {
      const entries = toolConfigToLlmIds.get(edge.source) ?? [];
      if (!entries.includes(edge.target)) {
        entries.push(edge.target);
      }
      toolConfigToLlmIds.set(edge.source, entries);
    }
  }

  const modelConfigIds = Object.values(configs)
    .filter(
      (config) => config.kind === "model_config" && nodesById.has(config.id),
    )
    .map((config) => config.id);
  const modelProviderIds = Object.values(configs)
    .filter(
      (config) => config.kind === "model_provider" && nodesById.has(config.id),
    )
    .map((config) => config.id);
  const toolConfigIds = Object.values(configs)
    .filter((config) => config.kind === "tool_config" && nodesById.has(config.id))
    .map((config) => config.id);

  const occupiedById = new Map(
    nodes.map((node) => [node.id, toRect(node)] as const),
  );
  const clusterGap = 72;

  const placeNode = (nodeId: string, preferred: XYPosition): void => {
    const currentNode = nodesById.get(nodeId);
    if (!currentNode) {
      return;
    }
    const width = readNodeWidth(currentNode) ?? DEFAULT_NODE_WIDTH;
    const height = readNodeHeight(currentNode) ?? DEFAULT_NODE_HEIGHT;
    occupiedById.delete(nodeId);
    const position = findNonOverlappingPosition(
      preferred,
      width,
      height,
      Array.from(occupiedById.values()),
    );
    const nextNode = { ...currentNode, position };
    nodesById.set(nodeId, nextNode);
    occupiedById.set(nodeId, {
      x: position.x,
      y: position.y,
      width,
      height,
    });
  };

  for (const modelConfigId of modelConfigIds) {
    const llmIds = configToLlmIds.get(modelConfigId) ?? [];
    const targetBounds = collectBounds(llmIds, nodesById);
    const modelConfigNode = nodesById.get(modelConfigId);
    if (!(targetBounds && modelConfigNode)) {
      continue;
    }
    const width = readNodeWidth(modelConfigNode) ?? DEFAULT_NODE_WIDTH;
    const height = readNodeHeight(modelConfigNode) ?? DEFAULT_NODE_HEIGHT;
    const preferred =
      direction === "LR"
        ? {
            x: (targetBounds.minX + targetBounds.maxX) / 2 - width / 2,
            y: targetBounds.minY - height - clusterGap,
          }
        : {
            x: targetBounds.minX - width - clusterGap,
            y: (targetBounds.minY + targetBounds.maxY) / 2 - height / 2,
          };
    placeNode(modelConfigId, preferred);
  }

  for (const modelProviderId of modelProviderIds) {
    const configIds = providerToConfigIds.get(modelProviderId) ?? [];
    const targetBounds = collectBounds(configIds, nodesById);
    const modelProviderNode = nodesById.get(modelProviderId);
    if (!(targetBounds && modelProviderNode)) {
      continue;
    }
    const width = readNodeWidth(modelProviderNode) ?? DEFAULT_NODE_WIDTH;
    const height = readNodeHeight(modelProviderNode) ?? DEFAULT_NODE_HEIGHT;
    const preferred =
      direction === "LR"
        ? {
            x: (targetBounds.minX + targetBounds.maxX) / 2 - width / 2,
            y: targetBounds.minY - height - clusterGap,
          }
        : {
            x: targetBounds.minX - width - clusterGap,
            y: (targetBounds.minY + targetBounds.maxY) / 2 - height / 2,
          };
    placeNode(modelProviderId, preferred);
  }

  for (const toolConfigId of toolConfigIds) {
    const llmIds = toolConfigToLlmIds.get(toolConfigId) ?? [];
    const targetBounds = collectBounds(llmIds, nodesById);
    const toolConfigNode = nodesById.get(toolConfigId);
    if (!(targetBounds && toolConfigNode)) {
      continue;
    }
    const width = readNodeWidth(toolConfigNode) ?? DEFAULT_NODE_WIDTH;
    const height = readNodeHeight(toolConfigNode) ?? DEFAULT_NODE_HEIGHT;
    const preferred =
      direction === "LR"
        ? {
            x: (targetBounds.minX + targetBounds.maxX) / 2 - width / 2,
            y: targetBounds.minY - height - clusterGap,
          }
        : {
            x: targetBounds.minX - width - clusterGap,
            y: (targetBounds.minY + targetBounds.maxY) / 2 - height / 2,
          };
    placeNode(toolConfigId, preferred);
  }

  return nodes.map((node) => nodesById.get(node.id) ?? node);
}
