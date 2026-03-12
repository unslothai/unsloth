// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, Node, XYPosition } from "@xyflow/react";
import type { RecipeGraphAuxNodeData } from "../../components/recipe-graph-aux-node";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../../constants";
import type { RecipeNode, LayoutDirection, NodeConfig } from "../../types";
import {
  getDefaultDataSourceHandle,
  getDefaultDataTargetHandle,
  getDefaultSemanticSourceHandle,
  getDefaultSemanticTargetHandle,
  HANDLE_IDS,
  isDataSourceHandle,
  isDataTargetHandle,
  isSemanticSourceHandle,
  isSemanticTargetHandle,
  normalizeRecipeHandleId,
} from "../handles";
import { readNodeHeight, readNodeWidth } from "../rf-node-dimensions";
import { isSemanticRelation } from "./relations";

type DisplayGraphInput = {
  nodes: RecipeNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  layoutDirection: LayoutDirection;
  auxNodePositions: Record<string, XYPosition>;
  llmAuxVisibility: Record<string, boolean>;
  runtime?: {
    runningNodeId: string | null;
    doneNodeIds: Set<string>;
    activeEdgeIds: Set<string>;
    executionLocked: boolean;
  };
};

export type DisplayGraph = {
  nodes: Array<Node<RecipeNode["data"] | RecipeGraphAuxNodeData>>;
  edges: Edge[];
};

function isAuxEdge(edge: Edge): boolean {
  return edge.source.startsWith("aux-") || edge.target.startsWith("aux-");
}

function normalizeEdge(
  edge: Edge,
  configs: Record<string, NodeConfig>,
  layoutDirection: LayoutDirection,
  activeEdgeIds: Set<string>,
  runningNodeId: string | null,
  doneNodeIds: Set<string>,
): Edge {
  const isActiveByRuntimeTarget =
    Boolean(runningNodeId) &&
    edge.target === runningNodeId &&
    !isAuxEdge(edge);
  const isActiveEdge = activeEdgeIds.has(edge.id) || isActiveByRuntimeTarget;
  const isAux = isAuxEdge(edge);
  if (isAux) {
    return {
      ...edge,
      type: "canvas",
      data: { ...(edge.data ?? {}), path: "smoothstep", active: isActiveEdge },
      animated: isActiveEdge,
    };
  }

  const isActiveReversedRuntimeEdge =
    Boolean(runningNodeId) &&
    isActiveEdge &&
    edge.source === runningNodeId &&
    doneNodeIds.has(edge.target);
  const displayEdge = isActiveReversedRuntimeEdge
    ? {
        ...edge,
        source: edge.target,
        target: edge.source,
        sourceHandle: getDefaultDataSourceHandle(layoutDirection),
        targetHandle: getDefaultDataTargetHandle(layoutDirection),
      }
    : edge;

  const source = configs[displayEdge.source];
  const target = configs[displayEdge.target];
  const semantic =
    displayEdge.type === "semantic" ||
    (Boolean(source && target) && isSemanticRelation(source, target));
  const sourceHandleNormalized = normalizeRecipeHandleId(displayEdge.sourceHandle);
  const targetHandleNormalized = normalizeRecipeHandleId(displayEdge.targetHandle);
  const semanticSourceDefault =
    source?.kind === "llm"
      ? getDefaultDataSourceHandle(layoutDirection)
      : getDefaultSemanticSourceHandle(layoutDirection);
  const semanticTargetDefault =
    target?.kind === "llm"
      ? getDefaultDataTargetHandle(layoutDirection)
      : getDefaultSemanticTargetHandle(layoutDirection);
  let sourceHandle = getDefaultDataSourceHandle(layoutDirection);
  let targetHandle = getDefaultDataTargetHandle(layoutDirection);

  if (semantic) {
    sourceHandle =
      isSemanticSourceHandle(sourceHandleNormalized) ||
      isDataSourceHandle(sourceHandleNormalized)
        ? sourceHandleNormalized ?? semanticSourceDefault
        : semanticSourceDefault;
    targetHandle =
      isSemanticTargetHandle(targetHandleNormalized) ||
      isDataTargetHandle(targetHandleNormalized)
        ? targetHandleNormalized ?? semanticTargetDefault
        : semanticTargetDefault;
    // LLM nodes only expose data lane handles; coerce legacy semantic handles.
    if (source?.kind === "llm" && isSemanticSourceHandle(sourceHandle)) {
      sourceHandle = semanticSourceDefault;
    }
    if (target?.kind === "llm" && isSemanticTargetHandle(targetHandle)) {
      targetHandle = semanticTargetDefault;
    }
  } else {
    sourceHandle = isDataSourceHandle(sourceHandleNormalized)
      ? sourceHandleNormalized ?? getDefaultDataSourceHandle(layoutDirection)
      : getDefaultDataSourceHandle(layoutDirection);
    targetHandle = isDataTargetHandle(targetHandleNormalized)
      ? targetHandleNormalized ?? getDefaultDataTargetHandle(layoutDirection)
      : getDefaultDataTargetHandle(layoutDirection);
  }

  return {
    ...displayEdge,
    type: semantic ? "semantic" : "canvas",
    data: semantic
      ? { ...(displayEdge.data ?? {}), active: isActiveEdge }
      : { ...(displayEdge.data ?? {}), path: "smoothstep", active: isActiveEdge },
    sourceHandle,
    targetHandle,
    animated: isActiveEdge,
  };
}

type AuxNodeItem = {
  key: string;
  data: RecipeGraphAuxNodeData;
};

type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

function toRect(
  position: XYPosition,
  width: number,
  height: number,
): Rect {
  return {
    x: position.x,
    y: position.y,
    width,
    height,
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
  for (let ring = 0; ring <= 10; ring += 1) {
    for (let dx = -ring; dx <= ring; dx += 1) {
      for (let dy = -ring; dy <= ring; dy += 1) {
        if (ring > 0 && Math.max(Math.abs(dx), Math.abs(dy)) !== ring) {
          continue;
        }
        const candidate = {
          x: preferred.x + dx * step,
          y: preferred.y + dy * step,
        };
        const rect = toRect(candidate, width, height);
        if (!occupied.some((other) => intersects(rect, other))) {
          return candidate;
        }
      }
    }
  }
  return preferred;
}

type HandleSide = "left" | "right" | "top" | "bottom";

const SIDE_TO_TARGET_HANDLE: Record<HandleSide, string> = {
  left: HANDLE_IDS.dataIn,
  right: HANDLE_IDS.dataInRight,
  top: HANDLE_IDS.dataInTop,
  bottom: HANDLE_IDS.dataInBottom,
};

function getTargetSide(
  handleId: string | null | undefined,
  direction: LayoutDirection,
): HandleSide {
  const normalized = normalizeRecipeHandleId(handleId);
  if (!normalized) {
    return direction === "TB" ? "top" : "left";
  }
  if (
    normalized === HANDLE_IDS.dataInRight ||
    normalized === HANDLE_IDS.semanticInRight
  ) {
    return "right";
  }
  if (
    normalized === HANDLE_IDS.dataInBottom ||
    normalized === HANDLE_IDS.semanticInBottom
  ) {
    return "bottom";
  }
  if (
    normalized === HANDLE_IDS.dataInTop ||
    normalized === HANDLE_IDS.semanticInTop
  ) {
    return "top";
  }
  return "left";
}

function getSourceSide(
  handleId: string | null | undefined,
  direction: LayoutDirection,
): HandleSide {
  const normalized = normalizeRecipeHandleId(handleId);
  if (!normalized) {
    return direction === "TB" ? "bottom" : "right";
  }
  if (
    normalized === HANDLE_IDS.dataOutLeft ||
    normalized === HANDLE_IDS.semanticOutLeft
  ) {
    return "left";
  }
  if (
    normalized === HANDLE_IDS.dataOutTop ||
    normalized === HANDLE_IDS.semanticOutTop
  ) {
    return "top";
  }
  if (
    normalized === HANDLE_IDS.dataOutBottom ||
    normalized === HANDLE_IDS.semanticOutBottom
  ) {
    return "bottom";
  }
  return "right";
}

function pickAuxTargetHandle(
  llmId: string,
  direction: LayoutDirection,
  edges: Edge[],
): string {
  const occupied = new Set<HandleSide>();
  for (const edge of edges) {
    if (isAuxEdge(edge)) {
      continue;
    }
    if (edge.target === llmId) {
      occupied.add(getTargetSide(edge.targetHandle, direction));
    }
    if (edge.source === llmId) {
      occupied.add(getSourceSide(edge.sourceHandle, direction));
    }
  }

  const priority: HandleSide[] =
    direction === "LR"
      ? ["left", "right", "bottom", "top"]
      : ["top", "bottom", "right", "left"];
  for (const side of priority) {
    if (!occupied.has(side)) {
      return SIDE_TO_TARGET_HANDLE[side];
    }
  }

  const fallback: HandleSide = direction === "LR" ? "bottom" : "right";
  return SIDE_TO_TARGET_HANDLE[fallback];
}

function getHandleSideFromTargetHandle(targetHandle: string): HandleSide {
  if (targetHandle === HANDLE_IDS.dataInRight) {
    return "right";
  }
  if (targetHandle === HANDLE_IDS.dataInTop) {
    return "top";
  }
  if (targetHandle === HANDLE_IDS.dataInBottom) {
    return "bottom";
  }
  return "left";
}

function pickAuxSourceHandle(
  auxPosition: XYPosition,
  auxWidth: number,
  auxHeight: number,
  llmPosition: XYPosition,
  llmWidth: number,
  llmHeight: number,
): string {
  const auxCenter = {
    x: auxPosition.x + auxWidth / 2,
    y: auxPosition.y + auxHeight / 2,
  };
  const llmCenter = {
    x: llmPosition.x + llmWidth / 2,
    y: llmPosition.y + llmHeight / 2,
  };
  const dx = llmCenter.x - auxCenter.x;
  const dy = llmCenter.y - auxCenter.y;

  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? HANDLE_IDS.llmInputOutRight : HANDLE_IDS.llmInputOutLeft;
  }
  return dy >= 0 ? HANDLE_IDS.llmInputOutBottom : HANDLE_IDS.llmInputOutTop;
}

type AppendAuxNodeAndEdgeInput = {
  auxNodes: Node<RecipeGraphAuxNodeData>[];
  auxEdges: Edge[];
  entry: {
    item: AuxNodeItem;
    auxId: string;
    width: number;
    height: number;
  };
  position: XYPosition;
  parentNode: Node<RecipeNode["data"] | RecipeGraphAuxNodeData>;
  parentWidth: number;
  parentHeight: number;
  auxTargetHandle: string;
};

function appendAuxNodeAndEdge({
  auxNodes,
  auxEdges,
  entry,
  position,
  parentNode,
  parentWidth,
  parentHeight,
  auxTargetHandle,
}: AppendAuxNodeAndEdgeInput): void {
  auxNodes.push({
    id: entry.auxId,
    type: "aux",
    data: entry.item.data,
    position,
    width: entry.width,
    height: entry.height,
    style: {
      width: entry.width,
      height: entry.height,
    },
    draggable: true,
    selectable: true,
    focusable: true,
    connectable: false,
  });

  auxEdges.push({
    id: `e-${entry.auxId}-${parentNode.id}`,
    source: entry.auxId,
    sourceHandle: pickAuxSourceHandle(
      position,
      entry.width,
      entry.height,
      parentNode.position,
      parentWidth,
      parentHeight,
    ),
    target: parentNode.id,
    targetHandle: auxTargetHandle,
    type: "canvas",
    data: { path: "auto" },
    selectable: false,
    focusable: false,
  });
}

export function deriveDisplayGraph({
  nodes,
  edges,
  configs,
  layoutDirection,
  auxNodePositions,
  llmAuxVisibility,
  runtime,
}: DisplayGraphInput): DisplayGraph {
  const executionLocked = runtime?.executionLocked ?? false;
  const runningNodeId = runtime?.runningNodeId ?? null;
  const doneNodeIds = runtime?.doneNodeIds ?? new Set<string>();
  const activeEdgeIds = runtime?.activeEdgeIds ?? new Set<string>();
  const displayNodes = nodes.map((node) => {
    const hasWidth =
      typeof node.width === "number" ||
      typeof node.style?.width === "number" ||
      (typeof node.style?.width === "string" &&
        Number.isFinite(Number.parseFloat(node.style.width)));
    const runtimeState: "idle" | "running" | "done" =
      node.id === runningNodeId
        ? "running"
        : doneNodeIds.has(node.id)
          ? "done"
          : "idle";
    if (hasWidth) {
      return {
        ...node,
        data: {
          ...node.data,
          runtimeState,
          executionLocked,
        },
      };
    }
    return {
      ...node,
      data: {
        ...node.data,
        runtimeState,
        executionLocked,
      },
      style: { ...node.style, width: DEFAULT_NODE_WIDTH },
    };
  });
  const auxNodes: Node<RecipeGraphAuxNodeData>[] = [];
  const auxEdges: Edge[] = [];
  const occupiedRects: Rect[] = displayNodes.map((node) =>
    toRect(
      node.position,
      readNodeWidth(node) ?? DEFAULT_NODE_WIDTH,
      readNodeHeight(node) ?? DEFAULT_NODE_HEIGHT,
    ),
  );

  for (const node of displayNodes) {
    const config = configs[node.id];
    if (!(config && config.kind === "llm")) {
      continue;
    }
    if (!llmAuxVisibility[config.id]) {
      continue;
    }
    const llmDirection = node.data.layoutDirection ?? layoutDirection;
    const auxTargetHandle = pickAuxTargetHandle(node.id, llmDirection, edges);
    const auxTargetSide = getHandleSideFromTargetHandle(auxTargetHandle);
    const items: AuxNodeItem[] = [];

    if (config.system_prompt.trim()) {
      items.push({
        key: "system",
        data: {
          kind: "llm-prompt-input",
          llmId: config.id,
          field: "system_prompt",
          title: "System Prompt",
          executionLocked,
        },
      });
    }

    if (config.prompt.trim()) {
      items.push({
        key: "prompt",
        data: {
          kind: "llm-prompt-input",
          llmId: config.id,
          field: "prompt",
          title: "Prompt",
          executionLocked,
        },
      });
    }

    if (config.llm_type === "judge") {
      (config.scores ?? []).forEach((_score, scoreIndex) => {
        items.push({
          key: `score-${scoreIndex}`,
          data: {
            kind: "llm-judge-score",
            llmId: config.id,
            scoreIndex,
            executionLocked,
          },
        });
      });
    }

    if (items.length === 0) {
      continue;
    }

    const parentWidth = readNodeWidth(node) ?? DEFAULT_NODE_WIDTH;
    const parentHeight = readNodeHeight(node) ?? DEFAULT_NODE_HEIGHT;
    const itemsWithLayout = items.map((item) => {
      const auxId = `aux-${node.id}-${item.key}`;
      return {
        item,
        auxId,
        width: DEFAULT_NODE_WIDTH,
        height: DEFAULT_NODE_HEIGHT,
      };
    });

    const gap = 24;
    const sideOffset = 48;
    const stackHorizontal =
      auxTargetSide === "top" || auxTargetSide === "bottom";

    if (stackHorizontal) {
      const totalWidth =
        itemsWithLayout.reduce((sum, entry) => sum + entry.width, 0) +
        (itemsWithLayout.length - 1) * gap;
      const startX = node.position.x + (parentWidth - totalWidth) / 2;
      let xCursor = startX;

      for (const entry of itemsWithLayout) {
        const preferredPosition = {
          x: xCursor,
          y:
            auxTargetSide === "top"
              ? node.position.y - entry.height - sideOffset
              : node.position.y + parentHeight + sideOffset,
        };
        const defaultPosition = findNonOverlappingPosition(
          preferredPosition,
          entry.width,
          entry.height,
          occupiedRects,
        );
        const position = auxNodePositions[entry.auxId] ?? defaultPosition;
        xCursor += entry.width + gap;

        occupiedRects.push(toRect(position, entry.width, entry.height));
        appendAuxNodeAndEdge({
          auxNodes,
          auxEdges,
          entry,
          position,
          parentNode: node,
          parentWidth,
          parentHeight,
          auxTargetHandle,
        });
      }
      continue;
    }

    const totalHeight =
      itemsWithLayout.reduce((sum, entry) => sum + entry.height, 0) +
      (itemsWithLayout.length - 1) * gap;
    const maxWidth = Math.max(...itemsWithLayout.map((entry) => entry.width));
    const baseX =
      auxTargetSide === "right"
        ? node.position.x + parentWidth + sideOffset
        : node.position.x - maxWidth - sideOffset;
    let yCursor = node.position.y + (parentHeight - totalHeight) / 2;

    for (const entry of itemsWithLayout) {
      const preferredPosition = {
        x: baseX + (maxWidth - entry.width),
        y: yCursor,
      };
      const defaultPosition = findNonOverlappingPosition(
        preferredPosition,
        entry.width,
        entry.height,
        occupiedRects,
      );
      const position = auxNodePositions[entry.auxId] ?? defaultPosition;
      yCursor += entry.height + gap;

      occupiedRects.push(toRect(position, entry.width, entry.height));
      appendAuxNodeAndEdge({
        auxNodes,
        auxEdges,
        entry,
        position,
        parentNode: node,
        parentWidth,
        parentHeight,
        auxTargetHandle,
      });
    }
  }

  return {
    nodes: [...displayNodes, ...auxNodes],
    edges: [...edges, ...auxEdges].map((edge) =>
      normalizeEdge(
        edge,
        configs,
        layoutDirection,
        activeEdgeIds,
        runningNodeId,
        doneNodeIds,
      ),
    ),
  };
}
