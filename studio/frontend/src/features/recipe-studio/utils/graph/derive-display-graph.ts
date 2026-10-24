// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, Node, XYPosition } from "@xyflow/react";
import type { RecipeGraphAuxNodeData } from "../../components/recipe-graph-aux-node";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../../constants";
import type { RecipeNode, NodeConfig } from "../../types";
import { HANDLE_IDS } from "../handles";
import {
  getAuxNodePorts,
  getRecipeNodePorts,
  REROUTE_PORTS,
  resolveRecipeEdgeHandles,
} from "../ports";
import { readNodeHeight, readNodeWidth } from "../rf-node-dimensions";
import { isSemanticRelation } from "./relations";
import { expandRerouteEdges, REROUTE_HEIGHT, REROUTE_WIDTH } from "./reroutes";

type DisplayGraphInput = {
  nodes: RecipeNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  auxNodePositions: Record<string, XYPosition>;
  reroutes?: Record<string, XYPosition>;
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
        sourceHandle: HANDLE_IDS.dataOut,
        targetHandle: HANDLE_IDS.dataIn,
      }
    : edge;

  const source = configs[displayEdge.source];
  const target = configs[displayEdge.target];
  const semantic =
    displayEdge.type === "semantic" ||
    (Boolean(source && target) && isSemanticRelation(source, target));
  // The handles follow from what the two cards are, so a stale stored handle can't drop a wire.
  const { sourceHandle, targetHandle } = isActiveReversedRuntimeEdge
    ? { sourceHandle: HANDLE_IDS.dataOut, targetHandle: HANDLE_IDS.dataIn }
    : resolveRecipeEdgeHandles(source, target);

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
  portLabel: string;
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
};

function appendAuxNodeAndEdge({
  auxNodes,
  auxEdges,
  entry,
  position,
  parentNode,
}: AppendAuxNodeAndEdgeInput): void {
  auxNodes.push({
    id: entry.auxId,
    type: "aux",
    data: { ...entry.item.data, ports: getAuxNodePorts(entry.item.portLabel) },
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
    sourceHandle: HANDLE_IDS.llmInputOut,
    target: parentNode.id,
    targetHandle: HANDLE_IDS.promptIn,
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
  auxNodePositions,
  reroutes = {},
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
    const items: AuxNodeItem[] = [];

    if (config.system_prompt.trim()) {
      items.push({
        key: "system",
        portLabel: "system",
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
        portLabel: "prompt",
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
          portLabel: "score",
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
    const totalHeight =
      itemsWithLayout.reduce((sum, entry) => sum + entry.height, 0) +
      (itemsWithLayout.length - 1) * gap;
    const maxWidth = Math.max(...itemsWithLayout.map((entry) => entry.width));
    // Prompt boxes feed the LLM's left side, so they sit to its left.
    const baseX = node.position.x - maxWidth - sideOffset;
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
      });
    }
  }

  const displayEdges = expandRerouteEdges(
    [...edges, ...auxEdges].map((edge) =>
      normalizeEdge(edge, configs, activeEdgeIds, runningNodeId, doneNodeIds),
    ),
    reroutes,
  );

  // Ports depend on what is wired in, so they are worked out after the edges.
  const auxOpen = new Set(auxEdges.map((edge) => edge.target));
  const dataInputs = new Set(
    displayEdges
      .filter((edge) => edge.targetHandle === HANDLE_IDS.dataIn)
      .map((edge) => edge.target),
  );
  const hasToolProfiles = Object.values(configs).some(
    (config) => config.kind === "tool_config",
  );
  const portedNodes = displayNodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      ports: getRecipeNodePorts(configs[node.id], {
        promptInput: auxOpen.has(node.id),
        toolsInput: hasToolProfiles,
        dataInput: dataInputs.has(node.id),
      }),
    },
  }));
  const rerouteNodes: Node<RecipeNode["data"] | RecipeGraphAuxNodeData>[] = Object.entries(
    reroutes,
  ).map(([id, position]) => ({
    id,
    type: "reroute",
    position,
    width: REROUTE_WIDTH,
    height: REROUTE_HEIGHT,
    data: { ports: REROUTE_PORTS } as unknown as RecipeNode["data"],
    selectable: true,
    draggable: true,
  }));

  return {
    nodes: [...portedNodes, ...auxNodes, ...rerouteNodes],
    edges: displayEdges,
  };
}
