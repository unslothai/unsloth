import type { Edge, Node, XYPosition } from "@xyflow/react";
import type { CanvasAuxNodeData } from "../../components/canvas-aux-node";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../../constants";
import type { CanvasNode, LayoutDirection, NodeConfig } from "../../types";
import { getLlmJudgeScoreHandleId, HANDLE_IDS } from "../handles";

type DisplayGraphInput = {
  nodes: CanvasNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  layoutDirection: LayoutDirection;
  auxNodePositions: Record<string, XYPosition>;
  auxNodeSizes: Record<string, { width: number; height: number }>;
};

export type DisplayGraph = {
  nodes: Array<Node<CanvasNode["data"] | CanvasAuxNodeData>>;
  edges: Edge[];
  auxNodeIds: string[];
  auxDefaults: Record<string, XYPosition>;
};

type AuxNodeItem = {
  key: string;
  targetHandle: string;
  data: CanvasAuxNodeData;
};

function getNodeWidth(node: Node): number {
  if (typeof node.width === "number" && Number.isFinite(node.width)) {
    return node.width;
  }
  if (typeof node.style?.width === "number" && Number.isFinite(node.style.width)) {
    return node.style.width;
  }
  if (typeof node.style?.width === "string") {
    const parsed = Number.parseFloat(node.style.width);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (
    typeof node.measured?.width === "number" &&
    Number.isFinite(node.measured.width)
  ) {
    return node.measured.width;
  }
  return DEFAULT_NODE_WIDTH;
}

function getNodeHeight(node: Node): number {
  if (typeof node.height === "number" && Number.isFinite(node.height)) {
    return node.height;
  }
  if (typeof node.style?.height === "number" && Number.isFinite(node.style.height)) {
    return node.style.height;
  }
  if (typeof node.style?.height === "string") {
    const parsed = Number.parseFloat(node.style.height);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (
    typeof node.measured?.height === "number" &&
    Number.isFinite(node.measured.height)
  ) {
    return node.measured.height;
  }
  return DEFAULT_NODE_HEIGHT;
}

export function deriveDisplayGraph({
  nodes,
  edges,
  configs,
  layoutDirection,
  auxNodePositions,
  auxNodeSizes,
}: DisplayGraphInput): DisplayGraph {
  const displayNodes = nodes.map((node) => {
    const hasWidth =
      typeof node.width === "number" ||
      typeof node.style?.width === "number" ||
      (typeof node.style?.width === "string" &&
        Number.isFinite(Number.parseFloat(node.style.width)));
    if (hasWidth) {
      return node;
    }
    return {
      ...node,
      style: { ...node.style, width: DEFAULT_NODE_WIDTH },
    };
  });
  const auxNodes: Node<CanvasAuxNodeData>[] = [];
  const auxEdges: Edge[] = [];
  const auxDefaults: Record<string, XYPosition> = {};
  const auxNodeIds: string[] = [];

  for (const node of displayNodes) {
    const config = configs[node.id];
    if (!(config && config.kind === "llm")) {
      continue;
    }
    const llmDirection = node.data.layoutDirection ?? layoutDirection;
    const items: AuxNodeItem[] = [];

    if (config.system_prompt.trim()) {
      items.push({
        key: "system",
        targetHandle: HANDLE_IDS.llmSystemIn,
        data: {
          kind: "llm-prompt-input",
          llmId: config.id,
          field: "system_prompt",
          title: "System Prompt",
          layoutDirection: llmDirection,
        },
      });
    }

    if (config.prompt.trim()) {
      items.push({
        key: "prompt",
        targetHandle: HANDLE_IDS.llmPromptIn,
        data: {
          kind: "llm-prompt-input",
          llmId: config.id,
          field: "prompt",
          title: "Prompt",
          layoutDirection: llmDirection,
        },
      });
    }

    if (config.llm_type === "judge") {
      (config.scores ?? []).forEach((_score, scoreIndex) => {
        items.push({
          key: `score-${scoreIndex}`,
          targetHandle: getLlmJudgeScoreHandleId(scoreIndex),
          data: {
            kind: "llm-judge-score",
            llmId: config.id,
            scoreIndex,
            layoutDirection: llmDirection,
          },
        });
      });
    }

    if (items.length === 0) {
      continue;
    }

    const parentWidth = getNodeWidth(node);
    const parentHeight = getNodeHeight(node);
    const itemsWithLayout = items.map((item) => {
      const auxId = `aux-${node.id}-${item.key}`;
      const savedSize = auxNodeSizes[auxId];
      return {
        item,
        auxId,
        width: savedSize?.width ?? DEFAULT_NODE_WIDTH,
        height: savedSize?.height ?? DEFAULT_NODE_HEIGHT,
      };
    });

    const gap = 24;
    const sideOffset = 48;

    if (llmDirection === "TB") {
      const totalWidth =
        itemsWithLayout.reduce((sum, entry) => sum + entry.width, 0) +
        (itemsWithLayout.length - 1) * gap;
      const startX = node.position.x + (parentWidth - totalWidth) / 2;
      let xCursor = startX;

      for (const entry of itemsWithLayout) {
        const defaultPosition = {
          x: xCursor,
          y: node.position.y - entry.height - sideOffset,
        };
        const position = auxNodePositions[entry.auxId] ?? defaultPosition;
        xCursor += entry.width + gap;

        auxNodeIds.push(entry.auxId);
        if (!auxNodePositions[entry.auxId]) {
          auxDefaults[entry.auxId] = defaultPosition;
        }

        auxNodes.push({
          id: entry.auxId,
          type: "aux",
          data: entry.item.data,
          position,
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
          id: `e-${entry.auxId}-${node.id}`,
          source: entry.auxId,
          sourceHandle: HANDLE_IDS.llmInputOut,
          target: node.id,
          targetHandle: entry.item.targetHandle,
          type: "canvas",
          data: { path: "auto" },
          selectable: false,
          focusable: false,
          style: { strokeWidth: 1.5, stroke: "var(--border)" },
        });
      }
      continue;
    }

    const totalHeight =
      itemsWithLayout.reduce((sum, entry) => sum + entry.height, 0) +
      (itemsWithLayout.length - 1) * gap;
    const maxWidth = Math.max(...itemsWithLayout.map((entry) => entry.width));
    const baseX = node.position.x - maxWidth - sideOffset;
    let yCursor = node.position.y + (parentHeight - totalHeight) / 2;

    for (const entry of itemsWithLayout) {
      const defaultPosition = {
        x: baseX + (maxWidth - entry.width),
        y: yCursor,
      };
      const position = auxNodePositions[entry.auxId] ?? defaultPosition;
      yCursor += entry.height + gap;

      auxNodeIds.push(entry.auxId);
      if (!auxNodePositions[entry.auxId]) {
        auxDefaults[entry.auxId] = defaultPosition;
      }

      auxNodes.push({
        id: entry.auxId,
        type: "aux",
        data: entry.item.data,
        position,
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
        id: `e-${entry.auxId}-${node.id}`,
        source: entry.auxId,
        sourceHandle: HANDLE_IDS.llmInputOut,
        target: node.id,
        targetHandle: entry.item.targetHandle,
        type: "canvas",
        data: { path: "auto" },
        selectable: false,
        focusable: false,
        style: { strokeWidth: 1.5, stroke: "var(--border)" },
      });
    }
  }

  return {
    nodes: [...displayNodes, ...auxNodes],
    edges: [...edges, ...auxEdges],
    auxNodeIds,
    auxDefaults,
  };
}
