import type { Edge, Node, XYPosition } from "@xyflow/react";
import type { CanvasAuxNodeData } from "../../components/canvas-aux-node";
import type { CanvasNode, LayoutDirection, NodeConfig } from "../../types";
import { getLlmJudgeScoreHandleId, HANDLE_IDS } from "../handles";

type DisplayGraphInput = {
  nodes: CanvasNode[];
  edges: Edge[];
  configs: Record<string, NodeConfig>;
  layoutDirection: LayoutDirection;
  auxNodePositions: Record<string, XYPosition>;
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

export function deriveDisplayGraph({
  nodes,
  edges,
  configs,
  layoutDirection,
  auxNodePositions,
}: DisplayGraphInput): DisplayGraph {
  const auxNodes: Node<CanvasAuxNodeData>[] = [];
  const auxEdges: Edge[] = [];
  const auxDefaults: Record<string, XYPosition> = {};
  const auxNodeIds: string[] = [];

  for (const node of nodes) {
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

    const itemSpan = 140;
    const itemCenterOffset = ((items.length - 1) * itemSpan) / 2;
    const horizontalSpan = 300;
    const horizontalCenterOffset = ((items.length - 1) * horizontalSpan) / 2;

    items.forEach((item, index) => {
      const auxId = `aux-${node.id}-${item.key}`;
      const defaultPosition =
        llmDirection === "TB"
          ? {
              x: node.position.x + index * horizontalSpan - horizontalCenterOffset,
              y: node.position.y - 210,
            }
          : {
              x: node.position.x - 330,
              y: node.position.y + index * itemSpan - itemCenterOffset,
            };
      const position = auxNodePositions[auxId] ?? defaultPosition;

      auxNodeIds.push(auxId);
      if (!auxNodePositions[auxId]) {
        auxDefaults[auxId] = defaultPosition;
      }

      auxNodes.push({
        id: auxId,
        type: "aux",
        data: item.data,
        position,
        draggable: true,
        selectable: true,
        focusable: true,
        connectable: false,
      });

      auxEdges.push({
        id: `e-${auxId}-${node.id}`,
        source: auxId,
        sourceHandle: HANDLE_IDS.llmInputOut,
        target: node.id,
        targetHandle: item.targetHandle,
        type: "canvas",
        data: { path: "auto" },
        selectable: false,
        focusable: false,
        style: { strokeWidth: 1.5, stroke: "var(--border)" },
      });
    });
  }

  return {
    nodes: [...nodes, ...auxNodes],
    edges: [...edges, ...auxEdges],
    auxNodeIds,
    auxDefaults,
  };
}
