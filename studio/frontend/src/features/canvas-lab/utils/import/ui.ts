import type { CanvasNode, NodeConfig } from "../../types";
import { nodeDataFromConfig } from "../index";
import { isRecord, readString } from "./helpers";

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
};

export function parseUi(
  ui: UiInput | null,
): {
  positions: Map<string, { x: number; y: number }>;
  edges: Array<{ from: string; to: string; type?: string }> | null;
} {
  const positions = new Map<string, { x: number; y: number }>();
  const edges: Array<{ from: string; to: string; type?: string }> = [];
  if (ui && Array.isArray(ui.nodes)) {
    for (const node of ui.nodes) {
      if (isRecord(node)) {
        const id = readString(node.id);
        const x = typeof node.x === "number" ? node.x : null;
        const y = typeof node.y === "number" ? node.y : null;
        if (id && x !== null && y !== null) {
          positions.set(id, { x, y });
        }
      }
    }
  }
  if (ui && Array.isArray(ui.edges)) {
    for (const edge of ui.edges) {
      if (isRecord(edge)) {
        const from = readString(edge.from);
        const to = readString(edge.to);
        if (from && to) {
          edges.push({
            from,
            to,
            type: readString(edge.type) ?? undefined,
          });
        }
      }
    }
  }
  return { positions, edges: edges.length > 0 ? edges : null };
}

export function buildNodes(
  configs: NodeConfig[],
  positions: Map<string, { x: number; y: number }>,
): CanvasNode[] {
  return configs.map((config, index) => {
    const position =
      positions.get(config.name) ?? ({ x: 0, y: index * 140 } as const);
    return {
      id: config.id,
      type: "builder",
      position,
      data: nodeDataFromConfig(config),
    };
  });
}
