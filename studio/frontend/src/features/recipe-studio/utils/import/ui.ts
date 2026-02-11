import type { RecipeNode, NodeConfig } from "../../types";
import { DEFAULT_NODE_WIDTH } from "../../constants";
import { nodeDataFromConfig } from "../index";
import { isRecord, readString } from "./helpers";

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
};

export function parseUi(
  ui: UiInput | null,
): {
  layouts: Map<string, { x: number; y: number; width?: number }>;
  edges: Array<{ from: string; to: string; type?: string }> | null;
} {
  const layouts = new Map<string, { x: number; y: number; width?: number }>();
  const edges: Array<{ from: string; to: string; type?: string }> = [];
  if (ui && Array.isArray(ui.nodes)) {
    for (const node of ui.nodes) {
      if (isRecord(node)) {
        const id = readString(node.id);
        const x = typeof node.x === "number" ? node.x : null;
        const y = typeof node.y === "number" ? node.y : null;
        const width = typeof node.width === "number" ? node.width : null;
        if (id && x !== null && y !== null) {
          layouts.set(id, {
            x,
            y,
            ...(width && width > 0 ? { width } : {}),
          });
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
  return { layouts, edges: edges.length > 0 ? edges : null };
}

export function buildNodes(
  configs: NodeConfig[],
  layouts: Map<string, { x: number; y: number; width?: number }>,
): RecipeNode[] {
  return configs.map((config, index) => {
    const fallbackLayout: { x: number; y: number; width?: number } = {
      x: 0,
      y: index * 140,
    };
    const layout =
      layouts.get(config.name) ?? fallbackLayout;
    return {
      id: config.id,
      type: "builder",
      position: { x: layout.x, y: layout.y },
      data: nodeDataFromConfig(config),
      style: { width: layout.width ?? DEFAULT_NODE_WIDTH },
    };
  });
}
