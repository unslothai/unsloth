// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeNode, NodeConfig } from "../../types";
import { DEFAULT_NODE_WIDTH } from "../../constants";
import { nodeDataFromConfig } from "../index";
import { isRecord, readString } from "./helpers";

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
  aux_nodes?: unknown;
  reroutes?: unknown;
};

type ParsedAuxNode = {
  llm: string;
  key: string;
  x: number;
  y: number;
};

export function parseUi(
  ui: UiInput | null,
): {
  layouts: Map<string, { x: number; y: number; width?: number }>;
  auxNodes: ParsedAuxNode[];
  edges: Array<{
    from: string;
    to: string;
    type?: string;
    via?: string[];
  }> | null;
  reroutes: Record<string, { x: number; y: number }>;
} {
  const layouts = new Map<string, { x: number; y: number; width?: number }>();
  const auxNodes: ParsedAuxNode[] = [];
  const edges: Array<{
    from: string;
    to: string;
    type?: string;
    via?: string[];
  }> = [];
  const reroutes: Record<string, { x: number; y: number }> = {};
  if (ui && Array.isArray(ui.reroutes)) {
    for (const node of ui.reroutes) {
      if (isRecord(node)) {
        const id = readString(node.id);
        if (id && typeof node.x === "number" && typeof node.y === "number") {
          reroutes[id] = { x: node.x, y: node.y };
        }
      }
    }
  }
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
          const via = Array.isArray(edge.via)
            ? edge.via.filter((id): id is string => typeof id === "string")
            : [];
          edges.push({
            from,
            to,
            type: readString(edge.type) ?? undefined,
            ...(via.length > 0 ? { via } : {}),
          });
        }
      }
    }
  }
  if (ui && Array.isArray(ui.aux_nodes)) {
    for (const node of ui.aux_nodes) {
      if (!isRecord(node)) {
        continue;
      }
      const llm = readString(node.llm);
      const key = readString(node.key);
      const x = typeof node.x === "number" ? node.x : null;
      const y = typeof node.y === "number" ? node.y : null;
      if (!(llm && key && x !== null && y !== null)) {
        continue;
      }
      auxNodes.push({ llm, key, x, y });
    }
  }
  return {
    layouts,
    auxNodes,
    edges: edges.length > 0 ? edges : null,
    reroutes,
  };
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
