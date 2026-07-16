// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeNode, NodeConfig } from "../../types";
import { DEFAULT_NODE_WIDTH } from "../../constants";
import { nodeDataFromConfig } from "../index";
import { normalizeRecipeHandleId } from "../handles";
import { isRecord, readString } from "./helpers";

type UiInput = {
  nodes?: unknown;
  edges?: unknown;
  aux_nodes?: unknown;
  layout_direction?: unknown;
  layoutDirection?: unknown;
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
    sourceHandle?: string;
    targetHandle?: string;
  }> | null;
  layoutDirection: "LR" | "TB" | null;
} {
  const layouts = new Map<string, { x: number; y: number; width?: number }>();
  const auxNodes: ParsedAuxNode[] = [];
  const edges: Array<{
    from: string;
    to: string;
    type?: string;
    sourceHandle?: string;
    targetHandle?: string;
  }> = [];
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
          const sourceHandle = normalizeRecipeHandleId(
            readString(edge.source_handle) ?? readString(edge.sourceHandle),
          );
          const targetHandle = normalizeRecipeHandleId(
            readString(edge.target_handle) ?? readString(edge.targetHandle),
          );
          edges.push({
            from,
            to,
            type: readString(edge.type) ?? undefined,
            sourceHandle: sourceHandle ?? undefined,
            targetHandle: targetHandle ?? undefined,
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
  const layoutDirectionRaw =
    readString(ui?.layout_direction) ?? readString(ui?.layoutDirection);
  const layoutDirection =
    layoutDirectionRaw === "TB"
      ? "TB"
      : layoutDirectionRaw === "LR"
        ? "LR"
        : null;

  return {
    layouts,
    auxNodes,
    edges: edges.length > 0 ? edges : null,
    layoutDirection,
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
