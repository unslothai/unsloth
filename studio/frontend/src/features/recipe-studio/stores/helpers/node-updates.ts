// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { XYPosition } from "@xyflow/react";
import { DEFAULT_NODE_WIDTH } from "../../constants";
import type {
  RecipeNode,
  LayoutDirection,
  NodeConfig,
} from "../../types";
import { nodeDataFromConfig } from "../../utils";
import { getConfigUiMode } from "../../components/inline/inline-policy";

export type NodeUpdateState = {
  configs: Record<string, NodeConfig>;
  nodes: RecipeNode[];
  nextId: number;
  nextY: number;
};

export type NodeUpdateResult = {
  configs: Record<string, NodeConfig>;
  nodes: RecipeNode[];
  nextId: number;
  nextY: number;
  activeConfigId: string;
  dialogOpen: boolean;
};

export function updateNodeData(
  nodes: RecipeNode[],
  id: string,
  config: NodeConfig,
  layoutDirection: LayoutDirection,
): RecipeNode[] {
  return nodes.map((node) =>
    node.id === id
      ? { ...node, data: nodeDataFromConfig(config, layoutDirection) }
      : node,
  );
}

export function buildNodeUpdate(
  state: NodeUpdateState,
  config: NodeConfig,
  layoutDirection: LayoutDirection,
  position?: XYPosition,
  openDialog = true,
): NodeUpdateResult {
  const node: RecipeNode = {
    id: config.id,
    type: "builder",
    position: position ?? { x: 0, y: state.nextY },
    data: nodeDataFromConfig(config, layoutDirection),
    style: { width: DEFAULT_NODE_WIDTH },
    selected: true,
  };
  const mode = getConfigUiMode(config);
  return {
    configs: { ...state.configs, [config.id]: config },
    nodes: [...state.nodes.map((item) => ({ ...item, selected: false })), node],
    nextId: state.nextId + 1,
    nextY: position ? state.nextY : state.nextY + 140,
    activeConfigId: config.id,
    dialogOpen: openDialog && mode === "dialog",
  };
}

export function applyLayoutDirectionToNodes(
  nodes: RecipeNode[],
  configs: Record<string, NodeConfig>,
  layoutDirection: LayoutDirection,
): RecipeNode[] {
  return nodes.map((node) => {
    const config = configs[node.id];
    if (config) {
      return { ...node, data: nodeDataFromConfig(config, layoutDirection) };
    }
    return {
      ...node,
      data: { ...node.data, layoutDirection },
    };
  });
}
