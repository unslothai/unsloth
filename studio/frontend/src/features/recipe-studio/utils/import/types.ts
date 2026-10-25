// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Edge, XYPosition } from "@xyflow/react";
import type {
  LayoutDirection,
  RecipeNode,
  RecipeProcessorConfig,
  NodeConfig,
} from "../../types";
import type { SuperCard } from "../graph/super-cards";

export type RecipeSnapshot = {
  configs: Record<string, NodeConfig>;
  nodes: RecipeNode[];
  edges: Edge[];
  auxNodePositions: Record<string, XYPosition>;
  processors: RecipeProcessorConfig[];
  layoutDirection: LayoutDirection;
  nextId: number;
  nextY: number;
  superCards?: Record<string, SuperCard>;
};

export type ImportResult = {
  errors: string[];
  snapshot: RecipeSnapshot | null;
};
