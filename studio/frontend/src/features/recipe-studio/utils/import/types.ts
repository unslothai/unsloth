import type { Edge } from "@xyflow/react";
import type {
  RecipeNode,
  RecipeProcessorConfig,
  NodeConfig,
} from "../../types";

export type RecipeSnapshot = {
  configs: Record<string, NodeConfig>;
  nodes: RecipeNode[];
  edges: Edge[];
  processors: RecipeProcessorConfig[];
  nextId: number;
  nextY: number;
};

export type ImportResult = {
  errors: string[];
  snapshot: RecipeSnapshot | null;
};
