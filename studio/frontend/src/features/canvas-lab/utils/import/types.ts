import type { Edge } from "@xyflow/react";
import type {
  CanvasNode,
  CanvasProcessorConfig,
  NodeConfig,
} from "../../types";

export type CanvasSnapshot = {
  configs: Record<string, NodeConfig>;
  nodes: CanvasNode[];
  edges: Edge[];
  processors: CanvasProcessorConfig[];
  nextId: number;
  nextY: number;
};

export type ImportResult = {
  errors: string[];
  snapshot: CanvasSnapshot | null;
};
