import type { Edge } from "@xyflow/react";
import type { CanvasNode, NodeConfig } from "../../types";

export type CanvasSnapshot = {
  configs: Record<string, NodeConfig>;
  nodes: CanvasNode[];
  edges: Edge[];
  nextId: number;
  nextY: number;
};

export type ImportResult = {
  errors: string[];
  snapshot: CanvasSnapshot | null;
};
