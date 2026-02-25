import type {
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  XYPosition,
} from "@xyflow/react";
import type { RecipeGraphAuxNodeData } from "../components/recipe-graph-aux-node";
import type { RecipeNodeData } from "../types";

type AnyNode = Node<RecipeNodeData | RecipeGraphAuxNodeData>;

export function applyAuxNodeChanges(
  changes: NodeChange<AnyNode>[],
  actions: {
    setAuxNodePosition: (id: string, position: XYPosition) => void;
    setAuxNodeSize: (
      id: string,
      size: { width: number; height: number },
    ) => void;
  },
): void {
  for (const change of changes) {
    if (!("id" in change) || !change.id.startsWith("aux-")) {
      continue;
    }
    if (change.type === "position") {
      const nextPosition = change.position ?? change.positionAbsolute;
      if (nextPosition) {
        actions.setAuxNodePosition(change.id, nextPosition);
      }
      continue;
    }
    if (
      change.type === "dimensions" &&
      change.dimensions &&
      change.dimensions.width > 0 &&
      change.dimensions.height > 0
    ) {
      actions.setAuxNodeSize(change.id, {
        width: change.dimensions.width,
        height: change.dimensions.height,
      });
    }
  }
}

export function filterNodeChangesByIds<T extends Node>(
  changes: NodeChange<T>[],
  ids: Set<string>,
): NodeChange<T>[] {
  return changes.filter(
    (change): change is NodeChange<T> => "id" in change && ids.has(change.id),
  );
}

export function filterEdgeChangesByIds(
  changes: EdgeChange<Edge>[],
  ids: Set<string>,
): EdgeChange<Edge>[] {
  return changes.filter(
    (change): change is EdgeChange<Edge> => "id" in change && ids.has(change.id),
  );
}

