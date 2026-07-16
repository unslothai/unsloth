// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  Edge,
  EdgeChange,
  Node,
  NodeChange,
  XYPosition,
} from "@xyflow/react";

export function applyAuxNodeChanges<T extends Node>(
  changes: NodeChange<T>[],
  actions: {
    setAuxNodePosition: (id: string, position: XYPosition) => void;
  },
): void {
  for (const change of changes) {
    if (!("id" in change) || !change.id.startsWith("aux-")) {
      continue;
    }
    if (change.type !== "position") {
      continue;
    }
    const nextPosition = change.position ?? change.positionAbsolute;
    if (!nextPosition) {
      continue;
    }
    actions.setAuxNodePosition(change.id, nextPosition);
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
