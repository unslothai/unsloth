// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode, Node } from "@xyflow/react";
import type { Rect } from "./orthogonal-router";

/**
 * Build absolute-positioned node rectangles for the orthogonal router from
 * React Flow's `nodeLookup`, skipping the edge's own endpoints (so the wire
 * can leave/enter its handles) and any node that isn't measured yet.
 */
export function collectEdgeObstacles<N extends Node>(
  nodeLookup: Map<string, InternalNode<N>>,
  sourceId: string,
  targetId: string,
): Rect[] {
  const rects: Rect[] = [];
  nodeLookup.forEach((node, id) => {
    if (id === sourceId || id === targetId) {
      return;
    }
    const width = node.measured?.width ?? node.width ?? null;
    const height = node.measured?.height ?? node.height ?? null;
    const position = node.internals.positionAbsolute ?? node.position;
    if (width == null || height == null || width <= 0 || height <= 0) {
      return;
    }
    rects.push({ x: position.x, y: position.y, width, height });
  });
  return rects;
}
