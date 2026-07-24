// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode, Node } from "@xyflow/react";
import type { Rect } from "./orthogonal-router";

/**
 * Build absolute-positioned node rectangles for the orthogonal router from
 * React Flow's `nodeLookup`.
 *
 * Every measured node is included — including the edge's own endpoints — so a
 * wire never routes under the card it attaches to. The router exits each handle
 * with a straight stub longer than the obstacle clearance, so the pin lead
 * clears its own (inflated) card without needing it excluded here.
 */
export function collectObstacles<N extends Node>(
  nodeLookup: Map<string, InternalNode<N>>,
): Rect[] {
  const rects: Rect[] = [];
  for (const node of nodeLookup.values()) {
    const width = node.measured?.width ?? node.width ?? null;
    const height = node.measured?.height ?? node.height ?? null;
    const position = node.internals.positionAbsolute ?? node.position;
    if (width == null || height == null || width <= 0 || height <= 0) {
      continue;
    }
    rects.push({ x: position.x, y: position.y, width, height });
  }
  return rects;
}
