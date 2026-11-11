// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  COLUMN_TYPE,
  GENERATE_TYPE,
  isRecipeGroup,
} from "../recipe/recipe-columns";
import type { DocGraph } from "../stores/doc-store";
import type { FlowGroup } from "../types";

/** The recipe a column or Recipe Dataset block belongs to: it only runs inside it, wherever it is dragged. */
export function ownRecipe(graph: DocGraph, nodeId: string): string | null {
  const node = graph.nodes[nodeId];
  if (!node || (node.type !== COLUMN_TYPE && node.type !== GENERATE_TYPE))
    return null;
  return node.group && isRecipeGroup(graph.groups[node.group])
    ? node.group
    : null;
}

/** A recipe frame grown to hold the blocks of its own that were dragged past its edge. */
export function holdRecipeBlocks(
  group: FlowGroup,
  positions: Record<string, [number, number]>,
  sizes: Map<string, { width: number; height: number }>,
): FlowGroup {
  const pad = 24;
  let [x0, y0] = group.rect;
  let x1 = x0 + group.rect[2];
  let y1 = y0 + group.rect[3];
  for (const [id, pos] of Object.entries(positions)) {
    const size = sizes.get(id) ?? { width: 260, height: 120 };
    x0 = Math.min(x0, pos[0] - pad);
    y0 = Math.min(y0, pos[1] - pad * 2);
    x1 = Math.max(x1, pos[0] + size.width + pad);
    y1 = Math.max(y1, pos[1] + size.height + pad);
  }
  const rect: [number, number, number, number] = [x0, y0, x1 - x0, y1 - y0];
  return rect.every((v, i) => v === group.rect[i]) ? group : { ...group, rect };
}
