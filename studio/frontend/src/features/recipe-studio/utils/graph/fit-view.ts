// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Node } from "@xyflow/react";

function isMarkdownNoteNode(node: Node): boolean {
  if (node.type !== "builder") {
    return false;
  }
  if (!node.data || typeof node.data !== "object") {
    return false;
  }
  return (node.data as { kind?: string }).kind === "note";
}

export function getFitNodeIdsIgnoringNotes(nodes: Node[]): Array<{ id: string }> {
  const nodesWithoutNotes = nodes.filter((node) => !isMarkdownNoteNode(node));
  const targetNodes = nodesWithoutNotes.length > 0 ? nodesWithoutNotes : nodes;
  return targetNodes.map((node) => ({ id: node.id }));
}
