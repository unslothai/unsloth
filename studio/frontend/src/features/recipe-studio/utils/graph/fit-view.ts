// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FitViewOptions, Node } from "@xyflow/react";

/** Cap auto-fit zoom so the view doesn't punch in too tight on small graphs. */
export const FIT_VIEW_MAX_ZOOM = 1.1;
export const FIT_VIEW_PADDING = 0.12;
export const FIT_VIEW_DURATION_MS = 340;

/** Markdown note nodes are decorative and should not affect the fitView bbox. */
function isMarkdownNoteNode(node: Node): boolean {
  if (node.type !== "builder") {
    return false;
  }
  if (!node.data || typeof node.data !== "object") {
    return false;
  }
  return (node.data as { kind?: string }).kind === "note";
}

/** Aux nodes (llm-prompt-input, llm-judge-score) are satellite overlays. */
function isAuxNode(node: Node): boolean {
  return node.type === "aux";
}

/**
 * Returns the primary workflow nodes that fitView should target.
 *
 * Excludes markdown notes and aux (LLM input overlay) nodes so the viewport
 * is framed around the primary workflow blocks. Falls back to all nodes if
 * filtering would leave an empty set.
 *
 * The returned array contains full {@link Node} objects so callers can inspect
 * `node.measured` without a second lookup pass.
 */
export function getFitViewTargetNodes(nodes: Node[]): Node[] {
  const primary = nodes.filter(
    (node) => !(isMarkdownNoteNode(node) || isAuxNode(node)),
  );
  return primary.length > 0 ? primary : nodes;
}

/**
 * Builds a standard {@link FitViewOptions} object targeting the primary
 * workflow nodes. Every call site that invokes `fitView` should go through
 * this helper so zoom, padding, and node filtering stay consistent.
 */
export function buildFitViewOptions(
  nodes: Node[],
  overrides?: Partial<FitViewOptions>,
): FitViewOptions {
  const targets = getFitViewTargetNodes(nodes);
  return {
    duration: FIT_VIEW_DURATION_MS,
    maxZoom: FIT_VIEW_MAX_ZOOM,
    padding: FIT_VIEW_PADDING,
    nodes: targets.map((n) => ({ id: n.id })),
    ...overrides,
  };
}
