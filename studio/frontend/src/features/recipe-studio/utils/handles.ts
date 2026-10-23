// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { Connection } from "@xyflow/react";

// Inputs sit on the left of a card, outputs on the right. See utils/ports.ts for which card
// shows which port and what it is labelled.
export const HANDLE_IDS = {
  dataIn: "data-in",
  dataOut: "data-out",
  semanticIn: "semantic-in",
  semanticOut: "semantic-out",
  modelIn: "model-in",
  toolsIn: "tools-in",
  promptIn: "prompt-in",
  llmInputOut: "llm-input-out-right",
  rerouteIn: "reroute-in",
  rerouteOut: "reroute-out",
} as const;

export type RecipeHandleId = (typeof HANDLE_IDS)[keyof typeof HANDLE_IDS];

// Handles older recipes may still reference, including the top and bottom ones from the
// top-to-bottom layout. Each maps to the left or right handle of the same lane; the edge is kept.
export const LEGACY_HANDLE_ALIAS_MAP: Record<string, string> = {
  "data-in-top": HANDLE_IDS.dataIn,
  "data-in-right": HANDLE_IDS.dataIn,
  "data-in-bottom": HANDLE_IDS.dataIn,
  "data-out-bottom": HANDLE_IDS.dataOut,
  "data-out-left": HANDLE_IDS.dataOut,
  "data-out-top": HANDLE_IDS.dataOut,
  "semantic-in-top": HANDLE_IDS.semanticIn,
  "semantic-in-left": HANDLE_IDS.semanticIn,
  "semantic-in-right": HANDLE_IDS.semanticIn,
  "semantic-in-bottom": HANDLE_IDS.semanticIn,
  "semantic-out-bottom": HANDLE_IDS.semanticOut,
  "semantic-out-right": HANDLE_IDS.semanticOut,
  "semantic-out-left": HANDLE_IDS.semanticOut,
  "semantic-out-top": HANDLE_IDS.semanticOut,
  "llm-input-out-bottom": HANDLE_IDS.llmInputOut,
  "llm-input-out-left": HANDLE_IDS.llmInputOut,
  "llm-input-out-top": HANDLE_IDS.llmInputOut,
};

const DATA_TARGET_HANDLES = new Set<string>([HANDLE_IDS.dataIn]);
const DATA_SOURCE_HANDLES = new Set<string>([HANDLE_IDS.dataOut]);
const SEMANTIC_TARGET_HANDLES = new Set<string>([
  HANDLE_IDS.semanticIn,
  HANDLE_IDS.modelIn,
  HANDLE_IDS.toolsIn,
]);
const SEMANTIC_SOURCE_HANDLES = new Set<string>([HANDLE_IDS.semanticOut]);

export function normalizeRecipeHandleId(
  handleId: string | null | undefined,
): string | null {
  if (!handleId) {
    return null;
  }
  return LEGACY_HANDLE_ALIAS_MAP[handleId] ?? handleId;
}

export type RecipeHandleSide = "left" | "right";

/** Side a handle sits on: every input is on the left, every output on the right. */
export function getRecipeHandleSide(
  _handleId: string | null | undefined,
  kind: "source" | "target",
): RecipeHandleSide {
  return kind === "target" ? "left" : "right";
}

/** Rewrites removed handle ids on a stored edge to their legal replacement. */
export function migrateRecipeEdgeHandles<
  T extends { sourceHandle?: string | null; targetHandle?: string | null },
>(edge: T): T {
  const sourceHandle = normalizeRecipeHandleId(edge.sourceHandle);
  const targetHandle = normalizeRecipeHandleId(edge.targetHandle);
  if (
    sourceHandle === (edge.sourceHandle ?? null) &&
    targetHandle === (edge.targetHandle ?? null)
  ) {
    return edge;
  }
  return { ...edge, sourceHandle, targetHandle };
}

export function normalizeRecipeConnectionHandles(
  connection: Connection,
): Connection {
  return {
    ...connection,
    sourceHandle: normalizeRecipeHandleId(connection.sourceHandle),
    targetHandle: normalizeRecipeHandleId(connection.targetHandle),
  };
}

function isKnownHandle(
  handleId: string | null | undefined,
  handles: Set<string>,
): boolean {
  if (!handleId) {
    return false;
  }
  return handles.has(normalizeRecipeHandleId(handleId) ?? "");
}

export function isDataTargetHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, DATA_TARGET_HANDLES);
}

export function isDataSourceHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, DATA_SOURCE_HANDLES);
}

export function isSemanticTargetHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, SEMANTIC_TARGET_HANDLES);
}

export function isSemanticSourceHandle(
  handleId: string | null | undefined,
): boolean {
  return isKnownHandle(handleId, SEMANTIC_SOURCE_HANDLES);
}
